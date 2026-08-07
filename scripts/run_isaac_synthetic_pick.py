#!/usr/bin/env python3
"""Execute a validated synthetic cuMotion pick trajectory in Isaac Sim.

This is an offline acceleration-drive tracking smoke test. It uses the exact
position/velocity samples stored by ``plan_synthetic_pick.py`` and steps PhysX,
but it does not model finger closure, pin attachment, contact, calibrated tool
mass properties, or the real Techman controller. It creates no ROS graph,
sensor input, network connection, or Watson command.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any

import numpy as np


ARENA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_USD = ARENA_DIR / "generated/isaac/6.0.1/tm5s_with_2fg7/tm5s_with_2fg7.usda"
DEFAULT_IMPORT_REPORT = ARENA_DIR / "outputs/isaac_sim/6.0.1/import_report.json"
DEFAULT_REPORT = ARENA_DIR / "outputs/isaac_sim/6.0.1/synthetic_pick_report.json"
EXPECTED_PYTHON = (3, 12)
EXPECTED_ISAAC_PACKAGES = {
    "isaacsim": "6.0.1.0",
    "isaacsim-asset": "6.0.1.0",
    "isaacsim-core": "6.0.1.0",
}
EXPECTED_DOF_NAMES = [f"joint_{index}" for index in range(1, 7)]
MAX_TRACKING_ERROR_RADIANS = 0.05
MAX_ENDPOINT_ERROR_RADIANS = 0.01
CAMERA_EYE = [1.35, 1.20, 1.05]
CAMERA_TARGET = [0.25, 0.0, 0.40]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--import-report", type=Path, default=DEFAULT_IMPORT_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Stop after this many simulated seconds; zero runs until the window closes.",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Run without wall-clock pacing; intended for headless validation.",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="Capture one rendered frame after one simulated second; refuses overwrite.",
    )
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def float64_sha256(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.asarray(array, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def validate_runtime() -> dict[str, str]:
    if sys.version_info[:2] != EXPECTED_PYTHON:
        raise RuntimeError(
            f"Isaac Sim 6.0.1 synthetic viewer requires Python 3.12; found {sys.version.split()[0]}"
        )
    versions: dict[str, str] = {}
    for package_name, expected in EXPECTED_ISAAC_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Missing required package: {package_name}") from exc
        if actual != expected:
            raise RuntimeError(f"{package_name} must be {expected}; found {actual}")
        versions[package_name] = actual
    return versions


def load_and_validate_plan(
    plan_path: Path,
    usd_path: Path,
    import_report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not plan_path.is_file():
        raise FileNotFoundError(f"Synthetic pick plan is missing: {plan_path}")
    if not usd_path.is_file():
        raise FileNotFoundError(f"Imported TM5S USD is missing: {usd_path}")
    if not import_report_path.is_file():
        raise FileNotFoundError(f"Isaac import report is missing: {import_report_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    import_report = json.loads(import_report_path.read_text(encoding="utf-8"))
    if plan.get("format_version") != 1:
        raise ValueError("Synthetic pick plan must use format_version: 1")
    if plan.get("camera_or_depth_used") is not False:
        raise ValueError("Synthetic plan must explicitly record camera_or_depth_used false")
    if plan.get("real_robot_commanded") is not False or plan.get("watson_connected") is not False:
        raise ValueError("Synthetic plan must explicitly record no Watson connection or command")
    if plan.get("planning_tool_frame") != "flange":
        raise ValueError("This viewer requires a cuMotion plan at the flange tool frame")
    if plan.get("accepted_candidate_count", 0) < 1 or not plan.get("selected", {}).get("accepted"):
        raise ValueError("Synthetic plan has no accepted selected candidate")
    if plan.get("tool_model", {}).get("quick_changer_robot_side_modeled") is not False:
        raise ValueError("Unexpected Quick Changer model state")
    isaac_execution = plan.get("isaac_execution", {})
    if isaac_execution.get("status") != "simulation_only_uncalibrated_acceleration_drive_smoke":
        raise ValueError("Synthetic plan lacks the expected simulation-only drive profile")
    finite_drive_values = [
        np.asarray(isaac_execution.get(name), dtype=np.float64)
        for name in ("proportional_gains", "derivative_gains")
    ]
    if any(array.shape != (6,) or not np.all(np.isfinite(array)) for array in finite_drive_values):
        raise ValueError("Synthetic plan drive gains must contain six finite values")
    if isaac_execution.get("save_to_usd") is not False:
        raise ValueError("Synthetic drive gains must not be persisted to the imported USD")

    urdf_hash = plan["input_artifacts"]["urdf"]["sha256"]
    if urdf_hash != import_report.get("source_urdf_sha256"):
        raise ValueError("cuMotion URDF hash does not match the URDF imported into Isaac")
    if sha256_file(usd_path) != import_report.get("output_usd_sha256"):
        raise ValueError("Isaac USD hash does not match the validated import report")
    if import_report.get("dof_count") != 6 or import_report.get("physx_dof_names") != EXPECTED_DOF_NAMES:
        raise ValueError("Isaac import report does not describe the expected six TM5S joints")

    control_dt = float(plan["control_dt_seconds"])
    if control_dt <= 0.0:
        raise ValueError("Plan control_dt_seconds must be positive")
    maximum_allowed_step = float(plan["maximum_control_step_rad"])
    previous_end: np.ndarray | None = None
    expected_stages = ["pregrasp", "grasp", "lift"]
    actual_stages = [stage["name"] for stage in plan["selected"]["stages"]]
    if actual_stages != expected_stages:
        raise ValueError(f"Expected stage order {expected_stages}; found {actual_stages}")
    for stage in plan["selected"]["stages"]:
        if not stage.get("accepted"):
            raise ValueError(f"Selected stage is not accepted: {stage['name']}")
        samples = stage.get("control_samples")
        if not isinstance(samples, list) or len(samples) < 2:
            raise ValueError(f"Stage {stage['name']} has no usable control samples")
        times = np.asarray([sample["time_seconds"] for sample in samples], dtype=np.float64)
        positions = [np.asarray(sample["joint_positions"], dtype=np.float64) for sample in samples]
        velocities = [np.asarray(sample["joint_velocities"], dtype=np.float64) for sample in samples]
        if any(array.shape != (6,) or not np.all(np.isfinite(array)) for array in positions + velocities):
            raise ValueError(f"Stage {stage['name']} contains invalid joint samples")
        if times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
            raise ValueError(f"Stage {stage['name']} sample times are not strictly increasing from zero")
        if not math.isclose(
            float(times[-1]),
            float(stage["trajectory_validation"]["duration_seconds"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"Stage {stage['name']} sample duration does not match validation")
        computed_hash = float64_sha256(positions + velocities)
        if computed_hash != stage["control_samples_float64_sha256"]:
            raise ValueError(f"Stage {stage['name']} control sample hash mismatch")
        maximum_step = float(np.max(np.abs(np.diff(np.asarray(positions), axis=0))))
        if maximum_step > maximum_allowed_step + 1.0e-12:
            raise ValueError(f"Stage {stage['name']} exceeds the maximum command step")
        if previous_end is not None and not np.allclose(positions[0], previous_end, atol=1.0e-12):
            raise ValueError(f"Stage {stage['name']} is discontinuous from the previous endpoint")
        previous_end = positions[-1]
    return plan, import_report


def build_command_cycle(plan: dict[str, Any]) -> list[dict[str, Any]]:
    control_dt = float(plan["control_dt_seconds"])
    hold_steps = max(1, int(round(float(plan["stage_hold_seconds"]) / control_dt)))
    commands: list[dict[str, Any]] = []
    stages = plan["selected"]["stages"]

    def append_stage(stage: dict[str, Any], direction: str) -> None:
        samples = stage["control_samples"]
        iterable = samples if direction == "forward" else reversed(samples)
        for sample_index, sample in enumerate(iterable):
            if commands and sample_index == 0:
                continue
            velocity_sign = 1.0 if direction == "forward" else -1.0
            commands.append(
                {
                    "phase": stage["name"] if direction == "forward" else f"reset_{stage['name']}",
                    "direction": direction,
                    "joint_positions": np.asarray(sample["joint_positions"], dtype=np.float64),
                    "joint_velocities": velocity_sign
                    * np.asarray(sample["joint_velocities"], dtype=np.float64),
                    "endpoint": False,
                    "hold": False,
                }
            )
        endpoint_position = commands[-1]["joint_positions"].copy()
        for hold_index in range(hold_steps):
            commands.append(
                {
                    "phase": f"hold_{stage['name']}" if direction == "forward" else "hold_ready",
                    "direction": direction,
                    "joint_positions": endpoint_position,
                    "joint_velocities": np.zeros(6, dtype=np.float64),
                    "endpoint": hold_index == hold_steps - 1,
                    "hold": True,
                }
            )

    for stage in stages:
        append_stage(stage, "forward")
    for reverse_index, stage in enumerate(reversed(stages)):
        append_stage(stage, "reverse")
        if reverse_index < len(stages) - 1:
            # Only hold after the final reset-to-ready segment.
            del commands[-hold_steps:]
    return commands


def axis_orientation(axis: np.ndarray, rot_matrix_to_quat: Any) -> np.ndarray:
    z_axis = axis / np.linalg.norm(axis)
    reference = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(reference, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return rot_matrix_to_quat(np.column_stack((x_axis, y_axis, z_axis)))


def add_visual_scene(stage: Any, plan: dict[str, Any]) -> list[str]:
    from isaacsim.core.api.objects import VisualCuboid, VisualCylinder, VisualSphere
    from isaacsim.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat
    from pxr import UsdLux

    created: list[str] = []
    role_colors = {
        "foam": np.array([0.18, 0.34, 0.20]),
        "tray_wall": np.array([0.28, 0.31, 0.36]),
        "specimen_body": np.array([0.36, 0.18, 0.08]),
        "other_pin_shaft": np.array([0.62, 0.64, 0.68]),
        "other_pin_head": np.array([0.78, 0.12, 0.12]),
    }
    for index, obstacle in enumerate(plan["selected"]["collision_obstacles"]):
        path = f"/SyntheticPick/CollisionProxy_{index:02d}_{obstacle['role']}"
        color = role_colors[obstacle["role"]]
        if obstacle["type"] == "cuboid":
            qx, qy, qz, qw = obstacle["quaternion_xyzw"]
            VisualCuboid(
                prim_path=path,
                position=np.asarray(obstacle["position_xyz_m"], dtype=np.float64),
                orientation=np.array([qw, qx, qy, qz], dtype=np.float64),
                scale=np.asarray(obstacle["side_lengths_m"], dtype=np.float64),
                size=1.0,
                color=color,
            )
        else:
            VisualSphere(
                prim_path=path,
                position=np.asarray(obstacle["position_xyz_m"], dtype=np.float64),
                radius=float(obstacle["radius_m"]),
                color=color,
            )
        created.append(path)

    selected_truth_id = int(plan["selected"]["truth_id"])
    selected_truth = next(
        truth
        for truth in plan["synthetic_scene"]["truth"]
        if int(truth["pin_id"]) == selected_truth_id
    )
    base = np.asarray(selected_truth["base"], dtype=np.float64)
    axis = np.asarray(selected_truth["axis_up"], dtype=np.float64)
    length = float(selected_truth["length_m"])
    pin_path = "/SyntheticPick/SelectedPin/Shaft"
    VisualCylinder(
        prim_path=pin_path,
        position=base + axis * length / 2.0,
        orientation=axis_orientation(axis, rot_matrix_to_quat),
        radius=0.0012,
        height=length,
        color=np.array([0.08, 0.95, 0.25]),
    )
    head_path = "/SyntheticPick/SelectedPin/Head"
    VisualSphere(
        prim_path=head_path,
        position=np.asarray(selected_truth["head"], dtype=np.float64),
        radius=0.004,
        color=np.array([0.08, 0.95, 0.25]),
    )
    created.extend([pin_path, head_path])

    target = plan["selected"]["target"]
    target_colors = {
        "pregrasp": np.array([0.10, 0.50, 1.00]),
        "grasp": np.array([1.00, 0.20, 0.10]),
        "lift": np.array([0.20, 1.00, 0.55]),
    }
    for name, position_key in [
        ("pregrasp", "pregrasp_position"),
        ("grasp", "grasp_position"),
        ("lift", "lift_position"),
    ]:
        path = f"/SyntheticPick/Targets/{name.capitalize()}"
        VisualSphere(
            prim_path=path,
            position=np.asarray(target[position_key], dtype=np.float64),
            radius=0.006,
            color=target_colors[name],
        )
        created.append(path)

    dome = UsdLux.DomeLight.Define(stage, "/SyntheticPick/Lights/Dome")
    dome.CreateIntensityAttr(850.0)
    key = UsdLux.DistantLight.Define(stage, "/SyntheticPick/Lights/Key")
    key.CreateIntensityAttr(2600.0)
    key.CreateAngleAttr(1.0)
    created.extend(["/SyntheticPick/Lights/Dome", "/SyntheticPick/Lights/Key"])
    return created


def create_status_panel(ui: Any, plan: dict[str, Any]) -> tuple[Any, Any, dict[str, bool]]:
    state = {"stop_requested": False}
    selected = plan["selected"]
    clearance_mm = selected["minimum_sampled_sphere_clearance_m"] * 1000.0
    window = ui.Window("TM5S Synthetic Pin Pick", width=480, height=285)
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label(
                "SIMULATION ONLY - NOT CONNECTED TO WATSON",
                style={"color": 0xFF5A5AFF, "font_size": 17},
            )
            ui.Label(
                f"Seed {plan['synthetic_scene']['seed']} | selected detection "
                f"{selected['detection_id']} | clearance {clearance_mm:.2f} mm"
            )
            status_label = ui.Label("Initialising acceleration-drive tracking...")
            ui.Label(
                "Blue/red/green dots: pregrasp, grasp, lift TCP targets.\n"
                "The selected pin stays fixed: no finger closure or attached-object model.\n"
                "Tool mass, Quick Changer, grasp point, and controller are uncalibrated.",
                word_wrap=True,
            )

            def request_stop() -> None:
                state["stop_requested"] = True

            ui.Button("Stop and close demo", height=32, clicked_fn=request_stop)
    return window, status_label, state


def main() -> int:
    args = build_parser().parse_args()
    package_versions = validate_runtime()
    plan_path = args.plan.expanduser().resolve()
    usd_path = args.usd.expanduser().resolve()
    import_report_path = args.import_report.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if args.duration_seconds < 0.0:
        raise ValueError("--duration-seconds must be non-negative")
    if args.headless and args.duration_seconds <= 0.0:
        raise ValueError("--headless requires a positive --duration-seconds")
    screenshot_path = args.screenshot.expanduser().resolve() if args.screenshot else None
    if screenshot_path is not None:
        if screenshot_path.exists():
            raise FileExistsError(f"Refusing to overwrite screenshot: {screenshot_path}")
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    plan, import_report = load_and_validate_plan(plan_path, usd_path, import_report_path)
    command_cycle = build_command_cycle(plan)
    physics_dt = float(plan["control_dt_seconds"])

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "hide_ui": args.headless,
            "width": 1280,
            "height": 720,
            "window_width": 1600,
            "window_height": 900,
            "renderer": "RaytracedLighting",
            "active_gpu": 0,
            "physics_gpu": 0,
            "multi_gpu": False,
            "max_gpu_count": 1,
            "fast_shutdown": True,
            "disable_viewport_updates": False,
            "open_usd": str(usd_path),
        }
    )
    exit_code = 1
    world = None
    panel_window = None
    started_wall = time.perf_counter()
    try:
        import omni.ui as ui
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.rendering_manager import ViewportManager
        from isaacsim.core.utils.types import ArticulationAction
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        World.clear_instance()
        stage = omni.usd.get_context().get_stage()
        default_prim = stage.GetDefaultPrim()
        if not default_prim.IsValid():
            raise RuntimeError("Imported TM5S stage has no default prim")
        asset_prim_path = str(default_prim.GetPath())
        scene_paths = add_visual_scene(stage, plan)
        world = World(
            physics_dt=physics_dt,
            rendering_dt=1.0 / 60.0,
            stage_units_in_meters=1.0,
            backend="numpy",
            device="cpu",
        )
        robot = world.scene.add(
            SingleArticulation(prim_path=asset_prim_path, name="tm5s_synthetic_pick")
        )
        world.reset()
        if not robot.handles_initialized:
            raise RuntimeError("TM5S articulation handles did not initialise")
        if robot.num_dof != 6 or list(robot.dof_names) != EXPECTED_DOF_NAMES:
            raise RuntimeError(
                f"Expected TM5S DOFs {EXPECTED_DOF_NAMES}; found {list(robot.dof_names)}"
            )
        dof_properties = robot.dof_properties
        joint_limits = np.column_stack(
            (
                np.asarray(dof_properties["lower"], dtype=np.float64),
                np.asarray(dof_properties["upper"], dtype=np.float64),
            )
        )
        all_positions = np.asarray(
            [command["joint_positions"] for command in command_cycle], dtype=np.float64
        )
        if np.any(all_positions < joint_limits[:, 0]) or np.any(all_positions > joint_limits[:, 1]):
            raise RuntimeError("Synthetic control trajectory exceeds imported joint limits")

        start_positions = all_positions[0]
        zeros = np.zeros(6, dtype=np.float64)
        robot.set_joints_default_state(positions=start_positions, velocities=zeros)
        robot.set_joint_positions(start_positions)
        robot.set_joint_velocities(zeros)
        controller = robot.get_articulation_controller()
        imported_proportional_gains, imported_derivative_gains = controller.get_gains()
        maximum_efforts = controller.get_max_efforts()
        configured_proportional_gains = np.asarray(
            plan["isaac_execution"]["proportional_gains"], dtype=np.float64
        )
        configured_derivative_gains = np.asarray(
            plan["isaac_execution"]["derivative_gains"], dtype=np.float64
        )
        controller.set_gains(
            kps=configured_proportional_gains,
            kds=configured_derivative_gains,
            save_to_usd=False,
        )
        applied_proportional_gains, applied_derivative_gains = controller.get_gains()
        if not np.allclose(applied_proportional_gains, configured_proportional_gains) or not np.allclose(
            applied_derivative_gains, configured_derivative_gains
        ):
            raise RuntimeError("Isaac controller did not apply the configured simulation gains")
        world.play()

        render = not args.headless or screenshot_path is not None
        render_interval = max(1, int(round((1.0 / 60.0) / physics_dt)))
        if render:
            viewport_ready, waited_frames = ViewportManager.wait_for_viewport(max_frames=120)
            if not viewport_ready:
                raise RuntimeError(f"Isaac viewport was not ready after {waited_frames} frames")
            ViewportManager.set_camera_view(
                ViewportManager.get_camera(), eye=CAMERA_EYE, target=CAMERA_TARGET
            )
        if args.headless:
            status_label = None
            panel_state = {"stop_requested": False}
        else:
            panel_window, status_label, panel_state = create_status_panel(ui, plan)

        settling_steps = max(1, int(round(0.25 / physics_dt)))
        start_action = ArticulationAction(joint_positions=start_positions, joint_velocities=zeros)
        for settling_index in range(settling_steps):
            robot.apply_action(start_action)
            world.step(render=render and settling_index % render_interval == 0)

        simulated_seconds = 0.0
        step_count = 0
        cycle_count = 0
        command_index = 0
        squared_position_error_sum = 0.0
        position_error_values = 0
        maximum_tracking_error = 0.0
        maximum_tracking_error_per_joint = np.zeros(6, dtype=np.float64)
        maximum_velocity_error = 0.0
        maximum_actual_velocity = np.zeros(6, dtype=np.float64)
        maximum_actual_acceleration = np.zeros(6, dtype=np.float64)
        previous_actual_velocity = np.asarray(robot.get_joint_velocities(), dtype=np.float64)
        endpoint_errors: list[dict[str, Any]] = []
        last_phase = ""
        screenshot_requested = False
        screenshot_capture = None
        screenshot_future = None
        interrupted = False

        try:
            while simulation_app.is_running() and not panel_state["stop_requested"]:
                step_started = time.perf_counter()
                command = command_cycle[command_index]
                action = ArticulationAction(
                    joint_positions=command["joint_positions"],
                    joint_velocities=command["joint_velocities"],
                )
                robot.apply_action(action)
                render_this_step = render and step_count % render_interval == 0
                world.step(render=render_this_step)
                actual_positions = np.asarray(robot.get_joint_positions(), dtype=np.float64)
                actual_velocities = np.asarray(robot.get_joint_velocities(), dtype=np.float64)
                if (
                    actual_positions.shape != (6,)
                    or actual_velocities.shape != (6,)
                    or not np.all(np.isfinite(actual_positions))
                    or not np.all(np.isfinite(actual_velocities))
                ):
                    raise RuntimeError("Isaac controller produced invalid joint state")
                position_error = np.abs(actual_positions - command["joint_positions"])
                velocity_error = np.abs(actual_velocities - command["joint_velocities"])
                maximum_tracking_error = max(maximum_tracking_error, float(np.max(position_error)))
                maximum_tracking_error_per_joint = np.maximum(
                    maximum_tracking_error_per_joint, position_error
                )
                maximum_velocity_error = max(maximum_velocity_error, float(np.max(velocity_error)))
                maximum_actual_velocity = np.maximum(maximum_actual_velocity, np.abs(actual_velocities))
                actual_acceleration = (actual_velocities - previous_actual_velocity) / physics_dt
                maximum_actual_acceleration = np.maximum(
                    maximum_actual_acceleration, np.abs(actual_acceleration)
                )
                previous_actual_velocity = actual_velocities
                squared_position_error_sum += float(np.dot(position_error, position_error))
                position_error_values += 6

                if command["phase"] != last_phase:
                    print(f"Synthetic pick: {command['phase']}")
                    if status_label is not None:
                        status_label.text = (
                            f"Motion: {command['phase']} | cycle {cycle_count + 1} | "
                            f"max error {maximum_tracking_error * 1000.0:.2f} mrad"
                        )
                    last_phase = command["phase"]
                if command["endpoint"]:
                    endpoint_errors.append(
                        {
                            "phase": command["phase"],
                            "cycle": cycle_count + 1,
                            "maximum_joint_error_rad": float(np.max(position_error)),
                        }
                    )

                simulated_seconds += physics_dt
                step_count += 1
                command_index += 1
                if command_index >= len(command_cycle):
                    command_index = 0
                    cycle_count += 1
                if (
                    screenshot_path is not None
                    and not screenshot_requested
                    and simulated_seconds >= 1.0
                ):
                    viewport = get_active_viewport()
                    if viewport is None:
                        raise RuntimeError("No active viewport is available for screenshot capture")
                    screenshot_capture = capture_viewport_to_file(
                        viewport, file_path=str(screenshot_path)
                    )
                    screenshot_future = asyncio.ensure_future(
                        screenshot_capture.wait_for_result(completion_frames=30)
                    )
                    screenshot_requested = True
                if args.duration_seconds > 0.0 and simulated_seconds >= args.duration_seconds:
                    break
                if not args.no_realtime:
                    remaining = physics_dt - (time.perf_counter() - step_started)
                    if remaining > 0.0:
                        time.sleep(remaining)
        except KeyboardInterrupt:
            interrupted = True
            print("Synthetic pick interrupted; writing report before closing")

        screenshot_wait_result = None
        if screenshot_path is not None and screenshot_future is None:
            raise RuntimeError("Screenshot was requested but the run ended before capture time")
        if screenshot_future is not None:
            for _ in range(180):
                if screenshot_future.done() or not simulation_app.is_running():
                    break
                world.render()
            if not screenshot_future.done():
                raise RuntimeError("Screenshot capture did not finish within 180 frames")
            screenshot_wait_result = bool(screenshot_future.result())
            if not screenshot_wait_result:
                raise RuntimeError("Isaac viewport screenshot capture returned failure")
            import omni.kit.renderer_capture

            omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
        screenshot_written = screenshot_path is None or screenshot_path.is_file()
        if not screenshot_written:
            raise RuntimeError(f"Screenshot capture did not complete: {screenshot_path}")

        maximum_endpoint_error = max(
            (record["maximum_joint_error_rad"] for record in endpoint_errors), default=0.0
        )
        print(
            "Synthetic pick tracking: "
            f"max={maximum_tracking_error:.6f} rad, "
            f"per_joint={maximum_tracking_error_per_joint.tolist()}, "
            f"endpoint={maximum_endpoint_error:.6f} rad, "
            f"kp={np.asarray(applied_proportional_gains).tolist()}, "
            f"kd={np.asarray(applied_derivative_gains).tolist()}"
        )
        if maximum_tracking_error > MAX_TRACKING_ERROR_RADIANS:
            raise RuntimeError(
                f"Acceleration-drive tracking exceeded {MAX_TRACKING_ERROR_RADIANS} rad: "
                f"{maximum_tracking_error}"
            )
        if maximum_endpoint_error > MAX_ENDPOINT_ERROR_RADIANS:
            raise RuntimeError(
                f"Endpoint tracking exceeded {MAX_ENDPOINT_ERROR_RADIANS} rad: "
                f"{maximum_endpoint_error}"
            )

        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv],
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "package_versions": package_versions,
            "validation_scope": "offline_physx_acceleration_drive_tracking_smoke",
            "source_plan": str(plan_path),
            "source_plan_sha256": sha256_file(plan_path),
            "source_usd": str(usd_path),
            "source_usd_sha256": sha256_file(usd_path),
            "source_import_report": str(import_report_path),
            "source_import_report_sha256": sha256_file(import_report_path),
            "headless": args.headless,
            "active_gpu": 0,
            "multi_gpu": False,
            "configured_physx_device": "cpu",
            "physx_dynamics_stepped": True,
            "physics_dt_seconds": physics_dt,
            "render_interval_physics_steps": render_interval,
            "physical_camera_or_depth_sensor_used": False,
            "viewport_camera_used": render,
            "ros_used": False,
            "real_robot_commanded": False,
            "watson_connected": False,
            "motion_mode": "physx_acceleration_drive_position_velocity_targets",
            "controller_calibrated_to_tm5s": False,
            "tool_mass_and_inertia_calibrated": False,
            "quick_changer_modeled": False,
            "fingers_fixed": True,
            "contact_or_grasp_simulated": False,
            "selected_pin_attached": False,
            "selected_pin_visual_only": True,
            "collision_proxies_visual_only_in_isaac": True,
            "collision_validation_source": "standalone_cumotion_sampled_xrdf_spheres",
            "selected_detection_id": plan["selected"]["detection_id"],
            "selected_truth_id": plan["selected"]["truth_id"],
            "minimum_planned_sampled_sphere_clearance_m": plan["selected"][
                "minimum_sampled_sphere_clearance_m"
            ],
            "synthetic_detection_error": plan["selected"]["synthetic_detection_error"],
            "joint_names": EXPECTED_DOF_NAMES,
            "joint_limits_radians": joint_limits.tolist(),
            "imported_proportional_gains": np.asarray(imported_proportional_gains).tolist(),
            "imported_derivative_gains": np.asarray(imported_derivative_gains).tolist(),
            "applied_simulation_proportional_gains": np.asarray(
                applied_proportional_gains
            ).tolist(),
            "applied_simulation_derivative_gains": np.asarray(
                applied_derivative_gains
            ).tolist(),
            "simulation_gains_saved_to_usd": False,
            "imported_maximum_efforts": np.asarray(maximum_efforts).tolist(),
            "simulated_seconds": simulated_seconds,
            "wall_seconds": time.perf_counter() - started_wall,
            "physics_step_count": step_count,
            "completed_cycles": cycle_count,
            "interrupted": interrupted,
            "stop_button_requested": panel_state["stop_requested"],
            "maximum_tracking_error_rad": maximum_tracking_error,
            "maximum_tracking_error_per_joint_rad": maximum_tracking_error_per_joint.tolist(),
            "rms_tracking_error_rad": math.sqrt(
                squared_position_error_sum / max(1, position_error_values)
            ),
            "maximum_velocity_tracking_error_rad_s": maximum_velocity_error,
            "maximum_actual_velocity_rad_s": maximum_actual_velocity.tolist(),
            "maximum_actual_acceleration_rad_s2": maximum_actual_acceleration.tolist(),
            "maximum_endpoint_error_rad": maximum_endpoint_error,
            "endpoint_errors": endpoint_errors,
            "tracking_error_limit_rad": MAX_TRACKING_ERROR_RADIANS,
            "endpoint_error_limit_rad": MAX_ENDPOINT_ERROR_RADIANS,
            "scene_paths": scene_paths,
            "camera_eye": CAMERA_EYE,
            "camera_target": CAMERA_TARGET,
            "screenshot": str(screenshot_path) if screenshot_path else None,
            "screenshot_written": screenshot_written,
            "screenshot_capture_scheduled": screenshot_capture is not None,
            "screenshot_wait_result": screenshot_wait_result,
            "import_report_urdf_hash_match": True,
            "import_report_usd_hash_match": True,
            "warning": (
                "Simulator drive tracking only. Missing calibrated Quick Changer, pin-grasp "
                "TCP, 2FG7 inertia, contact, finger, and Techman controller models prevent "
                "hardware dynamics or grasp-success conclusions."
            ),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Synthetic pick report: {report_path}")
        exit_code = 0
        return 0
    except KeyboardInterrupt:
        exit_code = 0
        return 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        panel_window = None
        if world is not None and simulation_app.is_running():
            world.stop()
        simulation_app.close(wait_for_replicator=False, exit_code=exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
