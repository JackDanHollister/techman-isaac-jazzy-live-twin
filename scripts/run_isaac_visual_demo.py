#!/usr/bin/env python3
"""Show the imported TM5S + 2FG7 moving in a camera-free Isaac Sim scene.

The demo is simulation-only. It creates no ROS, perception sensor, network, or
Watson connection. With physics paused, it animates the exact five raw knots
from the audited cuMotion static-obstacle benchmark and smoothly follows the
same joint-space lines between them. This is not a controller, dynamics,
calibration, or hardware-safety validation.
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
import yaml


ARENA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_USD = (
    ARENA_DIR
    / "generated/isaac/6.0.1/tm5s_with_2fg7/tm5s_with_2fg7.usda"
)
DEFAULT_CONFIG = ARENA_DIR / "config/isaac_visual_demo.yaml"
DEFAULT_REPORT = ARENA_DIR / "outputs/isaac_sim/6.0.1/visual_demo_report.json"
EXPECTED_PYTHON = (3, 12)
EXPECTED_ISAAC_PACKAGES = {
    "isaacsim": "6.0.1.0",
    "isaacsim-asset": "6.0.1.0",
    "isaacsim-core": "6.0.1.0",
}
EXPECTED_DOF_NAMES = [f"joint_{index}" for index in range(1, 7)]
PHYSICS_DT_SECONDS = 1.0 / 60.0
MAX_TRACKING_ERROR_RADIANS = 1.0e-3
MAX_COMMAND_STEP_RADIANS = 1.0e-2
CAMERA_EYE = [1.65, 1.45, 1.35]
CAMERA_TARGET = [0.0, -0.18, 0.60]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without a window; intended for automated validation.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Stop after this many simulated seconds; zero runs until the window closes.",
    )
    parser.add_argument(
        "--transition-seconds",
        type=float,
        default=None,
        help="Override the config transition duration.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=None,
        help="Override the config waypoint hold duration.",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Do not pace steps to wall-clock time; useful with --headless.",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="Capture one rendered frame to a new PNG path after one simulated second.",
    )
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_runtime() -> dict[str, str]:
    if sys.version_info[:2] != EXPECTED_PYTHON:
        raise RuntimeError(
            "Isaac Sim 6.0.1 visual demo requires Python 3.12; "
            f"found {sys.version.split()[0]}"
        )
    versions: dict[str, str] = {}
    for package_name, expected_version in EXPECTED_ISAAC_PACKAGES.items():
        try:
            actual_version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Missing required package: {package_name}") from exc
        if actual_version != expected_version:
            raise RuntimeError(
                f"{package_name} must be {expected_version}; found {actual_version}"
            )
        versions[package_name] = actual_version
    return versions


def load_visual_demo_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("format_version") != 1:
        raise ValueError("Visual demo config must use format_version: 1")
    raw_waypoints = config.get("waypoints")
    forward_sequence = config.get("forward_sequence")
    sequence = config.get("motion_sequence")
    if not isinstance(raw_waypoints, dict) or not raw_waypoints:
        raise ValueError("Visual demo config requires a non-empty waypoints mapping")
    if not isinstance(forward_sequence, list) or len(forward_sequence) < 2:
        raise ValueError("Visual demo config requires at least two forward waypoints")
    if not isinstance(sequence, list) or len(sequence) < 3:
        raise ValueError("Visual demo config requires at least three sequence entries")

    waypoints: dict[str, np.ndarray] = {}
    for name, values in raw_waypoints.items():
        waypoint = np.asarray(values, dtype=np.float64)
        if waypoint.shape != (6,) or not np.all(np.isfinite(waypoint)):
            raise ValueError(f"Waypoint {name!r} must contain six finite joint values")
        waypoints[str(name)] = waypoint
    unknown = [
        name
        for name in [*forward_sequence, *sequence]
        if name not in waypoints
    ]
    if unknown:
        raise ValueError(f"Motion sequence refers to unknown waypoints: {unknown}")
    if len(set(forward_sequence)) != len(forward_sequence):
        raise ValueError("Forward benchmark waypoint names must be unique")
    if set(waypoints) != set(forward_sequence):
        raise ValueError("Waypoints must exactly match the forward benchmark sequence")
    expected_ping_pong = [*forward_sequence, *reversed(forward_sequence[:-1])]
    if sequence != expected_ping_pong:
        raise ValueError(
            "Motion sequence must follow the benchmark forward and then retrace it"
        )

    obstacle = config.get("benchmark_obstacle")
    if not isinstance(obstacle, dict):
        raise ValueError("Visual demo config requires benchmark_obstacle")
    obstacle_position = np.asarray(obstacle.get("position"), dtype=np.float64)
    obstacle_sides = np.asarray(obstacle.get("side_lengths"), dtype=np.float64)
    if (
        obstacle_position.shape != (3,)
        or obstacle_sides.shape != (3,)
        or not np.all(np.isfinite(obstacle_position))
        or not np.all(np.isfinite(obstacle_sides))
        or np.any(obstacle_sides <= 0.0)
    ):
        raise ValueError("Benchmark obstacle requires a finite position and positive sides")

    forward_path = np.asarray(
        [waypoints[name] for name in forward_sequence], dtype="<f8"
    )
    forward_hash = hashlib.sha256(forward_path.tobytes(order="C")).hexdigest()
    expected_hash = config.get("source_forward_raw_path_float64_sha256")
    if forward_hash != expected_hash:
        raise ValueError(
            "Configured forward waypoint hash does not match the source benchmark hash"
        )
    transition_seconds = float(config.get("transition_seconds", 0.0))
    hold_seconds = float(config.get("hold_seconds", 0.0))
    if transition_seconds <= 0.0 or hold_seconds < 0.0:
        raise ValueError("transition_seconds must be positive and hold_seconds non-negative")
    return {
        **config,
        "waypoints": waypoints,
        "forward_sequence": [str(name) for name in forward_sequence],
        "motion_sequence": [str(name) for name in sequence],
        "benchmark_obstacle": {
            "position": obstacle_position,
            "side_lengths": obstacle_sides,
        },
        "computed_forward_raw_path_float64_sha256": forward_hash,
        "transition_seconds": transition_seconds,
        "hold_seconds": hold_seconds,
    }


def validate_benchmark_source(config: dict[str, Any]) -> dict[str, Any]:
    relative_bundle = Path(str(config.get("source_benchmark_bundle", "")))
    if relative_bundle.is_absolute():
        raise ValueError("source_benchmark_bundle must be relative to the demo directory")
    bundle = (ARENA_DIR / relative_bundle).resolve()
    allowed_root = (ARENA_DIR / "outputs/cumotion_benchmark").resolve()
    if bundle == allowed_root or allowed_root not in bundle.parents:
        raise ValueError("source_benchmark_bundle must stay under outputs/cumotion_benchmark")

    summary_path = bundle / "summary.json"
    audit_path = bundle / "mesh_audit.json"
    if not summary_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError(
            f"Visual demo benchmark evidence is incomplete under {bundle}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    source_case = config.get("source_case")
    matching_cases = [case for case in summary["cases"] if case["case"] == source_case]
    if len(matching_cases) != 1:
        raise ValueError(f"Benchmark summary does not contain one {source_case!r} case")
    case_summary = matching_cases[0]
    if (
        case_summary["accepted_paths"] != case_summary["trials"]
        or case_summary["trials"] <= 0
    ):
        raise ValueError(f"Source benchmark case {source_case!r} was not fully accepted")

    obstacle_audit = audit["obstacle_path_audit"]
    if obstacle_audit["case"] != source_case:
        raise ValueError("Mesh audit case does not match the visual demo source case")
    audit_hash = obstacle_audit["reconstruction"]["raw_path_float64_sha256"]
    if audit_hash != config["computed_forward_raw_path_float64_sha256"]:
        raise ValueError("Visual waypoints do not reproduce the independently audited path")
    audited_obstacle = obstacle_audit["obstacle"]
    if not np.array_equal(
        config["benchmark_obstacle"]["position"],
        np.asarray(audited_obstacle["position_m"], dtype=np.float64),
    ) or not np.array_equal(
        config["benchmark_obstacle"]["side_lengths"],
        np.asarray(audited_obstacle["side_lengths_m"], dtype=np.float64),
    ):
        raise ValueError("Displayed benchmark obstacle does not match the mesh audit")

    return {
        "bundle": str(bundle),
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "mesh_audit": str(audit_path),
        "mesh_audit_sha256": sha256_file(audit_path),
        "source_case": source_case,
        "accepted_paths": case_summary["accepted_paths"],
        "trials": case_summary["trials"],
        "minimum_sampled_sphere_clearance_m": case_summary[
            "minimum_sampled_sphere_clearance_m"
        ],
        "minimum_sampled_mesh_vertex_clearance_m": obstacle_audit["aggregate"][
            "minimum_sampled_mesh_vertex_clearance_m"
        ],
        "triangle_aabb_intersection_pairs": obstacle_audit["aggregate"][
            "triangle_aabb_intersection_pairs"
        ],
        "raw_path_float64_sha256": audit_hash,
        "raw_knots": obstacle_audit["reconstruction"]["raw_knots"],
        "audit_max_joint_step_rad": obstacle_audit["reconstruction"][
            "max_joint_step_rad"
        ],
    }


def smoothstep_cosine(amount: float) -> float:
    clamped = min(1.0, max(0.0, amount))
    return 0.5 - 0.5 * math.cos(math.pi * clamped)


def maximum_command_step_bound(
    config: dict[str, Any], transition_seconds: float
) -> float:
    """Return the maximum per-frame delta implied by cosine interpolation."""
    sequence = config["motion_sequence"]
    waypoints = config["waypoints"]
    largest_segment_delta = max(
        float(np.max(np.abs(waypoints[end] - waypoints[start])))
        for start, end in zip(sequence, sequence[1:])
    )
    return (
        largest_segment_delta
        * math.pi
        / (2.0 * transition_seconds)
        * PHYSICS_DT_SECONDS
    )


def command_for_time(
    config: dict[str, Any],
    simulated_seconds: float,
    transition_seconds: float,
    hold_seconds: float,
) -> tuple[np.ndarray, str, int]:
    sequence = config["motion_sequence"]
    waypoints = config["waypoints"]
    segment_seconds = transition_seconds + hold_seconds
    segments_per_cycle = len(sequence) - 1
    cycle_seconds = segment_seconds * segments_per_cycle
    cycle_index = int(simulated_seconds // cycle_seconds)
    within_cycle = simulated_seconds - cycle_index * cycle_seconds
    segment_index = min(int(within_cycle // segment_seconds), segments_per_cycle - 1)
    within_segment = within_cycle - segment_index * segment_seconds
    start_name = sequence[segment_index]
    end_name = sequence[segment_index + 1]
    alpha = smoothstep_cosine(min(within_segment, transition_seconds) / transition_seconds)
    command = waypoints[start_name] + alpha * (waypoints[end_name] - waypoints[start_name])
    status = f"{start_name} -> {end_name}"
    if within_segment >= transition_seconds:
        status = f"holding {end_name}"
    return command, status, cycle_index


def add_visual_workcell(stage: Any, config: dict[str, Any]) -> list[str]:
    from isaacsim.core.api.objects import VisualCuboid
    from pxr import UsdLux

    created_paths: list[str] = []

    def cuboid(path: str, position: list[float], scale: list[float], color: list[float]) -> None:
        VisualCuboid(
            prim_path=path,
            position=np.asarray(position),
            scale=np.asarray(scale),
            size=1.0,
            color=np.asarray(color),
        )
        created_paths.append(path)

    cuboid("/Demo/Floor", [0.0, 0.0, -0.015], [2.0, 2.0, 0.03], [0.16, 0.18, 0.22])
    obstacle = config["benchmark_obstacle"]
    cuboid(
        "/Demo/AuditedObstacle",
        obstacle["position"].tolist(),
        obstacle["side_lengths"].tolist(),
        [0.95, 0.35, 0.06],
    )

    dome = UsdLux.DomeLight.Define(stage, "/Demo/Lights/Dome")
    dome.CreateIntensityAttr(900.0)
    distant = UsdLux.DistantLight.Define(stage, "/Demo/Lights/Key")
    distant.CreateIntensityAttr(2500.0)
    distant.CreateAngleAttr(1.0)
    created_paths.extend(["/Demo/Lights/Dome", "/Demo/Lights/Key"])
    return created_paths


def create_status_panel(ui: Any) -> tuple[Any, Any, dict[str, bool]]:
    state = {"stop_requested": False}
    window = ui.Window("TM5S Camera-Free Visual Demo", width=440, height=225)
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label(
                "SIMULATION ONLY - NOT CONNECTED TO WATSON",
                style={"color": 0xFF5A5AFF, "font_size": 17},
            )
            ui.Label("Orange block: audited cuMotion benchmark obstacle")
            status_label = ui.Label("Initialising six-axis articulation...")
            ui.Label(
                "Physics-paused animation through the exact five cuMotion knots.\n"
                "Checks are sampled; this is not dynamics or controller validation.",
                word_wrap=True,
            )

            def request_stop() -> None:
                state["stop_requested"] = True

            ui.Button("Stop and close demo", height=32, clicked_fn=request_stop)
    return window, status_label, state


def main() -> int:
    args = build_parser().parse_args()
    package_versions = validate_runtime()
    usd_path = args.usd.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(
            f"Imported Isaac USD is missing: {usd_path}\n"
            "Run OMNI_KIT_ACCEPT_EULA=YES ./scripts/run_isaac_import.sh first."
        )
    if not config_path.is_file():
        raise FileNotFoundError(f"Visual demo config is missing: {config_path}")
    if args.duration_seconds < 0.0:
        raise ValueError("--duration-seconds must be non-negative")
    if args.headless and args.duration_seconds <= 0.0:
        raise ValueError("--headless requires a positive --duration-seconds")
    if args.screenshot is not None:
        screenshot_path = args.screenshot.expanduser().resolve()
        if screenshot_path.exists():
            raise FileExistsError(f"Refusing to overwrite screenshot: {screenshot_path}")
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        screenshot_path = None

    demo_config = load_visual_demo_config(config_path)
    benchmark_source = validate_benchmark_source(demo_config)
    transition_seconds = (
        args.transition_seconds
        if args.transition_seconds is not None
        else demo_config["transition_seconds"]
    )
    hold_seconds = (
        args.hold_seconds
        if args.hold_seconds is not None
        else demo_config["hold_seconds"]
    )
    if transition_seconds <= 0.0 or hold_seconds < 0.0:
        raise ValueError("Transition duration must be positive and hold duration non-negative")
    command_step_bound = maximum_command_step_bound(demo_config, transition_seconds)
    if command_step_bound > MAX_COMMAND_STEP_RADIANS:
        minimum_seconds = (
            command_step_bound
            / MAX_COMMAND_STEP_RADIANS
            * transition_seconds
        )
        raise ValueError(
            "Transition duration is too short for the validated replay sampling: "
            f"use at least {minimum_seconds:.3f} seconds"
        )

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
    started_wall_time = time.perf_counter()
    try:
        import omni.ui as ui
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.rendering_manager import ViewportManager
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        World.clear_instance()
        stage = omni.usd.get_context().get_stage()
        default_prim = stage.GetDefaultPrim()
        if not default_prim.IsValid():
            raise RuntimeError("Imported TM5S stage has no default prim")
        asset_prim_path = str(default_prim.GetPath())
        created_scene_paths = add_visual_workcell(stage, demo_config)

        world = World(
            physics_dt=PHYSICS_DT_SECONDS,
            rendering_dt=PHYSICS_DT_SECONDS,
            stage_units_in_meters=1.0,
            backend="numpy",
            device="cpu",
        )
        robot = world.scene.add(
            SingleArticulation(prim_path=asset_prim_path, name="tm5s_visual_demo")
        )
        world.reset()
        world.pause()
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
        if joint_limits.shape != (6, 2) or not np.all(np.isfinite(joint_limits)):
            raise RuntimeError(f"Unexpected TM5S joint limits: {joint_limits}")
        for waypoint_name, waypoint in demo_config["waypoints"].items():
            if np.any(waypoint < joint_limits[:, 0]) or np.any(
                waypoint > joint_limits[:, 1]
            ):
                raise RuntimeError(
                    f"Visual waypoint {waypoint_name!r} exceeds the imported joint limits"
                )

        start = demo_config["waypoints"][demo_config["forward_sequence"][0]]
        zeros = np.zeros(6, dtype=np.float64)
        robot.set_joints_default_state(positions=start, velocities=zeros)
        robot.set_joint_positions(start)
        robot.set_joint_velocities(zeros)

        render = not args.headless or screenshot_path is not None
        if render:
            viewport_ready, waited_frames = ViewportManager.wait_for_viewport(max_frames=120)
            if not viewport_ready:
                raise RuntimeError(
                    f"Isaac viewport was not ready after {waited_frames} frames"
                )
            ViewportManager.set_camera_view(
                ViewportManager.get_camera(), eye=CAMERA_EYE, target=CAMERA_TARGET
            )
        if not args.headless:
            panel_window, status_label, panel_state = create_status_panel(ui)
        else:
            status_label = None
            panel_state = {"stop_requested": False}

        simulated_seconds = 0.0
        step_count = 0
        max_tracking_error = 0.0
        last_status = ""
        completed_cycles = 0
        screenshot_requested = False
        screenshot_capture = None
        screenshot_future = None
        previous_command = start.copy()
        max_command_step = 0.0
        interrupted = False

        try:
            while simulation_app.is_running() and not panel_state["stop_requested"]:
                step_started = time.perf_counter()
                command, status, cycle_index = command_for_time(
                    demo_config,
                    simulated_seconds,
                    transition_seconds,
                    hold_seconds,
                )
                completed_cycles = max(completed_cycles, cycle_index)
                max_command_step = max(
                    max_command_step,
                    float(np.max(np.abs(command - previous_command))),
                )
                robot.set_joint_positions(command)
                robot.set_joint_velocities(zeros)
                actual = np.asarray(robot.get_joint_positions(), dtype=np.float64)
                if actual.shape != (6,) or not np.all(np.isfinite(actual)):
                    raise RuntimeError(f"Visual replay produced invalid joints: {actual}")
                max_tracking_error = max(
                    max_tracking_error,
                    float(np.max(np.abs(actual - command))),
                )
                if status != last_status:
                    print(f"Visual demo: {status}")
                    if status_label is not None:
                        status_label.text = f"Motion: {status} | cycle {cycle_index + 1}"
                    last_status = status

                # World.render pumps the viewport/UI while explicitly suppressing physics.
                world.render()
                simulated_seconds += PHYSICS_DT_SECONDS
                step_count += 1
                previous_command = command.copy()
                if (
                    screenshot_path is not None
                    and not screenshot_requested
                    and simulated_seconds >= 1.0
                ):
                    viewport = get_active_viewport()
                    if viewport is None:
                        raise RuntimeError(
                            "No active viewport is available for screenshot capture"
                        )
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
                    remaining = PHYSICS_DT_SECONDS - (time.perf_counter() - step_started)
                    if remaining > 0.0:
                        time.sleep(remaining)
        except KeyboardInterrupt:
            interrupted = True
            print("Visual demo interrupted; writing the run report before closing")

        if max_tracking_error > MAX_TRACKING_ERROR_RADIANS:
            raise RuntimeError(
                "Visual replay exceeded the kinematic tracking tolerance: "
                f"{max_tracking_error} rad"
            )
        if max_command_step > MAX_COMMAND_STEP_RADIANS + 1.0e-12:
            raise RuntimeError(
                "Visual replay exceeded the command sampling limit: "
                f"{max_command_step} rad"
            )
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
        if screenshot_path is not None and not screenshot_written:
            raise RuntimeError(f"Screenshot capture did not complete: {screenshot_path}")

        final_positions = np.asarray(robot.get_joint_positions(), dtype=np.float64)
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv],
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "package_versions": package_versions,
            "validation_scope": "visible_simulation_only_physics_paused_benchmark_animation",
            "source_usd": str(usd_path),
            "source_usd_sha256": sha256_file(usd_path),
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "headless": args.headless,
            "active_gpu": 0,
            "multi_gpu": False,
            "configured_physx_device": "cpu",
            "physx_dynamics_stepped": False,
            "physics_timeline_paused": True,
            "physical_camera_or_depth_sensor_used": False,
            "viewport_camera_used": render,
            "ros_used": False,
            "real_robot_commanded": False,
            "watson_connected": False,
            "motion_mode": "kinematic_joint_state_animation_physics_paused",
            "benchmark_source": benchmark_source,
            "benchmark_collision_scope": (
                "sampled provisional XRDF spheres plus independent sampled mesh-to-cuboid audit"
            ),
            "displayed_benchmark_obstacle_geometry_audited": True,
            "visual_floor_collision_audited": False,
            "visual_props_physics_collision_enabled": False,
            "calibration_status": "provisional_gripper_mount_and_tcp_demo_only",
            "forward_sequence": demo_config["forward_sequence"],
            "motion_sequence": demo_config["motion_sequence"],
            "transition_seconds": transition_seconds,
            "hold_seconds": hold_seconds,
            "simulated_seconds": simulated_seconds,
            "wall_seconds": time.perf_counter() - started_wall_time,
            "step_count": step_count,
            "completed_cycles": completed_cycles,
            "interrupted": interrupted,
            "stop_button_requested": panel_state["stop_requested"],
            "max_tracking_error_radians": max_tracking_error,
            "maximum_command_step_bound_radians": command_step_bound,
            "max_command_step_radians": max_command_step,
            "joint_limits_radians": joint_limits.tolist(),
            "final_joint_positions": final_positions.tolist(),
            "scene_paths": created_scene_paths,
            "camera_eye": CAMERA_EYE,
            "camera_target": CAMERA_TARGET,
            "screenshot": str(screenshot_path) if screenshot_path is not None else None,
            "screenshot_written": screenshot_written,
            "screenshot_capture_scheduled": screenshot_capture is not None,
            "screenshot_wait_result": screenshot_wait_result,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Visual demo report: {report_path}")
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
