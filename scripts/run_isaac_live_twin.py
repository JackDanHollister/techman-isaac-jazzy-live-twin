#!/usr/bin/env python3
"""Mirror Watson's live joint positions in a paused, read-only Isaac Sim view."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any

import numpy as np

from pin_axis_3d_sim.live_twin import (
    EXPECTED_JOINT_NAMES,
    JOINT_STATE_TOPIC,
    LiveJointStateBuffer,
    normalise_joint_state,
)


ARENA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_USD = (
    ARENA_DIR
    / "reference/seven_pin/isaac/tm5s_with_2fg7/tm5s_with_2fg7.usda"
)
DEFAULT_REPORT = ARENA_DIR / "outputs/isaac_sim/6.0.1/live_twin_report.json"
EXPECTED_PYTHON = (3, 12)
EXPECTED_ISAAC_PACKAGES = {
    "isaacsim": "6.0.1.0",
    "isaacsim-core": "6.0.1.0",
    "isaacsim-ros2": "6.0.1.0",
}
EXPECTED_DOMAIN_ID = "219"
EXPECTED_DISCOVERY_RANGE = "LOCALHOST"
EXPECTED_RMW = "rmw_fastrtps_cpp"
RENDER_DT_SECONDS = 1.0 / 60.0
MAX_PREFLIGHT_AGE_SECONDS = 180.0
MAX_TRACKING_ERROR_RADIANS = 1.0e-6
MAX_RENDERED_LINK_POSITION_ERROR_METERS = 1.0e-5
MAX_RENDERED_LINK_ORIENTATION_ERROR_RADIANS = 1.0e-5
CAMERA_EYE = [1.65, 1.45, 1.35]
CAMERA_TARGET = [0.0, -0.18, 0.60]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without a window; intended for integration validation.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Stop after this many wall-clock seconds; zero runs until closed.",
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=15.0,
        help="Fail if no valid Watson joint state arrives within this time.",
    )
    parser.add_argument(
        "--stale-after-seconds",
        type=float,
        default=0.25,
        help="Freeze and flag the mirror when the latest valid sample is older.",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="Capture one frame after the live stream has been stable for one second.",
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
            "Isaac Sim 6.0.1 live twin requires Python 3.12; "
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


def validate_ros_environment(environment: dict[str, str] | None = None) -> None:
    values = os.environ if environment is None else environment
    expected = {
        "ROS_DISTRO": "jazzy",
        "ROS_DOMAIN_ID": EXPECTED_DOMAIN_ID,
        "ROS_AUTOMATIC_DISCOVERY_RANGE": EXPECTED_DISCOVERY_RANGE,
        "RMW_IMPLEMENTATION": EXPECTED_RMW,
    }
    failures = [
        f"{name} must be {expected_value!r}; found {values.get(name)!r}"
        for name, expected_value in expected.items()
        if values.get(name) != expected_value
    ]
    if failures:
        raise RuntimeError("Isaac live-twin ROS isolation failed: " + "; ".join(failures))


def load_preflight_report(
    path: Path,
    *,
    now: datetime | None = None,
    max_age_seconds: float = MAX_PREFLIGHT_AGE_SECONDS,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    required = {
        "mode": "check",
        "status": "check_passed",
        "motion_commanded": False,
        "ros_domain_id": EXPECTED_DOMAIN_ID,
        "ros_automatic_discovery_range": EXPECTED_DISCOVERY_RANGE,
    }
    failures = [
        f"{name} is {report.get(name)!r}, expected {expected!r}"
        for name, expected in required.items()
        if report.get(name) != expected
    ]
    if report.get("health_failures") != []:
        failures.append(f"health_failures is {report.get('health_failures')!r}")
    stable_health = report.get("stable_health")
    if not isinstance(stable_health, dict):
        failures.append("stable_health is missing")
    else:
        if stable_health.get("robot_error") is not False:
            failures.append("stable health reports robot_error")
        if stable_health.get("robot_link") is not True:
            failures.append("stable health reports no robot link")
        if stable_health.get("error_code") != 0:
            failures.append(
                f"stable health error_code is {stable_health.get('error_code')!r}"
            )
    try:
        timestamp = datetime.fromisoformat(str(report["timestamp_utc"]))
        if timestamp.tzinfo is None:
            raise ValueError("timestamp has no timezone")
        current = datetime.now(timezone.utc) if now is None else now
        age = (current - timestamp.astimezone(timezone.utc)).total_seconds()
        if age < -1.0 or age > max_age_seconds:
            failures.append(
                f"preflight report age {age:.3f}s is outside [-1, {max_age_seconds}]s"
            )
    except (KeyError, TypeError, ValueError) as exc:
        failures.append(f"preflight timestamp is invalid: {exc}")
        age = float("nan")
    if failures:
        raise ValueError("Watson live-twin preflight failed: " + "; ".join(failures))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "timestamp_utc": report["timestamp_utc"],
        "age_seconds_at_launch": age,
        "initial_joint_positions": stable_health["feedback_joint_positions"],
    }


def add_live_scene(stage: Any) -> list[str]:
    from pxr import Gf, UsdGeom, UsdLux

    created_paths: list[str] = []
    floor = UsdGeom.Cube.Define(stage, "/LiveTwin/Floor")
    floor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.015))
    floor.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 0.015))
    floor.GetSizeAttr().Set(2.0)
    floor.GetDisplayColorAttr().Set([Gf.Vec3f(0.10, 0.13, 0.18)])
    dome = UsdLux.DomeLight.Define(stage, "/LiveTwin/Lights/Dome")
    dome.CreateIntensityAttr(950.0)
    key = UsdLux.DistantLight.Define(stage, "/LiveTwin/Lights/Key")
    key.CreateIntensityAttr(2800.0)
    key.CreateAngleAttr(1.0)
    created_paths.extend(
        ["/LiveTwin/Floor", "/LiveTwin/Lights/Dome", "/LiveTwin/Lights/Key"]
    )
    return created_paths


def create_status_panel(ui: Any) -> tuple[Any, Any, Any, dict[str, bool]]:
    state = {"stop_requested": False}
    window = ui.Window("Watson Live Isaac Twin", width=520, height=285)
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label(
                "LIVE ARM-JOINT MIRROR - CANNOT COMMAND WATSON",
                style={"color": 0xFF55DD55, "font_size": 18},
            )
            status_label = ui.Label(
                "WAITING FOR /watson/joint_states",
                style={"color": 0xFF44AAFF, "font_size": 17},
            )
            details_label = ui.Label("Initialising ROS subscription...")
            ui.Label(
                "Physics is paused. Joint positions are copied kinematically from the "
                "real arm; no extrapolation, controller, action client, or command "
                "publisher exists here.\nTool/QC geometry is provisional and not a "
                "calibrated collision twin.",
                word_wrap=True,
            )

            def request_stop() -> None:
                state["stop_requested"] = True

            ui.Button("Stop and close live twin", height=34, clicked_fn=request_stop)
    return window, status_label, details_label, state


def quaternion_angular_error_radians(first: np.ndarray, second: np.ndarray) -> float:
    """Return the shortest angular distance between two xyzw quaternions."""

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first.shape != (4,) or second.shape != (4,) or first_norm <= 0.0 or second_norm <= 0.0:
        raise ValueError("rendered-link quaternions must be non-zero four-vectors")
    dot = abs(float(np.dot(first / first_norm, second / second_norm)))
    return 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))


def rendered_link_pose(
    stage: Any,
    robot: Any,
    *,
    asset_prim_path: str,
    link_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read matching PhysX and USD world poses for one rendered robot link."""

    from pxr import Usd, UsdGeom

    body_names = tuple(robot._articulation_view.body_names)
    try:
        body_index = body_names.index(link_name)
    except ValueError as exc:
        raise RuntimeError(
            f"Imported articulation has no {link_name!r} body: {list(body_names)}"
        ) from exc
    physics_transforms = np.asarray(
        robot._articulation_view._physics_view.get_link_transforms(),
        dtype=np.float64,
    )
    if physics_transforms.shape != (1, robot.num_bodies, 7):
        raise RuntimeError(
            f"Unexpected PhysX link-transform shape: {physics_transforms.shape}"
        )
    physics_pose = physics_transforms[0, body_index]

    asset_prim = stage.GetPrimAtPath(asset_prim_path)
    visual_candidates = [
        prim for prim in Usd.PrimRange(asset_prim) if prim.GetName() == link_name
    ]
    if len(visual_candidates) != 1:
        raise RuntimeError(
            f"Expected one rendered {link_name!r} prim below {asset_prim_path}; "
            f"found {[str(prim.GetPath()) for prim in visual_candidates]}"
        )
    visual_prim = visual_candidates[0]
    matrix = UsdGeom.Xformable(visual_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    translation = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    rendered_position = np.asarray(
        [translation[0], translation[1], translation[2]], dtype=np.float64
    )
    rendered_orientation = np.asarray(
        [imaginary[0], imaginary[1], imaginary[2], rotation.GetReal()],
        dtype=np.float64,
    )
    return physics_pose[:3], physics_pose[3:], rendered_position, rendered_orientation


def sync_rendered_articulation_pose(world: Any, physx_interface: Any) -> None:
    """Publish a teleported articulation pose to USD without stepping dynamics."""

    world.physics_sim_view.update_articulations_kinematic()
    physx_interface.update_transformations(False, True, False)


def own_ros_graph_snapshot(node: Any) -> tuple[dict[str, Any], list[str]]:
    from rclpy.action.graph import (
        get_action_client_names_and_types_by_node,
        get_action_server_names_and_types_by_node,
    )

    name = node.get_name()
    namespace = node.get_namespace()
    publishers = node.get_publisher_names_and_types_by_node(name, namespace)
    subscriptions = node.get_subscriber_names_and_types_by_node(name, namespace)
    services = node.get_service_names_and_types_by_node(name, namespace)
    clients = node.get_client_names_and_types_by_node(name, namespace)
    action_clients = get_action_client_names_and_types_by_node(node, name, namespace)
    action_servers = get_action_server_names_and_types_by_node(node, name, namespace)
    snapshot = {
        "node_name": name,
        "node_namespace": namespace,
        "publishers": publishers,
        "subscriptions": subscriptions,
        "services": services,
        "clients": clients,
        "action_clients": action_clients,
        "action_servers": action_servers,
    }
    failures: list[str] = []
    expected_subscription = [(JOINT_STATE_TOPIC, ["sensor_msgs/msg/JointState"])]
    if subscriptions != expected_subscription:
        failures.append(f"unexpected subscriptions: {subscriptions}")
    if action_clients:
        failures.append(f"action clients are forbidden: {action_clients}")
    if action_servers:
        failures.append(f"action servers are forbidden: {action_servers}")
    command_tokens = (
        "execute_trajectory",
        "follow_joint_trajectory",
        "move_action",
        "sequence_move_group",
        "joint_trajectory",
    )
    for category, endpoints in (
        ("publisher", publishers),
        ("service", services),
        ("client", clients),
    ):
        forbidden = [
            endpoint
            for endpoint in endpoints
            if any(token in endpoint[0] for token in command_tokens)
        ]
        if forbidden:
            failures.append(f"forbidden command {category} endpoints: {forbidden}")
    return snapshot, failures


def main() -> int:
    args = build_parser().parse_args()
    package_versions = validate_runtime()
    validate_ros_environment()
    if args.duration_seconds < 0.0:
        raise ValueError("--duration-seconds must be non-negative")
    if args.headless and args.duration_seconds <= 0.0:
        raise ValueError("--headless requires a positive --duration-seconds")
    if args.startup_timeout_seconds <= 0.0:
        raise ValueError("--startup-timeout-seconds must be positive")

    usd_path = args.usd.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    preflight_path = args.preflight_report.expanduser().resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(f"Imported TM5S USD is missing: {usd_path}")
    preflight = load_preflight_report(preflight_path)
    if args.screenshot is None:
        screenshot_path = None
    else:
        screenshot_path = args.screenshot.expanduser().resolve()
        if screenshot_path.exists():
            raise FileExistsError(f"Refusing to overwrite screenshot: {screenshot_path}")
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)

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
    node = None
    subscription = None
    ros_context = None
    ros_executor = None
    panel_window = None
    started_wall_time = time.perf_counter()
    try:
        import omni.ui as ui
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.rendering_manager import ViewportManager
        from isaacsim.core.utils.extensions import enable_extension
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        from omni.physx import get_physx_interface

        enable_extension("isaacsim.ros2.bridge")
        for _ in range(4):
            simulation_app.update()

        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import JointState

        World.clear_instance()
        stage = omni.usd.get_context().get_stage()
        default_prim = stage.GetDefaultPrim()
        if not default_prim.IsValid():
            raise RuntimeError("Imported TM5S stage has no default prim")
        asset_prim_path = str(default_prim.GetPath())
        scene_paths = add_live_scene(stage)
        world = World(
            physics_dt=RENDER_DT_SECONDS,
            rendering_dt=RENDER_DT_SECONDS,
            stage_units_in_meters=1.0,
            backend="numpy",
            device="cpu",
        )
        robot = world.scene.add(
            SingleArticulation(prim_path=asset_prim_path, name="watson_live_twin")
        )
        world.reset()
        world.pause()
        if not robot.handles_initialized:
            raise RuntimeError("TM5S articulation handles did not initialise")
        articulation_joint_names = tuple(robot.dof_names)
        if not all(name in articulation_joint_names for name in EXPECTED_JOINT_NAMES):
            raise RuntimeError(
                "Imported articulation does not contain the six Techman joints: "
                f"{list(articulation_joint_names)}"
            )
        arm_dof_indices = np.asarray(
            [articulation_joint_names.index(name) for name in EXPECTED_JOINT_NAMES],
            dtype=np.int64,
        )
        dof_properties = robot.dof_properties
        articulation_limits = np.column_stack(
            (
                np.asarray(dof_properties["lower"], dtype=np.float64),
                np.asarray(dof_properties["upper"], dtype=np.float64),
            )
        )
        joint_limits = articulation_limits[arm_dof_indices]
        if joint_limits.shape != (6, 2) or not np.all(np.isfinite(joint_limits)):
            raise RuntimeError(f"Unexpected imported joint limits: {joint_limits}")
        initial_positions, _ = normalise_joint_state(
            EXPECTED_JOINT_NAMES,
            preflight["initial_joint_positions"],
            [0.0] * 6,
            joint_limits=joint_limits,
        )
        initial = np.asarray(initial_positions, dtype=np.float64)
        display_positions = np.zeros(robot.num_dof, dtype=np.float64)
        display_velocities = np.zeros(robot.num_dof, dtype=np.float64)
        display_positions[arm_dof_indices] = initial
        if np.any(display_positions < articulation_limits[:, 0]) or np.any(
            display_positions > articulation_limits[:, 1]
        ):
            raise RuntimeError("Initial live-twin state exceeds articulation limits")
        robot.set_joints_default_state(
            positions=display_positions,
            velocities=display_velocities,
        )
        robot.set_joint_positions(display_positions)
        robot.set_joint_velocities(display_velocities)
        physx_interface = get_physx_interface()
        sync_rendered_articulation_pose(world, physx_interface)

        buffer = LiveJointStateBuffer(stale_after_seconds=args.stale_after_seconds)
        ros_context = Context()
        rclpy.init(
            context=ros_context,
            signal_handler_options=SignalHandlerOptions.NO,
        )
        node = Node(
            "watson_live_twin",
            namespace="/isaac",
            context=ros_context,
            enable_rosout=False,
            start_parameter_services=False,
            enable_logger_service=False,
        )
        ros_executor = SingleThreadedExecutor(context=ros_context)
        ros_executor.add_node(node)

        def joint_state_callback(message: Any) -> None:
            try:
                buffer.accept(
                    message.name,
                    message.position,
                    message.velocity,
                    stamp_seconds=message.header.stamp.sec,
                    stamp_nanoseconds=message.header.stamp.nanosec,
                    joint_limits=joint_limits,
                )
            except ValueError:
                return

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        subscription = node.create_subscription(
            JointState,
            JOINT_STATE_TOPIC,
            joint_state_callback,
            qos,
        )
        for _ in range(10):
            ros_executor.spin_once(timeout_sec=0.02)
        graph_snapshot, graph_failures = own_ros_graph_snapshot(node)
        if graph_failures:
            raise RuntimeError(
                "Isaac live-twin command-path audit failed: "
                + "; ".join(graph_failures)
            )

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
        if args.headless:
            status_label = None
            details_label = None
            panel_state = {"stop_requested": False}
        else:
            panel_window, status_label, details_label, panel_state = create_status_panel(ui)

        loop_started = time.monotonic()
        last_status = ""
        max_tracking_error = 0.0
        max_rendered_link_position_error = 0.0
        max_rendered_link_orientation_error = 0.0
        max_message_age = 0.0
        max_display_step = 0.0
        previous_display = initial.copy()
        applied_updates = 0
        stale_frames = 0
        screenshot_requested = False
        screenshot_capture = None
        screenshot_future = None
        first_live_time = None

        while simulation_app.is_running() and not panel_state["stop_requested"]:
            frame_started = time.perf_counter()
            ros_executor.spin_once(timeout_sec=0.0)
            now = time.monotonic()
            status = buffer.status(now_monotonic_s=now)
            age = buffer.age_seconds(now_monotonic_s=now)
            if age is not None:
                max_message_age = max(max_message_age, age)
            if status == "LIVE" and buffer.sample is not None:
                command = np.asarray(buffer.sample.positions, dtype=np.float64)
                max_display_step = max(
                    max_display_step,
                    float(np.max(np.abs(command - previous_display))),
                )
                display_positions[arm_dof_indices] = command
                display_velocities[arm_dof_indices] = 0.0
                robot.set_joint_positions(display_positions)
                robot.set_joint_velocities(display_velocities)
                sync_rendered_articulation_pose(world, physx_interface)
                actual = np.asarray(robot.get_joint_positions(), dtype=np.float64)
                error = float(
                    np.max(np.abs(actual[arm_dof_indices] - command))
                )
                max_tracking_error = max(max_tracking_error, error)
                (
                    physics_link_position,
                    physics_link_orientation,
                    rendered_link_position,
                    rendered_link_orientation,
                ) = rendered_link_pose(
                    stage,
                    robot,
                    asset_prim_path=asset_prim_path,
                    link_name="link_6",
                )
                rendered_position_error = float(
                    np.linalg.norm(rendered_link_position - physics_link_position)
                )
                rendered_orientation_error = quaternion_angular_error_radians(
                    rendered_link_orientation, physics_link_orientation
                )
                max_rendered_link_position_error = max(
                    max_rendered_link_position_error, rendered_position_error
                )
                max_rendered_link_orientation_error = max(
                    max_rendered_link_orientation_error, rendered_orientation_error
                )
                previous_display = command.copy()
                applied_updates += 1
                if first_live_time is None:
                    first_live_time = now
            elif status == "STALE":
                stale_frames += 1

            if status != last_status:
                print(f"Isaac live twin: {status}", flush=True)
                last_status = status
            if status_label is not None:
                if status == "LIVE":
                    status_label.text = "LIVE - MIRRORING WATSON"
                    status_label.style = {"color": 0xFF55DD55, "font_size": 17}
                elif status == "STALE":
                    status_label.text = "STALE - MIRROR FROZEN"
                    status_label.style = {"color": 0xFF4444FF, "font_size": 17}
                else:
                    status_label.text = "WAITING FOR /watson/joint_states"
                    status_label.style = {"color": 0xFF44AAFF, "font_size": 17}
            if details_label is not None:
                age_text = "n/a" if age is None else f"{age * 1000.0:.1f} ms"
                j6_deg = float(np.degrees(previous_display[-1]))
                details_label.text = (
                    f"Rate: {buffer.observed_rate_hz():.1f} Hz | age: {age_text} | "
                    f"J6: {j6_deg:+.2f} deg | valid: {buffer.valid_messages} | "
                    f"invalid: {buffer.invalid_messages}"
                )

            world.render()
            if buffer.valid_messages == 0 and now - loop_started > args.startup_timeout_seconds:
                raise RuntimeError(
                    "No valid Watson joint state reached Isaac before the startup timeout"
                )
            if (
                screenshot_path is not None
                and first_live_time is not None
                and now - first_live_time >= 1.0
                and not screenshot_requested
            ):
                viewport = get_active_viewport()
                if viewport is None:
                    raise RuntimeError("No active viewport is available for screenshot")
                screenshot_capture = capture_viewport_to_file(
                    viewport, file_path=str(screenshot_path)
                )
                screenshot_future = asyncio.ensure_future(
                    screenshot_capture.wait_for_result(completion_frames=30)
                )
                screenshot_requested = True
            if args.duration_seconds > 0.0 and now - loop_started >= args.duration_seconds:
                break
            remaining = RENDER_DT_SECONDS - (time.perf_counter() - frame_started)
            if remaining > 0.0:
                time.sleep(remaining)

        if buffer.valid_messages == 0 or applied_updates == 0:
            raise RuntimeError("Isaac live twin ended without applying a Watson joint state")
        if max_tracking_error > MAX_TRACKING_ERROR_RADIANS:
            raise RuntimeError(
                f"Live mirror tracking error {max_tracking_error}rad exceeds "
                f"{MAX_TRACKING_ERROR_RADIANS}rad"
            )
        if max_rendered_link_position_error > MAX_RENDERED_LINK_POSITION_ERROR_METERS:
            raise RuntimeError(
                "Rendered link_6 position disagrees with PhysX by "
                f"{max_rendered_link_position_error}m"
            )
        if (
            max_rendered_link_orientation_error
            > MAX_RENDERED_LINK_ORIENTATION_ERROR_RADIANS
        ):
            raise RuntimeError(
                "Rendered link_6 orientation disagrees with PhysX by "
                f"{max_rendered_link_orientation_error}rad"
            )
        screenshot_wait_result = None
        if screenshot_path is not None and screenshot_future is None:
            raise RuntimeError("Screenshot was requested but never scheduled")
        if screenshot_future is not None:
            for _ in range(180):
                if screenshot_future.done() or not simulation_app.is_running():
                    break
                world.render()
            if not screenshot_future.done():
                raise RuntimeError("Screenshot capture did not finish within 180 frames")
            screenshot_wait_result = bool(screenshot_future.result())
            if not screenshot_wait_result:
                raise RuntimeError("Isaac screenshot capture returned failure")
            import omni.kit.renderer_capture

            omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
        screenshot_written = screenshot_path is None or screenshot_path.is_file()
        if not screenshot_written:
            raise RuntimeError(f"Screenshot capture did not complete: {screenshot_path}")

        final_articulation_positions = np.asarray(
            robot.get_joint_positions(),
            dtype=np.float64,
        )
        final_positions = final_articulation_positions[arm_dof_indices]
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv],
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "package_versions": package_versions,
            "status": "live_joint_mirror_passed",
            "validation_scope": "read_only_live_arm_joint_mirror",
            "source_usd": str(usd_path),
            "source_usd_sha256": sha256_file(usd_path),
            "preflight": preflight,
            "ros_domain_id": os.environ["ROS_DOMAIN_ID"],
            "ros_automatic_discovery_range": os.environ[
                "ROS_AUTOMATIC_DISCOVERY_RANGE"
            ],
            "rmw_implementation": os.environ["RMW_IMPLEMENTATION"],
            "source_topic": JOINT_STATE_TOPIC,
            "ros_graph": graph_snapshot,
            "robot_command_publishers": [],
            "robot_command_service_clients": [],
            "robot_command_action_clients": [],
            "real_robot_commanded": False,
            "physics_timeline_paused": True,
            "physx_dynamics_stepped": False,
            "motion_mode": "paused_kinematic_joint_position_mirror",
            "stale_behavior": "freeze_last_valid_pose_without_extrapolation",
            "stale_after_seconds": args.stale_after_seconds,
            "valid_messages": buffer.valid_messages,
            "invalid_messages": buffer.invalid_messages,
            "observed_message_rate_hz": buffer.observed_rate_hz(),
            "applied_updates": applied_updates,
            "stale_frames": stale_frames,
            "max_message_age_seconds": max_message_age,
            "max_tracking_error_radians": max_tracking_error,
            "max_rendered_link_position_error_meters": (
                max_rendered_link_position_error
            ),
            "max_rendered_link_orientation_error_radians": (
                max_rendered_link_orientation_error
            ),
            "final_physics_link_6_position_meters": physics_link_position.tolist(),
            "final_rendered_link_6_position_meters": rendered_link_position.tolist(),
            "max_display_step_radians": max_display_step,
            "joint_names": list(EXPECTED_JOINT_NAMES),
            "articulation_joint_names": list(articulation_joint_names),
            "joint_limits_radians": joint_limits.tolist(),
            "final_joint_positions": final_positions.tolist(),
            "tool_model_status": "provisional_2fg7_without_verified_watson_qc_yaw_or_tcp",
            "scene_paths": scene_paths,
            "camera_eye": CAMERA_EYE,
            "camera_target": CAMERA_TARGET,
            "headless": args.headless,
            "active_gpu": 0,
            "multi_gpu": False,
            "wall_seconds": time.perf_counter() - started_wall_time,
            "screenshot": str(screenshot_path) if screenshot_path is not None else None,
            "screenshot_written": screenshot_written,
            "screenshot_capture_scheduled": screenshot_capture is not None,
            "screenshot_wait_result": screenshot_wait_result,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Isaac live-twin report: {report_path}")
        exit_code = 0
        return 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        panel_window = None
        if node is not None:
            if subscription is not None:
                node.destroy_subscription(subscription)
            if ros_executor is not None:
                ros_executor.remove_node(node)
            node.destroy_node()
        if ros_executor is not None:
            ros_executor.shutdown(timeout_sec=1.0)
        if ros_context is not None:
            try:
                import rclpy

                if rclpy.ok(context=ros_context):
                    rclpy.shutdown(context=ros_context)
            except Exception:
                traceback.print_exc()
        if world is not None and simulation_app.is_running():
            world.stop()
        simulation_app.close(wait_for_replicator=False, exit_code=exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
