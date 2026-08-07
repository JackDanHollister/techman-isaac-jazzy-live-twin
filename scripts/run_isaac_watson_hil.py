#!/usr/bin/env python3
"""Run the one-shot Isaac GUI for Watson's seven-pin air replay.

Preview mode is entirely offline.  Dry-run and execute launch the existing
guarded Watson wrapper as the sole physical authority, mirror measured
``/watson/joint_states`` into the eight-DOF Isaac articulation, and use the
runner's structured events only for gripper and specimen presentation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import queue
import secrets
import signal
import stat
import subprocess
import sys
import threading
import time
import traceback
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ARENA_DIR = SCRIPT_DIR.parent
if str(ARENA_DIR) not in sys.path:
    sys.path.insert(0, str(ARENA_DIR))

import run_isaac_grasp_cycle as single
import run_isaac_live_twin as live
import run_isaac_multi_pin_verticalization as multi
from pin_axis_3d_sim.isaac_hil_timeline import subscribe_hil_timeline
from pin_axis_3d_sim.live_twin import (
    EXPECTED_JOINT_NAMES,
    JOINT_STATE_TOPIC,
    LiveJointStateBuffer,
)
from pin_axis_3d_sim.multi_pin_cycle import ARM_JOINT_NAMES, build_multi_pin_cycle
from pin_axis_3d_sim.watson_hil import (
    HilCoordinator,
    HilMode,
    HilState,
    build_runner_command,
    parse_hil_event_line,
    sanitized_runner_environment,
)
from pin_axis_3d_sim.watson_multi_pin_execution import (
    EXECUTION_ARM_TOKEN,
    GRIPPER_EXECUTION_TOKEN,
    load_execution_bundle,
)


DEFAULT_CONFIG = ARENA_DIR / "config/isaac_multi_pin_verticalization.yaml"
DEFAULT_WRAPPER = SCRIPT_DIR / "run_watson_multi_pin_air_replay.sh"
DEFAULT_REPORT = (
    ARENA_DIR / "outputs/isaac_sim/6.0.1/watson_hil_report.json"
)
RENDER_DT_SECONDS = 1.0 / 60.0
CHILD_STOP_TERM_AFTER_SECONDS = 105.0
CHILD_STOP_KILL_AFTER_SECONDS = 120.0
MAX_RUNNER_OUTPUT_LINES = 250


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in HilMode),
        default=HilMode.PREVIEW.value,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--wrapper", type=Path, default=DEFAULT_WRAPPER)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--auto-arm", action="store_true")
    parser.add_argument("--auto-play", action="store_true")
    parser.add_argument(
        "--no-realtime-preview",
        action="store_true",
        help="Advance preview once per render instead of pacing its control samples.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Close after this wall time; zero leaves the visible result open.",
    )
    parser.add_argument(
        "--stale-after-seconds",
        type=float,
        default=0.25,
        help="Freeze the physical mirror after this joint-state age.",
    )
    parser.add_argument("--camera-view", choices=("tray", "workcell"), default="tray")
    parser.add_argument("--arm-token", default="")
    parser.add_argument("--gripper-token", default="")
    parser.add_argument("--confirm-cell-clear", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> HilMode:
    mode = HilMode(args.mode)
    if not math.isfinite(args.duration_seconds) or args.duration_seconds < 0.0:
        raise ValueError("--duration-seconds must be finite and non-negative")
    if (
        not math.isfinite(args.stale_after_seconds)
        or args.stale_after_seconds <= 0.0
    ):
        raise ValueError("--stale-after-seconds must be finite and positive")
    if args.auto_play and not args.auto_arm:
        raise ValueError("--auto-play requires --auto-arm")
    if args.headless and not (args.auto_arm and args.auto_play):
        raise ValueError("headless HIL requires --auto-arm and --auto-play")
    if mode is HilMode.EXECUTE:
        if args.headless or args.auto_arm or args.auto_play:
            raise ValueError(
                "physical execute requires a visible, manually armed HIL window"
            )
        if (
            args.arm_token != EXECUTION_ARM_TOKEN
            or args.gripper_token != GRIPPER_EXECUTION_TOKEN
            or not args.confirm_cell_clear
        ):
            raise ValueError(
                "physical execute requires the exact arm/gripper tokens and "
                "--confirm-cell-clear"
            )
    elif args.arm_token or args.gripper_token or args.confirm_cell_clear:
        raise ValueError("preview/dry-run must not receive execute authorization")
    return mode


def _normal_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.parent.resolve() / expanded.name


def reserve_private_report(path: Path) -> tuple[Path, int, int]:
    destination = _normal_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        destination,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    sentinel = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "hil_report_reserved",
    }
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(sentinel, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    metadata = destination.lstat()
    return destination, metadata.st_dev, metadata.st_ino


def write_private_report(
    reservation: tuple[Path, int, int],
    payload: dict[str, Any],
) -> None:
    destination, device, inode = reservation
    before = destination.lstat()
    if (
        destination.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or (before.st_dev, before.st_ino) != (device, inode)
    ):
        raise RuntimeError("HIL report reservation changed")
    temporary = destination.with_name(
        f".{destination.name}.final-{os.getpid()}-{secrets.token_hex(6)}"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        current = destination.lstat()
        if (current.st_dev, current.st_ino) != (device, inode):
            raise RuntimeError("HIL report reservation was replaced")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def runner_report_path(hil_report: Path) -> Path:
    suffix = hil_report.suffix or ".json"
    stem = hil_report.name[: -len(suffix)] if hil_report.suffix else hil_report.name
    return hil_report.with_name(f"{stem}_watson_runner{suffix}")


def read_runner_report_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_size <= 0
        or metadata.st_size > 2 * 1024 * 1024
    ):
        raise RuntimeError(f"Watson runner report is not a private regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Watson runner report must contain one JSON object")
    return {
        "status": value.get("status"),
        "mode": value.get("mode"),
        "motion_commanded": value.get("motion_commanded"),
        "gripper_command_count": len(value.get("gripper_commands", [])),
        "stage_report_count": len(value.get("stage_reports", [])),
        "physical_estop_required": (
            value.get("status") == "stop_unverified_use_physical_estop"
        ),
    }


def make_event(sequence: int, name: str, **fields: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event_sequence": sequence,
        "event": name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }


def stream_child_output(
    process: subprocess.Popen[str],
    output_queue: queue.SimpleQueue[str],
) -> None:
    assert process.stdout is not None
    try:
        for line in process.stdout:
            output_queue.put(line)
    finally:
        process.stdout.close()


def create_panel(
    ui: Any,
    *,
    mode: HilMode,
    set_camera_view: Any,
) -> tuple[Any, Any, Any, dict[str, bool]]:
    flags = {
        "arm_requested": False,
        "disarm_requested": False,
        "stop_requested": False,
        "close_requested": False,
    }
    physical = mode is HilMode.EXECUTE
    title = (
        "Watson PHYSICAL HIL"
        if physical
        else f"Watson HIL - {mode.value.upper()}"
    )
    window = ui.Window(title, width=620, height=430)
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label(
                (
                    "PHYSICAL WATSON MOTION ENABLED"
                    if physical
                    else (
                        "READ-ONLY LIVE CHECK - NO MOTION"
                        if mode is HilMode.DRY_RUN
                        else "OFFLINE ISAAC PREVIEW - WATSON NOT CONNECTED"
                    )
                ),
                style={
                    "color": 0xFF3333FF if physical else 0xFF55DD55,
                    "font_size": 19,
                },
            )
            status_label = ui.Label(
                "DISARMED - click ARM, then the Isaac toolbar Play button",
                style={"color": 0xFF44AAFF, "font_size": 17},
            )
            details_label = ui.Label("One run per window. Pause or Stop cancels.")
            ui.Label(
                (
                    "The guarded Watson wrapper remains the only physical command "
                    "authority. Isaac mirrors measured arm joints; ordered runner "
                    "events animate the 2FG7 and specimens."
                    if mode is not HilMode.PREVIEW
                    else
                    "Preview uses the reviewed seven-pin choreography locally. "
                    "It creates no ROS graph, network connection, or robot process."
                ),
                word_wrap=True,
            )
            with ui.HStack(spacing=8, height=36):
                ui.Button(
                    (
                        "ARM ONE PHYSICAL REPLAY"
                        if physical
                        else f"ARM ONE {mode.value.upper()} RUN"
                    ),
                    clicked_fn=lambda: flags.__setitem__("arm_requested", True),
                )
                ui.Button(
                    "Disarm",
                    clicked_fn=lambda: flags.__setitem__("disarm_requested", True),
                )
            with ui.HStack(spacing=8, height=34):
                ui.Button("Focus tray", clicked_fn=lambda: set_camera_view("tray"))
                ui.Button(
                    "Whole workcell",
                    clicked_fn=lambda: set_camera_view("workcell"),
                )
            ui.Button(
                "STOP HIL / GUARDED CANCEL",
                height=42,
                clicked_fn=lambda: flags.__setitem__("stop_requested", True),
            )

            def close_window() -> None:
                flags["stop_requested"] = True
                flags["close_requested"] = True

            ui.Button("Close HIL window", height=32, clicked_fn=close_window)
    return window, status_label, details_label, flags


def status_style(state: HilState, *, estop: bool = False) -> dict[str, Any]:
    if estop or state is HilState.FAILED:
        colour = 0xFF3333FF
    elif state in {HilState.COMPLETED, HilState.STOPPED}:
        colour = 0xFF55DD55
    elif state in {HilState.ARMED, HilState.RUNNING}:
        colour = 0xFF44AAFF
    else:
        colour = 0xFFAAAAAA
    return {"color": colour, "font_size": 17}


def main() -> int:
    args = build_parser().parse_args()
    mode = validate_args(args)
    package_versions = single.validate_runtime()
    if mode is not HilMode.PREVIEW:
        live.validate_ros_environment()

    config_path = _normal_path(args.config)
    wrapper_path = _normal_path(args.wrapper)
    config = multi.load_and_validate_config(config_path)
    if mode is not HilMode.PREVIEW and not wrapper_path.is_file():
        raise FileNotFoundError(f"Watson wrapper is missing: {wrapper_path}")

    requested_hil_report = _normal_path(args.report)
    child_report_path = runner_report_path(requested_hil_report)
    if child_report_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite Watson runner report: {child_report_path}"
        )
    hil_reservation = reserve_private_report(requested_hil_report)
    hil_report_path = hil_reservation[0]
    child_report_path = runner_report_path(hil_report_path)

    gripper = config["gripper"]
    preview_commands = build_multi_pin_cycle(
        config["plan"],
        finger_open_m=float(gripper["open_position_m"]),
        finger_closed_m=float(gripper["closed_position_m"]),
        finger_speed_m_s=float(gripper["per_finger_speed_m_s"]),
        hold_seconds=float(gripper["hold_seconds"]),
    )
    physical_bundle = (
        None if mode is HilMode.PREVIEW else load_execution_bundle()
    )
    coordinator = HilCoordinator(mode)
    started_utc = datetime.now(timezone.utc).isoformat()
    started_wall = time.monotonic()
    report: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": started_utc,
        "runtime_scope": "isaac_watson_one_shot_hil",
        "mode": mode.value,
        "status": "initialising",
        "command": [sys.executable, *sys.argv],
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "package_versions": package_versions,
        "source_config": str(config_path),
        "source_plan": str(config["plan_path"]),
        "source_plan_sha256": config["plan_sha256"],
        "source_usd": str(config["usd_path"]),
        "watson_wrapper": str(wrapper_path) if mode is not HilMode.PREVIEW else None,
        "watson_runner_report": (
            str(child_report_path) if mode is not HilMode.PREVIEW else None
        ),
        "ros_used": mode is not HilMode.PREVIEW,
        "watson_connection_requested": mode is not HilMode.PREVIEW,
        "watson_connected": False if mode is HilMode.PREVIEW else None,
        "real_robot_command_authorized": mode is HilMode.EXECUTE,
        "real_robot_commanded": False,
        "arm_visual_authority": (
            "offline_reviewed_commands"
            if mode is HilMode.PREVIEW
            else "/watson/joint_states"
        ),
        "gripper_visual_authority": (
            "offline_reviewed_commands"
            if mode is HilMode.PREVIEW
            else "ordered_runner_events"
        ),
        "events": coordinator.events,
    }

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
            "open_usd": str(config["usd_path"]),
        }
    )

    exit_code = 1
    world = None
    panel_window = None
    timeline_subscriptions: tuple[Any, Any, Any] | None = None
    ros_context = None
    ros_executor = None
    ros_node = None
    ros_subscription = None
    child: subprocess.Popen[str] | None = None
    child_reader: threading.Thread | None = None
    child_output: queue.SimpleQueue[str] = queue.SimpleQueue()
    child_command: list[str] | None = None
    child_return_code: int | None = None
    child_stop_started: float | None = None
    child_term_sent = False
    child_kill_sent = False
    runner_lines: list[str] = []
    protocol_error: str | None = None
    graph_snapshot: dict[str, Any] | None = None
    joint_buffer: LiveJointStateBuffer | None = None
    applied_joint_updates = 0
    stale_frames = 0
    max_joint_age_seconds = 0.0
    max_tracking_error_radians = 0.0
    panel_flags: dict[str, bool] = {
        "arm_requested": False,
        "disarm_requested": False,
        "stop_requested": False,
        "close_requested": False,
    }
    timeline_flags = {
        "play_requested": False,
        "stop_requested": False,
        "suppress_next_stop": False,
    }

    try:
        import carb
        import omni.timeline
        import omni.ui as ui
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.rendering_manager import ViewportManager
        from isaacsim.core.utils.extensions import enable_extension
        from omni.physx import get_physx_interface

        if mode is not HilMode.PREVIEW:
            enable_extension("isaacsim.ros2.bridge")
            for _ in range(4):
                simulation_app.update()

        World.clear_instance()
        carb.settings.get_settings().set_bool(
            single.GUIDE_PURPOSE_DISPLAY_SETTING,
            False,
        )
        stage = omni.usd.get_context().get_stage()
        default_prim = stage.GetDefaultPrim()
        if not default_prim.IsValid():
            raise RuntimeError("Articulated Watson stage has no default prim")
        asset_prim_path = str(default_prim.GetPath())
        scene_paths, suppressed_scene_proxies = multi.add_static_scene(
            stage,
            config["plan"],
        )

        payload_ops: dict[int, Any] = {}
        initial_matrices: dict[int, Any] = {}
        destination_matrices: dict[int, Any] = {}
        for specimen, color in zip(
            config["plan"]["specimens"],
            config["payload"]["specimen_colors_rgb"],
        ):
            specimen_id = int(specimen["specimen_id"])
            payload_ops[specimen_id], _ = multi.add_payload_visual(
                stage,
                config["payload"],
                specimen,
                color,
            )
            initial_matrices[specimen_id] = multi.pose_matrix(
                multi._stage(
                    specimen,
                    "descend_tilted_grasp",
                )["target_pin_grasp_tcp_pose"]
            )
            destination_matrices[specimen_id] = multi.pose_matrix(
                multi._stage(
                    specimen,
                    "descend_vertical",
                )["target_pin_grasp_tcp_pose"]
            )
            payload_ops[specimen_id].Set(initial_matrices[specimen_id])

        world = World(
            physics_dt=float(config["plan"]["control_dt_seconds"]),
            rendering_dt=RENDER_DT_SECONDS,
            stage_units_in_meters=1.0,
            backend="numpy",
            device="cpu",
        )
        robot = world.scene.add(
            SingleArticulation(
                prim_path=asset_prim_path,
                name="watson_isaac_hil",
            )
        )
        world.reset()
        world.pause()
        if not robot.handles_initialized:
            raise RuntimeError("Watson HIL articulation handles did not initialise")
        if robot.num_dof != 8 or list(robot.dof_names) != config["expected_dof_names"]:
            raise RuntimeError(
                f"Expected eight DOFs {config['expected_dof_names']}; "
                f"found {list(robot.dof_names)}"
            )
        dof_index = {name: index for index, name in enumerate(robot.dof_names)}
        joint_limits = np.column_stack(
            (
                np.asarray(robot.dof_properties["lower"], dtype=np.float64),
                np.asarray(robot.dof_properties["upper"], dtype=np.float64),
            )
        )
        arm_limits = np.asarray(
            [joint_limits[dof_index[name]] for name in ARM_JOINT_NAMES],
            dtype=np.float64,
        )
        display_positions = np.zeros(8, dtype=np.float64)
        display_velocities = np.zeros(8, dtype=np.float64)
        if mode is HilMode.PREVIEW:
            initial_arm = np.asarray(
                preview_commands[0]["arm_positions"],
                dtype=np.float64,
            )
            preview_arm_positions = np.asarray(
                [command["arm_positions"] for command in preview_commands],
                dtype=np.float64,
            )
            if np.any(preview_arm_positions < arm_limits[:, 0]) or np.any(
                preview_arm_positions > arm_limits[:, 1]
            ):
                raise RuntimeError(
                    "Offline preview exceeds the imported arm-joint limits"
                )
        else:
            assert physical_bundle is not None
            initial_arm = np.asarray(
                physical_bundle.stages[0].start_positions,
                dtype=np.float64,
            )
        for index, name in enumerate(ARM_JOINT_NAMES):
            display_positions[dof_index[name]] = initial_arm[index]
        finger_position = float(gripper["open_position_m"])
        finger_target = finger_position
        for name in (gripper["leader_joint"], gripper["mimic_joint"]):
            display_positions[dof_index[name]] = finger_position
        if np.any(display_positions < joint_limits[:, 0]) or np.any(
            display_positions > joint_limits[:, 1]
        ):
            raise RuntimeError("Initial HIL articulation state exceeds imported limits")

        physx_interface = get_physx_interface()

        def apply_display() -> bool:
            if getattr(world, "physics_sim_view", None) is None:
                return False
            robot.set_joint_positions(display_positions)
            robot.set_joint_velocities(display_velocities)
            live.sync_rendered_articulation_pose(world, physx_interface)
            return True

        robot.set_joints_default_state(
            positions=display_positions,
            velocities=display_velocities,
        )
        apply_display()
        tcp_prim_path = config["import_report"]["expected_link_paths"][
            "pin_grasp_tcp"
        ]

        camera_evidence: dict[str, Any] = {}

        def set_camera_view(view: str) -> None:
            if view == "tray":
                eye = config["viewer"]["tray_camera_eye_xyz_m"]
                target = config["viewer"]["tray_camera_target_xyz_m"]
            elif view == "workcell":
                eye = config["viewer"]["workcell_camera_eye_xyz_m"]
                target = config["viewer"]["workcell_camera_target_xyz_m"]
            else:
                raise ValueError(f"unknown HIL camera view: {view}")
            camera_evidence.update(
                {"view": view, "eye_xyz_m": list(eye), "target_xyz_m": list(target)}
            )
            ViewportManager.set_camera_view(
                ViewportManager.get_camera(),
                eye=list(eye),
                target=list(target),
            )

        if not args.headless:
            ready, waited_frames = ViewportManager.wait_for_viewport(max_frames=120)
            if not ready:
                raise RuntimeError(
                    f"Isaac viewport was not ready after {waited_frames} frames"
                )
            set_camera_view(args.camera_view)
            (
                panel_window,
                status_label,
                details_label,
                panel_flags,
            ) = create_panel(ui, mode=mode, set_camera_view=set_camera_view)
        else:
            status_label = None
            details_label = None

        if mode is not HilMode.PREVIEW:
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

            joint_buffer = LiveJointStateBuffer(
                stale_after_seconds=args.stale_after_seconds
            )
            ros_context = Context()
            rclpy.init(
                context=ros_context,
                signal_handler_options=SignalHandlerOptions.NO,
            )
            ros_node = Node(
                "watson_hil_joint_mirror",
                namespace="/isaac",
                context=ros_context,
                enable_rosout=False,
                start_parameter_services=False,
                enable_logger_service=False,
            )
            ros_executor = SingleThreadedExecutor(context=ros_context)
            ros_executor.add_node(ros_node)

            def joint_state_callback(message: Any) -> None:
                assert joint_buffer is not None
                try:
                    joint_buffer.accept(
                        message.name,
                        message.position,
                        message.velocity,
                        stamp_seconds=message.header.stamp.sec,
                        stamp_nanoseconds=message.header.stamp.nanosec,
                        joint_limits=arm_limits,
                    )
                except ValueError:
                    return

            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            ros_subscription = ros_node.create_subscription(
                JointState,
                JOINT_STATE_TOPIC,
                joint_state_callback,
                qos,
            )
            graph_snapshot, graph_failures = live.own_ros_graph_snapshot(ros_node)
            if graph_failures:
                raise RuntimeError(
                    "Isaac HIL mirror created a command path: "
                    + "; ".join(graph_failures)
                )

        timeline = omni.timeline.get_timeline_interface()

        def on_timeline_play(_event: Any) -> None:
            timeline_flags["play_requested"] = True

        def on_timeline_stop(_event: Any) -> None:
            if timeline_flags["suppress_next_stop"]:
                timeline_flags["suppress_next_stop"] = False
            else:
                timeline_flags["stop_requested"] = True

        timeline_subscriptions = subscribe_hil_timeline(
            timeline,
            on_timeline_play,
            on_timeline_stop,
        )

        if args.auto_arm:
            panel_flags["arm_requested"] = True
        auto_play_issued = False
        preview_index = 0
        preview_accumulator = 0.0
        preview_last_wall = time.monotonic()
        attached_specimen: int | None = None
        current_stage = "waiting"
        last_runner_line = ""
        close_after_terminal = args.headless
        report["scene_paths"] = scene_paths
        report["suppressed_scene_proxies"] = suppressed_scene_proxies
        report["ros_graph"] = graph_snapshot
        report["status"] = "disarmed"

        def accept_event(event: dict[str, Any]) -> None:
            nonlocal attached_specimen, finger_target, current_stage
            accepted = coordinator.accept_event(event)
            event_name = accepted["event"]
            if event_name in {"stage_started", "stage_completed", "stage_failed"}:
                current_stage = (
                    f"{accepted.get('sequence_index')}: "
                    f"{accepted.get('stage_name')}"
                )
            if event_name == "gripper_started":
                finger_target = float(
                    gripper[
                        "closed_position_m"
                        if accepted["action"] == "close"
                        else "open_position_m"
                    ]
                )
            if event_name == "gripper_completed" and accepted["completed"]:
                finger_target = float(
                    gripper[
                        "closed_position_m"
                        if accepted["action"] == "close"
                        else "open_position_m"
                    ]
                )
                specimen_id = accepted.get("specimen_id")
                if accepted["action"] == "close" and specimen_id is not None:
                    if attached_specimen is not None:
                        raise RuntimeError(
                            "HIL received a second close while a specimen was attached"
                        )
                    attached_specimen = int(specimen_id)
                    payload_ops[attached_specimen].Set(
                        single.prim_world_matrix(stage, tcp_prim_path)
                    )
                elif accepted["action"] == "open" and specimen_id is not None:
                    if attached_specimen != int(specimen_id):
                        raise RuntimeError(
                            "HIL open event does not match the attached specimen"
                        )
                    payload_ops[int(specimen_id)].Set(
                        destination_matrices[int(specimen_id)]
                    )
                    attached_specimen = None

        while simulation_app.is_running():
            frame_started = time.monotonic()
            if ros_executor is not None:
                ros_executor.spin_once(timeout_sec=0.0)

            if panel_flags["arm_requested"]:
                panel_flags["arm_requested"] = False
                try:
                    if timeline.is_playing():
                        raise RuntimeError(
                            "stop the Isaac timeline before arming this HIL run"
                        )
                    coordinator.arm()
                    report["status"] = "armed_waiting_for_toolbar_play"
                except RuntimeError as exc:
                    last_runner_line = str(exc)
            if panel_flags["disarm_requested"]:
                panel_flags["disarm_requested"] = False
                try:
                    coordinator.disarm()
                    report["status"] = "disarmed"
                except RuntimeError as exc:
                    last_runner_line = str(exc)

            if args.auto_play and coordinator.state is HilState.ARMED and not auto_play_issued:
                auto_play_issued = True
                timeline.play()

            if timeline_flags["play_requested"]:
                timeline_flags["play_requested"] = False
                action = coordinator.on_play()
                if action == "launch":
                    timeline_flags["suppress_next_stop"] = True
                    timeline.pause()
                    report["status"] = "launch_requested"

            stop_requested = (
                panel_flags["stop_requested"] or timeline_flags["stop_requested"]
            )
            panel_flags["stop_requested"] = False
            timeline_flags["stop_requested"] = False

            if coordinator.state is HilState.LAUNCH_REQUESTED:
                if stop_requested:
                    coordinator.cancel_before_spawn()
                    report["status"] = "cancelled_before_launch"
                elif mode is HilMode.PREVIEW:
                    coordinator.runner_started()
                    accept_event(
                        make_event(
                            1,
                            "run_started",
                            mode=mode.value,
                            start_mode="offline_preview",
                            selected_stage_count=49,
                        )
                    )
                    report["status"] = "preview_running"
                    preview_last_wall = time.monotonic()
                else:
                    child_command = build_runner_command(
                        wrapper=wrapper_path,
                        mode=mode,
                        report=child_report_path,
                        arm_token=args.arm_token,
                        gripper_token=args.gripper_token,
                        confirm_cell_clear=args.confirm_cell_clear,
                    )
                    child = subprocess.Popen(
                        child_command,
                        cwd=ARENA_DIR,
                        env=sanitized_runner_environment(),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                        start_new_session=True,
                    )
                    coordinator.runner_started()
                    child_reader = threading.Thread(
                        target=stream_child_output,
                        args=(child, child_output),
                        name="watson-hil-output",
                        daemon=True,
                    )
                    child_reader.start()
                    report["status"] = "watson_wrapper_running"
                    report["watson_wrapper_pid"] = child.pid

            if stop_requested and coordinator.state in {
                HilState.RUNNING,
                HilState.LAUNCH_REQUESTED,
            }:
                stop_action = coordinator.on_stop()
                if mode is HilMode.PREVIEW and stop_action == "signal_stop":
                    accept_event(
                        make_event(
                            coordinator.last_event_sequence + 1,
                            "run_failed",
                            status="preview_stopped",
                            error="preview stop requested",
                            physical_estop_required=False,
                        )
                    )
                    coordinator.runner_exited(130)
                    report["status"] = "preview_stopped"
                elif child is not None and stop_action == "signal_stop":
                    child.send_signal(signal.SIGINT)
                    child_stop_started = time.monotonic()
                    report["status"] = "guarded_stop_requested"

            while True:
                try:
                    line = child_output.get_nowait()
                except queue.Empty:
                    break
                stripped = line.rstrip("\r\n")
                print(f"[Watson] {stripped}", flush=True)
                runner_lines.append(stripped)
                if len(runner_lines) > MAX_RUNNER_OUTPUT_LINES:
                    del runner_lines[: len(runner_lines) - MAX_RUNNER_OUTPUT_LINES]
                last_runner_line = stripped[-180:]
                try:
                    event = parse_hil_event_line(line)
                    if event is not None:
                        accept_event(event)
                except (RuntimeError, ValueError) as exc:
                    protocol_error = str(exc)
                    last_runner_line = f"HIL PROTOCOL ERROR: {exc}"
                    if coordinator.state in {
                        HilState.RUNNING,
                        HilState.LAUNCH_REQUESTED,
                    }:
                        stop_action = coordinator.on_stop()
                        if child is not None and stop_action == "signal_stop":
                            child.send_signal(signal.SIGINT)
                            child_stop_started = time.monotonic()

            if child is not None and child.poll() is not None and child_return_code is None:
                child_return_code = int(child.returncode)
                if child_reader is not None:
                    child_reader.join(timeout=2.0)
                while True:
                    try:
                        line = child_output.get_nowait()
                    except queue.Empty:
                        break
                    stripped = line.rstrip("\r\n")
                    runner_lines.append(stripped)
                    last_runner_line = stripped[-180:]
                    try:
                        event = parse_hil_event_line(line)
                        if event is not None:
                            accept_event(event)
                    except (RuntimeError, ValueError) as exc:
                        protocol_error = str(exc)
                        last_runner_line = f"HIL PROTOCOL ERROR: {exc}"
                if coordinator.state in {
                    HilState.RUNNING,
                    HilState.STOPPING,
                    HilState.LAUNCH_REQUESTED,
                }:
                    coordinator.runner_exited(child_return_code)
                report["status"] = coordinator.state.value

            if child_stop_started is not None and child is not None and child.poll() is None:
                stop_age = time.monotonic() - child_stop_started
                if stop_age >= CHILD_STOP_KILL_AFTER_SECONDS and not child_kill_sent:
                    child.kill()
                    child_kill_sent = True
                    protocol_error = (
                        "Watson wrapper exceeded guarded stop and TERM intervals"
                    )
                elif stop_age >= CHILD_STOP_TERM_AFTER_SECONDS and not child_term_sent:
                    child.terminate()
                    child_term_sent = True

            now = time.monotonic()
            frame_dt = max(0.0, now - frame_started)
            if mode is HilMode.PREVIEW and coordinator.state is HilState.RUNNING:
                if args.no_realtime_preview:
                    steps = 4096
                else:
                    preview_accumulator += now - preview_last_wall
                    steps = int(
                        preview_accumulator
                        / float(config["plan"]["control_dt_seconds"])
                    )
                    if steps > 0:
                        preview_accumulator -= (
                            steps * float(config["plan"]["control_dt_seconds"])
                        )
                preview_last_wall = now
                preview_batch_size = (
                    min(steps, 4096)
                    if args.no_realtime_preview
                    else min(steps, 64)
                )
                for _ in range(preview_batch_size):
                    command = preview_commands[preview_index]
                    for index, name in enumerate(ARM_JOINT_NAMES):
                        display_positions[dof_index[name]] = command["arm_positions"][
                            index
                        ]
                        display_velocities[dof_index[name]] = command["arm_velocities"][
                            index
                        ]
                    finger_position = float(command["finger_position_m"])
                    finger_target = finger_position
                    for name in (gripper["leader_joint"], gripper["mimic_joint"]):
                        display_positions[dof_index[name]] = finger_position
                    current_stage = (
                        f"specimen {command['specimen_id']}/7: {command['phase']}"
                    )
                    if not args.no_realtime_preview:
                        apply_display()
                        tcp_matrix = single.prim_world_matrix(stage, tcp_prim_path)
                        if attached_specimen is not None:
                            payload_ops[attached_specimen].Set(tcp_matrix)
                    if command["attachment_event"] == "attach":
                        attached_specimen = int(command["specimen_id"])
                        if not args.no_realtime_preview:
                            payload_ops[attached_specimen].Set(tcp_matrix)
                    elif command["attachment_event"] == "release":
                        specimen_id = int(command["specimen_id"])
                        payload_ops[specimen_id].Set(
                            destination_matrices[specimen_id]
                        )
                        attached_specimen = None
                    preview_index += 1
                    if preview_index >= len(preview_commands):
                        accept_event(
                            make_event(
                                coordinator.last_event_sequence + 1,
                                "run_completed",
                                mode=mode.value,
                                status="preview_completed",
                                motion_commanded=False,
                            )
                        )
                        coordinator.runner_exited(0)
                        report["status"] = "preview_completed"
                        break
                if args.no_realtime_preview:
                    apply_display()
                    if attached_specimen is not None:
                        payload_ops[attached_specimen].Set(
                            single.prim_world_matrix(stage, tcp_prim_path)
                        )
            elif mode is not HilMode.PREVIEW:
                assert joint_buffer is not None
                joint_status = joint_buffer.status(now_monotonic_s=now)
                age = joint_buffer.age_seconds(now_monotonic_s=now)
                if age is not None:
                    max_joint_age_seconds = max(max_joint_age_seconds, age)
                if joint_status == "LIVE" and joint_buffer.sample is not None:
                    for index, name in enumerate(ARM_JOINT_NAMES):
                        display_positions[dof_index[name]] = (
                            joint_buffer.sample.positions[index]
                        )
                        display_velocities[dof_index[name]] = (
                            joint_buffer.sample.velocities[index]
                        )
                    applied_joint_updates += 1
                elif joint_status == "STALE":
                    stale_frames += 1
                finger_step = (
                    float(gripper["per_finger_speed_m_s"])
                    * max(frame_dt, RENDER_DT_SECONDS)
                )
                delta = finger_target - finger_position
                finger_position += float(
                    np.clip(delta, -finger_step, finger_step)
                )
                for name in (gripper["leader_joint"], gripper["mimic_joint"]):
                    display_positions[dof_index[name]] = finger_position
                    display_velocities[dof_index[name]] = 0.0
                display_applied = apply_display()
                if (
                    display_applied
                    and joint_status == "LIVE"
                    and joint_buffer.sample is not None
                ):
                    readback = np.asarray(robot.get_joint_positions())
                    error = max(
                        abs(
                            float(readback[dof_index[name]])
                            - joint_buffer.sample.positions[index]
                        )
                        for index, name in enumerate(ARM_JOINT_NAMES)
                    )
                    max_tracking_error_radians = max(
                        max_tracking_error_radians,
                        error,
                    )
                if attached_specimen is not None:
                    payload_ops[attached_specimen].Set(
                        single.prim_world_matrix(stage, tcp_prim_path)
                    )

            if status_label is not None:
                if coordinator.physical_estop_required:
                    status_label.text = "USE PHYSICAL E-STOP - STOP NOT VERIFIED"
                elif coordinator.state is HilState.ARMED:
                    status_label.text = "ARMED ONCE - PRESS ISAAC TOOLBAR PLAY"
                elif coordinator.state is HilState.RUNNING:
                    mirror = ""
                    if joint_buffer is not None:
                        mirror = f" | mirror {joint_buffer.status(now_monotonic_s=now)}"
                    status_label.text = f"RUNNING | {current_stage}{mirror}"
                elif coordinator.state is HilState.STOPPING:
                    status_label.text = "STOPPING - GUARDED RECOVERY IN PROGRESS"
                elif coordinator.state is HilState.COMPLETED:
                    status_label.text = "COMPLETED - ONE-SHOT RUN FINISHED"
                elif coordinator.state is HilState.STOPPED:
                    status_label.text = "STOPPED"
                elif coordinator.state is HilState.FAILED:
                    status_label.text = "FAILED CLOSED"
                else:
                    status_label.text = (
                        "DISARMED - click ARM, then Isaac toolbar Play"
                    )
                status_label.style = status_style(
                    coordinator.state,
                    estop=coordinator.physical_estop_required,
                )
            if details_label is not None:
                details_label.text = (
                    last_runner_line
                    or "One-shot state: " + coordinator.state.value
                )

            if getattr(world, "physics_sim_view", None) is None:
                simulation_app.update()
            else:
                world.render()

            terminal = coordinator.state in {
                HilState.COMPLETED,
                HilState.STOPPED,
                HilState.FAILED,
            }
            if close_after_terminal and terminal:
                break
            if panel_flags["close_requested"] and (
                child is None or child.poll() is not None
            ):
                break
            if args.duration_seconds > 0.0 and now - started_wall >= args.duration_seconds:
                if coordinator.state in {
                    HilState.RUNNING,
                    HilState.LAUNCH_REQUESTED,
                }:
                    panel_flags["stop_requested"] = True
                elif child is None or child.poll() is not None:
                    break
            remaining = RENDER_DT_SECONDS - (time.monotonic() - frame_started)
            if remaining > 0.0 and not (
                args.no_realtime_preview and mode is HilMode.PREVIEW
            ):
                time.sleep(remaining)

        if coordinator.state is HilState.COMPLETED:
            exit_code = 0
        elif coordinator.state is HilState.STOPPED:
            exit_code = 130
        else:
            exit_code = 1
    except BaseException as exc:
        report["status"] = "hil_failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        if child is not None and child.poll() is None:
            try:
                child.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
            stop_started = time.monotonic()
            while child.poll() is None:
                elapsed = time.monotonic() - stop_started
                if elapsed >= CHILD_STOP_KILL_AFTER_SECONDS:
                    child.kill()
                elif elapsed >= CHILD_STOP_TERM_AFTER_SECONDS:
                    child.terminate()
                if simulation_app.is_running():
                    simulation_app.update()
                time.sleep(0.05)
            child_return_code = int(child.returncode)
        if child_reader is not None:
            child_reader.join(timeout=2.0)
        if ros_executor is not None and ros_node is not None:
            try:
                ros_executor.remove_node(ros_node)
            except Exception:
                pass
        if ros_node is not None:
            ros_node.destroy_node()
        if ros_context is not None:
            try:
                import rclpy

                if rclpy.ok(context=ros_context):
                    rclpy.shutdown(context=ros_context)
            except Exception:
                pass
        runner_report_summary = None
        if mode is not HilMode.PREVIEW:
            try:
                runner_report_summary = read_runner_report_summary(
                    child_report_path
                )
            except Exception as exc:
                if protocol_error is None:
                    protocol_error = f"runner report validation failed: {exc}"
        if runner_report_summary is not None:
            report["watson_connected"] = runner_report_summary["status"] not in {
                "reserved_before_live_contact",
                None,
            }
            report["real_robot_commanded"] = bool(
                runner_report_summary["motion_commanded"]
            )

        report.update(
            {
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_wall_seconds": time.monotonic() - started_wall,
                "status": report.get("status", coordinator.state.value),
                "coordinator_state": coordinator.state.value,
                "launch_consumed": coordinator.launch_consumed,
                "physical_estop_required": coordinator.physical_estop_required,
                "failure": coordinator.failure,
                "protocol_error": protocol_error,
                "events": coordinator.events,
                "child_command": child_command,
                "child_return_code": child_return_code,
                "child_term_sent": child_term_sent,
                "child_kill_sent": child_kill_sent,
                "runner_output_tail": runner_lines,
                "watson_runner_summary": runner_report_summary,
                "valid_joint_messages": (
                    joint_buffer.valid_messages if joint_buffer is not None else 0
                ),
                "invalid_joint_messages": (
                    joint_buffer.invalid_messages if joint_buffer is not None else 0
                ),
                "applied_joint_updates": applied_joint_updates,
                "stale_frames": stale_frames,
                "maximum_joint_age_seconds": max_joint_age_seconds,
                "maximum_joint_tracking_error_radians": (
                    max_tracking_error_radians
                ),
                "camera": (
                    camera_evidence if "camera_evidence" in locals() else {}
                ),
                "physics_timeline_used_as_one_shot_trigger": True,
                "physical_pause_supported": False,
                "pause_and_stop_request_guarded_cancel": True,
            }
        )
        if protocol_error is not None and exit_code == 0:
            exit_code = 1
            report["status"] = "hil_protocol_failed"
        try:
            write_private_report(hil_reservation, report)
            print(f"Isaac/Watson HIL report: {hil_report_path}", flush=True)
        except Exception:
            traceback.print_exc()
            exit_code = 1
        timeline_subscriptions = None
        panel_window = None
        if world is not None and simulation_app.is_running():
            try:
                world.stop()
            except Exception:
                pass
        try:
            simulation_app.close(
                wait_for_replicator=False,
                exit_code=exit_code,
            )
        except Exception:
            pass

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
