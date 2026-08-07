#!/usr/bin/env python3
"""Guarded ingress and synchronized seven-pin arm/2FG7 air replay for Watson.

``check`` and ``dry-run`` never send an action goal or construct a gripper
transport.  ``execute`` is locked to the exact private artifacts, named tool,
Listen1, graph owners, six-axis envelopes, immutable arm and gripper tokens,
and explicit cell-clear confirmation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Iterator, NamedTuple


SCRIPT_DIR = Path(__file__).resolve().parent
ARENA_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ARENA_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from pin_axis_3d_sim.controller_tool_state import (  # noqa: E402
    query_controller_tool_items,
)
from pin_axis_3d_sim.onrobot_control import (  # noqa: E402
    CONTROL_CONFIRMATION,
    FixedComputeBoxTransport,
    GripperAction,
    run_fixed_recovery_stop,
    run_guarded_command,
)
from pin_axis_3d_sim.watson_guard import (  # noqa: E402
    HealthSnapshot,
    JOINT_NAMES,
    health_failures,
)
from pin_axis_3d_sim.watson_multi_pin_execution import (  # noqa: E402
    DEFAULT_INGRESS_ARTIFACT,
    DEFAULT_RETIMED_ARTIFACT,
    EXECUTION_ARM_TOKEN,
    GRIPPER_EXECUTION_TOKEN,
    GRIPPER_POLICY,
    LIVE_GOAL_TOLERANCE_RAD,
    MAX_PROJECT_SPEED,
    READY_JOINT_POSITIONS_RAD,
    ExecutionBundle,
    StageSpec,
    build_robot_trajectory,
    exact_execute_project_speed_failures,
    exact_tool_audit_failures,
    gripper_after_stage_hook,
    live_stage_failures,
    live_start_errors,
    load_execution_bundle,
    stage_report,
    validate_execution_authorization,
    validate_robot_trajectory,
    validate_stage_live_first_wire_cubic,
)
from pin_axis_3d_sim.watson_hil import HIL_EVENT_PREFIX  # noqa: E402
from run_watson_guarded_demo import (  # noqa: E402
    EXECUTE_LOCK_PATH,
    ROBOT_INTERFACE,
    ROBOT_IP,
    ROBOT_MAC,
    ROBOT_SOURCE_IP,
    StopUnverifiedError,
    acquire_execute_lock,
    validate_execute_network,
)


SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
EXECUTE_ACTION = "/watson/execute_trajectory"
CONTROLLER_ACTION = "/watson/tmr_arm_controller/follow_joint_trajectory"
TOOL_SELECT_REQUEST_ID = "ToolSelect1"
TOOL_SELECT_SCRIPT = 'ChangeTCP("QC_2FG7_VENDOR")'
MOVEIT_SUCCESS = 1
GOAL_ACCEPTANCE_TIMEOUT_S = 10.0
POST_MOTION_STATIONARY_TIMEOUT_S = 8.0
ACTION_IDLE_TIMEOUT_S = 4.0
ACTION_STATUS_TYPE = "action_msgs/msg/GoalStatusArray"
REPORT_DIGEST_FIELD = "report_payload_sha256"
REPORT_SCHEMA_VERSION = 1
MAX_REPORT_BYTES = 2 * 1024 * 1024
COMPUTE_BOX_IP = os.environ.get("ONROBOT_COMPUTE_BOX_IP", "192.0.2.1")
POSITION_SOURCE_CACHE_SIZE = 8
POSITION_SOURCE_MAX_HEADER_SKEW_S = 0.030
POSITION_SOURCE_MAX_PAIR_AGE_S = 0.100
POSITION_SOURCE_MAX_PAIRED_DELTA_RAD = 0.005
_hil_event_sequence = 0
_hil_final_event_emitted = False


class ProcessSignalGate:
    """Block stop signals in every subsequently created thread.

    Live command dispatch runs only on the main thread. Signals remain blocked
    process-wide through thread inheritance and are consumed synchronously at
    explicit guard points. ``command_commit`` is the linearization point:
    a signal consumed before it blocks dispatch; one arriving after it causes
    the already-committed command to enter the normal cancel/STOP recovery path.
    """

    def __init__(self, signal_api: Any = signal) -> None:
        self._signal = signal_api
        self._signals = frozenset(
            (
                signal_api.SIGINT,
                signal_api.SIGTERM,
                signal_api.SIGHUP,
            )
        )
        self._latched_signal: int | None = None
        self._dispatch_active = False
        self._previous_mask = signal_api.pthread_sigmask(
            signal_api.SIG_BLOCK,
            self._signals,
        )
        self._previous_handlers = {}
        for stop_signal in self._signals:
            self._previous_handlers[stop_signal] = signal_api.getsignal(
                stop_signal
            )
            # Non-interactive shells may start a background child with SIGINT
            # ignored. Reset disposition while it is blocked so sigtimedwait
            # reliably receives every wrapper-forwarded stop signal.
            signal_api.signal(stop_signal, signal_api.SIG_DFL)

    @property
    def latched_signal(self) -> int | None:
        self.poll()
        return self._latched_signal

    def poll(self) -> int | None:
        if self._latched_signal is not None:
            return self._latched_signal
        signal_info = self._signal.sigtimedwait(self._signals, 0.0)
        if signal_info is not None:
            self._latched_signal = int(signal_info.si_signo)
        return self._latched_signal

    @contextmanager
    def command_commit(self, label: str) -> Iterator[None]:
        if not isinstance(label, str) or not label:
            raise ValueError("command dispatch label must be non-empty")
        if self._dispatch_active:
            raise RuntimeError("nested live command dispatch is forbidden")
        pending = self.poll()
        if pending is not None:
            raise InterruptedError(
                f"stop requested by signal {pending} before {label} dispatch"
            )
        self._dispatch_active = True
        try:
            yield
        finally:
            self._dispatch_active = False


class PositionSourcePairTracker:
    """Pair FeedbackState and JointState positions by their driver timestamp."""

    def __init__(self) -> None:
        self._samples: dict[
            str,
            dict[int, tuple[tuple[float, ...], float]],
        ] = {
            "feedback": {},
            "joint_state": {},
        }
        self._latest: dict[str, tuple[int, float]] = {}
        self._last_pair: dict[str, Any] | None = None

    def record(
        self,
        source: str,
        stamp_ns: int,
        positions: tuple[float, ...],
        received_at: float,
    ) -> None:
        if source not in self._samples:
            raise ValueError(f"unknown position source: {source}")
        if (
            len(positions) != len(JOINT_NAMES)
            or not all(math.isfinite(value) for value in positions)
        ):
            return
        stamp_ns = int(stamp_ns)
        received_at = float(received_at)
        samples = self._samples[source]
        samples[stamp_ns] = (positions, received_at)
        self._latest[source] = (stamp_ns, received_at)
        while len(samples) > POSITION_SOURCE_CACHE_SIZE:
            samples.pop(next(iter(samples)))

        other_source = (
            "joint_state" if source == "feedback" else "feedback"
        )
        other = self._samples[other_source].get(stamp_ns)
        if other is None:
            return
        other_positions, other_received_at = other
        feedback_positions = (
            positions if source == "feedback" else other_positions
        )
        joint_state_positions = (
            positions if source == "joint_state" else other_positions
        )
        self._last_pair = {
            "stamp_ns": stamp_ns,
            "matched_at": max(received_at, other_received_at),
            "maximum_delta_rad": max(
                abs(
                    feedback_positions[index]
                    - joint_state_positions[index]
                )
                for index in range(len(JOINT_NAMES))
            ),
        }

    def failures(self, *, now: float) -> list[str]:
        failures: list[str] = []
        if set(self._latest) != {"feedback", "joint_state"}:
            return ["timestamped Watson position sources are incomplete"]
        header_skew_s = abs(
            self._latest["feedback"][0]
            - self._latest["joint_state"][0]
        ) / 1_000_000_000.0
        if header_skew_s > POSITION_SOURCE_MAX_HEADER_SKEW_S:
            failures.append(
                "Watson position-source header skew is "
                f"{header_skew_s:.6f}s > "
                f"{POSITION_SOURCE_MAX_HEADER_SKEW_S:.6f}s"
            )
        if self._last_pair is None:
            failures.append(
                "no exact-timestamp Watson position-source pair was observed"
            )
            return failures
        pair_age_s = float(now) - float(self._last_pair["matched_at"])
        if pair_age_s < 0.0 or pair_age_s > POSITION_SOURCE_MAX_PAIR_AGE_S:
            failures.append(
                "exact-timestamp Watson position-source pair is stale "
                f"({pair_age_s:.6f}s > "
                f"{POSITION_SOURCE_MAX_PAIR_AGE_S:.6f}s)"
            )
        pair_delta_rad = float(self._last_pair["maximum_delta_rad"])
        if pair_delta_rad > POSITION_SOURCE_MAX_PAIRED_DELTA_RAD:
            failures.append(
                "exact-timestamp Watson position sources disagree "
                f"({pair_delta_rad:.6f}rad > "
                f"{POSITION_SOURCE_MAX_PAIRED_DELTA_RAD:.6f}rad)"
            )
        return failures


def message_stamp_ns(message: Any) -> int:
    """Return one validated ROS Header stamp as integer nanoseconds."""

    stamp = message.header.stamp
    seconds = int(stamp.sec)
    nanoseconds = int(stamp.nanosec)
    if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
        raise ValueError("invalid ROS message timestamp")
    return seconds * 1_000_000_000 + nanoseconds


def emit_hil_event(enabled: bool, event: str, **fields: Any) -> None:
    """Emit one machine-readable, non-authoritative GUI status event."""

    if not enabled:
        return
    if re.fullmatch(r"[a-z][a-z0-9_]*", event) is None:
        raise ValueError(f"invalid HIL event name: {event!r}")
    global _hil_event_sequence, _hil_final_event_emitted
    if _hil_final_event_emitted:
        raise RuntimeError("cannot emit a HIL event after the final run event")
    next_sequence = _hil_event_sequence + 1
    payload = {
        "schema_version": 1,
        "event_sequence": next_sequence,
        "event": event,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print(
        HIL_EVENT_PREFIX
        + json.dumps(
            payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    _hil_event_sequence = next_sequence
    if event in {"run_completed", "run_failed"}:
        _hil_final_event_emitted = True


def reset_hil_event_stream() -> None:
    """Start one fresh, process-local HIL event sequence."""

    global _hil_event_sequence, _hil_final_event_emitted
    _hil_event_sequence = 0
    _hil_final_event_emitted = False


def emit_hil_stage_failure(
    enabled: bool,
    stage: StageSpec,
    exc: BaseException,
    *,
    arm_stage_completed: bool,
) -> None:
    """Close only an arm stage that did not already report completion."""

    if arm_stage_completed:
        return
    emit_hil_event(
        enabled,
        "stage_failed",
        sequence_index=stage.sequence_index,
        stage_name=stage.stage_name,
        specimen_id=stage.specimen_id,
        error=f"{type(exc).__name__}: {exc}",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--mode",
        choices=("check", "dry-run", "execute"),
        default="check",
        help=(
            "check validates live gates without building a goal; dry-run also "
            "builds every exact ROS message; execute requires explicit arming"
        ),
    )
    parser.add_argument(
        "--offline-validate",
        action="store_true",
        help=(
            "validate the private artifacts without sourcing ROS, inspecting "
            "the network, starting Watson bring-up, or creating transports"
        ),
    )
    parser.add_argument("--namespace", default="/watson")
    parser.add_argument(
        "--retimed-artifact",
        type=Path,
        default=DEFAULT_RETIMED_ARTIFACT,
    )
    parser.add_argument(
        "--ingress-artifact",
        type=Path,
        default=DEFAULT_INGRESS_ARTIFACT,
    )
    parser.add_argument("--state-timeout", type=float, default=30.0)
    parser.add_argument("--service-timeout", type=float, default=20.0)
    parser.add_argument("--execution-timeout", type=float, default=120.0)
    parser.add_argument("--arm-token", default="")
    parser.add_argument(
        "--gripper-token",
        default="",
        help="Required exact 2FG7 confirmation token for execute",
    )
    parser.add_argument(
        "--confirm-cell-clear",
        action="store_true",
        help="Required for execute after checking the empty cell and E-stop",
    )
    parser.add_argument(
        "--resume-at-reviewed-ready",
        action="store_true",
        help=(
            "Execute only stages 1..49 after freshly proving the arm is at "
            "the exact reviewed READY boundary left by a completed ingress"
        ),
    )
    parser.add_argument(
        "--hil-events",
        action="store_true",
        help="Emit structured read-only status events for the Isaac HIL GUI",
    )
    parser.add_argument("--report", type=Path, default=None)
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    validate_execution_authorization(
        mode=args.mode,
        arm_token=args.arm_token,
        gripper_token=args.gripper_token,
        confirm_cell_clear=args.confirm_cell_clear,
        namespace=args.namespace,
    )
    if args.offline_validate and args.mode != "check":
        raise ValueError("--offline-validate requires the default --mode check")
    if args.resume_at_reviewed_ready and args.mode != "execute":
        raise ValueError("--resume-at-reviewed-ready requires --mode execute")
    for name in ("state_timeout", "service_timeout", "execution_timeout"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.mode == "execute":
        if os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST":
            raise ValueError(
                "execute requires ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST"
            )
        try:
            domain_id = int(os.environ.get("ROS_DOMAIN_ID", "-1"))
        except ValueError as exc:
            raise ValueError("execute requires an integer ROS_DOMAIN_ID") from exc
        if not 0 <= domain_id <= 232:
            raise ValueError(
                "execute requires an explicit ROS_DOMAIN_ID between 0 and 232"
            )


def select_execution_stages(
    bundle: ExecutionBundle,
    *,
    resume_at_reviewed_ready: bool,
) -> tuple[StageSpec, ...]:
    """Select only the fixed post-ingress boundary; arbitrary resume is absent."""

    if not resume_at_reviewed_ready:
        return bundle.stages
    if len(bundle.stages) != 50:
        raise ValueError("reviewed-ready resume requires the exact 50-stage bundle")
    ingress = bundle.stages[0]
    first_replay = bundle.stages[1]
    ready = tuple(float(value) for value in READY_JOINT_POSITIONS_RAD)
    if (
        ingress.sequence_index != 0
        or ingress.kind != "tool_aware_ingress"
        or ingress.specimen_id is not None
        or ingress.stage_index != -1
        or ingress.stage_name != "tool_aware_ready_ingress"
        or ingress.goal_positions != ready
    ):
        raise ValueError("reviewed-ready resume ingress boundary changed")
    if (
        first_replay.sequence_index != 1
        or first_replay.kind != "seven_pin_air_replay"
        or first_replay.specimen_id != 1
        or first_replay.stage_index != 0
        or first_replay.stage_name != "approach_tilted_pregrasp"
        or first_replay.start_positions != ready
        or first_replay.start_positions != ingress.goal_positions
    ):
        raise ValueError("reviewed-ready resume first replay boundary changed")
    return bundle.stages[1:]


def validate_air_replay_network() -> None:
    """Pin both Watson and its Compute Box to the dedicated physical NIC."""

    validate_execute_network()
    try:
        result = subprocess.run(
            ["ip", "-4", "route", "get", COMPUTE_BOX_IP],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"could not verify Compute Box route: {exc}") from exc
    route = result.stdout.strip()
    if (
        result.returncode != 0
        or f"dev {ROBOT_INTERFACE}" not in route
        or f"src {ROBOT_SOURCE_IP}" not in route
    ):
        raise ValueError(
            f"route to Compute Box {COMPUTE_BOX_IP} must use "
            f"dev {ROBOT_INTERFACE} src {ROBOT_SOURCE_IP}; observed: "
            f"{route or result.stderr.strip() or 'none'}"
        )


def load_ros() -> dict[str, Any]:
    try:
        import rclpy
        from action_msgs.msg import GoalStatus, GoalStatusArray
        from builtin_interfaces.msg import Duration
        from moveit_msgs.action import ExecuteTrajectory
        from moveit_msgs.msg import RobotTrajectory
        from rclpy.action import ActionClient
        from rclpy.action.graph import (
            get_action_client_names_and_types_by_node,
            get_action_server_names_and_types_by_node,
        )
        from rclpy.parameter_client import AsyncParameterClient
        from rclpy.qos import HistoryPolicy, qos_profile_action_status_default
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import JointState
        from tm_msgs.msg import FeedbackState
        from tm_msgs.srv import AskItem, AskSta, SendScript
        from trajectory_msgs.msg import JointTrajectoryPoint
    except Exception as exc:
        raise RuntimeError(
            "Source ROS 2 Jazzy and the Techman workspace install/setup.bash"
        ) from exc
    return {
        "rclpy": rclpy,
        "GoalStatus": GoalStatus,
        "GoalStatusArray": GoalStatusArray,
        "Duration": Duration,
        "ExecuteTrajectory": ExecuteTrajectory,
        "RobotTrajectory": RobotTrajectory,
        "JointTrajectoryPoint": JointTrajectoryPoint,
        "ActionClient": ActionClient,
        "AsyncParameterClient": AsyncParameterClient,
        "get_action_client_names_and_types_by_node": (
            get_action_client_names_and_types_by_node
        ),
        "get_action_server_names_and_types_by_node": (
            get_action_server_names_and_types_by_node
        ),
        "HistoryPolicy": HistoryPolicy,
        "qos_profile_action_status_default": qos_profile_action_status_default,
        "rmw_implementation_identifier": (
            rclpy.get_rmw_implementation_identifier()
        ),
        "SignalHandlerOptions": SignalHandlerOptions,
        "JointState": JointState,
        "FeedbackState": FeedbackState,
        "AskItem": AskItem,
        "AskSta": AskSta,
        "SendScript": SendScript,
    }


def report_payload_sha256(report: dict[str, Any]) -> str:
    payload = dict(report)
    payload.pop(REPORT_DIGEST_FIELD, None)
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class ReportReservation(NamedTuple):
    path: Path
    device: int
    inode: int
    content_sha256: str


def _normal_report_path(path: Path) -> Path:
    destination = path.expanduser()
    if not destination.is_absolute():
        destination = Path.cwd() / destination
    return destination.parent.resolve() / destination.name


def _encoded_report(report: dict[str, Any]) -> bytes:
    report[REPORT_DIGEST_FIELD] = report_payload_sha256(report)
    encoded = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    if len(encoded) > MAX_REPORT_BYTES:
        raise RuntimeError("air-replay report unexpectedly exceeds 2 MiB")
    return encoded


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def reserve_private_report(path: Path) -> ReportReservation:
    """Reserve and prove the report destination before any live contact."""

    destination = _normal_report_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sentinel = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "reserved_before_live_contact",
        "warning": (
            "The runner reserved this private path before live contact. "
            "If this remains, final execution state was not durably recorded."
        ),
        "runner_sha256": SCRIPT_SHA256,
    }
    encoded = _encoded_report(sentinel)
    descriptor = os.open(
        destination,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    metadata = destination.lstat()
    if (
        destination.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
    ):
        destination.unlink(missing_ok=True)
        raise RuntimeError("private air-replay report checks failed")
    _fsync_directory(destination.parent)
    return ReportReservation(
        path=destination,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        content_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def write_private_report(
    path: Path,
    report: dict[str, Any],
    *,
    reservation: ReportReservation | None = None,
) -> Path:
    destination = _normal_report_path(path)
    encoded = _encoded_report(report)
    if reservation is None:
        reservation = reserve_private_report(destination)
    if destination != reservation.path:
        raise RuntimeError("air-replay report reservation path changed")
    before = destination.lstat()
    if (
        destination.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or (before.st_dev, before.st_ino)
        != (reservation.device, reservation.inode)
        or hashlib.sha256(destination.read_bytes()).hexdigest()
        != reservation.content_sha256
    ):
        raise RuntimeError("air-replay report reservation changed")

    temporary = destination.with_name(
        f".{destination.name}.final-{os.getpid()}-{secrets.token_hex(8)}"
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
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        current = destination.lstat()
        if (current.st_dev, current.st_ino) != (
            reservation.device,
            reservation.inode,
        ):
            raise RuntimeError("air-replay report reservation was replaced")
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    metadata = destination.lstat()
    if (
        destination.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or hashlib.sha256(destination.read_bytes()).hexdigest()
        != hashlib.sha256(encoded).hexdigest()
    ):
        raise RuntimeError("final private air-replay report checks failed")
    print(f"Report: {destination}")
    return destination


def write_report_best_effort(
    path: Path,
    report: dict[str, Any],
    *,
    reservation: ReportReservation | None = None,
) -> None:
    try:
        write_private_report(path, report, reservation=reservation)
    except BaseException as exc:
        print(
            f"REPORT WRITE FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def guarded_gripper_transition(
    runtime: Any,
    transport: Any,
    action: GripperAction | str,
    *,
    confirmation: str,
    command_runner: Any = run_guarded_command,
) -> dict[str, Any]:
    """Run one fixed 2FG7 action between two stationary arm proofs."""

    parsed_action = (
        action if isinstance(action, GripperAction) else GripperAction(action)
    )
    transition: dict[str, Any] = {
        "action": parsed_action.value,
        "completed": False,
        "arm_health_before": None,
        "command_report": None,
        "arm_state_refresh_after_command": None,
        "arm_health_after": None,
        "arm_stop_unverified": False,
        "failures": [],
    }
    gripper_command_completed = False
    try:
        before = runtime.require_healthy(
            stationary=True,
            exact_project_speed=True,
        )
        transition["arm_health_before"] = asdict(before)
        if runtime.stop_requested:
            raise RuntimeError(
                f"stop requested by signal {runtime.stop_signal}"
            )
        command_kwargs = {
            "execute": True,
            "confirmation": confirmation,
            "transport": transport,
            "abort_requested": lambda: runtime.stop_requested,
        }
        dispatch_guard = getattr(runtime, "command_dispatch", None)
        if dispatch_guard is not None:
            command_kwargs["dispatch_guard"] = dispatch_guard
        command_report = command_runner(parsed_action, **command_kwargs)
        transition["command_report"] = command_report
        if (
            command_report.get("status") != "completed"
            or command_report.get("completed") is not True
        ):
            failures = command_report.get("failures")
            detail = (
                "; ".join(str(item) for item in failures)
                if isinstance(failures, list) and failures
                else "guarded 2FG7 helper did not report completion"
            )
            raise RuntimeError(detail)
        gripper_command_completed = True
        try:
            refreshed = runtime.refresh_after_blocking_gripper_call()
            transition["arm_state_refresh_after_command"] = asdict(refreshed)
            after = runtime.require_healthy(
                stationary=True,
                exact_project_speed=True,
            )
        except (OSError, ValueError, RuntimeError):
            transition["arm_stop_unverified"] = True
            raise
        transition["arm_health_after"] = asdict(after)
        transition["completed"] = True
    except (OSError, ValueError, RuntimeError) as exc:
        if gripper_command_completed and transition["arm_health_after"] is None:
            transition["arm_stop_unverified"] = True
        transition["failures"].append(str(exc))
    return transition


def gripper_command_may_be_moving(transition: dict[str, Any]) -> bool:
    """Conservatively identify an unresolved Compute Box command."""

    command_report = transition.get("command_report")
    if not isinstance(command_report, dict):
        return False
    if command_report.get("completed") is True:
        return False
    safety = command_report.get("safety_evidence")
    if not isinstance(safety, dict) or not safety.get(
        "gripper_command_may_have_been_sent"
    ):
        return False
    recovery = command_report.get("recovery_stop")
    return not (
        isinstance(recovery, dict)
        and recovery.get("attempted") is True
        and recovery.get("failures") == []
        and isinstance(recovery.get("state_after"), dict)
    )


def raise_gripper_transition_failure(
    context: str,
    transition: dict[str, Any],
) -> None:
    """Preserve an uncertain arm stop as an E-stop-class failure."""

    failures = transition.get("failures")
    detail = (
        "; ".join(str(item) for item in failures)
        if isinstance(failures, list) and failures
        else "guarded 2FG7 transition failed"
    )
    message = f"{context}: {detail}"
    if transition.get("arm_stop_unverified") is True:
        raise StopUnverifiedError(message + "; use the physical E-stop")
    raise RuntimeError(message)


def best_effort_gripper_stop(
    transport: Any,
    *,
    confirmation: str,
    command_runner: Any = run_guarded_command,
) -> dict[str, Any]:
    """Send fixed recovery STOP even when passive pre-state is unavailable."""

    try:
        if command_runner is run_guarded_command:
            return run_fixed_recovery_stop(
                confirmation=confirmation,
                transport=transport,
            )
        return command_runner(
            GripperAction.STOP,
            execute=True,
            confirmation=confirmation,
            transport=transport,
        )
    except BaseException as exc:
        return {
            "action": "stop",
            "status": "best_effort_stop_raised",
            "completed": False,
            "failures": [f"{type(exc).__name__}: {exc}"],
        }


def gripper_stop_verified(stop_report: Any) -> bool:
    """Require a completed STOP plus a fresh non-busy post-state."""

    if (
        not isinstance(stop_report, dict)
        or stop_report.get("status") != "completed"
        or stop_report.get("completed") is not True
    ):
        return False
    state_after = stop_report.get("state_after")
    state = (
        state_after.get("state")
        if isinstance(state_after, dict)
        else None
    )
    return isinstance(state, dict) and state.get("busy") is False


class AirReplayNode:
    """One exact MoveIt execution client plus read-only Watson state clients."""

    def __init__(
        self,
        args: argparse.Namespace,
        ros: dict[str, Any],
        signal_gate: ProcessSignalGate | None = None,
    ) -> None:
        self.args = args
        self.ros = ros
        self.rclpy = ros["rclpy"]
        self.namespace = "/" + args.namespace.strip("/")
        self.node = self.rclpy.create_node("watson_multi_pin_air_replay")
        self.feedback = None
        self.feedback_received_at = 0.0
        self.joint_positions: tuple[float, ...] | None = None
        self.joint_velocities: tuple[float, ...] = ()
        self.joint_state_received_at = 0.0
        self.position_source_pairs = PositionSourcePairTracker()
        self.execute_action_status = None
        self.controller_action_status = None
        self.execute_action_status_received_at = 0.0
        self.controller_action_status_received_at = 0.0
        self.execute_action_status_generation = 0
        self.controller_action_status_generation = 0
        self.signal_gate = signal_gate
        self._stop_requested = False
        self.stop_signal: int | None = None
        self.motion_command_sent = False
        self.active_goal_handle = None
        self.active_result_future = None
        self.tool_selection_evidence: dict[str, Any] = {
            "attempted": False,
            "request_id": TOOL_SELECT_REQUEST_ID,
            "script_sha256": hashlib.sha256(
                TOOL_SELECT_SCRIPT.encode("utf-8")
            ).hexdigest(),
            "response_received": False,
            "response_ok": None,
            "fresh_exact_tool_readback": None,
            "readback_failures": [],
        }

        self.node.create_subscription(
            ros["FeedbackState"],
            f"{self.namespace}/feedback_states",
            self._feedback_callback,
            20,
        )
        self.node.create_subscription(
            ros["JointState"],
            f"{self.namespace}/joint_states",
            self._joint_state_callback,
            20,
        )
        self.node.create_subscription(
            ros["GoalStatusArray"],
            f"{EXECUTE_ACTION}/_action/status",
            self._execute_status_callback,
            ros["qos_profile_action_status_default"],
        )
        self.node.create_subscription(
            ros["GoalStatusArray"],
            f"{CONTROLLER_ACTION}/_action/status",
            self._controller_status_callback,
            ros["qos_profile_action_status_default"],
        )
        self.execute_client = ros["ActionClient"](
            self.node,
            ros["ExecuteTrajectory"],
            EXECUTE_ACTION,
        )
        self.tool_client = self.node.create_client(
            ros["AskItem"],
            f"{self.namespace}/ask_item",
        )
        self.tool_select_client = None
        if args.mode == "execute":
            self.tool_select_client = self.node.create_client(
                ros["SendScript"],
                f"{self.namespace}/send_script",
            )
        self.status_client = self.node.create_client(
            ros["AskSta"],
            f"{self.namespace}/ask_sta",
        )
        self.move_group_parameters = ros["AsyncParameterClient"](
            self.node,
            f"{self.namespace}/move_group",
        )

    def _feedback_callback(self, message: Any) -> None:
        received_at = time.monotonic()
        self.feedback = message
        self.feedback_received_at = received_at
        positions = tuple(float(value) for value in message.joint_pos)
        self.position_source_pairs.record(
            "feedback",
            message_stamp_ns(message),
            positions,
            received_at,
        )

    def _joint_state_callback(self, message: Any) -> None:
        received_at = time.monotonic()
        positions = dict(zip(message.name, message.position))
        velocities = dict(zip(message.name, message.velocity))
        if not all(name in positions for name in JOINT_NAMES):
            return
        self.joint_positions = tuple(float(positions[name]) for name in JOINT_NAMES)
        self.joint_velocities = (
            tuple(float(velocities[name]) for name in JOINT_NAMES)
            if all(name in velocities for name in JOINT_NAMES)
            else ()
        )
        self.joint_state_received_at = received_at
        self.position_source_pairs.record(
            "joint_state",
            message_stamp_ns(message),
            self.joint_positions,
            received_at,
        )

    def _execute_status_callback(self, message: Any) -> None:
        self.execute_action_status = message
        self.execute_action_status_received_at = time.monotonic()
        self.execute_action_status_generation += 1

    def _controller_status_callback(self, message: Any) -> None:
        self.controller_action_status = message
        self.controller_action_status_received_at = time.monotonic()
        self.controller_action_status_generation += 1

    @property
    def stop_requested(self) -> bool:
        gate = getattr(self, "signal_gate", None)
        if gate is not None:
            signum = gate.poll()
            if signum is not None:
                self._stop_requested = True
                if self.stop_signal is None:
                    self.stop_signal = signum
        return bool(getattr(self, "_stop_requested", False))

    @stop_requested.setter
    def stop_requested(self, value: bool) -> None:
        self._stop_requested = bool(value)

    def request_stop(self, signum: int, _frame: Any = None) -> None:
        self._stop_requested = True
        self.stop_signal = signum

    def command_dispatch(self, label: str) -> Any:
        gate = getattr(self, "signal_gate", None)
        if gate is None:
            return nullcontext()
        return gate.command_commit(label)

    def snapshot(self) -> HealthSnapshot:
        if self.feedback is None or self.joint_positions is None:
            raise RuntimeError("Watson feedback and joint state are incomplete")
        feedback_velocities = tuple(
            float(value) for value in self.feedback.joint_vel
        )
        velocities = (
            feedback_velocities
            if len(feedback_velocities) == len(JOINT_NAMES)
            else self.joint_velocities
        )
        now = time.monotonic()
        return HealthSnapshot(
            is_svr_connected=bool(self.feedback.is_svr_connected),
            is_sct_connected=bool(self.feedback.is_sct_connected),
            tmsrv_cperr=int(self.feedback.tmsrv_cperr),
            tmscript_cperr=int(self.feedback.tmscript_cperr),
            tmsrv_dataerr=int(self.feedback.tmsrv_dataerr),
            tmscript_dataerr=int(self.feedback.tmscript_dataerr),
            is_data_table_correct=bool(self.feedback.is_data_table_correct),
            robot_link=bool(self.feedback.robot_link),
            robot_error=bool(self.feedback.robot_error),
            project_run=bool(self.feedback.project_run),
            project_pause=bool(self.feedback.project_pause),
            safetyguard_a=bool(self.feedback.safetyguard_a),
            e_stop=bool(self.feedback.e_stop),
            error_code=int(self.feedback.error_code),
            project_speed=int(self.feedback.project_speed),
            ma_mode=int(self.feedback.ma_mode),
            robot_light=int(self.feedback.robot_light),
            joint_positions=self.joint_positions,
            feedback_joint_positions=tuple(
                float(value) for value in self.feedback.joint_pos
            ),
            joint_velocities=velocities,
            feedback_age_s=now - self.feedback_received_at,
            joint_state_age_s=now - self.joint_state_received_at,
        )

    def spin_until_state(self) -> HealthSnapshot:
        deadline = time.monotonic() + self.args.state_timeout
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.feedback is not None and self.joint_positions is not None:
                return self.snapshot()
        raise RuntimeError("timed out waiting for Watson feedback and joints")

    def publisher_failures(self) -> list[str]:
        failures: list[str] = []
        for suffix in ("joint_states", "feedback_states"):
            topic = f"{self.namespace}/{suffix}"
            publishers = self.node.get_publishers_info_by_topic(topic)
            owners = sorted(
                (item.node_name, item.node_namespace) for item in publishers
            )
            if owners != [("tm_driver_node", self.namespace)]:
                failures.append(f"unexpected {topic} publishers: {owners}")
        return failures

    def graph_failures(self) -> list[str]:
        action_servers: list[tuple[str, str, str, list[str]]] = []
        action_clients: list[tuple[str, str, str, list[str]]] = []
        graph_errors: list[str] = []
        guarded_client_endpoints = {
            EXECUTE_ACTION,
            CONTROLLER_ACTION,
            f"{self.namespace}/move_action",
            f"{self.namespace}/sequence_move_group",
        }
        for node_name, node_namespace in self.node.get_node_names_and_namespaces():
            try:
                endpoints = self.ros[
                    "get_action_server_names_and_types_by_node"
                ](self.node, node_name, node_namespace)
                for endpoint, endpoint_types in endpoints:
                    if endpoint in {EXECUTE_ACTION, CONTROLLER_ACTION}:
                        action_servers.append(
                            (
                                endpoint,
                                node_name,
                                node_namespace,
                                list(endpoint_types),
                            )
                        )
            except Exception as exc:
                graph_errors.append(
                    f"action-server graph failed for "
                    f"{node_namespace}/{node_name}: {exc}"
                )
            try:
                endpoints = self.ros[
                    "get_action_client_names_and_types_by_node"
                ](self.node, node_name, node_namespace)
                for endpoint, endpoint_types in endpoints:
                    if endpoint in guarded_client_endpoints:
                        action_clients.append(
                            (
                                endpoint,
                                node_name,
                                node_namespace,
                                list(endpoint_types),
                            )
                        )
            except Exception as exc:
                graph_errors.append(
                    f"action-client graph failed for "
                    f"{node_namespace}/{node_name}: {exc}"
                )
        expected_servers = sorted(
            [
                (
                    EXECUTE_ACTION,
                    "move_group",
                    self.namespace,
                    ["moveit_msgs/action/ExecuteTrajectory"],
                ),
                (
                    CONTROLLER_ACTION,
                    "tm_driver_node",
                    self.namespace,
                    ["control_msgs/action/FollowJointTrajectory"],
                ),
            ]
        )
        expected_clients = sorted(
            [
                (
                    EXECUTE_ACTION,
                    "watson_multi_pin_air_replay",
                    "/",
                    ["moveit_msgs/action/ExecuteTrajectory"],
                ),
                (
                    CONTROLLER_ACTION,
                    "moveit_simple_controller_manager",
                    self.namespace,
                    ["control_msgs/action/FollowJointTrajectory"],
                ),
            ]
        )
        failures = list(graph_errors)
        if sorted(action_servers) != expected_servers:
            failures.append(
                f"unexpected Watson action servers: {sorted(action_servers)}"
            )
        if sorted(action_clients) != expected_clients:
            failures.append(
                f"unexpected Watson action clients: {sorted(action_clients)}"
            )
        return failures

    def wait_for_graph(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        failures = ["action graph has not been inspected"]
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            failures = self.publisher_failures() + self.graph_failures()
            if not failures:
                return
        raise RuntimeError("Watson graph provenance failed: " + "; ".join(failures))

    def _action_status_specs(
        self,
    ) -> tuple[tuple[str, str, str, str], ...]:
        return (
            (
                "execute",
                "MoveIt execute action",
                f"{EXECUTE_ACTION}/_action/status",
                "move_group",
            ),
            (
                "controller",
                "Techman controller action",
                f"{CONTROLLER_ACTION}/_action/status",
                "tm_driver_node",
            ),
        )

    def action_status_publisher_snapshot(
        self,
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        """Read action-status writer identity and discoverable durable QoS.

        DDS discovery does not carry publisher history or history depth. The
        RMW API therefore legitimately reports ``UNKNOWN`` and ``0`` for that
        pair. Reliability and durability do participate in DDS endpoint
        matching and remain exact requirements here.
        """

        expected_qos = self.ros["qos_profile_action_status_default"]
        snapshot: dict[str, dict[str, Any]] = {}
        failures: list[str] = []
        for key, label, topic, expected_node in self._action_status_specs():
            try:
                publishers = self.node.get_publishers_info_by_topic(topic)
            except Exception as exc:
                failures.append(f"{label} publisher graph failed: {exc}")
                continue
            if len(publishers) != 1:
                owners = sorted(
                    (item.node_name, item.node_namespace)
                    for item in publishers
                )
                failures.append(
                    f"{label} must have one status publisher, observed {owners}"
                )
                continue
            publisher = publishers[0]
            owner = (publisher.node_name, publisher.node_namespace)
            if owner != (expected_node, self.namespace):
                failures.append(
                    f"{label} status publisher owner is {owner}, expected "
                    f"{(expected_node, self.namespace)}"
                )
            if publisher.topic_type != ACTION_STATUS_TYPE:
                failures.append(
                    f"{label} status type is {publisher.topic_type!r}, expected "
                    f"{ACTION_STATUS_TYPE!r}"
                )
            qos = publisher.qos_profile
            for field in ("reliability", "durability"):
                observed = getattr(qos, field, None)
                expected = getattr(expected_qos, field, None)
                if observed != expected:
                    failures.append(
                        f"{label} status QoS {field} is {observed!r}, "
                        f"expected {expected!r}"
                    )
            observed_history = getattr(qos, "history", None)
            observed_depth = getattr(qos, "depth", None)
            expected_history = getattr(expected_qos, "history", None)
            expected_depth = getattr(expected_qos, "depth", None)
            if (
                observed_history == expected_history
                and observed_depth == expected_depth
            ):
                history_depth_evidence = "reported_exact"
            elif (
                observed_history == self.ros["HistoryPolicy"].UNKNOWN
                and observed_depth == 0
                and self.ros["rmw_implementation_identifier"]
                == "rmw_fastrtps_cpp"
            ):
                history_depth_evidence = "dds_discovery_unavailable"
            else:
                history_depth_evidence = "mismatch"
                failures.append(
                    f"{label} status QoS history/depth is "
                    f"{observed_history!r}/{observed_depth!r}, expected "
                    f"{expected_history!r}/{expected_depth!r} or the DDS "
                    "discovery sentinel UNKNOWN/0 from rmw_fastrtps_cpp"
                )
            endpoint_gid = tuple(int(value) for value in publisher.endpoint_gid)
            if not endpoint_gid or not any(endpoint_gid):
                failures.append(f"{label} status publisher GID is unavailable")
            snapshot[key] = {
                "label": label,
                "topic": topic,
                "node_name": publisher.node_name,
                "node_namespace": publisher.node_namespace,
                "topic_type": publisher.topic_type,
                "endpoint_gid": list(endpoint_gid),
                "qos": {
                    "history": getattr(
                        getattr(qos, "history", None),
                        "name",
                        str(getattr(qos, "history", None)),
                    ),
                    "depth": getattr(qos, "depth", None),
                    "reliability": getattr(
                        getattr(qos, "reliability", None),
                        "name",
                        str(getattr(qos, "reliability", None)),
                    ),
                    "durability": getattr(
                        getattr(qos, "durability", None),
                        "name",
                        str(getattr(qos, "durability", None)),
                    ),
                    "history_depth_evidence": history_depth_evidence,
                    "rmw_implementation_identifier": self.ros[
                        "rmw_implementation_identifier"
                    ],
                },
            }
        return snapshot, failures

    @staticmethod
    def _action_status_gid_map(
        snapshot: dict[str, dict[str, Any]],
    ) -> dict[str, tuple[int, ...]]:
        return {
            key: tuple(int(value) for value in item["endpoint_gid"])
            for key, item in snapshot.items()
        }

    def _status_message_failures(
        self,
        messages: dict[str, Any],
        *,
        require_all: bool,
    ) -> list[str]:
        terminal = {
            self.ros["GoalStatus"].STATUS_SUCCEEDED,
            self.ros["GoalStatus"].STATUS_CANCELED,
            self.ros["GoalStatus"].STATUS_ABORTED,
        }
        failures: list[str] = []
        for key, label, _topic, _owner in self._action_status_specs():
            statuses = messages.get(key)
            if statuses is None:
                if require_all:
                    failures.append(
                        f"{label} has no retained status evidence"
                    )
                continue
            nonterminal = [
                status.status
                for status in statuses.status_list
                if status.status not in terminal
            ]
            if nonterminal:
                failures.append(f"{label} has nonterminal status {nonterminal}")
        return failures

    def action_busy_failures(
        self,
        *,
        require_present: bool = False,
    ) -> list[str]:
        """Inspect the latest persistent status samples without wall-clock age."""

        return self._status_message_failures(
            {
                "execute": self.execute_action_status,
                "controller": self.controller_action_status,
            },
            require_all=require_present,
        )

    def require_action_idle(self) -> dict[str, Any]:
        """Read a fresh transient-local snapshot from both action servers.

        A new subscription is required for each ordinary idle gate so the
        current transient-local writers must replay their retained latest
        sample. Before this runner's first command, no retained sample is also
        valid after the full settle window: Jazzy publishes status only after a
        goal exists. Once any command has been attempted, both writers must
        replay terminal/empty state.
        """

        start_publishers, start_failures = (
            self.action_status_publisher_snapshot()
        )
        start_failures.extend(self.publisher_failures())
        start_failures.extend(self.graph_failures())
        if start_failures:
            raise RuntimeError(
                "Watson action-idle publisher provenance failed: "
                + "; ".join(start_failures)
            )
        start_gids = self._action_status_gid_map(start_publishers)
        if set(start_gids) != {"execute", "controller"}:
            raise RuntimeError(
                "Watson action-idle publisher provenance is incomplete"
            )

        messages: dict[str, Any] = {
            "execute": None,
            "controller": None,
        }
        received_at: dict[str, float | None] = {
            "execute": None,
            "controller": None,
        }
        subscriptions: list[Any] = []
        cleanup_failures: list[str] = []
        permanent_failures: list[str] = []
        deadline = time.monotonic() + ACTION_IDLE_TIMEOUT_S
        last_failures = ["fresh action-idle evidence was not observed"]
        proof: dict[str, Any] | None = None

        def capture(key: str) -> Any:
            def callback(message: Any) -> None:
                messages[key] = message
                received_at[key] = time.monotonic()

            return callback

        try:
            for key, _label, topic, _owner in self._action_status_specs():
                subscriptions.append(
                    self.node.create_subscription(
                        self.ros["GoalStatusArray"],
                        topic,
                        capture(key),
                        self.ros["qos_profile_action_status_default"],
                    )
                )
            while time.monotonic() < deadline:
                self.rclpy.spin_once(self.node, timeout_sec=0.05)
                current_publishers, provenance_failures = (
                    self.action_status_publisher_snapshot()
                )
                provenance_failures.extend(self.publisher_failures())
                provenance_failures.extend(self.graph_failures())
                current_gids = self._action_status_gid_map(
                    current_publishers
                )
                if current_gids != start_gids:
                    failure = (
                        "action status publisher GID changed during idle proof: "
                        f"{start_gids!r} -> {current_gids!r}"
                    )
                    if failure not in permanent_failures:
                        permanent_failures.append(failure)
                status_failures = self._status_message_failures(
                    messages,
                    require_all=self.motion_command_sent,
                )
                last_failures = (
                    permanent_failures
                    + provenance_failures
                    + status_failures
                )
                if (
                    not last_failures
                    and all(messages[key] is not None for key in messages)
                ):
                    proof = {
                        "verified": True,
                        "basis": "fresh_transient_local_terminal_status",
                        "publisher_snapshot": current_publishers,
                        "execute_status_reception_age_s": (
                            time.monotonic() - received_at["execute"]
                        ),
                        "controller_status_reception_age_s": (
                            time.monotonic() - received_at["controller"]
                        ),
                    }
                    break
            if proof is None:
                final_publishers, final_failures = (
                    self.action_status_publisher_snapshot()
                )
                final_failures.extend(self.publisher_failures())
                final_failures.extend(self.graph_failures())
                final_gids = self._action_status_gid_map(final_publishers)
                if final_gids != start_gids:
                    permanent_failures.append(
                        "action status publisher GID changed at idle-proof "
                        f"decision: {start_gids!r} -> {final_gids!r}"
                    )
                final_failures = (
                    permanent_failures
                    + final_failures
                    + self._status_message_failures(
                        messages,
                        require_all=self.motion_command_sent,
                    )
                )
                missing = [
                    key for key, message in messages.items()
                    if message is None
                ]
                if (
                    not final_failures
                    and not self.motion_command_sent
                    and len(missing) == len(messages)
                ):
                    proof = {
                        "verified": True,
                        "basis": (
                            "no_retained_sample_before_first_goal_"
                            "after_full_settle"
                        ),
                        "settle_seconds": ACTION_IDLE_TIMEOUT_S,
                        "missing_status_samples": missing,
                        "publisher_snapshot": final_publishers,
                    }
                else:
                    if (
                        not final_failures
                        and not self.motion_command_sent
                        and missing
                    ):
                        final_failures.append(
                            "only one action server replayed retained status; "
                            "pre-goal no-sample proof requires both absent"
                        )
                    last_failures = final_failures
        finally:
            for subscription in subscriptions:
                try:
                    destroyed = self.node.destroy_subscription(subscription)
                    if destroyed is False:
                        cleanup_failures.append(
                            "node rejected temporary status subscription "
                            "destruction"
                        )
                except Exception as exc:
                    cleanup_failures.append(
                        f"temporary status subscription cleanup raised: {exc}"
                    )
        if cleanup_failures:
            raise RuntimeError(
                "Watson action-idle temporary subscription cleanup failed: "
                + "; ".join(cleanup_failures)
            )
        if proof is not None:
            return proof
        raise RuntimeError(
            "Watson action-idle proof failed: " + "; ".join(last_failures)
        )

    def capture_action_status_checkpoint(self) -> dict[str, Any]:
        """Bind a future terminal proof to status events after the next send."""

        publishers, failures = self.action_status_publisher_snapshot()
        failures.extend(self.publisher_failures())
        failures.extend(self.graph_failures())
        if failures:
            raise RuntimeError(
                "action-status checkpoint provenance failed: "
                + "; ".join(failures)
            )
        return {
            "execute_generation": self.execute_action_status_generation,
            "controller_generation": self.controller_action_status_generation,
            "publisher_gids": self._action_status_gid_map(publishers),
        }

    def require_goal_specific_action_idle(
        self,
        checkpoint: dict[str, Any],
    ) -> dict[str, Any]:
        """Require terminal samples generated after one specific goal send."""

        expected_gids = checkpoint["publisher_gids"]
        deadline = time.monotonic() + ACTION_IDLE_TIMEOUT_S
        last_failures = ["goal-specific status evidence was not observed"]
        permanent_failures: list[str] = []
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            publishers, failures = self.action_status_publisher_snapshot()
            failures.extend(self.publisher_failures())
            failures.extend(self.graph_failures())
            current_gids = self._action_status_gid_map(publishers)
            if current_gids != expected_gids:
                failure = (
                    "action status publisher GID changed after goal send: "
                    f"{expected_gids!r} -> {current_gids!r}"
                )
                if failure not in permanent_failures:
                    permanent_failures.append(failure)
            generation_failures: list[str] = []
            if (
                self.execute_action_status_generation
                <= checkpoint["execute_generation"]
            ):
                generation_failures.append(
                    "MoveIt execute status did not advance after this goal send"
                )
            if (
                self.controller_action_status_generation
                <= checkpoint["controller_generation"]
            ):
                generation_failures.append(
                    "Techman controller status did not advance after this "
                    "goal send"
                )
            status_failures = self.action_busy_failures(
                require_present=True
            )
            last_failures = (
                permanent_failures
                + failures
                + generation_failures
                + status_failures
            )
            if not last_failures:
                return {
                    "verified": True,
                    "basis": "goal_specific_terminal_status_generations",
                    "execute_generation_before": checkpoint[
                        "execute_generation"
                    ],
                    "execute_generation_after": (
                        self.execute_action_status_generation
                    ),
                    "controller_generation_before": checkpoint[
                        "controller_generation"
                    ],
                    "controller_generation_after": (
                        self.controller_action_status_generation
                    ),
                    "publisher_snapshot": publishers,
                }
        raise RuntimeError(
            "Watson goal-specific action-idle proof failed: "
            + "; ".join(last_failures)
        )

    def require_execution_enabled(self) -> None:
        if not self.move_group_parameters.wait_for_services(
            timeout_sec=self.args.service_timeout
        ):
            raise RuntimeError("move_group parameter service is unavailable")
        future = self.move_group_parameters.get_parameters(
            ["allow_trajectory_execution"]
        )
        self.rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=self.args.service_timeout,
        )
        if not future.done() or future.result() is None:
            raise RuntimeError("could not read allow_trajectory_execution")
        values = future.result().values
        if len(values) != 1 or values[0].bool_value is not True:
            raise RuntimeError("MoveIt allow_trajectory_execution is not true")

    def require_listen1(self) -> dict[str, Any]:
        if not self.status_client.wait_for_service(
            timeout_sec=self.args.service_timeout
        ):
            raise RuntimeError("read-only ask_sta service is unavailable")
        request = self.ros["AskSta"].Request()
        request.subcmd = "00"
        request.subdata = ""
        request.wait_time = min(float(self.args.service_timeout), 2.0)
        future = self.status_client.call_async(request)
        self.rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=self.args.service_timeout,
        )
        if not future.done() or future.result() is None:
            raise RuntimeError("TMSTA 00 Listen query timed out")
        response = future.result()
        if (
            response.ok is not True
            or response.subcmd != "00"
            or response.subdata != "true,Listen1"
        ):
            raise RuntimeError(
                "Watson must be at exact Listen1; observed "
                f"ok={response.ok!r} subcmd={response.subcmd!r} "
                f"subdata={response.subdata!r}"
            )
        return {
            "ok": True,
            "subcmd": "00",
            "subdata": "true,Listen1",
            "read_only": True,
        }

    def read_tool_audit(
        self,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return query_controller_tool_items(
            node=self.node,
            rclpy=self.rclpy,
            ask_item_type=self.ros["AskItem"],
            client=self.tool_client,
            timeout_s=(
                self.args.service_timeout
                if timeout_s is None
                else float(timeout_s)
            ),
        )

    def require_exact_tool(self) -> dict[str, Any]:
        audit = self.read_tool_audit()
        failures = exact_tool_audit_failures(audit)
        if failures:
            raise RuntimeError("controller tool gate failed: " + "; ".join(failures))
        return audit

    def select_exact_tool(self) -> dict[str, Any]:
        """Select the one reviewed TCP, then prove its fresh readback."""

        evidence = self.tool_selection_evidence
        if self.args.mode != "execute" or self.tool_select_client is None:
            raise RuntimeError("tool selection exists only in execute mode")
        if re.fullmatch(r"[A-Za-z0-9]+", TOOL_SELECT_REQUEST_ID) is None:
            raise RuntimeError("tool selection request id must be alphanumeric")
        if self.stop_requested:
            raise RuntimeError(f"stop requested by signal {self.stop_signal}")
        if not self.tool_select_client.wait_for_service(
            timeout_sec=self.args.service_timeout
        ):
            raise RuntimeError("send_script service is unavailable")
        request = self.ros["SendScript"].Request()
        request.id = TOOL_SELECT_REQUEST_ID
        request.script = TOOL_SELECT_SCRIPT
        evidence["attempted"] = True
        with self.command_dispatch("ChangeTCP_QC_2FG7_VENDOR"):
            if self.stop_requested:
                raise RuntimeError(
                    f"stop requested by signal {self.stop_signal}"
                )
            future = self.tool_select_client.call_async(request)
        self.rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=self.args.service_timeout,
        )
        if (
            not future.done()
            or future.result() is None
            or future.result().ok is not True
        ):
            evidence["response_received"] = (
                future.done() and future.result() is not None
            )
            evidence["response_ok"] = (
                None
                if not evidence["response_received"]
                else bool(future.result().ok)
            )
            observed = (
                None
                if not future.done() or future.result() is None
                else bool(future.result().ok)
            )
            raise RuntimeError(
                "ChangeTCP QC_2FG7_VENDOR failed: "
                f"send_script ok={observed!r}"
            )
        evidence["response_received"] = True
        evidence["response_ok"] = True

        deadline = time.monotonic() + self.args.service_timeout
        last_failures = ["fresh exact tool readback was not observed"]
        while time.monotonic() < deadline:
            if self.stop_requested:
                raise RuntimeError(
                    f"stop requested by signal {self.stop_signal}"
                )
            # This also proves CPERR/dataerr remain clear after ChangeTCP.
            self.require_healthy(
                stationary=True,
                exact_project_speed=True,
            )
            remaining = max(0.05, deadline - time.monotonic())
            try:
                audit = self.read_tool_audit(timeout_s=min(2.0, remaining))
                last_failures = exact_tool_audit_failures(audit)
                evidence["fresh_exact_tool_readback"] = audit
                evidence["readback_failures"] = list(last_failures)
                if not last_failures:
                    return dict(evidence)
            except (OSError, ValueError, RuntimeError) as exc:
                last_failures = [str(exc)]
                evidence["readback_failures"] = list(last_failures)
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        raise RuntimeError(
            "ChangeTCP returned ok but exact QC_2FG7_VENDOR readback "
            "did not become healthy: "
            + "; ".join(last_failures)
        )

    def require_healthy(
        self,
        *,
        stationary: bool,
        exact_project_speed: bool = False,
        honor_stop_request: bool = True,
    ) -> HealthSnapshot:
        deadline = time.monotonic() + 1.0
        first_positions = None
        latest = None
        while time.monotonic() < deadline:
            if honor_stop_request and self.stop_requested:
                raise RuntimeError(f"stop requested by signal {self.stop_signal}")
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            latest = self.snapshot()
            if first_positions is None:
                first_positions = latest.feedback_joint_positions
            failures = health_failures(
                latest,
                max_project_speed=MAX_PROJECT_SPEED,
                require_stationary=stationary,
                require_auto_mode=True,
            )
            if exact_project_speed:
                failures.extend(
                    exact_execute_project_speed_failures(latest)
                )
            failures.extend(self.publisher_failures())
            if stationary and first_positions is not None:
                drift = max(
                    abs(latest.feedback_joint_positions[index] - first_positions[index])
                    for index in range(len(JOINT_NAMES))
                )
                if drift > 0.001:
                    failures.append(
                        f"stationary proof drift {drift:.6f}rad exceeds 0.001rad"
                    )
            if failures:
                raise RuntimeError("Watson health gate failed: " + "; ".join(failures))
        if latest is None:
            raise RuntimeError("no Watson state observed during health proof")
        return latest

    def refresh_after_blocking_gripper_call(self) -> HealthSnapshot:
        """Drain ROS state queued while the synchronous Compute Box call ran."""

        deadline = time.monotonic() + 1.0
        last_refresh_failures = ["fresh Watson state was not observed"]
        refreshable_prefixes = (
            "feedback is stale",
            "joint state is stale",
            "joint_states and feedback joint positions disagree",
        )
        while time.monotonic() < deadline:
            if self.stop_requested:
                raise RuntimeError(
                    f"stop requested by signal {self.stop_signal}"
                )
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            snapshot = self.snapshot()
            failures = health_failures(
                snapshot,
                max_project_speed=MAX_PROJECT_SPEED,
                require_stationary=True,
                require_auto_mode=True,
            )
            failures.extend(exact_execute_project_speed_failures(snapshot))
            failures.extend(self.publisher_failures())
            fatal = [
                failure
                for failure in failures
                if not failure.startswith(refreshable_prefixes)
            ]
            if fatal:
                raise RuntimeError(
                    "Watson post-gripper refresh gate failed: "
                    + "; ".join(fatal)
                )
            if not failures:
                return snapshot
            last_refresh_failures = failures
        raise RuntimeError(
            "Watson state did not refresh after the blocking gripper call: "
            + "; ".join(last_refresh_failures)
        )

    def complete_base_live_gate(
        self,
        expected_start: tuple[float, ...],
        *,
        exact_project_speed: bool,
    ) -> tuple[HealthSnapshot, dict[str, Any], dict[str, float]]:
        self.wait_for_graph()
        self.require_execution_enabled()
        listen = self.require_listen1()
        snapshot = self.require_healthy(
            stationary=True,
            exact_project_speed=exact_project_speed,
        )
        errors = live_start_errors(snapshot, expected_start)
        self.require_action_idle()
        return snapshot, listen, errors

    def complete_execute_live_gate(
        self,
        expected_start: tuple[float, ...],
    ) -> tuple[HealthSnapshot, dict[str, Any], dict[str, Any], dict[str, float]]:
        snapshot, listen, errors = self.complete_base_live_gate(
            expected_start,
            exact_project_speed=True,
        )
        tool = self.require_exact_tool()
        # Re-sample after the read-only tool queries so this is the final state
        # used by the immediately following first-cubic proof.
        snapshot = self.require_healthy(
            stationary=True,
            exact_project_speed=True,
        )
        errors = live_start_errors(snapshot, expected_start)
        # Tool queries and the sustained health proof spin ROS after the base
        # gate. Re-prove both action servers idle as the final blocking call
        # before the caller's live-q/v first-cubic proof and goal send.
        self.require_action_idle()
        return snapshot, listen, tool, errors

    def build_message(self, stage: StageSpec) -> Any:
        trajectory = build_robot_trajectory(
            stage,
            {
                "RobotTrajectory": self.ros["RobotTrajectory"],
                "JointTrajectoryPoint": self.ros["JointTrajectoryPoint"],
                "Duration": self.ros["Duration"],
            },
        )
        validate_robot_trajectory(stage, trajectory)
        return trajectory

    def _motion_monitor_failures(
        self,
        snapshot: HealthSnapshot,
        stage: StageSpec,
    ) -> list[str]:
        failures = health_failures(
            snapshot,
            max_project_speed=MAX_PROJECT_SPEED,
            # The Techman driver emits both topics from one stamped sample,
            # but rclpy may dispatch adjacent callbacks from different 15 ms
            # cycles while the arm is moving. The exact-stamp pair gate below
            # retains the original 0.005 rad integrity check without comparing
            # asynchronous positions.
            max_position_source_delta_rad=math.inf,
            require_stationary=False,
            require_auto_mode=True,
        )
        failures.extend(
            self.position_source_pairs.failures(now=time.monotonic())
        )
        failures.extend(exact_execute_project_speed_failures(snapshot))
        failures.extend(self.publisher_failures())
        failures.extend(live_stage_failures(snapshot, stage))
        if self.stop_requested:
            failures.append(f"stop requested by signal {self.stop_signal}")
        return failures

    def execute_stage(self, stage: StageSpec) -> dict[str, Any]:
        trajectory = self.build_message(stage)
        if not self.execute_client.wait_for_server(
            timeout_sec=self.args.service_timeout
        ):
            raise RuntimeError(f"{EXECUTE_ACTION} is unavailable")
        validate_air_replay_network()
        stationary, listen, tool, start_errors = self.complete_execute_live_gate(
            stage.start_positions
        )
        if self.stop_requested:
            raise RuntimeError(f"stop requested by signal {self.stop_signal}")
        snapshot = self.snapshot()
        drift = max(
            abs(
                snapshot.feedback_joint_positions[index]
                - stationary.feedback_joint_positions[index]
            )
            for index in range(len(JOINT_NAMES))
        )
        if drift > 0.001:
            raise RuntimeError(
                f"robot moved after stationary proof ({drift:.6f}rad)"
            )
        live_start_errors(snapshot, stage.start_positions)
        failures = self._motion_monitor_failures(snapshot, stage)
        if failures:
            raise RuntimeError(
                f"pre-send six-axis gate failed for {stage.stage_name}: "
                + "; ".join(failures)
            )
        first_wire_cubic_proof = validate_stage_live_first_wire_cubic(
            snapshot,
            stage,
        )

        goal = self.ros["ExecuteTrajectory"].Goal()
        goal.trajectory = trajectory
        goal.controller_names = ["tmr_arm_controller"]
        send_future = None
        goal_handle = None
        result_future = None
        send_attempted = False
        action_status_checkpoint = None
        started = time.monotonic()
        try:
            with self.command_dispatch(
                f"MoveIt_execute_{stage.sequence_index}_{stage.stage_name}"
            ):
                if self.stop_requested:
                    raise RuntimeError(
                        f"stop requested by signal {self.stop_signal}"
                    )
                action_status_checkpoint = (
                    self.capture_action_status_checkpoint()
                )
                send_attempted = True
                self.motion_command_sent = True
                send_future = self.execute_client.send_goal_async(goal)

            acceptance_deadline = time.monotonic() + GOAL_ACCEPTANCE_TIMEOUT_S
            acceptance_failures: list[str] = []
            while (
                not send_future.done()
                and time.monotonic() < acceptance_deadline
            ):
                self.rclpy.spin_once(self.node, timeout_sec=0.05)
                acceptance_failures.extend(
                    self._motion_monitor_failures(self.snapshot(), stage)
                )
            if not send_future.done():
                raise StopUnverifiedError(
                    f"goal acceptance is unknown for {stage.stage_name}; "
                    "use the physical E-stop"
                )
            goal_handle = send_future.result()
            if goal_handle is None:
                raise StopUnverifiedError(
                    f"goal acceptance returned no handle for {stage.stage_name}; "
                    "use the physical E-stop"
                )
            if not goal_handle.accepted:
                raise RuntimeError(f"MoveIt rejected {stage.stage_name}")
            result_future = goal_handle.get_result_async()
            self.active_goal_handle = goal_handle
            self.active_result_future = result_future
            if acceptance_failures:
                raise RuntimeError(
                    "health changed during goal acceptance: "
                    + "; ".join(sorted(set(acceptance_failures)))
                )

            deadline = time.monotonic() + self.args.execution_timeout
            while not result_future.done() and time.monotonic() < deadline:
                self.rclpy.spin_once(self.node, timeout_sec=0.05)
                # spin_once may have delivered the terminal result. Do not run
                # one final in-motion sample check after that terminal event.
                if result_future.done():
                    break
                failures = self._motion_monitor_failures(
                    self.snapshot(),
                    stage,
                )
                if failures:
                    raise RuntimeError(
                        f"health changed during {stage.stage_name}: "
                        + "; ".join(failures)
                    )
            if not result_future.done():
                raise RuntimeError(
                    f"execution timed out for {stage.stage_name}"
                )
            wrapped_result = result_future.result()
            if (
                wrapped_result is None
                or wrapped_result.status
                != self.ros["GoalStatus"].STATUS_SUCCEEDED
                or wrapped_result.result.error_code.val != MOVEIT_SUCCESS
            ):
                status = None if wrapped_result is None else wrapped_result.status
                code = (
                    None
                    if wrapped_result is None
                    else wrapped_result.result.error_code.val
                )
                raise RuntimeError(
                    f"MoveIt failed {stage.stage_name}: "
                    f"status={status}, error_code={code}"
                )
            stationary_failure = self.verify_stationary_after_motion()
            if stationary_failure is not None:
                raise StopUnverifiedError(
                    f"{stage.stage_name} reported success but "
                    f"{stationary_failure}; use the physical E-stop"
                )
            try:
                action_idle = self.require_goal_specific_action_idle(
                    action_status_checkpoint
                )
            except RuntimeError as exc:
                raise StopUnverifiedError(
                    f"{stage.stage_name} reported success but action-idle "
                    f"proof failed: {exc}; use the physical E-stop"
                ) from exc
            final_snapshot = self.snapshot()
            final_failures = health_failures(
                final_snapshot,
                max_project_speed=MAX_PROJECT_SPEED,
                require_stationary=True,
                require_auto_mode=True,
            )
            final_failures.extend(
                exact_execute_project_speed_failures(final_snapshot)
            )
            final_failures.extend(self.publisher_failures())
            if final_failures:
                raise RuntimeError(
                    f"post-stage health failed after {stage.stage_name}: "
                    + "; ".join(final_failures)
                )
            goal_error = max(
                max(
                    abs(
                        final_snapshot.joint_positions[index]
                        - stage.goal_positions[index]
                    )
                    for index in range(len(JOINT_NAMES))
                ),
                max(
                    abs(
                        final_snapshot.feedback_joint_positions[index]
                        - stage.goal_positions[index]
                    )
                    for index in range(len(JOINT_NAMES))
                ),
            )
            if goal_error > LIVE_GOAL_TOLERANCE_RAD:
                raise RuntimeError(
                    f"live goal mismatch after {stage.stage_name}: "
                    f"{goal_error:.6f}rad > "
                    f"{LIVE_GOAL_TOLERANCE_RAD:.6f}rad"
                )
            return {
                **stage_report(stage, "execution_passed_stationary"),
                "action_status": wrapped_result.status,
                "moveit_error_code": wrapped_result.result.error_code.val,
                "wall_duration_seconds": time.monotonic() - started,
                "live_start_errors": start_errors,
                "live_goal_error_rad": goal_error,
                "listen_gate": listen,
                "controller_tool_audit": tool,
                "stationary_to_send_drift_rad": drift,
                "live_first_wire_cubic_proof": first_wire_cubic_proof,
                "post_motion_stationary_verified": True,
                "post_motion_action_idle": action_idle,
                "final_health": asdict(final_snapshot),
                "gripper_after_stage_hook": gripper_after_stage_hook(stage),
            }
        except BaseException as exc:
            if not send_attempted:
                raise
            if goal_handle is None and send_future is not None and send_future.done():
                try:
                    goal_handle = send_future.result()
                except BaseException:
                    goal_handle = None
            if goal_handle is not None and goal_handle.accepted:
                if result_future is None:
                    try:
                        result_future = goal_handle.get_result_async()
                    except BaseException:
                        result_future = None
                cancellation_failures = self.cancel_execution(
                    goal_handle,
                    result_future,
                    action_status_checkpoint,
                )
                if cancellation_failures:
                    raise StopUnverifiedError(
                        f"{exc}; cancellation/stop proof failed: "
                        + "; ".join(cancellation_failures)
                        + "; use the physical E-stop"
                    ) from exc
                if isinstance(exc, StopUnverifiedError):
                    raise RuntimeError(
                        f"{exc}; cancellation and sustained stop were verified"
                    ) from exc
                raise
            if goal_handle is not None and not goal_handle.accepted:
                raise
            if isinstance(exc, StopUnverifiedError):
                raise
            raise StopUnverifiedError(
                f"{exc}; goal acceptance is unknown for {stage.stage_name}; "
                "use the physical E-stop"
            ) from exc
        finally:
            self.active_goal_handle = None
            self.active_result_future = None

    def cancel_execution(
        self,
        goal_handle: Any,
        result_future: Any,
        action_status_checkpoint: dict[str, Any],
    ) -> list[str]:
        failures: list[str] = []
        provisional_cancel_failures: list[str] = []
        allowed = {
            self.ros["GoalStatus"].STATUS_CANCELED,
            self.ros["GoalStatus"].STATUS_SUCCEEDED,
            self.ros["GoalStatus"].STATUS_ABORTED,
        }
        result_was_terminal = (
            result_future is not None
            and result_future.done()
            and result_future.result() is not None
            and result_future.result().status in allowed
        )
        if not result_was_terminal:
            try:
                cancel_future = goal_handle.cancel_goal_async()
                self.rclpy.spin_until_future_complete(
                    self.node,
                    cancel_future,
                    timeout_sec=3.0,
                )
                if not cancel_future.done() or cancel_future.result() is None:
                    provisional_cancel_failures.append(
                        "MoveIt did not acknowledge cancellation"
                    )
                elif not cancel_future.result().goals_canceling:
                    provisional_cancel_failures.append(
                        "MoveIt reported no active goal to cancel"
                    )
            except BaseException as exc:
                failures.append(
                    f"trajectory cancellation raised "
                    f"{type(exc).__name__}: {exc}"
                )
        terminal_proven = False
        try:
            if result_future is None:
                failures.append("no result future for terminal-state proof")
            else:
                deadline = time.monotonic() + 10.0
                while not result_future.done() and time.monotonic() < deadline:
                    self.rclpy.spin_once(self.node, timeout_sec=0.05)
                if not result_future.done() or result_future.result() is None:
                    failures.append("action did not reach a terminal state")
                else:
                    if result_future.result().status not in allowed:
                        failures.append(
                            "action terminal status was "
                            f"{result_future.result().status}"
                        )
                    else:
                        terminal_proven = True
        except BaseException as exc:
            failures.append(
                f"terminal-state proof raised {type(exc).__name__}: {exc}"
            )
        if not terminal_proven:
            failures.extend(provisional_cancel_failures)
        stationary_failure = self.verify_stationary_after_motion()
        if stationary_failure is not None:
            failures.append(stationary_failure)
        try:
            self.require_goal_specific_action_idle(action_status_checkpoint)
        except RuntimeError as exc:
            failures.append(
                f"post-cancel goal-specific action-idle proof failed: {exc}"
            )
        return failures

    def verify_stationary_after_motion(self) -> str | None:
        deadline = time.monotonic() + POST_MOTION_STATIONARY_TIMEOUT_S
        last_error = "fresh stationary feedback was not verified"
        while time.monotonic() < deadline:
            try:
                self.require_healthy(
                    stationary=True,
                    exact_project_speed=True,
                    honor_stop_request=False,
                )
                return None
            except RuntimeError as exc:
                last_error = str(exc)
        return f"sustained stationary feedback failed: {last_error}"

    def destroy(self) -> None:
        self.node.destroy_node()


def _bundle_report(bundle: ExecutionBundle) -> dict[str, Any]:
    return {
        "retimed_artifact": str(bundle.retimed_path),
        "retimed_file_sha256": bundle.retimed_file_sha256,
        "retimed_payload_sha256": bundle.retimed_payload_sha256,
        "retimed_wire_numeric_sha256": (
            bundle.retimed_wire_numeric_sha256
        ),
        "ingress_artifact": str(bundle.ingress_path),
        "ingress_file_sha256": bundle.ingress_file_sha256,
        "ingress_numeric_sha256": bundle.ingress_numeric_sha256,
        "stage_count": len(bundle.stages),
        "gripper_policy": dict(GRIPPER_POLICY),
    }


def main() -> int:
    reset_hil_event_stream()
    args = build_parser().parse_args()
    signal_gate = None
    if not args.offline_validate:
        # Bash starts an asynchronous child with SIGINT ignored. Reset and
        # block stop signals before artifact loading or report reservation, so
        # a GUI Stop cannot be discarded during runner startup.
        signal_gate = ProcessSignalGate()
    try:
        if signal_gate is not None and signal_gate.poll() is not None:
            raise InterruptedError(
                "stop requested before air-replay validation"
            )
        validate_cli(args)
        if GRIPPER_EXECUTION_TOKEN != CONTROL_CONFIRMATION:
            raise RuntimeError("arm runner and guarded 2FG7 token disagree")
        bundle = load_execution_bundle(
            args.retimed_artifact,
            args.ingress_artifact,
        )
        execution_stages = select_execution_stages(
            bundle,
            resume_at_reviewed_ready=args.resume_at_reviewed_ready,
        )
        if signal_gate is not None and signal_gate.poll() is not None:
            raise InterruptedError(
                "stop requested during air-replay validation"
            )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        emit_hil_event(
            args.hil_events,
            "run_failed",
            status="startup_validation_failed",
            error=str(exc),
            physical_estop_required=False,
        )
        return 2

    if args.offline_validate:
        print("Seven-pin air-replay offline validation: PASS")
        print(f"Exact stages: {len(bundle.stages)} (ingress plus 49)")
        print(f"Retimed artifact SHA-256: {bundle.retimed_file_sha256}")
        print(
            "Retimed wire-numeric SHA-256: "
            f"{bundle.retimed_wire_numeric_sha256}"
        )
        print(f"Ingress artifact SHA-256: {bundle.ingress_file_sha256}")
        print("ROS graph inspected: no")
        print("Network contacted: no")
        print("Arm or gripper transport created: no")
        emit_hil_event(
            args.hil_events,
            "run_completed",
            mode="offline-validate",
            status="offline_validation_passed",
            motion_commanded=False,
        )
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = args.report or (
        ARENA_DIR
        / "outputs/watson_guarded_demo/"
        f"{timestamp}_seven_pin_air_{args.mode}.json"
    )
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "status": "artifacts_validated",
        "namespace": "/" + args.namespace.strip("/"),
        "robot_ip": ROBOT_IP,
        "robot_interface": ROBOT_INTERFACE,
        "robot_source_ip": ROBOT_SOURCE_IP,
        "robot_mac": ROBOT_MAC,
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
        "ros_automatic_discovery_range": os.environ.get(
            "ROS_AUTOMATIC_DISCOVERY_RANGE"
        ),
        "runner_sha256": SCRIPT_SHA256,
        "bundle": _bundle_report(bundle),
        "execute_action": EXECUTE_ACTION,
        "controller_names": ["tmr_arm_controller"],
        "direct_controller_action_client_created": False,
        "gripper_action_clients_created": 0,
        "gripper_transport_created": False,
        "gripper_commands": [],
        "gripper_recovery_stops": [],
        "motion_commanded": False,
        "start_mode": (
            "fresh_reviewed_ready_resume"
            if args.resume_at_reviewed_ready
            else "tool_aware_ready_ingress"
        ),
        "execution_start_sequence_index": execution_stages[0].sequence_index,
        "selected_stage_count": len(execution_stages),
        "reviewed_ready_joint_positions_rad": (
            list(READY_JOINT_POSITIONS_RAD)
            if args.resume_at_reviewed_ready
            else None
        ),
        "stage_reports": [
            stage_report(
                stage,
                (
                    "resume_ingress_not_selected_pending_fresh_ready_gate"
                    if args.resume_at_reviewed_ready
                    and stage.sequence_index == 0
                    else "artifact_validated_pending_live_gate"
                ),
            )
            for stage in bundle.stages
        ],
    }
    try:
        report_reservation = reserve_private_report(report_path)
    except (OSError, ValueError, RuntimeError) as exc:
        print(
            f"ERROR: cannot reserve private report before live contact: {exc}",
            file=sys.stderr,
        )
        emit_hil_event(
            args.hil_events,
            "run_failed",
            status="report_reservation_failed",
            error=str(exc),
            physical_estop_required=False,
        )
        return 2

    ros = None
    runtime = None
    execute_lock = None
    gripper_transport = None
    gripper_may_be_moving = False
    try:
        validate_air_replay_network()
        if args.mode == "execute":
            execute_lock = acquire_execute_lock(EXECUTE_LOCK_PATH)
        ros = load_ros()
        ros["rclpy"].init(
            args=None,
            signal_handler_options=ros["SignalHandlerOptions"].NO,
        )
        runtime = AirReplayNode(args, ros, signal_gate=signal_gate)
        report["tool_selection"] = dict(runtime.tool_selection_evidence)

        initial = runtime.spin_until_state()
        tool_selection = None
        expected_execution_start = execution_stages[0].start_positions
        if args.mode == "execute":
            runtime.complete_base_live_gate(
                expected_execution_start,
                exact_project_speed=True,
            )
            tool_selection = runtime.select_exact_tool()
            report["tool_selection"] = dict(runtime.tool_selection_evidence)
            snapshot, listen, tool, start_errors = (
                runtime.complete_execute_live_gate(
                    expected_execution_start
                )
            )
        else:
            snapshot, listen, start_errors = runtime.complete_base_live_gate(
                expected_execution_start,
                exact_project_speed=False,
            )
            tool = runtime.read_tool_audit()
        report["initial_health"] = asdict(initial)
        report["stable_health"] = asdict(snapshot)
        report["listen_gate"] = listen
        report["controller_tool_audit"] = tool
        report["controller_tool_audit_failures"] = exact_tool_audit_failures(
            tool
        )
        report["tool_selection"] = tool_selection
        report["live_execution_start_errors"] = start_errors
        if not args.resume_at_reviewed_ready:
            report["live_ingress_start_errors"] = start_errors
        skipped_stage_reports = []
        if args.resume_at_reviewed_ready:
            skipped_stage_reports = [
                stage_report(
                    bundle.stages[0],
                    "skipped_after_fresh_reviewed_ready_gate",
                )
            ]
            report["resume_gate"] = {
                "status": "passed",
                "basis": (
                    "fresh stationary dual-feed READY state, exact Listen1, "
                    "speed 50, named tool, and idle action servers"
                ),
                "ingress_goal_sent_in_this_run": False,
                "live_start_errors": start_errors,
            }
        emit_hil_event(
            args.hil_events,
            "run_started",
            mode=args.mode,
            start_mode=report["start_mode"],
            selected_stage_count=len(execution_stages),
        )

        if args.mode == "check":
            tool_suffix = (
                "exact_tool"
                if not report["controller_tool_audit_failures"]
                else "tool_mismatch_reported"
            )
            report["status"] = (
                f"live_check_passed_read_only_{tool_suffix}_no_motion"
            )
            report["stage_reports"] = [
                stage_report(stage, "validated_not_built_or_sent")
                for stage in bundle.stages
            ]
            print("Seven-pin air-replay live check: PASS")
            if report["controller_tool_audit_failures"]:
                print(
                    "Read-only tool mismatch: "
                    + "; ".join(report["controller_tool_audit_failures"])
                )
            print("RobotTrajectory messages built: 0")
            print("Action goals sent: 0")
            write_private_report(
                report_path,
                report,
                reservation=report_reservation,
            )
            emit_hil_event(
                args.hil_events,
                "run_completed",
                mode=args.mode,
                status=report["status"],
                motion_commanded=False,
            )
            return 0

        built = []
        for stage in execution_stages:
            trajectory = runtime.build_message(stage)
            built.append(
                {
                    **stage_report(stage, "message_built_exact_not_sent"),
                    "message_joint_names": list(
                        trajectory.joint_trajectory.joint_names
                    ),
                    "message_acceleration_values_present": False,
                    "message_effort_values_present": False,
                }
            )
        report["stage_reports"] = skipped_stage_reports + built
        report["controller_messages_built"] = len(built)
        if args.mode == "dry-run":
            tool_suffix = (
                "exact_tool"
                if not report["controller_tool_audit_failures"]
                else "tool_mismatch_reported"
            )
            report["status"] = (
                f"live_dry_run_passed_read_only_{tool_suffix}_no_motion"
            )
            print("Seven-pin air-replay live dry-run: PASS")
            if report["controller_tool_audit_failures"]:
                print(
                    "Read-only tool mismatch: "
                    + "; ".join(report["controller_tool_audit_failures"])
                )
            print(f"Exact RobotTrajectory messages built: {len(built)}")
            print("Action goals sent: 0")
            write_private_report(
                report_path,
                report,
                reservation=report_reservation,
            )
            emit_hil_event(
                args.hil_events,
                "run_completed",
                mode=args.mode,
                status=report["status"],
                motion_commanded=False,
            )
            return 0

        print(
            "Execution ARMED: "
            + (
                "49 air-replay stages from freshly verified READY."
                if args.resume_at_reviewed_ready
                else "tool-aware ingress plus 49 air-replay arm stages."
            )
        )
        print(
            "2FG7 profile: open 39mm, close 1mm, 20N, 10%."
        )
        gripper_transport = FixedComputeBoxTransport()
        report["gripper_transport_created"] = True
        gripper_may_be_moving = True
        initial_context = (
            "before_reviewed_ready_resume"
            if args.resume_at_reviewed_ready
            else "before_tool_aware_ready_ingress"
        )
        emit_hil_event(
            args.hil_events,
            "gripper_started",
            action="open",
            context=initial_context,
            specimen_id=None,
            sequence_index=None,
        )
        initial_open = guarded_gripper_transition(
            runtime,
            gripper_transport,
            GripperAction.OPEN,
            confirmation=args.gripper_token,
        )
        gripper_may_be_moving = gripper_command_may_be_moving(initial_open)
        initial_open["context"] = initial_context
        report["gripper_commands"].append(initial_open)
        emit_hil_event(
            args.hil_events,
            "gripper_completed",
            action="open",
            completed=bool(initial_open["completed"]),
            context=initial_context,
            specimen_id=None,
            sequence_index=None,
        )
        if initial_open["completed"] is not True:
            report["stage_reports"] = skipped_stage_reports + [
                stage_report(
                    stage,
                    "not_attempted_after_initial_gripper_open_failure",
                )
                for stage in execution_stages
            ]
            raise_gripper_transition_failure(
                "initial guarded 2FG7 OPEN failed; ingress will not be sent",
                initial_open,
            )

        execution_reports: list[dict[str, Any]] = []
        for stage in execution_stages:
            emit_hil_event(
                args.hil_events,
                "stage_started",
                sequence_index=stage.sequence_index,
                stage_name=stage.stage_name,
                specimen_id=stage.specimen_id,
            )
            print(
                f"Executing stage {stage.sequence_index + 1}/"
                f"{len(bundle.stages)}: {stage.stage_name}",
                flush=True,
            )
            arm_stage_recorded = False
            try:
                executed_stage = runtime.execute_stage(stage)
                execution_reports.append(executed_stage)
                arm_stage_recorded = True
                emit_hil_event(
                    args.hil_events,
                    "stage_completed",
                    sequence_index=stage.sequence_index,
                    stage_name=stage.stage_name,
                    specimen_id=stage.specimen_id,
                    status=executed_stage["status"],
                )
                hook = gripper_after_stage_hook(stage)
                if hook is not None:
                    gripper_may_be_moving = True
                    hook_context = (
                        f"after_stage_{stage.sequence_index}_"
                        f"{stage.stage_name}"
                    )
                    emit_hil_event(
                        args.hil_events,
                        "gripper_started",
                        action=hook["action"],
                        context=hook_context,
                        specimen_id=stage.specimen_id,
                        sequence_index=stage.sequence_index,
                    )
                    transition = guarded_gripper_transition(
                        runtime,
                        gripper_transport,
                        GripperAction(hook["action"]),
                        confirmation=args.gripper_token,
                    )
                    gripper_may_be_moving = gripper_command_may_be_moving(
                        transition
                    )
                    transition["context"] = hook_context
                    report["gripper_commands"].append(transition)
                    emit_hil_event(
                        args.hil_events,
                        "gripper_completed",
                        action=hook["action"],
                        completed=bool(transition["completed"]),
                        context=hook_context,
                        specimen_id=stage.specimen_id,
                        sequence_index=stage.sequence_index,
                    )
                    executed_stage["gripper_after_stage_hook"] = {
                        **hook,
                        "executed": transition["completed"],
                        "guarded_command_index": (
                            len(report["gripper_commands"]) - 1
                        ),
                    }
                    if transition["completed"] is not True:
                        executed_stage["post_stage_failure"] = {
                            "component": "guarded_2fg7",
                            "action": hook["action"],
                            "error": "; ".join(transition["failures"]),
                            "next_arm_stage_sent": False,
                        }
                        raise_gripper_transition_failure(
                            f"guarded 2FG7 {hook['action'].upper()} failed "
                            f"after {stage.stage_name}; next arm stage will "
                            "not be sent",
                            transition,
                        )
            except BaseException as exc:
                emit_hil_stage_failure(
                    args.hil_events,
                    stage,
                    exc,
                    arm_stage_completed=arm_stage_recorded,
                )
                if arm_stage_recorded:
                    execution_reports[-1]["sequence_aborted_after_stage"] = True
                    execution_reports[-1]["sequence_error"] = str(exc)
                else:
                    execution_reports.append(
                        {
                            **stage_report(stage, "failed_closed_before_pass"),
                            "error": str(exc),
                        }
                    )
                report["stage_reports"] = (
                    skipped_stage_reports
                    + execution_reports
                    + [
                    stage_report(pending, "not_attempted_after_failure")
                    for pending in execution_stages[len(execution_reports) :]
                    ]
                )
                raise
        report["stage_reports"] = skipped_stage_reports + execution_reports
        report["motion_commanded"] = runtime.motion_command_sent
        observed_gripper_actions = [
            item["action"] for item in report["gripper_commands"]
        ]
        expected_gripper_actions = ["open"] + [
            action
            for _ in range(7)
            for action in ("close", "open")
        ]
        if observed_gripper_actions != expected_gripper_actions:
            raise RuntimeError(
                "guarded gripper sequence changed: "
                f"{observed_gripper_actions!r}"
            )
        report["status"] = (
            "execution_passed_reviewed_ready_resume_49_arm_stages_and_"
            "guarded_gripper_actions"
            if args.resume_at_reviewed_ready
            else "execution_passed_all_arm_stages_and_guarded_gripper_actions"
        )
        print("Seven-pin arm and guarded 2FG7 air replay: PASS")
        print(f"Guarded gripper actions: {len(report['gripper_commands'])}")
        write_private_report(
            report_path,
            report,
            reservation=report_reservation,
        )
        emit_hil_event(
            args.hil_events,
            "run_completed",
            mode=args.mode,
            status=report["status"],
            motion_commanded=runtime.motion_command_sent,
        )
        return 0
    except StopUnverifiedError as exc:
        if gripper_transport is not None and gripper_may_be_moving:
            stop_report = best_effort_gripper_stop(
                gripper_transport,
                confirmation=args.gripper_token,
            )
            stop_report["reason"] = "outer_stop_unverified"
            report["gripper_recovery_stops"].append(stop_report)
            gripper_may_be_moving = not gripper_stop_verified(stop_report)
        if runtime is not None:
            report["motion_commanded"] = runtime.motion_command_sent
            report["tool_selection"] = dict(
                runtime.tool_selection_evidence
            )
        report["status"] = "stop_unverified_use_physical_estop"
        report["error"] = str(exc)
        report["gripper_stop_unverified"] = gripper_may_be_moving
        print(f"EMERGENCY: {exc}", file=sys.stderr, flush=True)
        if gripper_may_be_moving:
            print(
                "EMERGENCY: 2FG7 STOP is unverified; use the physical E-stop.",
                file=sys.stderr,
                flush=True,
            )
        write_report_best_effort(
            report_path,
            report,
            reservation=report_reservation,
        )
        emit_hil_event(
            args.hil_events,
            "run_failed",
            status=report["status"],
            error=str(exc),
            physical_estop_required=True,
        )
        return 3
    except (OSError, ValueError, RuntimeError) as exc:
        if gripper_transport is not None and gripper_may_be_moving:
            stop_report = best_effort_gripper_stop(
                gripper_transport,
                confirmation=args.gripper_token,
            )
            stop_report["reason"] = "outer_failure"
            report["gripper_recovery_stops"].append(stop_report)
            gripper_may_be_moving = not gripper_stop_verified(stop_report)
        if runtime is not None:
            report["motion_commanded"] = runtime.motion_command_sent
            report["tool_selection"] = dict(
                runtime.tool_selection_evidence
            )
        report["status"] = (
            "stop_unverified_use_physical_estop"
            if gripper_may_be_moving
            else "failed_closed"
        )
        report["error"] = str(exc)
        report["gripper_stop_unverified"] = gripper_may_be_moving
        prefix = "EMERGENCY" if gripper_may_be_moving else "ERROR"
        print(f"{prefix}: {exc}", file=sys.stderr, flush=True)
        if gripper_may_be_moving:
            print(
                "EMERGENCY: 2FG7 STOP is unverified; use the physical E-stop.",
                file=sys.stderr,
                flush=True,
            )
        write_report_best_effort(
            report_path,
            report,
            reservation=report_reservation,
        )
        emit_hil_event(
            args.hil_events,
            "run_failed",
            status=report["status"],
            error=str(exc),
            physical_estop_required=bool(gripper_may_be_moving),
        )
        return 3 if gripper_may_be_moving else 1
    except BaseException as exc:
        if gripper_transport is not None and gripper_may_be_moving:
            stop_report = best_effort_gripper_stop(
                gripper_transport,
                confirmation=args.gripper_token,
            )
            stop_report["reason"] = "outer_unexpected_exception_or_signal"
            report["gripper_recovery_stops"].append(stop_report)
            gripper_may_be_moving = not gripper_stop_verified(stop_report)
        if runtime is not None:
            report["motion_commanded"] = runtime.motion_command_sent
            report["tool_selection"] = dict(
                runtime.tool_selection_evidence
            )
        report["status"] = (
            "stop_unverified_use_physical_estop"
            if gripper_may_be_moving
            else "unexpected_failure_stop_verified"
        )
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["gripper_stop_unverified"] = gripper_may_be_moving
        if gripper_may_be_moving:
            print(
                "EMERGENCY: 2FG7 STOP is unverified; use the physical E-stop.",
                file=sys.stderr,
                flush=True,
            )
        write_report_best_effort(
            report_path,
            report,
            reservation=report_reservation,
        )
        emit_hil_event(
            args.hil_events,
            "run_failed",
            status=report["status"],
            error=report["error"],
            physical_estop_required=bool(gripper_may_be_moving),
        )
        if gripper_may_be_moving:
            return 3
        raise
    finally:
        if runtime is not None:
            runtime.destroy()
        if ros is not None and ros["rclpy"].ok():
            ros["rclpy"].shutdown()
        if execute_lock is not None:
            execute_lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
