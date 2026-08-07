"""Pure orchestration state for the Isaac/Watson HIL presentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


HIL_EVENT_PREFIX = "WATSON_HIL_EVENT "
HIL_EVENT_NAMES = frozenset(
    {
        "stage_started",
        "stage_completed",
        "stage_failed",
        "gripper_started",
        "gripper_completed",
        "run_started",
        "run_completed",
        "run_failed",
    }
)


class HilMode(str, Enum):
    PREVIEW = "preview"
    DRY_RUN = "dry-run"
    EXECUTE = "execute"


class HilState(str, Enum):
    DISARMED = "disarmed"
    ARMED = "armed"
    LAUNCH_REQUESTED = "launch_requested"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


RUNNER_ENVIRONMENT_REMOVE = frozenset(
    {
        "AMENT_PREFIX_PATH",
        "COLCON_PREFIX_PATH",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "ROS_DISTRO",
        "ROS_PYTHON_VERSION",
        "ROS_VERSION",
    }
)


def validate_hil_event(payload: Any) -> dict[str, Any]:
    """Validate one versioned, non-authoritative runner status event."""

    if not isinstance(payload, dict):
        raise ValueError("HIL event must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("HIL event schema_version must be 1")
    event_sequence = payload.get("event_sequence")
    if not isinstance(event_sequence, int) or event_sequence < 1:
        raise ValueError("HIL event_sequence must be a positive integer")
    event = payload.get("event")
    if event not in HIL_EVENT_NAMES:
        raise ValueError(f"unknown HIL event: {event!r}")
    timestamp = payload.get("timestamp_utc")
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError("HIL event timestamp_utc is missing")
    if event in {"stage_started", "stage_completed", "stage_failed"}:
        if not isinstance(payload.get("sequence_index"), int):
            raise ValueError("stage HIL event sequence_index must be an integer")
        if not isinstance(payload.get("stage_name"), str):
            raise ValueError("stage HIL event stage_name must be a string")
        specimen_id = payload.get("specimen_id")
        if specimen_id is not None and (
            not isinstance(specimen_id, int) or not 1 <= specimen_id <= 7
        ):
            raise ValueError("stage HIL event specimen_id must be 1..7 or null")
        if event == "stage_failed" and not isinstance(payload.get("error"), str):
            raise ValueError("failed stage HIL event error must be a string")
    if event in {"gripper_started", "gripper_completed"}:
        if payload.get("action") not in {"open", "close"}:
            raise ValueError("gripper HIL action must be open or close")
        if not isinstance(payload.get("context"), str):
            raise ValueError("gripper HIL context must be a string")
        specimen_id = payload.get("specimen_id")
        if specimen_id is not None and (
            not isinstance(specimen_id, int) or not 1 <= specimen_id <= 7
        ):
            raise ValueError("gripper HIL specimen_id must be 1..7 or null")
        if event == "gripper_completed" and not isinstance(
            payload.get("completed"), bool
        ):
            raise ValueError("completed gripper HIL event needs a Boolean result")
    if event in {"run_started", "run_completed"}:
        if not isinstance(payload.get("mode"), str):
            raise ValueError("run HIL event mode must be a string")
    if event == "run_completed":
        if not isinstance(payload.get("status"), str):
            raise ValueError("completed HIL event status must be a string")
    if event == "run_failed":
        if not isinstance(payload.get("status"), str):
            raise ValueError("failed HIL event status must be a string")
        if not isinstance(payload.get("error"), str):
            raise ValueError("failed HIL event error must be a string")
        if not isinstance(payload.get("physical_estop_required"), bool):
            raise ValueError(
                "failed HIL event physical_estop_required must be Boolean"
            )
    return dict(payload)


def parse_hil_event_line(line: str) -> dict[str, Any] | None:
    """Parse one prefixed runner line; ordinary human output returns ``None``."""

    if not isinstance(line, str):
        raise TypeError("runner output line must be text")
    stripped = line.rstrip("\r\n")
    if not stripped.startswith(HIL_EVENT_PREFIX):
        return None
    encoded = stripped[len(HIL_EVENT_PREFIX) :]
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid HIL event JSON: {exc}") from exc
    return validate_hil_event(payload)


@dataclass
class HilCoordinator:
    """One-shot GUI arming and Play/Stop state machine."""

    mode: HilMode
    state: HilState = HilState.DISARMED
    launch_consumed: bool = False
    stop_signal_sent: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)
    failure: str | None = None
    physical_estop_required: bool = False
    last_event_sequence: int = 0

    def arm(self) -> None:
        if self.launch_consumed:
            raise RuntimeError("this HIL window already consumed its one run")
        if self.state is not HilState.DISARMED:
            raise RuntimeError(f"cannot arm HIL from {self.state.value}")
        self.state = HilState.ARMED

    def disarm(self) -> None:
        if self.state is not HilState.ARMED:
            raise RuntimeError(f"cannot disarm HIL from {self.state.value}")
        self.state = HilState.DISARMED

    def on_play(self) -> str:
        if self.state is HilState.ARMED and not self.launch_consumed:
            self.launch_consumed = True
            self.state = HilState.LAUNCH_REQUESTED
            return "launch"
        return "ignore"

    def runner_started(self) -> None:
        if self.state is not HilState.LAUNCH_REQUESTED:
            raise RuntimeError(f"runner cannot start from {self.state.value}")
        self.state = HilState.RUNNING

    def on_stop(self) -> str:
        if self.state in {HilState.LAUNCH_REQUESTED, HilState.RUNNING}:
            self.state = HilState.STOPPING
            if not self.stop_signal_sent:
                self.stop_signal_sent = True
                return "signal_stop"
        return "ignore"

    def cancel_before_spawn(self) -> None:
        if self.state is not HilState.LAUNCH_REQUESTED:
            raise RuntimeError(
                f"pre-spawn launch cannot be cancelled from {self.state.value}"
            )
        self.state = HilState.STOPPED

    def accept_event(self, payload: Any) -> dict[str, Any]:
        event = validate_hil_event(payload)
        expected_sequence = self.last_event_sequence + 1
        if event["event_sequence"] != expected_sequence:
            raise ValueError(
                "HIL event sequence changed: "
                f"expected {expected_sequence}, got {event['event_sequence']}"
            )
        self.last_event_sequence = event["event_sequence"]
        self.events.append(event)
        if event["event"] == "run_failed":
            self.failure = event["error"]
            self.physical_estop_required = event["physical_estop_required"]
        return event

    def runner_exited(self, return_code: int) -> None:
        if self.state not in {
            HilState.RUNNING,
            HilState.STOPPING,
            HilState.LAUNCH_REQUESTED,
        }:
            raise RuntimeError(f"runner cannot exit from {self.state.value}")
        completed = bool(self.events) and self.events[-1]["event"] == "run_completed"
        final_failed = bool(self.events) and self.events[-1]["event"] == "run_failed"
        failed = any(event["event"] == "run_failed" for event in self.events)
        if return_code == 0 and completed and not failed:
            self.state = HilState.COMPLETED
        elif (
            self.stop_signal_sent
            and final_failed
            and not self.physical_estop_required
        ):
            self.state = HilState.STOPPED
        else:
            self.state = HilState.FAILED
            if self.failure is None:
                self.failure = (
                    f"runner exited with status {return_code} without a verified "
                    "final lifecycle event"
                )


def safe_report_name(value: str) -> str:
    """Restrict GUI-selected report suffixes to simple local filenames."""

    if re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None:
        raise ValueError("HIL report name contains unsupported characters")
    return value


def sanitized_runner_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a native-ROS child environment without Isaac Python overlays."""

    values = dict(os.environ if environment is None else environment)
    for name in RUNNER_ENVIRONMENT_REMOVE:
        values.pop(name, None)
    values.update(
        {
            "ROS_DOMAIN_ID": "219",
            "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
            "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
        }
    )
    return values


def build_runner_command(
    *,
    wrapper: Path,
    mode: HilMode,
    report: Path,
    arm_token: str = "",
    gripper_token: str = "",
    confirm_cell_clear: bool = False,
) -> list[str]:
    """Build the only permitted HIL child command."""

    if mode is HilMode.PREVIEW:
        raise ValueError("preview mode must not launch the Watson wrapper")
    wrapper = wrapper.expanduser().resolve()
    report = report.expanduser().resolve()
    if not wrapper.is_file():
        raise FileNotFoundError(f"Watson wrapper is missing: {wrapper}")
    command = [
        str(wrapper),
        "--mode",
        mode.value,
        "--hil-events",
        "--report",
        str(report),
    ]
    if mode is HilMode.EXECUTE:
        if not arm_token or not gripper_token or not confirm_cell_clear:
            raise ValueError(
                "execute requires both exact tokens and confirmed cell clearance"
            )
        command.extend(
            [
                "--arm-token",
                arm_token,
                "--gripper-token",
                gripper_token,
                "--confirm-cell-clear",
            ]
        )
    elif arm_token or gripper_token or confirm_cell_clear:
        raise ValueError("dry-run must not accept execute authorization")
    return command
