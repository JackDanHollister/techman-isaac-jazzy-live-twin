"""Guarded direct control for Watson's OnRobot 2FG7 air replay.

The fixed command profile mirrors the direct Compute Box API settings already
used on Watson:

* device 0, 2FG7 type 17 / product 192, inward-facing fingers;
* 39 mm open;
* 1 mm close at 20 N and 10 percent speed; and
* the ``grip_external`` and ``stop`` HTTP GET routes.

The 1 mm close was physically exercised on Watson and is deliberate for this
empty air replay because the Isaac presentation closes the inward jaws through
the same full 39-to-1 mm gesture.

Dry-run is the default and returns before constructing a live transport.
Execution requires an exact immutable confirmation token, passive fresh state
before and after every command, and a fixed allowlisted URL.  This module does
not import ROS and has no Watson arm-control path.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Callable, ContextManager, Protocol
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from pin_axis_3d_sim.onrobot_state import (
    COMPUTE_BOX_ORIGIN,
    EXPECTED_DEVICE_TYPE,
    EXPECTED_EXTERNAL_RANGE_MM,
    EXPECTED_INTERNAL_RANGE_MM,
    EXPECTED_PRODUCT_CODE,
    LIVE_CONFIRMATION,
    LiveCapture,
    QualificationError,
    REPORT_DIGEST_FIELD,
    canonical_digest,
    capture_live_read_only,
    normalise_2fg7_state,
    write_private_report,
)


CONTROL_CONFIRMATION = "EXECUTE_WATSON_2FG7_AIR_REPLAY"
FIXED_DEVICE_ID = 0
OPEN_EXTERNAL_WIDTH_MM = 39.0
AIR_REPLAY_CLOSE_EXTERNAL_WIDTH_MM = 1.0
AIR_REPLAY_CLOSE_CONTACT_MAX_EXTERNAL_MM = 2.0
WIDTH_TOLERANCE_MM = 0.5
RANGE_TOLERANCE_MM = 0.25
INTERNAL_EXTERNAL_OFFSET_MM = 10.0
FORCE_N = 20
SPEED_PERCENT = 10
MAX_COMMAND_RESPONSE_BYTES = 4096


class GripperAction(str, Enum):
    """The only actuator operations exposed by the guarded helper."""

    OPEN = "open"
    CLOSE = "close"
    STOP = "stop"


@dataclass(frozen=True)
class CommandSpec:
    """One immutable allowlisted Compute Box GET command."""

    action: GripperAction
    url: str
    target_external_width_mm: float | None
    force_n: int | None
    speed_percent: int | None


_OPEN_URL = (
    f"{COMPUTE_BOX_ORIGIN}/api/dc/twofg/grip_external/"
    f"{FIXED_DEVICE_ID}/{int(OPEN_EXTERNAL_WIDTH_MM)}/{FORCE_N}/{SPEED_PERCENT}"
)
_CLOSE_URL = (
    f"{COMPUTE_BOX_ORIGIN}/api/dc/twofg/grip_external/"
    f"{FIXED_DEVICE_ID}/{int(AIR_REPLAY_CLOSE_EXTERNAL_WIDTH_MM)}/"
    f"{FORCE_N}/{SPEED_PERCENT}"
)
_STOP_URL = f"{COMPUTE_BOX_ORIGIN}/api/dc/twofg/stop/{FIXED_DEVICE_ID}"

_COMMAND_SPECS = {
    GripperAction.OPEN: CommandSpec(
        action=GripperAction.OPEN,
        url=_OPEN_URL,
        target_external_width_mm=OPEN_EXTERNAL_WIDTH_MM,
        force_n=FORCE_N,
        speed_percent=SPEED_PERCENT,
    ),
    GripperAction.CLOSE: CommandSpec(
        action=GripperAction.CLOSE,
        url=_CLOSE_URL,
        target_external_width_mm=AIR_REPLAY_CLOSE_EXTERNAL_WIDTH_MM,
        force_n=FORCE_N,
        speed_percent=SPEED_PERCENT,
    ),
    GripperAction.STOP: CommandSpec(
        action=GripperAction.STOP,
        url=_STOP_URL,
        target_external_width_mm=None,
        force_n=None,
        speed_percent=None,
    ),
}
ALLOWED_COMMAND_URLS = frozenset(spec.url for spec in _COMMAND_SPECS.values())


@dataclass(frozen=True)
class CommandResponse:
    """Minimal response evidence returned by an injected transport."""

    status_code: int
    body: str
    final_url: str


class OnRobotControlTransport(Protocol):
    """The small transport surface used by the guarded command workflow."""

    def passive_state(self, *, timeout_seconds: float) -> LiveCapture:
        """Capture one passive 2FG7 state event."""

    def command_get(
        self,
        *,
        url: str,
        timeout_seconds: float,
    ) -> CommandResponse:
        """Issue one fixed GET command."""


@dataclass(frozen=True)
class ControlTiming:
    """Bounded timing profile; shorter values may be injected for tests."""

    state_capture_timeout_seconds: float = 3.0
    command_timeout_seconds: float = 5.0
    poll_timeout_seconds: float = 20.0
    poll_interval_seconds: float = 0.1
    max_state_age_seconds: float = 5.0

    def __post_init__(self) -> None:
        values = {
            "state_capture_timeout_seconds": self.state_capture_timeout_seconds,
            "command_timeout_seconds": self.command_timeout_seconds,
            "poll_timeout_seconds": self.poll_timeout_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
            "max_state_age_seconds": self.max_state_age_seconds,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.state_capture_timeout_seconds > 5.0:
            raise ValueError("state_capture_timeout_seconds cannot exceed 5")
        if self.command_timeout_seconds > 10.0:
            raise ValueError("command_timeout_seconds cannot exceed 10")
        if self.poll_timeout_seconds > 30.0:
            raise ValueError("poll_timeout_seconds cannot exceed 30")
        if self.poll_interval_seconds > 1.0:
            raise ValueError("poll_interval_seconds cannot exceed 1")
        if self.max_state_age_seconds > 10.0:
            raise ValueError("max_state_age_seconds cannot exceed 10")


DEFAULT_TIMING = ControlTiming()


class _NoRedirectHandler(HTTPRedirectHandler):
    """Prevent an allowlisted request from being redirected elsewhere."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def _validate_command_url(url: str) -> None:
    if url not in ALLOWED_COMMAND_URLS:
        raise QualificationError(
            "gripper command URL is not one of the fixed Watson 2FG7 routes"
        )


def command_spec(action: GripperAction | str) -> CommandSpec:
    """Return one of the three immutable command specifications."""

    try:
        parsed = action if isinstance(action, GripperAction) else GripperAction(action)
    except (TypeError, ValueError) as exc:
        raise ValueError("action must be exactly open, close, or stop") from exc
    return _COMMAND_SPECS[parsed]


class FixedComputeBoxTransport:
    """Production HTTP transport pinned to the fixed Compute Box routes.

    Constructing this object opens no connection.  Its opener is injectable so
    the complete command behavior can be tested without network access.
    """

    def __init__(self, *, opener: Callable[..., Any] | None = None) -> None:
        # Never allow HTTP_PROXY/ALL_PROXY to carry a robot command away from
        # the already-validated direct robot NIC route.
        self._opener = opener or build_opener(
            ProxyHandler({}),
            _NoRedirectHandler(),
        ).open

    def passive_state(self, *, timeout_seconds: float) -> LiveCapture:
        return capture_live_read_only(
            confirmation=LIVE_CONFIRMATION,
            timeout_seconds=timeout_seconds,
            opener=self._opener,
        )

    def command_get(
        self,
        *,
        url: str,
        timeout_seconds: float,
    ) -> CommandResponse:
        _validate_command_url(url)
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive and finite")
        request = Request(
            url,
            headers={
                "Accept": "text/plain",
                "Cache-Control": "no-store",
            },
            method="GET",
        )
        response = self._opener(request, timeout=timeout_seconds)
        try:
            if hasattr(response, "__enter__"):
                with response:
                    return self._read_command_response(response, url)
            return self._read_command_response(response, url)
        finally:
            if not hasattr(response, "__enter__"):
                close = getattr(response, "close", None)
                if close is not None:
                    close()

    @staticmethod
    def _read_command_response(response: Any, requested_url: str) -> CommandResponse:
        final_url = (
            response.geturl()
            if callable(getattr(response, "geturl", None))
            else requested_url
        )
        _validate_command_url(final_url)
        if final_url != requested_url:
            raise QualificationError("Compute Box command response changed URL")
        body_bytes = response.read(MAX_COMMAND_RESPONSE_BYTES + 1)
        if len(body_bytes) > MAX_COMMAND_RESPONSE_BYTES:
            raise QualificationError("Compute Box command response is too large")
        try:
            body = body_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise QualificationError(
                "Compute Box command response is not UTF-8"
            ) from exc
        status = getattr(response, "status", None)
        if status is None and callable(getattr(response, "getcode", None)):
            status = response.getcode()
        if isinstance(status, bool) or not isinstance(status, int):
            raise QualificationError("Compute Box response has no HTTP status")
        return CommandResponse(
            status_code=status,
            body=body,
            final_url=final_url,
        )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_received_at(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise QualificationError("passive state has no timezone-aware receive time")
    return value.astimezone(timezone.utc)


def _state_record(
    capture: LiveCapture,
    *,
    now: datetime,
    max_age_seconds: float,
) -> tuple[dict[str, Any], list[str]]:
    captured_at = _parse_received_at(capture.received_at_utc)
    current = _parse_received_at(now)
    age_seconds = (current - captured_at).total_seconds()
    state = normalise_2fg7_state(capture.payload)
    failures: list[str] = []
    if age_seconds < -1.0:
        failures.append(
            f"passive state timestamp is {-age_seconds:.3f}s in the future"
        )
    elif age_seconds > max_age_seconds:
        failures.append(
            f"passive state is stale ({age_seconds:.3f}s > "
            f"{max_age_seconds:.3f}s)"
        )
    if state["device_id"] != FIXED_DEVICE_ID:
        failures.append(
            f"device_id is {state['device_id']}, expected {FIXED_DEVICE_ID}"
        )
    if state["device_type"] != EXPECTED_DEVICE_TYPE:
        failures.append(
            f"device_type is {state['device_type']}, "
            f"expected {EXPECTED_DEVICE_TYPE}"
        )
    if state["product_code"] != EXPECTED_PRODUCT_CODE:
        failures.append(
            f"product_code is {state['product_code']!r}, "
            f"expected {EXPECTED_PRODUCT_CODE}"
        )
    if state["finger_orientation_outward"]:
        failures.append("finger orientation is outward; inward is required")
    for label, actual, expected in (
        (
            "external minimum",
            state["external_width_mm"]["minimum"],
            EXPECTED_EXTERNAL_RANGE_MM[0],
        ),
        (
            "external maximum",
            state["external_width_mm"]["maximum"],
            EXPECTED_EXTERNAL_RANGE_MM[1],
        ),
        (
            "internal minimum",
            state["internal_width_mm"]["minimum"],
            EXPECTED_INTERNAL_RANGE_MM[0],
        ),
        (
            "internal maximum",
            state["internal_width_mm"]["maximum"],
            EXPECTED_INTERNAL_RANGE_MM[1],
        ),
    ):
        if not math.isclose(actual, expected, abs_tol=RANGE_TOLERANCE_MM):
            failures.append(
                f"{label} width is {actual:.3f}mm, expected {expected:.3f}mm"
            )
    external = state["external_width_mm"]["current"]
    internal = state["internal_width_mm"]["current"]
    if not (
        EXPECTED_EXTERNAL_RANGE_MM[0] - WIDTH_TOLERANCE_MM
        <= external
        <= EXPECTED_EXTERNAL_RANGE_MM[1] + WIDTH_TOLERANCE_MM
    ):
        failures.append(
            f"current external width {external:.6f}mm is outside the fixed "
            "1-39mm range plus 0.5mm readback tolerance"
        )
    if not (
        EXPECTED_INTERNAL_RANGE_MM[0] - WIDTH_TOLERANCE_MM
        <= internal
        <= EXPECTED_INTERNAL_RANGE_MM[1] + WIDTH_TOLERANCE_MM
    ):
        failures.append(
            f"current internal width {internal:.6f}mm is outside the fixed "
            "11-49mm range plus 0.5mm readback tolerance"
        )
    if not math.isclose(
        internal - external,
        INTERNAL_EXTERNAL_OFFSET_MM,
        abs_tol=WIDTH_TOLERANCE_MM,
    ):
        failures.append(
            "internal and external width readbacks do not retain the 10mm offset"
        )
    record = {
        "received_at_utc": captured_at.isoformat(),
        "age_seconds": age_seconds,
        "state": state,
        "payload_sha256": canonical_digest(capture.payload),
        "raw_payload_included": False,
    }
    return record, failures


def _operational_failures(
    state: dict[str, Any],
    *,
    allow_busy: bool,
    allow_error: bool,
    allow_grip: bool,
) -> list[str]:
    failures: list[str] = []
    if state["busy"] and not allow_busy:
        failures.append("2FG7 reports busy")
    if state["grip_detected"] and not allow_grip:
        failures.append("2FG7 reports an unexpected active grip during air replay")
    errors = state["errors"]
    if errors["error_bits"] != 0 and not allow_error:
        failures.append(
            "2FG7 status contains error bits "
            f"({errors['error_bits']}; raw status {errors['status_code']})"
        )
    if errors["not_calibrated"] and not allow_error:
        failures.append("2FG7 reports not calibrated")
    if errors["linear_sensor_error"] and not allow_error:
        failures.append("2FG7 reports a linear sensor error")
    return failures


def _precommand_failures(
    action: GripperAction,
    state: dict[str, Any],
) -> list[str]:
    if action is GripperAction.STOP:
        return []
    failures = _operational_failures(
        state,
        allow_busy=False,
        allow_error=False,
        # OPEN is the release operation and normally starts while a real
        # object, or Watson's inward fingertips at their mechanical meeting
        # point, still asserts grip detection.
        allow_grip=action is GripperAction.OPEN,
    )
    if action is GripperAction.CLOSE:
        external = state["external_width_mm"]["current"]
        internal = state["internal_width_mm"]["current"]
        if not math.isclose(
            external,
            OPEN_EXTERNAL_WIDTH_MM,
            abs_tol=WIDTH_TOLERANCE_MM,
        ) or not math.isclose(
            internal,
            OPEN_EXTERNAL_WIDTH_MM + INTERNAL_EXTERNAL_OFFSET_MM,
            abs_tol=WIDTH_TOLERANCE_MM,
        ):
            failures.append("close requires the synchronized 39mm open state")
    return failures


def _target_reached(spec: CommandSpec, state: dict[str, Any]) -> bool:
    if spec.action is GripperAction.STOP:
        return not state["busy"]
    assert spec.target_external_width_mm is not None
    external_target = spec.target_external_width_mm
    internal_target = external_target + INTERNAL_EXTERNAL_OFFSET_MM
    target_width_reached = (
        math.isclose(
            state["external_width_mm"]["current"],
            external_target,
            abs_tol=WIDTH_TOLERANCE_MM,
        )
        and math.isclose(
            state["internal_width_mm"]["current"],
            internal_target,
            abs_tol=WIDTH_TOLERANCE_MM,
        )
    )
    if state["busy"]:
        return False
    if spec.action is GripperAction.OPEN:
        return target_width_reached and not state["grip_detected"]
    if target_width_reached:
        return True
    return (
        state["grip_detected"]
        and state["external_width_mm"]["current"]
        <= AIR_REPLAY_CLOSE_CONTACT_MAX_EXTERNAL_MM
        and state["internal_width_mm"]["current"]
        <= (
            AIR_REPLAY_CLOSE_CONTACT_MAX_EXTERNAL_MM
            + INTERNAL_EXTERNAL_OFFSET_MM
        )
    )


def _response_record(response: CommandResponse, expected_url: str) -> dict[str, Any]:
    _validate_command_url(response.final_url)
    if response.final_url != expected_url:
        raise QualificationError("Compute Box command response changed URL")
    if response.status_code != 200:
        raise QualificationError(
            f"Compute Box command returned HTTP {response.status_code}"
        )
    body = response.body.strip()
    if body != "0":
        raise QualificationError(
            "Compute Box command did not return the established success token 0"
        )
    return {
        "status_code": response.status_code,
        "final_url": response.final_url,
        "body_sha256": hashlib.sha256(
            response.body.encode("utf-8")
        ).hexdigest(),
        "success_token_matched": True,
        "raw_body_included": False,
    }


def _poll_for_completion(
    transport: OnRobotControlTransport,
    spec: CommandSpec,
    *,
    timing: ControlTiming,
    monotonic: Callable[[], float],
    sleeper: Callable[[float], None],
    wall_clock: Callable[[], datetime],
    abort_requested: Callable[[], bool] | None = None,
) -> tuple[dict[str, Any], int]:
    deadline = monotonic() + timing.poll_timeout_seconds
    poll_count = 0
    last_record: dict[str, Any] | None = None
    while monotonic() < deadline:
        if abort_requested is not None and abort_requested():
            raise InterruptedError(
                f"abort requested while waiting for {spec.action.value}"
            )
        remaining = deadline - monotonic()
        capture = transport.passive_state(
            timeout_seconds=min(
                timing.state_capture_timeout_seconds,
                max(0.001, remaining),
            )
        )
        poll_count += 1
        if abort_requested is not None and abort_requested():
            raise InterruptedError(
                f"abort requested while waiting for {spec.action.value}"
            )
        record, failures = _state_record(
            capture,
            now=wall_clock(),
            max_age_seconds=timing.max_state_age_seconds,
        )
        last_record = record
        failures.extend(
            _operational_failures(
                record["state"],
                allow_busy=True,
                allow_error=False,
                # Grip is an expected intermediate/terminal state for CLOSE,
                # remains latched at the start of OPEN, and must not prevent a
                # STOP from proving that finger motion has ceased.
                allow_grip=True,
            )
        )
        if (
            spec.action is GripperAction.CLOSE
            and record["state"]["grip_detected"]
            and record["state"]["external_width_mm"]["current"]
            > AIR_REPLAY_CLOSE_CONTACT_MAX_EXTERNAL_MM
        ):
            failures.append(
                "2FG7 grip was detected above the fixed 2mm empty-air "
                "contact limit"
            )
        if failures:
            raise QualificationError("; ".join(failures))
        if _target_reached(spec, record["state"]):
            return record, poll_count
        remaining = deadline - monotonic()
        if remaining > 0.0:
            sleeper(min(timing.poll_interval_seconds, remaining))
    width_text = "unknown"
    if last_record is not None:
        width_text = (
            f"{last_record['state']['external_width_mm']['current']:.3f}mm"
        )
    raise TimeoutError(
        f"timed out waiting for {spec.action.value}; last external width "
        f"{width_text}"
    )


def _command_report_fields(spec: CommandSpec) -> dict[str, Any]:
    return {
        "action": spec.action.value,
        "method": "GET",
        "url": spec.url,
        "device_id": FIXED_DEVICE_ID,
        "target_external_width_mm": spec.target_external_width_mm,
        "force_n": spec.force_n,
        "speed_percent": spec.speed_percent,
    }


def _new_report(
    spec: CommandSpec,
    *,
    execute: bool,
    now: datetime,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "timestamp_utc": now.astimezone(timezone.utc).isoformat(),
        "component": "watson_onrobot_2fg7_guarded_control",
        "mode": "execute" if execute else "dry_run",
        "status": "pending" if execute else "dry_run",
        "completed": False,
        "command": _command_report_fields(spec),
        "fixed_profile": {
            "origin": COMPUTE_BOX_ORIGIN,
            "device_id": FIXED_DEVICE_ID,
            "device_type": EXPECTED_DEVICE_TYPE,
            "product_code": EXPECTED_PRODUCT_CODE,
            "finger_orientation": "inward",
            "open_external_width_mm": OPEN_EXTERNAL_WIDTH_MM,
            "air_replay_close_external_width_mm": (
                AIR_REPLAY_CLOSE_EXTERNAL_WIDTH_MM
            ),
            "air_replay_close_contact_max_external_width_mm": (
                AIR_REPLAY_CLOSE_CONTACT_MAX_EXTERNAL_MM
            ),
            "close_target_basis": (
                "Watson 1mm command; idle target readback or inward-fingertip "
                "contact at no more than 2mm completes the air gesture"
            ),
            "force_n": FORCE_N,
            "speed_percent": SPEED_PERCENT,
            "allowed_urls": sorted(ALLOWED_COMMAND_URLS),
        },
        "authorization": {
            "exact_confirmation_required": True,
            "confirmation_value_recorded": False,
            "accepted": False,
        },
        "state_before": None,
        "state_after": None,
        "poll_count": 0,
        "command_response": None,
        "recovery_stop": {
            "attempted": False,
            "response": None,
            "state_after": None,
            "poll_count": 0,
            "failures": [],
        },
        "failures": [],
        "transport_evidence": {
            "network_contact_attempted": False,
            "network_contacted": False,
            "command_attempted": False,
            "command_url": None,
        },
        "safety_evidence": {
            "ros_used": False,
            "watson_arm_contacted": False,
            "arm_motion_commanded": False,
            "gripper_commanded": False,
            "gripper_command_may_have_been_sent": False,
        },
    }


def _finalise_report(
    report: dict[str, Any],
    *,
    report_path: Path | None,
) -> dict[str, Any]:
    report.pop(REPORT_DIGEST_FIELD, None)
    report[REPORT_DIGEST_FIELD] = canonical_digest(report)
    if report_path is not None:
        write_private_report(report_path, report)
    return report


def _attempt_recovery_stop(
    transport: OnRobotControlTransport,
    *,
    timing: ControlTiming,
    monotonic: Callable[[], float],
    sleeper: Callable[[float], None],
    wall_clock: Callable[[], datetime],
) -> dict[str, Any]:
    recovery: dict[str, Any] = {
        "attempted": True,
        "response": None,
        "state_after": None,
        "poll_count": 0,
        "failures": [],
    }
    spec = command_spec(GripperAction.STOP)
    try:
        response = transport.command_get(
            url=spec.url,
            timeout_seconds=timing.command_timeout_seconds,
        )
        recovery["response"] = _response_record(response, spec.url)
        final, poll_count = _poll_for_completion(
            transport,
            spec,
            timing=timing,
            monotonic=monotonic,
            sleeper=sleeper,
            wall_clock=wall_clock,
        )
        recovery["state_after"] = final
        recovery["poll_count"] = poll_count
    except Exception as exc:
        recovery["failures"].append(str(exc))
    return recovery


def run_guarded_command(
    action: GripperAction | str,
    *,
    execute: bool = False,
    confirmation: str | None = None,
    transport: OnRobotControlTransport | None = None,
    report_path: Path | None = None,
    timing: ControlTiming = DEFAULT_TIMING,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    wall_clock: Callable[[], datetime] = _utc_now,
    abort_requested: Callable[[], bool] | None = None,
    dispatch_guard: Callable[[str], ContextManager[Any]] | None = None,
) -> dict[str, Any]:
    """Dry-run or execute one fixed 2FG7 command and return its report.

    A blocked operation is represented by ``status == "blocked"`` and never
    raises solely because a guard failed.  Transport or parsing failures are
    captured in the report.  Report-file creation failures still raise.
    """

    spec = command_spec(action)
    report = _new_report(spec, execute=execute, now=wall_clock())
    if not execute:
        return _finalise_report(report, report_path=report_path)
    if confirmation != CONTROL_CONFIRMATION:
        report["status"] = "blocked"
        report["failures"].append(
            f"execute mode requires exact confirmation {CONTROL_CONFIRMATION!r}"
        )
        return _finalise_report(report, report_path=report_path)
    report["authorization"]["accepted"] = True
    active_transport = transport or FixedComputeBoxTransport()
    if abort_requested is not None and abort_requested():
        report["status"] = "blocked"
        report["failures"].append("abort requested before gripper pre-command state")
        return _finalise_report(report, report_path=report_path)

    try:
        report["transport_evidence"]["network_contact_attempted"] = True
        before_capture = active_transport.passive_state(
            timeout_seconds=timing.state_capture_timeout_seconds
        )
        report["transport_evidence"]["network_contacted"] = True
        if abort_requested is not None and abort_requested():
            report["status"] = "blocked"
            report["failures"].append(
                "abort requested during gripper pre-command state"
            )
            return _finalise_report(report, report_path=report_path)
        before, failures = _state_record(
            before_capture,
            now=wall_clock(),
            max_age_seconds=timing.max_state_age_seconds,
        )
        report["state_before"] = before
        failures.extend(_precommand_failures(spec.action, before["state"]))
        if failures:
            report["status"] = "blocked"
            report["failures"].extend(failures)
            return _finalise_report(report, report_path=report_path)
    except Exception as exc:
        report["status"] = "blocked"
        report["failures"].append(f"pre-command passive state failed: {exc}")
        return _finalise_report(report, report_path=report_path)

    command_may_have_been_sent = False
    try:
        dispatch_context = (
            nullcontext()
            if dispatch_guard is None
            else dispatch_guard(f"2fg7_{spec.action.value}")
        )
        with dispatch_context:
            if abort_requested is not None and abort_requested():
                report["status"] = "blocked"
                report["failures"].append(
                    "abort requested immediately before gripper command"
                )
                return _finalise_report(report, report_path=report_path)
            report["transport_evidence"]["command_attempted"] = True
            report["transport_evidence"]["command_url"] = spec.url
            report["safety_evidence"][
                "gripper_command_may_have_been_sent"
            ] = True
            command_may_have_been_sent = True
            response = active_transport.command_get(
                url=spec.url,
                timeout_seconds=timing.command_timeout_seconds,
            )
        report["safety_evidence"]["gripper_commanded"] = True
        report["command_response"] = _response_record(response, spec.url)
        final, poll_count = _poll_for_completion(
            active_transport,
            spec,
            timing=timing,
            monotonic=monotonic,
            sleeper=sleeper,
            wall_clock=wall_clock,
            abort_requested=abort_requested,
        )
        report["state_after"] = final
        report["poll_count"] = poll_count
        report["status"] = "completed"
        report["completed"] = True
    except Exception as exc:
        report["status"] = "blocked"
        report["failures"].append(f"{spec.action.value} failed: {exc}")
        if command_may_have_been_sent and spec.action is not GripperAction.STOP:
            report["recovery_stop"] = _attempt_recovery_stop(
                active_transport,
                timing=timing,
                monotonic=monotonic,
                sleeper=sleeper,
                wall_clock=wall_clock,
            )
    return _finalise_report(report, report_path=report_path)


def run_fixed_recovery_stop(
    *,
    confirmation: str | None,
    transport: OnRobotControlTransport,
    timing: ControlTiming = DEFAULT_TIMING,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    wall_clock: Callable[[], datetime] = _utc_now,
) -> dict[str, Any]:
    """Send the fixed recovery STOP without depending on pre-state.

    This narrow path exists for the failure case where the passive state
    channel itself is unavailable.  A successful HTTP response is still not
    called complete until a fresh post-state proves the 2FG7 is not busy.
    """

    spec = command_spec(GripperAction.STOP)
    report = _new_report(spec, execute=True, now=wall_clock())
    report["recovery_only_no_prestate_required"] = True
    if confirmation != CONTROL_CONFIRMATION:
        report["status"] = "blocked"
        report["failures"].append(
            f"recovery STOP requires exact confirmation {CONTROL_CONFIRMATION!r}"
        )
        return _finalise_report(report, report_path=None)
    report["authorization"]["accepted"] = True
    try:
        report["transport_evidence"]["network_contact_attempted"] = True
        report["transport_evidence"]["command_attempted"] = True
        report["transport_evidence"]["command_url"] = spec.url
        report["safety_evidence"]["gripper_command_may_have_been_sent"] = True
        response = transport.command_get(
            url=spec.url,
            timeout_seconds=timing.command_timeout_seconds,
        )
        report["transport_evidence"]["network_contacted"] = True
        report["safety_evidence"]["gripper_commanded"] = True
        report["command_response"] = _response_record(response, spec.url)
        final, poll_count = _poll_for_completion(
            transport,
            spec,
            timing=timing,
            monotonic=monotonic,
            sleeper=sleeper,
            wall_clock=wall_clock,
        )
        report["state_after"] = final
        report["poll_count"] = poll_count
        report["status"] = "completed"
        report["completed"] = True
    except Exception as exc:
        report["status"] = "stop_sent_or_attempted_but_unverified"
        report["failures"].append(f"recovery stop failed: {exc}")
    return _finalise_report(report, report_path=None)


__all__ = [
    "AIR_REPLAY_CLOSE_EXTERNAL_WIDTH_MM",
    "AIR_REPLAY_CLOSE_CONTACT_MAX_EXTERNAL_MM",
    "ALLOWED_COMMAND_URLS",
    "COMPUTE_BOX_ORIGIN",
    "CONTROL_CONFIRMATION",
    "CommandResponse",
    "CommandSpec",
    "ControlTiming",
    "FIXED_DEVICE_ID",
    "FORCE_N",
    "FixedComputeBoxTransport",
    "GripperAction",
    "OPEN_EXTERNAL_WIDTH_MM",
    "OnRobotControlTransport",
    "SPEED_PERCENT",
    "command_spec",
    "run_guarded_command",
    "run_fixed_recovery_stop",
]
