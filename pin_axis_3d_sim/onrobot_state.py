"""Fail-closed, read-only qualification of an OnRobot 2FG7 state snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Any, Callable, Iterable
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


COMPUTE_BOX_HOST = os.environ.get("ONROBOT_COMPUTE_BOX_IP", "192.0.2.1")
COMPUTE_BOX_ORIGIN = f"http://{COMPUTE_BOX_HOST}"
SOCKET_IO_PATH = "/socket.io/"
LIVE_CONFIRMATION = "READ_ONLY_ONROBOT_STATE"
MAX_CAPTURE_BYTES = 2 * 1024 * 1024
EXPECTED_DEVICE_TYPE = 17
EXPECTED_PRODUCT_CODE = 192
EXPECTED_EXTERNAL_RANGE_MM = (1.0, 39.0)
EXPECTED_INTERNAL_RANGE_MM = (11.0, 49.0)
EXPECTED_OPEN_EXTERNAL_MM = 39.0
EXPECTED_OPEN_INTERNAL_MM = 49.0
ERROR_NOT_CALIBRATED = 8
ERROR_LINEAR_SENSOR = 16
STATUS_OPERATIONAL_MASK = 0x7
REPORT_DIGEST_FIELD = "report_payload_sha256"

_ACTUATOR_TOKENS = (
    "/api/dc/",
    "grip_external",
    "grip_internal",
    "release",
    "stop",
    "set_finger",
)
_KEY_CLEANER = re.compile(r"[^a-z0-9]+")

_ALIASES = {
    "device_id": (
        "deviceid",
        "deviceindex",
        "deviceidentifier",
    ),
    "device_type": ("devicetype", "devicetypeid"),
    "product_code": ("productcode",),
    "model": ("model", "modelname", "devicename", "productname", "typename", "name"),
    "orientation": (
        "fingerorientationoutward",
        "fingersorientationoutward",
        "fingerorientation",
        "orientationoutward",
        "isoutward",
    ),
    "external_min": (
        "minimumexternalgripwidth",
        "minexternalgripwidth",
        "externalgripwidthmin",
        "minimumexternalwidth",
        "minexternalwidth",
        "externalwidthmin",
    ),
    "external_max": (
        "maximumexternalgripwidth",
        "maxexternalgripwidth",
        "externalgripwidthmax",
        "maximumexternalwidth",
        "maxexternalwidth",
        "externalwidthmax",
    ),
    "external_current": (
        "currentexternalgripwidth",
        "actualexternalgripwidth",
        "externalgripwidth",
        "currentexternalwidth",
        "actualexternalwidth",
        "externalwidth",
    ),
    "internal_min": (
        "minimuminternalgripwidth",
        "mininternalgripwidth",
        "internalgripwidthmin",
        "minimuminternalwidth",
        "mininternalwidth",
        "internalwidthmin",
    ),
    "internal_max": (
        "maximuminternalgripwidth",
        "maxinternalgripwidth",
        "internalgripwidthmax",
        "maximuminternalwidth",
        "maxinternalwidth",
        "internalwidthmax",
    ),
    "internal_current": (
        "currentinternalgripwidth",
        "actualinternalgripwidth",
        "internalgripwidth",
        "currentinternalwidth",
        "actualinternalwidth",
        "internalwidth",
    ),
    "busy": ("busy", "isbusy", "devicebusy"),
    "grip_detected": (
        "gripdetected",
        "isgripdetected",
        "gripdetection",
        "objectdetected",
    ),
    "status_code": ("status", "statuscode", "error", "errorcode", "errors"),
    "not_calibrated": (
        "notcalibrated",
        "calibrationerror",
        "iscalibrationerror",
    ),
    "linear_sensor_error": (
        "linearsensorerror",
        "islinearsensorerror",
        "linearsensorfault",
    ),
}

_STATE_FIELDS = (
    "orientation",
    "external_min",
    "external_max",
    "external_current",
    "internal_min",
    "internal_max",
    "internal_current",
    "busy",
    "grip_detected",
    "status_code",
    "not_calibrated",
    "linear_sensor_error",
)


class QualificationError(ValueError):
    """Raised when a capture cannot be interpreted unambiguously."""


@dataclass(frozen=True)
class LiveCapture:
    """A single passive state event and transport-only evidence."""

    payload: Any
    received_at_utc: datetime
    transport: dict[str, Any]


@dataclass(frozen=True)
class _Candidate:
    value: dict[str, Any]
    path: tuple[str, ...]
    score: int


def _normalise_key(value: str) -> str:
    return _KEY_CLEANER.sub("", value.lower())


def _normalise_for_digest(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalise_for_digest(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        return [_normalise_for_digest(item) for item in value]
    if isinstance(value, tuple):
        return [_normalise_for_digest(item) for item in value]
    return value


def canonical_digest(value: Any) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible data."""

    encoded = json.dumps(
        _normalise_for_digest(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _capture_digest(value: Any) -> str:
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    if isinstance(value, str):
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
    return canonical_digest(_decode_nested_json(value))


def _decode_nested_json(value: Any, *, encoded_depth: int = 0) -> Any:
    if encoded_depth > 5:
        raise QualificationError("capture contains excessive nested JSON encoding")
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                return _decode_nested_json(
                    json.loads(stripped),
                    encoded_depth=encoded_depth + 1,
                )
            except json.JSONDecodeError:
                return value
        return value
    if isinstance(value, list):
        return [
            _decode_nested_json(item, encoded_depth=encoded_depth)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _decode_nested_json(item, encoded_depth=encoded_depth)
            for key, item in value.items()
        }
    return value


def decode_capture(raw: bytes | str | Any) -> Any:
    """Decode plain JSON or an Engine.IO/Socket.IO ``message`` event capture."""

    if not isinstance(raw, (bytes, str)):
        return _decode_nested_json(raw)
    if isinstance(raw, bytes):
        if len(raw) > MAX_CAPTURE_BYTES:
            raise QualificationError("capture exceeds the 2 MiB input limit")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise QualificationError("capture is not UTF-8") from exc
    else:
        text = raw
        if len(text.encode("utf-8")) > MAX_CAPTURE_BYTES:
            raise QualificationError("capture exceeds the 2 MiB input limit")
    text = text.strip()
    if not text:
        raise QualificationError("capture is empty")
    try:
        return _decode_nested_json(json.loads(text))
    except json.JSONDecodeError:
        pass

    message_payloads: list[Any] = []
    for packet in _engine_io_packets(text):
        if not packet.startswith("42"):
            continue
        try:
            event = json.loads(packet[2:])
        except json.JSONDecodeError as exc:
            raise QualificationError("malformed Socket.IO event packet") from exc
        if not isinstance(event, list) or len(event) < 2:
            raise QualificationError("malformed Socket.IO event envelope")
        if event[0] != "message":
            continue
        message_payloads.append(_decode_nested_json(event[1]))
    if not message_payloads:
        raise QualificationError("capture contains no inbound Socket.IO message event")
    if len(message_payloads) != 1:
        raise QualificationError(
            f"capture contains {len(message_payloads)} message events; exactly one is required"
        )
    return message_payloads[0]


def _engine_io_packets(text: str) -> list[str]:
    packets: list[str] = []
    for record in text.replace("\r\n", "\n").split("\x1e"):
        for line in record.splitlines():
            line = line.strip()
            if not line:
                continue
            length_prefix = re.fullmatch(r"(\d+):(.*)", line, flags=re.DOTALL)
            if length_prefix and int(length_prefix.group(1)) == len(
                length_prefix.group(2).encode("utf-8")
            ):
                line = length_prefix.group(2)
            packets.append(line)
    return packets


def _walk_dicts(
    value: Any,
    path: tuple[str, ...] = (),
) -> Iterable[tuple[tuple[str, ...], dict[str, Any]]]:
    if isinstance(value, dict):
        yield path, value
        for key, item in value.items():
            yield from _walk_dicts(item, path + (str(key),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_dicts(item, path + (f"[{index}]",))


def _flatten_scalars(
    value: dict[str, Any],
    path: tuple[str, ...] = (),
) -> dict[str, list[tuple[tuple[str, ...], Any]]]:
    flattened: dict[str, list[tuple[tuple[str, ...], Any]]] = {}
    for key, item in value.items():
        child_path = path + (str(key),)
        if isinstance(item, dict):
            nested = _flatten_scalars(item, child_path)
            for nested_key, entries in nested.items():
                flattened.setdefault(nested_key, []).extend(entries)
        elif not isinstance(item, (list, tuple, set)):
            flattened.setdefault(_normalise_key(str(key)), []).append((child_path, item))
    return flattened


def _alias_entries(
    flattened: dict[str, list[tuple[tuple[str, ...], Any]]],
    field: str,
) -> list[tuple[tuple[str, ...], Any]]:
    entries: list[tuple[tuple[str, ...], Any]] = []
    for alias in _ALIASES[field]:
        entries.extend(flattened.get(alias, ()))
    return entries


def _candidate_identity(
    flattened: dict[str, list[tuple[tuple[str, ...], Any]]],
) -> bool:
    for _, value in _alias_entries(flattened, "device_type"):
        if _try_int(value) == EXPECTED_DEVICE_TYPE:
            return True
    for _, value in _alias_entries(flattened, "product_code"):
        if _try_int(value) == EXPECTED_PRODUCT_CODE:
            return True
    for _, value in _alias_entries(flattened, "model"):
        model = _normalise_key(str(value))
        if model in {"2fg7", "twofg7"} or "twofg7" in model:
            return True
    return False


def _try_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    return None


def _candidate_score(
    flattened: dict[str, list[tuple[tuple[str, ...], Any]]],
) -> int:
    return sum(bool(_alias_entries(flattened, field)) for field in _STATE_FIELDS)


def _find_candidate(value: Any) -> dict[str, Any]:
    candidates: list[_Candidate] = []
    for path, mapping in _walk_dicts(value):
        flattened = _flatten_scalars(mapping)
        if _candidate_identity(flattened):
            candidates.append(
                _Candidate(mapping, path, _candidate_score(flattened))
            )
    if not candidates:
        raise QualificationError("capture contains no identifiable 2FG7 device")
    best_score = max(candidate.score for candidate in candidates)
    best = [candidate for candidate in candidates if candidate.score == best_score]
    minimal: list[_Candidate] = []
    for candidate in best:
        has_descendant = any(
            other.path != candidate.path
            and other.path[: len(candidate.path)] == candidate.path
            for other in best
        )
        if not has_descendant:
            minimal.append(candidate)
    if len(minimal) != 1:
        paths = [
            ".".join(candidate.path) if candidate.path else "<root>"
            for candidate in minimal
        ]
        raise QualificationError(
            "capture contains ambiguous 2FG7 candidates at " + ", ".join(paths)
        )
    return minimal[0].value


def _unique_raw(
    flattened: dict[str, list[tuple[tuple[str, ...], Any]]],
    field: str,
    *,
    required: bool,
) -> Any:
    entries = _alias_entries(flattened, field)
    if not entries:
        if required:
            raise QualificationError(f"2FG7 state is missing required field {field}")
        return None
    unique: list[Any] = []
    for _, value in entries:
        if not any(value == existing and type(value) is type(existing) for existing in unique):
            unique.append(value)
    if len(unique) != 1:
        paths = [".".join(path) for path, _ in entries]
        raise QualificationError(
            f"2FG7 state has conflicting values for {field} at {', '.join(paths)}"
        )
    return unique[0]


def _as_int(value: Any, field: str, *, minimum: int | None = None) -> int:
    result = _try_int(value)
    if result is None:
        raise QualificationError(f"2FG7 field {field} must be an integer")
    if minimum is not None and result < minimum:
        raise QualificationError(f"2FG7 field {field} must be at least {minimum}")
    return result


def _as_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise QualificationError(f"2FG7 field {field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise QualificationError(f"2FG7 field {field} must be numeric") from exc
    if not math.isfinite(result):
        raise QualificationError(f"2FG7 field {field} must be finite")
    return result


def _as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
    raise QualificationError(f"2FG7 field {field} must be boolean")


def _orientation_outward(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "outward":
            return True
        if lowered == "inward":
            return False
    return _as_bool(value, "finger_orientation_outward")


def normalise_2fg7_state(capture: Any) -> dict[str, Any]:
    """Extract exactly one 2FG7 and return a strict normalized state."""

    candidate = _find_candidate(_decode_nested_json(capture))
    flattened = _flatten_scalars(candidate)
    device_id = _as_int(
        _unique_raw(flattened, "device_id", required=True),
        "device_id",
        minimum=0,
    )
    device_type_raw = _unique_raw(flattened, "device_type", required=True)
    device_type = _as_int(device_type_raw, "device_type", minimum=0)
    product_code_raw = _unique_raw(flattened, "product_code", required=False)
    product_code = (
        None
        if product_code_raw is None
        else _as_int(product_code_raw, "product_code", minimum=0)
    )
    model_values = [
        str(value) for _, value in _alias_entries(flattened, "model")
    ]
    model = next(
        (
            value
            for value in model_values
            if "2fg7" in _normalise_key(value)
            or "twofg7" in _normalise_key(value)
        ),
        model_values[0] if model_values else None,
    )

    status_raw = _unique_raw(flattened, "status_code", required=True)
    status_code = _as_int(status_raw, "status_code", minimum=0)
    not_calibrated_raw = _unique_raw(
        flattened, "not_calibrated", required=False
    )
    linear_sensor_raw = _unique_raw(
        flattened, "linear_sensor_error", required=False
    )
    status_not_calibrated = bool(status_code & ERROR_NOT_CALIBRATED)
    status_linear_sensor = bool(status_code & ERROR_LINEAR_SENSOR)
    not_calibrated = (
        status_not_calibrated
        if not_calibrated_raw is None
        else _as_bool(not_calibrated_raw, "not_calibrated")
    )
    linear_sensor_error = (
        status_linear_sensor
        if linear_sensor_raw is None
        else _as_bool(linear_sensor_raw, "linear_sensor_error")
    )
    if (
        not_calibrated_raw is not None
        and not_calibrated != status_not_calibrated
    ):
        raise QualificationError(
            "2FG7 not_calibrated disagrees with status error bit 8"
        )
    if (
        linear_sensor_raw is not None
        and linear_sensor_error != status_linear_sensor
    ):
        raise QualificationError(
            "2FG7 linear_sensor_error disagrees with status error bit 16"
        )

    return {
        "device_id": device_id,
        "device_type": device_type,
        "product_code": product_code,
        "model": model,
        "finger_orientation_outward": _orientation_outward(
            _unique_raw(flattened, "orientation", required=True)
        ),
        "external_width_mm": {
            "minimum": _as_float(
                _unique_raw(flattened, "external_min", required=True),
                "external_min",
            ),
            "maximum": _as_float(
                _unique_raw(flattened, "external_max", required=True),
                "external_max",
            ),
            "current": _as_float(
                _unique_raw(flattened, "external_current", required=True),
                "external_current",
            ),
        },
        "internal_width_mm": {
            "minimum": _as_float(
                _unique_raw(flattened, "internal_min", required=True),
                "internal_min",
            ),
            "maximum": _as_float(
                _unique_raw(flattened, "internal_max", required=True),
                "internal_max",
            ),
            "current": _as_float(
                _unique_raw(flattened, "internal_current", required=True),
                "internal_current",
            ),
        },
        "busy": _as_bool(
            _unique_raw(flattened, "busy", required=True), "busy"
        ),
        "grip_detected": _as_bool(
            _unique_raw(flattened, "grip_detected", required=True),
            "grip_detected",
        ),
        "errors": {
            "status_code": status_code,
            "operational_status_bits": status_code & STATUS_OPERATIONAL_MASK,
            "error_bits": status_code & ~STATUS_OPERATIONAL_MASK,
            "not_calibrated": not_calibrated,
            "linear_sensor_error": linear_sensor_error,
            "other_status_bits": status_code
            & ~(
                STATUS_OPERATIONAL_MASK
                | ERROR_NOT_CALIBRATED
                | ERROR_LINEAR_SENSOR
            ),
        },
    }


def _parse_utc(value: datetime | str, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise QualificationError(f"{field} is not an ISO-8601 timestamp") from exc
    else:
        raise QualificationError(f"{field} is not an ISO-8601 timestamp")
    if parsed.tzinfo is None:
        raise QualificationError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def capture_timestamp(capture: Any) -> datetime | None:
    """Read a unique envelope timestamp without inspecting device timestamps."""

    if not isinstance(capture, dict):
        return None
    aliases = {
        "capturedatutc",
        "capturetimestamp",
        "capturetimestamputc",
        "receivedatutc",
    }
    matches = [
        value
        for key, value in capture.items()
        if _normalise_key(str(key)) in aliases
    ]
    if not matches:
        return None
    parsed = [_parse_utc(value, "capture timestamp") for value in matches]
    if any(value != parsed[0] for value in parsed[1:]):
        raise QualificationError("capture contains conflicting envelope timestamps")
    return parsed[0]


def qualification_failures(
    state: dict[str, Any],
    *,
    captured_at: datetime,
    now: datetime,
    max_age_seconds: float,
    range_tolerance_mm: float = 0.25,
    open_tolerance_mm: float = 0.5,
) -> tuple[list[str], float]:
    """Evaluate the exact Watson/Isaac air-demo starting state."""

    if max_age_seconds <= 0.0 or not math.isfinite(max_age_seconds):
        raise ValueError("max_age_seconds must be positive and finite")
    captured = _parse_utc(captured_at, "captured_at")
    current = _parse_utc(now, "now")
    age_seconds = (current - captured).total_seconds()
    failures: list[str] = []
    if age_seconds < -1.0:
        failures.append(
            f"capture timestamp is {-age_seconds:.3f}s in the future"
        )
    elif age_seconds > max_age_seconds:
        failures.append(
            f"capture is stale ({age_seconds:.3f}s > {max_age_seconds:.3f}s)"
        )
    if state["device_type"] != EXPECTED_DEVICE_TYPE:
        failures.append(
            f"device_type is {state['device_type']}, expected 2FG7 type 17"
        )
    if state["product_code"] != EXPECTED_PRODUCT_CODE:
        failures.append(
            f"product_code is {state['product_code']!r}, expected 192"
        )
    if state["finger_orientation_outward"]:
        failures.append("finger orientation is outward; inward is required")
    for name, actual, expected in (
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
        if not math.isclose(actual, expected, abs_tol=range_tolerance_mm):
            failures.append(
                f"{name} width is {actual:.3f}mm, expected {expected:.3f}mm"
            )
    external = state["external_width_mm"]
    internal = state["internal_width_mm"]
    if not (
        external["minimum"] - open_tolerance_mm
        <= external["current"]
        <= external["maximum"] + open_tolerance_mm
    ):
        failures.append(
            "current external width is outside its advertised range plus "
            "the open-state readback tolerance"
        )
    if not (
        internal["minimum"] - open_tolerance_mm
        <= internal["current"]
        <= internal["maximum"] + open_tolerance_mm
    ):
        failures.append(
            "current internal width is outside its advertised range plus "
            "the open-state readback tolerance"
        )
    if not math.isclose(
        external["current"], EXPECTED_OPEN_EXTERNAL_MM, abs_tol=open_tolerance_mm
    ):
        failures.append(
            "current external width is not synchronized to the 39mm open state"
        )
    if not math.isclose(
        internal["current"], EXPECTED_OPEN_INTERNAL_MM, abs_tol=open_tolerance_mm
    ):
        failures.append(
            "current internal width is not synchronized to the 49mm open state"
        )
    if state["busy"]:
        failures.append("2FG7 reports busy")
    if state["grip_detected"]:
        failures.append("2FG7 reports an active grip")
    errors = state["errors"]
    if errors["error_bits"] != 0:
        failures.append(
            "2FG7 status contains error bits "
            f"({errors['error_bits']}; raw status {errors['status_code']})"
        )
    if errors["not_calibrated"]:
        failures.append("2FG7 reports not calibrated")
    if errors["linear_sensor_error"]:
        failures.append("2FG7 reports a linear sensor error")
    return failures, age_seconds


def build_qualification_report(
    capture: Any,
    *,
    mode: str,
    captured_at: datetime | str | None,
    now: datetime | None = None,
    max_age_seconds: float = 5.0,
    transport: dict[str, Any] | None = None,
    capture_digest: str | None = None,
) -> dict[str, Any]:
    """Build a report even when parsing or qualification fails."""

    if mode not in {"offline_capture", "live_read_only_socketio"}:
        raise ValueError("unsupported qualification mode")
    report_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    failures: list[str] = []
    state: dict[str, Any] | None = None
    age_seconds: float | None = None
    try:
        decoded = decode_capture(capture)
        state = normalise_2fg7_state(decoded)
        if captured_at is None:
            captured_at = capture_timestamp(decoded)
        if captured_at is None:
            raise QualificationError("capture has no trusted envelope timestamp")
        captured = _parse_utc(captured_at, "captured_at")
        failures, age_seconds = qualification_failures(
            state,
            captured_at=captured,
            now=report_time,
            max_age_seconds=max_age_seconds,
        )
        captured_text: str | None = captured.isoformat()
    except (QualificationError, TypeError, ValueError) as exc:
        failures = [str(exc)]
        captured_text = None
    report: dict[str, Any] = {
        "format_version": 1,
        "timestamp_utc": report_time.isoformat(),
        "status": "qualified" if not failures else "blocked",
        "mode": mode,
        "ready_for_air_demo_sync": not failures,
        "capture": {
            "captured_at_utc": captured_text,
            "age_seconds": age_seconds,
            "max_age_seconds": max_age_seconds,
            "payload_sha256": capture_digest
            or _capture_digest(capture),
            "raw_payload_included": False,
        },
        "state": state,
        "failures": failures,
        "transport": transport
        or {
            "network_connection_opened": False,
            "origin": None,
            "socket_path": None,
            "inbound_events_accepted": [],
            "application_events_emitted": [],
            "http_methods": [],
        },
        "safety_evidence": {
            "watson_contacted": False,
            "ros_used": False,
            "motion_commanded": False,
            "gripper_commanded": False,
            "actuator_endpoint_used": False,
            "application_events_emitted": [],
        },
    }
    report[REPORT_DIGEST_FIELD] = canonical_digest(report)
    return report


def _validate_compute_box_origin(origin: str) -> None:
    parsed = urlsplit(origin)
    if (
        parsed.scheme != "http"
        or parsed.hostname != COMPUTE_BOX_HOST
        or parsed.port not in (None, 80)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise QualificationError(
            f"live reads are pinned to the Compute Box origin {COMPUTE_BOX_ORIGIN}"
        )


def _socket_url(sid: str | None = None) -> str:
    query: dict[str, str] = {
        "EIO": "4",
        "transport": "polling",
        "t": f"{time.monotonic_ns():x}",
    }
    if sid is not None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", sid):
            raise QualificationError("Socket.IO handshake returned an unsafe session ID")
        query["sid"] = sid
    url = f"{COMPUTE_BOX_ORIGIN}{SOCKET_IO_PATH}?{urlencode(query)}"
    lowered = url.lower()
    if any(token in lowered for token in _ACTUATOR_TOKENS):
        raise QualificationError("refusing actuator-like URL in read-only transport")
    return url


def _read_response(response: Any) -> str:
    data = response.read(MAX_CAPTURE_BYTES + 1)
    if len(data) > MAX_CAPTURE_BYTES:
        raise QualificationError("Compute Box response exceeds the 2 MiB limit")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QualificationError("Compute Box response is not UTF-8") from exc


def _request_text(
    opener: Callable[..., Any],
    *,
    method: str,
    url: str,
    timeout: float,
    body: str | None = None,
) -> str:
    data = None if body is None else body.encode("ascii")
    headers = {
        "Accept": "text/plain",
        "Cache-Control": "no-store",
    }
    if data is not None:
        headers["Content-Type"] = "text/plain;charset=UTF-8"
    request = Request(url, data=data, headers=headers, method=method)
    response = opener(request, timeout=timeout)
    if hasattr(response, "__enter__"):
        with response:
            return _read_response(response)
    try:
        return _read_response(response)
    finally:
        close = getattr(response, "close", None)
        if close is not None:
            close()


def capture_live_read_only(
    *,
    confirmation: str,
    timeout_seconds: float = 5.0,
    opener: Callable[..., Any] = urlopen,
    monotonic: Callable[[], float] = time.monotonic,
) -> LiveCapture:
    """Receive one passive Socket.IO ``message`` event from the Compute Box.

    The only outbound bodies are Socket.IO namespace connect (``40``) and,
    when requested by the server, Engine.IO pong (``3``). No application event
    is emitted and no ``/api/dc/`` route can be constructed.
    """

    if confirmation != LIVE_CONFIRMATION:
        raise QualificationError(
            f"live mode requires exact confirmation {LIVE_CONFIRMATION!r}"
        )
    if timeout_seconds <= 0.0 or not math.isfinite(timeout_seconds):
        raise ValueError("timeout_seconds must be positive and finite")
    _validate_compute_box_origin(COMPUTE_BOX_ORIGIN)
    deadline = monotonic() + timeout_seconds
    handshake = _request_text(
        opener,
        method="GET",
        url=_socket_url(),
        timeout=timeout_seconds,
    )
    packets = _engine_io_packets(handshake)
    open_packets = [packet for packet in packets if packet.startswith("0")]
    if len(open_packets) != 1:
        raise QualificationError("invalid Engine.IO handshake response")
    try:
        details = json.loads(open_packets[0][1:])
    except json.JSONDecodeError as exc:
        raise QualificationError("invalid Engine.IO handshake JSON") from exc
    if not isinstance(details, dict) or not isinstance(details.get("sid"), str):
        raise QualificationError("Engine.IO handshake did not supply a session ID")
    sid = details["sid"]
    connect_url = _socket_url(sid)
    _request_text(
        opener,
        method="POST",
        url=connect_url,
        timeout=max(0.001, deadline - monotonic()),
        body="40",
    )
    http_methods = ["GET", "POST"]
    protocol_bodies = ["40"]
    while monotonic() < deadline:
        response = _request_text(
            opener,
            method="GET",
            url=_socket_url(sid),
            timeout=max(0.001, deadline - monotonic()),
        )
        http_methods.append("GET")
        for packet in _engine_io_packets(response):
            if packet == "2":
                _request_text(
                    opener,
                    method="POST",
                    url=_socket_url(sid),
                    timeout=max(0.001, deadline - monotonic()),
                    body="3",
                )
                http_methods.append("POST")
                protocol_bodies.append("3")
                continue
            if not packet.startswith("42"):
                continue
            try:
                event = json.loads(packet[2:])
            except json.JSONDecodeError as exc:
                raise QualificationError("malformed inbound Socket.IO event") from exc
            if not isinstance(event, list) or len(event) < 2:
                raise QualificationError("malformed inbound Socket.IO event envelope")
            if event[0] != "message":
                continue
            received_at = datetime.now(timezone.utc)
            return LiveCapture(
                payload=_decode_nested_json(event[1]),
                received_at_utc=received_at,
                transport={
                    "network_connection_opened": True,
                    "origin": COMPUTE_BOX_ORIGIN,
                    "socket_path": SOCKET_IO_PATH,
                    "inbound_events_accepted": ["message"],
                    "application_events_emitted": [],
                    "http_methods": http_methods,
                    "transport_protocol_bodies_sent": protocol_bodies,
                },
            )
    raise QualificationError("timed out waiting for a passive 2FG7 message event")


def write_private_report(path: Path, report: dict[str, Any]) -> Path:
    """Create a mode-0600 JSON report without following links or overwriting."""

    target = path.expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    target = target.parent.resolve() / target.name
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(target, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            opened = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or stat.S_IMODE(opened.st_mode) != 0o600
                or opened.st_uid != os.geteuid()
                or opened.st_nlink != 1
            ):
                raise RuntimeError("qualification report failed private-file checks")
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    final = target.lstat()
    if (
        target.is_symlink()
        or not stat.S_ISREG(final.st_mode)
        or stat.S_IMODE(final.st_mode) != 0o600
        or final.st_uid != os.geteuid()
        or final.st_nlink != 1
    ):
        target.unlink(missing_ok=True)
        raise RuntimeError("qualification report failed final private-file checks")
    directory_fd = os.open(
        target.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return target
