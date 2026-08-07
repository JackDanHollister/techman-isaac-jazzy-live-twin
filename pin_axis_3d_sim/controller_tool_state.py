"""Read-only parsing and commissioning checks for TMflow tool settings."""

from __future__ import annotations

import json
import math
import re
from typing import Any


ITEM_SPECS: tuple[tuple[str, str, int | None], ...] = (
    ("TCP_Name", "active_tcp_name", None),
    ("TCP_Value", "tcp_value", 6),
    ("TCP_Mass", "mass_kg", 1),
    ("TCP_MOI", "principal_moi", 3),
    ("TCP_MCF", "mass_centre_frame", 6),
    ("Base_Name", "active_base_name", None),
    ("Base_Value", "base_value", 6),
)

# OnRobot publishes the dry QC-R + 2FG7 TCP, mass, and centre of gravity used
# for this exact record, but not principal moments of inertia.  TMflow exposes
# principal inertia under Additional Settings rather than requiring it to
# create/apply a tool.  Permit the observed zero vector only when every
# published field and the deliberately distinctive record name match this
# commissioned profile.  Other zero-inertia records continue to fail closed.
QC_2FG7_VENDOR_PROFILE: dict[str, str | float | tuple[float, ...]] = {
    "active_tcp_name": "QC_2FG7_VENDOR",
    "tcp_value": (0.0, 0.0, 138.6, 0.0, 0.0, 0.0),
    "mass_kg": 1.2,
    "principal_moi": (0.0, 0.0, 0.0),
    "mass_centre_frame": (0.0, 0.0, 62.52, 0.0, 0.0, 0.0),
}
PROFILE_LINEAR_TOLERANCE_MM = 0.01
PROFILE_ANGULAR_TOLERANCE_DEG = 0.001
PROFILE_MASS_TOLERANCE_KG = 0.001


def _finite_number(value: str, *, item: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{item} contains an invalid number: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{item} contains a non-finite number")
    return result


def parse_tmflow_item(item: str, response_value: str) -> str | float | list[float]:
    """Parse one exact ``AskItem`` response without accepting trailing data."""

    prefix = f"{item}="
    if not isinstance(response_value, str) or not response_value.startswith(prefix):
        raise ValueError(f"{item} response must start with {prefix!r}")
    payload = response_value[len(prefix) :]
    spec = next((entry for entry in ITEM_SPECS if entry[0] == item), None)
    if spec is None:
        raise ValueError(f"Unsupported TMflow item: {item!r}")
    expected_length = spec[2]
    if expected_length is None:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{item} is not a valid quoted string") from exc
        if not isinstance(parsed, str) or not parsed:
            raise ValueError(f"{item} must be a non-empty quoted string")
        return parsed
    if expected_length == 1:
        return _finite_number(payload, item=item)
    if re.fullmatch(r"\{[^{}]*\}", payload) is None:
        raise ValueError(f"{item} must be one flat brace-delimited vector")
    components = payload[1:-1].split(",") if payload[1:-1] else []
    if len(components) != expected_length:
        raise ValueError(
            f"{item} must contain {expected_length} values; found {len(components)}"
        )
    return [_finite_number(component.strip(), item=item) for component in components]


def parse_controller_tool_items(raw_items: dict[str, str]) -> dict[str, Any]:
    """Convert all required controller responses into stable report fields."""

    expected = {item for item, _, _ in ITEM_SPECS}
    if set(raw_items) != expected:
        missing = sorted(expected - set(raw_items))
        extra = sorted(set(raw_items) - expected)
        raise ValueError(f"TMflow item set mismatch: missing={missing}, extra={extra}")
    return {
        field: parse_tmflow_item(item, raw_items[item])
        for item, field, _ in ITEM_SPECS
    }


def matches_qc_2fg7_vendor_profile(settings: dict[str, Any]) -> bool:
    """Return whether all commissioned vendor fields match the named profile."""

    if settings.get("active_tcp_name") != QC_2FG7_VENDOR_PROFILE["active_tcp_name"]:
        return False
    mass = settings.get("mass_kg")
    if (
        isinstance(mass, bool)
        or not isinstance(mass, (int, float))
        or not math.isfinite(float(mass))
        or not math.isclose(
            float(mass),
            float(QC_2FG7_VENDOR_PROFILE["mass_kg"]),
            rel_tol=0.0,
            abs_tol=PROFILE_MASS_TOLERANCE_KG,
        )
    ):
        return False

    for field, expected in (
        ("tcp_value", QC_2FG7_VENDOR_PROFILE["tcp_value"]),
        ("principal_moi", QC_2FG7_VENDOR_PROFILE["principal_moi"]),
        ("mass_centre_frame", QC_2FG7_VENDOR_PROFILE["mass_centre_frame"]),
    ):
        actual = settings.get(field)
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
            if (
                isinstance(actual_value, bool)
                or not isinstance(actual_value, (int, float))
                or not math.isfinite(float(actual_value))
            ):
                return False
            tolerance = (
                PROFILE_ANGULAR_TOLERANCE_DEG
                if field != "principal_moi" and index >= 3
                else PROFILE_LINEAR_TOLERANCE_MM
            )
            if field == "principal_moi":
                tolerance = 1.0e-12
            if not math.isclose(
                float(actual_value),
                float(expected_value),
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                return False
    return True


def controller_tool_failures(settings: dict[str, Any]) -> list[str]:
    """Return fail-closed reasons that the active physical tool is uncommissioned."""

    failures: list[str] = []
    if settings.get("active_tcp_name") in {None, "", "RobotEndFlange"}:
        failures.append("active TCP is the bare RobotEndFlange record")

    tcp = settings.get("tcp_value")
    if not isinstance(tcp, list) or len(tcp) != 6:
        failures.append("active TCP value is unavailable or malformed")
    elif math.dist([float(value) for value in tcp[:3]], [0.0, 0.0, 0.0]) <= 1.0e-9:
        failures.append("active TCP translation is zero")

    mass = settings.get("mass_kg")
    if not isinstance(mass, (int, float)) or not math.isfinite(float(mass)) or mass <= 0.0:
        failures.append("active tool mass is zero or invalid")

    moi = settings.get("principal_moi")
    if not isinstance(moi, list) or len(moi) != 3:
        failures.append("active tool principal moments are unavailable or malformed")
    elif any(float(value) < 0.0 for value in moi):
        failures.append("active tool principal moments are all zero or invalid")
    elif (
        max(float(value) for value in moi) <= 0.0
        and not matches_qc_2fg7_vendor_profile(settings)
    ):
        failures.append("active tool principal moments are all zero or invalid")

    mass_centre = settings.get("mass_centre_frame")
    if not isinstance(mass_centre, list) or len(mass_centre) != 6:
        failures.append("active tool mass-centre frame is unavailable or malformed")
    elif math.dist(
        [float(value) for value in mass_centre[:3]], [0.0, 0.0, 0.0]
    ) <= 1.0e-9:
        failures.append("active tool mass-centre translation is zero")
    return failures


def query_controller_tool_items(
    *,
    node: Any,
    rclpy: Any,
    ask_item_type: Any,
    client: Any,
    timeout_s: float,
) -> dict[str, Any]:
    """Issue only TM Ethernet ``READ_STRING`` requests through ``AskItem``."""

    if timeout_s <= 0.0 or not math.isfinite(timeout_s):
        raise ValueError("controller tool query timeout must be positive and finite")
    if not client.wait_for_service(timeout_sec=timeout_s):
        raise RuntimeError("timed out waiting for the read-only ask_item service")

    raw: dict[str, str] = {}
    for index, (item, _, _) in enumerate(ITEM_SPECS, start=1):
        request = ask_item_type.Request()
        request.id = f"toolaudit{index}"
        request.item = item
        request.wait_time = min(timeout_s, 2.0)
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
        if not future.done():
            raise RuntimeError(f"timed out reading controller item {item}")
        response = future.result()
        if response is None or not response.ok:
            raise RuntimeError(f"controller rejected read-only item {item}")
        if response.id != request.id:
            raise RuntimeError(
                f"controller response ID mismatch for {item}: {response.id!r}"
            )
        raw[item] = response.value

    parsed = parse_controller_tool_items(raw)
    failures = controller_tool_failures(parsed)
    known_vendor_profile = matches_qc_2fg7_vendor_profile(parsed)
    return {
        "raw_items": raw,
        "settings": parsed,
        "promotion_passed": not failures,
        "promotion_failures": failures,
        "commissioning_basis": (
            "exact_qc_2fg7_vendor_profile_without_published_principal_moi"
            if known_vendor_profile
            else "structural_nonzero_tool_record"
        ),
        "known_vendor_profile_matched": known_vendor_profile,
        "read_only_items": [item for item, _, _ in ITEM_SPECS],
        "write_items_called": [],
        "motion_commanded": False,
    }
