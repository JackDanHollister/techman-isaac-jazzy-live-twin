"""Immutable inputs and pure guards for Watson's seven-pin air replay.

This module has no ROS imports, network calls, action clients, or controller
write path.  It accepts only the two private artifacts reviewed on 2026-07-23,
re-derives the retimed points, and exposes exact stage/message validation for
the separately guarded runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
from typing import Any, Mapping, Sequence

from .controller_tool_state import (
    QC_2FG7_VENDOR_PROFILE,
    matches_qc_2fg7_vendor_profile,
)
from .watson_guard import HealthSnapshot
from .watson_multi_pin_retime import (
    ARTIFACT_DIGEST_FIELD,
    DEFAULT_REVIEWED_PLAN,
    DERIVATIVE_LIMIT_FRACTION,
    JOINT_NAMES,
    JOINT_POSITION_LOWER_RAD,
    JOINT_POSITION_UPPER_RAD,
    JOINT_VELOCITY_LIMITS_RAD_S,
    PVTPoint,
    READY_JOINT_POSITIONS_RAD,
    canonical_digest,
    load_reviewed_plan,
    retime_ingress_control_samples,
    validate_live_first_wire_cubic,
    validate_retimed_artifact,
    validate_retimed_ingress_candidate,
)


ARENA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RETIMED_ARTIFACT = (
    ARENA_DIR
    / "local/execution/"
    "retimed_seven_pin_air_replay.json"
)
DEFAULT_INGRESS_ARTIFACT = (
    ARENA_DIR
    / "local/execution/"
    "tool_aware_ready_ingress.json"
)

EXPECTED_RETIMED_FILE_SHA256 = (
    "8f24ba8c8cf6f814ba12f33e8202cf214b4fd89cd7d9017d11f75d075c5400fb"
)
EXPECTED_RETIMED_PAYLOAD_SHA256 = (
    "f904428c355579d177edb34a560968aa6fa3a30fc355b286a107f35d313c2616"
)
EXPECTED_RETIMED_WIRE_NUMERIC_SHA256 = (
    "7da9b2450a7ee0841707ad702042e534387339d21cf8e0446cc6458d6c781b42"
)
EXPECTED_INGRESS_FILE_SHA256 = (
    "5c13f72b209781417448f48098c222077a5065809a05b7c39e46d898e713b018"
)
EXPECTED_INGRESS_NUMERIC_SHA256 = (
    "68e74b20cfefe28a4f9750a6c8767f988641a9daec7041ca95f63c8f0465940a"
)
EXPECTED_INGRESS_SOURCE_LIVE_CHECK_SHA256 = (
    "fe715c4451243bb5e46c6740bb05b1b1e0620142beef4e07df4a02a6ff11ea1b"
)
EXPECTED_INGRESS_SOURCE_LIVE_CHECK_DIGEST = (
    "60fcfbefcfa4a6c07cea046cb6338d5a561fe35d0d027f01ca01680a9f237755"
)
EXPECTED_TOOL_ASSET_SHA256 = {
    "urdf": "ee7dadbee3e898152948c133f859f7bd085c93614fc8274549158cca10a18d03",
    "xrdf": "41f1575758b5ee65b6d337c7159e6f3bb70eae3a32863f3dac58e727c859d5df",
    "asset_manifest": (
        "60835ef5ca0c6212633864b610be5fd4a222ffc6e84732b0760c939d61b8b993"
    ),
}

EXECUTION_ARM_TOKEN = "MOVE_WATSON_SEVEN_PIN_AIR_REPLAY"
GRIPPER_EXECUTION_TOKEN = "EXECUTE_WATSON_2FG7_AIR_REPLAY"
MAX_PRIVATE_ARTIFACT_BYTES = 8 * 1024 * 1024
LIVE_START_TOLERANCE_RAD = 0.001
LIVE_GOAL_TOLERANCE_RAD = 0.003
LIVE_POSITION_ENVELOPE_MARGIN_RAD = 0.005
LIVE_VELOCITY_ENVELOPE_MARGIN_RAD_S = 0.05
MAX_PROJECT_SPEED = 50
GRIPPER_POLICY = {
    "mode": "execute_only_guarded_compute_box",
    "action_clients_created": 0,
    "services_called": [],
    "topics_published": [],
    "check_dry_run_transport_created": False,
    "single_transport_per_execute_run": True,
    "open_external_width_mm": 39.0,
    "close_external_width_mm": 1.0,
    "close_contact_max_external_width_mm": 2.0,
    "close_completion": (
        "idle 1mm target readback or deliberate inward-fingertip contact "
        "at no more than 2mm in the confirmed-clear air replay"
    ),
    "open_completion_requires_grip_cleared": True,
    "force_n": 20,
    "speed_percent": 10,
    "commands": ["open", "close", "stop_recovery_only"],
}
GRIPPER_AFTER_STAGE_HOOKS = {
    "descend_tilted_grasp": "close",
    "descend_vertical": "open",
}


@dataclass(frozen=True)
class StageSpec:
    """One independently submitted controller stage."""

    sequence_index: int
    kind: str
    specimen_id: int | None
    stage_index: int
    stage_name: str
    points: tuple[PVTPoint, ...]
    position_minimum_rad: tuple[float, ...]
    position_maximum_rad: tuple[float, ...]
    maximum_cubic_velocity_rad_s: tuple[float, ...]
    points_sha256: str
    first_serialized_wire_point: Mapping[str, Any]
    serialized_wire_tokens_sha256: str

    @property
    def start_positions(self) -> tuple[float, ...]:
        return self.points[0].positions

    @property
    def goal_positions(self) -> tuple[float, ...]:
        return self.points[-1].positions

    @property
    def duration_s(self) -> float:
        return self.points[-1].time_s


@dataclass(frozen=True)
class ExecutionBundle:
    """The exact ingress plus 49 reviewed seven-pin arm stages."""

    retimed_path: Path
    ingress_path: Path
    retimed_file_sha256: str
    ingress_file_sha256: str
    retimed_payload_sha256: str
    retimed_wire_numeric_sha256: str
    ingress_numeric_sha256: str
    stages: tuple[StageSpec, ...]


def _absolute_without_final_symlink(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.parent.resolve() / expanded.name


def _read_private_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[Path, dict[str, Any], str]:
    normalized = _absolute_without_final_symlink(path)
    before = normalized.lstat()
    if (
        normalized.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > MAX_PRIVATE_ARTIFACT_BYTES
    ):
        raise ValueError(
            f"{label} must be a private mode-0600, owner-only, single-link "
            f"regular file: {normalized}"
        )
    descriptor = os.open(
        normalized,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"{label} changed while opening")
        chunks: list[bytes] = []
        remaining = MAX_PRIVATE_ARTIFACT_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    if len(data) > MAX_PRIVATE_ARTIFACT_BYTES:
        raise ValueError(f"{label} exceeds the accepted size")
    after = normalized.lstat()
    if (after.st_dev, after.st_ino, after.st_size) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
    ):
        raise ValueError(f"{label} changed while reading")
    observed_sha256 = hashlib.sha256(data).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: {observed_sha256} != {expected_sha256}"
        )
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return normalized, value, observed_sha256


def _finite_six(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != len(JOINT_NAMES):
        raise ValueError(f"{label} must contain six values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def _float64_rows_sha256(rows: Sequence[Sequence[float]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        values = tuple(float(value) for value in row)
        digest.update(struct.pack(f"<{len(values)}d", *values))
    return digest.hexdigest()


def _profile_as_lists() -> dict[str, Any]:
    return {
        "active_tcp_name": QC_2FG7_VENDOR_PROFILE["active_tcp_name"],
        "tcp_value": list(QC_2FG7_VENDOR_PROFILE["tcp_value"]),
        "mass_kg": QC_2FG7_VENDOR_PROFILE["mass_kg"],
        "principal_moi": list(QC_2FG7_VENDOR_PROFILE["principal_moi"]),
        "mass_centre_frame": list(
            QC_2FG7_VENDOR_PROFILE["mass_centre_frame"]
        ),
    }


def validate_ingress_artifact(
    ingress: dict[str, Any],
) -> dict[str, Any]:
    """Validate the exact tool-aware ingress and derive controller points."""

    exact_fields = {
        "format_version": 1,
        "status": "offline_tool_aware_ingress_validated_not_executable",
        "joint_names": list(JOINT_NAMES),
        "path_mode": "deterministic_straight_cspace",
        "tool_profile": "watson_qc_nominal",
        "controller_tool_profile": _profile_as_lists(),
        "linear_self_collision_samples": 1001,
        "linear_self_collision_count": 0,
        "control_dt_seconds": 1.0 / 300.0,
        "execution_time_scale": 1.5,
        "control_samples_float64_sha256": EXPECTED_INGRESS_NUMERIC_SHA256,
    }
    for field, expected in exact_fields.items():
        if ingress.get(field) != expected:
            raise ValueError(f"Ingress {field} changed")
    if (
        _finite_six(ingress.get("ready_joint_positions"), "ingress ready")
        != READY_JOINT_POSITIONS_RAD
    ):
        raise ValueError("Ingress ready pose changed")
    start = _finite_six(ingress.get("start_joint_positions"), "ingress start")

    scope = ingress.get("scope")
    expected_scope = {
        "ros_graph_created": False,
        "network_connection_opened": False,
        "watson_connected": False,
        "controller_trajectory_created": False,
        "real_robot_commanded": False,
        "gripper_commanded": False,
        "workcell_obstacles_modelled": False,
    }
    if scope != expected_scope:
        raise ValueError("Ingress safety scope changed")
    validation = ingress.get("trajectory_validation")
    if (
        not isinstance(validation, dict)
        or validation.get("sampled_self_collision") is not False
        or validation.get("derivative_limits_met") is not True
        or validation.get("minimum_sampled_sphere_clearance_m") is not None
        or validation.get("obstacle_clearance_status")
        != "empty_physical_cell_no_obstacles_modelled"
    ):
        raise ValueError("Ingress trajectory validation changed")

    artifacts = ingress.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Ingress artifacts are missing")
    for label, expected_sha256 in EXPECTED_TOOL_ASSET_SHA256.items():
        record = artifacts.get(label)
        if (
            not isinstance(record, dict)
            or record.get("sha256") != expected_sha256
        ):
            raise ValueError(f"Ingress {label} artifact hash changed")
    live_check = artifacts.get("live_check")
    if (
        not isinstance(live_check, dict)
        or live_check.get("sha256")
        != EXPECTED_INGRESS_SOURCE_LIVE_CHECK_SHA256
        or live_check.get("report_payload_sha256")
        != EXPECTED_INGRESS_SOURCE_LIVE_CHECK_DIGEST
    ):
        raise ValueError("Ingress source live-check hash or digest changed")

    samples = ingress.get("control_samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("Ingress control samples are missing")
    position_rows: list[tuple[float, ...]] = []
    velocity_rows: list[tuple[float, ...]] = []
    prior_time = -1.0
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or set(sample) != {
            "time_seconds",
            "joint_positions",
            "joint_velocities",
        }:
            raise ValueError(f"Ingress sample {index} has changed fields")
        time_s = float(sample["time_seconds"])
        if (
            not math.isfinite(time_s)
            or time_s < 0.0
            or time_s <= prior_time
        ):
            raise ValueError("Ingress times are not strictly increasing")
        if index == 0 and time_s != 0.0:
            raise ValueError("Ingress no longer starts at time zero")
        prior_time = time_s
        position_rows.append(
            _finite_six(sample["joint_positions"], "ingress positions")
        )
        velocity_rows.append(
            _finite_six(sample["joint_velocities"], "ingress velocities")
        )
    if position_rows[0] != start:
        raise ValueError("Ingress sample start changed")
    if position_rows[-1] != READY_JOINT_POSITIONS_RAD:
        raise ValueError("Ingress sample goal changed")
    if any(velocity_rows[0]) or any(velocity_rows[-1]):
        raise ValueError("Ingress endpoint velocity changed")
    numeric_sha256 = _float64_rows_sha256(position_rows + velocity_rows)
    if numeric_sha256 != EXPECTED_INGRESS_NUMERIC_SHA256:
        raise ValueError("Ingress numeric sample hash mismatch")

    candidate = retime_ingress_control_samples(samples)
    validate_retimed_ingress_candidate(candidate, samples)
    return candidate


def _point_from_record(record: Mapping[str, Any]) -> PVTPoint:
    time_ns = record.get("time_from_start_nanoseconds")
    if not isinstance(time_ns, int) or time_ns < 0:
        raise ValueError("Stage point has an invalid nanosecond duration")
    time_s = float(record.get("time_from_start_seconds", math.nan))
    if time_ns / 1_000_000_000 != time_s:
        raise ValueError("Stage point seconds and nanoseconds disagree")
    source_index = record.get("source_sample_index")
    if not isinstance(source_index, int):
        raise ValueError("Stage point source index changed")
    return PVTPoint(
        source_sample_index=source_index,
        time_s=time_s,
        positions=_finite_six(
            record.get("joint_positions_rad"), "stage positions"
        ),
        velocities=_finite_six(
            record.get("joint_velocities_rad_s"), "stage velocities"
        ),
    )


def _stage_points_sha256(points: Sequence[PVTPoint]) -> str:
    digest = hashlib.sha256()
    for point in points:
        digest.update(struct.pack("<q", point.source_sample_index))
        digest.update(
            struct.pack(
                "<q12d",
                round(point.time_s * 1_000_000_000),
                *point.positions,
                *point.velocities,
            )
        )
    return digest.hexdigest()


def _stage_spec(
    *,
    sequence_index: int,
    kind: str,
    specimen_id: int | None,
    stage_index: int,
    stage_name: str,
    record: Mapping[str, Any],
) -> StageSpec:
    point_records = record.get("controller_points")
    message_validation = record.get("message_point_diagnostic_validation")
    wire_validation = record.get("serialized_wire_internal_validation")
    wire_points = record.get("serialized_wire_points")
    wire_tokens_sha256 = record.get("serialized_wire_tokens_sha256")
    if (
        not isinstance(point_records, list)
        or not isinstance(message_validation, dict)
        or not isinstance(wire_validation, dict)
        or not isinstance(wire_points, list)
        or not wire_points
        or not isinstance(wire_points[0], dict)
        or not isinstance(wire_tokens_sha256, str)
        or len(wire_tokens_sha256) != 64
    ):
        raise ValueError(f"{stage_name} is missing points or validation")
    points = tuple(_point_from_record(item) for item in point_records)
    if len(points) < 2 or points[0].time_s != 0.0:
        raise ValueError(f"{stage_name} has invalid controller points")
    if any(points[0].velocities) or any(points[-1].velocities):
        raise ValueError(f"{stage_name} endpoint velocities are not zero")
    return StageSpec(
        sequence_index=sequence_index,
        kind=kind,
        specimen_id=specimen_id,
        stage_index=stage_index,
        stage_name=stage_name,
        points=points,
        position_minimum_rad=_finite_six(
            message_validation.get("position_minimum_rad"),
            f"{stage_name} position minimum",
        ),
        position_maximum_rad=_finite_six(
            message_validation.get("position_maximum_rad"),
            f"{stage_name} position maximum",
        ),
        maximum_cubic_velocity_rad_s=_finite_six(
            wire_validation.get("maximum_cubic_velocity_rad_s"),
            f"{stage_name} velocity maximum",
        ),
        points_sha256=_stage_points_sha256(points),
        first_serialized_wire_point=dict(wire_points[0]),
        serialized_wire_tokens_sha256=wire_tokens_sha256,
    )


def load_execution_bundle(
    retimed_path: Path = DEFAULT_RETIMED_ARTIFACT,
    ingress_path: Path = DEFAULT_INGRESS_ARTIFACT,
) -> ExecutionBundle:
    """Load, hash, and fully re-derive the only accepted execution bundle."""

    normalized_retimed, retimed, retimed_file_sha256 = _read_private_json(
        retimed_path,
        expected_sha256=EXPECTED_RETIMED_FILE_SHA256,
        label="retimed seven-pin artifact",
    )
    normalized_ingress, ingress, ingress_file_sha256 = _read_private_json(
        ingress_path,
        expected_sha256=EXPECTED_INGRESS_FILE_SHA256,
        label="tool-aware ingress artifact",
    )
    if (
        retimed.get(ARTIFACT_DIGEST_FIELD)
        != EXPECTED_RETIMED_PAYLOAD_SHA256
        or retimed.get(ARTIFACT_DIGEST_FIELD) != canonical_digest(retimed)
    ):
        raise ValueError("Retimed artifact payload digest changed")
    source_plan = load_reviewed_plan(DEFAULT_REVIEWED_PLAN)
    if (
        source_plan.get("model_status", {}).get("tool_profile")
        != "watson_qc_nominal"
        or source_plan.get("model_status", {}).get(
            "pin_grasp_tcp_planned_directly"
        )
        is not True
    ):
        raise ValueError("Reviewed source-plan tool profile changed")
    validate_retimed_artifact(retimed, source_plan)
    metrics = retimed.get("metrics")
    if (
        not isinstance(metrics, dict)
        or metrics.get("serialized_wire_numeric_points_float64_sha256")
        != EXPECTED_RETIMED_WIRE_NUMERIC_SHA256
    ):
        raise ValueError("Retimed aggregate wire-numeric SHA-256 changed")
    ingress_candidate = validate_ingress_artifact(ingress)

    stages: list[StageSpec] = [
        _stage_spec(
            sequence_index=0,
            kind="tool_aware_ingress",
            specimen_id=None,
            stage_index=-1,
            stage_name="tool_aware_ready_ingress",
            record=ingress_candidate,
        )
    ]
    stored_stages = retimed.get("stages")
    if not isinstance(stored_stages, list):
        raise ValueError("Retimed seven-pin stages are missing")
    for sequence_index, record in enumerate(stored_stages, start=1):
        if not isinstance(record, dict):
            raise ValueError("Retimed seven-pin stage must be an object")
        stages.append(
            _stage_spec(
                sequence_index=sequence_index,
                kind="seven_pin_air_replay",
                specimen_id=int(record["specimen_id"]),
                stage_index=int(record["stage_index"]),
                stage_name=str(record["stage_name"]),
                record=record,
            )
        )
    if len(stages) != 50:
        raise ValueError("Execution bundle must contain ingress plus 49 stages")
    for previous, current in zip(stages, stages[1:]):
        if previous.goal_positions != current.start_positions:
            raise ValueError(
                f"Stage boundary {previous.stage_name} -> "
                f"{current.stage_name} is not exact"
            )
    if stages[0].goal_positions != READY_JOINT_POSITIONS_RAD:
        raise ValueError("Ingress no longer ends at the reviewed ready pose")
    if stages[-1].goal_positions != READY_JOINT_POSITIONS_RAD:
        raise ValueError("Seven-pin replay no longer finishes at ready")
    return ExecutionBundle(
        retimed_path=normalized_retimed,
        ingress_path=normalized_ingress,
        retimed_file_sha256=retimed_file_sha256,
        ingress_file_sha256=ingress_file_sha256,
        retimed_payload_sha256=EXPECTED_RETIMED_PAYLOAD_SHA256,
        retimed_wire_numeric_sha256=(
            EXPECTED_RETIMED_WIRE_NUMERIC_SHA256
        ),
        ingress_numeric_sha256=EXPECTED_INGRESS_NUMERIC_SHA256,
        stages=tuple(stages),
    )


def validate_execution_authorization(
    *,
    mode: str,
    arm_token: str,
    gripper_token: str = "",
    confirm_cell_clear: bool,
    namespace: str,
) -> None:
    """Require one immutable token and explicit cell-clear confirmation."""

    if mode not in {"check", "dry-run", "execute"}:
        raise ValueError(f"Unsupported air-replay mode: {mode!r}")
    if "/" + namespace.strip("/") != "/watson":
        raise ValueError("Watson air replay is locked to namespace /watson")
    if mode == "execute":
        if arm_token != EXECUTION_ARM_TOKEN:
            raise ValueError(
                "--mode execute requires --arm-token "
                f"{EXECUTION_ARM_TOKEN}"
            )
        if gripper_token != GRIPPER_EXECUTION_TOKEN:
            raise ValueError(
                "--mode execute requires --gripper-token "
                f"{GRIPPER_EXECUTION_TOKEN}"
            )
        if not confirm_cell_clear:
            raise ValueError(
                "--mode execute requires --confirm-cell-clear"
            )
    elif arm_token or gripper_token or confirm_cell_clear:
        raise ValueError(
            "Arming arguments are accepted only with --mode execute"
        )


def exact_tool_audit_failures(audit: Any) -> list[str]:
    """Reject anything except the exact named vendor profile."""

    if not isinstance(audit, dict):
        return ["controller tool audit is missing"]
    failures: list[str] = []
    settings = audit.get("settings")
    if not isinstance(settings, dict):
        failures.append("controller tool settings are missing")
    elif not matches_qc_2fg7_vendor_profile(settings):
        failures.append("active controller tool is not exact QC_2FG7_VENDOR")
    if audit.get("promotion_passed") is not True:
        failures.append("controller tool promotion gate did not pass")
    if audit.get("known_vendor_profile_matched") is not True:
        failures.append("controller did not match the named vendor profile")
    if audit.get("write_items_called") != []:
        failures.append("controller tool audit called a write item")
    if audit.get("motion_commanded") is not False:
        failures.append("controller tool audit reports motion")
    return failures


def live_start_errors(
    snapshot: HealthSnapshot,
    expected_start: Sequence[float],
    *,
    tolerance_rad: float = LIVE_START_TOLERANCE_RAD,
) -> dict[str, float]:
    """Return exact-source errors for the two independent joint feeds."""

    expected = _finite_six(expected_start, "expected live start")
    if (
        len(snapshot.joint_positions) != len(JOINT_NAMES)
        or len(snapshot.feedback_joint_positions) != len(JOINT_NAMES)
    ):
        raise ValueError("Live start does not contain six joints")
    joint_state_error = max(
        abs(snapshot.joint_positions[index] - expected[index])
        for index in range(len(JOINT_NAMES))
    )
    feedback_error = max(
        abs(snapshot.feedback_joint_positions[index] - expected[index])
        for index in range(len(JOINT_NAMES))
    )
    if max(joint_state_error, feedback_error) > tolerance_rad:
        raise ValueError(
            "live start mismatch: "
            f"{max(joint_state_error, feedback_error):.6f}rad > "
            f"{tolerance_rad:.6f}rad"
        )
    return {
        "joint_state_error_rad": joint_state_error,
        "feedback_error_rad": feedback_error,
        "tolerance_rad": tolerance_rad,
    }


def exact_execute_project_speed_failures(
    snapshot: HealthSnapshot,
) -> list[str]:
    """Execution is reviewed only at exactly 50 percent TMflow speed."""

    if snapshot.project_speed == MAX_PROJECT_SPEED:
        return []
    return [
        "set TMflow project speed to 50 before execute "
        f"(observed {snapshot.project_speed})"
    ]


def validate_stage_live_first_wire_cubic(
    snapshot: HealthSnapshot,
    stage: StageSpec,
) -> dict[str, Any]:
    """Prove the driver's actual live-q/v seeded first transmitted cubic."""

    return validate_live_first_wire_cubic(
        snapshot.feedback_joint_positions,
        snapshot.joint_velocities,
        stage.first_serialized_wire_point,
    )


def live_stage_failures(
    snapshot: HealthSnapshot,
    stage: StageSpec,
) -> list[str]:
    """Bound every physical joint by the immutable cubic stage envelope."""

    positions = snapshot.feedback_joint_positions
    velocities = snapshot.joint_velocities
    if len(positions) != len(JOINT_NAMES) or len(velocities) != len(JOINT_NAMES):
        return ["live six-axis stage state is incomplete"]
    if not all(math.isfinite(value) for value in positions + velocities):
        return ["live six-axis stage state contains a non-finite value"]
    failures: list[str] = []
    for joint, joint_name in enumerate(JOINT_NAMES):
        low = max(
            JOINT_POSITION_LOWER_RAD[joint],
            stage.position_minimum_rad[joint]
            - LIVE_POSITION_ENVELOPE_MARGIN_RAD,
        )
        high = min(
            JOINT_POSITION_UPPER_RAD[joint],
            stage.position_maximum_rad[joint]
            + LIVE_POSITION_ENVELOPE_MARGIN_RAD,
        )
        if not low <= positions[joint] <= high:
            failures.append(
                f"{joint_name} live position {positions[joint]:.6f}rad left "
                f"stage envelope [{low:.6f}, {high:.6f}]rad"
            )
        velocity_cap = min(
            JOINT_VELOCITY_LIMITS_RAD_S[joint]
            * DERIVATIVE_LIMIT_FRACTION,
            stage.maximum_cubic_velocity_rad_s[joint]
            + LIVE_VELOCITY_ENVELOPE_MARGIN_RAD_S,
        )
        if abs(velocities[joint]) > velocity_cap:
            failures.append(
                f"{joint_name} live velocity {velocities[joint]:.6f}rad/s "
                f"exceeds stage cap {velocity_cap:.6f}rad/s"
            )
    return failures


def stage_report(stage: StageSpec, status: str) -> dict[str, Any]:
    return {
        "sequence_index": stage.sequence_index,
        "kind": stage.kind,
        "specimen_id": stage.specimen_id,
        "stage_index": stage.stage_index,
        "stage_name": stage.stage_name,
        "controller_point_count_including_zero_seed": len(stage.points),
        "driver_transmitted_point_count": len(stage.points) - 1,
        "duration_seconds": stage.duration_s,
        "start_joint_positions_rad": list(stage.start_positions),
        "goal_joint_positions_rad": list(stage.goal_positions),
        "points_sha256": stage.points_sha256,
        "serialized_wire_tokens_sha256": stage.serialized_wire_tokens_sha256,
        "status": status,
        "gripper": dict(GRIPPER_POLICY),
        "gripper_after_stage_hook": gripper_after_stage_hook(stage),
    }


def gripper_after_stage_hook(stage: StageSpec) -> dict[str, Any] | None:
    """Describe the execute-only gripper transition after a reviewed stage."""

    action = GRIPPER_AFTER_STAGE_HOOKS.get(stage.stage_name)
    if action is None or stage.kind != "seven_pin_air_replay":
        return None
    return {
        "timing": "after_stage",
        "action": action,
        "policy": "execute_only_guarded_compute_box_injection_point",
        "executed": False,
        "actuator_calls": [],
        "target_external_width_mm": 1.0 if action == "close" else 39.0,
        "force_n": 20,
        "speed_percent": 10,
    }


def build_robot_trajectory(
    stage: StageSpec,
    ros_types: Mapping[str, Any],
) -> Any:
    """Build one exact MoveIt RobotTrajectory without acceleration values."""

    trajectory = ros_types["RobotTrajectory"]()
    trajectory.joint_trajectory.joint_names = list(JOINT_NAMES)
    for point in stage.points:
        message = ros_types["JointTrajectoryPoint"]()
        message.positions = list(point.positions)
        message.velocities = list(point.velocities)
        message.accelerations = []
        message.effort = []
        nanoseconds = round(point.time_s * 1_000_000_000)
        message.time_from_start = ros_types["Duration"](
            sec=nanoseconds // 1_000_000_000,
            nanosec=nanoseconds % 1_000_000_000,
        )
        trajectory.joint_trajectory.points.append(message)
    validate_robot_trajectory(stage, trajectory)
    return trajectory


def validate_robot_trajectory(stage: StageSpec, trajectory: Any) -> None:
    """Prove the ROS message still contains the exact reviewed PVT points."""

    if list(trajectory.joint_trajectory.joint_names) != list(JOINT_NAMES):
        raise ValueError("RobotTrajectory joint order changed")
    multi = trajectory.multi_dof_joint_trajectory
    if list(multi.joint_names) or list(multi.points):
        raise ValueError("RobotTrajectory contains a multi-DOF command")
    messages = list(trajectory.joint_trajectory.points)
    if len(messages) != len(stage.points):
        raise ValueError("RobotTrajectory point count changed")
    for index, (expected, message) in enumerate(zip(stage.points, messages)):
        if tuple(float(value) for value in message.positions) != expected.positions:
            raise ValueError(f"RobotTrajectory point {index} positions changed")
        if tuple(float(value) for value in message.velocities) != expected.velocities:
            raise ValueError(f"RobotTrajectory point {index} velocities changed")
        if list(message.accelerations):
            raise ValueError(
                f"RobotTrajectory point {index} invents acceleration values"
            )
        if list(message.effort):
            raise ValueError(f"RobotTrajectory point {index} contains effort")
        observed_nanoseconds = (
            int(message.time_from_start.sec) * 1_000_000_000
            + int(message.time_from_start.nanosec)
        )
        expected_nanoseconds = round(expected.time_s * 1_000_000_000)
        if observed_nanoseconds != expected_nanoseconds:
            raise ValueError(f"RobotTrajectory point {index} time changed")
