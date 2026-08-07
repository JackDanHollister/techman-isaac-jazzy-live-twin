#!/usr/bin/env python3
"""Freeze a non-executable Watson arm intent from the seven-pin Isaac plan.

This tool consumes an already captured schema-8 read-only Watson check.  It
creates no ROS graph, opens no network connection, and deliberately does not
emit a controller trajectory.  The resulting private manifest records the
work still required before a separately implemented six-axis guard may arm.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any

import numpy as np
import yaml


ARENA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ARENA_DIR / "config/isaac_multi_pin_verticalization.yaml"
EXPECTED_PLAN_SHA256 = (
    "0c5cc66bd54510bc92e92fefc727555abfdf157e541e67755cdccaea565b2c5a"
)
EXPECTED_SAMPLE_SHA256 = (
    "ca8005329f64baf6e205813cece9801ce8f779862dbcd4dc4b0757e1bfcce8bf"
)
EXPECTED_JOINT_NAMES = [f"joint_{index}" for index in range(1, 7)]
EXPECTED_SPECIMEN_IDS = list(range(1, 8))
EXPECTED_STAGE_NAMES = [
    "approach_tilted_pregrasp",
    "descend_tilted_grasp",
    "lift_tilted",
    "reorient_vertical",
    "descend_vertical",
    "retreat_vertical",
    "return_ready",
]
EXPECTED_READY = np.asarray([0.0, 0.0, 1.5708, 0.0, 1.5708, 0.0])
REPORT_DIGEST_FIELD = "report_payload_sha256"
MANIFEST_STATUS = "blocked_pending_tool_commissioning_and_physical_retime"
MAX_LIVE_CHECK_BYTES = 8 * 1024 * 1024
MAX_LIVE_CHECK_AGE_SECONDS = 30 * 60
EXPECTED_LIVE_PROVENANCE = {
    "namespace": "/watson",
    "robot_ip": os.environ.get("TECHMAN_ROBOT_IP", "192.0.2.23"),
    "robot_interface": os.environ.get(
        "TECHMAN_ROBOT_INTERFACE", "enp1s0"
    ),
    "robot_source_ip": os.environ.get(
        "TECHMAN_ROBOT_SOURCE_IP", "192.0.2.100"
    ),
    "robot_mac": os.environ.get(
        "TECHMAN_ROBOT_MAC", "02:00:00:00:00:23"
    ).lower(),
}
SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
EXPECTED_RUNNER_SHA256 = hashlib.sha256(
    (ARENA_DIR / "scripts/run_watson_guarded_demo.py").read_bytes()
).hexdigest()
EXPECTED_GUARD_SHA256 = hashlib.sha256(
    (ARENA_DIR / "pin_axis_3d_sim/watson_guard.py").read_bytes()
).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(payload: dict[str, Any]) -> str:
    copy = dict(payload)
    copy.pop(REPORT_DIGEST_FIELD, None)
    return hashlib.sha256(
        json.dumps(
            copy,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def float64_sha256(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.asarray(array, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def finite_joint_vector(value: Any, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain six finite joint values")
    return vector


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def read_private_json(path: Path, label: str) -> dict[str, Any]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    if stat.S_IMODE(before.st_mode) != 0o600:
        raise ValueError(f"{label} must be private mode 0600: {path}")
    if before.st_uid != os.geteuid() or before.st_nlink != 1:
        raise ValueError(f"{label} must be owned by this user with one hard link")
    if before.st_size <= 0 or before.st_size > MAX_LIVE_CHECK_BYTES:
        raise ValueError(f"{label} has an invalid file size")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"{label} changed while it was opened")
        chunks: list[bytes] = []
        remaining = MAX_LIVE_CHECK_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    if len(data) > MAX_LIVE_CHECK_BYTES:
        raise ValueError(f"{label} exceeds the maximum accepted size")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot parse {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def validate_live_check(
    path: Path,
    *,
    now: datetime | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    report = read_private_json(path, "Live check")
    if report.get("schema_version") != 8:
        raise ValueError("Live check must use guarded-report schema 8")
    if report.get("mode") != "check" or report.get("status") != "check_passed":
        raise ValueError("Live evidence must be a passing read-only check report")
    if report.get("motion_commanded") is not False:
        raise ValueError("Live check must record motion_commanded false")
    if report.get(REPORT_DIGEST_FIELD) != canonical_digest(report):
        raise ValueError("Live check payload digest does not match its contents")
    for field, expected in EXPECTED_LIVE_PROVENANCE.items():
        if report.get(field) != expected:
            raise ValueError(f"Live check {field} does not identify Watson")
    if report.get("runner_source_sha256") != EXPECTED_RUNNER_SHA256:
        raise ValueError("Live check was not produced by the current guarded runner")
    if report.get("guard_source_sha256") != EXPECTED_GUARD_SHA256:
        raise ValueError("Live check was not produced with the current Watson guard")
    timestamp_value = report.get("timestamp_utc")
    try:
        timestamp = datetime.fromisoformat(str(timestamp_value))
    except ValueError as exc:
        raise ValueError("Live check timestamp is invalid") from exc
    if timestamp.tzinfo is None:
        raise ValueError("Live check timestamp must include a timezone")
    reference = now or datetime.now(timezone.utc)
    age_seconds = (reference.astimezone(timezone.utc) - timestamp.astimezone(timezone.utc)).total_seconds()
    if age_seconds < -5.0 or age_seconds > MAX_LIVE_CHECK_AGE_SECONDS:
        raise ValueError(
            f"Live check is stale or future-dated ({age_seconds:.1f} seconds old)"
        )
    if report.get("health_failures") != []:
        raise ValueError("Live check contains health failures")
    health = report.get("stable_health")
    if not isinstance(health, dict):
        raise ValueError("Live check is missing stable_health")
    required_true = (
        "is_svr_connected",
        "is_sct_connected",
        "is_data_table_correct",
        "robot_link",
        "project_run",
    )
    required_false = ("robot_error", "project_pause", "safetyguard_a", "e_stop")
    if any(health.get(field) is not True for field in required_true):
        raise ValueError("Live check lost a required healthy/Listen state")
    if any(health.get(field) is not False for field in required_false):
        raise ValueError("Live check reports a robot, pause, safeguard, or E-stop state")
    if health.get("error_code") != 0:
        raise ValueError("Live check reports a non-zero controller error")
    if report.get("controller_tool_settings_promotion_passed") is not False:
        raise ValueError("This blocked intent requires the observed uncommissioned tool")
    audit = report.get("controller_tool_audit")
    if not isinstance(audit, dict) or audit.get("promotion_passed") is not False:
        raise ValueError("Live check must contain the blocked controller tool audit")
    settings = audit.get("settings", {})
    if settings.get("active_tcp_name") != "RobotEndFlange":
        raise ValueError("Expected the observed bare RobotEndFlange controller record")
    if settings.get("mass_kg") != 0.0:
        raise ValueError("Expected the observed zero-mass controller tool record")
    if any(settings.get(field) != [0.0] * size for field, size in (
        ("tcp_value", 6),
        ("principal_moi", 3),
        ("mass_centre_frame", 6),
    )):
        raise ValueError("Expected the observed zeroed TCP/MOI/CoG controller record")
    joints = finite_joint_vector(health.get("feedback_joint_positions"), "live joints")
    return report, joints


def resolve_plan(config_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Multi-pin config must be a YAML object")
    scope = config.get("scope", {})
    for field in ("ros_used", "watson_connected", "real_robot_commanded"):
        if scope.get(field) is not False:
            raise ValueError(f"Multi-pin config must keep {field} false")
    binding = config.get("multi_pin_plan", {})
    if binding.get("sha256") != EXPECTED_PLAN_SHA256:
        raise ValueError("Multi-pin config no longer pins the reviewed plan hash")
    plan_path = Path(str(binding.get("path", "")))
    if not plan_path.is_absolute():
        plan_path = ARENA_DIR / plan_path
    plan_path = plan_path.resolve()
    if sha256_file(plan_path) != EXPECTED_PLAN_SHA256:
        raise ValueError("Reviewed multi-pin plan file hash mismatch")
    plan = read_json(plan_path, "multi-pin plan")
    return config, plan_path, plan


def validate_plan(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if plan.get("format_version") != 1 or plan.get("frame_id") != "base":
        raise ValueError("Reviewed plan format/base frame mismatch")
    if plan.get("planning_tool_frame") != "pin_grasp_tcp":
        raise ValueError("Reviewed plan must target pin_grasp_tcp")
    for field in ("ros_used", "watson_connected", "real_robot_commanded"):
        if plan.get(field) is not False:
            raise ValueError(f"Reviewed plan must keep {field} false")
    if plan.get("joint_names") != EXPECTED_JOINT_NAMES:
        raise ValueError("Reviewed plan joint order mismatch")
    if plan.get("specimen_ids") != EXPECTED_SPECIMEN_IDS:
        raise ValueError("Reviewed plan specimen order mismatch")
    ready = finite_joint_vector(plan.get("ready_joint_positions"), "ready pose")
    if not np.array_equal(ready, EXPECTED_READY):
        raise ValueError("Reviewed plan ready pose changed")
    specimens = plan.get("specimens")
    if not isinstance(specimens, list) or len(specimens) != 7:
        raise ValueError("Reviewed plan must contain seven specimens")
    top_validation = plan.get("validation", {})
    required_clearance = float(plan.get("required_sampled_sphere_clearance_m", 0.0))
    minimum_clearance = float(
        top_validation.get("minimum_sampled_sphere_clearance_m", 0.0)
    )
    if (
        top_validation.get("all_stages_accepted") is not True
        or top_validation.get("sampled_self_collision") is not False
        or top_validation.get("derivative_limits_met") is not True
        or top_validation.get("cycles_ready_to_ready") is not True
        or top_validation.get("adjacent_stage_positions_continuous") is not True
        or required_clearance < 0.004
        or minimum_clearance < required_clearance
    ):
        raise ValueError("Reviewed plan top-level acceptance/clearance gates failed")

    summaries: list[dict[str, Any]] = []
    position_arrays: list[np.ndarray] = []
    velocity_arrays: list[np.ndarray] = []
    peak_velocity = np.zeros(6)
    peak_acceleration = np.zeros(6)
    previous = ready.copy()
    total_samples = 0
    source_duration = 0.0
    for specimen_id, specimen in zip(EXPECTED_SPECIMEN_IDS, specimens):
        if specimen.get("specimen_id") != specimen_id:
            raise ValueError("Reviewed plan specimen IDs changed")
        stages = specimen.get("stages")
        if not isinstance(stages, list) or [stage.get("name") for stage in stages] != EXPECTED_STAGE_NAMES:
            raise ValueError(f"Specimen {specimen_id} stage order changed")
        for stage_index, stage in enumerate(stages):
            stage_validation = stage.get("trajectory_validation", {})
            if (
                stage.get("accepted") is not True
                or stage.get("path_found") is not True
                or stage.get("goal_tolerance_met") is not True
                or stage_validation.get("sampled_self_collision") is not False
                or stage_validation.get("derivative_limits_met") is not True
                or float(stage_validation.get("minimum_sampled_sphere_clearance_m", 0.0))
                < required_clearance
            ):
                raise ValueError("Reviewed plan contains a failed stage gate")
            start = finite_joint_vector(stage.get("start_joint_positions"), "stage start")
            end = finite_joint_vector(stage.get("end_joint_positions"), "stage end")
            if not np.array_equal(start, previous):
                raise ValueError("Reviewed plan lost exact stage continuity")
            samples = stage.get("control_samples")
            if not isinstance(samples, list) or len(samples) < 2:
                raise ValueError("Reviewed plan stage has insufficient samples")
            positions = np.asarray([sample.get("joint_positions") for sample in samples], dtype=np.float64)
            velocities = np.asarray([sample.get("joint_velocities") for sample in samples], dtype=np.float64)
            times = np.asarray([sample.get("time_seconds") for sample in samples], dtype=np.float64)
            if positions.shape != (len(samples), 6) or velocities.shape != positions.shape:
                raise ValueError("Reviewed plan sample shape changed")
            if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)) or not np.all(np.isfinite(times)):
                raise ValueError("Reviewed plan contains a non-finite sample")
            if times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
                raise ValueError("Reviewed plan sample times are not strictly increasing")
            if not np.array_equal(positions[0], start) or not np.array_equal(positions[-1], end):
                raise ValueError("Reviewed plan stage sample endpoints changed")
            stage_hash = float64_sha256(
                [row for row in positions] + [row for row in velocities]
            )
            if stage_hash != stage.get("control_samples_float64_sha256"):
                raise ValueError("Reviewed plan stage numeric hash mismatch")
            acceleration = np.asarray(stage_validation.get("maximum_acceleration_rad_s2"), dtype=np.float64)
            if acceleration.shape != (6,) or not np.all(np.isfinite(acceleration)):
                raise ValueError("Reviewed plan acceleration evidence changed")
            peak_velocity = np.maximum(peak_velocity, np.max(np.abs(velocities), axis=0))
            peak_acceleration = np.maximum(peak_acceleration, acceleration)
            position_arrays.extend(row for row in positions)
            velocity_arrays.extend(row for row in velocities)
            total_samples += len(samples)
            source_duration += float(times[-1])
            summaries.append(
                {
                    "specimen_id": specimen_id,
                    "stage_index": stage_index,
                    "stage_name": stage["name"],
                    "sample_count": len(samples),
                    "source_duration_seconds": float(times[-1]),
                    "start_joint_positions_rad": start.tolist(),
                    "end_joint_positions_rad": end.tolist(),
                    "positions_and_velocities_float64_sha256": stage_hash,
                    "controller_trajectory_created": False,
                }
            )
            previous = end
        if not np.array_equal(previous, ready):
            raise ValueError(f"Specimen {specimen_id} no longer returns exactly ready")

    aggregate_hash = float64_sha256(position_arrays + velocity_arrays)
    validation = top_validation
    if len(summaries) != 49 or total_samples != 18102:
        raise ValueError("Reviewed plan stage/sample count changed")
    if aggregate_hash != EXPECTED_SAMPLE_SHA256 or validation.get("control_samples_float64_sha256") != aggregate_hash:
        raise ValueError("Reviewed plan aggregate numeric hash mismatch")
    metrics = {
        "stage_count": len(summaries),
        "control_sample_count": total_samples,
        "control_dt_seconds": float(plan["control_dt_seconds"]),
        "source_stage_motion_duration_seconds": source_duration,
        "maximum_source_velocity_rad_s": peak_velocity.tolist(),
        "maximum_source_acceleration_rad_s2": peak_acceleration.tolist(),
        "maximum_observed_control_step_rad": float(validation["maximum_observed_control_step_rad"]),
        "minimum_sampled_robot_sphere_clearance_m": float(validation["minimum_sampled_sphere_clearance_m"]),
        "control_samples_float64_sha256": aggregate_hash,
    }
    return summaries, metrics


def build_manifest(
    config_path: Path,
    live_check_path: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    config, plan_path, plan = resolve_plan(config_path)
    stage_summaries, metrics = validate_plan(plan)
    live_report, live_joints = validate_live_check(live_check_path, now=now)
    ready = EXPECTED_READY.copy()
    delta = ready - live_joints
    blockers = [
        "live controller tool is the zeroed RobotEndFlange record; commission and reread the physical QC+2FG7 TCP, mass, MOI, and CoG",
        "the 300 Hz Isaac samples require controller-compatible >=25 ms PVT retiming/filter emulation and post-filter validation",
        "a new immutable six-axis guard is required; the proven J6-only guard must not be widened",
        "live-start to Isaac-ready ingress and ready-to-captured-start egress are unplanned and require separate review/authorization",
        "the live MoveIt scene must include the commissioned QC+2FG7 and confirmed physical clock registration",
        "no verified physical OnRobot 2FG7 command path exists; this intent is arm-only",
        "matching the Isaac presentation speed requires a separate bounded physical speed qualification",
        "real table/base registration and unmodelled EIH camera/cable geometry are not validated",
    ]
    manifest: dict[str, Any] = {
        "format_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": MANIFEST_STATUS,
        "mode": "offline_non_executable_arm_intent",
        "script_sha256": SCRIPT_SHA256,
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "source_plan": str(plan_path),
        "source_plan_sha256": EXPECTED_PLAN_SHA256,
        "source_numeric_sample_sha256": EXPECTED_SAMPLE_SHA256,
        "source_plan_metrics": metrics,
        "live_check": str(live_check_path.resolve()),
        "live_check_sha256": sha256_file(live_check_path),
        "live_check_payload_sha256": live_report[REPORT_DIGEST_FIELD],
        "live_check_status": live_report["status"],
        "live_project_speed": live_report["stable_health"].get("project_speed"),
        "live_controller_tool_promotion_passed": False,
        "captured_live_joint_positions_rad": live_joints.tolist(),
        "isaac_ready_joint_positions_rad": ready.tolist(),
        "live_to_ready_delta_rad": delta.tolist(),
        "maximum_live_to_ready_delta_rad": float(np.max(np.abs(delta))),
        "ingress_intent": {
            "from": "captured_live_joint_positions",
            "to": "isaac_ready_joint_positions",
            "status": "unplanned_not_a_controller_trajectory",
        },
        "reviewed_isaac_stage_intents": stage_summaries,
        "egress_intent": {
            "from": "isaac_ready_joint_positions",
            "to": "captured_live_joint_positions",
            "status": "unplanned_not_a_controller_trajectory",
        },
        "blockers": blockers,
        "stage_count": len(stage_summaries),
        "commands_gripper": False,
        "controller_trajectory_created": False,
        "command_path_created": False,
        "ros_used": False,
        "watson_connected": False,
        "network_connection_opened": False,
        "real_robot_commanded": False,
        "motion_commanded": False,
        "execution_authorized": False,
        "arm_token_accepted": False,
        "warning": "This manifest is evidence of blocked preparation, not permission or input for robot execution.",
    }
    manifest[REPORT_DIGEST_FIELD] = canonical_digest(manifest)
    return manifest


def write_private_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.parent.resolve() / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(manifest, stream, indent=2, allow_nan=False)
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
                raise RuntimeError("Blocked intent manifest failed private-file checks")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    final = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(final.st_mode)
        or stat.S_IMODE(final.st_mode) != 0o600
        or final.st_uid != os.geteuid()
        or final.st_nlink != 1
    ):
        path.unlink(missing_ok=True)
        raise RuntimeError("Blocked intent manifest failed final private-file checks")
    directory_fd = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--live-check", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = build_manifest(args.config.resolve(), args.live_check.resolve())
    write_private_manifest(args.output, manifest)
    print(f"Status: {manifest['status']}")
    print(f"Reviewed arm stages: {manifest['stage_count']}")
    print(f"Maximum live-to-ready delta: {manifest['maximum_live_to_ready_delta_rad']:.6f} rad")
    print("Motion commanded: false")
    print(f"Manifest: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
