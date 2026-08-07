#!/usr/bin/env python3
"""Plan a tool-aware, offline Watson ingress from a captured pose to demo ready.

The script consumes a private schema-8 read-only Watson check, but creates no
ROS graph, opens no network connection, and cannot command the physical robot.
It validates a deterministic straight c-space ingress with the commissioned
QC-R + inward 2FG7 cuMotion model and exports controller-independent samples
for a separate Techman retiming/guard step.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import stat
import sys
from typing import Any

import cumotion
import numpy as np

from plan_synthetic_pick import (
    EXPECTED_CUMOTION_VERSION,
    float64_sha256,
    maximum_control_step,
    samples_for_trajectory,
    sha256_file,
    validate_trajectory,
)


ARENA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_URDF = (
    ARENA_DIR
    / "generated/tool_profiles/watson_qc_nominal/cumotion/tm5s_with_2fg7.urdf"
)
DEFAULT_XRDF = (
    ARENA_DIR
    / "generated/tool_profiles/watson_qc_nominal/cumotion/tm5s_with_2fg7.xrdf"
)
DEFAULT_ASSET_MANIFEST = (
    ARENA_DIR
    / "generated/tool_profiles/watson_qc_nominal/cumotion/asset_manifest.json"
)
DEFAULT_OUTPUT = (
    ARENA_DIR
    / "outputs/watson_guarded_demo/20260723T_tool_aware_ready_ingress.json"
)
EXPECTED_ARTIFACT_SHA256 = {
    "urdf": "ee7dadbee3e898152948c133f859f7bd085c93614fc8274549158cca10a18d03",
    "xrdf": "41f1575758b5ee65b6d337c7159e6f3bb70eae3a32863f3dac58e727c859d5df",
    "asset_manifest": (
        "60835ef5ca0c6212633864b610be5fd4a222ffc6e84732b0760c939d61b8b993"
    ),
}
EXPECTED_JOINT_NAMES = [f"joint_{index}" for index in range(1, 7)]
EXPECTED_READY = np.asarray([0.0, 0.0, 1.5708, 0.0, 1.5708, 0.0])
EXPECTED_TOOL_SETTINGS = {
    "active_tcp_name": "QC_2FG7_VENDOR",
    "tcp_value": [0.0, 0.0, 138.6, 0.0, 0.0, 0.0],
    "mass_kg": 1.2,
    "principal_moi": [0.0, 0.0, 0.0],
    "mass_centre_frame": [0.0, 0.0, 62.52, 0.0, 0.0, 0.0],
}
REPORT_DIGEST_FIELD = "report_payload_sha256"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--live-check", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--xrdf", type=Path, default=DEFAULT_XRDF)
    parser.add_argument("--asset-manifest", type=Path, default=DEFAULT_ASSET_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--control-dt-seconds", type=float, default=1.0 / 300.0)
    parser.add_argument("--validation-dt-seconds", type=float, default=1.0 / 600.0)
    parser.add_argument(
        "--execution-time-scale",
        type=float,
        default=1.5,
        help="Slow the time-optimal offline ingress before Techman retiming.",
    )
    parser.add_argument("--maximum-control-step-rad", type=float, default=0.01)
    return parser


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


def private_json(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or path.is_symlink()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
    ):
        raise ValueError("live check must be a private 0600 regular file owned by this user")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("live check must contain one JSON object")
    return value


def finite_joint_vector(value: Any, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain six finite joint values")
    return vector


def require_close(actual: Any, expected: Any, label: str, tolerance: float) -> None:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if (
        actual_array.shape != expected_array.shape
        or not np.all(np.isfinite(actual_array))
        or not np.allclose(actual_array, expected_array, rtol=0.0, atol=tolerance)
    ):
        raise ValueError(f"live check {label} does not match the commissioned profile")


def validate_live_check(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    report = private_json(path)
    if (
        report.get("schema_version") != 8
        or report.get("mode") != "check"
        or report.get("status") != "check_passed"
        or report.get("motion_commanded") is not False
        or report.get("health_failures") != []
    ):
        raise ValueError("live check is not a passing schema-8 read-only report")
    if report.get(REPORT_DIGEST_FIELD) != canonical_digest(report):
        raise ValueError("live check payload digest is invalid")
    if report.get("controller_tool_settings_promotion_passed") is not True:
        raise ValueError("live check does not promote the commissioned controller tool")
    audit = report.get("controller_tool_audit")
    if not isinstance(audit, dict) or audit.get("promotion_passed") is not True:
        raise ValueError("live check controller tool audit did not pass")
    settings = audit.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("live check controller tool settings are missing")
    if settings.get("active_tcp_name") != EXPECTED_TOOL_SETTINGS["active_tcp_name"]:
        raise ValueError("live check active tool name is not QC_2FG7_VENDOR")
    require_close(
        settings.get("tcp_value"),
        EXPECTED_TOOL_SETTINGS["tcp_value"],
        "TCP",
        0.01,
    )
    require_close(
        [settings.get("mass_kg")],
        [EXPECTED_TOOL_SETTINGS["mass_kg"]],
        "mass",
        0.001,
    )
    require_close(
        settings.get("principal_moi"),
        EXPECTED_TOOL_SETTINGS["principal_moi"],
        "principal inertia",
        1.0e-12,
    )
    require_close(
        settings.get("mass_centre_frame"),
        EXPECTED_TOOL_SETTINGS["mass_centre_frame"],
        "mass centre",
        0.01,
    )
    health = report.get("stable_health")
    if not isinstance(health, dict):
        raise ValueError("live check stable health is missing")
    required_true = (
        "is_svr_connected",
        "is_sct_connected",
        "is_data_table_correct",
        "robot_link",
        "project_run",
    )
    required_false = ("robot_error", "project_pause", "safetyguard_a", "e_stop")
    if (
        any(health.get(field) is not True for field in required_true)
        or any(health.get(field) is not False for field in required_false)
        or health.get("error_code") != 0
    ):
        raise ValueError("live check health is not suitable for ingress planning")
    return report, finite_joint_vector(
        health.get("feedback_joint_positions"),
        "live feedback joints",
    )


def artifact(path: Path, label: str) -> dict[str, Any]:
    digest = sha256_file(path)
    if digest != EXPECTED_ARTIFACT_SHA256[label]:
        raise ValueError(f"{label} artifact hash changed")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
    }


def write_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> int:
    args = build_parser().parse_args()
    for field in ("live_check", "urdf", "xrdf", "asset_manifest", "output"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite ingress plan: {args.output}")
    for value, label in (
        (args.control_dt_seconds, "control dt"),
        (args.validation_dt_seconds, "validation dt"),
        (args.maximum_control_step_rad, "maximum control step"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be finite and positive")
    if (
        not math.isfinite(args.execution_time_scale)
        or args.execution_time_scale < 1.0
    ):
        raise ValueError("execution time scale must be finite and at least one")
    if getattr(cumotion, "__version__", None) != EXPECTED_CUMOTION_VERSION:
        raise RuntimeError(
            f"expected cuMotion {EXPECTED_CUMOTION_VERSION}; found "
            f"{getattr(cumotion, '__version__', 'unknown')}"
        )

    live_report, start = validate_live_check(args.live_check)
    artifacts = {
        "urdf": artifact(args.urdf, "urdf"),
        "xrdf": artifact(args.xrdf, "xrdf"),
        "asset_manifest": artifact(args.asset_manifest, "asset_manifest"),
        "live_check": {
            "path": str(args.live_check),
            "size_bytes": args.live_check.stat().st_size,
            "sha256": sha256_file(args.live_check),
            "report_payload_sha256": live_report[REPORT_DIGEST_FIELD],
        },
    }

    cumotion.set_log_level(cumotion.LogLevel.ERROR)
    robot = cumotion.load_robot_from_file(str(args.xrdf), str(args.urdf))
    if robot.num_cspace_coords() != 6:
        raise RuntimeError("ingress model must contain six c-space coordinates")
    kinematics = robot.kinematics()
    joint_names = [
        kinematics.cspace_coord_name(index)
        for index in range(kinematics.num_cspace_coords())
    ]
    if joint_names != EXPECTED_JOINT_NAMES:
        raise RuntimeError(f"unexpected ingress joint order: {joint_names}")

    world = cumotion.create_world()
    inspector = cumotion.create_robot_world_inspector(robot, world.add_world_view())
    if inspector.in_self_collision(start):
        raise RuntimeError("captured live configuration is in tool-aware self collision")
    if inspector.in_self_collision(EXPECTED_READY):
        raise RuntimeError("demo ready configuration is in tool-aware self collision")
    linear_collision_indices = [
        index
        for index, alpha in enumerate(np.linspace(0.0, 1.0, 1001))
        if inspector.in_self_collision(start + alpha * (EXPECTED_READY - start))
    ]
    if linear_collision_indices:
        raise RuntimeError(
            "deterministic straight ingress is in tool-aware self collision"
        )

    generator = cumotion.create_cspace_trajectory_generator(kinematics)
    trajectory = generator.generate_trajectory([start, EXPECTED_READY])
    if trajectory is None:
        raise RuntimeError("cuMotion could not time-parameterize the straight ingress")
    validation = validate_trajectory(
        trajectory,
        inspector,
        kinematics,
        args.validation_dt_seconds,
        args.execution_time_scale,
    )
    if validation["sampled_self_collision"] or not validation["derivative_limits_met"]:
        raise RuntimeError("tool-aware ingress trajectory validation failed")
    validation["minimum_sampled_sphere_clearance_m"] = None
    validation["obstacle_clearance_status"] = "empty_physical_cell_no_obstacles_modelled"

    samples = samples_for_trajectory(
        trajectory,
        args.control_dt_seconds,
        args.execution_time_scale,
    )
    samples[0]["joint_positions"] = start.tolist()
    samples[0]["joint_velocities"] = [0.0] * 6
    samples[-1]["joint_positions"] = EXPECTED_READY.tolist()
    samples[-1]["joint_velocities"] = [0.0] * 6
    maximum_step = maximum_control_step(samples)
    if maximum_step > args.maximum_control_step_rad + 1.0e-12:
        raise RuntimeError(
            f"ingress control step {maximum_step} exceeds "
            f"{args.maximum_control_step_rad}"
        )

    payload: dict[str, Any] = {
        "format_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "offline_tool_aware_ingress_validated_not_executable",
        "joint_names": joint_names,
        "start_joint_positions": start.tolist(),
        "ready_joint_positions": EXPECTED_READY.tolist(),
        "maximum_joint_displacement_rad": float(
            np.max(np.abs(EXPECTED_READY - start))
        ),
        "path_mode": "deterministic_straight_cspace",
        "tool_profile": "watson_qc_nominal",
        "controller_tool_profile": EXPECTED_TOOL_SETTINGS,
        "linear_self_collision_samples": 1001,
        "linear_self_collision_count": 0,
        "control_dt_seconds": args.control_dt_seconds,
        "execution_time_scale": args.execution_time_scale,
        "maximum_control_step_rad": maximum_step,
        "control_samples": samples,
        "control_samples_float64_sha256": float64_sha256(
            [
                np.asarray(sample["joint_positions"], dtype=np.float64)
                for sample in samples
            ]
            + [
                np.asarray(sample["joint_velocities"], dtype=np.float64)
                for sample in samples
            ]
        ),
        "trajectory_validation": validation,
        "artifacts": artifacts,
        "scope": {
            "ros_graph_created": False,
            "network_connection_opened": False,
            "watson_connected": False,
            "controller_trajectory_created": False,
            "real_robot_commanded": False,
            "gripper_commanded": False,
            "workcell_obstacles_modelled": False,
        },
        "provenance": {
            "command": [sys.executable, *sys.argv],
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "cumotion_version": cumotion.__version__,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    write_private_json(args.output, payload)
    print(f"Tool-aware straight ingress: PASS ({len(samples)} source samples)")
    print(
        "Duration: "
        f"{validation['duration_seconds']:.6f}s; max step: {maximum_step:.9f}rad"
    )
    print(f"Ingress artifact: {args.output}")
    print("ROS used: false; Watson connected: false; real robot commanded: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
