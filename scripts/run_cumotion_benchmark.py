#!/usr/bin/env python3
"""Benchmark standalone NVIDIA cuMotion against deterministic TM5S targets.

This runner is camera-free, ROS-free, and plan-only.  It never connects to or
commands Watson. Pose goals are either explicit synthetic targets or are produced
from known joint configurations via cuMotion forward kinematics. Errors are
measured at the deliberately limited ``flange`` tool frame.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cumotion
import numpy as np
import yaml


ARENA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ARENA_DIR / "generated/cumotion"
DEFAULT_PLANNER_CONFIG = ARENA_DIR / "config/tm5s_cumotion_planner.yaml"
DEFAULT_CASES = ARENA_DIR / "config/cumotion_benchmark_cases.yaml"
JOINT_NAMES = [f"joint_{index}" for index in range(1, 7)]
TOOL_FRAME = "flange"
# The denser independent audit found up to 1.573 mm of sampled mesh outside the
# sphere union. Require a rounded-up 2 mm world-clearance reserve so a merely
# positive sphere distance is never accepted as sufficient by this runner.
REQUIRED_SAMPLED_SPHERE_CLEARANCE_M = 0.002


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_MODEL_DIR / "tm5s_with_2fg7.urdf")
    parser.add_argument("--xrdf", type=Path, default=DEFAULT_MODEL_DIR / "tm5s_with_2fg7.xrdf")
    parser.add_argument("--planner-config", type=Path, default=DEFAULT_PLANNER_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--case",
        action="append",
        dest="selected_cases",
        help="Run only this named case; may be supplied more than once.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Result directory (default: outputs/cumotion_benchmark/<UTC timestamp>).",
    )
    return parser


def read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def validate_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required benchmark files are missing:\n  " + "\n  ".join(missing))


def vector(values: Any, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (6,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain six finite joint values")
    return result


def validate_cases(raw_cases: Any, selected: list[str] | None) -> list[dict[str, Any]]:
    if not isinstance(raw_cases, dict) or not isinstance(raw_cases.get("cases"), list):
        raise ValueError("Benchmark case file must contain a top-level 'cases' list")

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_case in raw_cases["cases"]:
        if not isinstance(raw_case, dict):
            raise ValueError("Each benchmark case must be a mapping")
        name = str(raw_case.get("name", ""))
        if not name or name in seen:
            raise ValueError(f"Case names must be present and unique: {name!r}")
        seen.add(name)
        if selected and name not in selected:
            continue
        mode = raw_case.get("mode")
        if mode not in {"cspace", "pose"}:
            raise ValueError(f"Case {name}: mode must be 'cspace' or 'pose'")
        case = dict(raw_case)
        case["start_joint_positions"] = vector(
            raw_case.get("start_joint_positions"), f"{name}.start_joint_positions"
        )
        if "target_joint_positions" in raw_case:
            case["target_joint_positions"] = vector(
                raw_case.get("target_joint_positions"), f"{name}.target_joint_positions"
            )
        elif mode == "pose" and isinstance(raw_case.get("target_pose"), dict):
            pose = raw_case["target_pose"]
            position = np.asarray(pose.get("position_xyz"), dtype=np.float64)
            quaternion = np.asarray(pose.get("quaternion_xyzw"), dtype=np.float64)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError(f"{name}.target_pose.position_xyz must contain three values")
            if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
                raise ValueError(f"{name}.target_pose.quaternion_xyzw must contain four values")
            quaternion_norm = float(np.linalg.norm(quaternion))
            if not np.isclose(quaternion_norm, 1.0, atol=1e-4):
                raise ValueError(f"{name}.target_pose quaternion must be normalized")
            case["target_pose"] = {
                "position_xyz": position,
                "quaternion_xyzw": quaternion / quaternion_norm,
            }
            case["target_joint_positions"] = None
        else:
            raise ValueError(
                f"Case {name}: provide target_joint_positions or, for pose mode, target_pose"
            )
        if raw_case.get("expected", "success") not in {"success", "failure"}:
            raise ValueError(f"Case {name}: expected must be success or failure")
        cases.append(case)

    if selected:
        unknown = sorted(set(selected) - seen)
        if unknown:
            raise ValueError(f"Unknown selected case(s): {', '.join(unknown)}")
    if not cases:
        raise ValueError("No benchmark cases selected")
    return cases


def add_obstacles(world: Any, case: dict[str, Any]) -> int:
    count = 0
    for obstacle_data in case.get("obstacles", []):
        obstacle_type = obstacle_data.get("type")
        position = np.asarray(obstacle_data.get("position"), dtype=np.float64)
        if position.shape != (3,):
            raise ValueError(f"{case['name']}: obstacle position must contain three values")
        pose = cumotion.Pose3.from_translation(position)
        if obstacle_type == "cuboid":
            obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
            side_lengths = np.asarray(obstacle_data.get("side_lengths"), dtype=np.float64)
            if side_lengths.shape != (3,) or np.any(side_lengths <= 0.0):
                raise ValueError(f"{case['name']}: cuboid side_lengths must be positive")
            obstacle.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS, side_lengths)
        elif obstacle_type == "sphere":
            obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.SPHERE)
            radius = float(obstacle_data.get("radius", 0.0))
            if radius <= 0.0:
                raise ValueError(f"{case['name']}: sphere radius must be positive")
            obstacle.set_attribute(cumotion.Obstacle.Attribute.RADIUS, radius)
        else:
            raise ValueError(f"{case['name']}: unsupported obstacle type {obstacle_type!r}")
        world.add_obstacle(obstacle, pose)
        count += 1
    return count


def path_length(path: list[np.ndarray]) -> float | None:
    if len(path) < 2:
        return 0.0 if path else None
    matrix = np.asarray(path, dtype=np.float64)
    return float(np.linalg.norm(np.diff(matrix, axis=0), axis=1).sum())


def densify_path(path: list[np.ndarray], max_joint_step: float = 0.01) -> list[np.ndarray]:
    if len(path) < 2:
        return path
    dense = [np.asarray(path[0], dtype=np.float64)]
    for start, end in zip(path, path[1:]):
        start_array = np.asarray(start, dtype=np.float64)
        end_array = np.asarray(end, dtype=np.float64)
        steps = max(1, int(np.ceil(np.max(np.abs(end_array - start_array)) / max_joint_step)))
        dense.extend(
            start_array + (end_array - start_array) * (index / steps)
            for index in range(1, steps + 1)
        )
    return dense


def validate_path_collisions(
    inspector: Any,
    path: list[np.ndarray],
    has_obstacles: bool,
) -> tuple[bool | None, float | None, int]:
    if not path:
        return None, None, 0
    validation_path = densify_path(path)
    self_collision = False
    minimum_clearance: float | None = None
    for configuration in validation_path:
        if inspector.in_self_collision(configuration):
            self_collision = True
        if has_obstacles:
            clearance = float(inspector.min_distance_to_obstacle(configuration))
            minimum_clearance = clearance if minimum_clearance is None else min(minimum_clearance, clearance)
    return self_collision, minimum_clearance, len(validation_path)


def target_pose_for_case(kinematics: Any, case: dict[str, Any]) -> Any:
    if case.get("target_pose") is None:
        return kinematics.pose(case["target_joint_positions"], TOOL_FRAME)
    position = case["target_pose"]["position_xyz"]
    x, y, z, w = case["target_pose"]["quaternion_xyzw"]
    return cumotion.Pose3(cumotion.Rotation3(w, x, y, z), position)


def plan_once(
    planner: Any,
    kinematics: Any,
    inspector: Any,
    joint_limits: list[tuple[float, float]],
    case: dict[str, Any],
    has_obstacles: bool,
) -> dict[str, Any]:
    start = case["start_joint_positions"]
    target_joints = case["target_joint_positions"]
    target_pose = target_pose_for_case(kinematics, case)

    planner.reset()
    started = time.perf_counter_ns()
    if case["mode"] == "cspace":
        if target_joints is None:
            raise ValueError(f"C-space case {case['name']} requires target_joint_positions")
        result = planner.plan_to_cspace_target(start, target_joints, True)
    else:
        result = planner.plan_to_pose_target(start, target_pose, True)
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0

    raw_path = list(result.path) if result.path_found else []
    interpolated_path = list(result.interpolated_path) if result.path_found else []
    validation_path = interpolated_path or raw_path
    self_collision, minimum_clearance, validation_samples = validate_path_collisions(
        inspector, validation_path, has_obstacles
    )

    trial: dict[str, Any] = {
        "path_found": bool(result.path_found),
        "latency_ms": latency_ms,
        "knots": len(raw_path),
        "interpolated_points": len(interpolated_path),
        "joint_path_length_rad": path_length(raw_path),
        "interpolated_joint_path_length_rad": path_length(interpolated_path),
        "sampled_self_collision_in_returned_path": self_collision,
        "minimum_sampled_sphere_clearance_m": minimum_clearance,
        "required_sampled_sphere_clearance_m": (
            REQUIRED_SAMPLED_SPHERE_CLEARANCE_M if has_obstacles else None
        ),
        "collision_validation_samples": validation_samples,
        "goal_max_joint_error_rad": None,
        "goal_translation_error_m": None,
        "goal_orientation_error_rad": None,
        "goal_tolerance_met": False,
        "final_joint_positions": None,
        "minimum_final_joint_limit_margin_rad": None,
        "accepted": False,
    }
    if not result.path_found:
        return trial

    final_configuration = np.asarray(validation_path[-1], dtype=np.float64)
    trial["final_joint_positions"] = final_configuration.tolist()
    trial["minimum_final_joint_limit_margin_rad"] = min(
        min(value - lower, upper - value)
        for value, (lower, upper) in zip(final_configuration, joint_limits)
    )
    final_pose = kinematics.pose(final_configuration, TOOL_FRAME)
    trial["goal_translation_error_m"] = float(
        np.linalg.norm(final_pose.translation - target_pose.translation)
    )
    trial["goal_orientation_error_rad"] = float(
        cumotion.Rotation3.distance(final_pose.rotation, target_pose.rotation)
    )
    if case["mode"] == "cspace":
        trial["goal_max_joint_error_rad"] = float(
            np.max(np.abs(final_configuration - target_joints))
        )
        trial["goal_tolerance_met"] = trial["goal_max_joint_error_rad"] <= 1e-8
    else:
        trial["goal_tolerance_met"] = (
            trial["goal_translation_error_m"] <= 0.0001
            and trial["goal_orientation_error_rad"] <= 0.005
        )
    trial["accepted"] = bool(
        trial["goal_tolerance_met"]
        and trial["sampled_self_collision_in_returned_path"] is False
        and (
            trial["minimum_sampled_sphere_clearance_m"] is None
            or trial["minimum_sampled_sphere_clearance_m"]
            >= REQUIRED_SAMPLED_SPHERE_CLEARANCE_M
        )
    )
    return trial


def percentile(values: list[float], amount: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), amount))


def summarize_case(case: dict[str, Any], trials: list[dict[str, Any]]) -> dict[str, Any]:
    paths_found = [trial for trial in trials if trial["path_found"]]
    accepted = [trial for trial in trials if trial["accepted"]]
    latencies = [float(trial["latency_ms"]) for trial in trials]
    return {
        "case": case["name"],
        "mode": case["mode"],
        "expected": case.get("expected", "success"),
        "trials": len(trials),
        "paths_found": len(paths_found),
        "path_found_rate": len(paths_found) / len(trials),
        "accepted_paths": len(accepted),
        "acceptance_rate": len(accepted) / len(trials),
        "latency_mean_ms": statistics.fmean(latencies),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_max_ms": max(latencies),
        "goal_translation_error_max_m": max(
            (trial["goal_translation_error_m"] for trial in paths_found), default=None
        ),
        "goal_orientation_error_max_rad": max(
            (trial["goal_orientation_error_rad"] for trial in paths_found), default=None
        ),
        "joint_path_length_mean_rad": statistics.fmean(
            trial["joint_path_length_rad"] for trial in paths_found
        )
        if paths_found
        else None,
        "minimum_sampled_sphere_clearance_m": min(
            (
                trial["minimum_sampled_sphere_clearance_m"]
                for trial in paths_found
                if trial["minimum_sampled_sphere_clearance_m"] is not None
            ),
            default=None,
        ),
    }


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = result.stdout.strip()
    return output or None


def gpu_process_snapshot() -> str | None:
    return command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )


def environment_record(args: argparse.Namespace, robot_description: Any) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cumotion_version": getattr(cumotion, "__version__", "unknown"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "joint_names": JOINT_NAMES,
        "cspace_coordinates": robot_description.num_cspace_coords(),
        "tool_frames": list(robot_description.tool_frame_names()),
        "tool_frame_used": TOOL_FRAME,
        "trials_per_case": args.trials,
        "warmups_per_case": args.warmups,
        "urdf": str(args.urdf.resolve()),
        "xrdf": str(args.xrdf.resolve()),
        "planner_config": str(args.planner_config.resolve()),
        "cases": str(args.cases.resolve()),
        "plan_only": True,
        "real_robot_commanded": False,
        "camera_or_depth_used": False,
        "calibration_status": "provisional gripper mount/TCP; flange benchmark only",
        "latency_scope": (
            "Synchronous cuMotion planning call only; excludes planner reset, model/planner "
            "construction, collision validation, result serialization, and process startup."
        ),
        "trial_semantics": (
            "Fixed seed with planner.reset(); repeated trials measure timing jitter of a "
            "deterministic path, not planning robustness."
        ),
        "collision_validation_scope": (
            "Joint path resampled to at most 0.01 rad per joint step and checked against the "
            "provisional, surface-audited XRDF sphere model. Obstacle paths must retain at "
            f"least {REQUIRED_SAMPLED_SPHERE_CLEARANCE_M:.3f} m sampled sphere clearance "
            "to reserve the known coverage uncertainty; this is not an independent mesh or "
            "hardware check."
        ),
        "required_sampled_sphere_clearance_m": REQUIRED_SAMPLED_SPHERE_CLEARANCE_M,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_results(
    output_dir: Path,
    environment: dict[str, Any],
    trial_records: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    input_paths: dict[str, Path],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "trials.jsonl").open("w", encoding="utf-8") as stream:
        for trial in trial_records:
            stream.write(json.dumps(trial) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps({"cases": summaries}, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    for label, source in input_paths.items():
        shutil.copy2(source, inputs_dir / f"{label}{source.suffix}")


def main() -> int:
    args = build_parser().parse_args()
    if args.trials < 1 or args.warmups < 0:
        raise ValueError("--trials must be positive and --warmups must not be negative")
    validate_files([args.urdf, args.xrdf, args.planner_config, args.cases])
    cases = validate_cases(read_yaml(args.cases), args.selected_cases)

    cumotion.set_log_level(cumotion.LogLevel.ERROR)
    load_started = time.perf_counter()
    robot_description = cumotion.load_robot_from_file(str(args.xrdf), str(args.urdf))
    load_seconds = time.perf_counter() - load_started
    if robot_description.num_cspace_coords() != 6:
        raise RuntimeError(
            f"Expected a six-axis robot; cuMotion loaded {robot_description.num_cspace_coords()} axes"
        )
    if TOOL_FRAME not in robot_description.tool_frame_names():
        raise RuntimeError(f"XRDF does not expose required tool frame: {TOOL_FRAME}")
    kinematics = robot_description.kinematics()
    joint_limits = []
    for index in range(kinematics.num_cspace_coords()):
        limits = kinematics.cspace_coord_limits(index)
        joint_limits.append((float(limits.lower), float(limits.upper)))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output or (ARENA_DIR / "outputs/cumotion_benchmark" / timestamp)
    trial_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    warmup_records: list[dict[str, Any]] = []
    preflight_records: list[dict[str, Any]] = []

    print(f"cuMotion {getattr(cumotion, '__version__', 'unknown')} | tool={TOOL_FRAME}")
    print(f"Loaded TM5S description in {load_seconds:.3f} s")
    for case in cases:
        world = cumotion.create_world()
        obstacle_count = add_obstacles(world, case)
        world_view = world.add_world_view()
        planner_config = cumotion.create_motion_planner_config_from_file(
            str(args.planner_config), robot_description, TOOL_FRAME, world_view
        )
        planner_config.set_param("enable_self_collision_checking", True)
        planner = cumotion.create_motion_planner(planner_config)
        inspector = cumotion.create_robot_world_inspector(robot_description, world.add_world_view())

        start = case["start_joint_positions"]
        target = case["target_joint_positions"]
        start_self_collision = bool(inspector.in_self_collision(start))
        target_self_collision = (
            bool(inspector.in_self_collision(target)) if target is not None else None
        )
        if start_self_collision or target_self_collision:
            pairs = {
                "start": inspector.frames_in_self_collision(start) if start_self_collision else [],
                "target": inspector.frames_in_self_collision(target) if target_self_collision else [],
            }
            raise RuntimeError(f"Case {case['name']} has a self-colliding endpoint: {pairs}")
        start_world_collision = bool(
            obstacle_count and inspector.in_collision_with_obstacle(start)
        )
        target_world_collision = bool(
            obstacle_count and target is not None and inspector.in_collision_with_obstacle(target)
        ) if target is not None else None
        start_clearance = (
            float(inspector.min_distance_to_obstacle(start)) if obstacle_count else None
        )
        target_clearance = (
            float(inspector.min_distance_to_obstacle(target))
            if obstacle_count and target is not None
            else None
        )
        if case.get("expected", "success") == "success" and (
            start_world_collision or target_world_collision
        ):
            raise RuntimeError(
                f"Success case {case['name']} has a world-colliding endpoint: "
                f"start={start_world_collision}, target={target_world_collision}"
            )
        preflight_records.append(
            {
                "case": case["name"],
                "expected": case.get("expected", "success"),
                "obstacle_count": obstacle_count,
                "start_self_collision": start_self_collision,
                "target_self_collision": target_self_collision,
                "target_joint_endpoint_known": target is not None,
                "start_world_collision": start_world_collision,
                "target_world_collision": target_world_collision,
                "start_world_clearance_m": start_clearance,
                "target_world_clearance_m": target_clearance,
            }
        )

        for warmup_index in range(args.warmups):
            warmup = plan_once(
                planner, kinematics, inspector, joint_limits, case, obstacle_count > 0
            )
            warmup_records.append(
                {"case": case["name"], "warmup": warmup_index + 1, **warmup}
            )

        case_trials: list[dict[str, Any]] = []
        for trial_index in range(args.trials):
            trial = plan_once(
                planner, kinematics, inspector, joint_limits, case, obstacle_count > 0
            )
            record = {"case": case["name"], "trial": trial_index + 1, **trial}
            trial_records.append(record)
            case_trials.append(record)
        summary = summarize_case(case, case_trials)
        summaries.append(summary)
        print(
            f"{case['name']}: {summary['accepted_paths']}/{summary['trials']} accepted, "
            f"p50={summary['latency_p50_ms']:.2f} ms, "
            f"p95={summary['latency_p95_ms']:.2f} ms"
        )

    environment = environment_record(args, robot_description)
    environment["robot_description_load_seconds"] = load_seconds
    environment["gpu_processes_after_warmup"] = gpu_process_snapshot()
    environment["case_preflight"] = preflight_records
    environment["warmups"] = warmup_records
    input_paths = {
        "urdf": args.urdf.resolve(),
        "xrdf": args.xrdf.resolve(),
        "planner_config": args.planner_config.resolve(),
        "cases": args.cases.resolve(),
    }
    manifest_path = args.urdf.resolve().parent / "asset_manifest.json"
    if manifest_path.is_file():
        input_paths["asset_manifest"] = manifest_path
    environment["command"] = [sys.executable, *sys.argv]
    environment["input_artifacts"] = {
        label: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for label, path in input_paths.items()
    }
    write_results(output_dir.resolve(), environment, trial_records, summaries, input_paths)
    print(f"Results: {output_dir.resolve()}")

    all_expectations_met = all(
        (summary["acceptance_rate"] == 1.0)
        if summary["expected"] == "success"
        else (summary["path_found_rate"] == 0.0)
        for summary in summaries
    )
    return 0 if all_expectations_met else 2


if __name__ == "__main__":
    raise SystemExit(main())
