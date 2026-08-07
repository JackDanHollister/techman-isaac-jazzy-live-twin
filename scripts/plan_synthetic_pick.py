#!/usr/bin/env python3
"""Plan and time-parameterize one synthetic pin pick with standalone cuMotion.

The input is a camera-free synthetic scene produced by ``run_pin_axis_demo.py``.
Every detected pin is tested through pregrasp, grasp, and lift. The accepted
candidate with the largest sampled collision-sphere clearance is written as a
controller-sampled trajectory for the standalone Isaac viewer.

This script creates no ROS graph, network connection, sensor input, or Watson
command. The current 2FG7 mount and virtual pin-grasp point are provisional.
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
import sys
import time
from typing import Any

import cumotion
import numpy as np
import yaml


ARENA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ARENA_DIR / "generated/cumotion"
DEFAULT_TASK_CONFIG = ARENA_DIR / "config/synthetic_pick_task.yaml"
DEFAULT_PLANNER_CONFIG = ARENA_DIR / "config/tm5s_cumotion_planner.yaml"
EXPECTED_CUMOTION_VERSION = "1.1.0"
STAGE_POSITION_KEYS = {
    "pregrasp": "pregrasp_position",
    "grasp": "grasp_position",
    "lift": "lift_position",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_MODEL_DIR / "tm5s_with_2fg7.urdf")
    parser.add_argument("--xrdf", type=Path, default=DEFAULT_MODEL_DIR / "tm5s_with_2fg7.xrdf")
    parser.add_argument("--planner-config", type=Path, default=DEFAULT_PLANNER_CONFIG)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON (default: beside result_json as synthetic_pick_plan.json).",
    )
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def float64_sha256(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.asarray(array, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def finite_vector(value: Any, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain {size} finite values")
    return result


def validate_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = [args.result_json, args.urdf, args.xrdf, args.planner_config, args.task_config]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing synthetic-pick inputs:\n  " + "\n  ".join(missing))
    if getattr(cumotion, "__version__", None) != EXPECTED_CUMOTION_VERSION:
        raise RuntimeError(
            f"Expected cuMotion {EXPECTED_CUMOTION_VERSION}; found "
            f"{getattr(cumotion, '__version__', 'unknown')}"
        )

    result = json.loads(args.result_json.read_text(encoding="utf-8"))
    task = read_yaml(args.task_config)
    if task.get("format_version") != 1:
        raise ValueError("Synthetic task config must use format_version: 1")
    if result.get("seed") != task.get("seed"):
        raise ValueError(
            f"Synthetic result seed {result.get('seed')} does not match task seed {task.get('seed')}"
        )
    if result.get("scene", {}).get("pin_count") != task.get("pin_count"):
        raise ValueError("Synthetic result pin count does not match task config")
    if result.get("frames", {}).get("target_frame") != task.get("frame_id"):
        raise ValueError("Synthetic result target frame does not match task config")
    if task.get("scope", {}).get("real_robot_commanded") is not False:
        raise ValueError("Task config must explicitly keep real_robot_commanded false")
    if task.get("scope", {}).get("camera_or_depth_used") is not False:
        raise ValueError("Task config must explicitly keep camera_or_depth_used false")
    return result, task


def rotation_aligning_z(axis: np.ndarray) -> Any:
    z_axis = axis / np.linalg.norm(axis)
    reference = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(reference, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return cumotion.Rotation3.from_matrix(np.column_stack((x_axis, y_axis, z_axis)))


def quaternion_xyzw(rotation: Any) -> list[float]:
    return [
        float(rotation.x()),
        float(rotation.y()),
        float(rotation.z()),
        float(rotation.w()),
    ]


def add_cuboid(
    world: Any,
    specs: list[dict[str, Any]],
    *,
    role: str,
    position: np.ndarray,
    side_lengths: np.ndarray,
    rotation: Any | None = None,
    source_id: int | None = None,
) -> None:
    obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
    obstacle.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS, side_lengths)
    pose = (
        cumotion.Pose3(rotation, position)
        if rotation is not None
        else cumotion.Pose3.from_translation(position)
    )
    world.add_obstacle(obstacle, pose)
    specs.append(
        {
            "type": "cuboid",
            "role": role,
            "source_id": source_id,
            "position_xyz_m": position.tolist(),
            "quaternion_xyzw": quaternion_xyzw(rotation or cumotion.Rotation3.identity()),
            "side_lengths_m": side_lengths.tolist(),
        }
    )


def add_sphere(
    world: Any,
    specs: list[dict[str, Any]],
    *,
    role: str,
    position: np.ndarray,
    radius: float,
    source_id: int | None = None,
) -> None:
    obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.SPHERE)
    obstacle.set_attribute(cumotion.Obstacle.Attribute.RADIUS, radius)
    world.add_obstacle(obstacle, cumotion.Pose3.from_translation(position))
    specs.append(
        {
            "type": "sphere",
            "role": role,
            "source_id": source_id,
            "position_xyz_m": position.tolist(),
            "radius_m": radius,
        }
    )


def build_candidate_world(
    result: dict[str, Any],
    task: dict[str, Any],
    selected_truth_id: int,
) -> tuple[Any, list[dict[str, Any]]]:
    world = cumotion.create_world()
    specs: list[dict[str, Any]] = []
    scene_config = result["scene"]["config"]
    collision = task["collision_scene"]
    center_x = float(scene_config["tray_center_x"])
    center_y = float(scene_config["tray_center_y"])
    foam_z = float(scene_config["foam_z"])
    size_x = float(scene_config["tray_size_x"])
    size_y = float(scene_config["tray_size_y"])
    foam_thickness = float(collision["foam_thickness_m"])
    wall_thickness = float(collision["tray_wall_thickness_m"])
    wall_height = float(collision["tray_wall_height_m"])

    add_cuboid(
        world,
        specs,
        role="foam",
        position=np.array([center_x, center_y, foam_z - foam_thickness / 2.0]),
        side_lengths=np.array([size_x, size_y, foam_thickness]),
    )
    wall_specs = [
        (
            np.array([center_x - size_x / 2.0 - wall_thickness / 2.0, center_y, foam_z + wall_height / 2.0]),
            np.array([wall_thickness, size_y + 2.0 * wall_thickness, wall_height]),
        ),
        (
            np.array([center_x + size_x / 2.0 + wall_thickness / 2.0, center_y, foam_z + wall_height / 2.0]),
            np.array([wall_thickness, size_y + 2.0 * wall_thickness, wall_height]),
        ),
        (
            np.array([center_x, center_y - size_y / 2.0 - wall_thickness / 2.0, foam_z + wall_height / 2.0]),
            np.array([size_x, wall_thickness, wall_height]),
        ),
        (
            np.array([center_x, center_y + size_y / 2.0 + wall_thickness / 2.0, foam_z + wall_height / 2.0]),
            np.array([size_x, wall_thickness, wall_height]),
        ),
    ]
    for position, sides in wall_specs:
        add_cuboid(world, specs, role="tray_wall", position=position, side_lengths=sides)

    specimen_padding = float(collision["specimen_padding_m"])
    shaft_width = float(collision["other_pin_shaft_width_m"])
    head_radius = float(collision["other_pin_head_radius_m"])
    for truth in result["scene"]["truth"]:
        truth_id = int(truth["pin_id"])
        specimen_sides = 2.0 * finite_vector(truth["specimen_radii"], 3, "specimen_radii")
        specimen_sides += 2.0 * specimen_padding
        add_cuboid(
            world,
            specs,
            role="specimen_body",
            source_id=truth_id,
            position=finite_vector(truth["specimen_center"], 3, "specimen_center"),
            side_lengths=specimen_sides,
        )
        if truth_id == selected_truth_id:
            continue
        base = finite_vector(truth["base"], 3, "pin_base")
        axis = finite_vector(truth["axis_up"], 3, "pin_axis")
        axis /= np.linalg.norm(axis)
        length = float(truth["length_m"])
        add_cuboid(
            world,
            specs,
            role="other_pin_shaft",
            source_id=truth_id,
            position=base + axis * length / 2.0,
            side_lengths=np.array([shaft_width, shaft_width, length]),
            rotation=rotation_aligning_z(axis),
        )
        add_sphere(
            world,
            specs,
            role="other_pin_head",
            source_id=truth_id,
            position=finite_vector(truth["head"], 3, "pin_head"),
            radius=head_radius,
        )
    return world, specs


def flange_pose(target: dict[str, Any], stage: str, task: dict[str, Any]) -> Any:
    tcp_position = finite_vector(target[STAGE_POSITION_KEYS[stage]], 3, f"{stage}_position")
    tool_z = finite_vector(target["tool_z_axis_robot"], 3, "tool_z_axis_robot")
    tool_model = task["tool_model"]
    flange_to_tcp = float(tool_model["pin_grasp_tcp_z_from_2fg7_origin_m"])
    mount_xyz = finite_vector(tool_model["flange_to_2fg7_origin_xyz_m"], 3, "mount_xyz")
    mount_rpy = finite_vector(tool_model["flange_to_2fg7_origin_rpy_rad"], 3, "mount_rpy")
    if not np.allclose(mount_xyz, 0.0) or not np.allclose(mount_rpy, 0.0):
        raise NotImplementedError(
            "The first synthetic benchmark supports only the current identity flange-to-2FG7 mount"
        )
    position = tcp_position - flange_to_tcp * tool_z
    qx, qy, qz, qw = finite_vector(target["quaternion_xyzw"], 4, "quaternion_xyzw")
    return cumotion.Pose3(cumotion.Rotation3(qw, qx, qy, qz), position)


def samples_for_trajectory(
    trajectory: Any, dt_seconds: float, time_scale: float
) -> list[dict[str, Any]]:
    duration = float(trajectory.domain().span()) * time_scale
    count = max(1, int(math.ceil(duration / dt_seconds)))
    times = np.linspace(0.0, duration, count + 1)
    samples: list[dict[str, Any]] = []
    upper = float(trajectory.domain().upper)
    for sample_time in times:
        trajectory_time = min(upper, float(sample_time) / time_scale)
        samples.append(
            {
                "time_seconds": float(sample_time),
                "joint_positions": np.asarray(
                    trajectory.eval(trajectory_time, 0), dtype=np.float64
                ).tolist(),
                "joint_velocities": (
                    np.asarray(trajectory.eval(trajectory_time, 1), dtype=np.float64)
                    / time_scale
                ).tolist(),
            }
        )
    return samples


def maximum_control_step(samples: list[dict[str, Any]]) -> float:
    positions = np.asarray(
        [sample["joint_positions"] for sample in samples], dtype=np.float64
    )
    if len(positions) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(positions, axis=0))))


def validate_trajectory(
    trajectory: Any,
    inspector: Any,
    kinematics: Any,
    validation_dt: float,
    time_scale: float,
) -> dict[str, Any]:
    time_optimal_duration = float(trajectory.domain().span())
    duration = time_optimal_duration * time_scale
    count = max(1, int(math.ceil(duration / validation_dt)))
    times = np.linspace(0.0, duration, count + 1)
    minimum_clearance = math.inf
    self_collision = False
    positions: list[np.ndarray] = []
    upper = float(trajectory.domain().upper)
    for sample_time in times:
        trajectory_time = min(upper, float(sample_time) / time_scale)
        configuration = np.asarray(
            trajectory.eval(trajectory_time, 0), dtype=np.float64
        )
        positions.append(configuration)
        self_collision = self_collision or bool(inspector.in_self_collision(configuration))
        minimum_clearance = min(
            minimum_clearance,
            float(inspector.min_distance_to_obstacle(configuration)),
        )

    velocity_limits = np.array(
        [kinematics.cspace_coord_velocity_limit(index) for index in range(kinematics.num_cspace_coords())]
    )
    acceleration_limits = np.array(
        [kinematics.cspace_coord_acceleration_limit(index) for index in range(kinematics.num_cspace_coords())]
    )
    jerk_limits = np.array(
        [kinematics.cspace_coord_jerk_limit(index) for index in range(kinematics.num_cspace_coords())]
    )
    time_optimal_maximum_velocity = np.asarray(
        trajectory.max_velocity_magnitude(), dtype=np.float64
    )
    time_optimal_maximum_acceleration = np.asarray(
        trajectory.max_acceleration_magnitude(), dtype=np.float64
    )
    time_optimal_maximum_jerk = np.asarray(
        trajectory.max_jerk_magnitude(), dtype=np.float64
    )
    maximum_velocity = time_optimal_maximum_velocity / time_scale
    maximum_acceleration = time_optimal_maximum_acceleration / (time_scale**2)
    maximum_jerk = time_optimal_maximum_jerk / (time_scale**3)
    tolerance = 1.0e-6
    return {
        "duration_seconds": duration,
        "time_optimal_duration_seconds": time_optimal_duration,
        "execution_time_scale": time_scale,
        "validation_samples": len(times),
        "minimum_sampled_sphere_clearance_m": minimum_clearance,
        "sampled_self_collision": self_collision,
        "maximum_velocity_rad_s": maximum_velocity.tolist(),
        "maximum_acceleration_rad_s2": maximum_acceleration.tolist(),
        "maximum_jerk_rad_s3": maximum_jerk.tolist(),
        "velocity_limits_rad_s": velocity_limits.tolist(),
        "acceleration_limits_rad_s2": acceleration_limits.tolist(),
        "jerk_limits_rad_s3": jerk_limits.tolist(),
        "derivative_limits_met": bool(
            np.all(maximum_velocity <= velocity_limits + tolerance)
            and np.all(maximum_acceleration <= acceleration_limits + tolerance)
            and np.all(maximum_jerk <= jerk_limits + tolerance)
        ),
        "sampled_positions_float64_sha256": float64_sha256(positions),
    }


def plan_candidate(
    robot_description: Any,
    kinematics: Any,
    trajectory_generator: Any,
    result: dict[str, Any],
    task: dict[str, Any],
    planner_config_path: Path,
    target: dict[str, Any],
    truth_id: int,
) -> dict[str, Any]:
    world, obstacle_specs = build_candidate_world(result, task, truth_id)
    planner_config = cumotion.create_motion_planner_config_from_file(
        str(planner_config_path),
        robot_description,
        task["robot"]["planning_tool_frame"],
        world.add_world_view(),
    )
    planner_config.set_param("enable_self_collision_checking", True)
    planner = cumotion.create_motion_planner(planner_config)
    inspector = cumotion.create_robot_world_inspector(robot_description, world.add_world_view())
    current = finite_vector(task["robot"]["ready_joint_positions"], 6, "ready_joint_positions")
    stages: list[dict[str, Any]] = []
    accepted = True
    reason: str | None = None
    translation_tolerance = float(task["planning"]["goal_translation_tolerance_m"])
    orientation_tolerance = float(task["planning"]["goal_orientation_tolerance_rad"])
    required_clearance = float(task["planning"]["required_sampled_sphere_clearance_m"])
    validation_dt = float(task["planning"]["collision_validation_dt_seconds"])
    control_dt = float(task["planning"]["control_dt_seconds"])
    maximum_allowed_control_step = float(task["planning"]["maximum_control_step_rad"])
    execution_time_scale = float(task["planning"]["execution_time_scale"])
    if execution_time_scale < 1.0:
        raise ValueError("execution_time_scale must be at least 1.0")

    if inspector.in_self_collision(current) or inspector.in_collision_with_obstacle(current):
        return {
            "detection_id": int(target["detection_id"]),
            "truth_id": truth_id,
            "accepted": False,
            "rejection_reason": "ready_configuration_in_collision",
            "minimum_sampled_sphere_clearance_m": None,
            "stages": [],
            "collision_obstacles": obstacle_specs,
        }

    for stage_name in task["planning"]["stages"]:
        target_pose = flange_pose(target, stage_name, task)
        planner.reset()
        started = time.perf_counter_ns()
        planning_result = planner.plan_to_pose_target(current, target_pose, True)
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        if not planning_result.path_found:
            accepted = False
            reason = f"{stage_name}_path_not_found"
            stages.append(
                {
                    "name": stage_name,
                    "path_found": False,
                    "planning_latency_ms": latency_ms,
                }
            )
            break

        raw_path = [np.asarray(point, dtype=np.float64) for point in planning_result.path]
        interpolated_path = [
            np.asarray(point, dtype=np.float64) for point in planning_result.interpolated_path
        ]
        if len(raw_path) < 2:
            accepted = False
            reason = f"{stage_name}_returned_short_path"
            break
        trajectory = trajectory_generator.generate_trajectory(raw_path)
        if trajectory is None:
            accepted = False
            reason = f"{stage_name}_trajectory_generation_failed"
            break
        validation = validate_trajectory(
            trajectory,
            inspector,
            kinematics,
            validation_dt,
            execution_time_scale,
        )
        final_configuration = raw_path[-1]
        actual_pose = kinematics.pose(final_configuration, task["robot"]["planning_tool_frame"])
        translation_error = float(np.linalg.norm(actual_pose.translation - target_pose.translation))
        orientation_error = float(cumotion.Rotation3.distance(actual_pose.rotation, target_pose.rotation))
        goal_tolerance_met = (
            translation_error <= translation_tolerance and orientation_error <= orientation_tolerance
        )
        stage_accepted = bool(
            goal_tolerance_met
            and not validation["sampled_self_collision"]
            and validation["minimum_sampled_sphere_clearance_m"] >= required_clearance
            and validation["derivative_limits_met"]
        )
        control_samples = samples_for_trajectory(
            trajectory, control_dt, execution_time_scale
        )
        maximum_sample_step = maximum_control_step(control_samples)
        stage_accepted = bool(
            stage_accepted and maximum_sample_step <= maximum_allowed_control_step + 1.0e-12
        )
        stages.append(
            {
                "name": stage_name,
                "path_found": True,
                "accepted": stage_accepted,
                "planning_latency_ms": latency_ms,
                "target_flange_pose": {
                    "position_xyz_m": np.asarray(target_pose.translation, dtype=np.float64).tolist(),
                    "quaternion_xyzw": quaternion_xyzw(target_pose.rotation),
                },
                "raw_knots": len(raw_path),
                "interpolated_points": len(interpolated_path),
                "raw_path": [point.tolist() for point in raw_path],
                "raw_path_float64_sha256": float64_sha256(raw_path),
                "goal_translation_error_m": translation_error,
                "goal_orientation_error_rad": orientation_error,
                "goal_tolerance_met": goal_tolerance_met,
                "trajectory_validation": validation,
                "maximum_control_step_rad": maximum_sample_step,
                "control_samples": control_samples,
                "control_samples_float64_sha256": float64_sha256(
                    [
                        np.asarray(sample["joint_positions"], dtype=np.float64)
                        for sample in control_samples
                    ]
                    + [
                        np.asarray(sample["joint_velocities"], dtype=np.float64)
                        for sample in control_samples
                    ]
                ),
            }
        )
        if not stage_accepted:
            accepted = False
            reason = f"{stage_name}_validation_failed"
            break
        current = final_configuration

    clearances = [
        stage["trajectory_validation"]["minimum_sampled_sphere_clearance_m"]
        for stage in stages
        if stage.get("accepted")
    ]
    return {
        "detection_id": int(target["detection_id"]),
        "truth_id": truth_id,
        "accepted": accepted and len(stages) == len(task["planning"]["stages"]),
        "rejection_reason": reason,
        "minimum_sampled_sphere_clearance_m": min(clearances) if clearances else None,
        "stages": stages,
        "collision_obstacles": obstacle_specs,
    }


def candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    """Keep selection evidence without duplicating every control sample and obstacle."""
    return {
        "detection_id": candidate["detection_id"],
        "truth_id": candidate["truth_id"],
        "accepted": candidate["accepted"],
        "rejection_reason": candidate["rejection_reason"],
        "minimum_sampled_sphere_clearance_m": candidate[
            "minimum_sampled_sphere_clearance_m"
        ],
        "collision_obstacle_count": len(candidate["collision_obstacles"]),
        "stages": [
            {key: value for key, value in stage.items() if key != "control_samples"}
            for stage in candidate["stages"]
        ],
    }


def main() -> int:
    args = build_parser().parse_args()
    args.result_json = args.result_json.expanduser().resolve()
    args.urdf = args.urdf.expanduser().resolve()
    args.xrdf = args.xrdf.expanduser().resolve()
    args.planner_config = args.planner_config.expanduser().resolve()
    args.task_config = args.task_config.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else args.result_json.with_name("synthetic_pick_plan.json")
    )
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite synthetic pick plan: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result, task = validate_inputs(args)
    cumotion.set_log_level(cumotion.LogLevel.ERROR)
    robot_description = cumotion.load_robot_from_file(str(args.xrdf), str(args.urdf))
    if robot_description.num_cspace_coords() != 6:
        raise RuntimeError("Synthetic pick benchmark requires a six-axis robot")
    expected_joint_names = list(task["robot"]["joint_names"])
    kinematics = robot_description.kinematics()
    actual_joint_names = [
        kinematics.cspace_coord_name(index) for index in range(kinematics.num_cspace_coords())
    ]
    if actual_joint_names != expected_joint_names:
        raise RuntimeError(f"Unexpected joint order: {actual_joint_names}")
    tool_frame = task["robot"]["planning_tool_frame"]
    if tool_frame not in robot_description.tool_frame_names():
        raise RuntimeError(f"Planning tool frame is not available in XRDF: {tool_frame}")

    matches = result["evaluation"]["matches"]
    truth_by_detection = {int(match["detection_id"]): int(match["truth_id"]) for match in matches}
    target_by_detection = {
        int(target["detection_id"]): target for target in result["alignment"]["targets"]
    }
    if set(truth_by_detection) != set(target_by_detection):
        raise ValueError("Every synthetic detection must have exactly one ground-truth match")

    trajectory_generator = cumotion.create_cspace_trajectory_generator(kinematics)
    candidates = [
        plan_candidate(
            robot_description,
            kinematics,
            trajectory_generator,
            result,
            task,
            args.planner_config,
            target_by_detection[detection_id],
            truth_by_detection[detection_id],
        )
        for detection_id in sorted(target_by_detection)
    ]
    accepted_candidates = [candidate for candidate in candidates if candidate["accepted"]]
    if not accepted_candidates:
        raise RuntimeError("No synthetic pin candidate passed all planning and trajectory checks")
    selected = max(
        accepted_candidates,
        key=lambda candidate: (
            candidate["minimum_sampled_sphere_clearance_m"],
            -candidate["detection_id"],
        ),
    )
    selected_match = next(
        match for match in matches if int(match["detection_id"]) == selected["detection_id"]
    )
    selected_target = target_by_detection[selected["detection_id"]]

    input_paths = {
        "synthetic_result": args.result_json,
        "urdf": args.urdf,
        "xrdf": args.xrdf,
        "planner_config": args.planner_config,
        "task_config": args.task_config,
    }
    payload = {
        "format_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cumotion_version": cumotion.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "validation_scope": "camera_free_synthetic_pin_cumotion_plan_and_time_parameterization",
        "camera_or_depth_used": False,
        "ros_used": False,
        "real_robot_commanded": False,
        "watson_connected": False,
        "planning_tool_frame": tool_frame,
        "tool_model": task["tool_model"],
        "tool_model_status": (
            "legacy CAD dry-run profile: provisional identity flange mount; pin-grasp "
            "point is the CAD fingertip plane, not the OnRobot nominal TCP; robot-side "
            "Quick Changer is omitted by this profile"
        ),
        "collision_model_status": (
            "provisional XRDF spheres plus synthetic primitive tray, specimens, and non-target pins"
        ),
        "selection_rule": task["planning"]["selection_rule"],
        "required_sampled_sphere_clearance_m": task["planning"][
            "required_sampled_sphere_clearance_m"
        ],
        "control_dt_seconds": task["planning"]["control_dt_seconds"],
        "maximum_control_step_rad": task["planning"]["maximum_control_step_rad"],
        "execution_time_scale": task["planning"]["execution_time_scale"],
        "stage_hold_seconds": task["planning"]["stage_hold_seconds"],
        "isaac_execution": task["isaac_execution"],
        "input_artifacts": {
            name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in input_paths.items()
        },
        "synthetic_scene": {
            "seed": result["seed"],
            "frame_id": result["frames"]["target_frame"],
            "pin_count": result["scene"]["pin_count"],
            "scene_config": result["scene"]["config"],
            "truth": result["scene"]["truth"],
            "evaluation": result["evaluation"],
        },
        "candidate_count": len(candidates),
        "accepted_candidate_count": len(accepted_candidates),
        "candidates": [candidate_summary(candidate) for candidate in candidates],
        "selected": {
            **selected,
            "synthetic_detection_error": selected_match,
            "target": selected_target,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Selected detection {selected['detection_id']} / truth pin {selected['truth_id']} "
        f"with {selected['minimum_sampled_sphere_clearance_m'] * 1000.0:.3f} mm "
        "minimum sampled sphere clearance"
    )
    print(f"Accepted candidates: {len(accepted_candidates)}/{len(candidates)}")
    print(f"Synthetic pick plan: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
