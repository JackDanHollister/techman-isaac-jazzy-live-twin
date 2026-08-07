#!/usr/bin/env python3
"""Plan seven offline pin verticalizations with standalone cuMotion 1.1.

Truth specimens 1 through 7 from the deterministic seed-1407 synthetic scene
are planned ready-to-ready.  Each cycle approaches the detected tilted pin,
lifts it, moves the pinch point to a deterministic placement with the pin
vertical, places it, retreats, and returns to the configured ready joints.

The planner uses ``pin_grasp_tcp`` directly.  It creates no ROS graph, opens no
Watson connection, and sends no command to a physical robot.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from plan_synthetic_pick import (
    EXPECTED_CUMOTION_VERSION,
    add_cuboid,
    add_sphere,
    finite_vector,
    float64_sha256,
    maximum_control_step,
    quaternion_xyzw,
    read_yaml,
    rotation_aligning_z,
    samples_for_trajectory,
    sha256_file,
    validate_trajectory,
)


ARENA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ARENA_DIR / "outputs/synthetic_pick_seed_1407/result.json"
DEFAULT_OUTPUT = (
    ARENA_DIR
    / "outputs/multi_pin_verticalization_seed_1407/multi_pin_verticalization_plan.json"
)
DEFAULT_MODEL_DIR = ARENA_DIR / "generated/tool_profiles/watson_qc_nominal/cumotion"
DEFAULT_URDF = DEFAULT_MODEL_DIR / "tm5s_with_2fg7.urdf"
DEFAULT_XRDF = DEFAULT_MODEL_DIR / "tm5s_with_2fg7.xrdf"
DEFAULT_ASSET_MANIFEST = DEFAULT_MODEL_DIR / "asset_manifest.json"
DEFAULT_PLANNER_CONFIG = ARENA_DIR / "config/tm5s_cumotion_planner.yaml"
DEFAULT_TASK_CONFIG = ARENA_DIR / "config/synthetic_pick_task.yaml"
DEFAULT_REQUIRED_CLEARANCE_M = 0.004

TOOL_FRAME = "pin_grasp_tcp"
SPECIMEN_IDS = list(range(1, 8))
DESTINATION_BASE_OVERRIDES_M = {
    5: np.array([0.40, 0.00, 0.0], dtype=np.float64),
    7: np.array([0.45, -0.05, 0.0], dtype=np.float64),
}
STAGE_NAMES = [
    "approach_tilted_pregrasp",
    "descend_tilted_grasp",
    "lift_tilted",
    "reorient_vertical",
    "descend_vertical",
    "retreat_vertical",
    "return_ready",
]
POSE_STAGE_SOURCE_KEYS = {
    "approach_tilted_pregrasp": "pregrasp_position",
    "descend_tilted_grasp": "grasp_position",
    "lift_tilted": "lift_position",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--result-json", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--xrdf", type=Path, default=DEFAULT_XRDF)
    parser.add_argument("--asset-manifest", type=Path, default=DEFAULT_ASSET_MANIFEST)
    parser.add_argument("--planner-config", type=Path, default=DEFAULT_PLANNER_CONFIG)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--control-dt-seconds",
        type=float,
        default=1.0 / 300.0,
        help="Controller sampling period for every exported stage.",
    )
    parser.add_argument(
        "--validation-dt-seconds",
        type=float,
        default=1.0 / 300.0,
        help="Sampling period for collision and self-collision validation.",
    )
    parser.add_argument(
        "--execution-time-scale",
        type=float,
        default=1.0,
        help="Trajectory time multiplier; values below 1 are rejected.",
    )
    parser.add_argument(
        "--maximum-control-step-rad",
        type=float,
        default=0.01,
        help="Maximum absolute adjacent joint-position step in the export.",
    )
    parser.add_argument(
        "--required-clearance-m",
        type=float,
        default=DEFAULT_REQUIRED_CLEARANCE_M,
        help="Minimum sampled robot-sphere clearance for this simulation plan.",
    )
    return parser


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def normalized(value: Any, label: str) -> np.ndarray:
    vector = finite_vector(value, 3, label)
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= 1.0e-12:
        raise ValueError(f"{label} must be non-zero")
    return vector / magnitude


def pose_record(pose: Any) -> dict[str, Any]:
    return {
        "position_xyz_m": np.asarray(pose.translation, dtype=np.float64).tolist(),
        "quaternion_xyzw": quaternion_xyzw(pose.rotation),
    }


def alignment_pose(target: dict[str, Any], position_key: str) -> Any:
    position = finite_vector(target[position_key], 3, position_key)
    qx, qy, qz, qw = finite_vector(target["quaternion_xyzw"], 4, "quaternion_xyzw")
    quaternion_norm = float(np.linalg.norm([qx, qy, qz, qw]))
    if not math.isclose(quaternion_norm, 1.0, abs_tol=2.0e-3):
        raise ValueError(f"Alignment quaternion is not unit length: {quaternion_norm}")
    return cumotion.Pose3(cumotion.Rotation3(qw, qx, qy, qz), position)


def vertical_rotation(target: dict[str, Any]) -> Any:
    """Level the detected pose while retaining its jaw yaw about world Z."""
    matrix = np.asarray(target["rotation_matrix_row_major"], dtype=np.float64)
    if matrix.shape != (9,) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation_matrix_row_major must contain nine finite values")
    initial = matrix.reshape(3, 3)
    x_axis = np.array([initial[0, 0], initial[1, 0], 0.0], dtype=np.float64)
    if float(np.linalg.norm(x_axis)) <= 1.0e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-10):
        raise RuntimeError("Failed to construct an orthonormal vertical tool pose")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-10):
        raise RuntimeError("Vertical tool pose is not right-handed")
    return cumotion.Rotation3.from_matrix(rotation)


def destination_base(truth: dict[str, Any]) -> tuple[np.ndarray, str]:
    truth_id = int(truth["pin_id"])
    source = finite_vector(truth["base"], 3, f"truth {truth_id} base")
    if truth_id in DESTINATION_BASE_OVERRIDES_M:
        return DESTINATION_BASE_OVERRIDES_M[truth_id].copy(), "reachable_clear_relocation"
    return source, "source_base_verticalization"


def scene_geometry(
    result: dict[str, Any],
    task: dict[str, Any],
    *,
    omitted_truth_id: int | None,
    verticalized_ids: set[int],
) -> tuple[Any, list[dict[str, Any]]]:
    """Build the primitive planning world for one ready-to-ready cycle.

    The currently held specimen is omitted because cuMotion 1.1 has no attached
    payload in this standalone path.  Pins completed in earlier cycles are moved
    to their vertical truth-base pose before planning the next cycle.
    """
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
    walls = [
        (
            np.array(
                [
                    center_x - size_x / 2.0 - wall_thickness / 2.0,
                    center_y,
                    foam_z + wall_height / 2.0,
                ]
            ),
            np.array([wall_thickness, size_y + 2.0 * wall_thickness, wall_height]),
        ),
        (
            np.array(
                [
                    center_x + size_x / 2.0 + wall_thickness / 2.0,
                    center_y,
                    foam_z + wall_height / 2.0,
                ]
            ),
            np.array([wall_thickness, size_y + 2.0 * wall_thickness, wall_height]),
        ),
        (
            np.array(
                [
                    center_x,
                    center_y - size_y / 2.0 - wall_thickness / 2.0,
                    foam_z + wall_height / 2.0,
                ]
            ),
            np.array([size_x, wall_thickness, wall_height]),
        ),
        (
            np.array(
                [
                    center_x,
                    center_y + size_y / 2.0 + wall_thickness / 2.0,
                    foam_z + wall_height / 2.0,
                ]
            ),
            np.array([size_x, wall_thickness, wall_height]),
        ),
    ]
    for position, sides in walls:
        add_cuboid(
            world,
            specs,
            role="tray_wall",
            position=position,
            side_lengths=sides,
        )

    padding = float(collision["specimen_padding_m"])
    shaft_width = float(collision["other_pin_shaft_width_m"])
    head_radius = float(collision["other_pin_head_radius_m"])
    for truth in result["scene"]["truth"]:
        truth_id = int(truth["pin_id"])
        if truth_id == omitted_truth_id:
            continue
        source_base = finite_vector(truth["base"], 3, f"truth {truth_id} base")
        base = (
            destination_base(truth)[0]
            if truth_id in verticalized_ids
            else source_base
        )
        initial_axis = normalized(truth["axis_up"], f"truth {truth_id} axis_up")
        axis = (
            np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if truth_id in verticalized_ids
            else initial_axis
        )
        length = float(truth["length_m"])
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError(f"truth {truth_id} has invalid pin length")
        specimen_center_initial = finite_vector(
            truth["specimen_center"], 3, f"truth {truth_id} specimen_center"
        )
        axial_center = float(np.dot(specimen_center_initial - source_base, initial_axis))
        specimen_center = (
            base + axis * axial_center
            if truth_id in verticalized_ids
            else specimen_center_initial
        )
        specimen_sides = (
            2.0 * finite_vector(
                truth["specimen_radii"], 3, f"truth {truth_id} specimen_radii"
            )
            + 2.0 * padding
        )
        add_cuboid(
            world,
            specs,
            role="specimen_body",
            source_id=truth_id,
            position=specimen_center,
            side_lengths=specimen_sides,
        )
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
            position=base + axis * length,
            radius=head_radius,
        )
    return world, specs


def validate_source(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    input_paths = [
        args.result_json,
        args.urdf,
        args.xrdf,
        args.asset_manifest,
        args.planner_config,
        args.task_config,
    ]
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing multi-pin inputs:\n  " + "\n  ".join(missing))
    if getattr(cumotion, "__version__", None) != EXPECTED_CUMOTION_VERSION:
        raise RuntimeError(
            f"Expected cuMotion {EXPECTED_CUMOTION_VERSION}; found "
            f"{getattr(cumotion, '__version__', 'unknown')}"
        )
    if args.control_dt_seconds <= 0.0 or not math.isfinite(args.control_dt_seconds):
        raise ValueError("control_dt_seconds must be finite and positive")
    if args.validation_dt_seconds <= 0.0 or not math.isfinite(args.validation_dt_seconds):
        raise ValueError("validation_dt_seconds must be finite and positive")
    if args.execution_time_scale < 1.0 or not math.isfinite(args.execution_time_scale):
        raise ValueError("execution_time_scale must be finite and at least 1.0")
    if args.maximum_control_step_rad <= 0.0 or not math.isfinite(
        args.maximum_control_step_rad
    ):
        raise ValueError("maximum_control_step_rad must be finite and positive")

    result = json.loads(args.result_json.read_text(encoding="utf-8"))
    task = read_yaml(args.task_config)
    if result.get("seed") != 1407:
        raise ValueError(f"Expected deterministic synthetic seed 1407; found {result.get('seed')}")
    if result.get("frames", {}).get("target_frame") != "base":
        raise ValueError("Synthetic alignment targets must be expressed in base")
    if task.get("format_version") != 1:
        raise ValueError("Synthetic task config must use format_version: 1")
    scope = task.get("scope", {})
    for key in ("camera_or_depth_used", "ros_used", "real_robot_commanded"):
        if scope.get(key) is not False:
            raise ValueError(f"Task scope must explicitly set {key}: false")

    truth_by_id = {int(item["pin_id"]): item for item in result["scene"]["truth"]}
    if not set(SPECIMEN_IDS).issubset(truth_by_id):
        raise ValueError("Synthetic truth does not contain every requested specimen ID 1..7")
    matches = result["evaluation"]["matches"]
    detection_by_truth = {int(item["truth_id"]): int(item["detection_id"]) for item in matches}
    if len(detection_by_truth) != len(matches):
        raise ValueError("Each truth pin must have exactly one detection match")
    targets_by_detection = {
        int(item["detection_id"]): item for item in result["alignment"]["targets"]
    }
    targets_by_truth: dict[int, dict[str, Any]] = {}
    for truth_id in SPECIMEN_IDS:
        detection_id = detection_by_truth.get(truth_id)
        if detection_id is None or detection_id not in targets_by_detection:
            raise ValueError(f"Truth specimen {truth_id} has no matched alignment target")
        target = targets_by_detection[detection_id]
        if target.get("frame_id") != "base":
            raise ValueError(f"Truth specimen {truth_id} target is not in base")
        targets_by_truth[truth_id] = target
    return result, task, truth_by_id, targets_by_truth


def stage_target_poses(
    truth: dict[str, Any],
    target: dict[str, Any],
    alignment_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    poses = {
        stage: alignment_pose(target, key)
        for stage, key in POSE_STAGE_SOURCE_KEYS.items()
    }
    base, placement_label = destination_base(truth)
    length = float(truth["length_m"])
    grip_below_head = float(alignment_config["grip_below_head"])
    lift_distance = float(alignment_config["lift_distance"])
    remaining = length - grip_below_head
    if remaining <= 0.0:
        raise ValueError("grip_below_head must be shorter than the selected pin")
    vertical_grasp = base + np.array([0.0, 0.0, remaining])
    vertical_lift = vertical_grasp + np.array([0.0, 0.0, lift_distance])
    rotation = vertical_rotation(target)
    poses.update(
        {
            "reorient_vertical": cumotion.Pose3(rotation, vertical_lift),
            "descend_vertical": cumotion.Pose3(rotation, vertical_grasp),
            "retreat_vertical": cumotion.Pose3(rotation, vertical_lift),
        }
    )
    geometry = {
        "grip_below_head_m": grip_below_head,
        "lift_distance_m": lift_distance,
        "remaining_pin_end_z_from_pinch_m": remaining,
        "placement_label": placement_label,
        "vertical_grasp_position_xyz_m": vertical_grasp.tolist(),
        "vertical_lift_position_xyz_m": vertical_lift.tolist(),
        "vertical_quaternion_xyzw": quaternion_xyzw(rotation),
    }
    return poses, geometry


def force_exact_sample_endpoints(
    samples: list[dict[str, Any]], start: np.ndarray, end: np.ndarray
) -> None:
    zero = np.zeros_like(start).tolist()
    samples[0]["joint_positions"] = start.tolist()
    samples[0]["joint_velocities"] = zero
    samples[-1]["joint_positions"] = end.tolist()
    samples[-1]["joint_velocities"] = zero


def plan_stage(
    *,
    name: str,
    planner: Any,
    trajectory_generator: Any,
    inspector: Any,
    kinematics: Any,
    current: np.ndarray,
    target_pose: Any | None,
    ready: np.ndarray,
    control_dt: float,
    validation_dt: float,
    execution_time_scale: float,
    maximum_allowed_control_step: float,
    required_clearance: float,
    translation_tolerance: float,
    orientation_tolerance: float,
) -> tuple[dict[str, Any], np.ndarray]:
    planner.reset()
    started = time.perf_counter_ns()
    if target_pose is None:
        planning_result = planner.plan_to_cspace_target(current, ready, True)
    else:
        planning_result = planner.plan_to_pose_target(current, target_pose, True)
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    if not planning_result.path_found:
        raise RuntimeError(f"{name}: cuMotion path not found")

    raw_path = [np.asarray(point, dtype=np.float64) for point in planning_result.path]
    interpolated_path = [
        np.asarray(point, dtype=np.float64) for point in planning_result.interpolated_path
    ]
    if len(raw_path) < 2:
        raise RuntimeError(f"{name}: cuMotion returned fewer than two path knots")
    raw_path[0] = current.copy()
    if target_pose is None:
        raw_path[-1] = ready.copy()
    final = raw_path[-1].copy()
    trajectory = trajectory_generator.generate_trajectory(raw_path)
    if trajectory is None:
        raise RuntimeError(f"{name}: time parameterization failed")

    trajectory_validation = validate_trajectory(
        trajectory,
        inspector,
        kinematics,
        validation_dt,
        execution_time_scale,
    )
    samples = samples_for_trajectory(trajectory, control_dt, execution_time_scale)
    force_exact_sample_endpoints(samples, current, final)
    maximum_step = maximum_control_step(samples)

    stage: dict[str, Any] = {
        "name": name,
        "path_found": True,
        "accepted": False,
        "planning_latency_ms": latency_ms,
        "start_joint_positions": current.tolist(),
        "end_joint_positions": final.tolist(),
        "raw_knots": len(raw_path),
        "interpolated_points": len(interpolated_path),
        "raw_path_float64_sha256": float64_sha256(raw_path),
        "trajectory_validation": trajectory_validation,
        "maximum_control_step_rad": maximum_step,
        "control_samples": samples,
        "control_samples_float64_sha256": float64_sha256(
            [np.asarray(sample["joint_positions"], dtype=np.float64) for sample in samples]
            + [np.asarray(sample["joint_velocities"], dtype=np.float64) for sample in samples]
        ),
    }
    if target_pose is None:
        joint_error = float(np.max(np.abs(final - ready)))
        goal_tolerance_met = joint_error <= 1.0e-12
        stage["target_joint_positions"] = ready.tolist()
        stage["goal_joint_error_rad"] = joint_error
    else:
        actual_pose = kinematics.pose(final, TOOL_FRAME)
        translation_error = float(
            np.linalg.norm(actual_pose.translation - target_pose.translation)
        )
        orientation_error = float(
            cumotion.Rotation3.distance(actual_pose.rotation, target_pose.rotation)
        )
        goal_tolerance_met = bool(
            translation_error <= translation_tolerance
            and orientation_error <= orientation_tolerance
        )
        stage["target_pin_grasp_tcp_pose"] = pose_record(target_pose)
        stage["goal_translation_error_m"] = translation_error
        stage["goal_orientation_error_rad"] = orientation_error
    stage["goal_tolerance_met"] = goal_tolerance_met

    accepted = bool(
        goal_tolerance_met
        and not trajectory_validation["sampled_self_collision"]
        and trajectory_validation["minimum_sampled_sphere_clearance_m"]
        >= required_clearance
        and trajectory_validation["derivative_limits_met"]
        and maximum_step <= maximum_allowed_control_step + 1.0e-12
    )
    stage["accepted"] = accepted
    if not accepted:
        reasons: list[str] = []
        if not goal_tolerance_met:
            reasons.append("goal_tolerance")
        if trajectory_validation["sampled_self_collision"]:
            reasons.append("sampled_self_collision")
        if (
            trajectory_validation["minimum_sampled_sphere_clearance_m"]
            < required_clearance
        ):
            reasons.append("sampled_obstacle_clearance")
        if not trajectory_validation["derivative_limits_met"]:
            reasons.append("derivative_limits")
        if maximum_step > maximum_allowed_control_step + 1.0e-12:
            reasons.append("control_step")
        raise RuntimeError(f"{name}: validation failed ({', '.join(reasons)})")
    return stage, final


def plan_specimen(
    *,
    robot_description: Any,
    kinematics: Any,
    trajectory_generator: Any,
    result: dict[str, Any],
    task: dict[str, Any],
    planner_config_path: Path,
    truth: dict[str, Any],
    target: dict[str, Any],
    verticalized_ids: set[int],
    ready: np.ndarray,
    args: argparse.Namespace,
    required_clearance: float,
) -> dict[str, Any]:
    truth_id = int(truth["pin_id"])
    world, obstacle_specs = scene_geometry(
        result,
        task,
        omitted_truth_id=truth_id,
        verticalized_ids=verticalized_ids,
    )
    planner_config = cumotion.create_motion_planner_config_from_file(
        str(planner_config_path),
        robot_description,
        TOOL_FRAME,
        world.add_world_view(),
    )
    planner_config.set_param("enable_self_collision_checking", True)
    planner = cumotion.create_motion_planner(planner_config)
    inspector = cumotion.create_robot_world_inspector(
        robot_description, world.add_world_view()
    )
    if inspector.in_self_collision(ready):
        raise RuntimeError(f"Specimen {truth_id}: ready configuration is in self-collision")
    if inspector.in_collision_with_obstacle(ready):
        raise RuntimeError(f"Specimen {truth_id}: ready configuration collides with the scene")

    target_poses, geometry = stage_target_poses(
        truth, target, result["alignment"]["config"]
    )
    current = ready.copy()
    stages: list[dict[str, Any]] = []
    for stage_name in STAGE_NAMES:
        stage, current = plan_stage(
            name=stage_name,
            planner=planner,
            trajectory_generator=trajectory_generator,
            inspector=inspector,
            kinematics=kinematics,
            current=current,
            target_pose=target_poses.get(stage_name),
            ready=ready,
            control_dt=args.control_dt_seconds,
            validation_dt=args.validation_dt_seconds,
            execution_time_scale=args.execution_time_scale,
            maximum_allowed_control_step=args.maximum_control_step_rad,
            required_clearance=required_clearance,
            translation_tolerance=float(
                task["planning"]["goal_translation_tolerance_m"]
            ),
            orientation_tolerance=float(
                task["planning"]["goal_orientation_tolerance_rad"]
            ),
        )
        stages.append(stage)

    if [stage["name"] for stage in stages] != STAGE_NAMES:
        raise RuntimeError(f"Specimen {truth_id}: stage ordering changed unexpectedly")
    if stages[0]["control_samples"][0]["joint_positions"] != ready.tolist():
        raise RuntimeError(f"Specimen {truth_id}: cycle does not start at ready")
    if stages[-1]["control_samples"][-1]["joint_positions"] != ready.tolist():
        raise RuntimeError(f"Specimen {truth_id}: cycle does not end exactly at ready")
    for previous, following in zip(stages, stages[1:]):
        if (
            previous["control_samples"][-1]["joint_positions"]
            != following["control_samples"][0]["joint_positions"]
        ):
            raise RuntimeError(f"Specimen {truth_id}: adjacent stages are discontinuous")

    initial_axis = normalized(
        target["pin_axis_up"], f"truth {truth_id} detected pin_axis_up"
    )
    truth_axis = normalized(truth["axis_up"], f"truth {truth_id} truth axis_up")
    truth_source_base = finite_vector(truth["base"], 3, "truth base")
    placed_base, placement_label = destination_base(truth)
    planned_source_base = finite_vector(
        target["grasp_position"], 3, "alignment grasp_position"
    ) + geometry["remaining_pin_end_z_from_pinch_m"] * finite_vector(
        target["tool_z_axis_robot"], 3, "alignment tool_z_axis_robot"
    )
    minimum_clearance = min(
        float(stage["trajectory_validation"]["minimum_sampled_sphere_clearance_m"])
        for stage in stages
    )
    return {
        "specimen_id": truth_id,
        "source_detection_id": int(target["detection_id"]),
        "frame_id": "base",
        "initial_axis_up": initial_axis.tolist(),
        "truth_axis_up": truth_axis.tolist(),
        "final_axis_up": [0.0, 0.0, 1.0],
        "source_base_xyz_m": planned_source_base.tolist(),
        "truth_source_base_xyz_m": truth_source_base.tolist(),
        "source_base_truth_error_m": float(
            np.linalg.norm(planned_source_base - truth_source_base)
        ),
        "base_xyz_m": placed_base.tolist(),
        "placement_label": placement_label,
        "pin_length_m": float(truth["length_m"]),
        **geometry,
        "source_alignment_target": target,
        "accepted": True,
        "minimum_sampled_sphere_clearance_m": minimum_clearance,
        "planning_collision_obstacles": obstacle_specs,
        "stages": stages,
    }


def main() -> int:
    args = build_parser().parse_args()
    for name in (
        "result_json",
        "urdf",
        "xrdf",
        "asset_manifest",
        "planner_config",
        "task_config",
        "output",
    ):
        setattr(args, name, resolved(getattr(args, name)))
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite multi-pin plan: {args.output}")

    result, task, truth_by_id, targets_by_truth = validate_source(args)
    required_clearance = (
        float(task["planning"]["required_sampled_sphere_clearance_m"])
        if args.required_clearance_m is None
        else float(args.required_clearance_m)
    )
    if required_clearance < 0.0 or not math.isfinite(required_clearance):
        raise ValueError("required_clearance_m must be finite and non-negative")

    cumotion.set_log_level(cumotion.LogLevel.ERROR)
    robot_description = cumotion.load_robot_from_file(str(args.xrdf), str(args.urdf))
    if robot_description.num_cspace_coords() != 6:
        raise RuntimeError("Multi-pin verticalization requires a six-axis robot")
    if TOOL_FRAME not in robot_description.tool_frame_names():
        raise RuntimeError(f"Required direct planning frame is absent from XRDF: {TOOL_FRAME}")
    kinematics = robot_description.kinematics()
    expected_joint_names = list(task["robot"]["joint_names"])
    actual_joint_names = [
        kinematics.cspace_coord_name(index)
        for index in range(kinematics.num_cspace_coords())
    ]
    if actual_joint_names != expected_joint_names:
        raise RuntimeError(f"Unexpected joint order: {actual_joint_names}")
    ready = finite_vector(task["robot"]["ready_joint_positions"], 6, "ready joints")
    trajectory_generator = cumotion.create_cspace_trajectory_generator(kinematics)

    specimens: list[dict[str, Any]] = []
    verticalized_ids: set[int] = set()
    for truth_id in SPECIMEN_IDS:
        print(f"Planning specimen {truth_id}/7...", flush=True)
        specimens.append(
            plan_specimen(
                robot_description=robot_description,
                kinematics=kinematics,
                trajectory_generator=trajectory_generator,
                result=result,
                task=task,
                planner_config_path=args.planner_config,
                truth=truth_by_id[truth_id],
                target=targets_by_truth[truth_id],
                verticalized_ids=verticalized_ids,
                ready=ready,
                args=args,
                required_clearance=required_clearance,
            )
        )
        verticalized_ids.add(truth_id)

    _, canonical_scene_obstacles = scene_geometry(
        result,
        task,
        omitted_truth_id=None,
        verticalized_ids=set(),
    )
    all_stages = [stage for specimen in specimens for stage in specimen["stages"]]
    all_samples = [sample for stage in all_stages for sample in stage["control_samples"]]
    aggregate_hash = float64_sha256(
        [np.asarray(sample["joint_positions"], dtype=np.float64) for sample in all_samples]
        + [np.asarray(sample["joint_velocities"], dtype=np.float64) for sample in all_samples]
    )
    input_paths = {
        "synthetic_result": args.result_json,
        "watson_qc_nominal_urdf": args.urdf,
        "watson_qc_nominal_xrdf": args.xrdf,
        "watson_qc_nominal_asset_manifest": args.asset_manifest,
        "planner_config": args.planner_config,
        "source_task_config": args.task_config,
    }
    payload = {
        "format_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "validation_scope": "offline_synthetic_multi_pin_verticalization_cumotion_1_1",
        "frame_id": "base",
        "planning_tool_frame": TOOL_FRAME,
        "ros_used": False,
        "watson_connected": False,
        "real_robot_commanded": False,
        "control_dt_seconds": args.control_dt_seconds,
        "validation_dt_seconds": args.validation_dt_seconds,
        "execution_time_scale": args.execution_time_scale,
        "maximum_control_step_rad": args.maximum_control_step_rad,
        "required_sampled_sphere_clearance_m": required_clearance,
        "ready_joint_positions": ready.tolist(),
        "joint_names": actual_joint_names,
        "specimen_ids": SPECIMEN_IDS,
        "specimens": specimens,
        "scene_obstacles": canonical_scene_obstacles,
        "scene_obstacle_count": len(canonical_scene_obstacles),
        "safety_scope": {
            "camera_or_depth_used": False,
            "ros_used": False,
            "watson_connected": False,
            "real_robot_commanded": False,
            "network_connection_opened": False,
            "simulation_only": True,
        },
        "model_status": {
            "tool_profile": "watson_qc_nominal",
            "pin_grasp_tcp_planned_directly": True,
            "attached_payload_collision_modelled": False,
            "collision_note": (
                "Robot spheres are checked against tray and non-target pin/specimen "
                "primitives; the held target payload is omitted during its cycle."
            ),
            "clearance_note": (
                "The explicit 4 mm simulation threshold is below the observed limiting "
                "ID7 reorient clearance (about 4.48348 mm); this is planning evidence, "
                "not a physically calibrated Watson clearance."
            ),
            "physical_calibration_status": "not_physically_calibrated_for_watson_execution",
        },
        "source_scene": {
            "seed": result["seed"],
            "pin_count": result["scene"]["pin_count"],
            "alignment_config": result["alignment"]["config"],
            "unmodified_truth_id": 0,
            "destination_base_overrides_m": {
                str(specimen_id): position.tolist()
                for specimen_id, position in DESTINATION_BASE_OVERRIDES_M.items()
            },
        },
        "provenance": {
            "command": [sys.executable, *sys.argv],
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cumotion_version": cumotion.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "input_artifacts": {
                name: artifact_record(path) for name, path in input_paths.items()
            },
        },
        "validation": {
            "all_stages_accepted": all(stage["accepted"] for stage in all_stages),
            "stage_count": len(all_stages),
            "control_sample_count": len(all_samples),
            "minimum_sampled_sphere_clearance_m": min(
                specimen["minimum_sampled_sphere_clearance_m"]
                for specimen in specimens
            ),
            "maximum_observed_control_step_rad": max(
                stage["maximum_control_step_rad"] for stage in all_stages
            ),
            "sampled_self_collision": any(
                stage["trajectory_validation"]["sampled_self_collision"]
                for stage in all_stages
            ),
            "derivative_limits_met": all(
                stage["trajectory_validation"]["derivative_limits_met"]
                for stage in all_stages
            ),
            "cycles_ready_to_ready": True,
            "adjacent_stage_positions_continuous": True,
            "control_samples_float64_sha256": aggregate_hash,
        },
    }
    if [item["specimen_id"] for item in specimens] != SPECIMEN_IDS:
        raise RuntimeError("Final specimen order is not exactly 1..7")
    if payload["validation"]["stage_count"] != len(SPECIMEN_IDS) * len(STAGE_NAMES):
        raise RuntimeError("Final plan does not contain exactly seven stages per specimen")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(f"Validated specimens: {len(specimens)}/7")
    print(f"Validated stages: {len(all_stages)}/{len(SPECIMEN_IDS) * len(STAGE_NAMES)}")
    print(f"Multi-pin verticalization plan: {args.output}")
    print(f"Plan sha256: {sha256_file(args.output)}")
    print("ROS used: false; Watson connected: false; real robot commanded: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
