"""Alignment target generation for pin-axis picking."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .detection import PinAxisDetection
from .geometry import (
    Plane,
    quaternion_xyzw_from_matrix,
    rotation_matrix_from_z_axis,
    serializable_vec,
)


@dataclass(frozen=True)
class AlignmentConfig:
    grip_below_head: float = 0.004
    pregrasp_clearance: float = 0.045
    lift_distance: float = 0.030
    centerline_visual_length: float = 0.075


@dataclass(frozen=True)
class GripperTarget:
    detection_id: int
    grip_point: np.ndarray
    pregrasp_position: np.ndarray
    grasp_position: np.ndarray
    lift_position: np.ndarray
    tool_z_axis_robot: np.ndarray
    rotation_matrix: np.ndarray
    quaternion_xyzw: np.ndarray
    pin_axis_up: np.ndarray

    def to_dict(self) -> dict:
        return {
            "detection_id": int(self.detection_id),
            "frame_id": "base",
            "tcp_model": "virtual_gripper_pinch_center",
            "convention": "tool local +Z points downward along approach, opposite pin_axis_up",
            "grip_point": serializable_vec(self.grip_point),
            "pregrasp_position": serializable_vec(self.pregrasp_position),
            "grasp_position": serializable_vec(self.grasp_position),
            "lift_position": serializable_vec(self.lift_position),
            "tool_z_axis_robot": serializable_vec(self.tool_z_axis_robot),
            "pin_axis_up": serializable_vec(self.pin_axis_up),
            "quaternion_xyzw": serializable_vec(self.quaternion_xyzw),
            "rotation_matrix_row_major": serializable_vec(self.rotation_matrix.reshape(-1)),
        }


def make_gripper_target(
    detection: PinAxisDetection,
    plane: Plane,
    *,
    config: AlignmentConfig | None = None,
) -> GripperTarget:
    """Create pregrasp/grasp/lift poses aligned to a detected pin axis."""
    cfg = config or AlignmentConfig()
    axis_up = detection.axis_up
    grip_point = detection.head - cfg.grip_below_head * axis_up
    pregrasp_position = grip_point + cfg.pregrasp_clearance * axis_up
    grasp_position = grip_point
    lift_position = grip_point + cfg.lift_distance * axis_up

    # Virtual tool local +Z points down from the flange/TCP toward the pin.
    tool_z_axis = -axis_up
    rot = rotation_matrix_from_z_axis(tool_z_axis, x_hint=plane.u_axis)
    quat = quaternion_xyzw_from_matrix(rot)

    return GripperTarget(
        detection_id=detection.detection_id,
        grip_point=grip_point,
        pregrasp_position=pregrasp_position,
        grasp_position=grasp_position,
        lift_position=lift_position,
        tool_z_axis_robot=tool_z_axis,
        rotation_matrix=rot,
        quaternion_xyzw=quat,
        pin_axis_up=axis_up,
    )


def make_targets(
    detections: list[PinAxisDetection],
    plane: Plane,
    *,
    config: AlignmentConfig | None = None,
) -> list[GripperTarget]:
    return [make_gripper_target(det, plane, config=config) for det in detections]


def alignment_metadata(config: AlignmentConfig | None = None) -> dict:
    cfg = config or AlignmentConfig()
    return {
        "config": asdict(cfg),
        "notes": [
            "Targets are virtual gripper pinch-center TCP poses.",
            "A real flange->gripper_tcp transform must be added before hardware use.",
            "Pregrasp/grasp/lift positions are expressed in the configured target frame.",
        ],
    }
