"""Pure scheduling and evidence helpers for the Isaac pin-grasp cycle."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np


ARM_JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 7))
PHASE_ORDER = (
    "approach_pregrasp",
    "hold_pregrasp",
    "descend_to_grasp",
    "hold_grasp_open",
    "close_gripper",
    "hold_grasp_closed",
    "lift_pin",
    "hold_lift",
    "replace_pin",
    "hold_replaced_closed",
    "release_pin",
    "open_gripper",
    "hold_replaced_open",
    "retreat_to_pregrasp",
    "return_ready",
    "hold_ready",
)


def float64_sha256(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.asarray(array, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _stage_samples(plan: dict[str, Any], name: str) -> list[dict[str, Any]]:
    matches = [stage for stage in plan["selected"]["stages"] if stage.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name!r} stage")
    samples = matches[0].get("control_samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError(f"Stage {name!r} has no usable control samples")
    return samples


def _sample_vectors(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(sample.get("joint_positions"), dtype=np.float64)
    velocities = np.asarray(sample.get("joint_velocities"), dtype=np.float64)
    if (
        positions.shape != (6,)
        or velocities.shape != (6,)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(velocities))
    ):
        raise ValueError("Each arm sample must contain six finite positions and velocities")
    return positions, velocities


def _command(
    phase: str,
    arm_positions: np.ndarray,
    arm_velocities: np.ndarray,
    finger_position_m: float,
    *,
    attachment_event: str | None = None,
    endpoint: bool = False,
) -> dict[str, Any]:
    if phase not in PHASE_ORDER:
        raise ValueError(f"Unknown grasp-cycle phase: {phase}")
    if attachment_event not in {None, "attach", "release"}:
        raise ValueError(f"Unknown attachment event: {attachment_event}")
    return {
        "phase": phase,
        "arm_positions": np.asarray(arm_positions, dtype=np.float64).copy(),
        "arm_velocities": np.asarray(arm_velocities, dtype=np.float64).copy(),
        "finger_position_m": float(finger_position_m),
        "attachment_event": attachment_event,
        "endpoint": bool(endpoint),
    }


def build_grasp_cycle(
    plan: dict[str, Any],
    *,
    finger_open_m: float,
    finger_closed_m: float,
    finger_speed_m_s: float,
    hold_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Build one reversible arm cycle with explicit close/attach/release phases."""

    control_dt = float(plan.get("control_dt_seconds", 0.0))
    if not math.isfinite(control_dt) or control_dt <= 0.0:
        raise ValueError("control_dt_seconds must be finite and positive")
    for label, value in (
        ("finger_open_m", finger_open_m),
        ("finger_closed_m", finger_closed_m),
        ("finger_speed_m_s", finger_speed_m_s),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")
    if finger_open_m < 0.0 or finger_closed_m <= finger_open_m:
        raise ValueError("Finger closure must increase from a non-negative open position")
    if finger_speed_m_s <= 0.0:
        raise ValueError("Finger speed must be positive")
    selected_hold = float(
        plan.get("stage_hold_seconds", 0.0) if hold_seconds is None else hold_seconds
    )
    if not math.isfinite(selected_hold) or selected_hold <= 0.0:
        raise ValueError("hold_seconds must be finite and positive")

    hold_steps = max(1, int(round(selected_hold / control_dt)))
    finger_motion_steps = max(
        1,
        int(math.ceil((finger_closed_m - finger_open_m) / finger_speed_m_s / control_dt)),
    )
    stages = {name: _stage_samples(plan, name) for name in ("pregrasp", "grasp", "lift")}
    commands: list[dict[str, Any]] = []

    def append_arm_stage(
        source_name: str,
        phase: str,
        *,
        reverse: bool,
        finger_position_m: float,
    ) -> None:
        source = list(reversed(stages[source_name])) if reverse else stages[source_name]
        for sample_index, sample in enumerate(source):
            if commands and sample_index == 0:
                continue
            positions, velocities = _sample_vectors(sample)
            commands.append(
                _command(
                    phase,
                    positions,
                    -velocities if reverse else velocities,
                    finger_position_m,
                    endpoint=sample_index == len(source) - 1,
                )
            )

    def append_hold(
        phase: str,
        arm_positions: np.ndarray,
        finger_position_m: float,
    ) -> None:
        zeros = np.zeros(6, dtype=np.float64)
        for index in range(hold_steps):
            commands.append(
                _command(
                    phase,
                    arm_positions,
                    zeros,
                    finger_position_m,
                    endpoint=index == hold_steps - 1,
                )
            )

    append_arm_stage(
        "pregrasp", "approach_pregrasp", reverse=False, finger_position_m=finger_open_m
    )
    pregrasp = commands[-1]["arm_positions"]
    append_hold("hold_pregrasp", pregrasp, finger_open_m)

    append_arm_stage(
        "grasp", "descend_to_grasp", reverse=False, finger_position_m=finger_open_m
    )
    grasp = commands[-1]["arm_positions"]
    append_hold("hold_grasp_open", grasp, finger_open_m)

    zeros = np.zeros(6, dtype=np.float64)
    for index in range(1, finger_motion_steps + 1):
        fraction = index / finger_motion_steps
        finger_position = finger_open_m + fraction * (finger_closed_m - finger_open_m)
        commands.append(
            _command(
                "close_gripper",
                grasp,
                zeros,
                finger_position,
                attachment_event="attach" if index == finger_motion_steps else None,
                endpoint=index == finger_motion_steps,
            )
        )
    append_hold("hold_grasp_closed", grasp, finger_closed_m)

    append_arm_stage(
        "lift", "lift_pin", reverse=False, finger_position_m=finger_closed_m
    )
    lift = commands[-1]["arm_positions"]
    append_hold("hold_lift", lift, finger_closed_m)

    append_arm_stage(
        "lift", "replace_pin", reverse=True, finger_position_m=finger_closed_m
    )
    replaced = commands[-1]["arm_positions"]
    append_hold("hold_replaced_closed", replaced, finger_closed_m)
    commands.append(
        _command(
            "release_pin",
            replaced,
            zeros,
            finger_closed_m,
            attachment_event="release",
            endpoint=True,
        )
    )

    for index in range(1, finger_motion_steps + 1):
        fraction = index / finger_motion_steps
        finger_position = finger_closed_m - fraction * (finger_closed_m - finger_open_m)
        commands.append(
            _command(
                "open_gripper",
                replaced,
                zeros,
                finger_position,
                endpoint=index == finger_motion_steps,
            )
        )
    append_hold("hold_replaced_open", replaced, finger_open_m)

    append_arm_stage(
        "grasp", "retreat_to_pregrasp", reverse=True, finger_position_m=finger_open_m
    )
    append_arm_stage(
        "pregrasp", "return_ready", reverse=True, finger_position_m=finger_open_m
    )
    ready = commands[-1]["arm_positions"]
    append_hold("hold_ready", ready, finger_open_m)
    validate_grasp_cycle(
        commands,
        plan=plan,
        finger_open_m=finger_open_m,
        finger_closed_m=finger_closed_m,
    )
    return commands


def validate_grasp_cycle(
    commands: list[dict[str, Any]],
    *,
    plan: dict[str, Any],
    finger_open_m: float,
    finger_closed_m: float,
) -> None:
    if not commands:
        raise ValueError("Grasp cycle is empty")
    observed_phases = tuple(dict.fromkeys(command["phase"] for command in commands))
    if observed_phases != PHASE_ORDER:
        raise ValueError(f"Unexpected phase order: {observed_phases}")
    attach_indices = [
        index for index, command in enumerate(commands) if command["attachment_event"] == "attach"
    ]
    release_indices = [
        index for index, command in enumerate(commands) if command["attachment_event"] == "release"
    ]
    if len(attach_indices) != 1 or len(release_indices) != 1:
        raise ValueError("Grasp cycle must contain one attach and one release event")
    if attach_indices[0] >= release_indices[0]:
        raise ValueError("Payload release must occur after attachment")

    positions = np.asarray([command["arm_positions"] for command in commands])
    if positions.shape != (len(commands), 6) or not np.all(np.isfinite(positions)):
        raise ValueError("Grasp-cycle arm positions are invalid")
    if not np.allclose(positions[0], positions[-1], rtol=0.0, atol=1.0e-12):
        raise ValueError("Grasp cycle does not return to its exact starting arm pose")
    maximum_step = float(np.max(np.abs(np.diff(positions, axis=0))))
    if maximum_step > float(plan["maximum_control_step_rad"]) + 1.0e-12:
        raise ValueError("Grasp cycle exceeds the source plan's arm command-step bound")

    finger_positions = np.asarray(
        [command["finger_position_m"] for command in commands], dtype=np.float64
    )
    if (
        not np.all(np.isfinite(finger_positions))
        or float(np.min(finger_positions)) < finger_open_m - 1.0e-12
        or float(np.max(finger_positions)) > finger_closed_m + 1.0e-12
    ):
        raise ValueError("Grasp cycle exceeds the finger travel range")
    if not math.isclose(float(finger_positions[attach_indices[0]]), finger_closed_m):
        raise ValueError("Payload attachment must occur only at full closure")
    if not math.isclose(float(finger_positions[release_indices[0]]), finger_closed_m):
        raise ValueError("Payload release must occur before the fingers open")


def cycle_evidence(commands: list[dict[str, Any]], control_dt_seconds: float) -> dict[str, Any]:
    arm_positions = [command["arm_positions"] for command in commands]
    arm_velocities = [command["arm_velocities"] for command in commands]
    finger_positions = np.asarray(
        [command["finger_position_m"] for command in commands], dtype=np.float64
    )
    phase_endpoints = []
    for phase in PHASE_ORDER:
        indices = [index for index, command in enumerate(commands) if command["phase"] == phase]
        endpoint = commands[indices[-1]]
        phase_endpoints.append(
            {
                "phase": phase,
                "first_command_index": indices[0],
                "last_command_index": indices[-1],
                "arm_joint_positions": endpoint["arm_positions"].tolist(),
                "finger_position_m": endpoint["finger_position_m"],
                "attachment_event": endpoint["attachment_event"],
            }
        )
    return {
        "arm_joint_names": list(ARM_JOINT_NAMES),
        "command_count": len(commands),
        "control_dt_seconds": float(control_dt_seconds),
        "cycle_duration_seconds": len(commands) * float(control_dt_seconds),
        "phase_order": list(PHASE_ORDER),
        "phase_endpoints": phase_endpoints,
        "arm_positions_float64_sha256": float64_sha256(arm_positions),
        "arm_positions_and_velocities_float64_sha256": float64_sha256(
            arm_positions + arm_velocities
        ),
        "finger_positions_float64_sha256": float64_sha256([finger_positions]),
        "maximum_arm_command_step_rad": float(
            np.max(np.abs(np.diff(np.asarray(arm_positions), axis=0)))
        ),
        "start_equals_end": bool(
            np.allclose(arm_positions[0], arm_positions[-1], rtol=0.0, atol=1.0e-12)
        ),
    }
