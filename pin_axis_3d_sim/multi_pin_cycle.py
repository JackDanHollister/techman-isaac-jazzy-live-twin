"""Pure scheduling, validation, and evidence for a seven-specimen pin cycle."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np


ARM_JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 7))
EXPECTED_SPECIMEN_COUNT = 7
SOURCE_STAGE_ORDER = (
    "approach_tilted_pregrasp",
    "descend_tilted_grasp",
    "lift_tilted",
    "reorient_vertical",
    "descend_vertical",
    "retreat_vertical",
    "return_ready",
)
PHASE_ORDER = (
    "approach_tilted_pregrasp",
    "hold_tilted_pregrasp",
    "descend_tilted_grasp",
    "hold_tilted_grasp_open",
    "close_gripper",
    "hold_tilted_grasp_closed",
    "lift_tilted",
    "reorient_vertical",
    "hold_vertical_lift",
    "descend_vertical",
    "hold_vertical_closed",
    "release_pin",
    "open_gripper",
    "hold_vertical_open",
    "retreat_vertical",
    "return_ready",
    "hold_ready",
)


def float64_sha256(arrays: list[np.ndarray]) -> str:
    """Hash numeric evidence with an explicit, platform-independent dtype."""

    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.asarray(array, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _finite_vector(value: Any, size: int, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain {size} finite values")
    return vector


def _sample_vectors(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    # The multi-pin planner uses compact q/qd keys.  Accepting the descriptive
    # keys as well keeps the pure scheduler useful with archived plan fixtures.
    positions = sample.get("q", sample.get("joint_positions"))
    velocities = sample.get("qd", sample.get("joint_velocities"))
    return (
        _finite_vector(positions, 6, "Each arm sample position vector"),
        _finite_vector(velocities, 6, "Each arm sample velocity vector"),
    )


def _validate_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    if plan.get("format_version") != 1:
        raise ValueError("Multi-pin plan format_version must be 1")
    control_dt = float(plan.get("control_dt_seconds", 0.0))
    maximum_step = float(plan.get("maximum_control_step_rad", 0.0))
    if not math.isfinite(control_dt) or control_dt <= 0.0:
        raise ValueError("control_dt_seconds must be finite and positive")
    if not math.isfinite(maximum_step) or maximum_step <= 0.0:
        raise ValueError("maximum_control_step_rad must be finite and positive")
    ready = _finite_vector(plan.get("ready_joint_positions"), 6, "ready_joint_positions")

    specimen_ids = plan.get("specimen_ids")
    specimens = plan.get("specimens")
    if not isinstance(specimen_ids, list) or len(specimen_ids) != EXPECTED_SPECIMEN_COUNT:
        raise ValueError("Multi-pin plan must configure exactly seven specimen_ids")
    if len(set(specimen_ids)) != EXPECTED_SPECIMEN_COUNT:
        raise ValueError("specimen_ids must be unique")
    if not isinstance(specimens, list) or len(specimens) != EXPECTED_SPECIMEN_COUNT:
        raise ValueError("Multi-pin plan must configure exactly seven specimens")
    if [specimen.get("specimen_id") for specimen in specimens] != specimen_ids:
        raise ValueError("specimens must match specimen_ids in the configured order")

    world_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    for specimen in specimens:
        specimen_id = specimen["specimen_id"]
        initial_axis = _finite_vector(
            specimen.get("initial_axis_up"), 3, f"Specimen {specimen_id} initial_axis_up"
        )
        final_axis = _finite_vector(
            specimen.get("final_axis_up"), 3, f"Specimen {specimen_id} final_axis_up"
        )
        if not math.isclose(
            float(np.linalg.norm(initial_axis)), 1.0, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise ValueError(f"Specimen {specimen_id} initial_axis_up must be unit length")
        if not np.allclose(final_axis, world_up, rtol=0.0, atol=1.0e-9):
            raise ValueError(f"Specimen {specimen_id} final_axis_up must be world vertical")
        _finite_vector(specimen.get("base_xyz_m"), 3, f"Specimen {specimen_id} base_xyz_m")
        remaining_pin_end = float(specimen.get("remaining_pin_end_z_from_pinch_m", math.nan))
        if not math.isfinite(remaining_pin_end) or remaining_pin_end <= 0.0:
            raise ValueError(
                f"Specimen {specimen_id} remaining_pin_end_z_from_pinch_m must be positive"
            )

        stages = specimen.get("stages")
        if not isinstance(stages, list):
            raise ValueError(f"Specimen {specimen_id} stages must be a list")
        stage_names = tuple(stage.get("name") for stage in stages)
        if stage_names != SOURCE_STAGE_ORDER:
            raise ValueError(f"Specimen {specimen_id} has unexpected stage order: {stage_names}")

        previous_endpoint = ready
        for stage in stages:
            samples = stage.get("control_samples")
            if not isinstance(samples, list) or len(samples) < 2:
                raise ValueError(
                    f"Specimen {specimen_id} stage {stage['name']!r} needs at least two samples"
                )
            vectors = [_sample_vectors(sample) for sample in samples]
            if not np.allclose(vectors[0][0], previous_endpoint, rtol=0.0, atol=1.0e-12):
                raise ValueError(
                    f"Specimen {specimen_id} stage {stage['name']!r} is discontinuous"
                )
            previous_endpoint = vectors[-1][0]
        if not np.allclose(previous_endpoint, ready, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Specimen {specimen_id} return_ready does not end at ready")
    return specimens


def _command(
    specimen_id: Any,
    specimen_index: int,
    phase: str,
    arm_positions: np.ndarray,
    arm_velocities: np.ndarray,
    finger_position_m: float,
    *,
    attachment_event: str | None = None,
    endpoint: bool = False,
) -> dict[str, Any]:
    if phase not in PHASE_ORDER:
        raise ValueError(f"Unknown multi-pin phase: {phase}")
    if attachment_event not in {None, "attach", "release"}:
        raise ValueError(f"Unknown attachment event: {attachment_event}")
    return {
        "specimen_id": specimen_id,
        "specimen_index": int(specimen_index),
        "phase": phase,
        "arm_positions": np.asarray(arm_positions, dtype=np.float64).copy(),
        "arm_velocities": np.asarray(arm_velocities, dtype=np.float64).copy(),
        "finger_position_m": float(finger_position_m),
        "attachment_event": attachment_event,
        "endpoint": bool(endpoint),
    }


def build_multi_pin_cycle(
    plan: dict[str, Any],
    *,
    finger_open_m: float,
    finger_closed_m: float,
    finger_speed_m_s: float,
    hold_seconds: float = 0.25,
) -> list[dict[str, Any]]:
    """Schedule seven ready-to-ready verticalisation cycles in plan order."""

    specimens = _validate_plan(plan)
    control_dt = float(plan["control_dt_seconds"])
    for label, value in (
        ("finger_open_m", finger_open_m),
        ("finger_closed_m", finger_closed_m),
        ("finger_speed_m_s", finger_speed_m_s),
        ("hold_seconds", hold_seconds),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")
    if finger_open_m < 0.0 or finger_closed_m <= finger_open_m:
        raise ValueError("Finger closure must increase from a non-negative open position")
    if finger_speed_m_s <= 0.0:
        raise ValueError("finger_speed_m_s must be positive")
    if hold_seconds <= 0.0:
        raise ValueError("hold_seconds must be positive")

    hold_steps = max(1, int(round(hold_seconds / control_dt)))
    finger_motion_steps = max(
        1,
        int(math.ceil((finger_closed_m - finger_open_m) / finger_speed_m_s / control_dt)),
    )
    zeros = np.zeros(6, dtype=np.float64)
    commands: list[dict[str, Any]] = []

    for specimen_index, specimen in enumerate(specimens):
        specimen_id = specimen["specimen_id"]
        stages = {stage["name"]: stage["control_samples"] for stage in specimen["stages"]}

        def append_arm_stage(name: str, finger_position_m: float) -> None:
            samples = stages[name]
            for sample_index, sample in enumerate(samples):
                # Every later stage starts with the preceding endpoint.  Keep
                # the first ready sample of each specimen, but avoid duplicates
                # at internal stage boundaries.
                if name != SOURCE_STAGE_ORDER[0] and sample_index == 0:
                    continue
                positions, velocities = _sample_vectors(sample)
                commands.append(
                    _command(
                        specimen_id,
                        specimen_index,
                        name,
                        positions,
                        velocities,
                        finger_position_m,
                        endpoint=sample_index == len(samples) - 1,
                    )
                )

        def append_hold(phase: str, arm_positions: np.ndarray, finger_position_m: float) -> None:
            for hold_index in range(hold_steps):
                commands.append(
                    _command(
                        specimen_id,
                        specimen_index,
                        phase,
                        arm_positions,
                        zeros,
                        finger_position_m,
                        endpoint=hold_index == hold_steps - 1,
                    )
                )

        append_arm_stage("approach_tilted_pregrasp", finger_open_m)
        tilted_pregrasp = commands[-1]["arm_positions"]
        append_hold("hold_tilted_pregrasp", tilted_pregrasp, finger_open_m)

        append_arm_stage("descend_tilted_grasp", finger_open_m)
        tilted_grasp = commands[-1]["arm_positions"]
        append_hold("hold_tilted_grasp_open", tilted_grasp, finger_open_m)

        for finger_index in range(1, finger_motion_steps + 1):
            fraction = finger_index / finger_motion_steps
            finger_position = finger_open_m + fraction * (finger_closed_m - finger_open_m)
            commands.append(
                _command(
                    specimen_id,
                    specimen_index,
                    "close_gripper",
                    tilted_grasp,
                    zeros,
                    finger_position,
                    attachment_event="attach" if finger_index == finger_motion_steps else None,
                    endpoint=finger_index == finger_motion_steps,
                )
            )
        append_hold("hold_tilted_grasp_closed", tilted_grasp, finger_closed_m)

        append_arm_stage("lift_tilted", finger_closed_m)
        append_arm_stage("reorient_vertical", finger_closed_m)
        vertical_lift = commands[-1]["arm_positions"]
        append_hold("hold_vertical_lift", vertical_lift, finger_closed_m)

        append_arm_stage("descend_vertical", finger_closed_m)
        vertical_placed = commands[-1]["arm_positions"]
        append_hold("hold_vertical_closed", vertical_placed, finger_closed_m)
        commands.append(
            _command(
                specimen_id,
                specimen_index,
                "release_pin",
                vertical_placed,
                zeros,
                finger_closed_m,
                attachment_event="release",
                endpoint=True,
            )
        )

        for finger_index in range(1, finger_motion_steps + 1):
            fraction = finger_index / finger_motion_steps
            finger_position = finger_closed_m - fraction * (finger_closed_m - finger_open_m)
            commands.append(
                _command(
                    specimen_id,
                    specimen_index,
                    "open_gripper",
                    vertical_placed,
                    zeros,
                    finger_position,
                    endpoint=finger_index == finger_motion_steps,
                )
            )
        append_hold("hold_vertical_open", vertical_placed, finger_open_m)

        append_arm_stage("retreat_vertical", finger_open_m)
        append_arm_stage("return_ready", finger_open_m)
        ready = commands[-1]["arm_positions"]
        append_hold("hold_ready", ready, finger_open_m)

    validate_multi_pin_cycle(
        commands,
        plan=plan,
        finger_open_m=finger_open_m,
        finger_closed_m=finger_closed_m,
        finger_speed_m_s=finger_speed_m_s,
    )
    return commands


def validate_multi_pin_cycle(
    commands: list[dict[str, Any]],
    *,
    plan: dict[str, Any],
    finger_open_m: float,
    finger_closed_m: float,
    finger_speed_m_s: float,
) -> None:
    """Validate phase, attachment, motion, and ready-pose invariants."""

    specimens = _validate_plan(plan)
    if not commands:
        raise ValueError("Multi-pin cycle is empty")
    control_dt = float(plan["control_dt_seconds"])
    maximum_step = float(plan["maximum_control_step_rad"])
    ready = np.asarray(plan["ready_joint_positions"], dtype=np.float64)

    expected_ids = [specimen["specimen_id"] for specimen in specimens]
    observed_ids = list(dict.fromkeys(command.get("specimen_id") for command in commands))
    if observed_ids != expected_ids:
        raise ValueError("Command specimen order does not match the source plan")

    carrying = False
    for specimen_index, specimen_id in enumerate(expected_ids):
        subset = [command for command in commands if command.get("specimen_id") == specimen_id]
        if not subset:
            raise ValueError(f"Specimen {specimen_id} has no commands")
        if any(command.get("specimen_index") != specimen_index for command in subset):
            raise ValueError(f"Specimen {specimen_id} has an invalid specimen_index")
        observed_phases = tuple(dict.fromkeys(command.get("phase") for command in subset))
        if observed_phases != PHASE_ORDER:
            raise ValueError(f"Specimen {specimen_id} has unexpected phase order")

        events = [command for command in subset if command.get("attachment_event") is not None]
        if [command["attachment_event"] for command in events] != ["attach", "release"]:
            raise ValueError(f"Specimen {specimen_id} must attach and release exactly once")
        positions = np.asarray([command["arm_positions"] for command in subset], dtype=np.float64)
        velocities = np.asarray([command["arm_velocities"] for command in subset], dtype=np.float64)
        if (
            positions.shape != (len(subset), 6)
            or velocities.shape != (len(subset), 6)
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(velocities))
        ):
            raise ValueError(f"Specimen {specimen_id} has invalid arm commands")
        if not np.allclose(positions[0], ready, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Specimen {specimen_id} does not start at ready")
        if not np.allclose(positions[-1], ready, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Specimen {specimen_id} does not end at ready")

    all_positions = np.asarray([command["arm_positions"] for command in commands], dtype=np.float64)
    if float(np.max(np.abs(np.diff(all_positions, axis=0)))) > maximum_step + 1.0e-12:
        raise ValueError("Multi-pin cycle exceeds the arm command-step bound")

    finger_positions = np.asarray(
        [command["finger_position_m"] for command in commands], dtype=np.float64
    )
    if (
        not np.all(np.isfinite(finger_positions))
        or float(np.min(finger_positions)) < finger_open_m - 1.0e-12
        or float(np.max(finger_positions)) > finger_closed_m + 1.0e-12
    ):
        raise ValueError("Multi-pin cycle exceeds the finger limits")
    maximum_finger_step = finger_speed_m_s * control_dt
    if float(np.max(np.abs(np.diff(finger_positions)))) > maximum_finger_step + 1.0e-12:
        raise ValueError("Multi-pin cycle exceeds the finger speed bound")

    for command in commands:
        event = command.get("attachment_event")
        if event == "attach":
            if carrying or not math.isclose(command["finger_position_m"], finger_closed_m):
                raise ValueError("Attachment requires an empty, fully closed gripper")
            carrying = True
        if carrying and not math.isclose(
            command["finger_position_m"], finger_closed_m, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("Fingers must remain closed while carrying a specimen")
        if event == "release":
            if not carrying:
                raise ValueError("Cannot release when no specimen is attached")
            carrying = False
    if carrying:
        raise ValueError("Multi-pin cycle ends with a specimen still attached")


def multi_pin_cycle_evidence(
    commands: list[dict[str, Any]], plan: dict[str, Any]
) -> dict[str, Any]:
    """Return deterministic overall and per-specimen scheduling evidence."""

    specimens = _validate_plan(plan)
    control_dt = float(plan["control_dt_seconds"])
    arm_positions = [command["arm_positions"] for command in commands]
    arm_velocities = [command["arm_velocities"] for command in commands]
    finger_positions = np.asarray(
        [command["finger_position_m"] for command in commands], dtype=np.float64
    )
    world_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    ready = np.asarray(plan["ready_joint_positions"], dtype=np.float64)
    specimen_evidence: list[dict[str, Any]] = []

    for specimen in specimens:
        specimen_id = specimen["specimen_id"]
        indices = [
            index for index, command in enumerate(commands) if command["specimen_id"] == specimen_id
        ]
        subset = [commands[index] for index in indices]
        phase_endpoints = []
        for phase in PHASE_ORDER:
            phase_indices = [index for index in indices if commands[index]["phase"] == phase]
            endpoint = commands[phase_indices[-1]]
            phase_endpoints.append(
                {
                    "phase": phase,
                    "command_count": len(phase_indices),
                    "first_command_index": phase_indices[0],
                    "last_command_index": phase_indices[-1],
                    "arm_joint_positions": endpoint["arm_positions"].tolist(),
                    "finger_position_m": endpoint["finger_position_m"],
                    "attachment_event": endpoint["attachment_event"],
                }
            )
        initial_axis = np.asarray(specimen["initial_axis_up"], dtype=np.float64)
        final_axis = np.asarray(specimen["final_axis_up"], dtype=np.float64)
        initial_tilt = math.degrees(
            math.acos(float(np.clip(np.dot(initial_axis, world_up), -1.0, 1.0)))
        )
        final_dot = float(np.clip(np.dot(final_axis, world_up), -1.0, 1.0))
        specimen_evidence.append(
            {
                "specimen_id": specimen_id,
                "first_command_index": indices[0],
                "last_command_index": indices[-1],
                "command_count": len(indices),
                "initial_axis_up": initial_axis.tolist(),
                "final_axis_up": final_axis.tolist(),
                "initial_tilt_degrees": initial_tilt,
                "final_axis_error": float(np.linalg.norm(final_axis - world_up)),
                "final_axis_error_degrees": math.degrees(math.acos(final_dot)),
                "attach_count": sum(
                    command["attachment_event"] == "attach" for command in subset
                ),
                "release_count": sum(
                    command["attachment_event"] == "release" for command in subset
                ),
                "starts_at_ready": bool(
                    np.allclose(subset[0]["arm_positions"], ready, rtol=0.0, atol=1.0e-12)
                ),
                "ends_at_ready": bool(
                    np.allclose(subset[-1]["arm_positions"], ready, rtol=0.0, atol=1.0e-12)
                ),
                "phase_endpoints": phase_endpoints,
            }
        )

    stream_records = [
        {
            "specimen_id": command["specimen_id"],
            "specimen_index": command["specimen_index"],
            "phase": command["phase"],
            "attachment_event": command["attachment_event"],
            "endpoint": command["endpoint"],
        }
        for command in commands
    ]
    geometry_arrays: list[np.ndarray] = []
    for specimen in specimens:
        geometry_arrays.extend(
            (
                np.asarray(specimen["initial_axis_up"]),
                np.asarray(specimen["final_axis_up"]),
                np.asarray(specimen["base_xyz_m"]),
                np.asarray([specimen["remaining_pin_end_z_from_pinch_m"]]),
            )
        )
    return {
        "format_version": 1,
        "arm_joint_names": list(ARM_JOINT_NAMES),
        "specimen_count": len(specimens),
        "specimen_ids": list(plan["specimen_ids"]),
        "command_count": len(commands),
        "control_dt_seconds": control_dt,
        "cycle_duration_seconds": len(commands) * control_dt,
        "phase_order_per_specimen": list(PHASE_ORDER),
        "specimens": specimen_evidence,
        "arm_positions_float64_sha256": float64_sha256(arm_positions),
        "arm_positions_and_velocities_float64_sha256": float64_sha256(
            arm_positions + arm_velocities
        ),
        "finger_positions_float64_sha256": float64_sha256([finger_positions]),
        "specimen_geometry_float64_sha256": float64_sha256(geometry_arrays),
        "command_stream_sha256": hashlib.sha256(
            json.dumps(stream_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "maximum_arm_command_step_rad": float(
            np.max(np.abs(np.diff(np.asarray(arm_positions), axis=0)))
        ),
        "maximum_finger_command_step_m": float(np.max(np.abs(np.diff(finger_positions)))),
        "all_specimens_ready_to_ready": all(
            item["starts_at_ready"] and item["ends_at_ready"] for item in specimen_evidence
        ),
        "all_final_axes_vertical": all(
            item["final_axis_error"] <= 1.0e-9 for item in specimen_evidence
        ),
    }
