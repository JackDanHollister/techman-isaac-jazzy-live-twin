from __future__ import annotations

import copy
import math
import unittest
from typing import Any

import numpy as np

from pin_axis_3d_sim.multi_pin_cycle import (
    PHASE_ORDER,
    SOURCE_STAGE_ORDER,
    build_multi_pin_cycle,
    multi_pin_cycle_evidence,
    validate_multi_pin_cycle,
)


CONTROL_DT_SECONDS = 0.1
MAXIMUM_CONTROL_STEP_RAD = 0.05
FINGER_OPEN_M = 0.0
FINGER_CLOSED_M = 0.02
FINGER_SPEED_M_S = 0.1


def _sample(q: np.ndarray, qd: np.ndarray) -> dict[str, list[float]]:
    return {"q": q.tolist(), "qd": qd.tolist()}


def synthetic_plan() -> dict[str, Any]:
    ready = np.zeros(6, dtype=np.float64)
    # Consecutive stage endpoints remain comfortably inside the 0.05 rad bound.
    endpoint_scalars = (0.01, 0.02, 0.03, 0.04, 0.03, 0.02, 0.0)
    specimen_ids = list(range(1, 8))
    specimens = []
    for specimen_index, specimen_id in enumerate(specimen_ids):
        angle_rad = math.radians(10.0 * specimen_id)
        initial_axis = [math.sin(angle_rad), 0.0, math.cos(angle_rad)]
        stages = []
        start = ready.copy()
        for stage_index, (name, endpoint_scalar) in enumerate(
            zip(SOURCE_STAGE_ORDER, endpoint_scalars, strict=True)
        ):
            endpoint = np.full(6, endpoint_scalar, dtype=np.float64)
            velocity = (endpoint - start) / CONTROL_DT_SECONDS
            stages.append(
                {
                    "name": name,
                    "control_samples": [
                        _sample(start, velocity),
                        _sample(endpoint, velocity),
                    ],
                }
            )
            start = endpoint
        specimens.append(
            {
                "specimen_id": specimen_id,
                "initial_axis_up": initial_axis,
                "final_axis_up": [0.0, 0.0, 1.0],
                "base_xyz_m": [0.1 * specimen_index, 0.2, 0.003],
                "remaining_pin_end_z_from_pinch_m": 0.004,
                "stages": stages,
            }
        )
    return {
        "format_version": 1,
        "control_dt_seconds": CONTROL_DT_SECONDS,
        "maximum_control_step_rad": MAXIMUM_CONTROL_STEP_RAD,
        "ready_joint_positions": ready.tolist(),
        "specimen_ids": specimen_ids,
        "specimens": specimens,
    }


def build_fixture() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan = synthetic_plan()
    commands = build_multi_pin_cycle(
        plan,
        finger_open_m=FINGER_OPEN_M,
        finger_closed_m=FINGER_CLOSED_M,
        finger_speed_m_s=FINGER_SPEED_M_S,
        hold_seconds=CONTROL_DT_SECONDS,
    )
    return plan, commands


class MultiPinCycleTests(unittest.TestCase):
    def test_seven_specimens_have_exact_phase_and_event_order(self) -> None:
        plan, commands = build_fixture()
        self.assertEqual(
            list(dict.fromkeys(command["specimen_id"] for command in commands)),
            list(range(1, 8)),
        )
        ready = np.asarray(plan["ready_joint_positions"])

        for specimen_index, specimen_id in enumerate(plan["specimen_ids"]):
            subset = [command for command in commands if command["specimen_id"] == specimen_id]
            self.assertEqual(len(subset), 20)
            self.assertTrue(all(command["specimen_index"] == specimen_index for command in subset))
            self.assertEqual(
                tuple(dict.fromkeys(command["phase"] for command in subset)), PHASE_ORDER
            )
            self.assertEqual(
                [command["attachment_event"] for command in subset if command["attachment_event"]],
                ["attach", "release"],
            )
            np.testing.assert_array_equal(subset[0]["arm_positions"], ready)
            np.testing.assert_array_equal(subset[-1]["arm_positions"], ready)

    def test_gripper_remains_closed_while_carrying_and_respects_speed(self) -> None:
        _, commands = build_fixture()
        maximum_finger_step = FINGER_SPEED_M_S * CONTROL_DT_SECONDS
        finger_positions = np.asarray(
            [command["finger_position_m"] for command in commands], dtype=np.float64
        )
        self.assertGreaterEqual(float(np.min(finger_positions)), FINGER_OPEN_M)
        self.assertLessEqual(float(np.max(finger_positions)), FINGER_CLOSED_M)
        self.assertLessEqual(
            float(np.max(np.abs(np.diff(finger_positions)))), maximum_finger_step + 1.0e-12
        )

        carrying = False
        for command in commands:
            if command["attachment_event"] == "attach":
                carrying = True
            if carrying:
                self.assertAlmostEqual(command["finger_position_m"], FINGER_CLOSED_M)
            if command["attachment_event"] == "release":
                carrying = False
        self.assertFalse(carrying)

    def test_arm_commands_are_finite_and_bounded(self) -> None:
        plan, commands = build_fixture()
        positions = np.asarray([command["arm_positions"] for command in commands])
        velocities = np.asarray([command["arm_velocities"] for command in commands])
        self.assertEqual(positions.shape, (140, 6))
        self.assertEqual(velocities.shape, (140, 6))
        self.assertTrue(np.all(np.isfinite(positions)))
        self.assertTrue(np.all(np.isfinite(velocities)))
        self.assertLessEqual(
            float(np.max(np.abs(np.diff(positions, axis=0)))),
            float(plan["maximum_control_step_rad"]),
        )

    def test_evidence_is_deterministic_and_records_verticalisation(self) -> None:
        plan, commands = build_fixture()
        evidence = multi_pin_cycle_evidence(commands, plan)
        repeated = multi_pin_cycle_evidence(
            build_multi_pin_cycle(
                plan,
                finger_open_m=FINGER_OPEN_M,
                finger_closed_m=FINGER_CLOSED_M,
                finger_speed_m_s=FINGER_SPEED_M_S,
                hold_seconds=CONTROL_DT_SECONDS,
            ),
            plan,
        )
        self.assertEqual(evidence, repeated)
        self.assertEqual(evidence["specimen_count"], 7)
        self.assertEqual(evidence["command_count"], 140)
        self.assertTrue(evidence["all_specimens_ready_to_ready"])
        self.assertTrue(evidence["all_final_axes_vertical"])
        self.assertAlmostEqual(evidence["maximum_arm_command_step_rad"], 0.02)
        for hash_name in (
            "arm_positions_float64_sha256",
            "arm_positions_and_velocities_float64_sha256",
            "finger_positions_float64_sha256",
            "specimen_geometry_float64_sha256",
            "command_stream_sha256",
        ):
            self.assertEqual(len(evidence[hash_name]), 64)

        for specimen_id, specimen in enumerate(evidence["specimens"], start=1):
            self.assertEqual(specimen["specimen_id"], specimen_id)
            self.assertAlmostEqual(specimen["initial_tilt_degrees"], 10.0 * specimen_id)
            self.assertEqual(specimen["final_axis_error"], 0.0)
            self.assertEqual(specimen["final_axis_error_degrees"], 0.0)
            self.assertEqual(specimen["attach_count"], 1)
            self.assertEqual(specimen["release_count"], 1)
            self.assertEqual(len(specimen["phase_endpoints"]), len(PHASE_ORDER))
            self.assertTrue(all(item["command_count"] >= 1 for item in specimen["phase_endpoints"]))

    def test_invalid_axis_and_carrying_state_are_rejected(self) -> None:
        plan = synthetic_plan()
        invalid_axis_plan = copy.deepcopy(plan)
        invalid_axis_plan["specimens"][0]["final_axis_up"] = [0.0, 1.0, 0.0]
        with self.assertRaisesRegex(ValueError, "final_axis_up must be world vertical"):
            build_multi_pin_cycle(
                invalid_axis_plan,
                finger_open_m=FINGER_OPEN_M,
                finger_closed_m=FINGER_CLOSED_M,
                finger_speed_m_s=FINGER_SPEED_M_S,
            )

        _, commands = build_fixture()
        invalid_commands = copy.deepcopy(commands)
        carrying_command = next(
            command for command in invalid_commands if command["phase"] == "lift_tilted"
        )
        carrying_command["finger_position_m"] = FINGER_CLOSED_M - 0.005
        with self.assertRaisesRegex(ValueError, "Fingers must remain closed while carrying"):
            validate_multi_pin_cycle(
                invalid_commands,
                plan=plan,
                finger_open_m=FINGER_OPEN_M,
                finger_closed_m=FINGER_CLOSED_M,
                finger_speed_m_s=FINGER_SPEED_M_S,
            )


if __name__ == "__main__":
    unittest.main()
