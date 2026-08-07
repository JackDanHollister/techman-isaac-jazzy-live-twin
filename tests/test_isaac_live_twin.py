from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

from pin_axis_3d_sim.live_twin import (
    EXPECTED_JOINT_NAMES,
    LiveJointStateBuffer,
    normalise_joint_state,
)


ARENA_DIR = Path(__file__).resolve().parents[1]


def load_viewer_module():
    path = ARENA_DIR / "scripts/run_isaac_live_twin.py"
    spec = importlib.util.spec_from_file_location("watson_isaac_live_twin", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load live-twin viewer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LiveTwinStateTests(unittest.TestCase):
    def test_joint_state_is_name_mapped_not_order_assumed(self) -> None:
        reverse_names = tuple(reversed(EXPECTED_JOINT_NAMES))
        reverse_positions = tuple(float(index) for index in reversed(range(6)))
        reverse_velocities = tuple(float(index) / 10.0 for index in reversed(range(6)))
        positions, velocities = normalise_joint_state(
            reverse_names,
            reverse_positions,
            reverse_velocities,
            joint_limits=[[-10.0, 10.0]] * 6,
        )
        self.assertEqual(positions, tuple(float(index) for index in range(6)))
        self.assertEqual(velocities, tuple(float(index) / 10.0 for index in range(6)))

    def test_malformed_or_out_of_limit_joint_state_is_rejected(self) -> None:
        zeros = [0.0] * 6
        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalise_joint_state(
                ["joint_1"] * 6,
                zeros,
                zeros,
                joint_limits=[[-1.0, 1.0]] * 6,
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            normalise_joint_state(
                EXPECTED_JOINT_NAMES,
                [0.0] * 5 + [math.nan],
                zeros,
                joint_limits=[[-1.0, 1.0]] * 6,
            )
        with self.assertRaisesRegex(ValueError, "outside imported limits"):
            normalise_joint_state(
                EXPECTED_JOINT_NAMES,
                [0.0] * 5 + [2.0],
                zeros,
                joint_limits=[[-1.0, 1.0]] * 6,
            )

    def test_buffer_freezes_by_classifying_old_data_stale(self) -> None:
        buffer = LiveJointStateBuffer(stale_after_seconds=0.25)
        self.assertEqual(buffer.status(now_monotonic_s=10.0), "WAITING")
        buffer.accept(
            EXPECTED_JOINT_NAMES,
            [0.0] * 6,
            [0.0] * 6,
            stamp_seconds=100,
            stamp_nanoseconds=5,
            joint_limits=[[-1.0, 1.0]] * 6,
            received_monotonic_s=10.0,
        )
        self.assertEqual(buffer.status(now_monotonic_s=10.25), "LIVE")
        self.assertEqual(buffer.status(now_monotonic_s=10.250001), "STALE")
        self.assertEqual(buffer.sample.positions, (0.0,) * 6)

    def test_invalid_update_does_not_replace_last_valid_sample(self) -> None:
        buffer = LiveJointStateBuffer()
        valid = buffer.accept(
            EXPECTED_JOINT_NAMES,
            [0.1] * 6,
            [0.0] * 6,
            stamp_seconds=1,
            stamp_nanoseconds=0,
            joint_limits=[[-1.0, 1.0]] * 6,
            received_monotonic_s=1.0,
        )
        with self.assertRaises(ValueError):
            buffer.accept(
                EXPECTED_JOINT_NAMES,
                [0.0] * 5 + [2.0],
                [0.0] * 6,
                stamp_seconds=1,
                stamp_nanoseconds=1,
                joint_limits=[[-1.0, 1.0]] * 6,
                received_monotonic_s=1.1,
            )
        self.assertIs(buffer.sample, valid)
        self.assertEqual(buffer.valid_messages, 1)
        self.assertEqual(buffer.invalid_messages, 1)


class LiveTwinLaunchTests(unittest.TestCase):
    def test_preflight_requires_fresh_read_only_watson_check(self) -> None:
        module = load_viewer_module()
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "check",
            "status": "check_passed",
            "motion_commanded": False,
            "ros_domain_id": "219",
            "ros_automatic_discovery_range": "LOCALHOST",
            "health_failures": [],
            "stable_health": {
                "robot_error": False,
                "robot_link": True,
                "error_code": 0,
                "feedback_joint_positions": [0.0] * 6,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "check.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            evidence = module.load_preflight_report(path)
            self.assertEqual(evidence["initial_joint_positions"], [0.0] * 6)
            report["motion_commanded"] = True
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "motion_commanded"):
                module.load_preflight_report(path)

    def test_viewer_source_has_no_robot_command_path_or_physics_step(self) -> None:
        source = (ARENA_DIR / "scripts/run_isaac_live_twin.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("create_subscription(", source)
        for forbidden in (
            "create_publisher(",
            "create_client(",
            "create_service(",
            "ActionClient(",
            "world.step(",
            "world.play(",
            "ArticulationAction(",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("world.pause()", source)
        self.assertIn("robot.set_joint_positions(", source)
        self.assertIn("update_articulations_kinematic()", source)
        self.assertIn("update_transformations(False, True, False)", source)
        self.assertIn("rendered_link_pose(", source)

    def test_quaternion_error_is_sign_invariant(self) -> None:
        module = load_viewer_module()
        first = [0.0, 0.0, 0.0, 1.0]
        second = [0.0, 0.0, 0.0, -1.0]
        self.assertEqual(module.quaternion_angular_error_radians(first, second), 0.0)

    def test_wrapper_requires_execution_disabled_and_guarded_preflight(self) -> None:
        source = (ARENA_DIR / "scripts/run_isaac_live_twin.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("allow_trajectory_execution", source)
        self.assertIn('"$execution_state" != *"False"*', source)
        self.assertIn("--mode check", source)
        self.assertIn("ROS_DOMAIN_ID=219", source)
        self.assertIn("ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST", source)


if __name__ == "__main__":
    unittest.main()
