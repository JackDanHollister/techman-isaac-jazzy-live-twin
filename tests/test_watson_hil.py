from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import unittest

from pin_axis_3d_sim.watson_hil import (
    HIL_EVENT_PREFIX,
    HilCoordinator,
    HilMode,
    HilState,
    build_runner_command,
    parse_hil_event_line,
    sanitized_runner_environment,
    validate_hil_event,
)
from pin_axis_3d_sim.watson_multi_pin_execution import (
    EXECUTION_ARM_TOKEN,
    GRIPPER_EXECUTION_TOKEN,
)


ARENA_DIR = Path(__file__).resolve().parents[1]
RUNNER = ARENA_DIR / "scripts/run_watson_multi_pin_air_replay.py"
WRAPPER = ARENA_DIR / "scripts/run_watson_multi_pin_air_replay.sh"


def event(name: str, **fields):
    return {
        "schema_version": 1,
        "event_sequence": fields.pop("event_sequence", 1),
        "event": name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }


class HilEventTests(unittest.TestCase):
    def test_ordinary_runner_output_is_ignored(self) -> None:
        self.assertIsNone(parse_hil_event_line("Executing stage 1/50\n"))

    def test_stage_and_gripper_events_validate(self) -> None:
        stage = event(
            "stage_started",
            sequence_index=2,
            stage_name="descend_tilted_grasp",
            specimen_id=1,
        )
        line = HIL_EVENT_PREFIX + json.dumps(stage) + "\n"
        self.assertEqual(parse_hil_event_line(line), stage)
        gripper = event(
            "gripper_completed",
            action="close",
            completed=True,
            context="after_stage_2_descend_tilted_grasp",
            specimen_id=1,
            sequence_index=2,
        )
        self.assertEqual(validate_hil_event(gripper), gripper)

    def test_invalid_event_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "schema_version"):
            validate_hil_event({"event": "stage_started"})
        with self.assertRaisesRegex(ValueError, "unknown HIL event"):
            validate_hil_event(event("launch_shell"))
        bad = event(
            "gripper_completed",
            action="close",
            completed="yes",
            context="after_stage",
            specimen_id=1,
        )
        with self.assertRaisesRegex(ValueError, "Boolean"):
            validate_hil_event(bad)
        with self.assertRaisesRegex(ValueError, "invalid HIL event JSON"):
            parse_hil_event_line(HIL_EVENT_PREFIX + "{")

    def test_runner_and_wrapper_expose_opt_in_event_protocol(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        shell = WRAPPER.read_text(encoding="utf-8")
        self.assertIn('parser.add_argument(\n        "--hil-events"', source)
        for name in (
            "run_started",
            "stage_started",
            "stage_completed",
            "stage_failed",
            "gripper_started",
            "gripper_completed",
            "run_completed",
            "run_failed",
        ):
            self.assertIn(f'"{name}"', source)
        self.assertIn("--hil-events", shell)


class HilCoordinatorTests(unittest.TestCase):
    def test_play_is_one_shot_and_requires_arming(self) -> None:
        coordinator = HilCoordinator(HilMode.EXECUTE)
        self.assertEqual(coordinator.on_play(), "ignore")
        coordinator.arm()
        self.assertEqual(coordinator.state, HilState.ARMED)
        self.assertEqual(coordinator.on_play(), "launch")
        self.assertEqual(coordinator.on_play(), "ignore")
        coordinator.runner_started()
        completed = event(
            "run_completed",
            event_sequence=1,
            mode="execute",
            status="passed",
            motion_commanded=True,
        )
        coordinator.accept_event(completed)
        coordinator.runner_exited(0)
        self.assertEqual(coordinator.state, HilState.COMPLETED)
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            coordinator.arm()

    def test_stop_signals_runner_once(self) -> None:
        coordinator = HilCoordinator(HilMode.DRY_RUN)
        coordinator.arm()
        self.assertEqual(coordinator.on_play(), "launch")
        coordinator.runner_started()
        self.assertEqual(coordinator.on_stop(), "signal_stop")
        self.assertEqual(coordinator.on_stop(), "ignore")
        coordinator.accept_event(
            event(
                "run_failed",
                status="unexpected_failure_stop_verified",
                error="stop requested",
                physical_estop_required=False,
            )
        )
        coordinator.runner_exited(130)
        self.assertEqual(coordinator.state, HilState.STOPPED)

    def test_stop_without_final_runner_proof_fails_closed(self) -> None:
        coordinator = HilCoordinator(HilMode.EXECUTE)
        coordinator.arm()
        coordinator.on_play()
        coordinator.runner_started()
        coordinator.on_stop()
        coordinator.runner_exited(137)
        self.assertEqual(coordinator.state, HilState.FAILED)
        self.assertIn("without a verified", coordinator.failure)

    def test_stop_before_spawn_cancels_without_launching(self) -> None:
        coordinator = HilCoordinator(HilMode.DRY_RUN)
        coordinator.arm()
        coordinator.on_play()
        coordinator.cancel_before_spawn()
        self.assertEqual(coordinator.state, HilState.STOPPED)
        self.assertFalse(coordinator.stop_signal_sent)

    def test_estop_failure_is_preserved(self) -> None:
        coordinator = HilCoordinator(HilMode.EXECUTE)
        coordinator.arm()
        coordinator.on_play()
        coordinator.runner_started()
        failed = event(
            "run_failed",
            event_sequence=1,
            status="stop_unverified_use_physical_estop",
            error="stop proof failed",
            physical_estop_required=True,
        )
        coordinator.accept_event(failed)
        coordinator.runner_exited(3)
        self.assertEqual(coordinator.state, HilState.FAILED)
        self.assertTrue(coordinator.physical_estop_required)
        self.assertEqual(coordinator.failure, "stop proof failed")


class HilRunnerLaunchTests(unittest.TestCase):
    def test_child_environment_removes_isaac_overlays(self) -> None:
        cleaned = sanitized_runner_environment(
            {
                "HOME": "/tmp/home",
                "PYTHONPATH": "/isaac/rclpy",
                "LD_LIBRARY_PATH": "/isaac/lib",
                "AMENT_PREFIX_PATH": "/isaac",
                "ROS_DISTRO": "jazzy",
            }
        )
        self.assertEqual(cleaned["HOME"], "/tmp/home")
        self.assertNotIn("PYTHONPATH", cleaned)
        self.assertNotIn("LD_LIBRARY_PATH", cleaned)
        self.assertNotIn("AMENT_PREFIX_PATH", cleaned)
        self.assertNotIn("ROS_DISTRO", cleaned)
        self.assertEqual(cleaned["ROS_DOMAIN_ID"], "219")
        self.assertEqual(cleaned["ROS_AUTOMATIC_DISCOVERY_RANGE"], "LOCALHOST")
        self.assertEqual(cleaned["RMW_IMPLEMENTATION"], "rmw_fastrtps_cpp")

    def test_execute_child_command_contains_exact_authorization(self) -> None:
        report = ARENA_DIR / "outputs/example_hil_report.json"
        command = build_runner_command(
            wrapper=WRAPPER,
            mode=HilMode.EXECUTE,
            report=report,
            arm_token=EXECUTION_ARM_TOKEN,
            gripper_token=GRIPPER_EXECUTION_TOKEN,
            confirm_cell_clear=True,
        )
        self.assertEqual(command[0], str(WRAPPER.resolve()))
        self.assertIn("--hil-events", command)
        self.assertIn(EXECUTION_ARM_TOKEN, command)
        self.assertIn(GRIPPER_EXECUTION_TOKEN, command)
        self.assertIn("--confirm-cell-clear", command)

    def test_preview_never_builds_a_physical_command(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not launch"):
            build_runner_command(
                wrapper=WRAPPER,
                mode=HilMode.PREVIEW,
                report=ARENA_DIR / "outputs/not_created.json",
            )


if __name__ == "__main__":
    unittest.main()
