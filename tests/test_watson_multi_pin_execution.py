from __future__ import annotations

import contextlib
import copy
from dataclasses import replace
import importlib.util
import io
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest

import pin_axis_3d_sim.watson_multi_pin_retime as retime_module
from pin_axis_3d_sim.watson_guard import HealthSnapshot
from pin_axis_3d_sim.watson_multi_pin_execution import (
    DEFAULT_INGRESS_ARTIFACT,
    DEFAULT_RETIMED_ARTIFACT,
    EXECUTION_ARM_TOKEN,
    GRIPPER_EXECUTION_TOKEN,
    GRIPPER_POLICY,
    LIVE_POSITION_ENVELOPE_MARGIN_RAD,
    PVTPoint,
    StageSpec,
    build_robot_trajectory,
    exact_execute_project_speed_failures,
    exact_tool_audit_failures,
    gripper_after_stage_hook,
    live_stage_failures,
    load_execution_bundle,
    validate_stage_live_first_wire_cubic,
    validate_execution_authorization,
    validate_robot_trajectory,
)


ARENA_DIR = Path(__file__).resolve().parents[1]
RUNNER = ARENA_DIR / "scripts/run_watson_multi_pin_air_replay.py"
WRAPPER = ARENA_DIR / "scripts/run_watson_multi_pin_air_replay.sh"


def load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "test_run_watson_multi_pin_air_replay",
        RUNNER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDuration:
    def __init__(self, *, sec: int = 0, nanosec: int = 0) -> None:
        self.sec = sec
        self.nanosec = nanosec


class FakeJointTrajectoryPoint:
    def __init__(self) -> None:
        self.positions = []
        self.velocities = []
        self.accelerations = []
        self.effort = []
        self.time_from_start = FakeDuration()


class FakeJointTrajectory:
    def __init__(self) -> None:
        self.joint_names = []
        self.points = []


class FakeMultiDOF:
    def __init__(self) -> None:
        self.joint_names = []
        self.points = []


class FakeRobotTrajectory:
    def __init__(self) -> None:
        self.joint_trajectory = FakeJointTrajectory()
        self.multi_dof_joint_trajectory = FakeMultiDOF()


def small_stage(name: str = "descend_tilted_grasp") -> StageSpec:
    start_positions = (0.0, 0.0, 1.5708, 0.0, 1.5708, 0.0)
    goal_positions = (
        0.0005,
        -0.0005,
        1.5713,
        0.0005,
        1.5703,
        0.0005,
    )
    points = (
        PVTPoint(
            source_sample_index=0,
            time_s=0.0,
            positions=start_positions,
            velocities=(0.0,) * 6,
        ),
        PVTPoint(
            source_sample_index=1,
            time_s=0.05,
            positions=goal_positions,
            velocities=(0.0,) * 6,
        ),
    )
    position_text = tuple(f"{math.degrees(value):.5f}" for value in goal_positions)
    velocity_text = ("0.00000",) * 6
    wire_positions = tuple(math.radians(float(value)) for value in position_text)
    return StageSpec(
        sequence_index=1,
        kind="seven_pin_air_replay",
        specimen_id=1,
        stage_index=1,
        stage_name=name,
        points=points,
        position_minimum_rad=(
            0.0,
            -0.0005,
            1.5708,
            0.0,
            1.5703,
            0.0,
        ),
        position_maximum_rad=(
            0.0005,
            0.0,
            1.5713,
            0.0005,
            1.5708,
            0.0005,
        ),
        maximum_cubic_velocity_rad_s=(0.015,) * 6,
        points_sha256="0" * 64,
        first_serialized_wire_point={
            "source_sample_index": 1,
            "wire_segment_duration_seconds_fixed_5": "0.05000",
            "wire_segment_duration_seconds": 0.05,
            "wire_segment_duration_ticks_1e5": 5000,
            "cumulative_wire_time_seconds": 0.05,
            "cumulative_wire_time_ticks_1e5": 5000,
            "joint_positions_degrees_fixed_5": list(position_text),
            "joint_velocities_degrees_per_second_fixed_5": list(
                velocity_text
            ),
            "joint_positions_rad_after_wire_roundtrip": list(
                wire_positions
            ),
            "joint_velocities_rad_s_after_wire_roundtrip": [0.0] * 6,
        },
        serialized_wire_tokens_sha256="1" * 64,
    )


def snapshot(
    positions=(0.0005, -0.0005, 1.5713, 0.0005, 1.5703, 0.0005),
    velocities=(0.0,) * 6,
) -> HealthSnapshot:
    return HealthSnapshot(
        is_svr_connected=True,
        is_sct_connected=True,
        tmsrv_cperr=0,
        tmscript_cperr=0,
        tmsrv_dataerr=0,
        tmscript_dataerr=0,
        is_data_table_correct=True,
        robot_link=True,
        robot_error=False,
        project_run=True,
        project_pause=False,
        safetyguard_a=False,
        e_stop=False,
        error_code=0,
        project_speed=50,
        ma_mode=0,
        robot_light=20,
        joint_positions=tuple(positions),
        feedback_joint_positions=tuple(positions),
        joint_velocities=tuple(velocities),
        feedback_age_s=0.01,
        joint_state_age_s=0.01,
    )


class ExactArtifactBundleTests(unittest.TestCase):
    @unittest.skipUnless(
        retime_module.TECHMAN_WORKSPACE.exists()
        and DEFAULT_RETIMED_ARTIFACT.exists()
        and DEFAULT_INGRESS_ARTIFACT.exists(),
        "reviewed tm_driver workspace or private execution artifacts are absent",
    )
    def test_private_bundle_is_exact_ingress_plus_49_stages(self) -> None:
        bundle = load_execution_bundle()
        self.assertEqual(len(bundle.stages), 50)
        self.assertEqual(bundle.stages[0].stage_name, "tool_aware_ready_ingress")
        self.assertEqual(bundle.stages[-1].stage_name, "return_ready")
        for previous, current in zip(bundle.stages, bundle.stages[1:]):
            self.assertEqual(previous.goal_positions, current.start_positions)
        for stage in bundle.stages:
            proof = validate_stage_live_first_wire_cubic(
                snapshot(positions=stage.start_positions),
                stage,
            )
            self.assertEqual(
                proof["status"],
                "validated_live_first_wire_cubic",
            )
        close_hooks = [
            gripper_after_stage_hook(stage)
            for stage in bundle.stages
            if stage.stage_name == "descend_tilted_grasp"
        ]
        open_hooks = [
            gripper_after_stage_hook(stage)
            for stage in bundle.stages
            if stage.stage_name == "descend_vertical"
        ]
        self.assertEqual(len(close_hooks), 7)
        self.assertEqual(len(open_hooks), 7)
        self.assertTrue(all(hook["action"] == "close" for hook in close_hooks))
        self.assertTrue(all(hook["action"] == "open" for hook in open_hooks))
        self.assertTrue(all(hook["executed"] is False for hook in close_hooks))
        self.assertTrue(all(hook["executed"] is False for hook in open_hooks))


class ExactTrajectoryMessageTests(unittest.TestCase):
    def types(self):
        return {
            "RobotTrajectory": FakeRobotTrajectory,
            "JointTrajectoryPoint": FakeJointTrajectoryPoint,
            "Duration": FakeDuration,
        }

    def test_builds_position_velocity_time_without_acceleration(self) -> None:
        stage = small_stage()
        trajectory = build_robot_trajectory(stage, self.types())
        validate_robot_trajectory(stage, trajectory)
        self.assertEqual(
            trajectory.joint_trajectory.joint_names,
            [f"joint_{index}" for index in range(1, 7)],
        )
        self.assertEqual(len(trajectory.joint_trajectory.points), 2)
        self.assertEqual(
            trajectory.joint_trajectory.points[-1].time_from_start.nanosec,
            50_000_000,
        )
        for point in trajectory.joint_trajectory.points:
            self.assertEqual(point.accelerations, [])
            self.assertEqual(point.effort, [])

    def test_rejects_invented_acceleration_or_changed_time(self) -> None:
        stage = small_stage()
        trajectory = build_robot_trajectory(stage, self.types())
        trajectory.joint_trajectory.points[1].accelerations = [0.0] * 6
        with self.assertRaisesRegex(ValueError, "invents acceleration"):
            validate_robot_trajectory(stage, trajectory)
        trajectory.joint_trajectory.points[1].accelerations = []
        trajectory.joint_trajectory.points[1].time_from_start.nanosec += 1
        with self.assertRaisesRegex(ValueError, "time changed"):
            validate_robot_trajectory(stage, trajectory)


class AirReplayGateTests(unittest.TestCase):
    def test_execute_requires_immutable_token_and_cell_clear(self) -> None:
        with self.assertRaisesRegex(ValueError, "arm-token"):
            validate_execution_authorization(
                mode="execute",
                arm_token="wrong",
                gripper_token=GRIPPER_EXECUTION_TOKEN,
                confirm_cell_clear=True,
                namespace="/watson",
            )
        with self.assertRaisesRegex(ValueError, "confirm-cell-clear"):
            validate_execution_authorization(
                mode="execute",
                arm_token=EXECUTION_ARM_TOKEN,
                gripper_token=GRIPPER_EXECUTION_TOKEN,
                confirm_cell_clear=False,
                namespace="/watson",
            )
        validate_execution_authorization(
            mode="execute",
            arm_token=EXECUTION_ARM_TOKEN,
            gripper_token=GRIPPER_EXECUTION_TOKEN,
            confirm_cell_clear=True,
            namespace="/watson",
        )

    def test_safe_modes_reject_arming_arguments(self) -> None:
        for mode in ("check", "dry-run"):
            with self.assertRaisesRegex(ValueError, "only with --mode execute"):
                validate_execution_authorization(
                    mode=mode,
                    arm_token=EXECUTION_ARM_TOKEN,
                    gripper_token="",
                    confirm_cell_clear=False,
                    namespace="/watson",
                )

    def test_execute_requires_independent_gripper_token(self) -> None:
        with self.assertRaisesRegex(ValueError, "gripper-token"):
            validate_execution_authorization(
                mode="execute",
                arm_token=EXECUTION_ARM_TOKEN,
                gripper_token="wrong",
                confirm_cell_clear=True,
                namespace="/watson",
            )

    def test_execute_speed_is_exactly_50(self) -> None:
        self.assertEqual(exact_execute_project_speed_failures(snapshot()), [])
        slow = snapshot()
        slow = HealthSnapshot(
            **{**slow.__dict__, "project_speed": 49}
        )
        self.assertEqual(
            exact_execute_project_speed_failures(slow),
            [
                "set TMflow project speed to 50 before execute (observed 49)"
            ],
        )

    def test_exact_named_vendor_tool_only(self) -> None:
        settings = {
            "active_tcp_name": "QC_2FG7_VENDOR",
            "tcp_value": [0.0, 0.0, 138.6, 0.0, 0.0, 0.0],
            "mass_kg": 1.2,
            "principal_moi": [0.0, 0.0, 0.0],
            "mass_centre_frame": [0.0, 0.0, 62.52, 0.0, 0.0, 0.0],
        }
        audit = {
            "settings": settings,
            "promotion_passed": True,
            "known_vendor_profile_matched": True,
            "write_items_called": [],
            "motion_commanded": False,
        }
        self.assertEqual(exact_tool_audit_failures(audit), [])
        renamed = copy.deepcopy(audit)
        renamed["settings"]["active_tcp_name"] = "RobotEndFlange"
        self.assertTrue(exact_tool_audit_failures(renamed))

    def test_live_monitor_checks_each_axis(self) -> None:
        stage = small_stage()
        self.assertEqual(live_stage_failures(snapshot(), stage), [])
        for joint in range(6):
            positions = list(snapshot().feedback_joint_positions)
            positions[joint] = (
                stage.position_maximum_rad[joint]
                + LIVE_POSITION_ENVELOPE_MARGIN_RAD
                + 0.001
            )
            failures = live_stage_failures(snapshot(positions=positions), stage)
            self.assertTrue(
                any(f"joint_{joint + 1}" in failure for failure in failures)
            )

    def test_live_first_wire_cubic_uses_actual_feedback_q_and_velocity(self) -> None:
        stage = small_stage()
        safe = snapshot(positions=stage.start_positions)
        proof = validate_stage_live_first_wire_cubic(safe, stage)
        self.assertEqual(
            proof["status"],
            "validated_live_first_wire_cubic",
        )
        unsafe_positions = list(stage.start_positions)
        unsafe_positions[0] -= 0.001
        with self.assertRaisesRegex(ValueError, "joint_1 cubic acceleration"):
            validate_stage_live_first_wire_cubic(
                snapshot(positions=unsafe_positions),
                stage,
            )

    def test_gripper_hooks_are_static_only(self) -> None:
        close = gripper_after_stage_hook(small_stage("descend_tilted_grasp"))
        opened = gripper_after_stage_hook(small_stage("descend_vertical"))
        self.assertEqual(close["action"], "close")
        self.assertEqual(opened["action"], "open")
        self.assertFalse(close["executed"])
        self.assertEqual(close["actuator_calls"], [])
        self.assertEqual(GRIPPER_POLICY["action_clients_created"], 0)
        self.assertEqual(
            GRIPPER_POLICY["close_contact_max_external_width_mm"],
            2.0,
        )
        self.assertTrue(
            GRIPPER_POLICY["open_completion_requires_grip_cleared"]
        )
        self.assertEqual(
            gripper_after_stage_hook(small_stage("return_ready")),
            None,
        )


class LauncherStaticTests(unittest.TestCase):
    def test_runner_has_no_direct_controller_or_gripper_action_client(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn(
            "from control_msgs.action import FollowJointTrajectory",
            source,
        )
        self.assertNotIn("ScriptExit", source)
        self.assertIn('TOOL_SELECT_REQUEST_ID = "ToolSelect1"', source)
        self.assertIn(
            "TOOL_SELECT_SCRIPT = 'ChangeTCP(\"QC_2FG7_VENDOR\")'",
            source,
        )
        self.assertIn('if args.mode == "execute":', source)
        self.assertIn('goal.controller_names = ["tmr_arm_controller"]', source)
        self.assertIn('EXECUTE_ACTION = "/watson/execute_trajectory"', source)

    def test_shell_owns_default_bringup_lifecycle(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("watson_bringup.launch.py", source)
        self.assertIn("allow_trajectory_execution:=true", source)
        self.assertIn("namespace:=watson", source)
        self.assertIn("--use-existing-stack", source)
        self.assertIn("--offline-validate", source)
        self.assertIn('ROBOT_INTERFACE="enp1s0"', source)
        self.assertIn('ROBOT_SOURCE_IP="192.0.2.100"', source)
        self.assertIn('COMPUTE_BOX_IP="192.0.2.1"', source)
        self.assertIn("stop_owned_stack", source)
        self.assertIn("setsid ros2 launch", source)
        self.assertLess(
            source.index('if [ "$offline_only" -eq 1 ]'),
            source.index('carrier_file="/sys/class/net/'),
        )
        self.assertLess(
            source.index('if [ "$help_only" -eq 1 ]'),
            source.index("source_setup /opt/ros/jazzy/setup.bash"),
        )
        self.assertIn(
            "execute mode must use the wrapper-owned, provenance-gated "
            "Watson stack",
            source,
        )


class RunnerInjectedBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_runner_module()

    def test_process_signal_gate_blocks_worker_delivery_and_latches_stop(self) -> None:
        code = f"""
import importlib.util
import os
import signal
import threading

runner_path = {str(RUNNER)!r}
spec = importlib.util.spec_from_file_location("signal_gate_probe", runner_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
gate = module.ProcessSignalGate()
worker = threading.Thread(
    target=lambda: os.kill(os.getpid(), signal.SIGTERM),
)
worker.start()
worker.join()
observed = gate.poll()
if observed != signal.SIGTERM:
    raise SystemExit(f"wrong latched signal: {{observed!r}}")
print("signal-gate-pass")
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ARENA_DIR,
            text=True,
            capture_output=True,
            timeout=5.0,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )
        self.assertIn("signal-gate-pass", completed.stdout)

    def test_process_signal_gate_defines_dispatch_linearization(self) -> None:
        class FakeSignalApi:
            SIGINT = 2
            SIGTERM = 15
            SIGHUP = 1
            SIG_BLOCK = 0
            SIG_DFL = object()

            def __init__(self) -> None:
                self.pending = []
                self.mask_calls = []
                self.handlers = {}

            def pthread_sigmask(self, how, signals):
                self.mask_calls.append((how, frozenset(signals)))
                return frozenset()

            def sigtimedwait(self, _signals, _timeout):
                if not self.pending:
                    return None
                return SimpleNamespace(si_signo=self.pending.pop(0))

            def getsignal(self, signum):
                return self.handlers.get(signum)

            def signal(self, signum, handler):
                self.handlers[signum] = handler

        before = FakeSignalApi()
        before_gate = self.runner.ProcessSignalGate(before)
        before.pending.append(before.SIGINT)
        with self.assertRaisesRegex(InterruptedError, "before arm dispatch"):
            with before_gate.command_commit("arm"):
                self.fail("pre-commit signal must prevent dispatch")

        after = FakeSignalApi()
        after_gate = self.runner.ProcessSignalGate(after)
        committed = False
        with after_gate.command_commit("arm"):
            committed = True
            after.pending.append(after.SIGTERM)
        self.assertTrue(committed)
        self.assertEqual(after_gate.poll(), after.SIGTERM)
        self.assertEqual(
            after.mask_calls,
            [(after.SIG_BLOCK, frozenset((1, 2, 15)))],
        )
        self.assertEqual(
            after.handlers,
            {1: after.SIG_DFL, 2: after.SIG_DFL, 15: after.SIG_DFL},
        )

    def test_hil_final_event_is_unique_and_last(self) -> None:
        self.runner.reset_hil_event_stream()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.runner.emit_hil_event(
                True,
                "run_started",
                mode="execute",
            )
            self.runner.emit_hil_event(
                True,
                "run_failed",
                status="failed_closed",
                error="deliberate test failure",
                physical_estop_required=False,
            )
            with self.assertRaisesRegex(RuntimeError, "after the final"):
                self.runner.emit_hil_event(
                    True,
                    "stage_started",
                    sequence_index=1,
                    stage_name="not_allowed_after_final",
                    specimen_id=1,
                )
            with self.assertRaisesRegex(RuntimeError, "after the final"):
                self.runner.emit_hil_event(
                    True,
                    "run_completed",
                    mode="execute",
                    status="must_not_duplicate_final",
                    motion_commanded=False,
                )
        payloads = [
            self.runner.json.loads(
                line[len(self.runner.HIL_EVENT_PREFIX) :]
            )
            for line in output.getvalue().splitlines()
        ]
        self.assertEqual(
            [payload["event"] for payload in payloads],
            ["run_started", "run_failed"],
        )
        self.assertEqual(
            [payload["event_sequence"] for payload in payloads],
            [1, 2],
        )

    def test_completed_arm_stage_is_not_reclassified_by_gripper_failure(
        self,
    ) -> None:
        stage = small_stage()
        self.runner.reset_hil_event_stream()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.runner.emit_hil_stage_failure(
                True,
                stage,
                RuntimeError("gripper hook failed"),
                arm_stage_completed=True,
            )
        self.assertEqual(output.getvalue(), "")

        with contextlib.redirect_stdout(output):
            self.runner.emit_hil_stage_failure(
                True,
                stage,
                RuntimeError("arm stage failed"),
                arm_stage_completed=False,
            )
        payload = self.runner.json.loads(
            output.getvalue().split(
                self.runner.HIL_EVENT_PREFIX,
                1,
            )[1]
        )
        self.assertEqual(payload["event"], "stage_failed")
        self.assertEqual(payload["sequence_index"], stage.sequence_index)
        self.assertIn("arm stage failed", payload["error"])

    def test_live_main_blocks_stop_signals_before_validation(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        main_source = source[source.index("def main()") :]
        self.assertLess(
            main_source.index("signal_gate = ProcessSignalGate()"),
            main_source.index("validate_cli(args)"),
        )
        self.assertLess(
            main_source.index("signal_gate = ProcessSignalGate()"),
            main_source.index("load_execution_bundle("),
        )
        wrapper = WRAPPER.read_text(encoding="utf-8")
        self.assertIn(
            "/usr/bin/env --default-signal=INT,TERM,HUP",
            wrapper,
        )

    def test_execution_enabled_uses_jazzy_parameter_services_api(self) -> None:
        calls = []

        class Future:
            def done(self):
                return True

            def result(self):
                return SimpleNamespace(
                    values=[SimpleNamespace(bool_value=True)]
                )

        class ParameterClient:
            def wait_for_services(self, **kwargs):
                calls.append(("wait_for_services", kwargs))
                return True

            def get_parameters(self, names):
                calls.append(("get_parameters", names))
                return Future()

        node = object.__new__(self.runner.AirReplayNode)
        node.args = SimpleNamespace(service_timeout=0.25)
        node.move_group_parameters = ParameterClient()
        node.node = object()
        node.rclpy = SimpleNamespace(
            spin_until_future_complete=lambda *_args, **_kwargs: None,
        )
        node.require_execution_enabled()
        self.assertEqual(
            calls,
            [
                ("wait_for_services", {"timeout_sec": 0.25}),
                ("get_parameters", ["allow_trajectory_execution"]),
            ],
        )

    def test_parser_rejects_abbreviated_mode(self) -> None:
        parser = self.runner.build_parser()
        self.assertFalse(parser.allow_abbrev)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--mo", "execute"])

    @unittest.skipUnless(
        retime_module.TECHMAN_WORKSPACE.exists()
        and DEFAULT_RETIMED_ARTIFACT.exists()
        and DEFAULT_INGRESS_ARTIFACT.exists(),
        "reviewed tm_driver workspace or private execution artifacts are absent",
    )
    def test_reviewed_ready_resume_selects_only_exact_stage_one_boundary(
        self,
    ) -> None:
        bundle = load_execution_bundle()
        selected = self.runner.select_execution_stages(
            bundle,
            resume_at_reviewed_ready=True,
        )
        self.assertEqual(len(selected), 49)
        self.assertEqual(selected[0].sequence_index, 1)
        self.assertEqual(selected[0].stage_name, "approach_tilted_pregrasp")
        self.assertEqual(
            selected[0].start_positions,
            bundle.stages[0].goal_positions,
        )
        self.assertEqual(
            self.runner.select_execution_stages(
                bundle,
                resume_at_reviewed_ready=False,
            ),
            bundle.stages,
        )

        changed_first = replace(
            bundle.stages[1],
            stage_name="tampered_resume_stage",
        )
        tampered = replace(
            bundle,
            stages=(bundle.stages[0], changed_first) + bundle.stages[2:],
        )
        with self.assertRaisesRegex(ValueError, "boundary changed"):
            self.runner.select_execution_stages(
                tampered,
                resume_at_reviewed_ready=True,
            )

    def test_reviewed_ready_resume_is_execute_only(self) -> None:
        parser = self.runner.build_parser()
        args = parser.parse_args(
            ["--mode", "check", "--resume-at-reviewed-ready"]
        )
        with self.assertRaisesRegex(ValueError, "requires --mode execute"):
            self.runner.validate_cli(args)

    def test_position_source_pair_allows_one_driver_cycle_of_motion(self) -> None:
        tracker = self.runner.PositionSourcePairTracker()
        zero = (0.0,) * 6
        tracker.record("feedback", 1_000_000_000, zero, 10.000)
        tracker.record("joint_state", 1_000_000_000, zero, 10.001)
        # The next FeedbackState can arrive before its matching JointState.
        # Latest-vs-latest differs by >0.005 rad, but the most recent exact
        # timestamp pair remains fresh and internally identical.
        tracker.record(
            "feedback",
            1_015_000_000,
            (0.007,) + zero[1:],
            10.015,
        )
        self.assertEqual(tracker.failures(now=10.016), [])

    def test_position_source_pair_rejects_same_stamp_corruption(self) -> None:
        tracker = self.runner.PositionSourcePairTracker()
        zero = (0.0,) * 6
        tracker.record("feedback", 2_000_000_000, zero, 20.000)
        tracker.record(
            "joint_state",
            2_000_000_000,
            (0.006,) + zero[1:],
            20.001,
        )
        self.assertTrue(
            any(
                "exact-timestamp Watson position sources disagree" in failure
                for failure in tracker.failures(now=20.002)
            )
        )

    def test_position_source_pair_rejects_stale_pair(self) -> None:
        tracker = self.runner.PositionSourcePairTracker()
        zero = (0.0,) * 6
        tracker.record("feedback", 3_000_000_000, zero, 30.000)
        tracker.record("joint_state", 3_000_000_000, zero, 30.001)
        self.assertTrue(
            any(
                "pair is stale" in failure
                for failure in tracker.failures(now=30.102)
            )
        )

    def test_position_source_pair_rejects_excessive_header_skew(self) -> None:
        tracker = self.runner.PositionSourcePairTracker()
        zero = (0.0,) * 6
        tracker.record("feedback", 4_000_000_000, zero, 40.000)
        tracker.record("joint_state", 4_000_000_000, zero, 40.001)
        tracker.record("feedback", 4_031_000_000, zero, 40.031)
        self.assertTrue(
            any(
                "header skew" in failure
                for failure in tracker.failures(now=40.032)
            )
        )

    def test_graph_rejects_interactive_planning_action_clients(self) -> None:
        runner = self.runner
        namespace = "/watson"

        class FakeGraphNode:
            def get_node_names_and_namespaces(self):
                return [
                    ("move_group", namespace),
                    ("tm_driver_node", namespace),
                    ("moveit_simple_controller_manager", namespace),
                    ("watson_multi_pin_air_replay", "/"),
                    ("interactive_rviz", namespace),
                ]

        def action_servers(_node, node_name, _node_namespace):
            if node_name == "move_group":
                return [
                    (
                        runner.EXECUTE_ACTION,
                        ["moveit_msgs/action/ExecuteTrajectory"],
                    )
                ]
            if node_name == "tm_driver_node":
                return [
                    (
                        runner.CONTROLLER_ACTION,
                        ["control_msgs/action/FollowJointTrajectory"],
                    )
                ]
            return []

        def exact_action_clients(
            _node,
            node_name,
            _node_namespace,
            *,
            interactive_endpoint,
        ):
            if node_name == "watson_multi_pin_air_replay":
                return [
                    (
                        runner.EXECUTE_ACTION,
                        ["moveit_msgs/action/ExecuteTrajectory"],
                    )
                ]
            if node_name == "moveit_simple_controller_manager":
                return [
                    (
                        runner.CONTROLLER_ACTION,
                        ["control_msgs/action/FollowJointTrajectory"],
                    )
                ]
            if node_name == "interactive_rviz":
                return [
                    (
                        interactive_endpoint,
                        ["moveit_msgs/action/MoveGroup"],
                    )
                ]
            return []

        for interactive_endpoint in (
            f"{namespace}/move_action",
            f"{namespace}/sequence_move_group",
        ):
            with self.subTest(endpoint=interactive_endpoint):
                node = object.__new__(runner.AirReplayNode)
                node.namespace = namespace
                node.node = FakeGraphNode()
                node.ros = {
                    "get_action_server_names_and_types_by_node": action_servers,
                    "get_action_client_names_and_types_by_node": (
                        lambda ros_node, node_name, node_namespace: (
                            exact_action_clients(
                                ros_node,
                                node_name,
                                node_namespace,
                                interactive_endpoint=interactive_endpoint,
                            )
                        )
                    ),
                }
                failures = node.graph_failures()
                self.assertTrue(
                    any(
                        "interactive_rviz" in failure
                        and interactive_endpoint in failure
                        for failure in failures
                    ),
                    failures,
                )

    def test_guarded_gripper_transition_brackets_command_with_arm_health(self) -> None:
        calls = []

        class Runtime:
            stop_requested = False
            stop_signal = None

            def require_healthy(self, **kwargs):
                calls.append(("health", kwargs))
                return snapshot()

            def refresh_after_blocking_gripper_call(self):
                calls.append(("refresh", {}))
                return snapshot()

        transport = object()

        def command_runner(action, **kwargs):
            calls.append(("command", action, kwargs))
            return {
                "status": "completed",
                "completed": True,
                "failures": [],
            }

        report = self.runner.guarded_gripper_transition(
            Runtime(),
            transport,
            self.runner.GripperAction.CLOSE,
            confirmation=GRIPPER_EXECUTION_TOKEN,
            command_runner=command_runner,
        )
        self.assertTrue(report["completed"])
        self.assertEqual([entry[0] for entry in calls], [
            "health",
            "command",
            "refresh",
            "health",
        ])
        self.assertIs(calls[1][2]["transport"], transport)
        self.assertFalse(calls[1][2]["abort_requested"]())
        self.assertTrue(calls[0][1]["stationary"])
        self.assertTrue(calls[0][1]["exact_project_speed"])

    def test_blocked_gripper_prevents_post_health_and_is_reported(self) -> None:
        health_calls = []

        class Runtime:
            stop_requested = False
            stop_signal = None

            def require_healthy(self, **kwargs):
                health_calls.append(kwargs)
                return snapshot()

            def refresh_after_blocking_gripper_call(self):
                raise AssertionError("blocked command must not refresh")

        def command_runner(_action, **_kwargs):
            return {
                "status": "blocked",
                "completed": False,
                "failures": ["fake block"],
                "safety_evidence": {
                    "gripper_command_may_have_been_sent": False,
                },
            }

        report = self.runner.guarded_gripper_transition(
            Runtime(),
            object(),
            self.runner.GripperAction.OPEN,
            confirmation=GRIPPER_EXECUTION_TOKEN,
            command_runner=command_runner,
        )
        self.assertFalse(report["completed"])
        self.assertEqual(len(health_calls), 1)
        self.assertIn("fake block", report["failures"])

    def test_post_gripper_arm_health_failure_marks_stop_unverified(self) -> None:
        health_calls = 0

        class Runtime:
            stop_requested = False
            stop_signal = None

            def require_healthy(self, **_kwargs):
                nonlocal health_calls
                health_calls += 1
                if health_calls == 2:
                    raise RuntimeError("fresh stationary arm proof vanished")
                return snapshot()

            def refresh_after_blocking_gripper_call(self):
                return snapshot()

        def command_runner(_action, **_kwargs):
            return {
                "status": "completed",
                "completed": True,
                "failures": [],
                "safety_evidence": {
                    "gripper_command_may_have_been_sent": True,
                },
            }

        report = self.runner.guarded_gripper_transition(
            Runtime(),
            object(),
            self.runner.GripperAction.OPEN,
            confirmation=GRIPPER_EXECUTION_TOKEN,
            command_runner=command_runner,
        )
        self.assertFalse(report["completed"])
        self.assertTrue(report["arm_stop_unverified"])
        self.assertIn("stationary arm proof vanished", report["failures"][0])
        with self.assertRaises(self.runner.StopUnverifiedError):
            self.runner.raise_gripper_transition_failure(
                "post-gripper gate",
                report,
            )

    def test_post_gripper_refresh_drains_stale_ros_callbacks(self) -> None:
        node = object.__new__(self.runner.AirReplayNode)
        node._stop_requested = False
        node.signal_gate = None
        node.stop_signal = None
        samples = [
            replace(
                snapshot(),
                feedback_age_s=1.4,
                joint_state_age_s=1.4,
            ),
            replace(
                snapshot(),
                feedback_age_s=0.01,
                joint_state_age_s=0.01,
            ),
        ]
        spins = 0

        def spin_once(_node, **_kwargs):
            nonlocal spins
            spins += 1

        node.node = object()
        node.rclpy = SimpleNamespace(spin_once=spin_once)
        node.snapshot = lambda: samples[min(spins - 1, 1)]
        node.publisher_failures = lambda: []
        refreshed = node.refresh_after_blocking_gripper_call()
        self.assertEqual(refreshed.feedback_age_s, 0.01)
        self.assertEqual(spins, 2)

    def test_tool_select_uses_only_exact_request_and_fresh_readback(self) -> None:
        runner = self.runner
        requests = []

        class Future:
            def done(self):
                return True

            def result(self):
                return SimpleNamespace(ok=True)

        class Client:
            def wait_for_service(self, **_kwargs):
                return True

            def call_async(self, request):
                requests.append(request)
                return Future()

        class Request:
            def __init__(self):
                self.id = ""
                self.script = ""

        node = object.__new__(runner.AirReplayNode)
        node.args = SimpleNamespace(mode="execute", service_timeout=0.1)
        node.tool_select_client = Client()
        node.ros = {"SendScript": SimpleNamespace(Request=Request)}
        node.node = object()
        node.rclpy = SimpleNamespace(
            spin_until_future_complete=lambda *_args, **_kwargs: None,
            spin_once=lambda *_args, **_kwargs: None,
        )
        node.stop_requested = False
        node.stop_signal = None
        node.tool_selection_evidence = {
            "attempted": False,
            "request_id": "ToolSelect1",
            "script_sha256": self.runner.hashlib.sha256(
                b'ChangeTCP("QC_2FG7_VENDOR")'
            ).hexdigest(),
            "response_received": False,
            "response_ok": None,
            "fresh_exact_tool_readback": None,
            "readback_failures": [],
        }
        node.require_healthy = lambda **_kwargs: snapshot()
        exact_audit = {
            "settings": {
                "active_tcp_name": "QC_2FG7_VENDOR",
                "tcp_value": [0.0, 0.0, 138.6, 0.0, 0.0, 0.0],
                "mass_kg": 1.2,
                "principal_moi": [0.0, 0.0, 0.0],
                "mass_centre_frame": [
                    0.0, 0.0, 62.52, 0.0, 0.0, 0.0
                ],
            },
            "promotion_passed": True,
            "known_vendor_profile_matched": True,
            "write_items_called": [],
            "motion_commanded": False,
        }
        node.read_tool_audit = lambda **_kwargs: exact_audit
        result = node.select_exact_tool()
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].id, "ToolSelect1")
        self.assertEqual(
            requests[0].script,
            'ChangeTCP("QC_2FG7_VENDOR")',
        )
        self.assertTrue(result["response_ok"])

    def test_tool_select_is_unavailable_in_read_only_modes(self) -> None:
        node = object.__new__(self.runner.AirReplayNode)
        node.args = SimpleNamespace(mode="dry-run")
        node.tool_select_client = None
        node.tool_selection_evidence = {}
        with self.assertRaisesRegex(RuntimeError, "only in execute mode"):
            node.select_exact_tool()

    def test_failed_tool_select_returns_before_any_action_goal(self) -> None:
        runner = self.runner

        class Future:
            def done(self):
                return True

            def result(self):
                return SimpleNamespace(ok=False)

        class Client:
            def wait_for_service(self, **_kwargs):
                return True

            def call_async(self, _request):
                return Future()

        class Request:
            id = ""
            script = ""

        node = object.__new__(runner.AirReplayNode)
        node.args = SimpleNamespace(mode="execute", service_timeout=0.1)
        node.tool_select_client = Client()
        node.ros = {"SendScript": SimpleNamespace(Request=Request)}
        node.node = object()
        node.rclpy = SimpleNamespace(
            spin_until_future_complete=lambda *_args, **_kwargs: None,
        )
        node.stop_requested = False
        node.stop_signal = None
        node.tool_selection_evidence = {
            "attempted": False,
            "request_id": "ToolSelect1",
            "script_sha256": "0" * 64,
            "response_received": False,
            "response_ok": None,
            "fresh_exact_tool_readback": None,
            "readback_failures": [],
        }
        with self.assertRaisesRegex(RuntimeError, "ChangeTCP"):
            node.select_exact_tool()
        self.assertTrue(node.tool_selection_evidence["attempted"])
        self.assertTrue(node.tool_selection_evidence["response_received"])
        self.assertFalse(node.tool_selection_evidence["response_ok"])

    def _action_status_test_node(
        self,
        *,
        retained_messages=None,
        motion_command_sent=False,
        wrong_qos=False,
        wrong_owner=False,
        wrong_type=False,
        wrong_depth=False,
        wrong_durability=False,
        zero_gid=False,
        unknown_history_depth=False,
        unknown_history_nonzero_depth=False,
        known_history_zero_depth=False,
        system_default_history_depth=False,
        keep_all_history_depth=False,
        mixed_history_depth=False,
        rmw_implementation_identifier="rmw_fastrtps_cpp",
        change_gid_after_spin=False,
        fail_second_subscription=False,
    ):
        runner = self.runner
        retained_messages = retained_messages or {}
        expected_qos = SimpleNamespace(
            history=1,
            depth=1,
            reliability=2,
            durability=3,
        )

        class FakeStatusNode:
            def __init__(self):
                self.subscriptions = []
                self.destroyed = []
                self.gid_version = 0

            def get_publishers_info_by_topic(self, topic):
                is_execute = topic.startswith(runner.EXECUTE_ACTION)
                node_name = "move_group" if is_execute else "tm_driver_node"
                if wrong_owner and is_execute:
                    node_name = "unexpected_move_group"
                history = 1
                depth = 2 if wrong_depth and is_execute else 1
                if unknown_history_depth or (
                    mixed_history_depth and is_execute
                ):
                    history = 3
                    depth = 0
                if unknown_history_nonzero_depth:
                    history = 3
                    depth = 1
                if known_history_zero_depth:
                    history = 1
                    depth = 0
                if system_default_history_depth:
                    history = 0
                    depth = 0
                if keep_all_history_depth:
                    history = 2
                    depth = 0
                qos = SimpleNamespace(
                    history=history,
                    depth=depth,
                    reliability=(
                        99 if wrong_qos and is_execute else 2
                    ),
                    durability=(
                        99 if wrong_durability and is_execute else 3
                    ),
                )
                gid_seed = 11 if is_execute else 22
                return [
                    SimpleNamespace(
                        node_name=node_name,
                        node_namespace="/watson",
                        topic_type=(
                            "wrong_msgs/msg/Status"
                            if wrong_type and is_execute
                            else runner.ACTION_STATUS_TYPE
                        ),
                        endpoint_gid=(
                            [0]
                            if zero_gid and is_execute
                            else [gid_seed + (100 * self.gid_version)]
                        ),
                        qos_profile=qos,
                    )
                ]

            def create_subscription(
                self,
                _message_type,
                topic,
                callback,
                _qos,
            ):
                if fail_second_subscription and len(self.subscriptions) == 1:
                    raise RuntimeError("injected second subscription failure")
                subscription = SimpleNamespace(
                    topic=topic,
                    callback=callback,
                    delivered=False,
                )
                self.subscriptions.append(subscription)
                return subscription

            def destroy_subscription(self, subscription):
                self.destroyed.append(subscription)
                return True

        fake_node = FakeStatusNode()

        def spin_once(_node, **_kwargs):
            if change_gid_after_spin:
                fake_node.gid_version = 1
            for subscription in fake_node.subscriptions:
                key = (
                    "execute"
                    if subscription.topic.startswith(runner.EXECUTE_ACTION)
                    else "controller"
                )
                message = retained_messages.get(key)
                if message is not None and not subscription.delivered:
                    subscription.delivered = True
                    subscription.callback(message)

        node = object.__new__(runner.AirReplayNode)
        node.namespace = "/watson"
        node.node = fake_node
        node.rclpy = SimpleNamespace(spin_once=spin_once)
        node.ros = {
            "GoalStatus": SimpleNamespace(
                STATUS_SUCCEEDED=4,
                STATUS_CANCELED=5,
                STATUS_ABORTED=6,
            ),
            "GoalStatusArray": object,
            "HistoryPolicy": SimpleNamespace(UNKNOWN=3),
            "qos_profile_action_status_default": expected_qos,
            "rmw_implementation_identifier": (
                rmw_implementation_identifier
            ),
        }
        node.execute_action_status = None
        node.controller_action_status = None
        node.execute_action_status_received_at = -1_000_000.0
        node.controller_action_status_received_at = -1_000_000.0
        node.execute_action_status_generation = 0
        node.controller_action_status_generation = 0
        node.motion_command_sent = motion_command_sent
        node.publisher_failures = lambda: []
        node.graph_failures = lambda: []
        return node, fake_node

    @staticmethod
    def _terminal_status(status=4):
        return SimpleNamespace(
            status_list=[SimpleNamespace(status=status)]
        )

    def _with_short_action_idle_timeout(self, callback):
        original_timeout = self.runner.ACTION_IDLE_TIMEOUT_S
        self.runner.ACTION_IDLE_TIMEOUT_S = 0.01
        try:
            return callback()
        finally:
            self.runner.ACTION_IDLE_TIMEOUT_S = original_timeout

    def test_action_idle_accepts_no_retained_sample_only_before_goal(self) -> None:
        node, fake = self._action_status_test_node()
        proof = self._with_short_action_idle_timeout(
            node.require_action_idle
        )
        self.assertTrue(proof["verified"])
        self.assertIn("before_first_goal", proof["basis"])
        self.assertEqual(
            sorted(proof["missing_status_samples"]),
            ["controller", "execute"],
        )
        self.assertEqual(len(fake.destroyed), 2)

        node, fake = self._action_status_test_node(
            motion_command_sent=True
        )
        with self.assertRaisesRegex(RuntimeError, "no retained status evidence"):
            self._with_short_action_idle_timeout(node.require_action_idle)
        self.assertEqual(len(fake.destroyed), 2)

        node, fake = self._action_status_test_node(
            retained_messages={"execute": self._terminal_status()},
        )
        with self.assertRaisesRegex(RuntimeError, "both absent"):
            self._with_short_action_idle_timeout(node.require_action_idle)
        self.assertEqual(len(fake.destroyed), 2)

    def test_action_idle_replays_terminal_status_without_wall_age(self) -> None:
        terminal = self._terminal_status()
        node, fake = self._action_status_test_node(
            retained_messages={
                "execute": terminal,
                "controller": terminal,
            },
            motion_command_sent=True,
        )
        node.execute_action_status_received_at = -1_000_000.0
        node.controller_action_status_received_at = -1_000_000.0
        proof = self._with_short_action_idle_timeout(
            node.require_action_idle
        )
        self.assertEqual(
            proof["basis"],
            "fresh_transient_local_terminal_status",
        )
        self.assertEqual(len(fake.destroyed), 2)

    def test_action_idle_accepts_dds_unknown_history_and_depth(self) -> None:
        node, fake = self._action_status_test_node(
            unknown_history_depth=True,
        )
        proof = self._with_short_action_idle_timeout(
            node.require_action_idle
        )
        self.assertTrue(proof["verified"])
        for publisher in proof["publisher_snapshot"].values():
            self.assertEqual(
                publisher["qos"]["history_depth_evidence"],
                "dds_discovery_unavailable",
            )
        self.assertEqual(len(fake.destroyed), 2)

    def test_action_idle_accepts_mixed_exact_and_unknown_history_depth(
        self,
    ) -> None:
        node, fake = self._action_status_test_node(
            mixed_history_depth=True,
        )
        proof = self._with_short_action_idle_timeout(
            node.require_action_idle
        )
        evidence = {
            key: publisher["qos"]["history_depth_evidence"]
            for key, publisher in proof["publisher_snapshot"].items()
        }
        self.assertEqual(evidence["execute"], "dds_discovery_unavailable")
        self.assertEqual(evidence["controller"], "reported_exact")
        self.assertEqual(len(fake.destroyed), 2)

    def test_action_idle_rejects_nonterminal_retained_status(self) -> None:
        node, fake = self._action_status_test_node(
            retained_messages={
                "execute": self._terminal_status(status=2),
                "controller": self._terminal_status(),
            },
        )
        with self.assertRaisesRegex(RuntimeError, "nonterminal status"):
            self._with_short_action_idle_timeout(node.require_action_idle)
        self.assertEqual(len(fake.destroyed), 2)

    def test_action_idle_rejects_wrong_qos_owner_and_gid_change(self) -> None:
        for keyword in (
            "wrong_qos",
            "wrong_owner",
            "wrong_type",
            "wrong_depth",
            "wrong_durability",
            "zero_gid",
            "unknown_history_nonzero_depth",
            "known_history_zero_depth",
            "system_default_history_depth",
            "keep_all_history_depth",
        ):
            with self.subTest(keyword=keyword):
                node, fake = self._action_status_test_node(
                    **{keyword: True}
                )
                with self.assertRaisesRegex(RuntimeError, "provenance"):
                    node.require_action_idle()
                self.assertEqual(fake.destroyed, [])

        for keyword in (
            "wrong_qos",
            "wrong_durability",
            "wrong_owner",
            "wrong_type",
            "zero_gid",
        ):
            with self.subTest(
                keyword=keyword,
                history_depth="UNKNOWN/0",
            ):
                node, fake = self._action_status_test_node(
                    unknown_history_depth=True,
                    **{keyword: True},
                )
                with self.assertRaisesRegex(RuntimeError, "provenance"):
                    node.require_action_idle()
                self.assertEqual(fake.destroyed, [])

        node, fake = self._action_status_test_node(
            unknown_history_depth=True,
            rmw_implementation_identifier="rmw_cyclonedds_cpp",
        )
        with self.assertRaisesRegex(RuntimeError, "rmw_fastrtps_cpp"):
            node.require_action_idle()
        self.assertEqual(fake.destroyed, [])

        node, fake = self._action_status_test_node(
            change_gid_after_spin=True,
        )
        with self.assertRaisesRegex(RuntimeError, "GID changed"):
            self._with_short_action_idle_timeout(node.require_action_idle)
        self.assertEqual(len(fake.destroyed), 2)

    def test_action_idle_destroys_partial_subscription_on_creation_error(
        self,
    ) -> None:
        node, fake = self._action_status_test_node(
            fail_second_subscription=True,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "injected second subscription failure",
        ):
            node.require_action_idle()
        self.assertEqual(len(fake.destroyed), 1)

    def test_goal_specific_idle_requires_both_generations_to_advance(
        self,
    ) -> None:
        terminal = self._terminal_status()
        node, _fake = self._action_status_test_node()
        node.execute_action_status = terminal
        node.controller_action_status = terminal
        node.execute_action_status_generation = 7
        node.controller_action_status_generation = 9
        checkpoint = node.capture_action_status_checkpoint()

        with self.assertRaisesRegex(RuntimeError, "did not advance"):
            self._with_short_action_idle_timeout(
                lambda: node.require_goal_specific_action_idle(checkpoint)
            )

        fired = False

        def advance_once(_node, **_kwargs):
            nonlocal fired
            if fired:
                return
            fired = True
            node._execute_status_callback(terminal)
            node._controller_status_callback(terminal)

        node.rclpy = SimpleNamespace(spin_once=advance_once)
        proof = self._with_short_action_idle_timeout(
            lambda: node.require_goal_specific_action_idle(checkpoint)
        )
        self.assertEqual(
            proof["basis"],
            "goal_specific_terminal_status_generations",
        )
        self.assertEqual(proof["execute_generation_before"], 7)
        self.assertEqual(proof["execute_generation_after"], 8)
        self.assertEqual(proof["controller_generation_before"], 9)
        self.assertEqual(proof["controller_generation_after"], 10)

        node, fake = self._action_status_test_node()
        node.execute_action_status = terminal
        node.controller_action_status = terminal
        checkpoint = node.capture_action_status_checkpoint()

        def change_writer_and_advance(_node, **_kwargs):
            fake.gid_version = 1
            node._execute_status_callback(terminal)
            node._controller_status_callback(terminal)

        node.rclpy = SimpleNamespace(spin_once=change_writer_and_advance)
        with self.assertRaisesRegex(RuntimeError, "GID changed"):
            self._with_short_action_idle_timeout(
                lambda: node.require_goal_specific_action_idle(checkpoint)
            )

    def test_success_and_cancel_use_pre_send_generation_checkpoint(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        execute_source = source[
            source.index("    def execute_stage("):
            source.index("    def cancel_execution(")
        ]
        self.assertLess(
            execute_source.index("capture_action_status_checkpoint()"),
            execute_source.index("send_goal_async(goal)"),
        )
        self.assertIn(
            "require_goal_specific_action_idle(",
            execute_source,
        )
        cancel_source = source[
            source.index("    def cancel_execution("):
            source.index("    def verify_stationary_after_motion(")
        ]
        self.assertIn(
            "require_goal_specific_action_idle(action_status_checkpoint)",
            cancel_source,
        )
        self.assertIn("provisional_cancel_failures", cancel_source)
        self.assertIn("if not terminal_proven:", cancel_source)
        self.assertIn(
            "if result_future.done():\n                    break",
            execute_source,
        )

    def test_gripper_stop_requires_fresh_nonbusy_poststate(self) -> None:
        completed = {
            "status": "completed",
            "completed": True,
            "state_after": {"state": {"busy": False}},
        }
        self.assertTrue(self.runner.gripper_stop_verified(completed))
        for changed in (
            {"status": "blocked"},
            {"completed": False},
            {"state_after": None},
            {"state_after": {"state": {"busy": True}}},
        ):
            candidate = {**completed, **changed}
            self.assertFalse(self.runner.gripper_stop_verified(candidate))

    def test_report_path_is_reserved_then_atomically_finalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            reservation = self.runner.reserve_private_report(path)
            reserved = self.runner.json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(reserved["status"], "reserved_before_live_contact")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            report = {"schema_version": 1, "status": "finished"}
            self.runner.write_private_report(
                path,
                report,
                reservation=reservation,
            )
            final = self.runner.json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(final["status"], "finished")
            self.assertEqual(
                final["report_payload_sha256"],
                self.runner.report_payload_sha256(final),
            )

    def test_existing_report_path_blocks_before_live_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "existing.json"
            path.write_text("do not overwrite", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                self.runner.reserve_private_report(path)
        source = RUNNER.read_text(encoding="utf-8")
        main_source = source[source.index("def main()") :]
        self.assertLess(
            main_source.index("reserve_private_report(report_path)"),
            main_source.index("validate_air_replay_network()"),
        )


if __name__ == "__main__":
    unittest.main()
