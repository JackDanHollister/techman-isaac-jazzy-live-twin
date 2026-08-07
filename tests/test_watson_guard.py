from __future__ import annotations

import io
import hashlib
import json
import math
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pin_axis_3d_sim.watson_guard import (
    FIRST_MOTION_PROFILE,
    HealthSnapshot,
    J6_QUALIFICATION_PROFILE,
    J6_SHOWCASE_PROFILE,
    JOINT_NAMES,
    TrajectorySample,
    get_j6_guard_profile,
    health_failures,
    j6_profile_targets,
    motion_envelope_failures,
    validate_trajectory_samples,
    wrist_check_targets,
)
from scripts.run_watson_guarded_demo import (
    ARM_TOKEN,
    ACTION_STATUS_SUCCEEDED,
    GUARD_SOURCE_SHA256,
    J6_QUALIFICATION_ARM_TOKEN,
    J6_SHOWCASE_ARM_TOKEN,
    J6_PLANNING_GOAL_TOLERANCE_RAD,
    MAX_PLANNED_GOAL_ERROR_RAD,
    MOVEIT_SUCCESS,
    REPORT_SCHEMA_VERSION,
    REPORT_DIGEST_FIELD,
    RUNNER_SOURCE_SHA256,
    ROBOT_INTERFACE,
    ROBOT_IP,
    ROBOT_MAC,
    ROBOT_SOURCE_IP,
    StopUnverifiedError,
    WatsonGuardNode,
    acquire_execute_lock,
    build_parser,
    main as guarded_main,
    report_payload_sha256,
    require_fresh_showcase_gate_before_send,
    validate_cli,
    validate_execute_network,
    validate_qualification_report,
    write_report,
    write_report_best_effort,
)


def healthy_snapshot(**overrides) -> HealthSnapshot:
    values = {
        "is_svr_connected": True,
        "is_sct_connected": True,
        "tmsrv_cperr": 0,
        "tmscript_cperr": 0,
        "tmsrv_dataerr": 0,
        "tmscript_dataerr": 0,
        "is_data_table_correct": True,
        "robot_link": True,
        "robot_error": False,
        "project_run": True,
        "project_pause": False,
        "safetyguard_a": False,
        "e_stop": False,
        "error_code": 0,
        "project_speed": 5,
        "ma_mode": 0,
        "robot_light": 21,
        "joint_positions": (0.0, 0.1, -0.2, 0.3, -0.4, 0.5),
        "feedback_joint_positions": (0.0, 0.1, -0.2, 0.3, -0.4, 0.5),
        "joint_velocities": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "feedback_age_s": 0.01,
        "joint_state_age_s": 0.01,
    }
    values.update(overrides)
    return HealthSnapshot(**values)


def qualification_trajectory_payload(
    start: tuple[float, ...],
    goal: tuple[float, ...],
    hard_reference: tuple[float, ...],
) -> tuple[dict, dict, dict]:
    profile = get_j6_guard_profile(J6_QUALIFICATION_PROFILE)
    zeros = (0.0,) * len(JOINT_NAMES)
    samples = (
        TrajectorySample(start, zeros, zeros, 0.0),
        TrajectorySample(goal, zeros, zeros, 2.0),
    )
    payload = {
        "joint_names": list(JOINT_NAMES),
        "multi_dof_joint_names": [],
        "multi_dof_point_count": 0,
        "points": [
            {
                "time_s": sample.time_s,
                "positions_rad": list(sample.positions),
                "velocities_rad_s": list(sample.velocities),
                "accelerations_rad_s2": list(sample.accelerations),
            }
            for sample in samples
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    metrics = validate_trajectory_samples(
        samples,
        expected_start=start,
        expected_goal=goal,
        hard_reference_start=hard_reference,
        max_goal_error_rad=MAX_PLANNED_GOAL_ERROR_RAD,
        max_excursion_rad=profile.hard_excursion_rad,
        max_sample_step_rad=profile.max_sample_step_rad,
        max_velocity_rad_s=profile.max_planned_velocity_rad_s,
        max_acceleration_rad_s2=profile.max_planned_acceleration_rad_s2,
        min_total_duration_s=profile.min_duration_s,
        max_total_duration_s=profile.max_duration_s,
        guard_profile=profile.name,
    )
    execution_metrics = validate_trajectory_samples(
        samples,
        expected_start=start,
        expected_goal=goal,
        hard_reference_start=hard_reference,
        hard_travel_start=start,
        execution_start_positions=start,
        execution_start_velocities=zeros,
        max_goal_error_rad=MAX_PLANNED_GOAL_ERROR_RAD,
        max_excursion_rad=profile.hard_excursion_rad,
        max_sample_step_rad=profile.max_sample_step_rad,
        max_velocity_rad_s=profile.max_planned_velocity_rad_s,
        max_acceleration_rad_s2=profile.max_planned_acceleration_rad_s2,
        min_total_duration_s=profile.min_duration_s,
        max_total_duration_s=profile.max_duration_s,
        guard_profile=profile.name,
    )
    return payload, metrics, execution_metrics


def qualification_report_payload(
    *,
    timestamp: datetime | None = None,
    **overrides,
) -> dict:
    profile = get_j6_guard_profile(J6_QUALIFICATION_PROFILE)
    reference = (0.4, -0.2, 2.0, -0.3, 1.5, 0.25)
    stages = j6_profile_targets(
        reference,
        guard_profile=J6_QUALIFICATION_PROFILE,
    )
    plans = []
    executions = []
    stage_start = reference
    for stage_name, stage_goal in stages:
        trajectory, metrics, execution_metrics = qualification_trajectory_payload(
            stage_start,
            stage_goal,
            reference,
        )
        plans.append(
            {
                "stage": stage_name,
                "hard_reference_start_rad": list(reference),
                "start_positions_rad": list(stage_start),
                "goal_positions_rad": list(stage_goal),
                "metrics": metrics,
                "trajectory": trajectory,
            }
        )
        executions.append(
            {
                "stage": stage_name,
                "action_status": ACTION_STATUS_SUCCEEDED,
                "moveit_error_code": MOVEIT_SUCCESS,
                "live_start_error_rad": 0.0,
                "live_goal_error_rad": 0.0,
                "physical_start_positions_rad": list(stage_start),
                "physical_start_velocities_rad_s": [0.0] * len(JOINT_NAMES),
                "physical_start_feedback_age_s": 0.01,
                "physical_start_joint_state_age_s": 0.01,
                "stationary_to_physical_start_drift_rad": 0.0,
                "post_motion_stationary_verified": True,
                "final_joint_positions_rad": list(stage_goal),
                "final_feedback_positions_rad": list(stage_goal),
                "final_joint_velocities_rad_s": [0.0] * len(JOINT_NAMES),
                "final_feedback_age_s": 0.01,
                "final_joint_state_age_s": 0.01,
                "execution_revalidation_metrics": execution_metrics,
            }
        )
        stage_start = stage_goal
    stable_health = healthy_snapshot(
        joint_positions=reference,
        feedback_joint_positions=reference,
        joint_velocities=(0.0,) * len(JOINT_NAMES),
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "timestamp_utc": (timestamp or datetime.now(timezone.utc)).isoformat(),
        "mode": "execute",
        "namespace": "/watson",
        "robot_ip": ROBOT_IP,
        "robot_interface": ROBOT_INTERFACE,
        "robot_source_ip": ROBOT_SOURCE_IP,
        "robot_mac": ROBOT_MAC,
        "runner_source_sha256": RUNNER_SOURCE_SHA256,
        "guard_source_sha256": GUARD_SOURCE_SHA256,
        "motion_profile": J6_QUALIFICATION_PROFILE,
        "motion_pattern": "j6_toward_zero_then_return",
        "amplitude_deg": profile.requested_amplitude_deg,
        "hard_max_excursion_deg": profile.hard_excursion_deg,
        "hard_max_excursion_rad": profile.hard_excursion_rad,
        "hard_max_planned_velocity_rad_s": profile.max_planned_velocity_rad_s,
        "hard_max_planned_acceleration_rad_s2": (
            profile.max_planned_acceleration_rad_s2
        ),
        "hard_max_live_velocity_rad_s": profile.max_live_velocity_rad_s,
        "hard_max_sample_step_rad": profile.max_sample_step_rad,
        "hard_duration_range_s": [profile.min_duration_s, profile.max_duration_s],
        "velocity_scaling": profile.velocity_scaling,
        "acceleration_scaling": profile.acceleration_scaling,
        "max_project_speed": profile.max_project_speed,
        "ros_domain_id": "219",
        "ros_automatic_discovery_range": "LOCALHOST",
        "publishes_joint_states": False,
        "direct_controller_goals": False,
        "commands_gripper": False,
        "queries_controller_tool_settings_read_only": True,
        "controller_tool_settings_promotion_passed": True,
        "motion_commanded": True,
        "health_failures": [],
        "initial_health": asdict(stable_health),
        "stable_health": asdict(stable_health),
        "hard_reference_start_rad": list(reference),
        "plans": plans,
        "execution": executions,
        "status": "execution_passed_and_returned",
    }
    report.update(overrides)
    return report


def write_qualification_report(root: Path, report: dict | None = None) -> Path:
    report_dir = root / "outputs" / "watson_guarded_demo"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "20260717T090000Z_execute.json"
    payload = report or qualification_report_payload()
    payload[REPORT_DIGEST_FIELD] = report_payload_sha256(payload)
    path.write_text(
        json.dumps(payload) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


class WatsonGuardTests(unittest.TestCase):
    def test_healthy_stationary_auto_state_passes(self):
        self.assertEqual(health_failures(healthy_snapshot()), [])

    def test_safety_and_connection_faults_fail_closed(self):
        failures = health_failures(
            healthy_snapshot(
                is_sct_connected=False,
                e_stop=True,
                project_speed=10,
                joint_velocities=(0.0, 0.0, 0.0, 0.0, 0.0, 0.02),
            )
        )
        self.assertTrue(any("Listen Node" in failure for failure in failures))
        self.assertTrue(any("E-stop" in failure for failure in failures))
        self.assertTrue(any("project_speed" in failure for failure in failures))
        self.assertTrue(any("stationary" in failure for failure in failures))

    def test_robot_light_provides_auto_mode_evidence_when_ma_mode_is_absent(self):
        for robot_light in (20, 21):
            self.assertEqual(
                health_failures(healthy_snapshot(ma_mode=0, robot_light=robot_light)),
                [],
            )
        for robot_light in (0, 3, 4, 9, 10, 14, 18, 22, 23, 24):
            failures = health_failures(
                healthy_snapshot(ma_mode=0, robot_light=robot_light)
            )
            self.assertTrue(any("Auto standby/running" in failure for failure in failures))

    def test_joint_state_and_feedback_positions_must_agree(self):
        failures = health_failures(
            healthy_snapshot(
                feedback_joint_positions=(0.0, 0.1, -0.2, 0.3, -0.4, 0.52)
            )
        )
        self.assertTrue(any("positions disagree" in failure for failure in failures))

    def test_wrist_check_moves_toward_zero_and_returns_exactly(self):
        current = (0.0, 0.1, -0.2, 0.3, -0.4, 0.5)
        stages = wrist_check_targets(current, amplitude_deg=1.0)
        self.assertEqual([stage[0] for stage in stages], ["wrist_check", "return_to_start"])
        self.assertAlmostEqual(stages[0][1][-1], current[-1] - math.radians(1.0))
        self.assertEqual(stages[1][1], current)

    def test_first_motion_amplitude_is_hard_capped(self):
        with self.assertRaisesRegex(ValueError, "at most 1 degree"):
            wrist_check_targets((0.0,) * 6, amplitude_deg=1.01)

    def test_locked_profile_targets_move_only_j6_and_return_exactly(self):
        for profile_name, expected_degrees in (
            (J6_QUALIFICATION_PROFILE, 6.0),
            (J6_SHOWCASE_PROFILE, 12.0),
        ):
            for start_j6, direction in ((0.5, -1.0), (-0.5, 1.0)):
                current = (0.0, 0.1, -0.2, 0.3, -0.4, start_j6)
                stages = j6_profile_targets(
                    current,
                    guard_profile=profile_name,
                )
                self.assertEqual(stages[0][1][:-1], current[:-1])
                self.assertAlmostEqual(
                    stages[0][1][-1],
                    start_j6 + direction * math.radians(expected_degrees),
                )
                self.assertEqual(stages[1], ("return_to_start", current))

    def test_locked_profile_trajectories_pass_exact_cubic_proof(self):
        start = (0.0,) * 6
        for profile_name in (J6_QUALIFICATION_PROFILE, J6_SHOWCASE_PROFILE):
            profile = get_j6_guard_profile(profile_name)
            goal = (0.0,) * 5 + (profile.requested_amplitude_rad,)
            samples = (
                TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.0),
                TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 2.0),
            )
            metrics = validate_trajectory_samples(
                samples,
                expected_start=start,
                expected_goal=goal,
                max_excursion_rad=profile.hard_excursion_rad,
                max_sample_step_rad=profile.max_sample_step_rad,
                max_velocity_rad_s=profile.max_planned_velocity_rad_s,
                max_acceleration_rad_s2=profile.max_planned_acceleration_rad_s2,
                min_total_duration_s=profile.min_duration_s,
                max_total_duration_s=profile.max_duration_s,
                guard_profile=profile.name,
            )
            self.assertLessEqual(
                metrics["max_interpolated_velocity_rad_s"],
                profile.max_planned_velocity_rad_s,
            )

    def test_locked_profile_requires_exact_requested_displacement(self):
        profile = get_j6_guard_profile(J6_QUALIFICATION_PROFILE)
        start = (0.0,) * 6
        wrong_goal = (0.0,) * 5 + (math.radians(5.99),)
        samples = (
            TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.0),
            TrajectorySample(wrong_goal, (0.0,) * 6, (0.0,) * 6, 2.0),
        )
        with self.assertRaisesRegex(ValueError, "exactly 6.0 degrees"):
            validate_trajectory_samples(
                samples,
                expected_start=start,
                expected_goal=wrong_goal,
                max_excursion_rad=10.0,
                max_sample_step_rad=10.0,
                max_velocity_rad_s=10.0,
                max_acceleration_rad_s2=10.0,
                guard_profile=profile.name,
            )

    def test_locked_profile_live_caps_cannot_be_widened_by_caller(self):
        profile = get_j6_guard_profile(J6_QUALIFICATION_PROFILE)
        start = (0.0,) * 6
        goal = (0.0,) * 5 + (profile.requested_amplitude_rad,)
        outside = (0.0,) * 5 + (math.radians(7.0001),)
        failures = motion_envelope_failures(
            healthy_snapshot(
                joint_positions=outside,
                feedback_joint_positions=outside,
                joint_velocities=(0.0,) * 6,
            ),
            expected_start=start,
            expected_goal=goal,
            max_target_overshoot_rad=10.0,
            max_live_velocity_rad_s=10.0,
            guard_profile=profile.name,
        )
        self.assertTrue(any("guarded interval" in failure for failure in failures))

        moving = (0.0,) * 5 + (profile.max_live_velocity_rad_s + 1e-6,)
        failures = motion_envelope_failures(
            healthy_snapshot(
                joint_positions=goal,
                feedback_joint_positions=goal,
                joint_velocities=moving,
            ),
            expected_start=start,
            expected_goal=goal,
            max_live_velocity_rad_s=10.0,
            guard_profile=profile.name,
        )
        self.assertTrue(any("velocity" in failure for failure in failures))

    def test_small_time_parameterized_trajectory_passes(self):
        start = (0.0,) * 6
        goal = (0.0, 0.0, 0.0, 0.0, 0.0, math.radians(0.9))
        samples = [
            TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.0),
            TrajectorySample(
                (0.0, 0.0, 0.0, 0.0, 0.0, math.radians(0.45)),
                (0.0,) * 6,
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.04),
                1.0,
            ),
            TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 2.0),
        ]
        metrics = validate_trajectory_samples(
            samples,
            expected_start=start,
            expected_goal=goal,
        )
        self.assertEqual(metrics["sample_count"], 3)
        self.assertAlmostEqual(metrics["duration_s"], 2.0)

    def test_trajectory_with_wrong_start_or_large_excursion_is_rejected(self):
        start = (0.0,) * 6
        bad_start = (0.0, 0.0, 0.0, 0.0, 0.0, 0.01)
        goal = (0.0, 0.0, 0.0, 0.0, 0.0, math.radians(1.0))
        samples = [
            TrajectorySample(bad_start, (0.0,) * 6, (0.0,) * 6, 0.0),
            TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 1.0),
        ]
        with self.assertRaisesRegex(ValueError, "start mismatch"):
            validate_trajectory_samples(
                samples,
                expected_start=start,
                expected_goal=goal,
            )

    def test_executed_trajectory_cannot_exceed_one_degree(self):
        start = (0.0,) * 6
        goal = (0.0,) * 5 + (math.radians(1.0),)
        beyond_cap = (0.0,) * 5 + (math.radians(1.02),)
        with self.assertRaisesRegex(ValueError, "guarded J6|one-degree"):
            validate_trajectory_samples(
                [
                    TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.0),
                    TrajectorySample(beyond_cap, (0.0,) * 6, (0.0,) * 6, 2.0),
                ],
                expected_start=start,
                expected_goal=goal,
                max_excursion_rad=0.04,
            )

    def test_live_envelope_cannot_expand_past_one_degree(self):
        start = healthy_snapshot().feedback_joint_positions
        goal = start[:-1] + (start[-1] - math.radians(0.9),)
        outside = start[:-1] + (start[-1] - math.radians(1.01),)
        failures = motion_envelope_failures(
            healthy_snapshot(
                joint_positions=outside,
                feedback_joint_positions=outside,
            ),
            expected_start=start,
            expected_goal=goal,
            max_target_overshoot_rad=0.01,
        )
        self.assertTrue(any("left guarded interval" in failure for failure in failures))

    def test_live_non_target_limit_cannot_be_widened_by_caller(self):
        start = healthy_snapshot().feedback_joint_positions
        goal = start[:-1] + (start[-1] - math.radians(0.9),)
        outside = (start[0] + 0.5,) + start[1:]
        failures = motion_envelope_failures(
            healthy_snapshot(
                joint_positions=outside,
                feedback_joint_positions=outside,
            ),
            expected_start=start,
            expected_goal=goal,
            max_non_target_excursion_rad=1.0,
        )
        self.assertTrue(any("non-target" in failure for failure in failures))

    def test_non_finite_limit_cannot_bypass_hard_excursion_cap(self):
        start = (0.0,) * 6
        goal = (0.0,) * 5 + (math.radians(0.9),)
        samples = [
            TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.0),
            TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 2.0),
        ]
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            validate_trajectory_samples(
                samples,
                expected_start=start,
                expected_goal=goal,
                max_excursion_rad=float("nan"),
            )

    def test_return_stage_remains_bounded_to_original_captured_pose(self):
        captured = (0.0, 0.1, -0.2, 0.3, -0.4, 0.5)
        return_start = captured[:-1] + (captured[-1] - math.radians(0.9),)
        outside_global_cap = captured[:-1] + (
            captured[-1] - math.radians(1.01),
        )
        samples = [
            TrajectorySample(return_start, (0.0,) * 6, (0.0,) * 6, 0.0),
            TrajectorySample(outside_global_cap, (0.0,) * 6, (0.0,) * 6, 1.0),
            TrajectorySample(captured, (0.0,) * 6, (0.0,) * 6, 2.0),
        ]
        with self.assertRaisesRegex(ValueError, "guarded J6|one-degree"):
            validate_trajectory_samples(
                samples,
                expected_start=return_start,
                expected_goal=captured,
                hard_reference_start=captured,
            )

        failures = motion_envelope_failures(
            healthy_snapshot(
                joint_positions=outside_global_cap,
                feedback_joint_positions=outside_global_cap,
            ),
            expected_start=return_start,
            expected_goal=captured,
            hard_reference_start=captured,
        )
        self.assertTrue(any("left guarded interval" in failure for failure in failures))

        safe_return = [
            TrajectorySample(return_start, (0.0,) * 6, (0.0,) * 6, 0.0),
            TrajectorySample(captured, (0.0,) * 6, (0.0,) * 6, 2.0),
        ]
        metrics = validate_trajectory_samples(
            safe_return,
            expected_start=return_start,
            expected_goal=captured,
            hard_reference_start=captured,
        )
        self.assertLessEqual(
            metrics["max_excursion_rad"],
            math.radians(1.0),
        )

    def test_missing_derivatives_and_fast_cubic_are_rejected(self):
        start = (0.0,) * 6
        goal = (0.0, 0.0, 0.0, 0.0, 0.0, math.radians(1.0))
        with self.assertRaisesRegex(ValueError, "six velocities"):
            validate_trajectory_samples(
                [
                    TrajectorySample(start, (), (), 0.0),
                    TrajectorySample(goal, (), (), 1.0),
                ],
                expected_start=start,
                expected_goal=goal,
            )

    def test_planned_non_target_cubic_limit_cannot_be_widened_by_caller(self):
        start = (0.0,) * 6
        goal = (0.0,) * 5 + (math.radians(0.9),)
        non_target_velocity = (0.05,) + (0.0,) * 5
        with self.assertRaisesRegex(ValueError, "non-target joint excursion"):
            validate_trajectory_samples(
                [
                    TrajectorySample(start, non_target_velocity, (0.0,) * 6, 0.0),
                    TrajectorySample(goal, non_target_velocity, (0.0,) * 6, 2.0),
                ],
                expected_start=start,
                expected_goal=goal,
                max_non_target_excursion_rad=1.0,
                max_endpoint_velocity_rad_s=0.1,
                max_velocity_rad_s=1.0,
                max_acceleration_rad_s2=1.0,
            )

    def test_execution_recheck_models_live_first_pvt_segment(self):
        start = (0.0,) * 6
        goal = (0.0,) * 5 + (math.radians(0.9),)
        samples = [
            TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.0),
            TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 30.0),
        ]
        live_start = (0.0,) * 5 + (0.0009,)
        live_velocity = (0.0,) * 5 + (0.0049,)
        with self.assertRaisesRegex(ValueError, "cubic joint_6 trajectory"):
            validate_trajectory_samples(
                samples,
                expected_start=start,
                expected_goal=goal,
                execution_start_positions=live_start,
                execution_start_velocities=live_velocity,
            )

    def test_execution_recheck_rejects_reverse_live_first_pvt_cubic(self):
        start = (0.0,) * 6
        goal = (0.0,) * 5 + (math.radians(0.9),)
        samples = [
            TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.0),
            TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 30.0),
        ]
        live_start = (0.0,) * 5 + (-0.0009,)
        live_velocity = (0.0,) * 5 + (-0.0049,)
        with self.assertRaisesRegex(ValueError, "cubic joint_6 trajectory"):
            validate_trajectory_samples(
                samples,
                expected_start=start,
                expected_goal=goal,
                execution_start_positions=live_start,
                execution_start_velocities=live_velocity,
            )

    def test_safe_live_first_pvt_segment_passes_execution_recheck(self):
        start = (0.0,) * 6
        goal = (0.0,) * 5 + (math.radians(0.9),)
        samples = [
            TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.0),
            TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 2.0),
        ]
        live_start = (0.0,) * 5 + (0.0002,)
        metrics = validate_trajectory_samples(
            samples,
            expected_start=start,
            expected_goal=goal,
            execution_start_positions=live_start,
            execution_start_velocities=(0.0,) * 6,
        )
        self.assertAlmostEqual(metrics["max_start_error_rad"], 0.0002)

    def test_live_velocity_quantization_keeps_exact_reverse_arc(self):
        recorded_start_j6 = 0.2636900823702944
        start = (0.0,) * 5 + (recorded_start_j6,)
        first_endpoint = (0.0,) * 5 + (0.2634900823702944,)
        goal = (0.0,) * 5 + (recorded_start_j6 - math.radians(0.9),)
        first_endpoint_velocity = (0.0,) * 5 + (-0.004,)
        samples = [
            TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.0),
            TrajectorySample(
                first_endpoint,
                first_endpoint_velocity,
                (0.0,) * 6,
                0.1,
            ),
            TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 1.256624138),
        ]
        observed_tm_velocity = (0.0,) * 5 + (1.409908760105131e-6,)

        metrics = validate_trajectory_samples(
            samples,
            expected_start=start,
            expected_goal=goal,
            execution_start_positions=start,
            execution_start_velocities=observed_tm_velocity,
        )

        direct_travel = abs(goal[-1] - start[-1])
        self.assertGreater(metrics["target_path_length_rad"], direct_travel)
        self.assertLess(metrics["target_path_length_rad"] - direct_travel, 1e-9)
        self.assertAlmostEqual(
            metrics["max_live_start_reverse_excursion_rad"],
            2.48130405111624e-11,
            delta=1e-15,
        )

    def test_live_reverse_allowance_is_hard_capped_at_two_microradians_per_second(self):
        start = (0.0,) * 6
        goal = (0.0,) * 5 + (-math.radians(0.9),)
        samples = [
            TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.0),
            TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 2.0),
        ]
        above_hard_cap = (0.0,) * 5 + (2.01e-6,)

        with self.assertRaisesRegex(ValueError, "cubic joint_6 trajectory reverses"):
            validate_trajectory_samples(
                samples,
                expected_start=start,
                expected_goal=goal,
                execution_start_positions=start,
                execution_start_velocities=above_hard_cap,
                max_live_start_reverse_velocity_rad_s=1.0,
            )

    def test_live_reverse_excursion_is_hard_capped_for_long_first_segment(self):
        start = (0.0,) * 6
        goal = (0.0,) * 5 + (-math.radians(0.9),)
        live_velocity = (0.0,) * 5 + (2e-6,)
        samples = [
            TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.0),
            TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 29.0),
            TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 30.0),
        ]

        with self.assertRaisesRegex(ValueError, "live-start cubic joint_6"):
            validate_trajectory_samples(
                samples,
                expected_start=start,
                expected_goal=goal,
                execution_start_positions=start,
                execution_start_velocities=live_velocity,
                max_live_start_reverse_excursion_rad=1.0,
            )

    def test_planned_reverse_allowance_remains_one_microradian_per_second(self):
        start = (0.0,) * 6
        goal = (0.0,) * 5 + (-math.radians(0.9),)
        reverse_velocity = (0.0,) * 5 + (1.5e-6,)
        samples = [
            TrajectorySample(start, reverse_velocity, (0.0,) * 6, 0.0),
            TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 2.0),
        ]

        with self.assertRaisesRegex(ValueError, "cubic joint_6 trajectory reverses"):
            validate_trajectory_samples(
                samples,
                expected_start=start,
                expected_goal=goal,
            )

    def test_live_non_target_offset_uses_physical_not_planning_limit(self):
        start = (0.0,) * 6
        goal = (0.0,) * 5 + (math.radians(0.9),)
        samples = [
            TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.0),
            TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 2.0),
        ]
        live_start = (0.0009,) + (0.0,) * 5
        metrics = validate_trajectory_samples(
            samples,
            expected_start=start,
            expected_goal=goal,
            execution_start_positions=live_start,
            execution_start_velocities=(0.0,) * 6,
        )
        self.assertGreater(metrics["max_non_target_excursion_rad"], 0.0008)

    def test_execution_recheck_uses_first_transmitted_pvt_endpoint(self):
        start = (0.0,) * 6
        goal = (0.0,) * 5 + (math.radians(0.9),)
        skipped = (0.0,) * 5 + (0.0001,)
        midpoint = (0.0,) * 5 + (math.radians(0.45),)
        samples = [
            TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.0),
            TrajectorySample(skipped, (0.0,) * 6, (0.0,) * 6, 0.01),
            TrajectorySample(midpoint, (0.0,) * 6, (0.0,) * 6, 1.0),
            TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 2.0),
        ]
        metrics = validate_trajectory_samples(
            samples,
            expected_start=start,
            expected_goal=goal,
            execution_start_positions=(0.0,) * 5 + (0.0002,),
            execution_start_velocities=(0.0,) * 6,
        )
        self.assertEqual(metrics["tm_pvt_skipped_points"], 1)
        self.assertAlmostEqual(metrics["first_pvt_segment_duration_s"], 1.0)

    def test_execution_start_arrays_fail_closed(self):
        start = (0.0,) * 6
        goal = (0.0,) * 5 + (math.radians(0.9),)
        samples = [
            TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.0),
            TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 2.0),
        ]
        with self.assertRaisesRegex(ValueError, "supplied together"):
            validate_trajectory_samples(
                samples,
                expected_start=start,
                expected_goal=goal,
                execution_start_positions=start,
            )
        with self.assertRaisesRegex(ValueError, "six joints"):
            validate_trajectory_samples(
                samples,
                expected_start=start,
                expected_goal=goal,
                execution_start_positions=(0.0,) * 5,
                execution_start_velocities=(0.0,) * 5,
            )
        with self.assertRaisesRegex(ValueError, "must be finite"):
            validate_trajectory_samples(
                samples,
                expected_start=start,
                expected_goal=goal,
                execution_start_positions=(0.0,) * 5 + (float("nan"),),
                execution_start_velocities=(0.0,) * 6,
            )

    def test_live_first_pvt_non_target_cubic_keeps_hard_physical_limit(self):
        start = (0.0,) * 6
        goal = (0.0,) * 5 + (math.radians(0.9),)
        samples = [
            TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.0),
            TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 30.0),
        ]
        live_start = (0.0029,) + (0.0,) * 5
        live_velocity = (0.0049,) + (0.0,) * 5
        with self.assertRaisesRegex(
            ValueError,
            "one-degree cap|non-target joint excursion",
        ):
            validate_trajectory_samples(
                samples,
                expected_start=start,
                expected_goal=goal,
                execution_start_positions=live_start,
                execution_start_velocities=live_velocity,
                max_start_error_rad=0.003,
                max_non_target_excursion_rad=1.0,
            )

    def test_physical_start_to_endpoint_travel_is_hard_capped(self):
        start = (0.0,) * 6
        goal = (0.0,) * 5 + (math.radians(0.9),)
        planned_endpoint = goal[:-1] + (goal[-1] + 0.00099,)
        samples = [
            TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.0),
            TrajectorySample(planned_endpoint, (0.0,) * 6, (0.0,) * 6, 2.0),
        ]
        live_start = (0.0,) * 5 + (-0.00099,)
        with self.assertRaisesRegex(ValueError, "guarded J6 segment|path length"):
            validate_trajectory_samples(
                samples,
                expected_start=start,
                expected_goal=goal,
                execution_start_positions=live_start,
                execution_start_velocities=(0.0,) * 6,
            )

    def test_tm_driver_short_final_segment_is_emulated(self):
        start = (0.0,) * 6
        midpoint = (0.0,) * 5 + (math.radians(0.5),)
        near_goal = (0.0,) * 5 + (math.radians(0.95),)
        goal = (0.0,) * 5 + (math.radians(1.0),)
        samples = [
            TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.0),
            TrajectorySample(midpoint, (0.0,) * 6, (0.0,) * 6, 1.0),
            TrajectorySample(near_goal, (0.0,) * 6, (0.0,) * 6, 2.0),
            TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 2.01),
        ]
        metrics = validate_trajectory_samples(
            samples,
            expected_start=start,
            expected_goal=goal,
        )
        self.assertEqual(metrics["sample_count"], 4)
        self.assertEqual(metrics["tm_pvt_sample_count"], 3)
        self.assertEqual(metrics["tm_pvt_skipped_points"], 1)

    def test_tm_driver_segment_threshold_is_not_caller_overridable(self):
        start = (0.0,) * 6
        unsafe = (0.0,) * 5 + (math.radians(5.0),)
        goal = (0.0,) * 5 + (math.radians(0.9),)
        samples = [
            TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.0),
            TrajectorySample(unsafe, (0.0,) * 6, (0.0,) * 6, 0.5),
            TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 2.0),
        ]
        with self.assertRaisesRegex(TypeError, "min_segment_duration_s"):
            validate_trajectory_samples(
                samples,
                expected_start=start,
                expected_goal=goal,
                min_segment_duration_s=1.0,
            )
        with self.assertRaisesRegex(ValueError, "guarded J6"):
            validate_trajectory_samples(
                samples,
                expected_start=start,
                expected_goal=goal,
            )

    def test_reverse_tolerance_cannot_hide_cubic_arc(self):
        start = (0.0,) * 6
        goal = (0.0,) * 5 + (math.radians(0.9),)
        reverse_velocity = (0.0,) * 5 + (-0.005,)
        samples = [
            TrajectorySample(start, reverse_velocity, (0.0,) * 6, 0.0),
            TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 4.0),
        ]
        with self.assertRaisesRegex(ValueError, "cubic joint_6 trajectory reverses"):
            validate_trajectory_samples(
                samples,
                expected_start=start,
                expected_goal=goal,
                max_reverse_velocity_rad_s=1.0,
                max_target_overshoot_rad=1.0,
            )

    def test_repeated_or_reverse_wrist_path_is_rejected(self):
        start = (0.0,) * 6
        goal = (0.0,) * 5 + (math.radians(1.0),)
        samples = [
            TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.0),
            TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 4.0),
            TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 8.0),
            TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 12.0),
        ]
        with self.assertRaisesRegex(ValueError, "reverses direction"):
            validate_trajectory_samples(
                samples,
                expected_start=start,
                expected_goal=goal,
            )

    def test_nonzero_first_time_and_moving_endpoint_are_rejected(self):
        start = (0.0,) * 6
        goal = (0.0,) * 5 + (math.radians(1.0),)
        with self.assertRaisesRegex(ValueError, "exactly 0"):
            validate_trajectory_samples(
                [
                    TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.1),
                    TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 2.0),
                ],
                expected_start=start,
                expected_goal=goal,
            )
        moving = (0.0,) * 5 + (0.01,)
        with self.assertRaisesRegex(ValueError, "endpoint velocity"):
            validate_trajectory_samples(
                [
                    TrajectorySample(start, moving, (0.0,) * 6, 0.0),
                    TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 2.0),
                ],
                expected_start=start,
                expected_goal=goal,
            )
        with self.assertRaisesRegex(ValueError, "cubic-interpolated trajectory velocity"):
            validate_trajectory_samples(
                [
                    TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.0),
                    TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 0.2),
                ],
                expected_start=start,
                expected_goal=goal,
            )

    def test_live_motion_envelope_uses_feedback_positions(self):
        start = (0.0, 0.1, -0.2, 0.3, -0.4, 0.5)
        goal = start[:-1] + (0.5 - math.radians(1.0),)
        safe = healthy_snapshot(
            joint_positions=start[:-1] + (0.495,),
            feedback_joint_positions=start[:-1] + (0.495,),
            joint_velocities=(0.0, 0.0, 0.0, 0.0, 0.0, -0.02),
        )
        self.assertEqual(
            motion_envelope_failures(safe, expected_start=start, expected_goal=goal),
            [],
        )
        unsafe = healthy_snapshot(
            joint_positions=(0.01,) + start[1:-1] + (0.495,),
            feedback_joint_positions=(0.01,) + start[1:-1] + (0.495,),
        )
        failures = motion_envelope_failures(
            unsafe,
            expected_start=start,
            expected_goal=goal,
        )
        self.assertTrue(any("non-target" in failure for failure in failures))


class WatsonDisplayOnlyConfigTests(unittest.TestCase):
    def test_rviz_view_has_no_motion_planning_command_plugin(self):
        demo_dir = Path(__file__).resolve().parents[1]
        config = (demo_dir / "config" / "watson_display_only.rviz").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("moveit_rviz_plugin/MotionPlanning", config)
        self.assertIn("moveit_rviz_plugin/Trajectory", config)
        self.assertIn("Trajectory Topic: /display_planned_path", config)
        self.assertIn("/watson/robot_description", config)

    def test_rviz_wrapper_uses_isolated_ros_domain(self):
        demo_dir = Path(__file__).resolve().parents[1]
        wrapper = (
            demo_dir / "scripts" / "launch_watson_display_only_rviz.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("ROS_DOMAIN_ID=219", wrapper)
        self.assertIn("ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST", wrapper)
        self.assertIn("watson_display_only.rviz", wrapper)
        self.assertIn(
            "/display_planned_path:=/watson/display_planned_path",
            wrapper,
        )

        guard_wrapper = (
            demo_dir / "scripts" / "run_watson_guarded_demo.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("ROS_DOMAIN_ID=219", guard_wrapper)
        self.assertIn("ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST", guard_wrapper)


class ImmediateFuture:
    def __init__(self, *, value=None, error: BaseException | None = None):
        self.value = value
        self.error = error

    def done(self):
        return True

    def result(self):
        if self.error is not None:
            raise self.error
        return self.value


class PendingFuture:
    def done(self):
        return False


class FakeExecuteTrajectory:
    class Goal:
        def __init__(self):
            self.trajectory = None
            self.controller_names = []


class FakeGoalStatus:
    STATUS_ACCEPTED = 1
    STATUS_EXECUTING = 2
    STATUS_CANCELING = 3
    STATUS_SUCCEEDED = 4
    STATUS_CANCELED = 5
    STATUS_ABORTED = 6


class AcceptedHandle:
    accepted = True

    def __init__(self, result_future=None, result_error: BaseException | None = None):
        self.result_future = result_future
        self.result_error = result_error

    def get_result_async(self):
        if self.result_error is not None:
            raise self.result_error
        return self.result_future


def fake_robot_trajectory(samples: tuple[TrajectorySample, ...]):
    points = []
    for sample in samples:
        seconds = int(sample.time_s)
        nanoseconds = int(round((sample.time_s - seconds) * 1_000_000_000))
        points.append(
            SimpleNamespace(
                positions=list(sample.positions),
                velocities=list(sample.velocities),
                accelerations=list(sample.accelerations),
                effort=[],
                time_from_start=SimpleNamespace(sec=seconds, nanosec=nanoseconds),
            )
        )
    return SimpleNamespace(
        joint_trajectory=SimpleNamespace(
            joint_names=list(JOINT_NAMES),
            points=points,
        ),
        multi_dof_joint_trajectory=SimpleNamespace(joint_names=[], points=[]),
    )


def execute_guard(send_future) -> tuple[WatsonGuardNode, tuple[float, ...], tuple[float, ...]]:
    start = healthy_snapshot().feedback_joint_positions
    goal = start[:-1] + (start[-1] - math.radians(0.9),)
    guard = WatsonGuardNode.__new__(WatsonGuardNode)
    guard.args = Namespace(
        service_timeout=0.01,
        max_project_speed=5,
        execution_timeout=0.01,
    )
    guard.namespace = "/watson"
    guard.execute_client = SimpleNamespace(
        wait_for_server=lambda timeout_sec: True,
        send_goal_async=lambda goal_message: send_future,
    )
    guard.command_endpoint_failures = Mock(return_value=[])
    guard.settle_action_status_callbacks = Mock(return_value=None)
    guard.action_busy_failures = Mock(return_value=[])
    guard.require_healthy = Mock(return_value=healthy_snapshot())
    guard.publisher_failures = Mock(return_value=[])
    guard.snapshot = Mock(return_value=healthy_snapshot())
    guard.verify_stationary_after_motion = Mock(return_value=None)
    guard.cancel_execution = Mock(return_value=[])
    guard.rclpy = SimpleNamespace(spin_once=lambda node, timeout_sec: None)
    guard.node = object()
    guard.ros = {
        "ExecuteTrajectory": FakeExecuteTrajectory,
        "GoalStatus": FakeGoalStatus,
    }
    guard.stop_requested = False
    guard.stop_signal = None
    guard.active_goal_handle = None
    guard.active_result_future = None
    guard.motion_command_sent = False
    return guard, start, goal


class WatsonExecuteFailureTests(unittest.TestCase):
    def call_execute(
        self,
        guard,
        start,
        goal,
        *,
        planned_samples=None,
        trajectory=None,
    ):
        if planned_samples is None:
            planned_samples = (
                TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.0),
                TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 2.0),
            )
        if trajectory is None:
            trajectory = fake_robot_trajectory(planned_samples)
        with patch(
            "scripts.run_watson_guarded_demo.validate_execute_network",
            return_value=None,
        ):
            return guard.execute_stage(
                stage_name="wrist_check",
                trajectory=trajectory,
                planned_samples=planned_samples,
                expected_start=start,
                expected_goal=goal,
                hard_reference_start=start,
            )

    def test_default_requested_motion_keeps_margin_under_hard_cap(self):
        self.assertEqual(build_parser().parse_args([]).amplitude_deg, 0.9)
        self.assertEqual(
            build_parser().parse_args([]).motion_profile,
            FIRST_MOTION_PROFILE,
        )

    def test_new_motion_profiles_are_immutable_and_resolve_effective_limits(self):
        args = build_parser().parse_args(
            ["--mode", "plan", "--motion-profile", J6_QUALIFICATION_PROFILE]
        )
        validate_cli(args)
        profile = get_j6_guard_profile(J6_QUALIFICATION_PROFILE)
        self.assertEqual(args.amplitude_deg, profile.requested_amplitude_deg)
        self.assertEqual(args.velocity_scaling, profile.velocity_scaling)
        self.assertEqual(args.acceleration_scaling, profile.acceleration_scaling)

        override = build_parser().parse_args(
            [
                "--mode",
                "plan",
                "--motion-profile",
                J6_QUALIFICATION_PROFILE,
                "--amplitude-deg",
                "6",
            ]
        )
        with self.assertRaisesRegex(ValueError, "immutable"):
            validate_cli(override)

    def test_new_profiles_require_distinct_execute_tokens(self):
        environment = {
            "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
            "ROS_DOMAIN_ID": "219",
        }
        args = build_parser().parse_args(
            [
                "--mode",
                "execute",
                "--motion-profile",
                J6_QUALIFICATION_PROFILE,
                "--arm-token",
                J6_QUALIFICATION_ARM_TOKEN,
                "--confirm-cell-clear",
            ]
        )
        with patch.dict("os.environ", environment, clear=True):
            validate_cli(args)

        showcase_without_evidence = build_parser().parse_args(
            [
                "--mode",
                "execute",
                "--motion-profile",
                J6_SHOWCASE_PROFILE,
                "--arm-token",
                J6_SHOWCASE_ARM_TOKEN,
                "--confirm-cell-clear",
            ]
        )
        with patch.dict("os.environ", environment, clear=True):
            with self.assertRaisesRegex(ValueError, "qualification-report"):
                validate_cli(showcase_without_evidence)

        for profile_name in (J6_QUALIFICATION_PROFILE, J6_SHOWCASE_PROFILE):
            wrong = build_parser().parse_args(
                [
                    "--mode",
                    "execute",
                    "--motion-profile",
                    profile_name,
                    "--arm-token",
                    ARM_TOKEN,
                    "--confirm-cell-clear",
                ]
            )
            with patch.dict("os.environ", environment, clear=True):
                with self.assertRaisesRegex(ValueError, "requires --arm-token"):
                    validate_cli(wrong)

    def test_valid_qualification_report_arms_showcase_and_records_hash(self):
        environment = {
            "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
            "ROS_DOMAIN_ID": "219",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = write_qualification_report(root)
            args = build_parser().parse_args(
                [
                    "--mode",
                    "execute",
                    "--motion-profile",
                    J6_SHOWCASE_PROFILE,
                    "--arm-token",
                    J6_SHOWCASE_ARM_TOKEN,
                    "--confirm-cell-clear",
                    "--qualification-report",
                    str(path),
                ]
            )
            with (
                patch(
                    "scripts.run_watson_guarded_demo.ARENA_DIR",
                    root,
                ),
                patch.dict("os.environ", environment, clear=True),
            ):
                validate_cli(args)
            self.assertTrue(args.qualification_gate["report_validation_passed"])
            self.assertFalse(args.qualification_gate["passed"])
            self.assertIsNone(args.qualification_gate["live_pose_match_passed"])
            self.assertEqual(
                args.qualification_gate["report_sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                args.qualification_gate["execution_stages"],
                ["j6_qualification", "return_to_start"],
            )

    def test_qualification_report_rejects_wrong_core_execution_state(self):
        cases = (
            ("plan mode", {"mode": "plan"}),
            ("wrong profile", {"motion_profile": J6_SHOWCASE_PROFILE}),
            ("failed status", {"status": "failed_closed"}),
            ("no command", {"motion_commanded": False}),
            ("old schema", {"schema_version": REPORT_SCHEMA_VERSION - 1}),
        )
        environment = {"ROS_DOMAIN_ID": "219"}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch("scripts.run_watson_guarded_demo.ARENA_DIR", root):
                for label, overrides in cases:
                    with self.subTest(label=label):
                        path = write_qualification_report(
                            root,
                            qualification_report_payload(**overrides),
                        )
                        with patch.dict("os.environ", environment, clear=True):
                            with self.assertRaisesRegex(ValueError, "qualification report"):
                                validate_qualification_report(path)

    def test_qualification_report_requires_full_runner_evidence(self):
        environment = {"ROS_DOMAIN_ID": "219"}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cases = []

            missing_health = qualification_report_payload()
            missing_health.pop("stable_health")
            cases.append(("stable_health", missing_health))

            missing_trajectory = qualification_report_payload()
            missing_trajectory["plans"][0].pop("trajectory")
            cases.append(("trajectory", missing_trajectory))

            changed_trajectory = qualification_report_payload()
            changed_trajectory["plans"][0]["trajectory"]["points"][1][
                "positions_rad"
            ][-1] += 0.001
            cases.append(("trajectory hash", changed_trajectory))

            missing_physical_start = qualification_report_payload()
            missing_physical_start["execution"][0].pop(
                "physical_start_positions_rad"
            )
            cases.append(("physical_start", missing_physical_start))

            missing_revalidation = qualification_report_payload()
            missing_revalidation["execution"][0].pop(
                "execution_revalidation_metrics"
            )
            cases.append(("recomputed metrics", missing_revalidation))

            with patch("scripts.run_watson_guarded_demo.ARENA_DIR", root):
                for expected_error, payload in cases:
                    with self.subTest(expected_error=expected_error):
                        path = write_qualification_report(root, payload)
                        with patch.dict("os.environ", environment, clear=True):
                            with self.assertRaisesRegex(
                                (ValueError, KeyError),
                                expected_error,
                            ):
                                validate_qualification_report(path)

    def test_qualification_report_rejects_stale_or_invalid_return_evidence(self):
        environment = {"ROS_DOMAIN_ID": "219"}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            now = datetime.now(timezone.utc)
            stale = qualification_report_payload(timestamp=now - timedelta(hours=3))
            path = write_qualification_report(root, stale)
            with (
                patch("scripts.run_watson_guarded_demo.ARENA_DIR", root),
                patch.dict("os.environ", environment, clear=True),
            ):
                with self.assertRaisesRegex(ValueError, "stale"):
                    validate_qualification_report(path, now=now)

            wrong_return = qualification_report_payload(timestamp=now)
            wrong_return["plans"][1]["goal_positions_rad"][0] += 0.01
            path = write_qualification_report(root, wrong_return)
            with (
                patch("scripts.run_watson_guarded_demo.ARENA_DIR", root),
                patch.dict("os.environ", environment, clear=True),
            ):
                with self.assertRaisesRegex(ValueError, "expected J6 sequence"):
                    validate_qualification_report(path, now=now)

            excessive_error = qualification_report_payload(timestamp=now)
            excessive_error["execution"][1]["live_goal_error_rad"] = 0.01
            path = write_qualification_report(root, excessive_error)
            with (
                patch("scripts.run_watson_guarded_demo.ARENA_DIR", root),
                patch.dict("os.environ", environment, clear=True),
            ):
                with self.assertRaisesRegex(ValueError, "live goal error"):
                    validate_qualification_report(path, now=now)

    def test_qualification_report_path_and_permissions_fail_closed(self):
        environment = {"ROS_DOMAIN_ID": "219"}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = write_qualification_report(root)
            with (
                patch("scripts.run_watson_guarded_demo.ARENA_DIR", root),
                patch.dict("os.environ", environment, clear=True),
            ):
                with self.assertRaisesRegex(ValueError, "absolute path"):
                    validate_qualification_report(Path(path.name))

                path.chmod(0o660)
                with self.assertRaisesRegex(ValueError, "group/world writable"):
                    validate_qualification_report(path)

            path.unlink()
            target = path.parent / "qualification-target.json"
            target.write_text(
                json.dumps(qualification_report_payload()),
                encoding="utf-8",
            )
            target.chmod(0o600)
            path.symlink_to(target)
            with (
                patch("scripts.run_watson_guarded_demo.ARENA_DIR", root),
                patch.dict("os.environ", environment, clear=True),
            ):
                with self.assertRaisesRegex(ValueError, "securely read"):
                    validate_qualification_report(path)

    def test_qualification_report_is_rejected_outside_showcase_and_as_output(self):
        environment = {
            "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
            "ROS_DOMAIN_ID": "219",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = write_qualification_report(root)
            unrelated = build_parser().parse_args(
                [
                    "--mode",
                    "plan",
                    "--motion-profile",
                    J6_QUALIFICATION_PROFILE,
                    "--qualification-report",
                    str(path),
                ]
            )
            with patch.dict("os.environ", environment, clear=True):
                with self.assertRaisesRegex(ValueError, "valid only"):
                    validate_cli(unrelated)

            overwrite = build_parser().parse_args(
                [
                    "--mode",
                    "execute",
                    "--motion-profile",
                    J6_SHOWCASE_PROFILE,
                    "--arm-token",
                    J6_SHOWCASE_ARM_TOKEN,
                    "--confirm-cell-clear",
                    "--qualification-report",
                    str(path),
                    "--report",
                    str(path),
                ]
            )
            with (
                patch("scripts.run_watson_guarded_demo.ARENA_DIR", root),
                patch.dict("os.environ", environment, clear=True),
            ):
                with self.assertRaisesRegex(ValueError, "cannot overwrite"):
                    validate_cli(overwrite)

    def test_missing_showcase_evidence_never_loads_ros_or_checks_send_route(self):
        environment = {
            "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
            "ROS_DOMAIN_ID": "219",
        }
        argv = [
            "run_watson_guarded_demo.py",
            "--mode",
            "execute",
            "--motion-profile",
            J6_SHOWCASE_PROFILE,
            "--arm-token",
            J6_SHOWCASE_ARM_TOKEN,
            "--confirm-cell-clear",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.dict("os.environ", environment, clear=True),
            patch("scripts.run_watson_guarded_demo.load_ros") as load_ros,
            patch(
                "scripts.run_watson_guarded_demo.validate_execute_network"
            ) as network_check,
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(guarded_main(), 2)
        load_ros.assert_not_called()
        network_check.assert_not_called()

    def test_showcase_evidence_is_rechecked_only_before_outbound_send(self):
        now = datetime.now(timezone.utc)
        gate = {
            "passed": True,
            "report_timestamp_utc": (now - timedelta(hours=2, seconds=1)).isoformat(),
        }
        showcase_args = Namespace(
            motion_profile=J6_SHOWCASE_PROFILE,
            qualification_gate=gate,
        )
        with self.assertRaisesRegex(RuntimeError, "expired"):
            require_fresh_showcase_gate_before_send(
                showcase_args,
                "j6_showcase",
                now=now,
            )
        self.assertGreater(gate["report_age_at_outbound_send_s"], 7200.0)

        require_fresh_showcase_gate_before_send(
            showcase_args,
            "return_to_start",
            now=now,
        )
        require_fresh_showcase_gate_before_send(
            Namespace(motion_profile=J6_QUALIFICATION_PROFILE),
            "j6_qualification",
            now=now,
        )

    def test_failed_outbound_freshness_recheck_never_sends(self):
        guard, start, goal = execute_guard(ImmediateFuture(value=None))
        guard.execute_client.send_goal_async = Mock(
            return_value=ImmediateFuture(value=None)
        )
        with patch(
            "scripts.run_watson_guarded_demo.require_fresh_showcase_gate_before_send",
            side_effect=RuntimeError("qualification evidence expired"),
        ):
            with self.assertRaisesRegex(RuntimeError, "expired"):
                self.call_execute(guard, start, goal)
        guard.execute_client.send_goal_async.assert_not_called()
        self.assertFalse(guard.motion_command_sent)

    def test_j6_plan_goal_tolerance_is_tighter_than_execution_validation(self):
        self.assertEqual(J6_PLANNING_GOAL_TOLERANCE_RAD, 0.0001)
        self.assertEqual(MAX_PLANNED_GOAL_ERROR_RAD, 0.0002)
        self.assertLess(
            J6_PLANNING_GOAL_TOLERANCE_RAD,
            MAX_PLANNED_GOAL_ERROR_RAD,
        )

    def test_execute_request_cannot_consume_the_one_degree_safety_margin(self):
        args = Namespace(
            amplitude_deg=1.0,
            velocity_scaling=0.01,
            acceleration_scaling=0.01,
            max_project_speed=5,
            mode="execute",
            namespace="/watson",
            group_name="tmr_arm",
            planning_frame="base",
            arm_token=ARM_TOKEN,
            confirm_cell_clear=True,
        )
        with patch.dict(
            "os.environ",
            {"ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST", "ROS_DOMAIN_ID": "219"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "capped.*0.9"):
                validate_cli(args)

    def test_execute_network_binds_the_commissioned_watson_mac(self):
        route = SimpleNamespace(
            returncode=0,
            stdout=(
                f"{ROBOT_IP} dev {ROBOT_INTERFACE} src {ROBOT_SOURCE_IP} uid 1003"
            ),
            stderr="",
        )
        neighbour = SimpleNamespace(
            returncode=0,
            stdout=f"{ROBOT_IP} lladdr {ROBOT_MAC} REACHABLE",
            stderr="",
        )
        with (
            patch("pathlib.Path.read_text", return_value="1"),
            patch(
                "scripts.run_watson_guarded_demo.subprocess.run",
                side_effect=[route, neighbour],
            ),
        ):
            validate_execute_network()

        wrong_neighbour = SimpleNamespace(
            returncode=0,
            stdout=f"{ROBOT_IP} lladdr 00:00:00:00:00:01 REACHABLE",
            stderr="",
        )
        with (
            patch("pathlib.Path.read_text", return_value="1"),
            patch(
                "scripts.run_watson_guarded_demo.subprocess.run",
                side_effect=[route, wrong_neighbour],
            ),
        ):
            with self.assertRaisesRegex(ValueError, "identity"):
                validate_execute_network()

    def test_send_result_exception_is_stop_unverified_and_recorded(self):
        guard, start, goal = execute_guard(
            ImmediateFuture(error=RuntimeError("send result failed"))
        )
        with self.assertRaisesRegex(StopUnverifiedError, "acceptance is unknown"):
            self.call_execute(guard, start, goal)
        self.assertTrue(guard.motion_command_sent)

    def test_execution_revalidation_failure_never_calls_send(self):
        guard, start, goal = execute_guard(ImmediateFuture(value=None))
        guard.execute_client.send_goal_async = Mock(return_value=ImmediateFuture(value=None))
        with patch(
            "scripts.run_watson_guarded_demo.validate_trajectory_samples",
            side_effect=ValueError("physical PVT rejected"),
        ):
            with self.assertRaisesRegex(ValueError, "physical PVT rejected"):
                self.call_execute(guard, start, goal)
        guard.execute_client.send_goal_async.assert_not_called()
        self.assertFalse(guard.motion_command_sent)

    def test_exact_outgoing_trajectory_must_match_validated_plan(self):
        guard, start, goal = execute_guard(ImmediateFuture(value=None))
        guard.execute_client.send_goal_async = Mock(return_value=ImmediateFuture(value=None))
        safe_samples = (
            TrajectorySample(start, (0.0,) * 6, (0.0,) * 6, 0.0),
            TrajectorySample(goal, (0.0,) * 6, (0.0,) * 6, 2.0),
        )
        unsafe_goal = goal[:-1] + (goal[-1] - math.radians(5.0),)
        unsafe_trajectory = fake_robot_trajectory(
            (
                safe_samples[0],
                TrajectorySample(unsafe_goal, (0.0,) * 6, (0.0,) * 6, 2.0),
            )
        )
        with self.assertRaisesRegex(RuntimeError, "payload changed"):
            self.call_execute(
                guard,
                start,
                goal,
                planned_samples=safe_samples,
                trajectory=unsafe_trajectory,
            )
        guard.execute_client.send_goal_async.assert_not_called()
        self.assertFalse(guard.motion_command_sent)

    def test_final_stop_request_prevents_send(self):
        guard, start, goal = execute_guard(ImmediateFuture(value=None))
        guard.execute_client.send_goal_async = Mock(return_value=ImmediateFuture(value=None))

        def request_stop_after_validation(*_args, **_kwargs):
            guard.stop_requested = True
            guard.stop_signal = 2
            return {}

        with patch(
            "scripts.run_watson_guarded_demo.validate_trajectory_samples",
            side_effect=request_stop_after_validation,
        ):
            with self.assertRaisesRegex(RuntimeError, "stop requested"):
                self.call_execute(guard, start, goal)
        guard.execute_client.send_goal_async.assert_not_called()
        self.assertFalse(guard.motion_command_sent)

    def test_stop_arriving_during_send_enters_verified_cancellation(self):
        result_future = PendingFuture()
        handle = AcceptedHandle(result_future=result_future)
        guard, start, goal = execute_guard(ImmediateFuture(value=None))

        def stop_during_send(_goal):
            guard.stop_requested = True
            guard.stop_signal = 2
            return ImmediateFuture(value=handle)

        guard.execute_client.send_goal_async = Mock(side_effect=stop_during_send)
        with self.assertRaisesRegex(RuntimeError, "health gate changed"):
            self.call_execute(guard, start, goal)
        guard.cancel_execution.assert_called_once_with(handle, result_future)
        self.assertTrue(guard.motion_command_sent)

    def test_pre_accept_health_change_prints_immediate_estop_warning(self):
        guard, start, goal = execute_guard(PendingFuture())

        def request_stop_during_acceptance(_node, timeout_sec):
            guard.stop_requested = True
            guard.stop_signal = 2

        guard.rclpy.spin_once = request_stop_during_acceptance
        stderr = io.StringIO()
        with patch(
            "scripts.run_watson_guarded_demo.GOAL_ACCEPTANCE_TIMEOUT_S",
            0.001,
        ), redirect_stderr(stderr):
            with self.assertRaisesRegex(StopUnverifiedError, "acceptance is unknown"):
                self.call_execute(guard, start, goal)
        warning = stderr.getvalue()
        self.assertIn("EMERGENCY:", warning)
        self.assertIn("physical E-stop immediately", warning)

    def test_get_result_exception_attempts_cancel_and_requires_stop_proof(self):
        handle = AcceptedHandle(result_error=RuntimeError("get result failed"))
        guard, start, goal = execute_guard(ImmediateFuture(value=handle))
        guard.cancel_execution.return_value = ["no result future"]
        with self.assertRaisesRegex(StopUnverifiedError, "not fully verified"):
            self.call_execute(guard, start, goal)
        guard.cancel_execution.assert_called_once_with(handle, None)

    def test_live_monitor_exception_invokes_verified_cancellation(self):
        result_future = PendingFuture()
        handle = AcceptedHandle(result_future=result_future)
        guard, start, goal = execute_guard(ImmediateFuture(value=handle))
        guard.snapshot.side_effect = [
            healthy_snapshot(),
            RuntimeError("feedback callback failed"),
        ]
        with self.assertRaisesRegex(RuntimeError, "feedback callback failed"):
            self.call_execute(guard, start, goal)
        guard.cancel_execution.assert_called_once_with(handle, result_future)

    def test_terminal_result_exception_invokes_verified_cancellation(self):
        result_future = ImmediateFuture(error=RuntimeError("terminal result failed"))
        handle = AcceptedHandle(result_future=result_future)
        guard, start, goal = execute_guard(ImmediateFuture(value=handle))
        with self.assertRaisesRegex(RuntimeError, "terminal result failed"):
            self.call_execute(guard, start, goal)
        guard.cancel_execution.assert_called_once_with(handle, result_future)

    def test_success_without_stationary_proof_is_emergency(self):
        wrapped = SimpleNamespace(
            status=FakeGoalStatus.STATUS_SUCCEEDED,
            result=SimpleNamespace(error_code=SimpleNamespace(val=MOVEIT_SUCCESS)),
        )
        result_future = ImmediateFuture(value=wrapped)
        handle = AcceptedHandle(result_future=result_future)
        guard, start, goal = execute_guard(ImmediateFuture(value=handle))
        guard.verify_stationary_after_motion.return_value = "stationary proof failed"
        guard.cancel_execution.return_value = ["stationary proof failed"]
        with self.assertRaisesRegex(StopUnverifiedError, "not fully verified"):
            self.call_execute(guard, start, goal)

    def test_terminal_feedback_must_remain_inside_hard_motion_envelope(self):
        wrapped = SimpleNamespace(
            status=FakeGoalStatus.STATUS_SUCCEEDED,
            result=SimpleNamespace(error_code=SimpleNamespace(val=MOVEIT_SUCCESS)),
        )
        handle = AcceptedHandle(result_future=ImmediateFuture(value=wrapped))
        guard, start, goal = execute_guard(ImmediateFuture(value=handle))
        outside = start[:-1] + (start[-1] - math.radians(1.01),)
        guard.snapshot.side_effect = [
            healthy_snapshot(),
            healthy_snapshot(
                joint_positions=outside,
                feedback_joint_positions=outside,
            ),
        ]
        with self.assertRaisesRegex(RuntimeError, "post-motion health gate"):
            self.call_execute(guard, start, goal)

    def test_cancel_exceptions_are_returned_not_raised(self):
        guard, _, _ = execute_guard(ImmediateFuture(value=None))
        handle = SimpleNamespace(
            cancel_goal_async=Mock(side_effect=RuntimeError("cancel transport failed"))
        )
        terminal = SimpleNamespace(status=FakeGoalStatus.STATUS_ABORTED)
        failures = WatsonGuardNode.cancel_execution(
            guard,
            handle,
            ImmediateFuture(value=terminal),
        )
        self.assertTrue(any("cancel transport failed" in failure for failure in failures))

    def test_execute_cli_requires_isolated_explicit_ros_domain(self):
        args = Namespace(
            amplitude_deg=0.9,
            velocity_scaling=0.01,
            acceleration_scaling=0.01,
            max_project_speed=5,
            mode="execute",
            namespace="/watson",
            group_name="tmr_arm",
            planning_frame="base",
            arm_token=ARM_TOKEN,
            confirm_cell_clear=True,
        )
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "DISCOVERY_RANGE"):
                validate_cli(args)
        with patch.dict(
            "os.environ",
            {"ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST", "ROS_DOMAIN_ID": "219"},
            clear=True,
        ):
            validate_cli(args)

    def test_execute_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "execute.lock"
            first = acquire_execute_lock(lock_path)
            try:
                with self.assertRaisesRegex(ValueError, "already holds"):
                    acquire_execute_lock(lock_path)
            finally:
                first.close()
            second = acquire_execute_lock(lock_path)
            second.close()

    def test_action_idle_allows_no_retained_goal_history(self):
        guard = WatsonGuardNode.__new__(WatsonGuardNode)
        guard.ros = {"GoalStatus": FakeGoalStatus}
        guard.execute_action_status = None
        guard.controller_action_status = None
        self.assertEqual(WatsonGuardNode.action_busy_failures(guard), [])

        empty_status = SimpleNamespace(status_list=[])
        guard.execute_action_status = empty_status
        guard.controller_action_status = empty_status
        self.assertEqual(WatsonGuardNode.action_busy_failures(guard), [])

    def test_action_idle_rejects_unknown_goal_status(self):
        guard = WatsonGuardNode.__new__(WatsonGuardNode)
        guard.ros = {"GoalStatus": FakeGoalStatus}
        unknown = SimpleNamespace(status_list=[SimpleNamespace(status=0)])
        guard.execute_action_status = unknown
        guard.controller_action_status = SimpleNamespace(status_list=[])
        failures = WatsonGuardNode.action_busy_failures(guard)
        self.assertTrue(any("unknown" in failure for failure in failures))

    def test_command_provenance_rejects_interactive_move_action_client(self):
        namespace = "/watson"
        graph_nodes = [
            ("move_group", namespace),
            ("tm_driver_node", namespace),
            ("moveit_simple_controller_manager", namespace),
            ("watson_guarded_demo", "/"),
            ("interactive_rviz", namespace),
        ]

        class FakeGraphNode:
            def get_node_names_and_namespaces(self):
                return graph_nodes

            def get_service_names_and_types_by_node(self, node_name, _node_namespace):
                if node_name == "move_group":
                    return [
                        (
                            f"{namespace}/plan_kinematic_path",
                            ["moveit_msgs/srv/GetMotionPlan"],
                        )
                    ]
                return []

        def action_servers(_node, node_name, _node_namespace):
            if node_name == "move_group":
                return [
                    (
                        f"{namespace}/execute_trajectory",
                        ["moveit_msgs/action/ExecuteTrajectory"],
                    )
                ]
            if node_name == "tm_driver_node":
                return [
                    (
                        f"{namespace}/tmr_arm_controller/follow_joint_trajectory",
                        ["control_msgs/action/FollowJointTrajectory"],
                    )
                ]
            return []

        def action_clients(_node, node_name, _node_namespace):
            if node_name == "watson_guarded_demo":
                return [
                    (
                        f"{namespace}/execute_trajectory",
                        ["moveit_msgs/action/ExecuteTrajectory"],
                    )
                ]
            if node_name == "moveit_simple_controller_manager":
                return [
                    (
                        f"{namespace}/tmr_arm_controller/follow_joint_trajectory",
                        ["control_msgs/action/FollowJointTrajectory"],
                    )
                ]
            if node_name == "interactive_rviz":
                return [
                    (
                        f"{namespace}/move_action",
                        ["moveit_msgs/action/MoveGroup"],
                    )
                ]
            return []

        guard = WatsonGuardNode.__new__(WatsonGuardNode)
        guard.namespace = namespace
        guard.node = FakeGraphNode()
        guard.ros = {
            "get_action_server_names_and_types_by_node": action_servers,
            "get_action_client_names_and_types_by_node": action_clients,
        }
        failures = WatsonGuardNode.command_endpoint_failures(
            guard,
            require_execute=True,
        )
        self.assertTrue(any("interactive_rviz" in failure for failure in failures))

    def test_best_effort_report_does_not_raise(self):
        with patch(
            "scripts.run_watson_guarded_demo.write_report",
            side_effect=OSError("disk unavailable"),
        ):
            write_report_best_effort(Path("unused.json"), {})

    def test_report_writer_sets_private_mode_and_payload_digest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "report.json"
            report = {"schema_version": REPORT_SCHEMA_VERSION, "status": "test"}
            with redirect_stdout(io.StringIO()):
                write_report(path, report)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                stored[REPORT_DIGEST_FIELD],
                report_payload_sha256(stored),
            )


if __name__ == "__main__":
    unittest.main()
