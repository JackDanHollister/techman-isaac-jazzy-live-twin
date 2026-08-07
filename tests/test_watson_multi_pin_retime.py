from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

import pin_axis_3d_sim.watson_multi_pin_retime as retime_module
from pin_axis_3d_sim.watson_multi_pin_retime import (
    ARTIFACT_DIGEST_FIELD,
    ARTIFACT_STATUS,
    DEFAULT_REVIEWED_PLAN,
    DERIVATIVE_LIMIT_FRACTION,
    EXPECTED_PLAN_SHA256,
    EXPECTED_SOURCE_NUMERIC_SHA256,
    GLOBAL_TIME_SCALE,
    JOINT_NAMES,
    PVTPoint,
    TM_DRIVER_BINARY_SHA256,
    TM_DRIVER_COMMAND_HEADER_SHA256,
    TM_DRIVER_COMMAND_SOURCE_SHA256,
    TM_DRIVER_MIN_SEGMENT_DURATION_S,
    TM_DRIVER_MOVEIT_SOURCE_SHA256,
    build_retimed_artifact,
    canonical_digest,
    emulate_tm_driver_selection,
    load_reviewed_plan,
    retime_ingress_control_samples,
    serialize_filtered_message_points_to_wire,
    validate_live_first_wire_cubic,
    validate_retimed_artifact,
    validate_retimed_ingress_candidate,
    validate_reviewed_plan,
    validate_six_axis_pvt,
    verify_installed_tm_driver_provenance,
    write_private_artifact,
)

TM_DRIVER_WORKSPACE_PRESENT = retime_module.TECHMAN_WORKSPACE.exists()
requires_tm_driver_workspace = unittest.skipUnless(
    TM_DRIVER_WORKSPACE_PRESENT,
    "lab-local tm_driver workspace is absent: "
    f"{retime_module.TECHMAN_WORKSPACE}",
)


def point(index: int, time_s: float, joint_1: float = 0.0) -> PVTPoint:
    return PVTPoint(
        source_sample_index=index,
        time_s=time_s,
        positions=(joint_1, 0.0, 1.5708, 0.0, 1.5708, 0.0),
        velocities=(0.0,) * len(JOINT_NAMES),
    )


class InstalledDriverFilterEmulationTests(unittest.TestCase):
    def test_matches_interior_skip_and_short_final_replacement(self) -> None:
        source = tuple(
            point(index, time_s)
            for index, time_s in enumerate((0.0, 0.01, 0.02, 0.03, 0.04))
        )
        selected, skipped = emulate_tm_driver_selection(source)
        self.assertEqual([item.source_sample_index for item in selected], [0, 4])
        self.assertEqual(skipped, 3)
        self.assertAlmostEqual(selected[-1].time_s, 0.04)

    def test_filter_rejects_when_no_endpoint_can_be_transmitted(self) -> None:
        with self.assertRaisesRegex(ValueError, "no prior PVT endpoint"):
            emulate_tm_driver_selection((point(0, 0.0), point(1, 0.01)))


class InstalledWireSerializerTests(unittest.TestCase):
    @requires_tm_driver_workspace
    def test_exact_installed_components_and_semantics_are_hash_pinned(self) -> None:
        provenance = verify_installed_tm_driver_provenance()
        self.assertEqual(
            provenance["moveit_filter_source"]["sha256"],
            TM_DRIVER_MOVEIT_SOURCE_SHA256,
        )
        self.assertEqual(
            provenance["pvt_serializer_source"]["sha256"],
            TM_DRIVER_COMMAND_SOURCE_SHA256,
        )
        self.assertEqual(
            provenance["pvt_serializer_header"]["sha256"],
            TM_DRIVER_COMMAND_HEADER_SHA256,
        )
        self.assertEqual(
            provenance["tm_driver_binary"]["sha256"],
            TM_DRIVER_BINARY_SHA256,
        )
        semantics = provenance["verified_semantics"]
        self.assertEqual(semantics["decimal_places"], 5)
        self.assertEqual(semantics["wire_time_kind"], "relative_segment_seconds")
        self.assertFalse(semantics["zero_time_message_seed_transmitted"])

    @requires_tm_driver_workspace
    def test_changed_installed_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "tm_ros2_moveit_sct.cpp"
            changed.write_text("changed", encoding="utf-8")
            with mock.patch.object(
                retime_module,
                "TM_DRIVER_MOVEIT_SOURCE_PATH",
                changed,
            ):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    verify_installed_tm_driver_provenance()

    def test_fixed_five_degree_and_relative_time_round_trip_is_exact(self) -> None:
        message = (
            point(0, 0.0),
            PVTPoint(
                source_sample_index=1,
                time_s=0.025006,
                positions=(math.radians(1.23456789), 0.0, 1.5708, 0.0, 1.5708, 0.0),
                velocities=(math.radians(-2.34567891), 0.0, 0.0, 0.0, 0.0, 0.0),
            ),
            PVTPoint(
                source_sample_index=2,
                time_s=0.050013,
                positions=(math.radians(1.3), 0.0, 1.5708, 0.0, 1.5708, 0.0),
                velocities=(0.0,) * 6,
            ),
        )
        wire = serialize_filtered_message_points_to_wire(message)
        self.assertEqual(wire[0].position_degrees_text[0], "1.23457")
        self.assertEqual(wire[0].velocity_degrees_s_text[0], "-2.34568")
        self.assertEqual(wire[0].segment_duration_text, "0.02501")
        self.assertEqual(wire[1].segment_duration_text, "0.02501")
        self.assertEqual(wire[1].cumulative_time_ticks_1e5, 5002)
        self.assertEqual(wire[1].cumulative_time_s, 0.05002)
        self.assertEqual(
            wire[0].positions_rad[0],
            math.radians(float(wire[0].position_degrees_text[0])),
        )


class ImmutableSixAxisPVTTests(unittest.TestCase):
    def test_small_six_axis_candidate_passes_without_filter_changes(self) -> None:
        points = (point(0, 0.0), point(1, 0.05, joint_1=0.0005))
        metrics = validate_six_axis_pvt(points)
        self.assertEqual(metrics["post_candidate_filter_skipped_points"], 0)
        self.assertEqual(metrics["point_count_including_zero_seed"], 2)

    def test_sub_25ms_candidate_fails_filter_proof(self) -> None:
        candidate = (
            point(0, 0.0),
            point(1, 0.01, joint_1=0.0001),
            point(2, 0.03, joint_1=0.0002),
        )
        with self.assertRaisesRegex(ValueError, "changed by the tm_driver filter"):
            validate_six_axis_pvt(candidate)

    def test_immutable_velocity_limit_cannot_be_widened(self) -> None:
        candidate = (point(0, 0.0), point(1, 0.06, joint_1=0.3))
        with self.assertRaisesRegex(ValueError, "joint_1 cubic velocity"):
            validate_six_axis_pvt(candidate)

    def test_exact_cubic_acceleration_limit_is_enforced(self) -> None:
        candidate = (point(0, 0.0), point(1, 0.025, joint_1=0.01))
        with self.assertRaisesRegex(ValueError, "joint_1 cubic acceleration"):
            validate_six_axis_pvt(candidate)

    def test_message_only_proof_is_explicitly_not_a_wire_claim(self) -> None:
        metrics = validate_six_axis_pvt(
            (point(0, 0.0), point(1, 0.05, joint_1=0.0005))
        )
        self.assertFalse(metrics["wire_serialization_included"])
        self.assertFalse(metrics["physical_wire_proof_claimed"])


class ArbitraryIngressRetimeTests(unittest.TestCase):
    @staticmethod
    def samples(
        times: tuple[float, ...],
        *,
        final_joint_1: float = 0.0,
    ) -> list[dict]:
        result = []
        denominator = max(len(times) - 1, 1)
        for index, time_s in enumerate(times):
            result.append(
                {
                    "time_from_start_seconds": time_s,
                    "joint_positions_rad": [
                        final_joint_1 * index / denominator,
                        0.0,
                        1.5708,
                        0.0,
                        1.5708,
                        0.0,
                    ],
                    "joint_velocities_rad_s": [0.0] * 6,
                }
            )
        return result

    @requires_tm_driver_workspace
    def test_public_ingress_retimer_is_offline_and_filter_stable(self) -> None:
        samples = self.samples((0.0, 0.01, 0.02, 0.03))
        candidate = retime_ingress_control_samples(samples)
        metrics = validate_retimed_ingress_candidate(candidate, samples)
        self.assertEqual(candidate["source_filter_skipped_points"], 2)
        self.assertEqual(len(candidate["controller_points"]), 2)
        self.assertEqual(metrics["post_candidate_filter_skipped_points"], 0)
        self.assertEqual(len(candidate["serialized_wire_points"]), 1)
        self.assertFalse(candidate["message_point_physical_wire_proof_claimed"])
        self.assertFalse(
            candidate["full_physical_wire_trajectory_proof_claimed"]
        )
        self.assertEqual(
            candidate["controller_points"][-1]["time_from_start_nanoseconds"],
            31_500_000,
        )
        self.assertTrue(candidate["safety"]["offline_only"])
        for field, value in candidate["safety"].items():
            if field != "offline_only":
                self.assertIs(value, False, field)

    @requires_tm_driver_workspace
    def test_retimed_ingress_point_tamper_is_rejected(self) -> None:
        samples = self.samples((0.0, 0.01, 0.02, 0.03))
        candidate = retime_ingress_control_samples(samples)
        tampered = copy.deepcopy(candidate)
        tampered["controller_points"][-1]["joint_positions_rad"][0] += 1e-6
        with self.assertRaisesRegex(ValueError, "does not exactly derive"):
            validate_retimed_ingress_candidate(tampered, samples)

    @requires_tm_driver_workspace
    def test_retimed_ingress_wire_text_tamper_is_rejected(self) -> None:
        samples = self.samples((0.0, 0.01, 0.02, 0.03))
        candidate = retime_ingress_control_samples(samples)
        tampered = copy.deepcopy(candidate)
        tampered["serialized_wire_points"][0][
            "wire_segment_duration_seconds_fixed_5"
        ] = "0.03151"
        with self.assertRaisesRegex(ValueError, "does not exactly derive"):
            validate_retimed_ingress_candidate(tampered, samples)

    @requires_tm_driver_workspace
    def test_live_first_q_v_changes_exact_wire_acceleration(self) -> None:
        samples = self.samples((0.0, 0.05), final_joint_1=0.0005)
        candidate = retime_ingress_control_samples(samples)
        first_wire = candidate["serialized_wire_points"][0]
        start_positions = samples[0]["joint_positions_rad"]
        stationary = validate_live_first_wire_cubic(
            start_positions,
            [0.0] * 6,
            first_wire,
        )
        moving = validate_live_first_wire_cubic(
            start_positions,
            [0.001, 0.0, 0.0, 0.0, 0.0, 0.0],
            first_wire,
        )
        self.assertTrue(stationary["physical_wire_proof_claimed"])
        self.assertNotEqual(
            stationary["maximum_cubic_acceleration_rad_s2"][0],
            moving["maximum_cubic_acceleration_rad_s2"][0],
        )
        with self.assertRaisesRegex(ValueError, "joint_1 cubic acceleration"):
            validate_live_first_wire_cubic(
                start_positions,
                [-0.03, 0.0, 0.0, 0.0, 0.0, 0.0],
                first_wire,
            )
        longer = retime_ingress_control_samples(
            self.samples((0.0, 0.03, 0.06), final_joint_1=0.0005)
        )
        with self.assertRaisesRegex(ValueError, "requires the first"):
            validate_live_first_wire_cubic(
                start_positions,
                [0.0] * 6,
                longer["serialized_wire_points"][1],
            )

    def test_ingress_still_rejects_immutable_velocity_limit(self) -> None:
        samples = self.samples((0.0, 0.05), final_joint_1=0.3)
        with self.assertRaisesRegex(ValueError, "joint_1 cubic velocity"):
            retime_ingress_control_samples(samples)

    def test_ingress_rejects_trajectory_too_short_for_driver(self) -> None:
        samples = self.samples((0.0, 0.01))
        with self.assertRaisesRegex(ValueError, "no prior PVT endpoint"):
            retime_ingress_control_samples(samples)

    def test_ingress_rejects_missing_time_cleanly(self) -> None:
        samples = self.samples((0.0, 0.03))
        del samples[1]["time_from_start_seconds"]
        with self.assertRaisesRegex(ValueError, "time must be a finite number"):
            retime_ingress_control_samples(samples)


@requires_tm_driver_workspace
class ReviewedSevenPinRetimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = load_reviewed_plan(DEFAULT_REVIEWED_PLAN)
        cls.artifact = build_retimed_artifact(
            DEFAULT_REVIEWED_PLAN,
            now=datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc),
        )

    def resigned(self, artifact: dict) -> dict:
        artifact[ARTIFACT_DIGEST_FIELD] = canonical_digest(artifact)
        return artifact

    def test_exact_reviewed_plan_builds_expected_offline_candidate(self) -> None:
        artifact = self.artifact
        metrics = validate_retimed_artifact(artifact, self.plan)
        self.assertEqual(artifact["status"], ARTIFACT_STATUS)
        self.assertEqual(artifact["source_plan_sha256"], EXPECTED_PLAN_SHA256)
        self.assertEqual(
            artifact["source_numeric_sample_sha256"],
            EXPECTED_SOURCE_NUMERIC_SHA256,
        )
        self.assertEqual(artifact["format_version"], 2)
        self.assertEqual(artifact["retiming"]["global_time_scale"], GLOBAL_TIME_SCALE)
        self.assertEqual(metrics["stage_count"], 49)
        self.assertEqual(metrics["source_sample_count"], 18102)
        self.assertEqual(
            metrics["message_point_count_including_stage_zero_seeds"], 2280
        )
        self.assertEqual(
            metrics["driver_transmitted_wire_endpoint_count"], 2231
        )
        self.assertEqual(metrics["source_filter_skipped_points"], 15822)
        self.assertEqual(
            metrics["post_message_candidate_filter_skipped_points"], 0
        )
        self.assertEqual(metrics["wire_internal_stage_count_validated"], 49)
        self.assertEqual(
            metrics["wire_first_cubic_count_pending_live_validation"], 49
        )
        self.assertGreaterEqual(
            metrics["wire_internal_minimum_segment_duration_seconds"],
            TM_DRIVER_MIN_SEGMENT_DURATION_S,
        )
        self.assertLessEqual(
            metrics["wire_internal_maximum_acceleration_limit_utilization"],
            DERIVATIVE_LIMIT_FRACTION,
        )
        self.assertFalse(metrics["message_point_physical_wire_proof_claimed"])
        self.assertFalse(metrics["full_physical_wire_stage_proof_claimed"])
        self.assertFalse(artifact["proof_scope"]["live_first_cubics_validated"])
        self.assertEqual(
            artifact["installed_tm_driver_provenance"]["tm_driver_binary"][
                "sha256"
            ],
            TM_DRIVER_BINARY_SHA256,
        )
        self.assertTrue(artifact["safety"]["offline_only"])
        for field, value in artifact["safety"].items():
            if field != "offline_only":
                self.assertIs(value, False, field)

    def test_all_stage_boundaries_are_exact_ready_to_ready(self) -> None:
        ready = tuple(self.artifact["ready_joint_positions_rad"])
        previous = ready
        for stage in self.artifact["stages"]:
            start = tuple(stage["start_joint_positions_rad"])
            end = tuple(stage["end_joint_positions_rad"])
            first = tuple(stage["controller_points"][0]["joint_positions_rad"])
            last = tuple(stage["controller_points"][-1]["joint_positions_rad"])
            self.assertEqual(start, previous)
            self.assertEqual(first, start)
            self.assertEqual(last, end)
            self.assertEqual(
                stage["controller_points"][0]["joint_velocities_rad_s"],
                [0.0] * 6,
            )
            self.assertEqual(
                stage["controller_points"][-1]["joint_velocities_rad_s"],
                [0.0] * 6,
            )
            self.assertEqual(
                len(stage["serialized_wire_points"]),
                len(stage["controller_points"]) - 1,
            )
            self.assertEqual(
                stage["serialized_wire_internal_validation"]["status"],
                "validated_internal_wire_cubics_first_live_cubic_pending",
            )
            self.assertFalse(stage["full_physical_wire_stage_proof_claimed"])
            self.assertFalse(stage["first_wire_cubic"]["validated_offline"])
            previous = end
            if stage["stage_name"] == "return_ready":
                self.assertEqual(previous, ready)

    def test_message_only_v1_artifact_is_rejected(self) -> None:
        legacy = copy.deepcopy(self.artifact)
        legacy["format_version"] = 1
        self.resigned(legacy)
        with self.assertRaisesRegex(ValueError, "unrounded v1 evidence"):
            validate_retimed_artifact(legacy, self.plan)

    def test_plan_file_byte_tamper_is_rejected_before_retime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / "plan.json"
            tampered.write_bytes(DEFAULT_REVIEWED_PLAN.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_reviewed_plan(tampered)

    def test_in_memory_numeric_tamper_is_rejected(self) -> None:
        tampered = dict(self.plan)
        tampered["specimens"] = list(self.plan["specimens"])
        specimen = dict(tampered["specimens"][0])
        tampered["specimens"][0] = specimen
        specimen["stages"] = list(specimen["stages"])
        stage = dict(specimen["stages"][0])
        specimen["stages"][0] = stage
        stage["control_samples"] = list(stage["control_samples"])
        sample = dict(stage["control_samples"][1])
        stage["control_samples"][1] = sample
        sample["joint_positions"] = list(sample["joint_positions"])
        sample["joint_positions"][0] += 1e-8
        with self.assertRaisesRegex(ValueError, "stage numeric hash"):
            validate_reviewed_plan(tampered)

    def test_resigned_point_tamper_cannot_change_reviewed_path(self) -> None:
        tampered = copy.deepcopy(self.artifact)
        tampered["stages"][0]["controller_points"][1][
            "joint_positions_rad"
        ][0] += 1e-6
        self.resigned(tampered)
        with self.assertRaisesRegex(ValueError, "does not exactly derive"):
            validate_retimed_artifact(tampered, self.plan)

    def test_resigned_limit_tamper_cannot_widen_guard(self) -> None:
        tampered = copy.deepcopy(self.artifact)
        tampered["immutable_six_axis_limits"][
            "joint_velocity_limits_rad_s"
        ][0] = 999.0
        self.resigned(tampered)
        with self.assertRaisesRegex(ValueError, "Immutable six-axis limits"):
            validate_retimed_artifact(tampered, self.plan)

    def test_resigned_filter_metadata_tamper_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.artifact)
        tampered["stages"][0]["source_filter_skipped_points"] -= 1
        self.resigned(tampered)
        with self.assertRaisesRegex(ValueError, "does not exactly derive"):
            validate_retimed_artifact(tampered, self.plan)

    def test_resigned_wire_value_tamper_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.artifact)
        tampered["stages"][0]["serialized_wire_points"][0][
            "joint_positions_degrees_fixed_5"
        ][0] = "999.00000"
        self.resigned(tampered)
        with self.assertRaisesRegex(ValueError, "does not exactly derive"):
            validate_retimed_artifact(tampered, self.plan)

    def test_payload_digest_tamper_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.artifact)
        tampered["source_plan"] = "/changed/without/resigning.json"
        with self.assertRaisesRegex(ValueError, "payload digest"):
            validate_retimed_artifact(tampered, self.plan)

    def test_private_writer_is_exclusive_and_mode_0600(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "retimed.json"
            written = write_private_artifact(output, self.artifact)
            self.assertEqual(stat.S_IMODE(written.stat().st_mode), 0o600)
            stored = json.loads(written.read_text(encoding="utf-8"))
            self.assertEqual(
                stored[ARTIFACT_DIGEST_FIELD], canonical_digest(stored)
            )
            with self.assertRaises(FileExistsError):
                write_private_artifact(output, self.artifact)

    def test_private_writer_rejects_invalid_digest(self) -> None:
        tampered = copy.deepcopy(self.artifact)
        tampered["warning"] = "changed"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "retimed.json"
            with self.assertRaisesRegex(ValueError, "invalid digest"):
                write_private_artifact(output, tampered)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
