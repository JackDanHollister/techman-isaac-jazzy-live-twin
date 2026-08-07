from __future__ import annotations

import hashlib
import json
import math
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pin_axis_3d_sim.grasp_cycle import (
    ARM_JOINT_NAMES,
    PHASE_ORDER,
    build_grasp_cycle,
    cycle_evidence,
)


ARENA_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ARENA_DIR / "config/isaac_grasp_cycle.yaml"
EXPECTED_CYCLE_HASHES = {
    "arm_positions_float64_sha256": (
        "b918710fd3c509f08091940ba93f2c19a6c17f1a389525826acf39fa62406272"
    ),
    "arm_positions_and_velocities_float64_sha256": (
        "92ab47a074576afa1f6d816752c2467396a508d590d40f0519545e13d9e1c86b"
    ),
    "finger_positions_float64_sha256": (
        "bd3bc810308640c5070db57fbf485df09eaf6461bf6d5edcd66e822653209f9c"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_arena_path(value: str) -> Path:
    path = (ARENA_DIR / value).resolve()
    path.relative_to(ARENA_DIR.resolve())
    return path


def vector(element: ET.Element, attribute: str) -> tuple[float, ...]:
    return tuple(float(value) for value in element.get(attribute, "").split())


class GraspCycleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        plan_path = resolve_arena_path(cls.config["arm_choreography"]["source_plan"])
        cls.plan_path = plan_path
        if not plan_path.is_file():
            raise unittest.SkipTest("Generated synthetic-pick plan is not present")
        cls.plan = json.loads(plan_path.read_text(encoding="utf-8"))
        gripper = cls.config["gripper_motion"]
        cls.commands = build_grasp_cycle(
            cls.plan,
            finger_open_m=float(gripper["open_position_m"]),
            finger_closed_m=float(gripper["closed_position_m"]),
            finger_speed_m_s=float(gripper["per_finger_speed_m_s"]),
            hold_seconds=float(gripper["hold_seconds"]),
        )
        cls.evidence = cycle_evidence(
            cls.commands,
            float(cls.plan["control_dt_seconds"]),
        )

    def phase_commands(self, phase: str) -> list[dict[str, Any]]:
        return [command for command in self.commands if command["phase"] == phase]

    def test_phase_and_attachment_event_order_is_exact(self) -> None:
        observed_phases = tuple(
            dict.fromkeys(command["phase"] for command in self.commands)
        )
        self.assertEqual(observed_phases, PHASE_ORDER)
        events = [
            (index, command["phase"], command["attachment_event"])
            for index, command in enumerate(self.commands)
            if command["attachment_event"] is not None
        ]
        self.assertEqual(
            [(phase, event) for _, phase, event in events],
            [("close_gripper", "attach"), ("release_pin", "release")],
        )
        attach_index, release_index = events[0][0], events[1][0]
        lift_index = next(
            index
            for index, command in enumerate(self.commands)
            if command["phase"] == "lift_pin"
        )
        open_index = next(
            index
            for index, command in enumerate(self.commands)
            if command["phase"] == "open_gripper"
        )
        self.assertLess(attach_index, lift_index)
        self.assertLess(lift_index, release_index)
        self.assertLess(release_index, open_index)

    def test_arm_cycle_is_an_exact_six_joint_round_trip(self) -> None:
        arm_positions = np.asarray(
            [command["arm_positions"] for command in self.commands],
            dtype=np.float64,
        )
        self.assertEqual(arm_positions.shape, (len(self.commands), 6))
        self.assertEqual(tuple(self.evidence["arm_joint_names"]), ARM_JOINT_NAMES)
        np.testing.assert_array_equal(arm_positions[0], arm_positions[-1])
        self.assertTrue(self.evidence["start_equals_end"])

        descend_endpoint = self.phase_commands("descend_to_grasp")[-1]["arm_positions"]
        replaced_endpoint = self.phase_commands("replace_pin")[-1]["arm_positions"]
        np.testing.assert_allclose(descend_endpoint, replaced_endpoint, atol=1.0e-12)
        pregrasp_endpoint = self.phase_commands("approach_pregrasp")[-1]["arm_positions"]
        retreat_endpoint = self.phase_commands("retreat_to_pregrasp")[-1]["arm_positions"]
        np.testing.assert_allclose(pregrasp_endpoint, retreat_endpoint, atol=1.0e-12)
        self.assertLessEqual(
            self.evidence["maximum_arm_command_step_rad"],
            float(self.plan["maximum_control_step_rad"]) + 1.0e-12,
        )

    def test_fingers_close_hold_and_reopen_within_speed_and_travel(self) -> None:
        gripper = self.config["gripper_motion"]
        open_position = float(gripper["open_position_m"])
        closed_position = float(gripper["closed_position_m"])
        control_dt = float(self.plan["control_dt_seconds"])
        maximum_step = float(gripper["per_finger_speed_m_s"]) * control_dt

        closing = np.asarray(
            [command["finger_position_m"] for command in self.phase_commands("close_gripper")]
        )
        opening = np.asarray(
            [command["finger_position_m"] for command in self.phase_commands("open_gripper")]
        )
        self.assertTrue(np.all(np.diff(closing) > 0.0))
        self.assertTrue(np.all(np.diff(opening) < 0.0))
        self.assertAlmostEqual(float(closing[-1]), closed_position)
        self.assertAlmostEqual(float(opening[-1]), open_position)
        self.assertLessEqual(float(np.max(np.diff(closing))), maximum_step + 1.0e-12)
        self.assertLessEqual(float(np.max(-np.diff(opening))), maximum_step + 1.0e-12)

        attach = next(
            command for command in self.commands if command["attachment_event"] == "attach"
        )
        release = next(
            command for command in self.commands if command["attachment_event"] == "release"
        )
        self.assertAlmostEqual(attach["finger_position_m"], closed_position)
        self.assertAlmostEqual(release["finger_position_m"], closed_position)
        for phase in ("lift_pin", "hold_lift", "replace_pin", "hold_replaced_closed"):
            self.assertTrue(
                all(
                    math.isclose(command["finger_position_m"], closed_position)
                    for command in self.phase_commands(phase)
                )
            )

    def test_pinned_inputs_and_cycle_evidence_are_stable(self) -> None:
        choreography = self.config["arm_choreography"]
        asset = self.config["articulated_asset"]
        metadata_path = resolve_arena_path(asset["tool_metadata"])
        import_report_path = resolve_arena_path(asset["import_report"])
        staged_manifest_path = resolve_arena_path(asset["staged_manifest"])
        self.assertEqual(sha256_file(self.plan_path), choreography["source_plan_sha256"])
        self.assertEqual(sha256_file(metadata_path), asset["tool_metadata_sha256"])
        self.assertEqual(
            sha256_file(import_report_path), asset["import_report_sha256"]
        )
        self.assertEqual(
            sha256_file(staged_manifest_path), asset["staged_manifest_sha256"]
        )

        gripper = self.config["gripper_motion"]
        repeated_commands = build_grasp_cycle(
            self.plan,
            finger_open_m=float(gripper["open_position_m"]),
            finger_closed_m=float(gripper["closed_position_m"]),
            finger_speed_m_s=float(gripper["per_finger_speed_m_s"]),
            hold_seconds=float(gripper["hold_seconds"]),
        )
        repeated_evidence = cycle_evidence(
            repeated_commands,
            float(self.plan["control_dt_seconds"]),
        )
        self.assertEqual(self.evidence, repeated_evidence)
        self.assertEqual(self.evidence["command_count"], 9109)
        for field, expected in EXPECTED_CYCLE_HASHES.items():
            self.assertEqual(self.evidence[field], expected)

    def test_scope_and_payload_geometry_match_the_10mm_baseline(self) -> None:
        scope = self.config["scope"]
        self.assertEqual(scope["mode"], "isaac_only_kinematic_grasp_cycle")
        for field in (
            "ros_used",
            "watson_connected",
            "real_robot_commanded",
            "contact_physics_simulated",
            "physical_camera_or_depth_used",
        ):
            self.assertIs(scope[field], False)

        payload = self.config["payload_visual"]
        self.assertEqual(payload["attachment_mode"], "kinematic_visual_follow")
        clear_start = float(payload["clear_start_z_from_pinch_m"])
        specimen_near = float(payload["specimen_near_z_from_pinch_m"])
        clear_length = float(payload["clear_pin_length_m"])
        self.assertAlmostEqual(clear_length, 0.010, places=12)
        self.assertAlmostEqual(specimen_near - clear_start, 0.010, places=12)
        self.assertAlmostEqual((clear_start + specimen_near) / 2.0, 0.0, places=12)
        self.assertAlmostEqual(
            specimen_near,
            float(payload["pinch_to_specimen_m"]),
            places=12,
        )
        specimen_half_z = 0.5 * float(payload["specimen_scale_xyz_m"][2])
        self.assertAlmostEqual(
            float(payload["specimen_center_z_from_pinch_m"]) - specimen_half_z,
            specimen_near,
            places=12,
        )
        self.assertAlmostEqual(
            float(payload["pin_head_center_z_from_pinch_m"])
            + float(payload["pin_head_radius_m"]),
            clear_start,
            places=12,
        )

        metadata_path = resolve_arena_path(self.config["articulated_asset"]["tool_metadata"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        baseline = metadata["application_pin_baseline"]
        start_xyz = np.asarray(
            baseline["clear_section_start_xyz_from_2fg7_device_origin_m"],
            dtype=np.float64,
        )
        pinch_xyz = np.asarray(
            baseline["pinch_xyz_from_2fg7_device_origin_m"], dtype=np.float64
        )
        specimen_xyz = np.asarray(
            baseline["specimen_near_point_xyz_from_2fg7_device_origin_m"],
            dtype=np.float64,
        )
        self.assertAlmostEqual(float(np.linalg.norm(specimen_xyz - start_xyz)), 0.010)
        np.testing.assert_allclose(pinch_xyz, (start_xyz + specimen_xyz) / 2.0, atol=1.0e-12)

    def test_0_8mm_pin_fits_inside_the_1mm_closed_gap(self) -> None:
        payload = self.config["payload_visual"]
        gripper = self.config["gripper_motion"]
        pin_diameter = 2.0 * float(payload["pin_radius_m"])
        closed_gap = float(gripper["closed_gap_m"])
        self.assertAlmostEqual(pin_diameter, 0.0008, places=12)
        self.assertAlmostEqual(closed_gap, 0.001, places=12)
        self.assertLess(pin_diameter, closed_gap)
        self.assertAlmostEqual(closed_gap - pin_diameter, 0.0002, places=12)


class ArticulatedAssetInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.asset = cls.config["articulated_asset"]
        cls.expected_dofs = list(cls.asset["expected_dof_names"])

    def test_staged_articulated_urdf_topology_and_mimic_if_present(self) -> None:
        manifest_path = resolve_arena_path(self.asset["staged_manifest"])
        if not manifest_path.is_file():
            self.skipTest("Generated articulated staging manifest is not present")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["asset_mode"], "isaac_articulated")
        self.assertIs(manifest["xrdf"], None)
        self.assertIs(manifest["offline_only"], True)
        self.assertEqual(manifest["moving_joints"], self.expected_dofs)

        urdf_path = manifest_path.parent / "tm5s_with_2fg7.urdf"
        if not urdf_path.is_file():
            self.skipTest("Generated articulated staged URDF is not present")
        root = ET.parse(urdf_path).getroot()
        moving_joints = [
            joint
            for joint in root.findall("joint")
            if joint.get("type") not in {None, "fixed"}
        ]
        self.assertEqual([joint.get("name") for joint in moving_joints], self.expected_dofs)
        joint_by_name = {joint.get("name"): joint for joint in moving_joints}
        for arm_joint in ARM_JOINT_NAMES:
            self.assertEqual(joint_by_name[arm_joint].get("type"), "revolute")

        leader_name = self.config["gripper_motion"]["leader_joint"]
        follower_name = self.config["gripper_motion"]["mimic_joint"]
        leader = joint_by_name[leader_name]
        follower = joint_by_name[follower_name]
        self.assertEqual(leader.get("type"), "prismatic")
        self.assertEqual(follower.get("type"), "prismatic")
        self.assertEqual(vector(leader.find("axis"), "xyz"), (-1.0, 0.0, 0.0))
        self.assertEqual(vector(follower.find("axis"), "xyz"), (1.0, 0.0, 0.0))
        self.assertIsNone(leader.find("mimic"))
        mimic = follower.find("mimic")
        self.assertIsNotNone(mimic)
        self.assertEqual(mimic.get("joint"), leader_name)
        self.assertAlmostEqual(float(mimic.get("multiplier")), 1.0)

        for joint in (leader, follower):
            limit = joint.find("limit")
            self.assertAlmostEqual(float(limit.get("lower")), 0.0)
            self.assertAlmostEqual(float(limit.get("upper")), 0.019)
            self.assertAlmostEqual(float(limit.get("velocity")), 0.225)
        self.assertIsNone(root.find("ros2_control"))
        self.assertFalse(root.findall("transmission"))

    def test_import_report_hashes_and_eight_dof_contract_if_present(self) -> None:
        report_path = resolve_arena_path(self.asset["import_report"])
        usd_path = resolve_arena_path(self.asset["usd"])
        if not report_path.is_file() or not usd_path.is_file():
            self.skipTest("Generated articulated Isaac report/USD is not present")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["validation_profile"], self.asset["profile"])
        self.assertEqual(report["dof_count"], 8)
        self.assertEqual(report["moving_joints"], self.expected_dofs)
        self.assertEqual(report["physx_dof_names"], self.expected_dofs)
        self.assertIs(report["real_robot_commanded"], False)
        self.assertEqual(sha256_file(usd_path), report["output_usd_sha256"])

        source_urdf = Path(report["source_urdf"]).resolve()
        self.assertTrue(source_urdf.is_relative_to(ARENA_DIR.resolve()))
        self.assertEqual(sha256_file(source_urdf), report["source_urdf_sha256"])
        joint_types = report["source_urdf_topology"]["joint_types"]
        self.assertEqual(joint_types[self.config["gripper_motion"]["leader_joint"]], "prismatic")
        self.assertEqual(joint_types[self.config["gripper_motion"]["mimic_joint"]], "prismatic")
        drive_validation = report["drive_validation"]
        self.assertIs(drive_validation[self.config["gripper_motion"]["leader_joint"]]["independent_drive"], True)
        self.assertIs(drive_validation[self.config["gripper_motion"]["mimic_joint"]]["mimic_controlled"], True)


if __name__ == "__main__":
    unittest.main()
