from __future__ import annotations

import copy
from pathlib import Path
import unittest

from pin_axis_3d_sim.controller_commissioning import (
    REQUIRED_PHYSICAL_ITEMS,
    load_offline_commissioning_manifest,
    offline_commissioning_manifest_failures,
    parse_offline_commissioning_manifest,
    validate_offline_commissioning_manifest,
)


ARENA_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = (
    ARENA_DIR / "config" / "watson_controller_commissioning_offline_template.json"
)
RUNNER_PATH = ARENA_DIR / "scripts" / "run_watson_guarded_demo.py"
MODULE_PATH = (
    ARENA_DIR / "pin_axis_3d_sim" / "controller_commissioning.py"
)


class OfflineCommissioningTemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_offline_commissioning_manifest(TEMPLATE_PATH)

    def test_shipped_template_is_valid_but_never_promotes(self) -> None:
        result = validate_offline_commissioning_manifest(self.manifest)
        self.assertTrue(result["offline_template_integrity_passed"])
        self.assertFalse(result["commissioning_ready"])
        self.assertFalse(result["controller_application_authorized"])
        self.assertFalse(result["applied_to_controller"])
        self.assertFalse(result["motion_commanded"])
        self.assertFalse(result["promotion_passed"])
        self.assertEqual(
            result["required_physical_items"], list(REQUIRED_PHYSICAL_ITEMS)
        )

    def test_vendor_nominal_dry_stack_values_are_exact_candidates(self) -> None:
        candidate = self.manifest["safe_precomputations"]
        self.assertEqual(
            candidate["nominal_flange_to_2fg7_vendor_tcp_xyz_m"],
            [0.0, 0.0, 0.1386],
        )
        self.assertEqual(candidate["nominal_dry_mass_kg"], 1.2)
        self.assertEqual(
            candidate["nominal_dry_cog_xyz_from_robot_flange_m"],
            [0.0, 0.0, 0.06252],
        )
        expected_cog_z = (0.06 * 0.004 + 1.14 * (0.0136 + 0.052)) / 1.2
        self.assertAlmostEqual(
            candidate["nominal_dry_cog_xyz_from_robot_flange_m"][2],
            expected_cog_z,
            places=12,
        )
        self.assertIn("not_a_calibrated_pinch_tcp", candidate["status"])

    def test_confirmed_inspection_is_recorded_and_remaining_values_are_unknown(self) -> None:
        required = self.manifest["required_physical_measurements"]
        revision = required["quick_changer_revision"]
        self.assertEqual(revision["item_number"], "109498")
        self.assertEqual(revision["value"], "QC-R v3")
        self.assertEqual(revision["ip_classification"], "IP67")
        self.assertFalse(required["adapter_k_presence"]["value"])
        self.assertIsNone(required["quick_changer_keyed_yaw"]["value_rad"])
        clock = required["quick_changer_keyed_yaw"]["clock_orientation"]
        self.assertEqual(clock["twelve_oclock_reference"], "tm_eih_camera")
        self.assertEqual(clock["quick_release_control"], "12_oclock_facing_tm_eih_camera")
        self.assertEqual(clock["cable_wrap"], "3_oclock")
        self.assertEqual(clock["cable_end_socket"], "9_oclock")
        blockers = self.manifest["promotion_blockers"]
        self.assertNotIn(
            "installed Quick Changer revision has not been identified", blockers
        )
        self.assertNotIn(
            "Adapter K presence has not been physically reverified", blockers
        )
        self.assertIn(
            "approved v2-labelled working CAD has not been registered to the confirmed physical clock orientation",
            blockers,
        )
        self.assertIsNone(
            required["calibrated_pinch_tcp"]["xyz_from_robot_flange_m"]
        )
        self.assertIsNone(
            required["calibrated_pinch_tcp"]["rpy_from_robot_flange_rad"]
        )
        for field in ("mass_kg", "cog_xyz_from_tcp_m", "principal_moi_kg_m2"):
            self.assertIsNone(required["workpiece_payload"][field])
        self.assertIsNone(
            required["physical_dry_tool_principal_moi"][
                "principal_moi_kg_m2"
            ]
        )

    def test_rejects_any_safety_flag_that_implies_controller_activity(self) -> None:
        unsafe_values = {
            "offline_only": False,
            "controller_contacted": True,
            "controller_write_authorized": True,
            "controller_write_performed": True,
            "controller_application_authorized": True,
            "motion_commanded": True,
            "applied_to_controller": True,
        }
        for field, unsafe_value in unsafe_values.items():
            with self.subTest(field=field):
                altered = copy.deepcopy(self.manifest)
                altered["safety"][field] = unsafe_value
                failures = offline_commissioning_manifest_failures(altered)
                self.assertTrue(any(field in failure for failure in failures))

    def test_rejects_filled_controller_placeholders_or_changed_physical_record(self) -> None:
        alterations = (
            ("controller_entry_placeholders", "tcp_name", "WatsonQC2FG7"),
            ("controller_entry_placeholders", "mass_kg", 1.2),
            ("required_physical_measurements", "quick_changer_revision", {}),
        )
        for section, field, value in alterations:
            with self.subTest(section=section, field=field):
                altered = copy.deepcopy(self.manifest)
                altered[section][field] = value
                self.assertTrue(offline_commissioning_manifest_failures(altered))

        altered = copy.deepcopy(self.manifest)
        altered["required_physical_measurements"][
            "physical_dry_tool_principal_moi"
        ]["principal_moi_kg_m2"] = [0.001, 0.001, 0.001]
        self.assertTrue(offline_commissioning_manifest_failures(altered))

    def test_rejects_changed_candidates_extra_keys_and_nonfinite_json(self) -> None:
        altered = copy.deepcopy(self.manifest)
        altered["safe_precomputations"]["nominal_dry_mass_kg"] = 1.21
        self.assertTrue(offline_commissioning_manifest_failures(altered))

        altered = copy.deepcopy(self.manifest)
        altered["controller_command"] = "SetTcp(...)"
        self.assertTrue(offline_commissioning_manifest_failures(altered))

        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            parse_offline_commissioning_manifest('{"x": 1, "x": 2}')
        with self.assertRaisesRegex(ValueError, "non-standard JSON number"):
            parse_offline_commissioning_manifest('{"x": NaN}')

    def test_artifact_cannot_satisfy_the_schema_8_live_guard(self) -> None:
        self.assertNotIn("schema_version", self.manifest)
        self.assertNotIn("mode", self.manifest)
        self.assertNotIn("controller_tool_audit", self.manifest)
        self.assertNotIn("controller_tool_settings_promotion_passed", self.manifest)
        relationship = self.manifest["guard_relationship"]
        self.assertEqual(relationship["observed_watson_report_schema_version"], 8)
        self.assertFalse(relationship["accepted_as_live_controller_tool_audit"])
        self.assertFalse(relationship["can_bypass_live_ask_item_query"])
        self.assertFalse(relationship["promotion_passed"])

        runner_source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("REPORT_SCHEMA_VERSION = 8", runner_source)
        self.assertIn(
            "controller_tool_audit = guard.read_controller_tool_settings(",
            runner_source,
        )
        self.assertIn(
            'if args.mode == "execute" and not controller_tool_audit["promotion_passed"]:',
            runner_source,
        )

    def test_validator_source_has_no_controller_network_motion_or_write_path(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        for forbidden in (
            "import rclpy",
            "import socket",
            "import subprocess",
            "import requests",
            "import urllib",
            ".write_text(",
            ".write_bytes(",
            "SetTcp",
            "SetLoad",
            "send_goal",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
