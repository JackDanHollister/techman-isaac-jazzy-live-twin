#!/usr/bin/env python3
"""Focused tests for the local official OnRobot CAD visual pipeline."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ARENA_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ARENA_DIR / "scripts/prepare_onrobot_official_visuals.py"
TOOL_MODEL_PATH = ARENA_DIR / "config/onrobot_2fg7_tool_model.json"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_onrobot_official_visuals", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OfficialOnRobotVisualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()
        _, cls.visual_spec = cls.module.load_visual_spec(TOOL_MODEL_PATH)

    def test_qc_registration_maps_axial_negative_y_to_positive_z(self) -> None:
        registration = self.visual_spec["assets"]["qc_robot_side"]["registration"]
        self.assertEqual(
            self.module.apply_registration((0.001, -0.0136, 0.002), registration),
            (0.001, 0.002, 0.0136),
        )

    def test_explicit_step_arguments_have_python_safe_destinations(self) -> None:
        args = self.module.build_parser().parse_args(
            ["--2fg7-step", "/tmp/2fg7.step", "--qc-step", "/tmp/qc.step"]
        )
        self.assertEqual(args.two_fg7_step, Path("/tmp/2fg7.step"))
        self.assertEqual(args.qc_step, Path("/tmp/qc.step"))

    def test_2fg7_registration_places_mating_plane_at_link_origin(self) -> None:
        registration = self.visual_spec["assets"]["two_fg7"]["registration"]
        origin = tuple(registration["source_origin_xyz_m"])
        registered_origin = self.module.apply_registration(origin, registration)
        self.assertTrue(all(abs(component) < 1.0e-15 for component in registered_origin))
        source_forward = (origin[0], origin[1] + 0.125, origin[2])
        registered_forward = self.module.apply_registration(source_forward, registration)
        self.assertAlmostEqual(registered_forward[2], 0.125)

    def test_expected_bounds_accept_converter_roundoff_but_reject_scale_drift(self) -> None:
        registration = self.visual_spec["assets"]["two_fg7"]["registration"]
        expected = registration["expected_bounds_m"]
        self.module.validate_bounds(
            tuple(value + 1.0e-7 for value in expected["minimum"]),
            tuple(value - 1.0e-7 for value in expected["maximum"]),
            registration,
            "two_fg7",
        )
        with self.assertRaisesRegex(ValueError, "registered maximum axis 2"):
            self.module.validate_bounds(
                tuple(expected["minimum"]),
                (expected["maximum"][0], expected["maximum"][1], 0.14315 * 1000.0),
                registration,
                "two_fg7",
            )


if __name__ == "__main__":
    unittest.main()
