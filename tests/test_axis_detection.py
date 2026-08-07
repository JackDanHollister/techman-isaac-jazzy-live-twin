from __future__ import annotations

import unittest

import numpy as np

from pin_axis_3d_sim.alignment import make_targets
from pin_axis_3d_sim.detection import detect_pin_axes
from pin_axis_3d_sim.evaluation import evaluate_detections
from pin_axis_3d_sim.synthetic_scene import SceneConfig, generate_scene


class PinAxisDetectionTests(unittest.TestCase):
    def test_synthetic_scene_recovers_most_pin_axes(self):
        scene = generate_scene(
            seed=21,
            pin_count=10,
            config=SceneConfig(scanner_noise_std=0.00035, max_pin_tilt_deg=10.0),
        )
        result = detect_pin_axes(scene.points, seed=99)
        evaluation = evaluate_detections(scene.truth, result.detections)

        self.assertGreaterEqual(evaluation["matched_count"], 8)
        self.assertLessEqual(evaluation["false_positive_count"], 2)
        self.assertIsNotNone(evaluation["mean_angular_error_deg"])
        self.assertLess(evaluation["mean_angular_error_deg"], 4.0)
        self.assertIsNotNone(evaluation["mean_axis_lateral_error_m"])
        self.assertLess(evaluation["mean_axis_lateral_error_m"], 0.0045)

    def test_gripper_targets_align_tool_z_opposite_pin_axis(self):
        scene = generate_scene(seed=8, pin_count=6)
        result = detect_pin_axes(scene.points, seed=109)
        self.assertGreater(len(result.detections), 0)

        targets = make_targets(result.detections, result.plane)
        for target in targets:
            dot = float(np.dot(target.tool_z_axis_robot, target.pin_axis_up))
            self.assertAlmostEqual(dot, -1.0, places=5)
            self.assertAlmostEqual(float(np.linalg.norm(target.quaternion_xyzw)), 1.0, places=5)
            self.assertGreater(
                float(np.dot(target.pregrasp_position - target.grasp_position, target.pin_axis_up)),
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
