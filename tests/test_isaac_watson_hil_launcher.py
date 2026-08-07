from __future__ import annotations

from pathlib import Path
import unittest


ARENA_DIR = Path(__file__).resolve().parents[1]
LAUNCHER = ARENA_DIR / "scripts/run_isaac_watson_hil.sh"


class IsaacWatsonHilLauncherTests(unittest.TestCase):
    def test_launcher_uses_isolated_isaac_jazzy_runtime(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("isaac-sim-6.0", source)
        self.assertIn("isaacsim.ros2.core/jazzy", source)
        self.assertIn('"isaacsim": "6.0.1.0"', source)
        self.assertIn('"isaacsim-core": "6.0.1.0"', source)
        self.assertIn('"isaacsim-ros2": "6.0.1.0"', source)
        self.assertIn("ROS_DOMAIN_ID=219", source)
        self.assertIn("ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST", source)
        self.assertIn("RMW_IMPLEMENTATION=rmw_fastrtps_cpp", source)

    def test_launcher_defaults_to_preview_and_timestamped_report(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("launch_args+=(--mode preview)", source)
        self.assertIn("${STAMP}_watson_hil.json", source)
        self.assertIn('launch_args+=(--report "$DEFAULT_REPORT")', source)
        self.assertIn('launch_args+=("$@")', source)
        self.assertIn('"$HIL_SCRIPT" "${launch_args[@]}"', source)

    def test_launcher_validates_local_inputs_without_robot_preflight(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        for required in (
            "run_isaac_watson_hil.py",
            "isaac_multi_pin_verticalization.yaml",
            "reference/seven_pin/isaac",
            "EULA_ACCEPTED",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "run_watson_guarded_demo.sh",
            "ros2 param",
            "ping ",
            "192.0.2.23",
            "/sys/class/net/",
        ):
            self.assertNotIn(forbidden, source)

    def test_gui_survives_toolbar_stop_invalidating_the_physics_view(self) -> None:
        source = (
            ARENA_DIR / "scripts/run_isaac_watson_hil.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'getattr(world, "physics_sim_view", None) is None',
            source,
        )
        self.assertIn("display_applied = apply_display()", source)
        self.assertIn("simulation_app.update()", source)


if __name__ == "__main__":
    unittest.main()
