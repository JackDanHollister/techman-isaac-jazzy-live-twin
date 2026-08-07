from __future__ import annotations

import importlib.util
import json
import math
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:  # Keep the original lightweight demo test suite runnable.
    yaml = None

try:
    import cumotion
except ImportError:
    cumotion = None


ARENA_DIR = Path(__file__).resolve().parents[1]


def load_isaac_importer_module():
    importer_path = ARENA_DIR / "scripts/import_tm5s_isaac_sim.py"
    spec = importlib.util.spec_from_file_location("tm5s_isaac_importer", importer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load importer module: {importer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tool_builder_module():
    builder_path = ARENA_DIR / "scripts/build_tm5s_2fg7_urdf.py"
    spec = importlib.util.spec_from_file_location("tm5s_2fg7_builder", builder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load tool builder module: {builder_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IsaacImportSafetyTests(unittest.TestCase):
    def test_generated_output_cleanup_is_project_contained(self) -> None:
        importer = load_isaac_importer_module()
        allowed = ARENA_DIR / "generated/isaac/6.0.1-test"
        self.assertEqual(importer.validate_generated_output_dir(allowed), allowed.resolve())
        with self.assertRaisesRegex(ValueError, "must stay within"):
            importer.validate_generated_output_dir(Path("/outside/project"))
        with self.assertRaisesRegex(ValueError, "not the root itself"):
            importer.validate_generated_output_dir(ARENA_DIR / "generated/isaac")

    def test_importer_defaults_to_strict_legacy_profile(self) -> None:
        importer = load_isaac_importer_module()
        args = importer.build_parser().parse_args([])
        self.assertEqual(args.validation_profile, "legacy_fixed_2fg7")

    def test_reviewed_import_profiles_match_exact_source_topology(self) -> None:
        importer = load_isaac_importer_module()
        sources = {
            "legacy_fixed_2fg7": (
                ARENA_DIR / "generated/cumotion/tm5s_with_2fg7.urdf"
            ),
            "watson_qc_nominal": (
                ARENA_DIR
                / "generated/tool_profiles/watson_qc_nominal/tm5s_with_2fg7.urdf"
            ),
        }
        if any(not source.is_file() for source in sources.values()):
            self.skipTest("Generated reviewed-profile URDFs are not present")
        for profile_name, source in sources.items():
            with self.subTest(profile=profile_name):
                topology = importer.validate_source_urdf_topology(
                    source,
                    importer.VALIDATION_PROFILES[profile_name],
                )
                self.assertEqual(topology["profile"], profile_name)
                self.assertEqual(topology["joint_types"]["joint_6"], "revolute")
                self.assertEqual(topology["joint_types"]["base_fixed_joint"], "fixed")

    def test_profile_mismatch_is_rejected_before_import(self) -> None:
        importer = load_isaac_importer_module()
        watson_source = (
            ARENA_DIR
            / "generated/tool_profiles/watson_qc_nominal/tm5s_with_2fg7.urdf"
        )
        if not watson_source.is_file():
            self.skipTest("Generated Watson-profile URDF is not present")
        with self.assertRaisesRegex(ValueError, "does not match validation profile"):
            importer.validate_source_urdf_topology(
                watson_source,
                importer.VALIDATION_PROFILES["legacy_fixed_2fg7"],
            )

    def test_importer_rejects_unstaged_package_meshes(self) -> None:
        importer = load_isaac_importer_module()
        unstaged = (
            ARENA_DIR
            / "generated/tool_profiles/watson_qc_nominal/tm5s_with_2fg7.urdf"
        )
        if not unstaged.is_file():
            self.skipTest("Generated unstaged Watson-profile URDF is not present")
        with self.assertRaisesRegex(ValueError, "staged cuMotion URDF"):
            importer.validate_source_urdf_meshes(unstaged)

    def test_importer_accepts_fully_staged_profile_meshes(self) -> None:
        importer = load_isaac_importer_module()
        staged = (
            ARENA_DIR
            / "generated/tool_profiles/watson_qc_nominal/cumotion/tm5s_with_2fg7.urdf"
        )
        if not staged.is_file():
            self.skipTest("Generated staged Watson-profile URDF is not present")
        evidence = importer.validate_source_urdf_meshes(staged)
        self.assertTrue(evidence["all_meshes_resolved"])
        self.assertGreaterEqual(evidence["mesh_reference_count"], 20)


class IsaacVisualDemoConfigTests(unittest.TestCase):
    @staticmethod
    def load_visual_demo_module():
        script_path = ARENA_DIR / "scripts/run_isaac_visual_demo.py"
        spec = importlib.util.spec_from_file_location("tm5s_isaac_visual_demo", script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load visual demo module: {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_visual_sequence_retraces_the_audited_path(self) -> None:
        module = self.load_visual_demo_module()
        config = module.load_visual_demo_config(
            ARENA_DIR / "config/isaac_visual_demo.yaml"
        )
        forward = config["forward_sequence"]
        self.assertEqual(
            config["motion_sequence"], [*forward, *reversed(forward[:-1])]
        )
        self.assertEqual(
            config["computed_forward_raw_path_float64_sha256"],
            "e5d75dde824a1542b0928bdb1e406318f036d1353676c904d0e86b6305f16012",
        )

    def test_visual_source_evidence_matches_config(self) -> None:
        module = self.load_visual_demo_module()
        config = module.load_visual_demo_config(
            ARENA_DIR / "config/isaac_visual_demo.yaml"
        )
        try:
            evidence = module.validate_benchmark_source(config)
        except FileNotFoundError:
            self.skipTest("Generated cuMotion benchmark evidence is not present")
        self.assertEqual(evidence["source_case"], "static_obstacle_detour")
        self.assertEqual(evidence["accepted_paths"], evidence["trials"])
        self.assertEqual(evidence["triangle_aabb_intersection_pairs"], 0)
        self.assertGreater(evidence["minimum_sampled_mesh_vertex_clearance_m"], 0.0)

    def test_visual_interpolation_starts_and_ends_exactly(self) -> None:
        module = self.load_visual_demo_module()
        config = module.load_visual_demo_config(
            ARENA_DIR / "config/isaac_visual_demo.yaml"
        )
        transition = config["transition_seconds"]
        hold = config["hold_seconds"]
        start, _, _ = module.command_for_time(config, 0.0, transition, hold)
        first_target, _, _ = module.command_for_time(
            config, transition, transition, hold
        )
        np.testing.assert_allclose(
            start, config["waypoints"][config["forward_sequence"][0]]
        )
        np.testing.assert_allclose(
            first_target,
            config["waypoints"][config["forward_sequence"][1]],
        )
        self.assertEqual(module.smoothstep_cosine(0.0), 0.0)
        self.assertTrue(math.isclose(module.smoothstep_cosine(1.0), 1.0))

    def test_visual_interpolation_respects_validated_sampling_step(self) -> None:
        module = self.load_visual_demo_module()
        config = module.load_visual_demo_config(
            ARENA_DIR / "config/isaac_visual_demo.yaml"
        )
        bound = module.maximum_command_step_bound(
            config, config["transition_seconds"]
        )
        self.assertLessEqual(bound, module.MAX_COMMAND_STEP_RADIANS)


class SyntheticPickArtifactTests(unittest.TestCase):
    @staticmethod
    def load_synthetic_viewer_module():
        script_path = ARENA_DIR / "scripts/run_isaac_synthetic_pick.py"
        spec = importlib.util.spec_from_file_location("tm5s_synthetic_pick_viewer", script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load synthetic viewer module: {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @unittest.skipIf(yaml is None, "PyYAML is required for the synthetic task config")
    def test_task_config_keeps_tool_assumptions_explicit(self) -> None:
        with (ARENA_DIR / "config/synthetic_pick_task.yaml").open(
            "r", encoding="utf-8"
        ) as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(config["format_version"], 1)
        self.assertEqual(config["frame_id"], "base")
        self.assertEqual(config["robot"]["planning_tool_frame"], "flange")
        self.assertFalse(config["tool_model"]["quick_changer_robot_side_modeled"])
        self.assertAlmostEqual(
            config["tool_model"]["pin_grasp_tcp_z_from_2fg7_origin_m"], 0.16225
        )
        self.assertAlmostEqual(
            config["tool_model"]["onrobot_nominal_device_tcp_z_m"], 0.125
        )
        self.assertEqual(
            config["isaac_execution"]["status"],
            "simulation_only_uncalibrated_acceleration_drive_smoke",
        )
        self.assertFalse(config["isaac_execution"]["save_to_usd"])

    def test_validated_plan_replays_continuously_if_present(self) -> None:
        plan_path = (
            ARENA_DIR
            / "outputs/synthetic_pick_seed_1407/synthetic_pick_plan.json"
        )
        usd_path = (
            ARENA_DIR
            / "generated/isaac/6.0.1/tm5s_with_2fg7/tm5s_with_2fg7.usda"
        )
        import_report = ARENA_DIR / "outputs/isaac_sim/6.0.1/import_report.json"
        if not plan_path.is_file() or not usd_path.is_file() or not import_report.is_file():
            self.skipTest("Validated synthetic pick and Isaac artifacts are not present")
        module = self.load_synthetic_viewer_module()
        plan, _ = module.load_and_validate_plan(plan_path, usd_path, import_report)
        commands = module.build_command_cycle(plan)
        positions = np.asarray(
            [command["joint_positions"] for command in commands], dtype=np.float64
        )
        self.assertGreater(len(commands), 1000)
        self.assertLessEqual(
            float(np.max(np.abs(np.diff(positions, axis=0)))),
            float(plan["maximum_control_step_rad"]) + 1.0e-12,
        )
        np.testing.assert_allclose(positions[0], positions[-1], atol=1.0e-12)
        self.assertEqual(plan["accepted_candidate_count"], plan["candidate_count"])
        self.assertEqual(plan["selected"]["detection_id"], 3)
        self.assertGreater(
            plan["selected"]["minimum_sampled_sphere_clearance_m"],
            plan["required_sampled_sphere_clearance_m"],
        )


class ToolModelProfileTests(unittest.TestCase):
    @staticmethod
    def build_profile(profile: str, *, finger_joints: str = "fixed"):
        module = load_tool_builder_module()
        args = module.build_parser().parse_args(
            ["--tool-profile", profile, "--finger-joints", finger_joints]
        )
        model = module.load_tool_model(args)
        root = ET.Element("robot", {"name": "test"})
        ET.SubElement(root, "link", {"name": "flange"})
        module.add_gripper(root, args, model)
        return module, args, model, root

    def test_profiles_keep_sources_confirmed_hardware_and_unknowns_explicit(self) -> None:
        data = json.loads(
            (ARENA_DIR / "config/onrobot_2fg7_tool_model.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(data["format_version"], 1)
        self.assertEqual(data["shared"]["finger_configuration"], "inwards")
        self.assertFalse(data["shared"]["adapter_k"]["present"])
        self.assertFalse(
            data["shared"]["adapter_k"]["physical_reverification_required"]
        )
        self.assertEqual(
            data["profiles"]["watson_qc_nominal"]["qc_variant"],
            "onrobot_109498_qc_r_v3",
        )
        quick_changer = data["shared"]["standard_quick_changer_robot_side"]
        self.assertEqual(quick_changer["assembly_type"], "single_standard_robot_side")
        self.assertEqual(quick_changer["item_number"], "109498")
        self.assertEqual(quick_changer["version"], "QC-R v3")
        self.assertEqual(quick_changer["ip_classification"], "IP67")
        clock = quick_changer["physical_clock_orientation"]
        self.assertEqual(clock["twelve_oclock_reference"], "tm_eih_camera")
        self.assertEqual(clock["quick_release_control"], "12_oclock_facing_tm_eih_camera")
        self.assertEqual(clock["cable_wrap"], "3_oclock")
        self.assertEqual(clock["cable_end_socket"], "9_oclock")
        self.assertAlmostEqual(quick_changer["body_diameter_m"], 0.071)
        self.assertAlmostEqual(quick_changer["maximum_width_m"], 0.08425)
        self.assertAlmostEqual(
            quick_changer["maximum_radial_reach_from_axis_m"], 0.04875
        )
        self.assertEqual(quick_changer["cog_xyz_m"], [0.0, 0.0, 0.004])
        commissioning = data["shared"]["controller_commissioning"]
        self.assertFalse(commissioning["promotion_gate"]["passed"])
        self.assertEqual(
            commissioning["latest_read_only_observation"]["active_tcp_name"],
            "RobotEndFlange",
        )
        profile = data["profiles"]["watson_qc_nominal"]
        self.assertIn("10mm", profile["pin_grasp_status"])
        self.assertIn("not_physically_calibrated", profile["pin_grasp_status"])
        contact = data["shared"]["frames_from_cad_origin"][
            "inward_finger_contact_face"
        ]
        self.assertEqual(contact["axial_bounds_z_m"], [0.11975, 0.15975])
        self.assertEqual(contact["lateral_center_xy_m"], [0.0, 0.0])
        self.assertAlmostEqual(contact["axial_center_z_m"], 0.13975)
        self.assertEqual(
            data["shared"]["geometry_scope"]["excluded"],
            ["cables", "cable_routing", "cable_wrap_geometry"],
        )
        baseline = profile["application_pin_baseline"]
        self.assertEqual(baseline["lateral_alignment"], "centred_between_inward_fingers")
        self.assertAlmostEqual(baseline["clear_pin_length_before_specimen_m"], 0.01)
        self.assertAlmostEqual(baseline["pinch_to_specimen_m"], 0.005)
        self.assertEqual(
            baseline["pinch_xyz_from_2fg7_device_origin_m"],
            profile["pin_grasp_tcp_xyz_m"],
        )

    def test_official_cad_visual_provenance_and_registration_are_pinned(self) -> None:
        data = json.loads(
            (ARENA_DIR / "config/onrobot_2fg7_tool_model.json").read_text(
                encoding="utf-8"
            )
        )
        visuals = data["shared"]["official_cad_visuals"]
        self.assertIn("no_redistribution_permission", visuals["distribution_policy"])
        self.assertEqual(visuals["output_units"], "metres")
        self.assertIn("reference_only_not_consumed", visuals["integration_status"])
        availability = visuals["installed_qc_cad_availability"]
        self.assertEqual(availability["item_number"], "109498")
        self.assertIn("exact_installed_v3_mesh_unavailable", availability["status"])
        self.assertIn(
            "user_approved_v2_labelled_working_approximation",
            availability["status"],
        )
        self.assertIn("not_vendor_confirmed_exact", availability["status"])
        self.assertEqual(
            visuals["assets"]["two_fg7"]["sha256"],
            "4aa791f431604588d854fa8f591002dbaf48fd7448c9eebe7b83533397d8aacd",
        )
        self.assertEqual(
            visuals["assets"]["qc_robot_side"]["sha256"],
            "ac3cb44b90b27b172423400def5150a9af8b6e9338baf358f413118cbf5f38db",
        )
        self.assertIn(
            "v2_labelled_reference_not_item_tagged_or_confirmed",
            visuals["assets"]["qc_robot_side"]["registration"]["status"],
        )
        two_fg7_bounds = visuals["assets"]["two_fg7"]["registration"][
            "expected_bounds_m"
        ]
        self.assertEqual(two_fg7_bounds["minimum"][2], -0.0132)
        self.assertEqual(two_fg7_bounds["maximum"][2], 0.14315)
        fixed_pose = visuals["assets"]["two_fg7"]["registration"][
            "fixed_finger_pose"
        ]
        self.assertEqual(fixed_pose["orientation"], "outwards")
        self.assertEqual(fixed_pose["geometric_inner_gap_m"], 0.0254)
        self.assertEqual(fixed_pose["watson_archived_orientation"], "inwards")
        self.assertIn("does_not_match", fixed_pose["status"])

    def test_default_outputs_are_profile_scoped(self) -> None:
        module = load_tool_builder_module()
        args = module.build_parser().parse_args([])
        model = module.load_tool_model(args)
        output, metadata = module.resolve_output_paths(args, model)
        expected_dir = ARENA_DIR / "generated/tool_profiles/legacy_cad_dry_run"
        self.assertEqual(output, expected_dir / "tm5s_with_2fg7.urdf")
        self.assertEqual(metadata, expected_dir / "tm5s_with_2fg7_metadata.json")

    def test_profile_wrappers_do_not_overwrite_pinned_shared_assets(self) -> None:
        for name in (
            "setup_cumotion_benchmark.sh",
            "run_cumotion_benchmark.sh",
            "launch_tm5s_2fg7_rviz.sh",
            "launch_tm5s_2fg7_moveit_demo.sh",
            "play_random_pin_alignment_demo.sh",
        ):
            with self.subTest(script=name):
                text = (ARENA_DIR / "scripts" / name).read_text(encoding="utf-8")
                self.assertIn("generated/tool_profiles", text)
                self.assertNotIn(
                    'URDF_PATH="$ARENA_DIR/generated/tm5s_with_2fg7.urdf"',
                    text,
                )

    def test_watson_profile_applies_qc_offset_exactly_once(self) -> None:
        module, args, model, root = self.build_profile("watson_qc_nominal")
        qc_joint = root.find("./joint[@name='onrobot_qc_robot_side_to_2fg7_origin']/origin")
        self.assertIsNotNone(qc_joint)
        self.assertAlmostEqual(float(qc_joint.get("xyz").split()[2]), 0.0136)
        metadata = module.build_metadata(
            args,
            model,
            module.SourceRobot(ET.ElementTree(root), "unit-test", None),
        )
        self.assertEqual(metadata["frame_xyz_from_flange_m"]["onrobot_nominal_tcp"], [0.0, 0.0, 0.1386])
        self.assertEqual(metadata["frame_xyz_from_flange_m"]["finger_tip_plane"], [0.0, 0.0, 0.17585])
        self.assertEqual(metadata["frame_xyz_from_flange_m"]["pin_grasp_tcp"], [0.0, 0.0, 0.17085])
        baseline = metadata["application_pin_baseline"]
        self.assertEqual(
            baseline["clear_section_start_xyz_from_2fg7_device_origin_m"],
            [0.0, 0.0, 0.15225],
        )
        self.assertEqual(
            baseline["specimen_near_point_xyz_from_2fg7_device_origin_m"],
            [0.0, 0.0, 0.16225],
        )
        self.assertAlmostEqual(metadata["dynamics"]["total_mass_kg"], 1.2)
        self.assertAlmostEqual(
            metadata["dynamics"]["aggregate_cog_xyz_from_mount_m"][2],
            0.06252,
        )
        inertia = metadata["dynamics"][
            "aggregate_inertia_at_cog_tool_axes_kg_m2"
        ]
        self.assertAlmostEqual(inertia["ixx"], 0.002693018704375)
        self.assertAlmostEqual(inertia["iyy"], 0.002983623704375)
        self.assertAlmostEqual(inertia["izz"], 0.00130163046875)

        qc_link = root.find("./link[@name='onrobot_qc_robot_side_link']")
        self.assertIsNotNone(qc_link)
        visual = qc_link.find("./visual/geometry/cylinder")
        collision = qc_link.find("./collision/geometry/cylinder")
        self.assertAlmostEqual(float(visual.get("radius")), 0.0355)
        self.assertAlmostEqual(float(collision.get("radius")), 0.04875)
        self.assertAlmostEqual(float(visual.get("length")), 0.0161)

    def test_legacy_and_watson_profiles_differ_by_qc_stack(self) -> None:
        results = {}
        for profile in ("legacy_cad_dry_run", "watson_qc_nominal"):
            module, args, model, root = self.build_profile(profile)
            results[profile] = module.build_metadata(
                args,
                model,
                module.SourceRobot(ET.ElementTree(root), "unit-test", None),
            )
        legacy_z = results["legacy_cad_dry_run"]["frame_xyz_from_flange_m"]["onrobot_nominal_tcp"][2]
        watson_z = results["watson_qc_nominal"]["frame_xyz_from_flange_m"]["onrobot_nominal_tcp"][2]
        self.assertAlmostEqual(watson_z - legacy_z, 0.0136)

    def test_tool_frames_and_proxy_mass_are_explicit(self) -> None:
        _, _, _, root = self.build_profile("watson_qc_nominal")
        link_names = {link.get("name") for link in root.findall("link")}
        self.assertTrue(
            {
                "onrobot_2fg7_origin",
                "onrobot_nominal_tcp",
                "finger_tip_plane",
                "pin_grasp_tcp",
                "gripper_tcp",
                "onrobot_qc_robot_side_link",
            }.issubset(link_names)
        )
        fingertip_parent = root.find(
            "./joint[@name='onrobot_2fg7_base_link_to_finger_tip_plane']/parent"
        )
        self.assertEqual(fingertip_parent.get("link"), "onrobot_2fg7_base_link")
        masses = [float(mass.get("value")) for mass in root.findall(".//inertial/mass")]
        self.assertAlmostEqual(sum(masses), 1.2, places=9)
        for inertia in root.findall(".//inertial/inertia"):
            diagonal = [float(inertia.get(axis)) for axis in ("ixx", "iyy", "izz")]
            self.assertTrue(all(value > 0.0 for value in diagonal))
            self.assertLessEqual(diagonal[0], diagonal[1] + diagonal[2])
            self.assertLessEqual(diagonal[1], diagonal[0] + diagonal[2])
            self.assertLessEqual(diagonal[2], diagonal[0] + diagonal[1])

    def test_cad_registration_moves_only_the_cad_tip_frame(self) -> None:
        module = load_tool_builder_module()
        args = module.build_parser().parse_args(
            [
                "--tool-profile",
                "watson_qc_nominal",
                "--cad-origin-xyz",
                "0",
                "0",
                "-0.01",
            ]
        )
        model = module.load_tool_model(args)
        root = ET.Element("robot", {"name": "test"})
        ET.SubElement(root, "link", {"name": "flange"})
        module.add_gripper(root, args, model)
        metadata = module.build_metadata(
            args,
            model,
            module.SourceRobot(ET.ElementTree(root), "unit-test", None),
        )
        self.assertEqual(
            metadata["frame_xyz_from_flange_m"]["onrobot_nominal_tcp"],
            [0.0, 0.0, 0.1386],
        )
        self.assertAlmostEqual(
            metadata["frame_xyz_from_flange_m"]["finger_tip_plane"][2],
            0.16585,
        )

    def test_prismatic_fingers_use_relative_speed_semantics(self) -> None:
        _, _, model, root = self.build_profile(
            "watson_qc_nominal", finger_joints="prismatic"
        )
        motion = model.shared["finger_motion"]
        self.assertEqual(motion["maximum_relative_gap_speed_m_s"], 0.45)
        self.assertEqual(motion["maximum_per_finger_speed_m_s"], 0.225)
        self.assertEqual(
            motion["recommended_initial_simulation_per_finger_speed_m_s"],
            0.0225,
        )
        for side in ("left", "right"):
            joint = root.find(f"./joint[@name='onrobot_2fg7_{side}_finger_joint']")
            self.assertIsNotNone(joint)
            limit = joint.find("limit")
            self.assertEqual(float(limit.get("lower")), 0.0)
            self.assertEqual(float(limit.get("upper")), 0.019)
            self.assertEqual(float(limit.get("velocity")), 0.225)
        right_mimic = root.find(
            "./joint[@name='onrobot_2fg7_right_finger_joint']/mimic"
        )
        self.assertEqual(right_mimic.get("joint"), "onrobot_2fg7_left_finger_joint")
        self.assertEqual(float(right_mimic.get("multiplier")), 1.0)


@unittest.skipIf(yaml is None, "PyYAML is only required by the cuMotion environment")
class CumotionBenchmarkConfigTests(unittest.TestCase):
    def test_cases_are_valid_and_unique(self) -> None:
        with (ARENA_DIR / "config/cumotion_benchmark_cases.yaml").open(
            "r", encoding="utf-8"
        ) as stream:
            data = yaml.safe_load(stream)
        cases = data["cases"]
        self.assertGreaterEqual(len(cases), 4)
        self.assertEqual(len({case["name"] for case in cases}), len(cases))
        for case in cases:
            self.assertIn(case["mode"], {"cspace", "pose"})
            self.assertEqual(len(case["start_joint_positions"]), 6)
            self.assertTrue(
                len(case.get("target_joint_positions", [])) == 6
                or (
                    case["mode"] == "pose"
                    and len(case.get("target_pose", {}).get("position_xyz", [])) == 3
                    and len(case.get("target_pose", {}).get("quaternion_xyzw", [])) == 4
                )
            )

    def test_planner_config_matches_six_axis_robot(self) -> None:
        with (ARENA_DIR / "config/tm5s_cumotion_planner.yaml").open(
            "r", encoding="utf-8"
        ) as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(len(config["distance_metric_weights"]), 6)
        self.assertTrue(config["enable_self_collision_checking"])
        self.assertEqual(config["cuda_tree_params"]["num_nodes_cpu_gpu_crossover"], 0)

    def test_generated_sampled_sphere_coverage_audit_if_present(self) -> None:
        manifest_path = ARENA_DIR / "generated/cumotion/asset_manifest.json"
        if not manifest_path.is_file():
            self.skipTest("Generated cuMotion assets are not present")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        margin = manifest["surface_coverage_margin_m"]
        self.assertGreaterEqual(manifest["collision_sphere_count"], 100)
        for link, audit in manifest["sphere_coverage_audit"].items():
            with self.subTest(link=link):
                self.assertLessEqual(audit["audited_max_uncovered_gap_m"], -margin + 1e-9)

    def test_watson_fixed_tool_stack_exclusions_if_present(self) -> None:
        xrdf_path = (
            ARENA_DIR
            / "generated/tool_profiles/watson_qc_nominal/cumotion/tm5s_with_2fg7.xrdf"
        )
        if not xrdf_path.is_file():
            self.skipTest("Generated Watson candidate assets are not present")
        with xrdf_path.open("r", encoding="utf-8") as stream:
            xrdf = yaml.safe_load(stream)
        ignored = xrdf["self_collision"]["ignore"]
        self.assertIn("onrobot_2fg7_base_link", ignored["link_6"])
        self.assertIn("onrobot_qc_robot_side_link", ignored["link_6"])
        self.assertIn(
            "onrobot_2fg7_base_link",
            ignored["onrobot_qc_robot_side_link"],
        )

    @unittest.skipIf(cumotion is None, "cuMotion is only installed in its isolated environment")
    def test_generated_robot_loads_with_six_axes_if_present(self) -> None:
        model_dir = ARENA_DIR / "generated/cumotion"
        urdf_path = model_dir / "tm5s_with_2fg7.urdf"
        xrdf_path = model_dir / "tm5s_with_2fg7.xrdf"
        if not urdf_path.is_file() or not xrdf_path.is_file():
            self.skipTest("Generated cuMotion assets are not present")
        robot = cumotion.load_robot_from_file(str(xrdf_path), str(urdf_path))
        self.assertEqual(robot.num_cspace_coords(), 6)
        self.assertIn("flange", robot.tool_frame_names())

if __name__ == "__main__":
    unittest.main()
