from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from pin_axis_3d_sim.workcell_scan import (
    application_pin_guide_points,
    author_usd_points,
    author_usd_visual_polyline,
    deterministic_even_point_cap,
    deterministic_voxel_downsample,
    drawer_top_rim_outline_points,
    load_ply_xyzrgb,
    registration_validation_failures,
    transform_points,
    transform_points_matrix,
)


def write_binary_ply(path: Path, records: np.ndarray, declared_count: int | None = None) -> None:
    count = records.shape[0] if declared_count is None else declared_count
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment test fixture with non-canonical property order\n"
        f"element vertex {count}\n"
        "property uchar blue\n"
        "property float x\n"
        "property uchar red\n"
        "property float y\n"
        "property float z\n"
        "property uchar green\n"
        "property float confidence\n"
        "element face 0\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        records.tofile(stream)


class BinaryPlyLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dtype = np.dtype(
            [
                ("blue", "u1"),
                ("x", "<f4"),
                ("red", "u1"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("green", "u1"),
                ("confidence", "<f4"),
            ]
        )

    def test_loads_binary_little_endian_xyzrgb_by_property_name(self) -> None:
        records = np.array(
            [
                (30, 1000.0, 10, 2000.0, 3000.0, 20, 0.9),
                (60, -5.5, 40, 6.5, 7.5, 50, 0.8),
            ],
            dtype=self.dtype,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.ply"
            write_binary_ply(path, records)
            points, colors = load_ply_xyzrgb(path)

        np.testing.assert_allclose(
            points,
            [[1000.0, 2000.0, 3000.0], [-5.5, 6.5, 7.5]],
        )
        np.testing.assert_array_equal(colors, [[10, 20, 30], [40, 50, 60]])
        self.assertEqual(points.dtype, np.float64)
        self.assertEqual(colors.dtype, np.uint8)

    def test_rejects_truncated_vertex_data(self) -> None:
        records = np.array([(3, 1.0, 1, 2.0, 3.0, 2, 0.5)], dtype=self.dtype)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.ply"
            write_binary_ply(path, records, declared_count=2)
            with self.assertRaisesRegex(ValueError, "truncated PLY vertex data"):
                load_ply_xyzrgb(path)


class PointTransformTests(unittest.TestCase):
    def test_applies_mm_scale_rotation_then_translation(self) -> None:
        half_root_two = np.sqrt(0.5)
        transformed = transform_points(
            np.array([[1000.0, 0.0, 0.0], [0.0, 2000.0, 0.0]]),
            scale=0.001,
            translation_xyz=[0.1, 0.2, 0.3],
            quaternion_xyzw=[0.0, 0.0, half_root_two, half_root_two],
        )
        np.testing.assert_allclose(
            transformed,
            [[0.1, 1.2, 0.3], [-1.9, 0.2, 0.3]],
            atol=1e-12,
        )

    def test_rejects_zero_quaternion(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-zero magnitude"):
            transform_points(
                np.zeros((1, 3)),
                scale=0.001,
                translation_xyz=[0.0, 0.0, 0.0],
                quaternion_xyzw=[0.0, 0.0, 0.0, 0.0],
            )

    def test_applies_proper_rotation_matrix_transform(self) -> None:
        transformed = transform_points_matrix(
            np.array([[1.0, 0.0, 0.0]]),
            scale=2.0,
            translation_xyz=[0.1, 0.2, 0.3],
            rotation_matrix=[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        )
        np.testing.assert_allclose(transformed, [[0.1, 2.2, 0.3]])

    def test_rejects_reflection_matrix(self) -> None:
        with self.assertRaisesRegex(ValueError, "proper orthonormal"):
            transform_points_matrix(
                np.zeros((1, 3)),
                scale=1.0,
                translation_xyz=[0.0, 0.0, 0.0],
                rotation_matrix=[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            )


class VoxelDownsampleTests(unittest.TestCase):
    def test_averages_voxels_independently_of_input_order(self) -> None:
        points = np.array(
            [
                [1.2, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [2.1, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [1.4, 0.0, 0.0],
            ]
        )
        colors = np.array(
            [[100, 110, 120], [20, 30, 40], [200, 210, 220], [10, 20, 30], [120, 130, 140]],
            dtype=np.uint8,
        )
        result_points, result_colors = deterministic_voxel_downsample(
            points,
            colors,
            voxel_size_m=1.0,
            max_points=None,
        )
        reversed_points, reversed_colors = deterministic_voxel_downsample(
            points[::-1],
            colors[::-1],
            voxel_size_m=1.0,
            max_points=None,
        )

        np.testing.assert_allclose(result_points, [[0.15, 0.0, 0.0], [1.3, 0.0, 0.0], [2.1, 0.0, 0.0]])
        np.testing.assert_array_equal(
            result_colors,
            [[15, 25, 35], [110, 120, 130], [200, 210, 220]],
        )
        np.testing.assert_array_equal(result_points, reversed_points)
        np.testing.assert_array_equal(result_colors, reversed_colors)

    def test_point_cap_is_reproducible_and_includes_sorted_extents(self) -> None:
        points = np.column_stack((np.arange(6, dtype=float), np.zeros((6, 2))))
        colors = np.column_stack(
            (np.arange(6, dtype=np.uint8), np.zeros((6, 2), dtype=np.uint8))
        )
        result_points, result_colors = deterministic_voxel_downsample(
            points,
            colors,
            voxel_size_m=0.5,
            max_points=3,
        )
        np.testing.assert_array_equal(result_points[:, 0], [0.0, 2.0, 5.0])
        np.testing.assert_array_equal(result_colors[:, 0], [0, 2, 5])

    def test_even_point_cap_can_be_applied_after_voxel_count_validation(self) -> None:
        points = np.column_stack((np.arange(6, dtype=float), np.zeros((6, 2))))
        colors = np.zeros((6, 3), dtype=np.uint8)
        result_points, _ = deterministic_even_point_cap(points, colors, max_points=3)
        np.testing.assert_array_equal(result_points[:, 0], [0.0, 2.0, 5.0])


class RegistrationEvidenceTests(unittest.TestCase):
    def test_reports_failed_metric_without_promoting_visual_registration(self) -> None:
        validation = {"rotation_determinant": 1.0, "sift_3d_inliers": 20}
        thresholds = {
            "rotation_determinant_abs_error_max": 1.0e-5,
            "sift_3d_inliers_min": 25,
        }
        failures = registration_validation_failures(validation, thresholds)
        self.assertTrue(any("sift_3d_inliers" in failure for failure in failures))
        self.assertTrue(any("missing threshold" in failure for failure in failures))


class DrawerOutlineTests(unittest.TestCase):
    def test_builds_only_a_closed_top_rim_outline(self) -> None:
        outline = drawer_top_rim_outline_points(
            center_xy_mm=[0.0, 0.0],
            size_xy_mm=[2.0, 4.0],
            short_axis_yaw_deg=0.0,
            rim_height_above_foam_mm=1.0,
            foam_plane_z_coefficients_mm=[0.0, 0.0, 0.0],
            secondary_to_primary_rotation=np.eye(3),
            secondary_to_primary_translation_mm=[0.0, 0.0, 0.0],
            scale_to_metres=0.001,
            primary_to_base_translation_m=[0.0, 0.0, 0.0],
            primary_to_base_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
        )
        np.testing.assert_allclose(
            outline,
            [
                [-0.001, -0.002, 0.001],
                [0.001, -0.002, 0.001],
                [0.001, 0.002, 0.001],
                [-0.001, 0.002, 0.001],
                [-0.001, -0.002, 0.001],
            ],
        )


class ApplicationPinGuideTests(unittest.TestCase):
    def test_builds_ten_millimetre_section_and_closed_perpendicular_ring(self) -> None:
        clear_start = np.array([0.0, 0.0, 0.15225])
        pinch = np.array([0.0, 0.0, 0.15725])
        specimen_near = np.array([0.0, 0.0, 0.16225])
        axis = np.array([0.0, 0.0, 1.0])

        guide = application_pin_guide_points(
            clear_start_xyz=clear_start,
            pinch_xyz=pinch,
            specimen_near_xyz=specimen_near,
            axis=axis,
            boundary_radius_m=0.012,
            segments=12,
        )

        np.testing.assert_allclose(
            guide["bare_section"],
            [clear_start, specimen_near],
        )
        np.testing.assert_allclose(guide["pinch"], pinch)
        self.assertAlmostEqual(
            float(np.linalg.norm(guide["bare_section"][1] - guide["bare_section"][0])),
            0.010,
        )
        self.assertAlmostEqual(float(np.linalg.norm(pinch - clear_start)), 0.005)
        self.assertAlmostEqual(float(np.linalg.norm(specimen_near - pinch)), 0.005)

        boundary = guide["specimen_boundary"]
        self.assertEqual(boundary.shape, (13, 3))
        np.testing.assert_allclose(boundary[0], boundary[-1], atol=1.0e-12)
        offsets = boundary[:-1] - specimen_near
        np.testing.assert_allclose(offsets @ axis, 0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            np.linalg.norm(offsets, axis=1),
            0.012,
            atol=1.0e-12,
        )

    def test_rejects_pinch_that_is_not_the_section_midpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "midpoint"):
            application_pin_guide_points(
                clear_start_xyz=[0.0, 0.0, 0.15225],
                pinch_xyz=[0.0, 0.0, 0.15625],
                specimen_near_xyz=[0.0, 0.0, 0.16225],
                axis=[0.0, 0.0, 1.0],
                boundary_radius_m=0.012,
                segments=12,
            )


class UsdAuthoringTests(unittest.TestCase):
    def test_authors_vertex_interpolated_display_colors(self) -> None:
        try:
            from pxr import Usd, UsdGeom
        except ImportError:
            self.skipTest("pxr is only present in the Isaac Sim environment")

        stage = Usd.Stage.CreateInMemory()
        result = author_usd_points(
            stage,
            "/Workcell/TestScan",
            np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
            np.array([[255, 0, 0], [0, 128, 255]], dtype=np.uint8),
            point_width_m=0.001,
        )

        self.assertTrue(result.GetPrim().IsValid())
        self.assertEqual(result.GetWidthsInterpolation(), UsdGeom.Tokens.constant)
        display_color = result.GetDisplayColorPrimvar()
        self.assertEqual(display_color.GetInterpolation(), UsdGeom.Tokens.vertex)
        self.assertEqual(len(display_color.Get()), 2)

    def test_drawer_outline_is_guide_only_without_physics_apis(self) -> None:
        try:
            from pxr import Usd, UsdGeom, UsdPhysics
        except ImportError:
            self.skipTest("pxr is only present in the Isaac Sim environment")

        stage = Usd.Stage.CreateInMemory()
        curve = author_usd_visual_polyline(
            stage,
            "/Workcell/ProvisionalDrawerOuterTopRim",
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            [255, 145, 35],
            width_m=0.0015,
            opacity=0.9,
            geometry_status="provisional_outer_top_rim_only",
        )
        prim = curve.GetPrim()
        self.assertEqual(curve.GetPurposeAttr().Get(), UsdGeom.Tokens.guide)
        self.assertTrue(prim.GetAttribute("magi:visualOnly").Get())
        self.assertFalse(prim.GetAttribute("magi:collisionQualified").Get())
        self.assertFalse(prim.HasAPI(UsdPhysics.CollisionAPI))
        self.assertFalse(prim.HasAPI(UsdPhysics.RigidBodyAPI))
        if hasattr(UsdPhysics, "MassAPI"):
            self.assertFalse(prim.HasAPI(UsdPhysics.MassAPI))

        visible_overlay = author_usd_visual_polyline(
            stage,
            "/Workcell/VisibleApplicationGuide",
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]]),
            [50, 220, 255],
            width_m=0.0012,
            opacity=1.0,
            geometry_status="user_selected_10mm_clear_pin_section",
            usd_purpose="default",
        )
        overlay_prim = visible_overlay.GetPrim()
        self.assertEqual(
            visible_overlay.GetPurposeAttr().Get(),
            UsdGeom.Tokens.default_,
        )
        self.assertTrue(overlay_prim.GetAttribute("magi:visualOnly").Get())
        self.assertFalse(overlay_prim.HasAPI(UsdPhysics.CollisionAPI))
        self.assertFalse(overlay_prim.HasAPI(UsdPhysics.RigidBodyAPI))
        if hasattr(UsdPhysics, "MassAPI"):
            self.assertFalse(overlay_prim.HasAPI(UsdPhysics.MassAPI))


class WorkcellEvidenceConfigTests(unittest.TestCase):
    def test_two_capture_fusion_stays_visual_only_and_unregistered_to_watson(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config/workcell_scan.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["point_cloud"]["capture_id"], "240mm")
        self.assertEqual(config["secondary_point_cloud"]["capture_id"], "440mm")
        registration = config["capture_registration"]
        self.assertEqual(
            registration["status"],
            "passed_visual_fusion_only_not_metrology_or_robot_registration",
        )
        rotation = np.asarray(registration["secondary_to_primary_rotation"])
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=7)
        self.assertGreaterEqual(
            registration["conservative_uncertainty_translation_mm"], 0.5
        )
        self.assertGreaterEqual(
            registration["conservative_uncertainty_rotation_deg"], 0.4
        )
        self.assertTrue(config["scope"]["captures_registered_to_each_other"])
        self.assertTrue(config["scope"]["captures_stitched"])
        self.assertFalse(config["scope"]["registered_to_watson_base"])
        self.assertFalse(config["scope"]["collision_enabled"])
        self.assertFalse(
            config["scan_derived_drawer_geometry"]["collision_qualified"]
        )
        robot = config["robot"]
        self.assertEqual(robot["asset_profile"], "watson_qc_nominal")
        self.assertEqual(
            robot["asset_path"],
            "generated/isaac/6.0.1-watson-qc-10mm/tm5s_with_2fg7/tm5s_with_2fg7.usda",
        )
        self.assertEqual(
            robot["asset_sha256"],
            "d05dee28e5bd81ce4564f6ef52f7c2084f1020e1db81d215e834875fca5aa0bc",
        )
        self.assertEqual(
            robot["source_urdf_sha256"],
            "ee7dadbee3e898152948c133f859f7bd085c93614fc8274549158cca10a18d03",
        )
        self.assertEqual(
            robot["source_urdf"],
            "generated/tool_profiles/watson_qc_nominal/cumotion/tm5s_with_2fg7.urdf",
        )
        self.assertEqual(
            robot["import_report"],
            "outputs/isaac_sim/6.0.1/watson_qc_10mm_import_report.json",
        )
        self.assertEqual(
            robot["import_report_sha256"],
            "8f62dc63514f8840a753abb629f2dcc83dfa6e2294594e7256b99e7945f11bc3",
        )
        self.assertEqual(
            robot["tool_metadata_sha256"],
            "7d35a8b32f91b2850e8a3ced05d221508e09a6a8225f34f5c15f509518c52017",
        )
        self.assertEqual(
            robot["quick_changer_revision_status"],
            "physically_confirmed_qc_r_v3_ip67",
        )
        self.assertEqual(
            robot["quick_changer_keyed_yaw_status"],
            "physical_clock_features_confirmed_numeric_working_cad_registration_pending",
        )
        self.assertEqual(
            robot["pin_grasp_tcp_status"],
            "user_selected_10mm_cad_relative_baseline_not_physically_calibrated",
        )
        application_visual = config["application_grasp_visual"]
        self.assertTrue(application_visual["enabled"])
        self.assertEqual(application_visual["parent_frame"], "onrobot_2fg7_origin")
        self.assertEqual(application_visual["visual_root_name"], "ApplicationPinGuide")
        self.assertEqual(
            application_visual["bare_section"]["display_color_rgb"],
            [50, 220, 255],
        )
        self.assertEqual(application_visual["bare_section"]["line_width_m"], 0.0012)
        self.assertEqual(application_visual["bare_section"]["opacity"], 1.0)
        self.assertEqual(
            application_visual["pinch_marker"]["source"],
            "imported_gripper_tcp_sphere",
        )
        self.assertEqual(application_visual["pinch_marker"]["display_color_name"], "blue")
        self.assertEqual(application_visual["pinch_marker"]["display_radius_m"], 0.0015)
        self.assertEqual(
            application_visual["specimen_near_boundary"]["marker_radius_m"],
            0.012,
        )
        self.assertEqual(
            application_visual["specimen_near_boundary"]["marker_segments"],
            48,
        )
        self.assertEqual(
            application_visual["specimen_near_boundary"]["display_color_rgb"],
            [255, 145, 35],
        )
        self.assertEqual(application_visual["purpose"], "guide")
        self.assertEqual(application_visual["usd_render_purpose"], "default")
        self.assertTrue(application_visual["visual_only"])
        self.assertFalse(application_visual["collision_enabled"])
        manifest_path = config_path.parents[1] / config["point_cloud"][
            "provenance_manifest"
        ]
        if not manifest_path.is_file():
            self.skipTest("Generated workcell-scan provenance is not present")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_registration = manifest["capture_registration"]
        self.assertTrue(manifest_registration["validation_gates_passed"])
        self.assertEqual(
            config["capture_registration"]["validation_thresholds"],
            manifest_registration["validation_thresholds"],
        )
        self.assertEqual(
            registration_validation_failures(
                manifest_registration["validation"],
                manifest_registration["validation_thresholds"],
            ),
            [],
        )
        manifest_drawer = manifest["scan_derived_drawer_geometry"]
        self.assertEqual(
            config["scan_derived_drawer_geometry"]["outer_top_rim_size_xy_mm"],
            manifest_drawer["outer_top_rim_size_xy_mm"],
        )
        self.assertTrue(manifest_drawer["visual_only"])
        self.assertFalse(manifest_drawer["collision_qualified"])


if __name__ == "__main__":
    unittest.main()
