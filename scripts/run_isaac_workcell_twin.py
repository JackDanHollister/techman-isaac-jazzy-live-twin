#!/usr/bin/env python3
"""Display the TM5S and a measured-scale pinned-insect scan in Isaac Sim.

This viewer is simulation-only and creates no ROS graph, sensor connection,
Watson network connection, controller, or robot command. The point cloud is a
visual reference, not physics or cuMotion collision geometry.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import struct
import sys
import time
import traceback
from typing import Any

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


ARENA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_USD = (
    ARENA_DIR
    / "generated/isaac/6.0.1-watson-qc-10mm/"
    "tm5s_with_2fg7/tm5s_with_2fg7.usda"
)
DEFAULT_CONFIG = ARENA_DIR / "config/workcell_scan.yaml"
DEFAULT_REPORT = ARENA_DIR / "outputs/isaac_sim/6.0.1/workcell_twin_report.json"
EXPECTED_PYTHON = (3, 12)
EXPECTED_ISAAC_PACKAGES = {
    "isaacsim": "6.0.1.0",
    "isaacsim-asset": "6.0.1.0",
    "isaacsim-core": "6.0.1.0",
}
EXPECTED_DOF_NAMES = [f"joint_{index}" for index in range(1, 7)]
RENDER_DT_SECONDS = 1.0 / 60.0
MAX_TRACKING_ERROR_RADIANS = 1.0e-6
MAX_RENDERED_LINK_POSITION_ERROR_METERS = 1.0e-5
MAX_RENDERED_LINK_ORIENTATION_ERROR_RADIANS = 1.0e-5
GUIDE_PURPOSE_DISPLAY_SETTING = "/persistent/app/hydra/displayPurpose/guide"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--camera-view",
        choices=("workcell", "application"),
        default="workcell",
        help="Start with the whole-workcell view or a close application-pin view.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Stop after this many wall-clock seconds; zero runs until closed.",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="Capture one rendered frame to a new PNG path after one second.",
    )
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_dimensions(path: Path) -> list[int]:
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Screenshot is not a valid PNG: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ValueError(f"Screenshot has invalid dimensions: {width}x{height}")
    return [width, height]


def validate_runtime() -> dict[str, str]:
    if sys.version_info[:2] != EXPECTED_PYTHON:
        raise RuntimeError(
            "Isaac Sim 6.0.1 workcell viewer requires Python 3.12; "
            f"found {sys.version.split()[0]}"
        )
    versions: dict[str, str] = {}
    for package_name, expected_version in EXPECTED_ISAAC_PACKAGES.items():
        try:
            actual_version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Missing required package: {package_name}") from exc
        if actual_version != expected_version:
            raise RuntimeError(
                f"{package_name} must be {expected_version}; found {actual_version}"
            )
        versions[package_name] = actual_version
    return versions


def resolve_arena_file(value: Any, *, label: str, allowed_root: Path) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        raise ValueError(f"{label} must be relative to the demo directory")
    resolved = (ARENA_DIR / relative).resolve()
    root = allowed_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"{label} must stay below {root}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def validate_imported_asset_artifacts(
    import_report: dict[str, Any],
    robot_asset_path: Path,
) -> dict[str, dict[str, Any]]:
    """Verify every USD payload recorded by the hash-bound import report."""

    artifacts = import_report.get("asset_artifacts")
    listed_files = import_report.get("asset_files")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Robot import report has no asset-artifact manifest")
    if not isinstance(listed_files, list) or set(listed_files) != set(artifacts):
        raise ValueError("Robot import report asset_files and asset_artifacts disagree")

    artifact_root = robot_asset_path.parents[1].resolve()
    validated: dict[str, dict[str, Any]] = {}
    selected_asset_found = False
    for relative_name in sorted(artifacts):
        relative_path = Path(relative_name)
        if relative_path.is_absolute():
            raise ValueError("Robot asset-artifact paths must be relative")
        artifact_path = (artifact_root / relative_path).resolve()
        if artifact_path == artifact_root or artifact_root not in artifact_path.parents:
            raise ValueError("Robot asset artifact escapes its generated asset directory")
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Robot asset artifact is missing: {artifact_path}")
        evidence = artifacts[relative_name]
        if not isinstance(evidence, dict):
            raise ValueError(f"Invalid asset-artifact evidence for {relative_name}")
        expected_size = evidence.get("size_bytes")
        expected_sha256 = str(evidence.get("sha256", ""))
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
            or artifact_path.stat().st_size != expected_size
        ):
            raise ValueError(f"Robot asset artifact size mismatch: {relative_name}")
        actual_sha256 = sha256_file(artifact_path)
        if len(expected_sha256) != 64 or actual_sha256 != expected_sha256:
            raise ValueError(f"Robot asset artifact hash mismatch: {relative_name}")
        selected_asset_found |= artifact_path == robot_asset_path
        validated[relative_name] = {
            "path": str(artifact_path),
            "size_bytes": expected_size,
            "sha256": actual_sha256,
        }
    if not selected_asset_found:
        raise ValueError("Selected robot USD is absent from the asset-artifact manifest")
    return validated


def finite_vector(value: Any, *, length: int, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain {length} finite numbers")
    return vector


def load_workcell_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or raw.get("format_version") != 1:
        raise ValueError("Workcell scan config must use format_version: 1")
    if raw.get("status") != "provisional_visual_reference":
        raise ValueError("Workcell config must retain provisional_visual_reference status")

    point_cloud = raw.get("point_cloud")
    secondary_point_cloud = raw.get("secondary_point_cloud")
    capture_registration = raw.get("capture_registration")
    drawer_geometry = raw.get("scan_derived_drawer_geometry")
    scan_to_base = raw.get("scan_to_base")
    robot = raw.get("robot")
    application_grasp_visual = raw.get("application_grasp_visual")
    viewer = raw.get("viewer")
    scope = raw.get("scope")
    if not all(
        isinstance(section, dict)
        for section in (
            point_cloud,
            secondary_point_cloud,
            capture_registration,
            drawer_geometry,
            scan_to_base,
            robot,
            application_grasp_visual,
            viewer,
            scope,
        )
    ):
        raise ValueError("Workcell config is missing a required mapping")

    scan_root = ARENA_DIR / "outputs/workcell_scan"
    point_cloud_path = resolve_arena_file(
        point_cloud.get("path"), label="point_cloud.path", allowed_root=scan_root
    )
    manifest_path = resolve_arena_file(
        point_cloud.get("provenance_manifest"),
        label="point_cloud.provenance_manifest",
        allowed_root=scan_root,
    )
    actual_hash = sha256_file(point_cloud_path)
    expected_hash = str(point_cloud.get("sha256", ""))
    if actual_hash != expected_hash:
        raise ValueError(
            f"Point-cloud SHA-256 mismatch: expected {expected_hash}; found {actual_hash}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "decoded_visual_reference_not_robot_registered":
        raise ValueError("Point-cloud provenance does not retain its unregistered status")
    secondary_point_cloud_path = resolve_arena_file(
        secondary_point_cloud.get("path"),
        label="secondary_point_cloud.path",
        allowed_root=scan_root,
    )
    secondary_manifest_path = resolve_arena_file(
        secondary_point_cloud.get("provenance_manifest"),
        label="secondary_point_cloud.provenance_manifest",
        allowed_root=scan_root,
    )
    if secondary_manifest_path != manifest_path:
        raise ValueError("Both point clouds must share the same provenance manifest")
    secondary_point_cloud_sha256 = sha256_file(secondary_point_cloud_path)
    if secondary_point_cloud_sha256 != str(secondary_point_cloud.get("sha256", "")):
        raise ValueError("Secondary point-cloud SHA-256 mismatch")
    if (
        point_cloud.get("capture_id") != "240mm"
        or secondary_point_cloud.get("capture_id") != "440mm"
        or secondary_point_cloud.get("input_units") != "millimetres"
    ):
        raise ValueError("Workcell fusion must retain the reviewed 240mm/440mm capture order")
    secondary_expected_points = int(
        secondary_point_cloud.get("expected_valid_points", 0)
    )
    if secondary_expected_points <= 0:
        raise ValueError("secondary_point_cloud.expected_valid_points must be positive")
    registration_status = (
        "passed_visual_fusion_only_not_metrology_or_robot_registration"
    )
    if capture_registration.get("status") != registration_status:
        raise ValueError("Capture registration must retain its visual-only status")
    registration_evidence_path = resolve_arena_file(
        capture_registration.get("evidence"),
        label="capture_registration.evidence",
        allowed_root=scan_root,
    )
    if (
        registration_evidence_path != manifest_path
        or capture_registration.get("evidence_section") != "capture_registration"
    ):
        raise ValueError("Capture registration must point to its manifest section")
    manifest_registration = manifest.get("capture_registration", {})
    if manifest_registration.get("status") != registration_status:
        raise ValueError("Provenance manifest lacks the reviewed visual registration")
    secondary_to_primary_rotation = np.asarray(
        capture_registration.get("secondary_to_primary_rotation"), dtype=np.float64
    )
    secondary_to_primary_translation_mm = finite_vector(
        capture_registration.get("secondary_to_primary_translation_mm"),
        length=3,
        label="capture_registration.secondary_to_primary_translation_mm",
    )
    if secondary_to_primary_rotation.shape != (3, 3) or not np.all(
        np.isfinite(secondary_to_primary_rotation)
    ):
        raise ValueError("capture_registration rotation must be a finite 3x3 matrix")
    determinant = float(np.linalg.det(secondary_to_primary_rotation))
    orthogonality_error = float(
        np.linalg.norm(
            secondary_to_primary_rotation @ secondary_to_primary_rotation.T
            - np.eye(3)
        )
    )
    if abs(determinant - 1.0) > 1.0e-5 or orthogonality_error > 1.0e-5:
        raise ValueError("capture_registration rotation is not a proper rotation")
    if not np.allclose(
        secondary_to_primary_rotation,
        np.asarray(manifest_registration.get("target_to_source_rotation")),
        atol=1.0e-12,
    ) or not np.allclose(
        secondary_to_primary_translation_mm,
        np.asarray(manifest_registration.get("target_to_source_translation_mm")),
        atol=1.0e-12,
    ):
        raise ValueError("Capture registration disagrees with its provenance manifest")
    source_to_target_rotation = np.asarray(
        manifest_registration.get("source_to_target_rotation"), dtype=np.float64
    )
    source_to_target_translation_mm = finite_vector(
        manifest_registration.get("source_to_target_translation_mm"),
        length=3,
        label="manifest source_to_target_translation_mm",
    )
    if source_to_target_rotation.shape != (3, 3) or not np.all(
        np.isfinite(source_to_target_rotation)
    ):
        raise ValueError("Manifest source-to-target rotation must be a finite 3x3 matrix")
    if not np.allclose(
        secondary_to_primary_rotation @ source_to_target_rotation,
        np.eye(3),
        atol=2.0e-8,
    ) or not np.allclose(
        secondary_to_primary_rotation @ source_to_target_translation_mm
        + secondary_to_primary_translation_mm,
        np.zeros(3),
        atol=2.0e-6,
    ):
        raise ValueError("Capture registration forward/inverse transforms do not compose")
    validation_thresholds = capture_registration.get("validation_thresholds")
    if not isinstance(validation_thresholds, dict):
        raise ValueError("capture_registration.validation_thresholds must be a mapping")
    if validation_thresholds != manifest_registration.get("validation_thresholds"):
        raise ValueError("Capture registration thresholds disagree with the manifest")
    validation_failures = registration_validation_failures(
        manifest_registration.get("validation", {}),
        validation_thresholds,
    )
    if validation_failures or manifest_registration.get("validation_gates_passed") is not True:
        raise ValueError(
            "Capture registration validation gates failed: "
            + "; ".join(validation_failures or ["manifest pass flag is not true"])
        )
    uncertainty_translation_mm = float(
        capture_registration.get("conservative_uncertainty_translation_mm", 0.0)
    )
    uncertainty_rotation_deg = float(
        capture_registration.get("conservative_uncertainty_rotation_deg", 0.0)
    )
    if uncertainty_translation_mm < 0.5 or uncertainty_rotation_deg < 0.4:
        raise ValueError("Capture registration must retain conservative uncertainty")
    expected_combined_points = int(
        capture_registration.get("expected_combined_valid_points", 0)
    )
    expected_unique_voxels = int(
        capture_registration.get("expected_unique_voxels_before_cap_at_0_5mm", 0)
    )
    if (
        expected_combined_points
        != int(manifest_registration.get("combined_valid_points", 0))
        or expected_unique_voxels
        != int(manifest_registration.get("combined_unique_voxels_at_0_5mm", 0))
        or expected_combined_points <= 0
        or expected_unique_voxels <= 0
    ):
        raise ValueError("Capture fusion counts disagree with the provenance manifest")
    if (
        drawer_geometry.get("status") != "provisional_outer_top_rim_only"
        or drawer_geometry.get("frame") != "zivid_440mm_saved_local"
        or drawer_geometry.get("collision_qualified") is not False
        or drawer_geometry.get("underside_observed") is not False
        or drawer_geometry.get("inner_dimensions_verified") is not False
        or drawer_geometry.get("table_or_robot_datum_observed") is not False
        or drawer_geometry.get("visual_only") is not True
    ):
        raise ValueError("Scan-derived drawer geometry must remain provisional and non-colliding")
    drawer_evidence_path = resolve_arena_file(
        drawer_geometry.get("evidence"),
        label="scan_derived_drawer_geometry.evidence",
        allowed_root=scan_root,
    )
    if (
        drawer_evidence_path != manifest_path
        or drawer_geometry.get("evidence_section")
        != "scan_derived_drawer_geometry"
    ):
        raise ValueError("Drawer geometry must point to its manifest section")
    manifest_drawer_geometry = manifest.get("scan_derived_drawer_geometry", {})
    if (
        manifest_drawer_geometry.get("status")
        != drawer_geometry.get("status")
        or manifest_drawer_geometry.get("frame") != drawer_geometry.get("frame")
        or manifest_drawer_geometry.get("visual_only") is not True
        or manifest_drawer_geometry.get("collision_qualified") is not False
    ):
        raise ValueError("Drawer geometry status disagrees with its provenance manifest")
    drawer_size_xy_mm = finite_vector(
        drawer_geometry.get("outer_top_rim_size_xy_mm"),
        length=2,
        label="scan_derived_drawer_geometry.outer_top_rim_size_xy_mm",
    )
    drawer_center_xy_mm = finite_vector(
        drawer_geometry.get("outer_top_rim_center_xy_mm"),
        length=2,
        label="scan_derived_drawer_geometry.outer_top_rim_center_xy_mm",
    )
    drawer_foam_plane_mm = finite_vector(
        drawer_geometry.get("foam_plane_z_a_x_plus_b_y_plus_c_mm"),
        length=3,
        label="scan_derived_drawer_geometry foam plane",
    )
    drawer_rim_height_mm = float(
        drawer_geometry.get("rim_height_above_foam_mm", 0.0)
    )
    drawer_yaw_deg = float(drawer_geometry.get("short_axis_yaw_deg", np.nan))
    if (
        np.any(drawer_size_xy_mm <= 0.0)
        or not np.isfinite(drawer_rim_height_mm)
        or drawer_rim_height_mm <= 0.0
        or not np.isfinite(drawer_yaw_deg)
    ):
        raise ValueError("Scan-derived drawer dimensions must be positive")
    manifest_drawer_plane = manifest_drawer_geometry.get("foam_plane_z_mm", {}).get(
        "a_x_plus_b_y_plus_c"
    )
    if not all(
        np.allclose(config_value, manifest_value, atol=1.0e-12)
        for config_value, manifest_value in (
            (
                drawer_size_xy_mm,
                manifest_drawer_geometry.get("outer_top_rim_size_xy_mm"),
            ),
            (
                drawer_center_xy_mm,
                manifest_drawer_geometry.get("outer_top_rim_center_xy_mm"),
            ),
            (drawer_foam_plane_mm, manifest_drawer_plane),
            (
                np.asarray([drawer_rim_height_mm, drawer_yaw_deg]),
                np.asarray(
                    [
                        manifest_drawer_geometry.get("rim_height_above_foam_mm"),
                        manifest_drawer_geometry.get("short_axis_yaw_deg"),
                    ]
                ),
            ),
        )
    ):
        raise ValueError("Drawer geometry values disagree with the provenance manifest")
    manifest_drawer_uncertainty = manifest_drawer_geometry.get("uncertainty", {})
    if not np.allclose(
        [
            float(drawer_geometry.get("dimension_uncertainty_mm", 0.0)),
            float(drawer_geometry.get("center_uncertainty_mm", 0.0)),
            float(drawer_geometry.get("yaw_uncertainty_deg", 0.0)),
        ],
        [
            manifest_drawer_uncertainty.get("dimension_mm"),
            manifest_drawer_uncertainty.get("center_mm"),
            manifest_drawer_uncertainty.get("yaw_deg"),
        ],
        atol=1.0e-12,
    ):
        raise ValueError("Drawer uncertainty disagrees with the provenance manifest")
    drawer_outline = drawer_geometry.get("visual_top_rim_outline")
    if not isinstance(drawer_outline, dict):
        raise ValueError("scan_derived_drawer_geometry visual outline must be a mapping")
    drawer_outline_prim_path = str(drawer_outline.get("prim_path", ""))
    drawer_outline_color = finite_vector(
        drawer_outline.get("display_color_rgb"),
        length=3,
        label="drawer visual outline display_color_rgb",
    )
    drawer_outline_opacity = float(drawer_outline.get("opacity", np.nan))
    drawer_outline_width_m = float(drawer_outline.get("line_width_m", 0.0))
    if (
        drawer_outline.get("enabled") is not True
        or drawer_outline.get("geometry_type")
        != "basis_curves_top_rim_outline"
        or drawer_outline.get("purpose") != "guide"
        or drawer_outline.get("collision_enabled") is not False
        or not drawer_outline_prim_path.startswith("/Workcell/")
        or drawer_outline_prim_path == str(point_cloud.get("prim_path", ""))
        or np.any(drawer_outline_color < 0.0)
        or np.any(drawer_outline_color > 255.0)
        or not np.isfinite(drawer_outline_opacity)
        or not 0.0 <= drawer_outline_opacity <= 1.0
        or not np.isfinite(drawer_outline_width_m)
        or drawer_outline_width_m <= 0.0
    ):
        raise ValueError("Drawer top-rim outline must remain a valid visual-only guide")

    scale = float(point_cloud.get("scale_to_metres", 0.0))
    voxel_size = float(point_cloud.get("voxel_size_m", 0.0))
    point_width = float(point_cloud.get("point_width_m", 0.0))
    maximum_points = int(point_cloud.get("maximum_render_points", 0))
    expected_points = int(point_cloud.get("expected_valid_points", 0))
    if point_cloud.get("input_units") != "millimetres" or not np.isclose(scale, 0.001):
        raise ValueError("The Zivid scan must retain its measured millimetre-to-metre scale")
    if voxel_size <= 0.0 or point_width <= 0.0 or maximum_points <= 0 or expected_points <= 0:
        raise ValueError("Point-cloud render settings must be positive")
    prim_path = str(point_cloud.get("prim_path", ""))
    if not prim_path.startswith("/Workcell/"):
        raise ValueError("Point-cloud prim_path must stay below /Workcell")

    translation = finite_vector(
        scan_to_base.get("translation_xyz_m"), length=3, label="scan_to_base.translation_xyz_m"
    )
    quaternion = finite_vector(
        scan_to_base.get("quaternion_xyzw"), length=4, label="scan_to_base.quaternion_xyzw"
    )
    if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1.0e-8):
        raise ValueError("scan_to_base quaternion must be unit length")
    if scan_to_base.get("registration_status") != "provisional_not_measured":
        raise ValueError("scan_to_base must remain explicitly provisional")
    if scan_to_base.get("frame_id") != "base":
        raise ValueError("scan_to_base.frame_id must be base")

    expected_scope = {
        "visual_only": True,
        "collision_enabled": False,
        "registered_to_watson_base": False,
        "captures_registered_to_each_other": True,
        "captures_stitched": True,
        "real_robot_commanded": False,
    }
    failures = [
        f"scope.{name} must be {expected!r}; found {scope.get(name)!r}"
        for name, expected in expected_scope.items()
        if scope.get(name) is not expected
    ]
    if failures:
        raise ValueError("; ".join(failures))

    asset_profile = str(robot.get("asset_profile", ""))
    if asset_profile != "watson_qc_nominal":
        raise ValueError("robot.asset_profile must be watson_qc_nominal")
    expected_robot_status = {
        "quick_changer_assembly_type": "single_standard_robot_side",
        "quick_changer_revision_status": "physically_confirmed_qc_r_v3_ip67",
        "quick_changer_keyed_yaw_status": (
            "physical_clock_features_confirmed_numeric_working_cad_registration_pending"
        ),
        "pin_grasp_tcp_status": (
            "user_selected_10mm_cad_relative_baseline_not_physically_calibrated"
        ),
        "controller_tool_settings_status": (
            "observed_bare_robot_end_flange_zero_tcp_payload_and_cog"
        ),
    }
    for name, expected in expected_robot_status.items():
        if robot.get(name) != expected:
            raise ValueError(f"robot.{name} must be {expected!r}")

    robot_asset_path = resolve_arena_file(
        robot.get("asset_path"),
        label="robot.asset_path",
        allowed_root=ARENA_DIR / "generated/isaac",
    )
    robot_asset_sha256 = sha256_file(robot_asset_path)
    if robot_asset_sha256 != str(robot.get("asset_sha256", "")):
        raise ValueError("Robot USD SHA-256 does not match the workcell config")
    source_urdf_path = resolve_arena_file(
        robot.get("source_urdf"),
        label="robot.source_urdf",
        allowed_root=ARENA_DIR / "generated/tool_profiles",
    )
    expected_source_urdf_sha256 = str(robot.get("source_urdf_sha256", ""))
    if (
        len(expected_source_urdf_sha256) != 64
        or sha256_file(source_urdf_path) != expected_source_urdf_sha256
    ):
        raise ValueError("Robot source URDF does not match its configured SHA-256")

    import_report_path = resolve_arena_file(
        robot.get("import_report"),
        label="robot.import_report",
        allowed_root=ARENA_DIR / "outputs/isaac_sim",
    )
    import_report_sha256 = sha256_file(import_report_path)
    if import_report_sha256 != str(robot.get("import_report_sha256", "")):
        raise ValueError("Robot import-report SHA-256 does not match the workcell config")
    import_report = json.loads(import_report_path.read_text(encoding="utf-8"))
    asset_artifacts = validate_imported_asset_artifacts(
        import_report,
        robot_asset_path,
    )
    import_failures = []
    if import_report.get("validation_profile") != asset_profile:
        import_failures.append("validation profile mismatch")
    if import_report.get("source_urdf_sha256") != expected_source_urdf_sha256:
        import_failures.append("source URDF hash mismatch")
    if Path(str(import_report.get("source_urdf", ""))).resolve() != source_urdf_path:
        import_failures.append("source URDF path mismatch")
    if Path(str(import_report.get("output_usd", ""))).resolve() != robot_asset_path:
        import_failures.append("output USD path mismatch")
    if import_report.get("output_usd_sha256") != robot_asset_sha256:
        import_failures.append("output USD hash mismatch")
    if import_report.get("unresolved_dependencies"):
        import_failures.append("unresolved USD dependencies")
    if import_report.get("dof_count") != 6:
        import_failures.append("validated DOF count is not six")
    if import_report.get("real_robot_commanded") is not False:
        import_failures.append("import report lacks explicit no-command evidence")
    if import_failures:
        raise ValueError("Robot import provenance failed: " + "; ".join(import_failures))

    tool_metadata_path = resolve_arena_file(
        robot.get("tool_metadata"),
        label="robot.tool_metadata",
        allowed_root=ARENA_DIR / "generated/tool_profiles",
    )
    tool_metadata_sha256 = sha256_file(tool_metadata_path)
    if tool_metadata_sha256 != str(robot.get("tool_metadata_sha256", "")):
        raise ValueError("Tool metadata SHA-256 does not match the workcell config")
    tool_metadata = json.loads(tool_metadata_path.read_text(encoding="utf-8"))
    quick_changer = tool_metadata.get("quick_changer_configuration", {})
    if (
        tool_metadata.get("tool_profile") != asset_profile
        or tool_metadata.get("quick_changer_mode") != "standard_robot_side_nominal"
        or quick_changer.get("assembly_type") != "single_standard_robot_side"
    ):
        raise ValueError("Tool metadata is not the reviewed single-QC Watson profile")

    geometry_scope = tool_metadata.get("geometry_scope")
    if (
        not isinstance(geometry_scope, dict)
        or not {"cables", "cable_routing", "cable_wrap_geometry"}.issubset(
            set(geometry_scope.get("excluded", []))
        )
    ):
        raise ValueError("Tool metadata must retain the user-selected cable exclusion")
    application_baseline = tool_metadata.get("application_pin_baseline")
    if not isinstance(application_baseline, dict):
        raise ValueError("Tool metadata has no application pin baseline")
    pin_axis = finite_vector(
        application_baseline.get("pin_axis_from_gripper_toward_specimen"),
        length=3,
        label="tool metadata application pin axis",
    )
    clear_start = finite_vector(
        application_baseline.get("clear_section_start_xyz_from_2fg7_device_origin_m"),
        length=3,
        label="tool metadata clear pin start",
    )
    pinch = finite_vector(
        application_baseline.get("pinch_xyz_from_2fg7_device_origin_m"),
        length=3,
        label="tool metadata pin pinch",
    )
    specimen_near = finite_vector(
        application_baseline.get("specimen_near_point_xyz_from_2fg7_device_origin_m"),
        length=3,
        label="tool metadata specimen-near point",
    )
    clear_pin_length_m = float(
        application_baseline.get("clear_pin_length_before_specimen_m", np.nan)
    )
    pinch_to_specimen_m = float(
        application_baseline.get("pinch_to_specimen_m", np.nan)
    )
    frame_status = tool_metadata.get("frame_status", {})
    if (
        not np.isclose(np.linalg.norm(pin_axis), 1.0, rtol=0.0, atol=1.0e-12)
        or not np.isclose(clear_pin_length_m, 0.010, rtol=0.0, atol=1.0e-12)
        or not np.isclose(pinch_to_specimen_m, 0.005, rtol=0.0, atol=1.0e-12)
        or application_baseline.get("lateral_alignment")
        != "centred_between_inward_fingers"
        or application_baseline.get("grasp_point_on_clear_section") != "midpoint"
        or application_baseline.get("specimen_geometry") != "variable_not_fixed"
        or application_baseline.get("status")
        != "user_selected_10mm_initial_baseline_2026_07_21"
        or frame_status.get("pin_grasp_tcp")
        != expected_robot_status["pin_grasp_tcp_status"]
        or not np.allclose(
            finite_vector(
                tool_metadata.get("pin_grasp_tcp_xyz_from_device_origin_m"),
                length=3,
                label="tool metadata pin-grasp TCP",
            ),
            pinch,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise ValueError("Tool metadata does not retain the selected centred 10 mm baseline")

    bare_section_visual = application_grasp_visual.get("bare_section")
    pinch_marker_visual = application_grasp_visual.get("pinch_marker")
    specimen_boundary_visual = application_grasp_visual.get(
        "specimen_near_boundary"
    )
    if not all(
        isinstance(section, dict)
        for section in (
            bare_section_visual,
            pinch_marker_visual,
            specimen_boundary_visual,
        )
    ):
        raise ValueError("application_grasp_visual is missing a style mapping")
    visual_root_name = str(application_grasp_visual.get("visual_root_name", ""))
    if (
        application_grasp_visual.get("enabled") is not True
        or application_grasp_visual.get("status")
        != "user_selected_10mm_baseline_visual_only"
        or application_grasp_visual.get("parent_frame") != "onrobot_2fg7_origin"
        or not visual_root_name.isidentifier()
        or application_grasp_visual.get("purpose") != "guide"
        or application_grasp_visual.get("usd_render_purpose") != "default"
        or application_grasp_visual.get("visual_only") is not True
        or application_grasp_visual.get("collision_enabled") is not False
        or pinch_marker_visual.get("source") != "imported_gripper_tcp_sphere"
        or pinch_marker_visual.get("display_color_name") != "blue"
        or specimen_boundary_visual.get("meaning")
        != "axial_start_of_specimen_not_specimen_size_or_shape"
    ):
        raise ValueError("Application pin visual must retain its reviewed guide-only status")
    bare_section_color = finite_vector(
        bare_section_visual.get("display_color_rgb"),
        length=3,
        label="application bare-section colour",
    )
    specimen_boundary_color = finite_vector(
        specimen_boundary_visual.get("display_color_rgb"),
        length=3,
        label="application specimen-boundary colour",
    )
    bare_section_width_m = float(bare_section_visual.get("line_width_m", 0.0))
    bare_section_opacity = float(bare_section_visual.get("opacity", np.nan))
    pinch_marker_radius_m = float(pinch_marker_visual.get("display_radius_m", 0.0))
    specimen_boundary_radius_m = float(
        specimen_boundary_visual.get("marker_radius_m", 0.0)
    )
    specimen_boundary_segments = specimen_boundary_visual.get("marker_segments")
    specimen_boundary_width_m = float(
        specimen_boundary_visual.get("line_width_m", 0.0)
    )
    specimen_boundary_opacity = float(
        specimen_boundary_visual.get("opacity", np.nan)
    )
    if (
        np.any(bare_section_color < 0.0)
        or np.any(bare_section_color > 255.0)
        or np.any(specimen_boundary_color < 0.0)
        or np.any(specimen_boundary_color > 255.0)
        or not np.isfinite(bare_section_width_m)
        or bare_section_width_m <= 0.0
        or not np.isfinite(bare_section_opacity)
        or not 0.0 <= bare_section_opacity <= 1.0
        or not np.isfinite(pinch_marker_radius_m)
        or pinch_marker_radius_m <= 0.0
        or pinch_marker_radius_m >= clear_pin_length_m / 2.0
        or not np.isfinite(specimen_boundary_radius_m)
        or specimen_boundary_radius_m <= 0.0
        or isinstance(specimen_boundary_segments, bool)
        or not isinstance(specimen_boundary_segments, int)
        or specimen_boundary_segments < 8
        or not np.isfinite(specimen_boundary_width_m)
        or specimen_boundary_width_m <= 0.0
        or not np.isfinite(specimen_boundary_opacity)
        or not 0.0 <= specimen_boundary_opacity <= 1.0
    ):
        raise ValueError("Application pin visual styling is invalid")
    application_guide_geometry = application_pin_guide_points(
        clear_start,
        pinch,
        specimen_near,
        pin_axis,
        specimen_boundary_radius_m,
        specimen_boundary_segments,
    )
    if (
        not np.isclose(
            np.linalg.norm(specimen_near - clear_start),
            clear_pin_length_m,
            rtol=0.0,
            atol=1.0e-12,
        )
        or not np.isclose(
            np.linalg.norm(specimen_near - pinch),
            pinch_to_specimen_m,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise ValueError("Application pin metadata distances are internally inconsistent")

    expected_link_paths = import_report.get("expected_link_paths")
    required_application_paths = {
        "onrobot_2fg7_origin",
        "gripper_tcp",
        "pin_grasp_tcp",
    }
    default_prim_path = str(import_report.get("default_prim", ""))
    if not isinstance(expected_link_paths, dict) or not required_application_paths.issubset(
        expected_link_paths
    ):
        raise ValueError("Robot import report lacks application-guide link paths")
    for name in required_application_paths:
        expected_path = expected_link_paths[name]
        if (
            not isinstance(expected_path, str)
            or not expected_path.startswith(default_prim_path + "/")
        ):
            raise ValueError(f"Invalid import-report path for {name}")
    application_parent_prim_path = expected_link_paths["onrobot_2fg7_origin"]
    application_guide_root_prim_path = (
        f"{application_parent_prim_path}/{visual_root_name}"
    )
    application_clear_section_prim_path = (
        f"{application_guide_root_prim_path}/ClearPinSection"
    )
    application_specimen_boundary_prim_path = (
        f"{application_guide_root_prim_path}/SpecimenNearBoundary"
    )
    application_pinch_marker_prim_path = expected_link_paths["gripper_tcp"] + "/sphere"

    controller_tool_audit_path = resolve_arena_file(
        robot.get("controller_tool_audit"),
        label="robot.controller_tool_audit",
        allowed_root=ARENA_DIR / "outputs/watson_controller_tool_audit",
    )
    controller_tool_audit_sha256 = sha256_file(controller_tool_audit_path)
    if controller_tool_audit_sha256 != str(
        robot.get("controller_tool_audit_sha256", "")
    ):
        raise ValueError("Controller tool-audit SHA-256 does not match")
    controller_tool_audit = json.loads(
        controller_tool_audit_path.read_text(encoding="utf-8")
    )
    controller_settings = controller_tool_audit.get("controller_tool_audit", {}).get(
        "settings", {}
    )
    if (
        controller_tool_audit.get("status") != "captured_promotion_blocked"
        or controller_tool_audit.get("motion_commanded") is not False
        or controller_settings.get("active_tcp_name") != "RobotEndFlange"
        or controller_settings.get("tcp_value") != [0.0] * 6
        or controller_settings.get("mass_kg") != 0.0
    ):
        raise ValueError("Controller tool audit does not prove the blocked bare-flange state")

    joint_names = list(robot.get("joint_names", []))
    joint_positions = finite_vector(
        robot.get("display_joint_positions"), length=6, label="robot.display_joint_positions"
    )
    if joint_names != EXPECTED_DOF_NAMES:
        raise ValueError(f"robot.joint_names must be {EXPECTED_DOF_NAMES}")
    camera_eye = finite_vector(viewer.get("camera_eye_xyz_m"), length=3, label="viewer.camera_eye_xyz_m")
    camera_target = finite_vector(
        viewer.get("camera_target_xyz_m"), length=3, label="viewer.camera_target_xyz_m"
    )
    application_camera_eye_offset = finite_vector(
        viewer.get("application_camera_eye_offset_from_pinch_in_2fg7_frame_m"),
        length=3,
        label="viewer.application_camera_eye_offset_from_pinch_in_2fg7_frame_m",
    )
    application_camera_target_offset = finite_vector(
        viewer.get("application_camera_target_offset_from_pinch_in_2fg7_frame_m"),
        length=3,
        label="viewer.application_camera_target_offset_from_pinch_in_2fg7_frame_m",
    )
    if np.linalg.norm(application_camera_eye_offset - application_camera_target_offset) < 0.1:
        raise ValueError("Application camera eye and target must be at least 0.1 m apart")
    floor_top = float(viewer.get("floor_top_z_m"))
    if not np.isfinite(floor_top):
        raise ValueError("viewer.floor_top_z_m must be finite")

    return {
        "raw": raw,
        "point_cloud_path": point_cloud_path,
        "point_cloud_sha256": actual_hash,
        "secondary_point_cloud_path": secondary_point_cloud_path,
        "secondary_point_cloud_sha256": secondary_point_cloud_sha256,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "robot_asset_profile": asset_profile,
        "robot_asset_path": robot_asset_path,
        "robot_asset_sha256": robot_asset_sha256,
        "robot_asset_artifacts": asset_artifacts,
        "robot_import_report_path": import_report_path,
        "robot_import_report_sha256": import_report_sha256,
        "robot_source_urdf_path": source_urdf_path,
        "robot_source_urdf_sha256": expected_source_urdf_sha256,
        "tool_metadata_path": tool_metadata_path,
        "tool_metadata_sha256": tool_metadata_sha256,
        "robot_status": expected_robot_status,
        "scale": scale,
        "voxel_size_m": voxel_size,
        "maximum_render_points": maximum_points,
        "expected_valid_points": expected_points,
        "secondary_expected_valid_points": secondary_expected_points,
        "expected_combined_valid_points": expected_combined_points,
        "expected_unique_voxels_before_cap": expected_unique_voxels,
        "point_width_m": point_width,
        "prim_path": prim_path,
        "translation_xyz_m": translation,
        "quaternion_xyzw": quaternion,
        "secondary_to_primary_rotation": secondary_to_primary_rotation,
        "secondary_to_primary_translation_mm": secondary_to_primary_translation_mm,
        "primary_to_secondary_rotation": source_to_target_rotation,
        "primary_to_secondary_translation_mm": source_to_target_translation_mm,
        "capture_registration_status": registration_status,
        "capture_registration_uncertainty_translation_mm": uncertainty_translation_mm,
        "capture_registration_uncertainty_rotation_deg": uncertainty_rotation_deg,
        "capture_registration_validation_thresholds": validation_thresholds,
        "scan_derived_drawer_geometry": drawer_geometry,
        "drawer_size_xy_mm": drawer_size_xy_mm,
        "drawer_center_xy_mm": drawer_center_xy_mm,
        "drawer_short_axis_yaw_deg": drawer_yaw_deg,
        "drawer_rim_height_mm": drawer_rim_height_mm,
        "drawer_foam_plane_mm": drawer_foam_plane_mm,
        "drawer_outline_prim_path": drawer_outline_prim_path,
        "drawer_outline_color_rgb": drawer_outline_color,
        "drawer_outline_opacity": drawer_outline_opacity,
        "drawer_outline_width_m": drawer_outline_width_m,
        "controller_tool_audit_path": controller_tool_audit_path,
        "controller_tool_audit_sha256": controller_tool_audit_sha256,
        "application_pin_baseline": application_baseline,
        "application_pin_axis": pin_axis,
        "application_clear_start": clear_start,
        "application_pinch": pinch,
        "application_specimen_near": specimen_near,
        "application_clear_pin_length_m": clear_pin_length_m,
        "application_pinch_to_specimen_m": pinch_to_specimen_m,
        "application_guide_geometry": application_guide_geometry,
        "application_parent_prim_path": application_parent_prim_path,
        "application_guide_root_prim_path": application_guide_root_prim_path,
        "application_clear_section_prim_path": application_clear_section_prim_path,
        "application_specimen_boundary_prim_path": (
            application_specimen_boundary_prim_path
        ),
        "application_pinch_marker_parent_prim_path": expected_link_paths[
            "gripper_tcp"
        ],
        "application_pinch_marker_prim_path": application_pinch_marker_prim_path,
        "application_pin_grasp_tcp_prim_path": expected_link_paths["pin_grasp_tcp"],
        "application_bare_section_color_rgb": bare_section_color,
        "application_bare_section_width_m": bare_section_width_m,
        "application_bare_section_opacity": bare_section_opacity,
        "application_pinch_marker_radius_m": pinch_marker_radius_m,
        "application_specimen_boundary_color_rgb": specimen_boundary_color,
        "application_specimen_boundary_radius_m": specimen_boundary_radius_m,
        "application_specimen_boundary_segments": specimen_boundary_segments,
        "application_specimen_boundary_width_m": specimen_boundary_width_m,
        "application_specimen_boundary_opacity": specimen_boundary_opacity,
        "tool_geometry_scope": geometry_scope,
        "joint_positions": joint_positions,
        "camera_eye_xyz_m": camera_eye,
        "camera_target_xyz_m": camera_target,
        "application_camera_eye_offset": application_camera_eye_offset,
        "application_camera_target_offset": application_camera_target_offset,
        "floor_top_z_m": floor_top,
    }


def add_application_pin_guide(stage: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Author hash-bound, visual-only pin and specimen-start guides."""

    from pxr import Sdf, UsdGeom, UsdPhysics

    required_existing_paths = (
        config["application_parent_prim_path"],
        config["application_pinch_marker_parent_prim_path"],
        config["application_pinch_marker_prim_path"],
        config["application_pin_grasp_tcp_prim_path"],
    )
    for prim_path in required_existing_paths:
        if not stage.GetPrimAtPath(prim_path).IsValid():
            raise RuntimeError(f"Imported application-guide prim is missing: {prim_path}")

    marker_parent = stage.GetPrimAtPath(
        config["application_pinch_marker_parent_prim_path"]
    )
    marker_translation = marker_parent.GetAttribute("xformOp:translate").Get()
    marker_translation = np.asarray(marker_translation, dtype=np.float64)
    if not np.allclose(
        marker_translation,
        config["application_pinch"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Imported blue TCP marker is not at the selected 10 mm midpoint"
        )
    pin_grasp_prim = stage.GetPrimAtPath(
        config["application_pin_grasp_tcp_prim_path"]
    )
    if pin_grasp_prim.GetParent() != marker_parent:
        raise RuntimeError("Imported pin-grasp TCP is not attached to the blue marker")
    pin_grasp_translation = np.asarray(
        pin_grasp_prim.GetAttribute("xformOp:translate").Get(),
        dtype=np.float64,
    )
    if not np.allclose(pin_grasp_translation, np.zeros(3), rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Imported pin-grasp TCP is offset from the blue marker")

    root_path = config["application_guide_root_prim_path"]
    if stage.GetPrimAtPath(root_path).IsValid():
        raise RuntimeError(f"Application guide path already exists: {root_path}")
    guide_root = UsdGeom.Xform.Define(stage, root_path)
    guide_root.CreatePurposeAttr().Set(UsdGeom.Tokens.default_)
    guide_root_prim = guide_root.GetPrim()
    guide_root_prim.CreateAttribute(
        "magi:visualOnly", Sdf.ValueTypeNames.Bool, custom=True
    ).Set(True)
    guide_root_prim.CreateAttribute(
        "magi:collisionQualified", Sdf.ValueTypeNames.Bool, custom=True
    ).Set(False)
    guide_root_prim.CreateAttribute(
        "magi:geometryStatus", Sdf.ValueTypeNames.String, custom=True
    ).Set("user_selected_10mm_application_baseline")

    geometry = config["application_guide_geometry"]
    bare_section = author_usd_visual_polyline(
        stage,
        config["application_clear_section_prim_path"],
        geometry["bare_section"],
        config["application_bare_section_color_rgb"],
        config["application_bare_section_width_m"],
        config["application_bare_section_opacity"],
        geometry_status="user_selected_10mm_clear_pin_section",
        usd_purpose="default",
    )
    specimen_boundary = author_usd_visual_polyline(
        stage,
        config["application_specimen_boundary_prim_path"],
        geometry["specimen_boundary"],
        config["application_specimen_boundary_color_rgb"],
        config["application_specimen_boundary_width_m"],
        config["application_specimen_boundary_opacity"],
        geometry_status="variable_specimen_near_boundary",
        usd_purpose="default",
    )

    authored_prims = (
        guide_root_prim,
        bare_section.GetPrim(),
        specimen_boundary.GetPrim(),
    )
    forbidden_apis = (
        UsdPhysics.CollisionAPI,
        UsdPhysics.RigidBodyAPI,
        UsdPhysics.MassAPI,
    )
    for prim in authored_prims:
        if any(prim.HasAPI(api) for api in forbidden_apis):
            raise RuntimeError(f"Application guide acquired a physics API: {prim.GetPath()}")
    marker_prim = stage.GetPrimAtPath(config["application_pinch_marker_prim_path"])
    if marker_prim.GetTypeName() != "Sphere" or any(
        marker_prim.HasAPI(api) for api in forbidden_apis
    ):
        raise RuntimeError("Imported blue pinch marker is not a visual-only sphere")
    marker_sphere = UsdGeom.Sphere(marker_prim)
    imported_marker_radius_m = float(marker_sphere.GetRadiusAttr().Get())
    marker_sphere.GetRadiusAttr().Set(config["application_pinch_marker_radius_m"])
    if not np.isclose(
        float(marker_sphere.GetRadiusAttr().Get()),
        config["application_pinch_marker_radius_m"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("Blue pinch-marker display radius override did not apply")

    return {
        "status": "user_selected_10mm_baseline_visual_only",
        "parent_frame": "onrobot_2fg7_origin",
        "parent_prim_path": config["application_parent_prim_path"],
        "visual_root_prim_path": root_path,
        "bare_section_prim_path": config["application_clear_section_prim_path"],
        "pinch_marker_prim_path": config["application_pinch_marker_prim_path"],
        "pin_grasp_tcp_prim_path": config["application_pin_grasp_tcp_prim_path"],
        "specimen_near_boundary_prim_path": config[
            "application_specimen_boundary_prim_path"
        ],
        "pin_axis_from_gripper_toward_specimen": config[
            "application_pin_axis"
        ].tolist(),
        "clear_section_start_xyz_from_2fg7_device_origin_m": config[
            "application_clear_start"
        ].tolist(),
        "pinch_xyz_from_2fg7_device_origin_m": config[
            "application_pinch"
        ].tolist(),
        "specimen_near_point_xyz_from_2fg7_device_origin_m": config[
            "application_specimen_near"
        ].tolist(),
        "clear_pin_length_m": config["application_clear_pin_length_m"],
        "pinch_to_specimen_m": config["application_pinch_to_specimen_m"],
        "pinch_location": "midpoint",
        "pinch_marker_source": "imported_blue_gripper_tcp_sphere",
        "pinch_marker_imported_radius_m": imported_marker_radius_m,
        "pinch_marker_display_radius_m": config[
            "application_pinch_marker_radius_m"
        ],
        "pinch_marker_radius_override_scope": "viewer_session_only",
        "specimen_boundary_radius_m": config[
            "application_specimen_boundary_radius_m"
        ],
        "specimen_boundary_segments": config[
            "application_specimen_boundary_segments"
        ],
        "specimen_geometry": "variable_not_fixed",
        "purpose": "guide",
        "usd_render_purpose": "default",
        "visual_only": True,
        "collision_qualified": False,
        "collision_api_authored": False,
        "rigid_body_api_authored": False,
        "mass_api_authored": False,
    }


def add_visual_scene(
    stage: Any,
    config: dict[str, Any],
    points: np.ndarray,
    colors: np.ndarray,
    drawer_outline_points: np.ndarray,
) -> tuple[list[str], dict[str, Any]]:
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

    floor_thickness = 0.02
    floor = UsdGeom.Cube.Define(stage, "/Workcell/Floor")
    floor.AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, config["floor_top_z_m"] - floor_thickness / 2.0)
    )
    floor.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, floor_thickness / 2.0))
    floor.GetSizeAttr().Set(2.0)
    floor.GetDisplayColorAttr().Set([Gf.Vec3f(0.10, 0.13, 0.18)])
    author_usd_points(
        stage,
        config["prim_path"],
        points,
        colors,
        config["point_width_m"],
    )
    drawer_outline = author_usd_visual_polyline(
        stage,
        config["drawer_outline_prim_path"],
        drawer_outline_points,
        config["drawer_outline_color_rgb"],
        config["drawer_outline_width_m"],
        config["drawer_outline_opacity"],
        geometry_status="provisional_outer_top_rim_only",
    )
    drawer_outline_prim = drawer_outline.GetPrim()
    if drawer_outline_prim.HasAPI(
        UsdPhysics.CollisionAPI
    ) or drawer_outline_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("Provisional drawer outline acquired a physics API")
    application_guide = add_application_pin_guide(stage, config)
    dome = UsdLux.DomeLight.Define(stage, "/Workcell/Lights/Dome")
    dome.CreateIntensityAttr(950.0)
    key = UsdLux.DistantLight.Define(stage, "/Workcell/Lights/Key")
    key.CreateIntensityAttr(2800.0)
    key.CreateAngleAttr(1.0)
    return [
        "/Workcell/Floor",
        config["prim_path"],
        config["drawer_outline_prim_path"],
        config["application_guide_root_prim_path"],
        config["application_clear_section_prim_path"],
        config["application_pinch_marker_prim_path"],
        config["application_specimen_boundary_prim_path"],
        "/Workcell/Lights/Dome",
        "/Workcell/Lights/Key",
    ], application_guide


def create_status_panel(
    ui: Any,
    raw_points: int,
    rendered_points: int,
    set_camera_view: Any,
) -> tuple[Any, dict[str, bool]]:
    state = {"stop_requested": False}
    window = ui.Window("TM5S Measured-Scale Workcell Twin", width=600, height=370)
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label(
                "FUSED REAL ZIVID SCANS - PROVISIONAL WATSON POSE",
                style={"color": 0xFF44AAFF, "font_size": 18},
            )
            ui.Label(f"{raw_points:,} measured points | {rendered_points:,} rendered points")
            ui.Label(
                "CYAN: 10 mm clear pin | BLUE: midpoint pinch TCP | "
                "ORANGE: variable specimen-near boundary",
                word_wrap=True,
            )
            ui.Label(
                "The coloured application geometry is a configurable visual guide, "
                "not scan-derived specimen geometry and not a collision model.",
                word_wrap=True,
            )
            ui.Label(
                "The two same-drawer captures are registered to each other and retain "
                "measured scale. Their pose relative to Watson is not measured. Visual "
                "reference only; no collision geometry, ROS, or robot-command path exists.",
                word_wrap=True,
            )
            with ui.HStack(height=34, spacing=8):
                ui.Button(
                    "Focus 10 mm pin guide",
                    clicked_fn=lambda: set_camera_view("application"),
                )
                ui.Button(
                    "Show whole workcell",
                    clicked_fn=lambda: set_camera_view("workcell"),
                )

            def request_stop() -> None:
                state["stop_requested"] = True

            ui.Button("Stop and close viewer", height=34, clicked_fn=request_stop)
    return window, state


def bounds(points: np.ndarray) -> dict[str, list[float]]:
    return {
        "minimum_xyz_m": np.min(points, axis=0).tolist(),
        "maximum_xyz_m": np.max(points, axis=0).tolist(),
        "extent_xyz_m": np.ptp(points, axis=0).tolist(),
    }


def quaternion_angular_error_radians(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != (4,) or second.shape != (4,):
        raise ValueError("Rendered-link quaternions must be four-vectors")
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 0.0 or second_norm <= 0.0:
        raise ValueError("Rendered-link quaternions must be non-zero")
    dot = abs(float(np.dot(first / first_norm, second / second_norm)))
    return 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))


def rendered_link_pose(
    stage: Any,
    robot: Any,
    *,
    asset_prim_path: str,
    link_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read matching PhysX and rendered-USD world poses for one robot link."""

    from pxr import Usd, UsdGeom

    body_names = tuple(robot._articulation_view.body_names)
    try:
        body_index = body_names.index(link_name)
    except ValueError as exc:
        raise RuntimeError(
            f"Imported articulation has no {link_name!r} body: {list(body_names)}"
        ) from exc
    physics_transforms = np.asarray(
        robot._articulation_view._physics_view.get_link_transforms(), dtype=np.float64
    )
    if physics_transforms.shape != (1, robot.num_bodies, 7):
        raise RuntimeError(
            f"Unexpected PhysX link-transform shape: {physics_transforms.shape}"
        )
    physics_pose = physics_transforms[0, body_index]

    asset_prim = stage.GetPrimAtPath(asset_prim_path)
    visual_candidates = [
        prim for prim in Usd.PrimRange(asset_prim) if prim.GetName() == link_name
    ]
    if len(visual_candidates) != 1:
        raise RuntimeError(
            f"Expected one rendered {link_name!r} prim below {asset_prim_path}; "
            f"found {[str(prim.GetPath()) for prim in visual_candidates]}"
        )
    matrix = UsdGeom.Xformable(visual_candidates[0]).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    translation = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    rendered_position = np.asarray(
        [translation[0], translation[1], translation[2]], dtype=np.float64
    )
    rendered_orientation = np.asarray(
        [imaginary[0], imaginary[1], imaginary[2], rotation.GetReal()], dtype=np.float64
    )
    return physics_pose[:3], physics_pose[3:], rendered_position, rendered_orientation


def main() -> int:
    args = build_parser().parse_args()
    package_versions = validate_runtime()
    if args.duration_seconds < 0.0:
        raise ValueError("--duration-seconds must be non-negative")
    if args.headless and args.duration_seconds <= 0.0:
        raise ValueError("--headless requires a positive --duration-seconds")

    usd_path = args.usd.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(f"Imported TM5S USD is missing: {usd_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Workcell config is missing: {config_path}")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite workcell report: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    screenshot_path = args.screenshot.expanduser().resolve() if args.screenshot else None
    if screenshot_path is not None:
        if screenshot_path.exists():
            raise FileExistsError(f"Refusing to overwrite screenshot: {screenshot_path}")
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_workcell_config(config_path)
    if usd_path != config["robot_asset_path"]:
        raise ValueError(
            "--usd must match the hash-bound robot asset in workcell_scan.yaml: "
            f"expected={config['robot_asset_path']}, actual={usd_path}"
        )
    if sha256_file(usd_path) != config["robot_asset_sha256"]:
        raise ValueError("Selected robot USD no longer matches its validated hash")
    source_points, source_colors = load_ply_xyzrgb(config["point_cloud_path"])
    if source_points.shape[0] != config["expected_valid_points"]:
        raise ValueError(
            f"Expected {config['expected_valid_points']} points; found {source_points.shape[0]}"
        )
    secondary_points, secondary_colors = load_ply_xyzrgb(
        config["secondary_point_cloud_path"]
    )
    if secondary_points.shape[0] != config["secondary_expected_valid_points"]:
        raise ValueError(
            f"Expected {config['secondary_expected_valid_points']} secondary points; "
            f"found {secondary_points.shape[0]}"
        )
    secondary_in_primary_mm = transform_points_matrix(
        secondary_points,
        scale=1.0,
        translation_xyz=config["secondary_to_primary_translation_mm"],
        rotation_matrix=config["secondary_to_primary_rotation"],
    )
    fused_source_points_mm = np.vstack((source_points, secondary_in_primary_mm))
    fused_source_colors = np.vstack((source_colors, secondary_colors))
    if fused_source_points_mm.shape[0] != config["expected_combined_valid_points"]:
        raise ValueError(
            "Combined valid-point count disagrees with provenance: "
            f"expected={config['expected_combined_valid_points']}, "
            f"actual={fused_source_points_mm.shape[0]}"
        )
    primary_in_secondary_mm = transform_points_matrix(
        source_points,
        scale=1.0,
        translation_xyz=config["primary_to_secondary_translation_mm"],
        rotation_matrix=config["primary_to_secondary_rotation"],
    )
    canonical_voxel_points, _ = deterministic_voxel_downsample(
        np.vstack((primary_in_secondary_mm, secondary_points)),
        fused_source_colors,
        voxel_size_m=0.5,
        max_points=None,
    )
    if canonical_voxel_points.shape[0] != config["expected_unique_voxels_before_cap"]:
        raise ValueError(
            "Canonical fused-voxel count disagrees with provenance: "
            f"expected={config['expected_unique_voxels_before_cap']}, "
            f"actual={canonical_voxel_points.shape[0]}"
        )
    del canonical_voxel_points, primary_in_secondary_mm
    transformed_points = transform_points(
        fused_source_points_mm,
        config["scale"],
        config["translation_xyz_m"],
        config["quaternion_xyzw"],
    )
    voxel_points, voxel_colors = deterministic_voxel_downsample(
        transformed_points,
        fused_source_colors,
        config["voxel_size_m"],
        max_points=None,
    )
    render_points, render_colors = deterministic_even_point_cap(
        voxel_points,
        voxel_colors,
        config["maximum_render_points"],
    )
    if render_points.shape[0] > config["maximum_render_points"]:
        raise RuntimeError("Point-cloud downsampling exceeded maximum_render_points")
    drawer_outline_points = drawer_top_rim_outline_points(
        center_xy_mm=config["drawer_center_xy_mm"],
        size_xy_mm=config["drawer_size_xy_mm"],
        short_axis_yaw_deg=config["drawer_short_axis_yaw_deg"],
        rim_height_above_foam_mm=config["drawer_rim_height_mm"],
        foam_plane_z_coefficients_mm=config["drawer_foam_plane_mm"],
        secondary_to_primary_rotation=config["secondary_to_primary_rotation"],
        secondary_to_primary_translation_mm=config[
            "secondary_to_primary_translation_mm"
        ],
        scale_to_metres=config["scale"],
        primary_to_base_translation_m=config["translation_xyz_m"],
        primary_to_base_quaternion_xyzw=config["quaternion_xyzw"],
    )

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "hide_ui": args.headless,
            "width": 1280,
            "height": 720,
            "window_width": 1600,
            "window_height": 900,
            "renderer": "RaytracedLighting",
            "active_gpu": 0,
            "physics_gpu": 0,
            "multi_gpu": False,
            "max_gpu_count": 1,
            "fast_shutdown": True,
            "disable_viewport_updates": False,
            "open_usd": str(usd_path),
        }
    )
    exit_code = 1
    world = None
    panel_window = None
    started_wall_time = time.perf_counter()
    try:
        import carb
        import omni.ui as ui
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.rendering_manager import ViewportManager
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        from omni.physx import get_physx_interface
        from pxr import Gf, Usd, UsdGeom

        World.clear_instance()
        # Imported collision proxies also use USD purpose=guide. Keep that
        # global category hidden and render our non-physical application
        # overlays with purpose=default plus explicit magi:visualOnly metadata.
        carb.settings.get_settings().set_bool(GUIDE_PURPOSE_DISPLAY_SETTING, False)
        if carb.settings.get_settings().get_as_bool(GUIDE_PURPOSE_DISPLAY_SETTING):
            raise RuntimeError("Isaac viewport collision-guide display could not be disabled")
        stage = omni.usd.get_context().get_stage()
        default_prim = stage.GetDefaultPrim()
        if not default_prim.IsValid():
            raise RuntimeError("Imported TM5S stage has no default prim")
        asset_prim_path = str(default_prim.GetPath())
        scene_paths, application_pin_guide = add_visual_scene(
            stage,
            config,
            render_points,
            render_colors,
            drawer_outline_points,
        )
        world = World(
            physics_dt=RENDER_DT_SECONDS,
            rendering_dt=RENDER_DT_SECONDS,
            stage_units_in_meters=1.0,
            backend="numpy",
            device="cpu",
        )
        robot = world.scene.add(
            SingleArticulation(prim_path=asset_prim_path, name="tm5s_workcell_twin")
        )
        world.reset()
        world.pause()
        if not robot.handles_initialized:
            raise RuntimeError("TM5S articulation handles did not initialise")
        if robot.num_dof != 6 or list(robot.dof_names) != EXPECTED_DOF_NAMES:
            raise RuntimeError(
                f"Expected TM5S DOFs {EXPECTED_DOF_NAMES}; found {list(robot.dof_names)}"
            )
        joint_limits = np.column_stack(
            (
                np.asarray(robot.dof_properties["lower"], dtype=np.float64),
                np.asarray(robot.dof_properties["upper"], dtype=np.float64),
            )
        )
        display_positions = config["joint_positions"]
        if np.any(display_positions < joint_limits[:, 0]) or np.any(
            display_positions > joint_limits[:, 1]
        ):
            raise RuntimeError("Configured display pose exceeds imported TM5S joint limits")
        zeros = np.zeros(6, dtype=np.float64)
        robot.set_joints_default_state(positions=display_positions, velocities=zeros)
        robot.set_joint_positions(display_positions)
        robot.set_joint_velocities(zeros)
        world.physics_sim_view.update_articulations_kinematic()
        get_physx_interface().update_transformations(False, True, False)
        joint_readback = np.asarray(robot.get_joint_positions(), dtype=np.float64)
        joint_readback_error = float(np.max(np.abs(joint_readback - display_positions)))
        if joint_readback_error > MAX_TRACKING_ERROR_RADIANS:
            raise RuntimeError(
                "Static display joint readback exceeded tolerance: "
                f"{joint_readback_error}rad"
            )
        (
            physics_link_position,
            physics_link_orientation,
            rendered_link_position,
            rendered_link_orientation,
        ) = rendered_link_pose(
            stage,
            robot,
            asset_prim_path=asset_prim_path,
            link_name="link_6",
        )
        rendered_link_position_error = float(
            np.linalg.norm(rendered_link_position - physics_link_position)
        )
        rendered_link_orientation_error = quaternion_angular_error_radians(
            rendered_link_orientation, physics_link_orientation
        )
        if rendered_link_position_error > MAX_RENDERED_LINK_POSITION_ERROR_METERS:
            raise RuntimeError(
                "Rendered link_6 position disagrees with PhysX by "
                f"{rendered_link_position_error}m"
            )
        if rendered_link_orientation_error > MAX_RENDERED_LINK_ORIENTATION_ERROR_RADIANS:
            raise RuntimeError(
                "Rendered link_6 orientation disagrees with PhysX by "
                f"{rendered_link_orientation_error}rad"
            )

        device_origin_matrix = UsdGeom.Xformable(
            stage.GetPrimAtPath(config["application_parent_prim_path"])
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        def application_local_to_world(local_xyz: np.ndarray) -> np.ndarray:
            local = config["application_pinch"] + local_xyz
            transformed = device_origin_matrix.Transform(Gf.Vec3d(*local.tolist()))
            return np.asarray(transformed, dtype=np.float64)

        application_camera_eye_world = application_local_to_world(
            config["application_camera_eye_offset"]
        )
        application_camera_target_world = application_local_to_world(
            config["application_camera_target_offset"]
        )

        def set_camera_view(view_name: str) -> None:
            if view_name == "application":
                eye = application_camera_eye_world
                target = application_camera_target_world
            elif view_name == "workcell":
                eye = config["camera_eye_xyz_m"]
                target = config["camera_target_xyz_m"]
            else:
                raise ValueError(f"Unknown camera view: {view_name}")
            ViewportManager.set_camera_view(
                ViewportManager.get_camera(),
                eye=eye.tolist(),
                target=target.tolist(),
            )

        render = not args.headless or screenshot_path is not None
        if render:
            viewport_ready, waited_frames = ViewportManager.wait_for_viewport(max_frames=120)
            if not viewport_ready:
                raise RuntimeError(f"Isaac viewport was not ready after {waited_frames} frames")
            set_camera_view(args.camera_view)
        if args.headless:
            panel_state = {"stop_requested": False}
        else:
            panel_window, panel_state = create_status_panel(
                ui,
                fused_source_points_mm.shape[0],
                render_points.shape[0],
                set_camera_view,
            )

        loop_started = time.perf_counter()
        rendered_frames = 0
        screenshot_requested = False
        screenshot_capture = None
        screenshot_future = None
        while simulation_app.is_running() and not panel_state["stop_requested"]:
            frame_started = time.perf_counter()
            world.render()
            rendered_frames += 1
            elapsed = time.perf_counter() - loop_started
            if screenshot_path is not None and elapsed >= 1.0 and not screenshot_requested:
                viewport = get_active_viewport()
                if viewport is None:
                    raise RuntimeError("No active viewport is available for screenshot")
                screenshot_capture = capture_viewport_to_file(
                    viewport, file_path=str(screenshot_path)
                )
                screenshot_future = asyncio.ensure_future(
                    screenshot_capture.wait_for_result(completion_frames=30)
                )
                screenshot_requested = True
            if args.duration_seconds > 0.0 and elapsed >= args.duration_seconds:
                break
            remaining = RENDER_DT_SECONDS - (time.perf_counter() - frame_started)
            if remaining > 0.0:
                time.sleep(remaining)

        screenshot_wait_result = None
        if screenshot_path is not None and screenshot_future is None:
            raise RuntimeError("Screenshot was requested but the viewer ended before capture time")
        if screenshot_future is not None:
            for _ in range(180):
                if screenshot_future.done() or not simulation_app.is_running():
                    break
                world.render()
            if not screenshot_future.done():
                raise RuntimeError("Screenshot capture did not finish within 180 frames")
            screenshot_wait_result = bool(screenshot_future.result())
            if not screenshot_wait_result:
                raise RuntimeError("Isaac screenshot capture returned failure")
        if rendered_frames <= 0:
            raise RuntimeError("Viewer ended without rendering a frame")
        screenshot_written = screenshot_path.is_file() if screenshot_path is not None else None
        if screenshot_path is not None and not screenshot_written:
            raise RuntimeError(f"Screenshot capture did not complete: {screenshot_path}")
        screenshot_sha256 = sha256_file(screenshot_path) if screenshot_path is not None else None
        screenshot_size = png_dimensions(screenshot_path) if screenshot_path is not None else None

        report = {
            "format_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "mode": "static_visual_workcell_twin",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "package_versions": package_versions,
            "usd": str(usd_path),
            "usd_sha256": sha256_file(usd_path),
            "robot_model": {
                "asset_profile": config["robot_asset_profile"],
                "source_urdf": str(config["robot_source_urdf_path"]),
                "source_urdf_sha256": config["robot_source_urdf_sha256"],
                "asset_artifact_count": len(config["robot_asset_artifacts"]),
                "asset_artifacts": config["robot_asset_artifacts"],
                "import_report": str(config["robot_import_report_path"]),
                "import_report_sha256": config["robot_import_report_sha256"],
                "tool_metadata": str(config["tool_metadata_path"]),
                "tool_metadata_sha256": config["tool_metadata_sha256"],
                **config["robot_status"],
            },
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "provenance_manifest": str(config["manifest_path"]),
            "provenance_manifest_sha256": config["manifest_sha256"],
            "point_cloud": str(config["point_cloud_path"]),
            "point_cloud_sha256": config["point_cloud_sha256"],
            "secondary_point_cloud": str(config["secondary_point_cloud_path"]),
            "secondary_point_cloud_sha256": config[
                "secondary_point_cloud_sha256"
            ],
            "primary_source_points": int(source_points.shape[0]),
            "secondary_source_points": int(secondary_points.shape[0]),
            "source_points": int(fused_source_points_mm.shape[0]),
            "canonical_unique_voxels_before_cap": config[
                "expected_unique_voxels_before_cap"
            ],
            "viewer_frame_unique_voxels_before_cap": int(voxel_points.shape[0]),
            "rendered_points": int(render_points.shape[0]),
            "source_bounds_after_transform": bounds(transformed_points),
            "rendered_bounds": bounds(render_points),
            "scan_to_base": {
                "translation_xyz_m": config["translation_xyz_m"].tolist(),
                "quaternion_xyzw": config["quaternion_xyzw"].tolist(),
                "registration_status": "provisional_not_measured",
            },
            "capture_registration": {
                "status": config["capture_registration_status"],
                "secondary_to_primary_rotation": config[
                    "secondary_to_primary_rotation"
                ].tolist(),
                "secondary_to_primary_translation_mm": config[
                    "secondary_to_primary_translation_mm"
                ].tolist(),
                "conservative_uncertainty_translation_mm": config[
                    "capture_registration_uncertainty_translation_mm"
                ],
                "conservative_uncertainty_rotation_deg": config[
                    "capture_registration_uncertainty_rotation_deg"
                ],
                "validation_thresholds": config[
                    "capture_registration_validation_thresholds"
                ],
                "validation_gates_passed": True,
            },
            "scan_derived_drawer_geometry": config[
                "scan_derived_drawer_geometry"
            ],
            "drawer_top_rim_visual": {
                "prim_path": config["drawer_outline_prim_path"],
                "points_in_base_m": drawer_outline_points.tolist(),
                "purpose": "guide",
                "visual_only": True,
                "collision_qualified": False,
                "collision_api_authored": False,
                "rigid_body_api_authored": False,
            },
            "application_pin_guide": {
                **application_pin_guide,
                "global_usd_guide_category_display_enabled": False,
                "tool_metadata": str(config["tool_metadata_path"]),
                "tool_metadata_sha256": config["tool_metadata_sha256"],
                "geometry_scope": config["tool_geometry_scope"],
            },
            "controller_tool_audit": str(config["controller_tool_audit_path"]),
            "controller_tool_audit_sha256": config[
                "controller_tool_audit_sha256"
            ],
            "point_width_m": config["point_width_m"],
            "voxel_size_m": config["voxel_size_m"],
            "scene_paths": scene_paths,
            "asset_prim_path": asset_prim_path,
            "joint_names": EXPECTED_DOF_NAMES,
            "display_joint_positions": display_positions.tolist(),
            "joint_readback_positions": joint_readback.tolist(),
            "maximum_joint_readback_error_radians": joint_readback_error,
            "physics_link_6_position_meters": physics_link_position.tolist(),
            "rendered_link_6_position_meters": rendered_link_position.tolist(),
            "rendered_link_6_position_error_meters": rendered_link_position_error,
            "rendered_link_6_orientation_error_radians": rendered_link_orientation_error,
            "headless": args.headless,
            "initial_camera_view": args.camera_view,
            "application_camera_eye_world_m": application_camera_eye_world.tolist(),
            "application_camera_target_world_m": (
                application_camera_target_world.tolist()
            ),
            "rendered_frames": rendered_frames,
            "elapsed_wall_seconds": time.perf_counter() - started_wall_time,
            "screenshot": str(screenshot_path) if screenshot_path else None,
            "screenshot_written": screenshot_written,
            "screenshot_sha256": screenshot_sha256,
            "screenshot_dimensions_pixels": screenshot_size,
            "screenshot_capture_scheduled": screenshot_capture is not None,
            "screenshot_wait_result": screenshot_wait_result,
            "visual_only": True,
            "collision_enabled": False,
            "registered_to_watson_base": False,
            "captures_registered_to_each_other": True,
            "captures_stitched_visual_only": True,
            "ros_used": False,
            "watson_connected": False,
            "real_robot_commanded": False,
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        exit_code = 0
    except Exception:
        traceback.print_exc()
    finally:
        panel_window = None
        if world is not None:
            try:
                world.stop()
            except Exception:
                pass
        simulation_app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
