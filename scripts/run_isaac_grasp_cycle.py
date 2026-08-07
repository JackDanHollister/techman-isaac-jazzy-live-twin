#!/usr/bin/env python3
"""Run the full Isaac-only 2FG7 pin pickup and replacement choreography.

The arm follows an existing deterministic six-joint cuMotion choreography. A
separate articulated Watson/QC/2FG7 presentation asset opens and closes both
inward fingers. The pin/specimen is attached to the application TCP with a
scripted visual transform after closure, carried through lift/replacement, and
released before the fingers reopen. Contact, friction, force, and physical
grasp success are deliberately not claimed. No ROS graph, network connection,
controller, or Watson command is created.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import struct
import sys
import time
import traceback
from typing import Any

import numpy as np
import yaml

from pin_axis_3d_sim.grasp_cycle import (
    ARM_JOINT_NAMES,
    PHASE_ORDER,
    build_grasp_cycle,
    cycle_evidence,
    float64_sha256,
)


ARENA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ARENA_DIR / "config/isaac_grasp_cycle.yaml"
DEFAULT_REPORT = ARENA_DIR / "outputs/isaac_sim/6.0.1/grasp_cycle_report.json"
EXPECTED_PYTHON = (3, 12)
EXPECTED_ISAAC_PACKAGES = {
    "isaacsim": "6.0.1.0",
    "isaacsim-asset": "6.0.1.0",
    "isaacsim-core": "6.0.1.0",
}
MAX_KINEMATIC_READBACK_ERROR = 1.0e-6
MAX_FINGER_MIRROR_ERROR_M = 1.0e-7
MAX_RELEASE_POSITION_ERROR_M = 1.0e-6
MAX_RELEASE_ORIENTATION_ERROR_RAD = 1.0e-6
IMPORTED_LIMIT_TOLERANCE = 1.0e-7
MIN_LIFT_DISPLACEMENT_M = 0.02
GUIDE_PURPOSE_DISPLAY_SETTING = "/persistent/app/hydra/displayPurpose/guide"
WHOLE_WORKCELL_CAMERA_EYE = [1.35, 1.20, 1.05]
WHOLE_WORKCELL_CAMERA_TARGET = [0.25, 0.0, 0.40]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--trajectory-export",
        type=Path,
        default=None,
        help="Arm-only simulation-preview export (default: beside the report).",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Stop after this many complete cycles; zero runs until the window closes.",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Do not pace simulation time to wall time; intended for headless proof runs.",
    )
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument(
        "--camera-view",
        choices=["grasp", "workcell"],
        default="grasp",
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
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Screenshot is not a valid PNG: {path}")
    return list(struct.unpack(">II", header[16:24]))


def validate_runtime() -> dict[str, str]:
    if sys.version_info[:2] != EXPECTED_PYTHON:
        raise RuntimeError(
            f"Isaac grasp cycle requires Python 3.12; found {sys.version.split()[0]}"
        )
    versions: dict[str, str] = {}
    for package_name, expected in EXPECTED_ISAAC_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Missing required package: {package_name}") from exc
        if actual != expected:
            raise RuntimeError(f"{package_name} must be {expected}; found {actual}")
        versions[package_name] = actual
    return versions


def contained_artifact(relative_path: str, label: str) -> Path:
    path = (ARENA_DIR / relative_path).resolve()
    try:
        path.relative_to(ARENA_DIR.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must remain inside the demo directory: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def finite_vector(value: Any, length: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain {length} finite values")
    return result


def validate_source_plan(plan: dict[str, Any]) -> None:
    if plan.get("format_version") != 1:
        raise ValueError("Source choreography plan must use format_version 1")
    if plan.get("ros_used") is not False:
        raise ValueError("Source choreography must explicitly record ros_used false")
    if plan.get("watson_connected") is not False or plan.get("real_robot_commanded") is not False:
        raise ValueError("Source choreography must explicitly record no Watson connection or command")
    if plan.get("planning_tool_frame") != "flange":
        raise ValueError("Source choreography must plan the six-axis flange frame")
    stages = plan.get("selected", {}).get("stages", [])
    if [stage.get("name") for stage in stages] != ["pregrasp", "grasp", "lift"]:
        raise ValueError("Source choreography stages must be pregrasp, grasp, lift")
    previous_endpoint: np.ndarray | None = None
    for stage in stages:
        samples = stage.get("control_samples", [])
        positions = [finite_vector(sample.get("joint_positions"), 6, "joint positions") for sample in samples]
        velocities = [finite_vector(sample.get("joint_velocities"), 6, "joint velocities") for sample in samples]
        if len(samples) < 2:
            raise ValueError(f"Source stage has too few samples: {stage.get('name')}")
        if float64_sha256(positions + velocities) != stage.get("control_samples_float64_sha256"):
            raise ValueError(f"Source stage sample hash mismatch: {stage.get('name')}")
        if previous_endpoint is not None and not np.allclose(
            positions[0], previous_endpoint, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(f"Source stage is discontinuous: {stage.get('name')}")
        previous_endpoint = positions[-1]


def validate_asset_artifacts(
    import_report: dict[str, Any],
    *,
    repository_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    output_dir = Path(import_report["output_directory"]).expanduser()
    if not output_dir.is_absolute() and repository_root is not None:
        output_dir = repository_root / output_dir
    output_dir = output_dir.resolve()
    if repository_root is not None:
        try:
            output_dir.relative_to(repository_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Imported asset directory escapes the repository: {output_dir}"
            ) from exc
    artifacts = import_report.get("asset_artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Articulated import report has no asset artifact manifest")
    validated: dict[str, dict[str, Any]] = {}
    for relative_path, evidence in sorted(artifacts.items()):
        artifact_path = (output_dir / relative_path).resolve()
        try:
            artifact_path.relative_to(output_dir)
        except ValueError as exc:
            raise ValueError(f"Imported artifact escapes its output directory: {relative_path}") from exc
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Imported artifact is missing: {artifact_path}")
        if artifact_path.stat().st_size != int(evidence["size_bytes"]):
            raise ValueError(f"Imported artifact size mismatch: {artifact_path}")
        actual_hash = sha256_file(artifact_path)
        if actual_hash != evidence["sha256"]:
            raise ValueError(f"Imported artifact hash mismatch: {artifact_path}")
        validated[relative_path] = {
            "path": str(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
            "sha256": actual_hash,
        }
    return validated


def load_and_validate_config(config_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("format_version") != 1:
        raise ValueError("Isaac grasp-cycle config must use format_version 1")
    scope = raw.get("scope", {})
    expected_false = (
        "ros_used",
        "watson_connected",
        "real_robot_commanded",
        "contact_physics_simulated",
        "physical_camera_or_depth_used",
    )
    if any(scope.get(field) is not False for field in expected_false):
        raise ValueError("Isaac grasp-cycle scope must explicitly disable ROS, Watson, and contact")

    choreography = raw["arm_choreography"]
    plan_path = contained_artifact(choreography["source_plan"], "source arm choreography")
    plan_hash = sha256_file(plan_path)
    if plan_hash != choreography["source_plan_sha256"]:
        raise ValueError("Source arm choreography hash mismatch")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_source_plan(plan)
    if choreography.get("source_profile") != "legacy_cad_dry_run":
        raise ValueError("The current reviewed arm choreography must retain its legacy source label")
    if "applies_only" not in str(choreography.get("collision_clearance_status", "")):
        raise ValueError("Current-QC collision limitations must remain explicit")

    asset = raw["articulated_asset"]
    usd_path = contained_artifact(asset["usd"], "articulated Isaac USD")
    import_report_path = contained_artifact(asset["import_report"], "articulated import report")
    manifest_path = contained_artifact(asset["staged_manifest"], "articulated staging manifest")
    metadata_path = contained_artifact(asset["tool_metadata"], "articulated tool metadata")
    if sha256_file(import_report_path) != asset["import_report_sha256"]:
        raise ValueError("Articulated import report hash mismatch")
    if sha256_file(manifest_path) != asset["staged_manifest_sha256"]:
        raise ValueError("Articulated staging manifest hash mismatch")
    if sha256_file(metadata_path) != asset["tool_metadata_sha256"]:
        raise ValueError("Articulated tool metadata hash mismatch")

    import_report = json.loads(import_report_path.read_text(encoding="utf-8"))
    expected_names = list(asset["expected_dof_names"])
    if import_report.get("validation_profile") != asset["profile"]:
        raise ValueError("Articulated import profile mismatch")
    if import_report.get("dof_count") != 8 or import_report.get("physx_dof_names") != expected_names:
        raise ValueError("Articulated import report does not contain the expected eight DOFs")
    if Path(import_report["output_usd"]).resolve() != usd_path:
        raise ValueError("Articulated import report references a different root USD")
    if sha256_file(usd_path) != import_report["output_usd_sha256"]:
        raise ValueError("Articulated root USD hash mismatch")
    source_urdf_path = Path(import_report["source_urdf"]).resolve()
    if not source_urdf_path.is_file() or sha256_file(source_urdf_path) != import_report["source_urdf_sha256"]:
        raise ValueError("Articulated source URDF hash mismatch")
    asset_artifacts = validate_asset_artifacts(import_report)

    staged_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if staged_manifest.get("asset_mode") != "isaac_articulated":
        raise ValueError("Staged model is not marked as an Isaac articulated asset")
    if staged_manifest.get("moving_joints") != expected_names:
        raise ValueError("Staged articulated joint order mismatch")
    if staged_manifest.get("xrdf") is not None:
        raise ValueError("Isaac articulated staging must not masquerade as a cuMotion XRDF bundle")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("tool_profile") != "watson_qc_nominal":
        raise ValueError("Articulated presentation asset must use the Watson QC profile")
    if metadata.get("finger_joints") != "prismatic" or metadata.get("finger_configuration") != "inwards":
        raise ValueError("Articulated presentation asset must use inward prismatic fingers")
    if metadata.get("finger_position_m") != 0.0:
        raise ValueError("Articulated source must use zero baked finger offset")
    baseline = metadata.get("application_pin_baseline", {})
    if not math.isclose(float(baseline.get("clear_pin_length_before_specimen_m")), 0.010):
        raise ValueError("Articulated tool metadata lost the 10 mm application baseline")
    if not math.isclose(float(baseline.get("pinch_to_specimen_m")), 0.005):
        raise ValueError("Articulated tool metadata lost the 5 mm pinch midpoint")
    if metadata.get("frame_xyz_from_flange_m", {}).get("pin_grasp_tcp") != [0.0, 0.0, 0.17085]:
        raise ValueError("Articulated tool metadata lost the provisional flange-to-pinch frame")

    gripper = raw["gripper_motion"]
    finger_motion = metadata["finger_motion"]
    expected_motion = {
        "open_position_m": 0.0,
        "closed_position_m": float(finger_motion["per_finger_travel_m"]),
        "open_gap_m": float(finger_motion["external_gap_open_m"]),
        "closed_gap_m": float(finger_motion["external_gap_closed_m"]),
        "per_finger_speed_m_s": float(
            finger_motion["recommended_initial_simulation_per_finger_speed_m_s"]
        ),
    }
    for name, expected in expected_motion.items():
        if not math.isclose(float(gripper[name]), expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"Gripper motion config disagrees with metadata: {name}")
    if gripper["leader_joint"] != expected_names[6] or gripper["mimic_joint"] != expected_names[7]:
        raise ValueError("Configured finger leader/follower order is incorrect")
    imported_mimic = import_report.get("imported_mimic_joints", {}).get(gripper["mimic_joint"], {})
    if (
        imported_mimic.get("source_joint") != gripper["leader_joint"]
        or imported_mimic.get("multiplier") != 1.0
        or imported_mimic.get("offset") != 0.0
    ):
        raise ValueError("Isaac import did not preserve the exact 2FG7 mimic relationship")

    payload = raw["payload_visual"]
    if payload.get("attachment_mode") != "kinematic_visual_follow":
        raise ValueError("The first grasp cycle must retain the reviewed visual attachment mode")
    if not math.isclose(float(payload["clear_pin_length_m"]), 0.010):
        raise ValueError("Payload clear-pin section must remain exactly 10 mm")
    if not math.isclose(float(payload["pinch_to_specimen_m"]), 0.005):
        raise ValueError("Payload pinch must remain at the 10 mm section midpoint")
    if 2.0 * float(payload["pin_radius_m"]) >= float(gripper["closed_gap_m"]):
        raise ValueError("Configured visual pin diameter must fit within the closed jaw gap")
    specimen_scale = finite_vector(
        payload["specimen_scale_xyz_m"], 3, "specimen_scale_xyz_m"
    )
    specimen_near = (
        float(payload["specimen_center_z_from_pinch_m"]) - 0.5 * specimen_scale[2]
    )
    if not math.isclose(
        specimen_near,
        float(payload["specimen_near_z_from_pinch_m"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Visible specimen must begin exactly after the 10 mm clear section")
    head_near = (
        float(payload["pin_head_center_z_from_pinch_m"])
        + float(payload["pin_head_radius_m"])
    )
    if not math.isclose(
        head_near,
        float(payload["clear_start_z_from_pinch_m"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Visible pin head must meet the start of the clear shaft")

    viewer = raw["viewer"]
    render_hz = float(viewer["render_hz"])
    if not math.isfinite(render_hz) or render_hz <= 0.0:
        raise ValueError("Viewer render_hz must be finite and positive")
    if viewer.get("screenshot_phase") not in PHASE_ORDER:
        raise ValueError("Viewer screenshot phase is not part of the grasp cycle")
    finite_vector(
        viewer["grasp_camera_eye_offset_from_pinch_m"], 3, "grasp camera eye offset"
    )
    finite_vector(
        viewer["grasp_camera_target_offset_from_pinch_m"],
        3,
        "grasp camera target offset",
    )

    return {
        "raw": raw,
        "plan": plan,
        "plan_path": plan_path,
        "plan_sha256": plan_hash,
        "usd_path": usd_path,
        "import_report": import_report,
        "import_report_path": import_report_path,
        "import_report_sha256": sha256_file(import_report_path),
        "asset_artifacts": asset_artifacts,
        "staged_manifest": staged_manifest,
        "staged_manifest_path": manifest_path,
        "staged_manifest_sha256": sha256_file(manifest_path),
        "metadata": metadata,
        "metadata_path": metadata_path,
        "metadata_sha256": sha256_file(metadata_path),
        "expected_dof_names": expected_names,
        "gripper": gripper,
        "payload": payload,
        "viewer": viewer,
    }


def set_gprim_color(gprim: Any, color: tuple[float, float, float], opacity: float = 1.0) -> None:
    from pxr import Gf, UsdGeom

    gprim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*color)])
    gprim.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set([float(opacity)])


def add_payload_visual(stage: Any, payload: dict[str, Any]) -> tuple[Any, list[str]]:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    root_path = payload["root_prim_path"]
    if stage.GetPrimAtPath(root_path).IsValid():
        raise RuntimeError(f"Payload prim already exists: {root_path}")
    root = UsdGeom.Xform.Define(stage, root_path)
    root_op = root.AddTransformOp(precision=UsdGeom.XformOp.PrecisionDouble)
    root_op.Set(Gf.Matrix4d(1.0))

    clear_start = float(payload["clear_start_z_from_pinch_m"])
    specimen_near = float(payload["specimen_near_z_from_pinch_m"])
    remaining_end = float(payload["remaining_pin_end_z_from_pinch_m"])
    clear_length = specimen_near - clear_start
    if not math.isclose(clear_length, float(payload["clear_pin_length_m"]), abs_tol=1.0e-12):
        raise ValueError("Payload clear-section endpoints do not span exactly 10 mm")

    created = [root_path]
    clear = UsdGeom.Cylinder.Define(stage, f"{root_path}/ClearPin10mm")
    clear.CreateAxisAttr(UsdGeom.Tokens.z)
    clear.CreateRadiusAttr(float(payload["pin_radius_m"]))
    clear.CreateHeightAttr(clear_length)
    UsdGeom.Xformable(clear).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, 0.5 * (clear_start + specimen_near))
    )
    set_gprim_color(clear, (0.05, 0.85, 1.0))
    created.append(str(clear.GetPath()))

    remainder = UsdGeom.Cylinder.Define(stage, f"{root_path}/RemainingPin")
    remainder.CreateAxisAttr(UsdGeom.Tokens.z)
    remainder.CreateRadiusAttr(float(payload["pin_radius_m"]))
    remainder.CreateHeightAttr(remaining_end - specimen_near)
    UsdGeom.Xformable(remainder).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, 0.5 * (specimen_near + remaining_end))
    )
    set_gprim_color(remainder, (0.68, 0.72, 0.78))
    created.append(str(remainder.GetPath()))

    head = UsdGeom.Sphere.Define(stage, f"{root_path}/PinHead")
    head.CreateRadiusAttr(float(payload["pin_head_radius_m"]))
    UsdGeom.Xformable(head).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, float(payload["pin_head_center_z_from_pinch_m"]))
    )
    set_gprim_color(head, (0.12, 0.92, 0.35))
    created.append(str(head.GetPath()))

    specimen = UsdGeom.Sphere.Define(stage, f"{root_path}/Specimen")
    specimen.CreateRadiusAttr(0.5)
    specimen_xform = UsdGeom.Xformable(specimen)
    specimen_xform.AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, float(payload["specimen_center_z_from_pinch_m"]))
    )
    specimen_xform.AddScaleOp().Set(Gf.Vec3f(*finite_vector(
        payload["specimen_scale_xyz_m"], 3, "specimen_scale_xyz_m"
    ).tolist()))
    set_gprim_color(specimen, (0.72, 0.30, 0.08))
    created.append(str(specimen.GetPath()))

    for path in created:
        prim = stage.GetPrimAtPath(path)
        prim.CreateAttribute("magi:visualOnly", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
        prim.CreateAttribute("magi:collisionQualified", Sdf.ValueTypeNames.Bool, custom=True).Set(False)
        prim.CreateAttribute("magi:geometryStatus", Sdf.ValueTypeNames.String, custom=True).Set(
            str(payload["geometry_status"])
        )
        if (
            prim.HasAPI(UsdPhysics.CollisionAPI)
            or prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or prim.HasAPI(UsdPhysics.MassAPI)
        ):
            raise RuntimeError(f"Visual payload unexpectedly acquired a physics API: {path}")
    return root_op, created


def add_static_scene(stage: Any, plan: dict[str, Any]) -> list[str]:
    from isaacsim.core.api.objects import VisualCuboid, VisualSphere
    from pxr import UsdLux

    selected_truth_id = int(plan["selected"]["truth_id"])
    role_colors = {
        "foam": np.array([0.18, 0.34, 0.20]),
        "tray_wall": np.array([0.28, 0.31, 0.36]),
        "specimen_body": np.array([0.32, 0.16, 0.07]),
        "other_pin_shaft": np.array([0.62, 0.64, 0.68]),
        "other_pin_head": np.array([0.78, 0.12, 0.12]),
    }
    created: list[str] = []
    for index, obstacle in enumerate(plan["selected"]["collision_obstacles"]):
        if (
            obstacle["role"] == "specimen_body"
            and int(obstacle.get("source_id", -1)) == selected_truth_id
        ):
            continue
        path = f"/GraspCycle/Scene/Proxy_{index:02d}_{obstacle['role']}"
        color = role_colors[obstacle["role"]]
        if obstacle["type"] == "cuboid":
            qx, qy, qz, qw = obstacle["quaternion_xyzw"]
            VisualCuboid(
                prim_path=path,
                position=np.asarray(obstacle["position_xyz_m"], dtype=np.float64),
                orientation=np.array([qw, qx, qy, qz], dtype=np.float64),
                scale=np.asarray(obstacle["side_lengths_m"], dtype=np.float64),
                size=1.0,
                color=color,
            )
        else:
            VisualSphere(
                prim_path=path,
                position=np.asarray(obstacle["position_xyz_m"], dtype=np.float64),
                radius=float(obstacle["radius_m"]),
                color=color,
            )
        created.append(path)

    dome = UsdLux.DomeLight.Define(stage, "/GraspCycle/Lights/Dome")
    dome.CreateIntensityAttr(950.0)
    key = UsdLux.DistantLight.Define(stage, "/GraspCycle/Lights/Key")
    key.CreateIntensityAttr(3000.0)
    key.CreateAngleAttr(1.0)
    created.extend(["/GraspCycle/Lights/Dome", "/GraspCycle/Lights/Key"])
    return created


def prim_world_matrix(stage: Any, prim_path: str) -> Any:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Required transform prim is missing: {prim_path}")
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def matrix_translation(matrix: Any) -> np.ndarray:
    value = matrix.ExtractTranslation()
    return np.asarray([value[0], value[1], value[2]], dtype=np.float64)


def matrix_rotation_error_radians(first: Any, second: Any) -> float:
    first_quat = first.ExtractRotationQuat()
    second_quat = second.ExtractRotationQuat()
    first_array = np.asarray([first_quat.GetReal(), *first_quat.GetImaginary()], dtype=np.float64)
    second_array = np.asarray([second_quat.GetReal(), *second_quat.GetImaginary()], dtype=np.float64)
    first_array /= np.linalg.norm(first_array)
    second_array /= np.linalg.norm(second_array)
    return 2.0 * math.acos(float(np.clip(abs(np.dot(first_array, second_array)), -1.0, 1.0)))


def create_status_panel(ui: Any, set_camera_view: Any) -> tuple[Any, Any, dict[str, bool]]:
    state = {"stop_requested": False}
    window = ui.Window("Watson 2FG7 Pin Grasp Cycle", width=530, height=330)
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label(
                "ISAAC SIMULATION ONLY - WATSON NOT CONNECTED",
                style={"color": 0xFF5A5AFF, "font_size": 17},
            )
            ui.Label("Approach > close > lift > replace > release > retreat")
            status_label = ui.Label("Initialising articulated grasp cycle...")
            ui.Label(
                "Both inward 2FG7 jaw joints move from a 39 mm to 1 mm gap.\n"
                "The pin is centred on the provisional 10 mm bare section.\n"
                "Pickup uses visual TCP-follow attachment; contact/friction are not simulated.",
                word_wrap=True,
            )
            with ui.HStack(spacing=8, height=32):
                ui.Button("Focus grasp", clicked_fn=lambda: set_camera_view("grasp"))
                ui.Button("Whole workcell", clicked_fn=lambda: set_camera_view("workcell"))

            def request_stop() -> None:
                state["stop_requested"] = True

            ui.Button("Stop and close demo", height=32, clicked_fn=request_stop)
    return window, status_label, state


def write_arm_preview_export(
    path: Path,
    commands: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    evidence: dict[str, Any],
) -> None:
    control_dt = float(config["plan"]["control_dt_seconds"])
    payload = {
        "format_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "simulation_preview_only_not_a_watson_command",
        "source_plan": str(config["plan_path"]),
        "source_plan_sha256": config["plan_sha256"],
        "presentation_asset": str(config["usd_path"]),
        "presentation_asset_sha256": sha256_file(config["usd_path"]),
        "joint_names": list(ARM_JOINT_NAMES),
        "control_dt_seconds": control_dt,
        "cycle_evidence": evidence,
        "samples": [
            {
                "time_seconds": index * control_dt,
                "phase": command["phase"],
                "joint_positions": command["arm_positions"].tolist(),
                "joint_velocities": command["arm_velocities"].tolist(),
            }
            for index, command in enumerate(commands)
        ],
        "gripper_samples_excluded": True,
        "ros_used": False,
        "watson_connected": False,
        "real_robot_commanded": False,
        "hardware_gate": (
            "Replan and retime through the guarded Watson/MoveIt path after physical tool "
            "commissioning; never send this Isaac preview file directly to the controller."
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    package_versions = validate_runtime()
    config_path = args.config.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    trajectory_path = (
        args.trajectory_export.expanduser().resolve()
        if args.trajectory_export
        else report_path.with_name(f"{report_path.stem}_watson_preview.json")
    )
    screenshot_path = args.screenshot.expanduser().resolve() if args.screenshot else None
    if args.cycles < 0:
        raise ValueError("--cycles must be non-negative")
    target_cycles = 1 if args.headless and args.cycles == 0 else args.cycles
    if args.headless and target_cycles <= 0:
        raise ValueError("Headless grasp-cycle validation requires at least one complete cycle")
    for output_path, label in (
        (report_path, "report"),
        (trajectory_path, "trajectory export"),
        (screenshot_path, "screenshot"),
    ):
        if output_path is None:
            continue
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite {label}: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
    config = load_and_validate_config(config_path)
    gripper = config["gripper"]
    commands = build_grasp_cycle(
        config["plan"],
        finger_open_m=float(gripper["open_position_m"]),
        finger_closed_m=float(gripper["closed_position_m"]),
        finger_speed_m_s=float(gripper["per_finger_speed_m_s"]),
        hold_seconds=float(gripper["hold_seconds"]),
    )
    control_dt = float(config["plan"]["control_dt_seconds"])
    evidence = cycle_evidence(commands, control_dt)
    render_interval = max(
        1,
        int(round((1.0 / float(config["viewer"]["render_hz"])) / control_dt)),
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
            "open_usd": str(config["usd_path"]),
        }
    )
    exit_code = 1
    world = None
    panel_window = None
    started_wall = time.perf_counter()
    try:
        import carb
        import omni.ui as ui
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.rendering_manager import ViewportManager
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        from omni.physx import get_physx_interface

        World.clear_instance()
        carb.settings.get_settings().set_bool(GUIDE_PURPOSE_DISPLAY_SETTING, False)
        if carb.settings.get_settings().get_as_bool(GUIDE_PURPOSE_DISPLAY_SETTING):
            raise RuntimeError("Isaac viewport collision-guide display could not be disabled")
        stage = omni.usd.get_context().get_stage()
        default_prim = stage.GetDefaultPrim()
        if not default_prim.IsValid():
            raise RuntimeError("Articulated Watson stage has no default prim")
        asset_prim_path = str(default_prim.GetPath())
        scene_paths = add_static_scene(stage, config["plan"])
        payload_transform_op, payload_paths = add_payload_visual(stage, config["payload"])
        world = World(
            physics_dt=control_dt,
            rendering_dt=1.0 / float(config["viewer"]["render_hz"]),
            stage_units_in_meters=1.0,
            backend="numpy",
            device="cpu",
        )
        robot = world.scene.add(
            SingleArticulation(prim_path=asset_prim_path, name="watson_2fg7_grasp_cycle")
        )
        world.reset()
        world.pause()
        if not robot.handles_initialized:
            raise RuntimeError("Articulated Watson handles did not initialise")
        if list(robot.dof_names) != config["expected_dof_names"] or robot.num_dof != 8:
            raise RuntimeError(
                f"Expected articulated DOFs {config['expected_dof_names']}; found {list(robot.dof_names)}"
            )
        dof_index = {name: index for index, name in enumerate(robot.dof_names)}
        joint_limits = np.column_stack(
            (
                np.asarray(robot.dof_properties["lower"], dtype=np.float64),
                np.asarray(robot.dof_properties["upper"], dtype=np.float64),
            )
        )

        def full_state(command: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
            positions = np.zeros(8, dtype=np.float64)
            velocities = np.zeros(8, dtype=np.float64)
            for index, name in enumerate(ARM_JOINT_NAMES):
                positions[dof_index[name]] = command["arm_positions"][index]
                velocities[dof_index[name]] = command["arm_velocities"][index]
            finger_position = float(command["finger_position_m"])
            positions[dof_index[gripper["leader_joint"]]] = finger_position
            positions[dof_index[gripper["mimic_joint"]]] = finger_position
            return positions, velocities

        all_positions = np.asarray([full_state(command)[0] for command in commands])
        # PhysX exposes the URDF's 0.019 m finger limit as float32
        # 0.018999999389..., so use a tight representation tolerance here.
        below_limits = all_positions < joint_limits[:, 0] - IMPORTED_LIMIT_TOLERANCE
        above_limits = all_positions > joint_limits[:, 1] + IMPORTED_LIMIT_TOLERANCE
        if np.any(below_limits) or np.any(above_limits):
            invalid_dofs = []
            for index, name in enumerate(robot.dof_names):
                if np.any(below_limits[:, index]) or np.any(above_limits[:, index]):
                    invalid_dofs.append(
                        {
                            "joint": name,
                            "command_min": float(np.min(all_positions[:, index])),
                            "command_max": float(np.max(all_positions[:, index])),
                            "imported_lower": float(joint_limits[index, 0]),
                            "imported_upper": float(joint_limits[index, 1]),
                        }
                    )
            raise RuntimeError(
                "Grasp cycle exceeds an imported articulation limit: "
                f"{invalid_dofs}"
            )

        def apply_kinematic_state(positions: np.ndarray, velocities: np.ndarray) -> None:
            robot.set_joint_positions(positions)
            robot.set_joint_velocities(velocities)
            world.physics_sim_view.update_articulations_kinematic()
            get_physx_interface().update_transformations(False, True, False)

        grasp_command = next(
            command for command in commands if command["phase"] == "hold_grasp_open"
        )
        grasp_positions, grasp_velocities = full_state(grasp_command)
        robot.set_joints_default_state(positions=grasp_positions, velocities=grasp_velocities)
        apply_kinematic_state(grasp_positions, grasp_velocities)
        tcp_prim_path = config["import_report"]["expected_link_paths"]["pin_grasp_tcp"]
        initial_payload_matrix = prim_world_matrix(stage, tcp_prim_path)
        payload_transform_op.Set(initial_payload_matrix)
        initial_payload_position = matrix_translation(initial_payload_matrix)

        start_positions, start_velocities = full_state(commands[0])
        robot.set_joints_default_state(positions=start_positions, velocities=start_velocities)
        apply_kinematic_state(start_positions, start_velocities)

        camera_view_evidence: dict[str, Any] = {}

        def set_camera_view(view_name: str) -> None:
            if view_name == "grasp":
                from pxr import Gf

                tcp_matrix = prim_world_matrix(stage, tcp_prim_path)
                eye_offset = finite_vector(
                    config["viewer"]["grasp_camera_eye_offset_from_pinch_m"],
                    3,
                    "grasp camera eye offset",
                )
                target_offset = finite_vector(
                    config["viewer"]["grasp_camera_target_offset_from_pinch_m"],
                    3,
                    "grasp camera target offset",
                )
                eye_value = tcp_matrix.Transform(Gf.Vec3d(*eye_offset.tolist()))
                target_value = tcp_matrix.Transform(Gf.Vec3d(*target_offset.tolist()))
                eye = np.asarray(eye_value, dtype=np.float64)
                target = np.asarray(target_value, dtype=np.float64)
            elif view_name == "workcell":
                eye = np.asarray(WHOLE_WORKCELL_CAMERA_EYE, dtype=np.float64)
                target = np.asarray(WHOLE_WORKCELL_CAMERA_TARGET, dtype=np.float64)
            else:
                raise ValueError(f"Unknown camera view: {view_name}")
            camera_view_evidence.update(
                {
                    "view": view_name,
                    "eye_xyz_m": eye.tolist(),
                    "target_xyz_m": target.tolist(),
                }
            )
            print(
                f"Isaac camera {view_name}: eye={eye.tolist()} target={target.tolist()}"
            )
            ViewportManager.set_camera_view(
                ViewportManager.get_camera(), eye=eye.tolist(), target=target.tolist()
            )

        render_animation = not args.headless
        if render_animation:
            viewport_ready, waited_frames = ViewportManager.wait_for_viewport(max_frames=120)
            if not viewport_ready:
                raise RuntimeError(f"Isaac viewport was not ready after {waited_frames} frames")
            set_camera_view(args.camera_view)
        if args.headless:
            status_label = None
            panel_state = {"stop_requested": False}
        else:
            panel_window, status_label, panel_state = create_status_panel(ui, set_camera_view)

        step_count = 0
        command_index = 0
        completed_cycles = 0
        rendered_frames = 0
        attached = False
        attach_events = 0
        release_events = 0
        maximum_readback_error = 0.0
        maximum_arm_readback_error = 0.0
        maximum_finger_readback_error = 0.0
        maximum_finger_mirror_error = 0.0
        observed_finger_min = math.inf
        observed_finger_max = -math.inf
        maximum_payload_displacement = 0.0
        lift_hold_displacement = 0.0
        maximum_attached_transform_error = 0.0
        release_position_errors: list[float] = []
        release_orientation_errors: list[float] = []
        last_phase = ""
        screenshot_requested = False
        screenshot_capture = None
        screenshot_future = None
        screenshot_phase_observed: str | None = None
        interrupted = False

        try:
            while simulation_app.is_running() and not panel_state["stop_requested"]:
                step_started = time.perf_counter()
                command = commands[command_index]
                positions, velocities = full_state(command)
                apply_kinematic_state(positions, velocities)
                tcp_matrix = prim_world_matrix(stage, tcp_prim_path)

                if attached:
                    payload_transform_op.Set(tcp_matrix)
                if command["attachment_event"] == "attach":
                    payload_transform_op.Set(tcp_matrix)
                    attached = True
                    attach_events += 1
                elif command["attachment_event"] == "release":
                    if not attached:
                        raise RuntimeError("Release event occurred while the payload was not attached")
                    payload_transform_op.Set(tcp_matrix)
                    attached = False
                    release_events += 1
                    release_matrix = prim_world_matrix(
                        stage, config["payload"]["root_prim_path"]
                    )
                    release_position_errors.append(
                        float(
                            np.linalg.norm(
                                matrix_translation(release_matrix) - initial_payload_position
                            )
                        )
                    )
                    release_orientation_errors.append(
                        matrix_rotation_error_radians(release_matrix, initial_payload_matrix)
                    )

                payload_matrix = prim_world_matrix(
                    stage, config["payload"]["root_prim_path"]
                )
                payload_displacement = float(
                    np.linalg.norm(matrix_translation(payload_matrix) - initial_payload_position)
                )
                maximum_payload_displacement = max(
                    maximum_payload_displacement, payload_displacement
                )
                if command["phase"] == "hold_lift":
                    lift_hold_displacement = max(lift_hold_displacement, payload_displacement)
                if attached:
                    matrix_error = float(
                        np.max(
                            np.abs(
                                np.asarray(payload_matrix, dtype=np.float64)
                                - np.asarray(tcp_matrix, dtype=np.float64)
                            )
                        )
                    )
                    maximum_attached_transform_error = max(
                        maximum_attached_transform_error, matrix_error
                    )

                readback = np.asarray(robot.get_joint_positions(), dtype=np.float64)
                if readback.shape != (8,) or not np.all(np.isfinite(readback)):
                    raise RuntimeError("Articulated kinematic readback is invalid")
                error = np.abs(readback - positions)
                maximum_readback_error = max(maximum_readback_error, float(np.max(error)))
                arm_error = max(float(error[dof_index[name]]) for name in ARM_JOINT_NAMES)
                maximum_arm_readback_error = max(maximum_arm_readback_error, arm_error)
                leader_value = float(readback[dof_index[gripper["leader_joint"]]])
                follower_value = float(readback[dof_index[gripper["mimic_joint"]]])
                target_finger = float(command["finger_position_m"])
                maximum_finger_readback_error = max(
                    maximum_finger_readback_error,
                    abs(leader_value - target_finger),
                    abs(follower_value - target_finger),
                )
                maximum_finger_mirror_error = max(
                    maximum_finger_mirror_error, abs(leader_value - follower_value)
                )
                observed_finger_min = min(observed_finger_min, leader_value, follower_value)
                observed_finger_max = max(observed_finger_max, leader_value, follower_value)

                render_this_step = (
                    render_animation
                    and step_count % render_interval == 0
                )
                if render_this_step:
                    world.render()
                    rendered_frames += 1
                if command["phase"] != last_phase:
                    print(f"Isaac grasp cycle: {command['phase']}")
                    if status_label is not None:
                        gap_mm = (
                            float(gripper["open_gap_m"])
                            - 2.0 * float(command["finger_position_m"])
                        ) * 1000.0
                        status_label.text = (
                            f"{command['phase']} | cycle {completed_cycles + 1} | "
                            f"jaw gap {gap_mm:.1f} mm | payload "
                            f"{'ATTACHED' if attached else 'FREE'}"
                        )
                    last_phase = command["phase"]

                if (
                    screenshot_path is not None
                    and not screenshot_requested
                    and command["phase"] == config["viewer"]["screenshot_phase"]
                    and render_this_step
                ):
                    if args.camera_view == "grasp":
                        set_camera_view("grasp")
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
                    screenshot_phase_observed = command["phase"]

                step_count += 1
                command_index += 1
                if command_index >= len(commands):
                    completed_cycles += 1
                    command_index = 0
                    final_matrix = prim_world_matrix(
                        stage, config["payload"]["root_prim_path"]
                    )
                    final_error = float(
                        np.linalg.norm(matrix_translation(final_matrix) - initial_payload_position)
                    )
                    final_orientation_error = matrix_rotation_error_radians(
                        final_matrix, initial_payload_matrix
                    )
                    if final_error > MAX_RELEASE_POSITION_ERROR_M:
                        raise RuntimeError(
                            f"Returned payload missed its initial pose by {final_error} m"
                        )
                    if final_orientation_error > MAX_RELEASE_ORIENTATION_ERROR_RAD:
                        raise RuntimeError(
                            "Returned payload missed its initial orientation by "
                            f"{final_orientation_error} rad"
                        )
                    payload_transform_op.Set(initial_payload_matrix)
                    if target_cycles > 0 and completed_cycles >= target_cycles:
                        break
                if not args.no_realtime:
                    remaining = control_dt - (time.perf_counter() - step_started)
                    if remaining > 0.0:
                        time.sleep(remaining)
        except KeyboardInterrupt:
            interrupted = True
            print("Isaac grasp cycle interrupted; writing available evidence")

        if args.headless and screenshot_path is not None:
            # Keep the validation pass renderer-free and fast. Once a complete
            # cycle has returned to ready, revisit the exact saved lift command
            # only long enough to capture a static visual proof, then restore the
            # final ready/open state.
            lift_command = next(
                command
                for command in reversed(commands)
                if command["phase"] == config["viewer"]["screenshot_phase"]
            )
            lift_positions, lift_velocities = full_state(lift_command)
            apply_kinematic_state(lift_positions, lift_velocities)
            payload_transform_op.Set(prim_world_matrix(stage, tcp_prim_path))
            viewport_ready, waited_frames = ViewportManager.wait_for_viewport(max_frames=120)
            if not viewport_ready:
                raise RuntimeError(
                    f"Isaac viewport was not ready after {waited_frames} frames"
                )
            # The first few headless viewport frames may apply Isaac's delayed
            # auto-framing. Warm the viewport before setting the evidence camera,
            # then reinforce it once after the renderer has consumed the pose.
            for _ in range(8):
                world.render()
                rendered_frames += 1
            set_camera_view(args.camera_view)
            for _ in range(8):
                world.render()
                rendered_frames += 1
            set_camera_view(args.camera_view)
            for _ in range(2):
                world.render()
                rendered_frames += 1
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
            screenshot_phase_observed = lift_command["phase"]
            for _ in range(90):
                if screenshot_future.done() or not simulation_app.is_running():
                    break
                world.render()
                rendered_frames += 1
            if not screenshot_future.done():
                raise RuntimeError(
                    "Static lift-pose screenshot did not finish within 90 frames"
                )
            final_positions, final_velocities = full_state(commands[-1])
            apply_kinematic_state(final_positions, final_velocities)
            payload_transform_op.Set(initial_payload_matrix)

        screenshot_wait_result = None
        if screenshot_path is not None and screenshot_future is None:
            raise RuntimeError("Screenshot was requested but the lift-hold phase was not captured")
        if screenshot_future is not None:
            for _ in range(180):
                if screenshot_future.done() or not simulation_app.is_running():
                    break
                world.render()
                rendered_frames += 1
            if not screenshot_future.done():
                raise RuntimeError("Screenshot capture did not finish within 180 frames")
            screenshot_wait_result = bool(screenshot_future.result())
            if not screenshot_wait_result:
                raise RuntimeError("Isaac screenshot capture returned failure")
            import omni.kit.renderer_capture

            omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()

        if target_cycles > 0 and completed_cycles < target_cycles:
            raise RuntimeError(
                f"Grasp cycle stopped after {completed_cycles}/{target_cycles} complete cycles"
            )
        if completed_cycles < 1:
            raise RuntimeError("No complete grasp cycle was observed")
        if maximum_readback_error > MAX_KINEMATIC_READBACK_ERROR:
            raise RuntimeError(
                f"Kinematic articulation readback exceeded tolerance: {maximum_readback_error}"
            )
        if maximum_finger_mirror_error > MAX_FINGER_MIRROR_ERROR_M:
            raise RuntimeError(
                f"2FG7 finger symmetry exceeded tolerance: {maximum_finger_mirror_error}"
            )
        if not math.isclose(
            observed_finger_min,
            float(gripper["open_position_m"]),
            rel_tol=0.0,
            abs_tol=MAX_FINGER_MIRROR_ERROR_M,
        ) or not math.isclose(
            observed_finger_max,
            float(gripper["closed_position_m"]),
            rel_tol=0.0,
            abs_tol=MAX_FINGER_MIRROR_ERROR_M,
        ):
            raise RuntimeError("The proof did not observe both full-open and full-closed finger states")
        if lift_hold_displacement < MIN_LIFT_DISPLACEMENT_M:
            raise RuntimeError(
                f"Payload lift displacement was too small: {lift_hold_displacement} m"
            )
        maximum_release_position_error = max(release_position_errors, default=math.inf)
        maximum_release_orientation_error = max(release_orientation_errors, default=math.inf)
        if maximum_release_position_error > MAX_RELEASE_POSITION_ERROR_M:
            raise RuntimeError(
                "Payload replacement exceeded position tolerance: "
                f"{maximum_release_position_error} m"
            )
        if maximum_release_orientation_error > MAX_RELEASE_ORIENTATION_ERROR_RAD:
            raise RuntimeError(
                "Payload replacement exceeded orientation tolerance: "
                f"{maximum_release_orientation_error} rad"
            )
        if attach_events != completed_cycles or release_events != completed_cycles:
            raise RuntimeError("Each complete cycle must contain exactly one attach and release event")

        screenshot_written = screenshot_path.is_file() if screenshot_path is not None else None
        if screenshot_path is not None and not screenshot_written:
            raise RuntimeError(f"Screenshot capture did not complete: {screenshot_path}")
        write_arm_preview_export(
            trajectory_path,
            commands,
            config=config,
            evidence=evidence,
        )
        report = {
            "format_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "mode": "isaac_only_articulated_2fg7_kinematic_pin_grasp_cycle",
            "command": [sys.executable, *sys.argv],
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "package_versions": package_versions,
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "source_plan": str(config["plan_path"]),
            "source_plan_sha256": config["plan_sha256"],
            "source_plan_profile": "legacy_cad_dry_run",
            "source_plan_usage": "six_joint_visual_choreography_only",
            "current_qc_collision_clearance_revalidated": False,
            "current_qc_collision_clearance_status": config["raw"]["arm_choreography"][
                "collision_clearance_status"
            ],
            "articulated_asset": {
                "profile": config["raw"]["articulated_asset"]["profile"],
                "usd": str(config["usd_path"]),
                "usd_sha256": sha256_file(config["usd_path"]),
                "source_urdf": config["import_report"]["source_urdf"],
                "source_urdf_sha256": config["import_report"]["source_urdf_sha256"],
                "import_report": str(config["import_report_path"]),
                "import_report_sha256": config["import_report_sha256"],
                "staged_manifest": str(config["staged_manifest_path"]),
                "staged_manifest_sha256": config["staged_manifest_sha256"],
                "tool_metadata": str(config["metadata_path"]),
                "tool_metadata_sha256": config["metadata_sha256"],
                "asset_artifact_count": len(config["asset_artifacts"]),
                "asset_artifacts": config["asset_artifacts"],
                "dof_names": list(robot.dof_names),
                "dof_count": robot.num_dof,
                "mimic_joint": config["import_report"]["imported_mimic_joints"],
            },
            "cycle": evidence,
            "completed_cycles": completed_cycles,
            "physics_dynamics_stepped": False,
            "motion_mode": "paused_physx_kinematic_joint_animation",
            "arm_joint_readback_max_error": maximum_arm_readback_error,
            "all_joint_readback_max_error": maximum_readback_error,
            "finger_joint_readback_max_error_m": maximum_finger_readback_error,
            "finger_mirror_max_error_m": maximum_finger_mirror_error,
            "finger_open_position_observed_m": observed_finger_min,
            "finger_closed_position_observed_m": observed_finger_max,
            "jaw_open_gap_m": float(gripper["open_gap_m"]),
            "jaw_closed_gap_m": float(gripper["closed_gap_m"]),
            "finger_motion_is_articulated": True,
            "payload": {
                "attachment_mode": "kinematic_visual_follow",
                "contact_physics_simulated": False,
                "collision_enabled": False,
                "rigid_body_enabled": False,
                "mass_api_enabled": False,
                "clear_pin_length_m": float(config["payload"]["clear_pin_length_m"]),
                "pinch_to_specimen_m": float(config["payload"]["pinch_to_specimen_m"]),
                "attach_events": attach_events,
                "release_events": release_events,
                "maximum_attached_transform_error": maximum_attached_transform_error,
                "maximum_displacement_m": maximum_payload_displacement,
                "lift_hold_displacement_m": lift_hold_displacement,
                "maximum_release_position_error_m": maximum_release_position_error,
                "maximum_release_orientation_error_rad": maximum_release_orientation_error,
                "visual_prim_paths": payload_paths,
            },
            "scene_paths": scene_paths,
            "control_dt_seconds": control_dt,
            "render_interval_control_steps": render_interval,
            "rendered_frames": rendered_frames,
            "headless": args.headless,
            "no_realtime": args.no_realtime,
            "active_gpu": 0,
            "multi_gpu": False,
            "configured_physx_device": "cpu",
            "physical_camera_or_depth_sensor_used": False,
            "viewport_camera_used": render_animation or screenshot_path is not None,
            "viewport_camera": camera_view_evidence or None,
            "ros_used": False,
            "watson_connected": False,
            "real_robot_commanded": False,
            "interrupted": interrupted,
            "stop_button_requested": panel_state["stop_requested"],
            "trajectory_export": str(trajectory_path),
            "trajectory_export_sha256": sha256_file(trajectory_path),
            "trajectory_export_status": "simulation_preview_only_not_a_watson_command",
            "screenshot": str(screenshot_path) if screenshot_path else None,
            "screenshot_phase": screenshot_phase_observed,
            "screenshot_written": screenshot_written,
            "screenshot_sha256": sha256_file(screenshot_path) if screenshot_path else None,
            "screenshot_dimensions": png_dimensions(screenshot_path) if screenshot_path else None,
            "screenshot_capture_scheduled": screenshot_capture is not None,
            "screenshot_wait_result": screenshot_wait_result,
            "wall_seconds": time.perf_counter() - started_wall,
            "warning": (
                "Visible choreography only. The articulated fingers and 10 mm placement are "
                "deterministic, but contact, friction, force, current-QC trajectory clearance, "
                "physical TCP calibration, controller timing, and grasp success are unvalidated."
            ),
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Isaac grasp-cycle report: {report_path}")
        print(f"Watson arm preview (not executable): {trajectory_path}")
        exit_code = 0
        return 0
    except KeyboardInterrupt:
        exit_code = 0
        return 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        panel_window = None
        if world is not None and simulation_app.is_running():
            world.stop()
        simulation_app.close(wait_for_replicator=False, exit_code=exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
