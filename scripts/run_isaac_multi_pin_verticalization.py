#!/usr/bin/env python3
"""Run the Isaac-only seven-specimen pin verticalisation presentation.

The existing eight-DOF Watson/QC/2FG7 presentation articulation follows a
deterministic, offline multi-pin plan.  Each synthetic specimen is represented
by its own visual-only payload.  The active payload follows the application TCP
exactly between the explicit attach and release events; released payloads stay
at their planned world-vertical destination.  No ROS graph, network
connection, Watson command, contact, friction, or grasp physics is created.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any

import numpy as np
import yaml

ARENA_DIR = Path(__file__).resolve().parents[1]
if str(ARENA_DIR) not in sys.path:
    sys.path.insert(0, str(ARENA_DIR))

import run_isaac_grasp_cycle as single
from pin_axis_3d_sim.multi_pin_cycle import (
    ARM_JOINT_NAMES,
    PHASE_ORDER,
    SOURCE_STAGE_ORDER as STAGE_ORDER,
    build_multi_pin_cycle,
    multi_pin_cycle_evidence,
)


DEFAULT_CONFIG = ARENA_DIR / "config/isaac_multi_pin_verticalization.yaml"
DEFAULT_REPORT = (
    ARENA_DIR
    / "outputs/isaac_sim/6.0.1/multi_pin_verticalization_report.json"
)
EXPECTED_SPECIMEN_IDS = list(range(1, 8))
MAX_KINEMATIC_READBACK_ERROR = 1.0e-6
MAX_FINGER_MIRROR_ERROR_M = 1.0e-7
MAX_ATTACHED_TRANSFORM_ERROR = 1.0e-9
MAX_INITIAL_AXIS_ERROR_RAD = 1.0e-5
MAX_FINAL_AXIS_ERROR_RAD = 1.0e-6
MAX_FINAL_BASE_ERROR_M = 1.0e-6
MAX_RELEASE_TCP_POSITION_ERROR_M = 1.0e-5
# The cuMotion plan accepts endpoint orientation residuals below 0.002 rad.
# Final released visuals are independently snapped and gated at 1e-6 rad.
MAX_RELEASE_TCP_ORIENTATION_ERROR_RAD = 0.002
MIN_LIFT_DISPLACEMENT_M = 0.020
SCENE_SUPPRESSED_ROLES = {
    "specimen_body",
    "pin_shaft",
    "pin_head",
    "other_pin_shaft",
    "other_pin_head",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Whole seven-specimen cycles; zero loops until the window closes.",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Do not pace simulation time to wall time; intended for headless proof runs.",
    )
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--camera-view", choices=["tray", "workcell"], default="tray")
    return parser


def _stage(specimen: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in specimen["stages"] if item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(
            f"Specimen {specimen.get('specimen_id')} must contain one {name!r} stage"
        )
    return matches[0]


def _portable_report_path(value: Any, label: str) -> Path:
    """Resolve a report path relative to this repository and contain it."""

    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = ARENA_DIR / path
    path = path.resolve()
    try:
        path.relative_to(ARENA_DIR.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository: {path}") from exc
    return path


def _validate_pose(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a pose object")
    single.finite_vector(value.get("position_xyz_m"), 3, f"{label} position")
    quaternion = single.finite_vector(
        value.get("quaternion_xyzw"), 4, f"{label} quaternion"
    )
    if not math.isclose(
        float(np.linalg.norm(quaternion)), 1.0, rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise ValueError(f"{label} quaternion must be normalised")


def validate_multi_pin_plan(plan: dict[str, Any]) -> None:
    if plan.get("format_version") != 1:
        raise ValueError("Multi-pin plan must use format_version 1")
    if plan.get("frame_id") != "base":
        raise ValueError("Multi-pin plan must use the Watson base frame")
    if plan.get("planning_tool_frame") != "pin_grasp_tcp":
        raise ValueError("Multi-pin plan must target the application pin_grasp_tcp")
    if plan.get("ros_used") is not False:
        raise ValueError("Multi-pin plan must explicitly record ros_used false")
    if plan.get("watson_connected") is not False:
        raise ValueError("Multi-pin plan must explicitly record watson_connected false")
    if plan.get("real_robot_commanded") is not False:
        raise ValueError("Multi-pin plan must explicitly record real_robot_commanded false")
    control_dt = float(plan.get("control_dt_seconds", 0.0))
    if not math.isfinite(control_dt) or control_dt <= 0.0:
        raise ValueError("Multi-pin plan control_dt_seconds must be positive")
    maximum_step = float(plan.get("maximum_control_step_rad", 0.0))
    if not math.isfinite(maximum_step) or maximum_step <= 0.0:
        raise ValueError("Multi-pin plan maximum_control_step_rad must be positive")
    single.finite_vector(plan.get("ready_joint_positions"), 6, "ready joints")
    required_clearance = float(plan.get("required_sampled_sphere_clearance_m", 0.0))
    validation = plan.get("validation", {})
    minimum_clearance = float(
        validation.get("minimum_sampled_sphere_clearance_m", 0.0)
    )
    if required_clearance < 0.004 or minimum_clearance < required_clearance:
        raise ValueError("Multi-pin plan lost its required 4 mm robot-sphere clearance")
    if validation.get("all_stages_accepted") is not True:
        raise ValueError("Multi-pin plan must accept every planned stage")
    if validation.get("sampled_self_collision") is not False:
        raise ValueError("Multi-pin plan reports a sampled self collision")
    if validation.get("derivative_limits_met") is not True:
        raise ValueError("Multi-pin plan violates a derivative limit")
    model_status = plan.get("model_status", {})
    if model_status.get("tool_profile") != "watson_qc_nominal":
        raise ValueError("Multi-pin plan must retain the reviewed Watson QC tool profile")
    if model_status.get("attached_payload_collision_modelled") is not False:
        raise ValueError("Attached-payload collision scope must remain explicit")
    if plan.get("specimen_ids") != EXPECTED_SPECIMEN_IDS:
        raise ValueError("Multi-pin plan must select specimen IDs 1 through 7 in order")
    specimens = plan.get("specimens")
    if not isinstance(specimens, list) or len(specimens) != 7:
        raise ValueError("Multi-pin plan must contain exactly seven specimens")
    if [item.get("specimen_id") for item in specimens] != EXPECTED_SPECIMEN_IDS:
        raise ValueError("Multi-pin specimen order must be exactly 1 through 7")

    for specimen in specimens:
        specimen_id = int(specimen["specimen_id"])
        initial_axis = single.finite_vector(
            specimen.get("initial_axis_up"), 3, f"specimen {specimen_id} initial axis"
        )
        final_axis = single.finite_vector(
            specimen.get("final_axis_up"), 3, f"specimen {specimen_id} final axis"
        )
        if not math.isclose(float(np.linalg.norm(initial_axis)), 1.0, abs_tol=1.0e-6):
            raise ValueError(f"Specimen {specimen_id} initial axis must be normalised")
        if not np.array_equal(final_axis, np.array([0.0, 0.0, 1.0])):
            raise ValueError(f"Specimen {specimen_id} final axis must be exact world +Z")
        single.finite_vector(
            specimen.get("base_xyz_m"), 3, f"specimen {specimen_id} base"
        )
        single.finite_vector(
            specimen.get("source_base_xyz_m"),
            3,
            f"specimen {specimen_id} source base",
        )
        remaining_end = float(specimen.get("remaining_pin_end_z_from_pinch_m", 0.0))
        if not math.isfinite(remaining_end) or remaining_end <= 0.0:
            raise ValueError(f"Specimen {specimen_id} remaining pin endpoint is invalid")
        stages = specimen.get("stages")
        if not isinstance(stages, list) or tuple(
            item.get("name") for item in stages
        ) != tuple(STAGE_ORDER):
            raise ValueError(f"Specimen {specimen_id} stage order is invalid")
        for stage in stages:
            samples = stage.get("control_samples")
            if not isinstance(samples, list) or len(samples) < 2:
                raise ValueError(
                    f"Specimen {specimen_id} stage {stage.get('name')} has too few samples"
                )
            target_pose = stage.get("target_pin_grasp_tcp_pose")
            if stage.get("name") in {"descend_tilted_grasp", "descend_vertical"}:
                _validate_pose(
                    target_pose,
                    f"specimen {specimen_id} stage {stage.get('name')}",
                )
            elif target_pose is not None:
                _validate_pose(
                    target_pose,
                    f"specimen {specimen_id} stage {stage.get('name')}",
                )
            previous_time = -math.inf
            for sample in samples:
                sample_time = float(sample.get("time_seconds", math.nan))
                if not math.isfinite(sample_time) or sample_time < previous_time:
                    raise ValueError("Stage sample times must be finite and monotonic")
                previous_time = sample_time
                single.finite_vector(
                    sample.get("q", sample.get("joint_positions")),
                    6,
                    "joint positions",
                )
                single.finite_vector(
                    sample.get("qd", sample.get("joint_velocities")),
                    6,
                    "joint velocities",
                )


def load_and_validate_config(config_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("format_version") != 1:
        raise ValueError("Isaac multi-pin config must use format_version 1")
    scope = raw.get("scope", {})
    for field in (
        "ros_used",
        "watson_connected",
        "real_robot_commanded",
        "contact_physics_simulated",
        "physical_camera_or_depth_used",
    ):
        if scope.get(field) is not False:
            raise ValueError(f"Multi-pin scope must explicitly set {field} false")

    plan_config = raw["multi_pin_plan"]
    plan_path = single.contained_artifact(plan_config["path"], "multi-pin plan")
    plan_sha256 = single.sha256_file(plan_path)
    if plan_sha256 != plan_config["sha256"]:
        raise ValueError("Multi-pin plan hash mismatch")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_multi_pin_plan(plan)
    if plan_config.get("specimen_ids") != EXPECTED_SPECIMEN_IDS:
        raise ValueError("Configured specimen IDs must be exactly 1 through 7")
    placement = plan_config.get("placement")
    if not isinstance(placement, str) or not placement:
        raise ValueError("Multi-pin placement policy must be recorded")

    asset = raw["articulated_asset"]
    usd_path = single.contained_artifact(asset["usd"], "articulated Isaac USD")
    import_report_path = single.contained_artifact(
        asset["import_report"], "articulated import report"
    )
    manifest_path = single.contained_artifact(
        asset["staged_manifest"], "articulated staging manifest"
    )
    metadata_path = single.contained_artifact(
        asset["tool_metadata"], "articulated tool metadata"
    )
    for path, expected, label in (
        (import_report_path, asset["import_report_sha256"], "import report"),
        (manifest_path, asset["staged_manifest_sha256"], "staging manifest"),
        (metadata_path, asset["tool_metadata_sha256"], "tool metadata"),
    ):
        if single.sha256_file(path) != expected:
            raise ValueError(f"Articulated {label} hash mismatch")

    import_report = json.loads(import_report_path.read_text(encoding="utf-8"))
    expected_names = list(asset["expected_dof_names"])
    if asset.get("profile") != "watson_qc_articulated_2fg7":
        raise ValueError("Multi-pin viewer requires the articulated Watson/QC/2FG7 profile")
    if import_report.get("validation_profile") != asset["profile"]:
        raise ValueError("Articulated import profile mismatch")
    if import_report.get("dof_count") != 8:
        raise ValueError("Articulated import must contain eight DOFs")
    if import_report.get("physx_dof_names") != expected_names:
        raise ValueError("Articulated import joint order mismatch")
    if _portable_report_path(
        import_report["output_usd"],
        "articulated output USD",
    ) != usd_path:
        raise ValueError("Articulated import report references a different USD")
    if single.sha256_file(usd_path) != import_report["output_usd_sha256"]:
        raise ValueError("Articulated root USD hash mismatch")
    source_urdf = _portable_report_path(
        import_report["source_urdf"],
        "articulated source URDF",
    )
    if not source_urdf.is_file():
        raise FileNotFoundError(f"Articulated source URDF is missing: {source_urdf}")
    if single.sha256_file(source_urdf) != import_report["source_urdf_sha256"]:
        raise ValueError("Articulated source URDF hash mismatch")
    asset_artifacts = single.validate_asset_artifacts(
        import_report,
        repository_root=ARENA_DIR,
    )

    staged_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if staged_manifest.get("asset_mode") != "isaac_articulated":
        raise ValueError("Staged asset is not marked as articulated")
    if staged_manifest.get("moving_joints") != expected_names:
        raise ValueError("Staged articulated joint order mismatch")
    if staged_manifest.get("xrdf") is not None:
        raise ValueError("Articulated display asset must not masquerade as cuMotion input")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("tool_profile") != "watson_qc_nominal":
        raise ValueError("Articulated tool profile mismatch")
    if metadata.get("finger_joints") != "prismatic":
        raise ValueError("2FG7 fingers must be prismatic")
    if metadata.get("finger_configuration") != "inwards":
        raise ValueError("2FG7 fingers must retain their inward configuration")

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
        if not math.isclose(
            float(gripper[name]), expected, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(f"Gripper config disagrees with tool metadata: {name}")
    if gripper["leader_joint"] != expected_names[6]:
        raise ValueError("Configured 2FG7 leader joint is incorrect")
    if gripper["mimic_joint"] != expected_names[7]:
        raise ValueError("Configured 2FG7 mimic joint is incorrect")

    payload = raw["payload_visual"]
    if payload.get("attachment_mode") != "kinematic_visual_follow":
        raise ValueError("Multi-pin payloads must use kinematic visual following")
    if payload.get("root_prim_prefix") != "/MultiPinCycle/Payload_":
        raise ValueError("Multi-pin payload root prefix is fixed for evidence")
    if not math.isclose(float(payload["clear_pin_length_m"]), 0.010):
        raise ValueError("Multi-pin clear section must remain exactly 10 mm")
    if not math.isclose(float(payload["pinch_to_specimen_m"]), 0.005):
        raise ValueError("Multi-pin pinch must remain at the 10 mm midpoint")
    if 2.0 * float(payload["pin_radius_m"]) >= float(gripper["closed_gap_m"]):
        raise ValueError("Visual pin diameter does not fit the closed jaw gap")
    specimen_scale = single.finite_vector(
        payload["specimen_scale_xyz_m"], 3, "specimen visual scale"
    )
    specimen_near = float(payload["specimen_center_z_from_pinch_m"]) - 0.5 * float(
        specimen_scale[2]
    )
    if not math.isclose(
        specimen_near,
        float(payload["specimen_near_z_from_pinch_m"]),
        abs_tol=1.0e-12,
    ):
        raise ValueError("Specimen visual must begin after the exact 10 mm clear section")
    colors = payload.get("specimen_colors_rgb")
    if not isinstance(colors, list) or len(colors) != 7:
        raise ValueError("Exactly seven specimen colours are required")
    for index, color in enumerate(colors, start=1):
        value = single.finite_vector(color, 3, f"specimen {index} colour")
        if np.any(value < 0.0) or np.any(value > 1.0):
            raise ValueError("Specimen colours must be in the range 0..1")

    viewer = raw["viewer"]
    if not math.isfinite(float(viewer["render_hz"])) or float(viewer["render_hz"]) <= 0:
        raise ValueError("Viewer render_hz must be positive")
    for name in (
        "tray_camera_eye_xyz_m",
        "tray_camera_target_xyz_m",
        "workcell_camera_eye_xyz_m",
        "workcell_camera_target_xyz_m",
    ):
        single.finite_vector(viewer[name], 3, name)
    if viewer.get("screenshot_phase") not in PHASE_ORDER:
        raise ValueError("Configured screenshot phase is not part of the cycle")

    return {
        "raw": raw,
        "plan": plan,
        "plan_path": plan_path,
        "plan_sha256": plan_sha256,
        "usd_path": usd_path,
        "import_report": import_report,
        "import_report_path": import_report_path,
        "asset_artifacts": asset_artifacts,
        "staged_manifest_path": manifest_path,
        "metadata_path": metadata_path,
        "expected_dof_names": expected_names,
        "gripper": gripper,
        "payload": payload,
        "viewer": viewer,
    }


def pose_matrix(pose: dict[str, Any]) -> Any:
    from pxr import Gf

    position = single.finite_vector(pose["position_xyz_m"], 3, "pose position")
    qx, qy, qz, qw = single.finite_vector(
        pose["quaternion_xyzw"], 4, "pose quaternion"
    )
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(Gf.Quatd(float(qw), Gf.Vec3d(float(qx), float(qy), float(qz))))
    matrix.SetTranslateOnly(Gf.Vec3d(*position.tolist()))
    return matrix


def pin_axis_up(matrix: Any) -> np.ndarray:
    """Return base-to-head axis; payload local +Z intentionally points to the base."""

    from pxr import Gf

    value = matrix.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
    axis = np.asarray([value[0], value[1], value[2]], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("Payload transform produced an invalid pin axis")
    return axis / norm


def pin_base_position(matrix: Any, remaining_end_m: float) -> np.ndarray:
    from pxr import Gf

    value = matrix.Transform(Gf.Vec3d(0.0, 0.0, float(remaining_end_m)))
    return np.asarray([value[0], value[1], value[2]], dtype=np.float64)


def axis_error_radians(observed: np.ndarray, expected: np.ndarray) -> float:
    return math.acos(float(np.clip(np.dot(observed, expected), -1.0, 1.0)))


def add_payload_visual(
    stage: Any,
    payload: dict[str, Any],
    specimen: dict[str, Any],
    color: list[float],
) -> tuple[Any, list[str]]:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    specimen_id = int(specimen["specimen_id"])
    root_path = f"{payload['root_prim_prefix']}{specimen_id}"
    if stage.GetPrimAtPath(root_path).IsValid():
        raise RuntimeError(f"Payload prim already exists: {root_path}")
    root = UsdGeom.Xform.Define(stage, root_path)
    root_op = root.AddTransformOp(precision=UsdGeom.XformOp.PrecisionDouble)
    root_op.Set(Gf.Matrix4d(1.0))

    clear_start = float(payload["clear_start_z_from_pinch_m"])
    specimen_near = float(payload["specimen_near_z_from_pinch_m"])
    remaining_end = float(specimen["remaining_pin_end_z_from_pinch_m"])
    if remaining_end <= specimen_near:
        raise ValueError(
            f"Specimen {specimen_id} remaining endpoint does not extend beyond its body"
        )
    created = [root_path]
    clear = UsdGeom.Cylinder.Define(stage, f"{root_path}/ClearPin10mm")
    clear.CreateAxisAttr(UsdGeom.Tokens.z)
    clear.CreateRadiusAttr(float(payload["pin_radius_m"]))
    clear.CreateHeightAttr(specimen_near - clear_start)
    UsdGeom.Xformable(clear).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, 0.5 * (clear_start + specimen_near))
    )
    single.set_gprim_color(clear, (0.05, 0.85, 1.0))
    created.append(str(clear.GetPath()))

    remainder = UsdGeom.Cylinder.Define(stage, f"{root_path}/RemainingPin")
    remainder.CreateAxisAttr(UsdGeom.Tokens.z)
    remainder.CreateRadiusAttr(float(payload["pin_radius_m"]))
    remainder.CreateHeightAttr(remaining_end - specimen_near)
    UsdGeom.Xformable(remainder).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, 0.5 * (specimen_near + remaining_end))
    )
    single.set_gprim_color(remainder, (0.68, 0.72, 0.78))
    created.append(str(remainder.GetPath()))

    head = UsdGeom.Sphere.Define(stage, f"{root_path}/PinHead")
    head.CreateRadiusAttr(float(payload["pin_head_radius_m"]))
    UsdGeom.Xformable(head).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, float(payload["pin_head_center_z_from_pinch_m"]))
    )
    single.set_gprim_color(head, (0.12, 0.92, 0.35))
    created.append(str(head.GetPath()))

    body = UsdGeom.Sphere.Define(stage, f"{root_path}/Specimen")
    body.CreateRadiusAttr(0.5)
    body_xform = UsdGeom.Xformable(body)
    body_xform.AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, float(payload["specimen_center_z_from_pinch_m"]))
    )
    body_xform.AddScaleOp().Set(
        Gf.Vec3f(*single.finite_vector(
            payload["specimen_scale_xyz_m"], 3, "specimen scale"
        ).tolist())
    )
    single.set_gprim_color(body, tuple(float(item) for item in color))
    created.append(str(body.GetPath()))

    for path in created:
        prim = stage.GetPrimAtPath(path)
        prim.CreateAttribute("magi:visualOnly", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
        prim.CreateAttribute(
            "magi:collisionQualified", Sdf.ValueTypeNames.Bool, custom=True
        ).Set(False)
        prim.CreateAttribute("magi:specimenId", Sdf.ValueTypeNames.Int, custom=True).Set(
            specimen_id
        )
        if (
            prim.HasAPI(UsdPhysics.CollisionAPI)
            or prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or prim.HasAPI(UsdPhysics.MassAPI)
        ):
            raise RuntimeError(f"Visual payload unexpectedly has a physics API: {path}")
    return root_op, created


def add_static_scene(stage: Any, plan: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    from isaacsim.core.api.objects import VisualCuboid, VisualSphere
    from pxr import UsdLux

    role_colors = {
        "foam": np.array([0.18, 0.34, 0.20]),
        "tray_wall": np.array([0.28, 0.31, 0.36]),
    }
    created: list[str] = []
    suppressed: list[dict[str, Any]] = []
    for index, obstacle in enumerate(plan.get("scene_obstacles", [])):
        role = str(obstacle.get("role"))
        if role in SCENE_SUPPRESSED_ROLES:
            suppressed.append(
                {
                    "index": index,
                    "role": role,
                    "source_id": obstacle.get("source_id"),
                }
            )
            continue
        if role not in role_colors:
            raise ValueError(f"Unsupported static scene role: {role}")
        path = f"/MultiPinCycle/Scene/Proxy_{index:02d}_{role}"
        color = role_colors[role]
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
        elif obstacle["type"] == "sphere":
            VisualSphere(
                prim_path=path,
                position=np.asarray(obstacle["position_xyz_m"], dtype=np.float64),
                radius=float(obstacle["radius_m"]),
                color=color,
            )
        else:
            raise ValueError(f"Unsupported static obstacle type: {obstacle['type']}")
        created.append(path)

    dome = UsdLux.DomeLight.Define(stage, "/MultiPinCycle/Lights/Dome")
    dome.CreateIntensityAttr(950.0)
    key = UsdLux.DistantLight.Define(stage, "/MultiPinCycle/Lights/Key")
    key.CreateIntensityAttr(3000.0)
    key.CreateAngleAttr(1.0)
    created.extend(["/MultiPinCycle/Lights/Dome", "/MultiPinCycle/Lights/Key"])
    return created, suppressed


def create_status_panel(ui: Any, set_camera_view: Any) -> tuple[Any, Any, dict[str, bool]]:
    state = {"stop_requested": False}
    window = ui.Window("Seven-Specimen Pin Verticalisation", width=550, height=345)
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label(
                "ISAAC ONLY - WATSON AND ROS NOT CONNECTED",
                style={"color": 0xFF5A5AFF, "font_size": 17},
            )
            ui.Label("Pick each specimen > lift > rotate vertical > place at planned base")
            status_label = ui.Label("Initialising seven-specimen cycle...")
            ui.Label(
                "Seven independent visual payloads use the centred 10 mm grasp baseline.\n"
                "The active payload follows the TCP; released pins stay upright.\n"
                "Contact, friction, grasp force, and physical execution are not simulated.",
                word_wrap=True,
            )
            with ui.HStack(spacing=8, height=32):
                ui.Button("Focus tray", clicked_fn=lambda: set_camera_view("tray"))
                ui.Button("Whole workcell", clicked_fn=lambda: set_camera_view("workcell"))

            def request_stop() -> None:
                state["stop_requested"] = True

            ui.Button("Stop and close demo", height=32, clicked_fn=request_stop)
    return window, status_label, state


def main() -> int:
    args = build_parser().parse_args()
    package_versions = single.validate_runtime()
    config_path = args.config.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    screenshot_path = args.screenshot.expanduser().resolve() if args.screenshot else None
    if args.cycles < 0:
        raise ValueError("--cycles must be non-negative")
    target_cycles = 1 if args.headless and args.cycles == 0 else args.cycles
    if args.headless and target_cycles <= 0:
        raise ValueError("Headless validation requires at least one seven-specimen cycle")
    for output_path, label in ((report_path, "report"), (screenshot_path, "screenshot")):
        if output_path is None:
            continue
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite {label}: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_and_validate_config(config_path)
    gripper = config["gripper"]
    commands = build_multi_pin_cycle(
        config["plan"],
        finger_open_m=float(gripper["open_position_m"]),
        finger_closed_m=float(gripper["closed_position_m"]),
        finger_speed_m_s=float(gripper["per_finger_speed_m_s"]),
        hold_seconds=float(gripper["hold_seconds"]),
    )
    control_dt = float(config["plan"]["control_dt_seconds"])
    evidence = multi_pin_cycle_evidence(commands, config["plan"])
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
        carb.settings.get_settings().set_bool(single.GUIDE_PURPOSE_DISPLAY_SETTING, False)
        if carb.settings.get_settings().get_as_bool(single.GUIDE_PURPOSE_DISPLAY_SETTING):
            raise RuntimeError("Isaac collision-guide display could not be disabled")
        stage = omni.usd.get_context().get_stage()
        default_prim = stage.GetDefaultPrim()
        if not default_prim.IsValid():
            raise RuntimeError("Articulated Watson stage has no default prim")
        asset_prim_path = str(default_prim.GetPath())
        scene_paths, suppressed_scene_proxies = add_static_scene(stage, config["plan"])

        payload_ops: dict[int, Any] = {}
        payload_paths: dict[int, list[str]] = {}
        initial_matrices: dict[int, Any] = {}
        destination_matrices: dict[int, Any] = {}
        remaining_endpoints: dict[int, float] = {}
        base_targets: dict[int, np.ndarray] = {}
        source_base_targets: dict[int, np.ndarray] = {}
        initial_pose_errors: dict[int, dict[str, float]] = {}
        destination_pose_errors: dict[int, dict[str, float]] = {}
        for specimen, color in zip(
            config["plan"]["specimens"], config["payload"]["specimen_colors_rgb"]
        ):
            specimen_id = int(specimen["specimen_id"])
            payload_ops[specimen_id], payload_paths[specimen_id] = add_payload_visual(
                stage, config["payload"], specimen, color
            )
            initial_matrix = pose_matrix(
                _stage(specimen, "descend_tilted_grasp")["target_pin_grasp_tcp_pose"]
            )
            destination_matrix = pose_matrix(
                _stage(specimen, "descend_vertical")["target_pin_grasp_tcp_pose"]
            )
            initial_matrices[specimen_id] = initial_matrix
            destination_matrices[specimen_id] = destination_matrix
            remaining_end = float(specimen["remaining_pin_end_z_from_pinch_m"])
            remaining_endpoints[specimen_id] = remaining_end
            base_target = np.asarray(specimen["base_xyz_m"], dtype=np.float64)
            base_targets[specimen_id] = base_target
            source_base_target = np.asarray(
                specimen["source_base_xyz_m"], dtype=np.float64
            )
            source_base_targets[specimen_id] = source_base_target
            initial_axis_error = axis_error_radians(
                pin_axis_up(initial_matrix),
                np.asarray(specimen["initial_axis_up"], dtype=np.float64),
            )
            initial_base_error = float(
                np.linalg.norm(
                    pin_base_position(initial_matrix, remaining_end)
                    - source_base_target
                )
            )
            destination_axis_error = axis_error_radians(
                pin_axis_up(destination_matrix), np.array([0.0, 0.0, 1.0])
            )
            destination_base_error = float(
                np.linalg.norm(
                    pin_base_position(destination_matrix, remaining_end) - base_target
                )
            )
            initial_pose_errors[specimen_id] = {
                "axis_error_rad": initial_axis_error,
                "base_error_m": initial_base_error,
            }
            destination_pose_errors[specimen_id] = {
                "axis_error_rad": destination_axis_error,
                "base_error_m": destination_base_error,
            }
            if initial_axis_error > MAX_INITIAL_AXIS_ERROR_RAD:
                raise RuntimeError(f"Specimen {specimen_id} initial pose lost its pin axis")
            # The initial TCP targets are derived from detector lines whereas
            # source_base_xyz_m is retained truth/provenance.  Preserve the
            # planned TCP transform exactly and report, rather than gate, this
            # detector-to-source-reference offset.
            if destination_axis_error > MAX_FINAL_AXIS_ERROR_RAD:
                raise RuntimeError(f"Specimen {specimen_id} destination is not world vertical")
            if destination_base_error > MAX_FINAL_BASE_ERROR_M:
                raise RuntimeError(f"Specimen {specimen_id} destination moved its base")
            payload_ops[specimen_id].Set(initial_matrix)

        world = World(
            physics_dt=control_dt,
            rendering_dt=1.0 / float(config["viewer"]["render_hz"]),
            stage_units_in_meters=1.0,
            backend="numpy",
            device="cpu",
        )
        robot = world.scene.add(
            SingleArticulation(
                prim_path=asset_prim_path,
                name="watson_2fg7_multi_pin_verticalization",
            )
        )
        world.reset()
        world.pause()
        if not robot.handles_initialized:
            raise RuntimeError("Articulated Watson handles did not initialise")
        if list(robot.dof_names) != config["expected_dof_names"] or robot.num_dof != 8:
            raise RuntimeError(
                f"Expected eight DOFs {config['expected_dof_names']}; "
                f"found {list(robot.dof_names)}"
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
        below = all_positions < joint_limits[:, 0] - single.IMPORTED_LIMIT_TOLERANCE
        above = all_positions > joint_limits[:, 1] + single.IMPORTED_LIMIT_TOLERANCE
        if np.any(below) or np.any(above):
            invalid = []
            for index, name in enumerate(robot.dof_names):
                if np.any(below[:, index]) or np.any(above[:, index]):
                    invalid.append(
                        {
                            "joint": name,
                            "command_min": float(np.min(all_positions[:, index])),
                            "command_max": float(np.max(all_positions[:, index])),
                            "imported_lower": float(joint_limits[index, 0]),
                            "imported_upper": float(joint_limits[index, 1]),
                        }
                    )
            raise RuntimeError(f"Multi-pin cycle exceeds articulation limits: {invalid}")

        def apply_kinematic_state(positions: np.ndarray, velocities: np.ndarray) -> None:
            robot.set_joint_positions(positions)
            robot.set_joint_velocities(velocities)
            world.physics_sim_view.update_articulations_kinematic()
            get_physx_interface().update_transformations(False, True, False)

        start_positions, start_velocities = full_state(commands[0])
        robot.set_joints_default_state(positions=start_positions, velocities=start_velocities)
        apply_kinematic_state(start_positions, start_velocities)
        tcp_prim_path = config["import_report"]["expected_link_paths"]["pin_grasp_tcp"]

        camera_view_evidence: dict[str, Any] = {}

        def set_camera_view(view_name: str) -> None:
            if view_name == "tray":
                eye = single.finite_vector(
                    config["viewer"]["tray_camera_eye_xyz_m"], 3, "tray camera eye"
                )
                target = single.finite_vector(
                    config["viewer"]["tray_camera_target_xyz_m"], 3, "tray camera target"
                )
            elif view_name == "workcell":
                eye = single.finite_vector(
                    config["viewer"]["workcell_camera_eye_xyz_m"],
                    3,
                    "workcell camera eye",
                )
                target = single.finite_vector(
                    config["viewer"]["workcell_camera_target_xyz_m"],
                    3,
                    "workcell camera target",
                )
            else:
                raise ValueError(f"Unknown camera view: {view_name}")
            camera_view_evidence.update(
                {"view": view_name, "eye_xyz_m": eye.tolist(), "target_xyz_m": target.tolist()}
            )
            print(f"Isaac camera {view_name}: eye={eye.tolist()} target={target.tolist()}")
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
        attached_specimen: int | None = None
        attach_events = {specimen_id: 0 for specimen_id in EXPECTED_SPECIMEN_IDS}
        release_events = {specimen_id: 0 for specimen_id in EXPECTED_SPECIMEN_IDS}
        lift_displacements = {specimen_id: 0.0 for specimen_id in EXPECTED_SPECIMEN_IDS}
        release_tcp_position_errors = {
            specimen_id: [] for specimen_id in EXPECTED_SPECIMEN_IDS
        }
        release_tcp_orientation_errors = {
            specimen_id: [] for specimen_id in EXPECTED_SPECIMEN_IDS
        }
        final_axis_errors = {specimen_id: [] for specimen_id in EXPECTED_SPECIMEN_IDS}
        final_base_errors = {specimen_id: [] for specimen_id in EXPECTED_SPECIMEN_IDS}
        maximum_readback_error = 0.0
        maximum_arm_readback_error = 0.0
        maximum_finger_readback_error = 0.0
        maximum_finger_mirror_error = 0.0
        maximum_attached_transform_error = 0.0
        observed_finger_min = math.inf
        observed_finger_max = -math.inf
        last_phase_key: tuple[int, str] | None = None
        screenshot_requested = False
        screenshot_capture = None
        screenshot_future = None
        screenshot_phase_observed: str | None = None
        interrupted = False

        def reset_payloads() -> None:
            for specimen_id in EXPECTED_SPECIMEN_IDS:
                payload_ops[specimen_id].Set(initial_matrices[specimen_id])

        def validate_final_payloads() -> None:
            for specimen_id in EXPECTED_SPECIMEN_IDS:
                matrix = single.prim_world_matrix(
                    stage, f"{config['payload']['root_prim_prefix']}{specimen_id}"
                )
                axis_error = axis_error_radians(
                    pin_axis_up(matrix), np.array([0.0, 0.0, 1.0])
                )
                base_error = float(
                    np.linalg.norm(
                        pin_base_position(matrix, remaining_endpoints[specimen_id])
                        - base_targets[specimen_id]
                    )
                )
                final_axis_errors[specimen_id].append(axis_error)
                final_base_errors[specimen_id].append(base_error)
                if axis_error > MAX_FINAL_AXIS_ERROR_RAD:
                    raise RuntimeError(
                        f"Specimen {specimen_id} final axis error is {axis_error} rad"
                    )
                if base_error > MAX_FINAL_BASE_ERROR_M:
                    raise RuntimeError(
                        f"Specimen {specimen_id} final base error is {base_error} m"
                    )

        try:
            while simulation_app.is_running() and not panel_state["stop_requested"]:
                step_started = time.perf_counter()
                command = commands[command_index]
                specimen_id = int(command["specimen_id"])
                positions, velocities = full_state(command)
                apply_kinematic_state(positions, velocities)
                tcp_matrix = single.prim_world_matrix(stage, tcp_prim_path)

                if attached_specimen is not None:
                    payload_ops[attached_specimen].Set(tcp_matrix)
                event = command.get("attachment_event")
                if event == "attach":
                    if attached_specimen is not None:
                        raise RuntimeError("A new payload attached before the previous release")
                    payload_ops[specimen_id].Set(tcp_matrix)
                    attached_specimen = specimen_id
                    attach_events[specimen_id] += 1
                elif event == "release":
                    if attached_specimen != specimen_id:
                        raise RuntimeError("Release event does not match the attached specimen")
                    destination_matrix = destination_matrices[specimen_id]
                    release_tcp_position_errors[specimen_id].append(
                        float(
                            np.linalg.norm(
                                single.matrix_translation(tcp_matrix)
                                - single.matrix_translation(destination_matrix)
                            )
                        )
                    )
                    release_tcp_orientation_errors[specimen_id].append(
                        single.matrix_rotation_error_radians(tcp_matrix, destination_matrix)
                    )
                    # The stored destination is the exact endpoint of the reviewed
                    # vertical TCP stage.  Snap only at release so the free visual
                    # payload has exact +Z pin-axis and planned-base evidence.
                    payload_ops[specimen_id].Set(destination_matrix)
                    attached_specimen = None
                    release_events[specimen_id] += 1

                payload_matrix = single.prim_world_matrix(
                    stage, f"{config['payload']['root_prim_prefix']}{specimen_id}"
                )
                if attached_specimen == specimen_id:
                    transform_error = float(
                        np.max(
                            np.abs(
                                np.asarray(payload_matrix, dtype=np.float64)
                                - np.asarray(tcp_matrix, dtype=np.float64)
                            )
                        )
                    )
                    maximum_attached_transform_error = max(
                        maximum_attached_transform_error, transform_error
                    )
                if command["phase"] in {"lift_tilted", "hold_lift_tilted"}:
                    displacement = float(
                        np.linalg.norm(
                            single.matrix_translation(payload_matrix)
                            - single.matrix_translation(initial_matrices[specimen_id])
                        )
                    )
                    lift_displacements[specimen_id] = max(
                        lift_displacements[specimen_id], displacement
                    )

                readback = np.asarray(robot.get_joint_positions(), dtype=np.float64)
                if readback.shape != (8,) or not np.all(np.isfinite(readback)):
                    raise RuntimeError("Articulated joint readback is invalid")
                error = np.abs(readback - positions)
                maximum_readback_error = max(maximum_readback_error, float(np.max(error)))
                maximum_arm_readback_error = max(
                    maximum_arm_readback_error,
                    max(float(error[dof_index[name]]) for name in ARM_JOINT_NAMES),
                )
                leader_value = float(readback[dof_index[gripper["leader_joint"]]])
                follower_value = float(readback[dof_index[gripper["mimic_joint"]]])
                finger_target = float(command["finger_position_m"])
                maximum_finger_readback_error = max(
                    maximum_finger_readback_error,
                    abs(leader_value - finger_target),
                    abs(follower_value - finger_target),
                )
                maximum_finger_mirror_error = max(
                    maximum_finger_mirror_error, abs(leader_value - follower_value)
                )
                observed_finger_min = min(observed_finger_min, leader_value, follower_value)
                observed_finger_max = max(observed_finger_max, leader_value, follower_value)

                render_this_step = render_animation and step_count % render_interval == 0
                if render_this_step:
                    world.render()
                    rendered_frames += 1
                phase_key = (specimen_id, str(command["phase"]))
                if phase_key != last_phase_key:
                    specimen_number = int(command.get("specimen_index", specimen_id - 1)) + 1
                    print(
                        f"Isaac multi-pin cycle: specimen {specimen_number}/7 "
                        f"{command['phase']}"
                    )
                    if status_label is not None:
                        gap_mm = (
                            float(gripper["open_gap_m"])
                            - 2.0 * float(command["finger_position_m"])
                        ) * 1000.0
                        status_label.text = (
                            f"specimen {specimen_number}/7 | {command['phase']} | "
                            f"cycle {completed_cycles + 1} | jaw gap {gap_mm:.1f} mm | "
                            f"{'ATTACHED' if attached_specimen is not None else 'FREE'}"
                        )
                    last_phase_key = phase_key

                is_last_specimen = specimen_id == EXPECTED_SPECIMEN_IDS[-1]
                if (
                    screenshot_path is not None
                    and not args.headless
                    and not screenshot_requested
                    and is_last_specimen
                    and command["phase"] == config["viewer"]["screenshot_phase"]
                    and render_this_step
                ):
                    set_camera_view(args.camera_view)
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
                    if attached_specimen is not None:
                        raise RuntimeError("Seven-specimen cycle ended with a payload attached")
                    validate_final_payloads()
                    completed_cycles += 1
                    command_index = 0
                    if target_cycles > 0 and completed_cycles >= target_cycles:
                        break
                    reset_payloads()
                    last_phase_key = None
                if not args.no_realtime:
                    remaining = control_dt - (time.perf_counter() - step_started)
                    if remaining > 0.0:
                        time.sleep(remaining)
        except KeyboardInterrupt:
            interrupted = True
            print("Isaac multi-pin cycle interrupted; writing available evidence")

        if args.headless and screenshot_path is not None:
            viewport_ready, waited_frames = ViewportManager.wait_for_viewport(max_frames=120)
            if not viewport_ready:
                raise RuntimeError(
                    f"Isaac viewport was not ready after {waited_frames} frames"
                )
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
            screenshot_phase_observed = config["viewer"]["screenshot_phase"]
            for _ in range(90):
                if screenshot_future.done() or not simulation_app.is_running():
                    break
                world.render()
                rendered_frames += 1
            if not screenshot_future.done():
                raise RuntimeError("Final upright screenshot did not finish within 90 frames")

        screenshot_wait_result = None
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
                f"Stopped after {completed_cycles}/{target_cycles} complete seven-pin cycles"
            )
        if completed_cycles < 1:
            raise RuntimeError("No complete seven-specimen cycle was observed")
        if maximum_readback_error > MAX_KINEMATIC_READBACK_ERROR:
            raise RuntimeError(
                f"Kinematic readback exceeded tolerance: {maximum_readback_error}"
            )
        if maximum_finger_mirror_error > MAX_FINGER_MIRROR_ERROR_M:
            raise RuntimeError(
                f"2FG7 finger mirror error exceeded tolerance: {maximum_finger_mirror_error}"
            )
        if maximum_attached_transform_error > MAX_ATTACHED_TRANSFORM_ERROR:
            raise RuntimeError(
                "Visual TCP following exceeded tolerance: "
                f"{maximum_attached_transform_error}"
            )
        if not math.isclose(
            observed_finger_min,
            float(gripper["open_position_m"]),
            abs_tol=MAX_FINGER_MIRROR_ERROR_M,
        ) or not math.isclose(
            observed_finger_max,
            float(gripper["closed_position_m"]),
            abs_tol=MAX_FINGER_MIRROR_ERROR_M,
        ):
            raise RuntimeError("Proof did not observe fully open and closed finger states")
        for specimen_id in EXPECTED_SPECIMEN_IDS:
            if attach_events[specimen_id] != completed_cycles:
                raise RuntimeError(f"Specimen {specimen_id} attach count is invalid")
            if release_events[specimen_id] != completed_cycles:
                raise RuntimeError(f"Specimen {specimen_id} release count is invalid")
            if lift_displacements[specimen_id] < MIN_LIFT_DISPLACEMENT_M:
                raise RuntimeError(
                    f"Specimen {specimen_id} lift was only "
                    f"{lift_displacements[specimen_id]} m"
                )
            if max(release_tcp_position_errors[specimen_id], default=math.inf) > (
                MAX_RELEASE_TCP_POSITION_ERROR_M
            ):
                raise RuntimeError(f"Specimen {specimen_id} missed destination TCP position")
            if max(release_tcp_orientation_errors[specimen_id], default=math.inf) > (
                MAX_RELEASE_TCP_ORIENTATION_ERROR_RAD
            ):
                raise RuntimeError(f"Specimen {specimen_id} missed destination TCP orientation")

        screenshot_written = screenshot_path.is_file() if screenshot_path else None
        if screenshot_path is not None and not screenshot_written:
            raise RuntimeError(f"Screenshot capture did not complete: {screenshot_path}")

        per_specimen = []
        for specimen in config["plan"]["specimens"]:
            specimen_id = int(specimen["specimen_id"])
            per_specimen.append(
                {
                    "specimen_id": specimen_id,
                    "source_detection_id": specimen.get("source_detection_id"),
                    "remaining_pin_end_z_from_pinch_m": remaining_endpoints[specimen_id],
                    "initial_axis_up": specimen["initial_axis_up"],
                    "final_axis_target": [0.0, 0.0, 1.0],
                    "base_xyz_m": specimen["base_xyz_m"],
                    "source_base_xyz_m": specimen["source_base_xyz_m"],
                    "placement_label": specimen.get("placement_label"),
                    "attach_events": attach_events[specimen_id],
                    "release_events": release_events[specimen_id],
                    "maximum_lift_displacement_m": lift_displacements[specimen_id],
                    "initial_pose_axis_error_rad": initial_pose_errors[specimen_id][
                        "axis_error_rad"
                    ],
                    "initial_pose_base_error_m": initial_pose_errors[specimen_id][
                        "base_error_m"
                    ],
                    "destination_pose_axis_error_rad": destination_pose_errors[
                        specimen_id
                    ]["axis_error_rad"],
                    "destination_pose_base_error_m": destination_pose_errors[
                        specimen_id
                    ]["base_error_m"],
                    "maximum_release_tcp_position_error_m": max(
                        release_tcp_position_errors[specimen_id]
                    ),
                    "maximum_release_tcp_orientation_error_rad": max(
                        release_tcp_orientation_errors[specimen_id]
                    ),
                    "maximum_final_axis_error_rad": max(final_axis_errors[specimen_id]),
                    "maximum_final_base_error_m": max(final_base_errors[specimen_id]),
                    "visual_prim_paths": payload_paths[specimen_id],
                }
            )

        report = {
            "format_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "mode": "isaac_only_seven_pin_verticalization",
            "command": [sys.executable, *sys.argv],
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "package_versions": package_versions,
            "config": str(config_path),
            "config_sha256": single.sha256_file(config_path),
            "multi_pin_plan": str(config["plan_path"]),
            "multi_pin_plan_sha256": config["plan_sha256"],
            "placement_policy": config["raw"]["multi_pin_plan"]["placement"],
            "planning_clearance": {
                "model": "reviewed_qc_robot_spheres_without_attached_payload_geometry",
                "required_sampled_sphere_clearance_m": config["plan"][
                    "required_sampled_sphere_clearance_m"
                ],
                "minimum_sampled_sphere_clearance_m": config["plan"]["validation"][
                    "minimum_sampled_sphere_clearance_m"
                ],
                "all_stages_accepted": config["plan"]["validation"][
                    "all_stages_accepted"
                ],
                "sampled_self_collision": config["plan"]["validation"][
                    "sampled_self_collision"
                ],
                "attached_payload_geometry_validated": False,
            },
            "articulated_asset": {
                "profile": config["raw"]["articulated_asset"]["profile"],
                "usd": str(config["usd_path"]),
                "usd_sha256": single.sha256_file(config["usd_path"]),
                "import_report": str(config["import_report_path"]),
                "import_report_sha256": single.sha256_file(config["import_report_path"]),
                "staged_manifest": str(config["staged_manifest_path"]),
                "staged_manifest_sha256": single.sha256_file(
                    config["staged_manifest_path"]
                ),
                "tool_metadata": str(config["metadata_path"]),
                "tool_metadata_sha256": single.sha256_file(config["metadata_path"]),
                "asset_artifact_count": len(config["asset_artifacts"]),
                "asset_artifacts": config["asset_artifacts"],
                "dof_names": list(robot.dof_names),
                "dof_count": robot.num_dof,
            },
            "cycle": evidence,
            "completed_cycles": completed_cycles,
            "specimen_count": 7,
            "specimens": per_specimen,
            "total_attach_events": sum(attach_events.values()),
            "total_release_events": sum(release_events.values()),
            "all_final_axes_world_positive_z": True,
            "all_final_bases_at_planned_destinations": True,
            "physics_dynamics_stepped": False,
            "motion_mode": "paused_physx_kinematic_joint_animation",
            "arm_joint_readback_max_error": maximum_arm_readback_error,
            "all_joint_readback_max_error": maximum_readback_error,
            "finger_joint_readback_max_error_m": maximum_finger_readback_error,
            "finger_mirror_max_error_m": maximum_finger_mirror_error,
            "finger_open_position_observed_m": observed_finger_min,
            "finger_closed_position_observed_m": observed_finger_max,
            "finger_motion_is_articulated": True,
            "payload_attachment_mode": "kinematic_visual_follow",
            "maximum_attached_transform_error": maximum_attached_transform_error,
            "contact_physics_simulated": False,
            "payload_collision_enabled": False,
            "payload_rigid_body_enabled": False,
            "payload_mass_api_enabled": False,
            "scene_paths": scene_paths,
            "suppressed_duplicate_truth_proxy_count": len(suppressed_scene_proxies),
            "suppressed_duplicate_truth_proxies": suppressed_scene_proxies,
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
            "screenshot": str(screenshot_path) if screenshot_path else None,
            "screenshot_phase": screenshot_phase_observed,
            "screenshot_written": screenshot_written,
            "screenshot_sha256": (
                single.sha256_file(screenshot_path) if screenshot_path else None
            ),
            "screenshot_dimensions": (
                single.png_dimensions(screenshot_path) if screenshot_path else None
            ),
            "screenshot_capture_scheduled": screenshot_capture is not None,
            "screenshot_wait_result": screenshot_wait_result,
            "wall_seconds": time.perf_counter() - started_wall,
            "warning": (
                "Isaac visual choreography only. The source plan validates reviewed QC "
                "robot-sphere clearance, but attached-payload geometry, contact, friction, "
                "force, physical TCP calibration, controller timing, and grasp success "
                "remain unvalidated."
            ),
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Isaac multi-pin report: {report_path}")
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
