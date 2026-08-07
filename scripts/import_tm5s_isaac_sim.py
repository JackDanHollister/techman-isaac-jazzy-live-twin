#!/usr/bin/env python3
"""Validate a known TM5S + 2FG7 URDF profile in Isaac Sim 6.0.1.

Run this only with the isolated Isaac Sim Python 3.12 environment.  The
wrapper refuses to start until the NVIDIA Omniverse EULA has been explicitly
accepted.  The script is headless, creates no ROS connections, and never
commands Watson.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import NamedTuple
from urllib.parse import unquote, urlparse


ARENA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_URDF = ARENA_DIR / "generated/cumotion/tm5s_with_2fg7.urdf"
GENERATED_ISAAC_ROOT = (ARENA_DIR / "generated/isaac").resolve()
DEFAULT_OUTPUT_DIR = GENERATED_ISAAC_ROOT / "6.0.1"
DEFAULT_REPORT = ARENA_DIR / "outputs/isaac_sim/6.0.1/import_report.json"
PHYSICS_DT_SECONDS = 1.0 / 60.0
EXPECTED_PYTHON = (3, 12)
EXPECTED_ISAAC_PACKAGES = {
    "isaacsim": "6.0.1.0",
    "isaacsim-asset": "6.0.1.0",
    "isaacsim-core": "6.0.1.0",
}
EXPECTED_JOINTS = frozenset(f"joint_{index}" for index in range(1, 7))
ARTICULATED_FINGER_JOINTS = frozenset(
    {
        "onrobot_2fg7_left_finger_joint",
        "onrobot_2fg7_right_finger_joint",
    }
)
LEGACY_EXPECTED_LINKS = frozenset(
    {
        "base",
        "flange",
        "gripper_tcp",
        *(f"link_{index}" for index in range(7)),
        "onrobot_2fg7_base_link",
        "onrobot_2fg7_left_finger_link",
        "onrobot_2fg7_right_finger_link",
    }
)
CORE_JOINT_BODIES = {
    **{
        f"joint_{index}": (f"link_{index - 1}", f"link_{index}")
        for index in range(1, 7)
    },
    "flange_fixed_joint": ("link_6", "flange"),
}
LEGACY_JOINT_BODIES = {
    **CORE_JOINT_BODIES,
    "base_fixed_joint": ("base", "link_0"),
    "flange_to_onrobot_2fg7_base": ("flange", "onrobot_2fg7_base_link"),
    "onrobot_2fg7_left_finger_joint": (
        "onrobot_2fg7_base_link",
        "onrobot_2fg7_left_finger_link",
    ),
    "onrobot_2fg7_right_finger_joint": (
        "onrobot_2fg7_base_link",
        "onrobot_2fg7_right_finger_link",
    ),
    "onrobot_2fg7_base_to_gripper_tcp": (
        "onrobot_2fg7_base_link",
        "gripper_tcp",
    ),
}
WATSON_QC_EXPECTED_LINKS = frozenset(
    {
        "base",
        "flange",
        "finger_tip_plane",
        "gripper_tcp",
        "onrobot_2fg7_base_link",
        "onrobot_2fg7_left_finger_link",
        "onrobot_2fg7_origin",
        "onrobot_2fg7_right_finger_link",
        "onrobot_nominal_tcp",
        "onrobot_qc_robot_side_link",
        "pin_grasp_tcp",
        *(f"link_{index}" for index in range(7)),
    }
)
WATSON_QC_JOINT_BODIES = {
    **CORE_JOINT_BODIES,
    "base_fixed_joint": ("base", "link_0"),
    "flange_to_onrobot_qc_robot_side": ("flange", "onrobot_qc_robot_side_link"),
    "onrobot_qc_robot_side_to_2fg7_origin": (
        "onrobot_qc_robot_side_link",
        "onrobot_2fg7_origin",
    ),
    "onrobot_2fg7_origin_to_base": (
        "onrobot_2fg7_origin",
        "onrobot_2fg7_base_link",
    ),
    "onrobot_2fg7_left_finger_joint": (
        "onrobot_2fg7_base_link",
        "onrobot_2fg7_left_finger_link",
    ),
    "onrobot_2fg7_right_finger_joint": (
        "onrobot_2fg7_base_link",
        "onrobot_2fg7_right_finger_link",
    ),
    "onrobot_2fg7_origin_to_onrobot_nominal_tcp": (
        "onrobot_2fg7_origin",
        "onrobot_nominal_tcp",
    ),
    "onrobot_2fg7_base_link_to_finger_tip_plane": (
        "onrobot_2fg7_base_link",
        "finger_tip_plane",
    ),
    "onrobot_2fg7_origin_to_gripper_tcp": (
        "onrobot_2fg7_origin",
        "gripper_tcp",
    ),
    "gripper_tcp_to_pin_grasp_tcp": ("gripper_tcp", "pin_grasp_tcp"),
}


class ValidationProfile(NamedTuple):
    """Exact source topology and allowed Isaac importer representation."""

    name: str
    expected_links: frozenset[str]
    expected_joint_bodies: dict[str, tuple[str, str]]
    moving_joint_types: dict[str, str]
    mimic_joints: dict[str, tuple[str, float, float]]
    allowed_collapsed_fixed_joint_origins: dict[
        str,
        tuple[tuple[float, float, float], tuple[float, float, float]],
    ]
    tcp_marker_mass_kg: float
    tcp_marker_inertia_kg_m2: float

    @property
    def expected_moving_joints(self) -> frozenset[str]:
        return frozenset(self.moving_joint_types)

    @property
    def expected_fixed_joints(self) -> frozenset[str]:
        return frozenset(self.expected_joint_bodies) - self.expected_moving_joints


ARM_MOVING_JOINT_TYPES = {joint_name: "revolute" for joint_name in EXPECTED_JOINTS}


VALIDATION_PROFILES = {
    "legacy_fixed_2fg7": ValidationProfile(
        name="legacy_fixed_2fg7",
        expected_links=LEGACY_EXPECTED_LINKS,
        expected_joint_bodies=LEGACY_JOINT_BODIES,
        moving_joint_types=ARM_MOVING_JOINT_TYPES,
        mimic_joints={},
        allowed_collapsed_fixed_joint_origins={
            "flange_fixed_joint": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        },
        tcp_marker_mass_kg=0.001,
        tcp_marker_inertia_kg_m2=0.4 * 0.001 * 0.004**2,
    ),
    "watson_qc_nominal": ValidationProfile(
        name="watson_qc_nominal",
        expected_links=WATSON_QC_EXPECTED_LINKS,
        expected_joint_bodies=WATSON_QC_JOINT_BODIES,
        moving_joint_types=ARM_MOVING_JOINT_TYPES,
        mimic_joints={},
        allowed_collapsed_fixed_joint_origins={
            "flange_fixed_joint": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            "onrobot_qc_robot_side_to_2fg7_origin": (
                (0.0, 0.0, 0.0136),
                (0.0, 0.0, 0.0),
            ),
            "onrobot_2fg7_origin_to_onrobot_nominal_tcp": (
                (0.0, 0.0, 0.125),
                (0.0, 0.0, 0.0),
            ),
            "onrobot_2fg7_base_link_to_finger_tip_plane": (
                (0.0, 0.0, 0.16225),
                (0.0, 0.0, 0.0),
            ),
            "gripper_tcp_to_pin_grasp_tcp": (
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
        },
        tcp_marker_mass_kg=1.0e-6,
        tcp_marker_inertia_kg_m2=0.4 * 1.0e-6 * 0.004**2,
    ),
    "watson_qc_articulated_2fg7": ValidationProfile(
        name="watson_qc_articulated_2fg7",
        expected_links=WATSON_QC_EXPECTED_LINKS,
        expected_joint_bodies=WATSON_QC_JOINT_BODIES,
        moving_joint_types={
            **ARM_MOVING_JOINT_TYPES,
            "onrobot_2fg7_left_finger_joint": "prismatic",
            "onrobot_2fg7_right_finger_joint": "prismatic",
        },
        mimic_joints={
            "onrobot_2fg7_right_finger_joint": (
                "onrobot_2fg7_left_finger_joint",
                1.0,
                0.0,
            )
        },
        allowed_collapsed_fixed_joint_origins={
            "flange_fixed_joint": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            "onrobot_qc_robot_side_to_2fg7_origin": (
                (0.0, 0.0, 0.0136),
                (0.0, 0.0, 0.0),
            ),
            "onrobot_2fg7_origin_to_onrobot_nominal_tcp": (
                (0.0, 0.0, 0.125),
                (0.0, 0.0, 0.0),
            ),
            "onrobot_2fg7_base_link_to_finger_tip_plane": (
                (0.0, 0.0, 0.16225),
                (0.0, 0.0, 0.0),
            ),
            "gripper_tcp_to_pin_grasp_tcp": (
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
        },
        tcp_marker_mass_kg=1.0e-6,
        tcp_marker_inertia_kg_m2=0.4 * 1.0e-6 * 0.004**2,
    ),
}
DEFAULT_VALIDATION_PROFILE = "legacy_fixed_2fg7"
MAX_JOINT_DRIFT_RADIANS = 1.0e-3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = result.stdout.strip()
    return output or None


def installed_isaac_package_versions() -> dict[str, str]:
    versions = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "")
        if name.lower().startswith("isaacsim"):
            versions[name] = distribution.version
    return dict(sorted(versions.items()))


def validate_runtime() -> dict[str, str]:
    if sys.version_info[:2] != EXPECTED_PYTHON:
        raise RuntimeError(
            "Isaac Sim 6.0.1 validation requires Python 3.12; "
            f"found {sys.version.split()[0]}"
        )
    package_versions: dict[str, str] = {}
    for package_name, expected_version in EXPECTED_ISAAC_PACKAGES.items():
        try:
            installed_version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Missing required package: {package_name}") from exc
        if installed_version != expected_version:
            raise RuntimeError(
                f"{package_name} must be {expected_version}; found {installed_version}"
            )
        package_versions[package_name] = installed_version
    return package_versions


def validate_source_urdf_topology(
    urdf_path: Path,
    profile: ValidationProfile,
) -> dict[str, object]:
    """Require the input URDF to match a reviewed profile before Isaac starts."""
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Could not parse source URDF: {urdf_path}: {exc}") from exc
    if root.tag != "robot":
        raise ValueError(f"Source URDF root must be <robot>; found <{root.tag}>")

    links = [element.get("name") for element in root.findall("link")]
    if any(not name for name in links):
        raise ValueError("Source URDF contains an unnamed top-level link")
    duplicate_links = sorted(name for name in set(links) if links.count(name) != 1)
    if duplicate_links:
        raise ValueError(f"Source URDF contains duplicate links: {duplicate_links}")

    joint_types: dict[str, str] = {}
    joint_bodies: dict[str, tuple[str, str]] = {}
    joint_mimics: dict[str, tuple[str, float, float]] = {}
    joint_origins: dict[
        str,
        tuple[tuple[float, float, float], tuple[float, float, float]],
    ] = {}

    def vector_attribute(
        element: ET.Element | None,
        name: str,
    ) -> tuple[float, float, float]:
        value = element.get(name, "0 0 0") if element is not None else "0 0 0"
        try:
            parsed = tuple(float(component) for component in value.split())
        except ValueError as exc:
            raise ValueError(f"Invalid URDF {name} vector: {value!r}") from exc
        if len(parsed) != 3:
            raise ValueError(f"URDF {name} vector must have three components: {value!r}")
        return parsed

    for element in root.findall("joint"):
        joint_name = element.get("name")
        joint_type = element.get("type")
        if not joint_name or not joint_type:
            raise ValueError("Source URDF contains an unnamed or untyped top-level joint")
        if joint_name in joint_types:
            raise ValueError(f"Source URDF contains duplicate joint: {joint_name}")
        parent = element.find("parent")
        child = element.find("child")
        parent_link = parent.get("link") if parent is not None else None
        child_link = child.get("link") if child is not None else None
        if not parent_link or not child_link:
            raise ValueError(f"Source URDF joint has incomplete topology: {joint_name}")
        joint_types[joint_name] = joint_type
        joint_bodies[joint_name] = (parent_link, child_link)
        mimic = element.find("mimic")
        if mimic is not None:
            source_joint = mimic.get("joint")
            if not source_joint:
                raise ValueError(f"Source URDF mimic joint has no source: {joint_name}")
            joint_mimics[joint_name] = (
                source_joint,
                float(mimic.get("multiplier", "1.0")),
                float(mimic.get("offset", "0.0")),
            )
        origin = element.find("origin")
        joint_origins[joint_name] = (
            vector_attribute(origin, "xyz"),
            vector_attribute(origin, "rpy"),
        )

    expected_joint_types = {
        joint_name: profile.moving_joint_types.get(joint_name, "fixed")
        for joint_name in profile.expected_joint_bodies
    }
    differences = []
    if set(links) != profile.expected_links:
        differences.append(
            "links="
            f"{sorted(set(links).symmetric_difference(profile.expected_links))}"
        )
    if joint_types != expected_joint_types:
        differences.append(
            "joint_types="
            f"expected={expected_joint_types}, actual={joint_types}"
        )
    if joint_bodies != profile.expected_joint_bodies:
        differences.append(
            "joint_bodies="
            f"expected={profile.expected_joint_bodies}, actual={joint_bodies}"
        )
    if joint_mimics != profile.mimic_joints:
        differences.append(
            "joint_mimics="
            f"expected={profile.mimic_joints}, actual={joint_mimics}"
        )
    collapsed_origin_mismatches = {
        joint_name: {
            "expected": expected_origin,
            "actual": joint_origins.get(joint_name),
        }
        for joint_name, expected_origin in (
            profile.allowed_collapsed_fixed_joint_origins.items()
        )
        if joint_origins.get(joint_name) != expected_origin
    }
    if collapsed_origin_mismatches:
        differences.append(f"collapsed_joint_origins={collapsed_origin_mismatches}")
    if differences:
        raise ValueError(
            f"Source URDF does not match validation profile {profile.name!r}: "
            + "; ".join(differences)
        )
    return {
        "profile": profile.name,
        "links": sorted(links),
        "joint_types": dict(sorted(joint_types.items())),
        "joint_bodies": {
            name: list(bodies) for name, bodies in sorted(joint_bodies.items())
        },
        "joint_mimics": {
            name: {
                "source_joint": values[0],
                "multiplier": values[1],
                "offset": values[2],
            }
            for name, values in sorted(joint_mimics.items())
        },
        "joint_origins": {
            name: {"xyz": list(origin[0]), "rpy": list(origin[1])}
            for name, origin in sorted(joint_origins.items())
        },
    }


def validate_source_urdf_meshes(urdf_path: Path) -> dict[str, object]:
    """Require every source mesh to be directly resolvable before Isaac starts."""

    root = ET.parse(urdf_path).getroot()
    resolved_paths: list[Path] = []
    package_references: list[str] = []
    unsupported_references: list[str] = []
    missing_paths: list[Path] = []
    for mesh in root.findall(".//mesh"):
        reference = str(mesh.get("filename", ""))
        if reference.startswith("package://"):
            package_references.append(reference)
            continue
        if reference.startswith("file://"):
            parsed = urlparse(reference)
            resolved = Path(unquote(parsed.path)).resolve()
        elif "://" in reference:
            unsupported_references.append(reference)
            continue
        else:
            resolved = (urdf_path.parent / reference).resolve()
        resolved_paths.append(resolved)
        if not resolved.is_file():
            missing_paths.append(resolved)
    if package_references:
        raise ValueError(
            "Source URDF contains package:// meshes that the isolated Isaac wrapper "
            "cannot resolve; import the staged cuMotion URDF instead: "
            f"{sorted(set(package_references))}"
        )
    if unsupported_references:
        raise ValueError(
            f"Source URDF contains unsupported mesh URIs: {sorted(set(unsupported_references))}"
        )
    if missing_paths:
        raise FileNotFoundError(
            f"Source URDF mesh files are missing: {[str(path) for path in missing_paths]}"
        )
    if not resolved_paths:
        raise ValueError("Source URDF contains no mesh geometry")
    return {
        "mesh_reference_count": len(resolved_paths),
        "unique_resolved_mesh_count": len(set(resolved_paths)),
        "all_meshes_resolved": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--validation-profile",
        choices=sorted(VALIDATION_PROFILES),
        default=DEFAULT_VALIDATION_PROFILE,
        help=(
            "Exact reviewed URDF topology to require before import; the legacy "
            "fixed-2FG7 profile remains the default"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Isaac Sim 6 Asset Structure 3 output directory; must be a child of "
            f"{GENERATED_ISAAC_ROOT}"
        ),
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--benchmark-steps", type=int, default=1000)
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    return parser


def validate_generated_output_dir(output_dir: Path) -> Path:
    """Keep importer cleanup inside the demo's ignored generated/isaac tree."""
    resolved_output_dir = output_dir.expanduser().resolve()
    try:
        relative_output_dir = resolved_output_dir.relative_to(GENERATED_ISAAC_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"--output-dir must stay within {GENERATED_ISAAC_ROOT}: "
            f"{resolved_output_dir}"
        ) from exc
    if not relative_output_dir.parts:
        raise ValueError(
            f"--output-dir must be a child of {GENERATED_ISAAC_ROOT}, not the root itself"
        )
    return resolved_output_dir


def main() -> int:
    args = build_parser().parse_args()
    package_versions = validate_runtime()
    urdf_path = args.urdf.resolve()
    validation_profile = VALIDATION_PROFILES[args.validation_profile]
    output_dir = validate_generated_output_dir(args.output_dir)
    report_path = args.report.resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(
            f"Prepared URDF is missing: {urdf_path}\n"
            "Run scripts/setup_cumotion_benchmark.sh first."
        )
    source_urdf_topology = validate_source_urdf_topology(
        urdf_path,
        validation_profile,
    )
    source_urdf_meshes = validate_source_urdf_meshes(urdf_path)
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.benchmark_steps <= 0:
        raise ValueError("--benchmark-steps must be positive")
    if args.benchmark_repeats <= 0:
        raise ValueError("--benchmark-repeats must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # URDF Importer 3 creates a package directory named after the source URDF.
    # Remove only that generated package so a rerun cannot validate stale USD.
    generated_package_dir = output_dir / urdf_path.stem
    if generated_package_dir.is_symlink():
        raise RuntimeError(
            f"Refusing to remove a generated-package symlink: {generated_package_dir}"
        )
    if generated_package_dir.exists() and not generated_package_dir.is_dir():
        raise RuntimeError(
            f"Expected generated package path to be a directory: {generated_package_dir}"
        )
    if generated_package_dir.is_dir():
        shutil.rmtree(generated_package_dir)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": True,
            "disable_viewport_updates": True,
            # Isaac Sim defaults multi_gpu to True.  This workstation has one
            # NVIDIA compute GPU plus an unsupported Intel display adapter, so
            # keep Kit on GPU 0 and avoid PCIe P2P setup.  The simple
            # articulation benchmark below deliberately uses CPU PhysX.
            "active_gpu": 0,
            "physics_gpu": 0,
            "multi_gpu": False,
            "max_gpu_count": 1,
            "fast_shutdown": True,
        }
    )
    exit_code = 1
    try:
        import numpy as np
        import omni.kit.app
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.prims import get_articulation_root_api_prim_path
        import isaacsim.core.utils.stage as stage_utils
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdUtils

        required_extensions = (
            "omni.scene.optimizer.core",
            "isaacsim.robot.schema",
            "isaacsim.asset.importer.urdf",
        )
        for extension_name in required_extensions:
            if not enable_extension(extension_name):
                raise RuntimeError(f"Could not enable Isaac Sim extension: {extension_name}")
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

        stage_utils.create_new_stage()
        simulation_app.update()

        import_config = URDFImporterConfig(
            urdf_path=str(urdf_path),
            usd_path=str(output_dir),
            robot_type="Manipulator",
            merge_mesh=False,
            merge_fixed_joints=False,
            collision_from_visuals=False,
            allow_self_collision=False,
            fix_base=True,
            joint_drive_type="acceleration",
            joint_target_type="position",
            override_joint_stiffness=1000.0,
            override_joint_damping=100.0,
            run_asset_transformer=True,
            run_multi_physics_conversion=True,
        )
        import_started = time.perf_counter()
        output_path = Path(URDFImporter(import_config).import_urdf()).resolve()
        import_elapsed_seconds = time.perf_counter() - import_started
        expected_output_path = (
            output_dir / urdf_path.stem / f"{urdf_path.stem}.usda"
        ).resolve()
        if output_path != expected_output_path:
            raise RuntimeError(
                "URDF importer returned an unexpected Asset Structure 3 path: "
                f"expected={expected_output_path}, actual={output_path}"
            )
        if not output_path.is_file():
            raise RuntimeError(f"URDF importer returned a missing USD: {output_path}")
        try:
            output_path.relative_to(output_dir)
        except ValueError as exc:
            raise RuntimeError(
                f"URDF importer wrote outside the requested directory: {output_path}"
            ) from exc
        simulation_app.update()
        if output_path.stat().st_size == 0:
            raise RuntimeError(f"Isaac Sim wrote an empty USD: {output_path}")

        stage = Usd.Stage.Open(str(output_path))
        if stage is None:
            raise RuntimeError(f"Isaac Sim wrote an unreadable USD: {output_path}")
        default_prim = stage.GetDefaultPrim()
        if not default_prim.IsValid():
            raise RuntimeError("Imported USD has no default robot prim")
        default_prim_path = str(default_prim.GetPath())
        physics_variant_selection = (
            default_prim.GetVariantSets().GetVariantSet("Physics").GetVariantSelection()
        )
        if physics_variant_selection != "physx":
            raise RuntimeError(
                "Imported asset did not select its PhysX variant: "
                f"{physics_variant_selection!r}"
            )
        stage_meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        if not np.isclose(stage_meters_per_unit, 1.0):
            raise RuntimeError(
                f"Imported USD stage units are not metres: {stage_meters_per_unit}"
            )
        _, _, unresolved_dependencies = UsdUtils.ComputeAllDependencies(str(output_path))
        unresolved_dependencies = sorted(str(path) for path in unresolved_dependencies)
        if unresolved_dependencies:
            raise RuntimeError(
                f"Imported USD has unresolved dependencies: {unresolved_dependencies}"
            )

        stage_up_axis = UsdGeom.GetStageUpAxis(stage)
        if stage_up_axis != UsdGeom.Tokens.z:
            raise RuntimeError(f"Imported USD is not Z-up: {stage_up_axis}")

        prims = list(stage.Traverse())
        joint_prims = [prim for prim in prims if prim.IsA(UsdPhysics.Joint)]
        duplicate_joint_names = sorted(
            name
            for name in {prim.GetName() for prim in joint_prims}
            if sum(prim.GetName() == name for prim in joint_prims) != 1
        )
        if duplicate_joint_names:
            raise RuntimeError(f"Imported USD has duplicate joint names: {duplicate_joint_names}")
        joint_prim_by_name = {prim.GetName(): prim for prim in joint_prims}
        moving_joint_prims = [
            prim
            for prim in joint_prims
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)
        ]
        moving_joints = sorted(prim.GetName() for prim in moving_joint_prims)
        unexpected_or_missing_joints = sorted(
            validation_profile.expected_moving_joints.symmetric_difference(moving_joints)
        )
        if unexpected_or_missing_joints:
            raise RuntimeError(
                "Imported moving joint set does not match the validation profile: "
                f"found={moving_joints}, difference={unexpected_or_missing_joints}"
            )

        articulation_roots = sorted(
            str(prim.GetPath())
            for prim in prims
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        )
        if len(articulation_roots) != 1:
            raise RuntimeError(
                f"Expected one articulation root; found {len(articulation_roots)}: "
                f"{articulation_roots}"
            )

        expected_link_paths: dict[str, str] = {}
        missing_links = []
        ambiguous_links: dict[str, list[str]] = {}
        for link_name in sorted(validation_profile.expected_links):
            matching_xforms = [
                prim
                for prim in prims
                if prim.GetName() == link_name and prim.IsA(UsdGeom.Xform)
            ]
            if not matching_xforms:
                missing_links.append(link_name)
            elif len(matching_xforms) > 1:
                ambiguous_links[link_name] = [str(prim.GetPath()) for prim in matching_xforms]
            else:
                expected_link_paths[link_name] = str(matching_xforms[0].GetPath())
        if missing_links:
            raise RuntimeError(f"Isaac import did not preserve required links: {missing_links}")
        if ambiguous_links:
            raise RuntimeError(f"Isaac import produced ambiguous link Xforms: {ambiguous_links}")

        tcp_prim = stage.GetPrimAtPath(expected_link_paths["gripper_tcp"])
        if not tcp_prim.HasAPI(UsdPhysics.MassAPI):
            raise RuntimeError("gripper_tcp marker has no explicit mass properties")
        tcp_mass_api = UsdPhysics.MassAPI(tcp_prim)
        tcp_marker_mass_kg = float(tcp_mass_api.GetMassAttr().Get())
        tcp_marker_diagonal_inertia = np.asarray(
            tcp_mass_api.GetDiagonalInertiaAttr().Get(), dtype=np.float64
        )
        if not np.isclose(
            tcp_marker_mass_kg,
            validation_profile.tcp_marker_mass_kg,
            rtol=1.0e-6,
        ):
            raise RuntimeError(f"Unexpected gripper_tcp marker mass: {tcp_marker_mass_kg}")
        if tcp_marker_diagonal_inertia.shape != (3,) or not np.allclose(
            tcp_marker_diagonal_inertia,
            validation_profile.tcp_marker_inertia_kg_m2,
            rtol=1.0e-5,
        ):
            raise RuntimeError(
                "Unexpected gripper_tcp marker inertia: "
                f"{tcp_marker_diagonal_inertia.tolist()}"
            )

        fixed_joint_prims = [prim for prim in joint_prims if prim.IsA(UsdPhysics.FixedJoint)]
        fixed_joint_names = {prim.GetName() for prim in fixed_joint_prims}
        required_explicit_fixed_joints = (
            validation_profile.expected_fixed_joints
            - validation_profile.allowed_collapsed_fixed_joint_origins.keys()
        )
        missing_fixed_joints = sorted(required_explicit_fixed_joints - fixed_joint_names)
        if missing_fixed_joints:
            raise RuntimeError(f"Isaac import lost fixed joints: {missing_fixed_joints}")

        # URDF-to-USD ghost-link simplification may omit a reviewed fixed joint
        # while retaining its child anchor Xform.  Verify the exact reviewed
        # source transform rather than accepting arbitrary omission.
        collapsed_fixed_joints = []
        for joint_name, (expected_xyz, expected_rpy) in sorted(
            validation_profile.allowed_collapsed_fixed_joint_origins.items()
        ):
            if joint_name in fixed_joint_names:
                continue
            parent_link, child_link = validation_profile.expected_joint_bodies[joint_name]
            child_prim = stage.GetPrimAtPath(expected_link_paths[child_link])
            if str(child_prim.GetParent().GetPath()) != expected_link_paths[parent_link]:
                raise RuntimeError(
                    f"Collapsed {joint_name} child is not directly beneath {parent_link}"
                )
            child_xformable = UsdGeom.Xformable(child_prim)
            child_local_transform = child_xformable.GetLocalTransformation()
            if expected_rpy != (0.0, 0.0, 0.0):
                raise RuntimeError(
                    f"Collapsed {joint_name} has an unsupported non-zero reviewed RPY"
                )
            expected_transform = Gf.Matrix4d(1.0)
            expected_transform.SetTranslate(Gf.Vec3d(*expected_xyz))
            if child_xformable.GetResetXformStack() or not Gf.IsClose(
                child_local_transform, expected_transform, 1.0e-9
            ):
                raise RuntimeError(
                    f"Collapsed {joint_name} did not preserve its reviewed transform"
                )
            collapsed_fixed_joints.append(joint_name)

        joint_body_names: dict[str, list[str]] = {}
        for joint_name, expected_bodies in validation_profile.expected_joint_bodies.items():
            if joint_name == "base_fixed_joint":
                continue
            if (
                joint_name
                in validation_profile.allowed_collapsed_fixed_joint_origins
                and joint_name not in joint_prim_by_name
            ):
                continue
            prim = joint_prim_by_name.get(joint_name)
            if prim is None:
                raise RuntimeError(f"Imported USD is missing joint topology for {joint_name}")
            joint = UsdPhysics.Joint(prim)
            body_0 = joint.GetBody0Rel().GetTargets()
            body_1 = joint.GetBody1Rel().GetTargets()
            if len(body_0) != 1 or len(body_1) != 1:
                raise RuntimeError(
                    f"Joint {joint_name} does not have exactly one body on each side"
                )
            actual_bodies = (body_0[0].name, body_1[0].name)
            if actual_bodies != expected_bodies:
                raise RuntimeError(
                    f"Joint {joint_name} topology mismatch: "
                    f"expected={expected_bodies}, actual={actual_bodies}"
                )
            joint_body_names[joint_name] = list(actual_bodies)

        base_fixed_joint = UsdPhysics.Joint(joint_prim_by_name["base_fixed_joint"])
        base_body_0 = base_fixed_joint.GetBody0Rel().GetTargets()
        base_body_1 = base_fixed_joint.GetBody1Rel().GetTargets()
        if base_body_0 != [default_prim.GetPath()] or [str(path) for path in base_body_1] != [
            expected_link_paths["link_0"]
        ]:
            raise RuntimeError(
                "base_fixed_joint does not connect the asset root to TM5S link_0"
            )

        def nearest_rigid_body_path(path: object) -> str | None:
            prim = stage.GetPrimAtPath(path)
            while prim.IsValid() and not prim.IsPseudoRoot():
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    return str(prim.GetPath())
                prim = prim.GetParent()
            return None

        world_anchor_joints = []
        for prim in fixed_joint_prims:
            joint = UsdPhysics.Joint(prim)
            body_0 = joint.GetBody0Rel().GetTargets()
            body_1 = joint.GetBody1Rel().GetTargets()
            rigid_body_0 = nearest_rigid_body_path(body_0[0]) if body_0 else None
            rigid_body_1 = nearest_rigid_body_path(body_1[0]) if body_1 else None
            if (rigid_body_0 is None) != (rigid_body_1 is None):
                world_anchor_joints.append(str(prim.GetPath()))
        expected_world_anchor = str(joint_prim_by_name["base_fixed_joint"].GetPath())
        if world_anchor_joints != [expected_world_anchor]:
            raise RuntimeError(
                "fix_base=True did not produce exactly the expected world-to-link_0 anchor: "
                f"{world_anchor_joints}"
            )

        imported_mimic_joints: dict[str, dict[str, object]] = {}
        for prim in moving_joint_prims:
            relationship = prim.GetRelationship("newton:mimicJoint")
            targets = relationship.GetTargets() if relationship.IsValid() else []
            if not targets:
                continue
            coef_0_attr = prim.GetAttribute("newton:mimicCoef0")
            coef_1_attr = prim.GetAttribute("newton:mimicCoef1")
            imported_mimic_joints[prim.GetName()] = {
                "source_joint_path": str(targets[0]),
                "source_joint": targets[0].name,
                "multiplier": float(coef_1_attr.Get()),
                "offset": float(coef_0_attr.Get()),
            }
        expected_imported_mimics = {
            joint_name: {
                "source_joint": values[0],
                "multiplier": values[1],
                "offset": values[2],
            }
            for joint_name, values in validation_profile.mimic_joints.items()
        }
        actual_imported_mimics = {
            joint_name: {
                "source_joint": values["source_joint"],
                "multiplier": values["multiplier"],
                "offset": values["offset"],
            }
            for joint_name, values in imported_mimic_joints.items()
        }
        if actual_imported_mimics != expected_imported_mimics:
            raise RuntimeError(
                "Imported mimic joints do not match the validation profile: "
                f"expected={expected_imported_mimics}, actual={actual_imported_mimics}"
            )

        drive_validation: dict[str, dict[str, float | str | bool]] = {}
        expected_angular_stiffness = float(np.deg2rad(1000.0))
        expected_angular_damping = float(np.deg2rad(100.0))
        for prim in moving_joint_prims:
            drive_axis = "linear" if prim.IsA(UsdPhysics.PrismaticJoint) else "angular"
            is_mimic = prim.GetName() in validation_profile.mimic_joints
            if not prim.HasAPI(UsdPhysics.DriveAPI, drive_axis):
                if is_mimic:
                    drive_validation[prim.GetName()] = {
                        "axis": drive_axis,
                        "mimic_controlled": True,
                        "independent_drive": False,
                    }
                    continue
                raise RuntimeError(
                    f"Moving joint has no {drive_axis} drive: {prim.GetName()}"
                )
            drive = UsdPhysics.DriveAPI(prim, drive_axis)
            drive_type = drive.GetTypeAttr().Get()
            stiffness = float(drive.GetStiffnessAttr().Get())
            damping = float(drive.GetDampingAttr().Get())
            if drive_type != "acceleration":
                raise RuntimeError(
                    f"Moving joint drive is not acceleration mode: {prim.GetName()}={drive_type}"
                )
            if drive_axis == "angular":
                if not np.isclose(stiffness, expected_angular_stiffness, rtol=1.0e-5):
                    raise RuntimeError(
                        f"Unexpected drive stiffness for {prim.GetName()}: {stiffness}"
                    )
                if not np.isclose(damping, expected_angular_damping, rtol=1.0e-5):
                    raise RuntimeError(
                        f"Unexpected drive damping for {prim.GetName()}: {damping}"
                    )
            elif not math.isfinite(stiffness) or not math.isfinite(damping):
                raise RuntimeError(
                    f"Prismatic drive gains must be finite for {prim.GetName()}"
                )
            drive_validation[prim.GetName()] = {
                "axis": drive_axis,
                "type": drive_type,
                "stiffness_usd_per_degree": stiffness,
                "damping_usd_per_degree": damping,
                "mimic_controlled": is_mimic,
                "independent_drive": not is_mimic,
            }

        robot_link_targets = [
            str(path)
            for path in default_prim.GetRelationship("isaac:physics:robotLinks").GetTargets()
        ]
        robot_joint_targets = [
            str(path)
            for path in default_prim.GetRelationship("isaac:physics:robotJoints").GetTargets()
        ]
        physical_rigid_body_paths = sorted(
            str(prim.GetPath())
            for prim in prims
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        )
        physical_joint_paths = sorted(str(prim.GetPath()) for prim in joint_prims)
        robot_schema_missing_link_api_paths = sorted(
            str(prim.GetPath())
            for prim in prims
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and "IsaacLinkAPI" not in prim.GetAppliedSchemas()
        )
        robot_schema_missing_joint_api_paths = sorted(
            str(prim.GetPath())
            for prim in joint_prims
            if "IsaacJointAPI" not in prim.GetAppliedSchemas()
        )
        robot_schema_api_coverage_incomplete = bool(
            robot_schema_missing_link_api_paths or robot_schema_missing_joint_api_paths
        )
        geometry_prims = list(Usd.PrimRange(default_prim, Usd.TraverseInstanceProxies()))
        mesh_prims = [prim for prim in geometry_prims if prim.IsA(UsdGeom.Mesh)]
        empty_mesh_face_paths = sorted(
            str(prim.GetPath())
            for prim in mesh_prims
            if not UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get()
        )
        empty_mesh_point_paths = sorted(
            str(prim.GetPath())
            for prim in mesh_prims
            if not UsdGeom.Mesh(prim).GetPointsAttr().Get()
        )
        mesh_count = len(mesh_prims)
        collision_prim_paths = sorted(
            str(prim.GetPath())
            for prim in geometry_prims
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        )
        if mesh_count == 0:
            raise RuntimeError("Imported USD contains no resolved mesh geometry")
        if empty_mesh_face_paths or empty_mesh_point_paths:
            raise RuntimeError(
                "Imported USD contains empty meshes: "
                f"faces={empty_mesh_face_paths}, points={empty_mesh_point_paths}"
            )
        if not collision_prim_paths:
            raise RuntimeError("Imported USD contains no collision-enabled prims")

        # Reopen the saved asset and require the profile's exact live PhysX articulation.
        stage = None
        World.clear_instance()
        if not stage_utils.open_stage(str(output_path)):
            raise RuntimeError("Isaac Sim could not reopen the generated USD")
        simulation_app.update()
        live_stage = omni.usd.get_context().get_stage()
        asset_prim_path = str(live_stage.GetDefaultPrim().GetPath())
        resolved_articulation_path = get_articulation_root_api_prim_path(asset_prim_path)
        world = World(
            physics_dt=PHYSICS_DT_SECONDS,
            rendering_dt=PHYSICS_DT_SECONDS,
            stage_units_in_meters=1.0,
            backend="numpy",
            device="cpu",
        )
        robot = world.scene.add(
            SingleArticulation(
                prim_path=asset_prim_path,
                name="tm5s_with_2fg7",
            )
        )
        world.reset()
        world.step(render=False)
        if not robot.handles_initialized:
            raise RuntimeError("PhysX articulation handles did not initialize")
        if (
            robot.num_dof != len(validation_profile.expected_moving_joints)
            or set(robot.dof_names) != validation_profile.expected_moving_joints
        ):
            raise RuntimeError(
                f"Unexpected live PhysX DOFs: count={robot.num_dof}, names={robot.dof_names}"
            )
        initial_joint_positions = np.asarray(robot.get_joint_positions(), dtype=np.float64)
        if initial_joint_positions.shape != (robot.num_dof,) or not np.all(
            np.isfinite(initial_joint_positions)
        ):
            raise RuntimeError(f"Invalid initial PhysX joint state: {initial_joint_positions}")
        for _ in range(args.warmup_steps):
            world.step(render=False)
        warmup_joint_positions = np.asarray(robot.get_joint_positions(), dtype=np.float64)
        if not np.all(np.isfinite(warmup_joint_positions)):
            raise RuntimeError("PhysX articulation became invalid during warmup")
        physics_dt_seconds = float(world.get_physics_dt())
        benchmark_elapsed_seconds = []
        benchmark_step_deltas = []
        simulated_seconds_per_repeat = []
        for _ in range(args.benchmark_repeats):
            start_step_index = int(world.current_time_step_index)
            start_simulation_time = float(world.current_time)
            benchmark_started = time.perf_counter()
            for _ in range(args.benchmark_steps):
                world.step(render=False)
            benchmark_elapsed_seconds.append(time.perf_counter() - benchmark_started)
            step_delta = int(world.current_time_step_index) - start_step_index
            simulated_seconds = float(world.current_time) - start_simulation_time
            if step_delta != args.benchmark_steps:
                raise RuntimeError(
                    "PhysX benchmark did not advance the requested number of steps: "
                    f"expected={args.benchmark_steps}, actual={step_delta}"
                )
            if not np.isclose(
                simulated_seconds,
                step_delta * physics_dt_seconds,
                rtol=1.0e-9,
                atol=1.0e-6,
            ):
                raise RuntimeError(
                    "PhysX benchmark time did not match the observed step advancement: "
                    f"steps={step_delta}, seconds={simulated_seconds}"
                )
            benchmark_step_deltas.append(step_delta)
            simulated_seconds_per_repeat.append(simulated_seconds)
        final_joint_positions = np.asarray(robot.get_joint_positions(), dtype=np.float64)
        if not np.all(np.isfinite(final_joint_positions)):
            raise RuntimeError("PhysX articulation became numerically invalid during smoke steps")
        joint_drift_radians = np.abs(final_joint_positions - warmup_joint_positions)
        max_joint_drift_radians = float(np.max(joint_drift_radians))
        if max_joint_drift_radians > MAX_JOINT_DRIFT_RADIANS:
            raise RuntimeError(
                "PhysX articulation drifted beyond the idle smoke-test limit: "
                f"{max_joint_drift_radians} rad"
            )
        world.stop()

        steps_per_second = [
            step_delta / elapsed
            for step_delta, elapsed in zip(benchmark_step_deltas, benchmark_elapsed_seconds)
        ]
        real_time_factor = [
            simulated_seconds / elapsed
            for simulated_seconds, elapsed in zip(
                simulated_seconds_per_repeat, benchmark_elapsed_seconds
            )
        ]
        asset_paths = sorted(
            path
            for path in output_path.parent.rglob("*")
            if path.is_file()
        )
        asset_files = [str(path.relative_to(output_dir)) for path in asset_paths]
        asset_artifacts = {
            str(path.relative_to(output_dir)): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in asset_paths
        }

        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv],
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "package_versions": package_versions,
            "installed_isaac_package_versions": installed_isaac_package_versions(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
            "validation_scope": "standalone_asset_and_physx_only",
            "validation_profile": validation_profile.name,
            "source_urdf_topology": source_urdf_topology,
            "source_urdf_meshes": source_urdf_meshes,
            "source_urdf": str(urdf_path),
            "source_urdf_sha256": sha256_file(urdf_path),
            "output_directory": str(output_dir),
            "output_usd": str(output_path),
            "output_usd_sha256": sha256_file(output_path),
            "import_elapsed_seconds": import_elapsed_seconds,
            "asset_files": asset_files,
            "asset_artifacts": asset_artifacts,
            "default_prim": default_prim_path,
            "physics_variant_selection": physics_variant_selection,
            "stage_meters_per_unit": stage_meters_per_unit,
            "stage_up_axis": stage_up_axis,
            "unresolved_dependencies": unresolved_dependencies,
            "articulation_roots": articulation_roots,
            "moving_joints": moving_joints,
            "dof_count": len(moving_joints),
            "fixed_joints": sorted(fixed_joint_names),
            "collapsed_fixed_joints": collapsed_fixed_joints,
            "collapsed_identity_fixed_joints": [
                joint_name
                for joint_name in collapsed_fixed_joints
                if validation_profile.allowed_collapsed_fixed_joint_origins[joint_name]
                == ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            ],
            "joint_body_names": joint_body_names,
            "imported_mimic_joints": imported_mimic_joints,
            "drive_validation": drive_validation,
            "world_anchor_joints": world_anchor_joints,
            "expected_link_paths": expected_link_paths,
            "tcp_marker_mass_kg": tcp_marker_mass_kg,
            "tcp_marker_diagonal_inertia_kg_m2": tcp_marker_diagonal_inertia.tolist(),
            "robot_schema_link_targets_diagnostic_only": robot_link_targets,
            "robot_schema_joint_targets_diagnostic_only": robot_joint_targets,
            "physical_rigid_body_paths": physical_rigid_body_paths,
            "physical_joint_paths": physical_joint_paths,
            "robot_schema_missing_link_api_paths": robot_schema_missing_link_api_paths,
            "robot_schema_missing_joint_api_paths": robot_schema_missing_joint_api_paths,
            "robot_schema_api_coverage_incomplete_observed_importer_limitation": (
                robot_schema_api_coverage_incomplete
            ),
            "mesh_count": mesh_count,
            "empty_mesh_face_paths": empty_mesh_face_paths,
            "empty_mesh_point_paths": empty_mesh_point_paths,
            "collision_prim_count": len(collision_prim_paths),
            "collision_prim_paths": collision_prim_paths,
            "preserved_required_links": sorted(validation_profile.expected_links),
            "missing_required_links": missing_links,
            "resolved_articulation_path": resolved_articulation_path,
            "physx_dof_names": list(robot.dof_names),
            "initial_joint_positions": initial_joint_positions.tolist(),
            "joint_positions_after_warmup": warmup_joint_positions.tolist(),
            "joint_positions_after_benchmark": final_joint_positions.tolist(),
            "joint_drift_radians_after_warmup": joint_drift_radians.tolist(),
            "max_joint_drift_radians_after_warmup": max_joint_drift_radians,
            "physics_benchmark": {
                "physics_dt_seconds": physics_dt_seconds,
                "warmup_steps": args.warmup_steps,
                "measured_steps_per_repeat": args.benchmark_steps,
                "repeats": args.benchmark_repeats,
                "observed_step_deltas": benchmark_step_deltas,
                "elapsed_seconds_per_repeat": benchmark_elapsed_seconds,
                "steps_per_second_per_repeat": steps_per_second,
                "steps_per_second_p05": float(np.percentile(steps_per_second, 5)),
                "steps_per_second_p50": float(np.percentile(steps_per_second, 50)),
                "steps_per_second_p95": float(np.percentile(steps_per_second, 95)),
                "simulated_seconds_per_repeat": simulated_seconds_per_repeat,
                "real_time_factor_per_repeat": real_time_factor,
                "real_time_factor_p05": float(np.percentile(real_time_factor, 5)),
                "real_time_factor_p50": float(np.percentile(real_time_factor, 50)),
                "real_time_factor_p95": float(np.percentile(real_time_factor, 95)),
                "rendering_enabled": False,
            },
            "fixed_base": True,
            "fixed_joints_merged": False,
            "urdf_importer_api": "URDFImporter 3",
            "headless": True,
            "fast_shutdown": True,
            "active_gpu": 0,
            "multi_gpu": False,
            "physx_dynamics_device": "cpu",
            "camera_or_depth_used": False,
            "isaac_ros_or_moveit_validated": False,
            "real_robot_commanded": False,
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        print(f"Isaac USD: {output_path}")
        print(f"Articulation roots: {articulation_roots}")
        print(f"PhysX DOFs: {list(robot.dof_names)}")
        print(
            "Headless PhysX: "
            f"{np.percentile(steps_per_second, 50):.1f} steps/s median, "
            f"{np.percentile(real_time_factor, 50):.1f}x real time median "
            f"over {args.benchmark_repeats} x {args.benchmark_steps} steps"
        )
        print(f"Import report: {report_path}")
        exit_code = 0
        return 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(wait_for_replicator=False, exit_code=exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
