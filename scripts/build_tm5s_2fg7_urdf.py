#!/usr/bin/env python3
"""Generate a self-contained TM5S URDF with an OnRobot 2FG7 gripper attached.

The generated model is for RViz/MoveIt dry-run work in the testing arena. It
does not change the installed TM5S MoveIt package and it should not be treated
as a calibrated hardware model until the real flange-to-gripper transform is
measured.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, NamedTuple


ARENA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_OUTPUT_ROOT = ARENA_DIR / "generated/tool_profiles"
DEFAULT_TOOL_MODEL = ARENA_DIR / "config/onrobot_2fg7_tool_model.json"
VENDOR_DIR = ARENA_DIR / "vendor/onrobot_2fg7"
MESH_DIR = VENDOR_DIR / "meshes"
TM5S_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
MARKER_RADIUS_M = 0.004

FINGER_TRANSFORMS = {
    "outwards": {
        "left": {
            "joint_xyz": (0.032239, -0.029494, 0.12005),
            "visual_xyz": (-0.032239, 0.029494, -0.12005),
            "axis": (-1.0, 0.0, 0.0),
            "mesh": "left_finger_link.stl",
            "inertial_xyz": (0.004247703322, 0.02949695948, 0.015234215486),
        },
        "right": {
            "joint_xyz": (-0.054361, -0.029494, 0.12005),
            "visual_xyz": (0.054361, 0.029494, -0.12005),
            "axis": (1.0, 0.0, 0.0),
            "mesh": "right_finger_link.stl",
            "inertial_xyz": (0.017874296678, 0.02949104052, 0.015234215486),
        },
    },
    "inwards": {
        "left": {
            "joint_xyz": (0.032239, -0.029494, 0.12005),
            "visual_xyz": (-0.032239, 0.029494, -0.12005),
            "axis": (-1.0, 0.0, 0.0),
            "mesh": "inwards/left_finger_link.stl",
            "inertial_xyz": (-0.007725703322, 0.02949104052, 0.015234215486),
        },
        "right": {
            "joint_xyz": (-0.054361, -0.029494, 0.12005),
            "visual_xyz": (0.054361, 0.029494, -0.12005),
            "axis": (1.0, 0.0, 0.0),
            "mesh": "inwards/right_finger_link.stl",
            "inertial_xyz": (0.029847703322, 0.02949695948, 0.015234215486),
        },
    },
}


class SourceRobot(NamedTuple):
    tree: ET.ElementTree
    source: str
    source_dir: Path | None


class ResolvedToolModel(NamedTuple):
    config_path: Path
    profile_name: str
    shared: dict[str, Any]
    profile: dict[str, Any]
    finger_configuration: str
    mount_xyz: tuple[float, float, float]
    mount_rpy: tuple[float, float, float]
    cad_origin_xyz: tuple[float, float, float]
    cad_origin_rpy: tuple[float, float, float]
    pin_grasp_xyz: tuple[float, float, float]
    pin_grasp_rpy: tuple[float, float, float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tm5s-urdf",
        type=Path,
        default=None,
        help=(
            "Optional pre-expanded TM5S URDF. By default, the script uses the "
            "installed tm_description xacro from the currently sourced ROS workspace."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output URDF (default: generated/tool_profiles/<profile>/tm5s_with_2fg7.urdf).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Output metadata JSON (default: beside the selected profile URDF).",
    )
    parser.add_argument("--tool-model", type=Path, default=DEFAULT_TOOL_MODEL)
    parser.add_argument(
        "--tool-profile",
        default=None,
        help="Named profile from the tool-model JSON (defaults to its default_profile).",
    )
    parser.add_argument(
        "--finger-configuration",
        choices=["outwards", "inwards"],
        default=None,
        help="Override the profile's vendored 2FG7 finger mesh set.",
    )
    parser.add_argument(
        "--finger-position",
        type=float,
        default=0.0,
        help="Finger prismatic offset in metres. For fixed fingers this bakes the visual pose.",
    )
    parser.add_argument(
        "--finger-joints",
        choices=["fixed", "prismatic"],
        default="fixed",
        help="Use fixed fingers for MoveIt display stability, or prismatic joints for RViz joint demos.",
    )
    parser.add_argument(
        "--mount-rpy",
        "--flange-to-gripper-rpy",
        dest="mount_rpy",
        nargs=3,
        type=float,
        default=None,
        metavar=("R", "P", "Y"),
        help="Override the profile mount rotation at the TM flange, radians.",
    )
    parser.add_argument(
        "--mount-xyz",
        "--flange-to-gripper-xyz",
        dest="mount_xyz",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override the profile mount translation at the TM flange, metres.",
    )
    parser.add_argument(
        "--cad-origin-xyz",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override the 2FG7 device-origin to vendored-CAD-origin translation.",
    )
    parser.add_argument(
        "--cad-origin-rpy",
        nargs=3,
        type=float,
        default=None,
        metavar=("R", "P", "Y"),
        help="Override the 2FG7 device-origin to vendored-CAD-origin rotation.",
    )
    parser.add_argument(
        "--pin-grasp-tcp-xyz",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override the application pin-grasp frame from the 2FG7 device origin.",
    )
    parser.add_argument(
        "--pin-grasp-tcp-rpy",
        nargs=3,
        type=float,
        default=None,
        metavar=("R", "P", "Y"),
        help="Override the application pin-grasp frame rotation, radians.",
    )
    parser.add_argument(
        "--tcp-z",
        type=float,
        default=None,
        help="Deprecated compatibility override for pin-grasp TCP Z only.",
    )
    return parser


def fmt(values) -> str:
    return " ".join(f"{float(v):.8g}" for v in values)


def vector3(value: Any, field: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field} must contain exactly three values")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{field} contains a non-finite value")
    return result


def validate_application_pin_baseline(
    profile: dict[str, Any],
    pin_grasp_xyz: tuple[float, float, float],
) -> None:
    baseline = profile.get("application_pin_baseline")
    if baseline is None:
        return

    axis = vector3(
        baseline.get("pin_axis_from_gripper_toward_specimen"),
        "application_pin_baseline.pin_axis_from_gripper_toward_specimen",
    )
    axis_norm = math.sqrt(sum(component * component for component in axis))
    if not math.isclose(axis_norm, 1.0, abs_tol=1.0e-12):
        raise ValueError("application pin axis must be a unit vector")

    clear_length = float(baseline.get("clear_pin_length_before_specimen_m"))
    if not math.isfinite(clear_length) or clear_length <= 0.0:
        raise ValueError("application clear pin length must be finite and positive")
    if baseline.get("grasp_point_on_clear_section") != "midpoint":
        raise ValueError("only a midpoint application pin grasp is currently supported")

    clear_start = vector3(
        baseline.get("clear_section_start_xyz_from_2fg7_device_origin_m"),
        "application_pin_baseline.clear_section_start",
    )
    configured_pinch = vector3(
        baseline.get("pinch_xyz_from_2fg7_device_origin_m"),
        "application_pin_baseline.pinch",
    )
    specimen_near = vector3(
        baseline.get("specimen_near_point_xyz_from_2fg7_device_origin_m"),
        "application_pin_baseline.specimen_near_point",
    )
    expected_specimen = tuple(
        clear_start[index] + clear_length * axis[index] for index in range(3)
    )
    expected_pinch = tuple(
        clear_start[index] + 0.5 * clear_length * axis[index] for index in range(3)
    )
    for label, actual, expected in (
        ("specimen near point", specimen_near, expected_specimen),
        ("configured pinch", configured_pinch, expected_pinch),
        ("profile pin-grasp TCP", pin_grasp_xyz, expected_pinch),
    ):
        if not all(
            math.isclose(actual[index], expected[index], abs_tol=1.0e-12)
            for index in range(3)
        ):
            raise ValueError(f"application {label} is inconsistent with the 10 mm baseline")

    pinch_to_specimen = float(baseline.get("pinch_to_specimen_m"))
    if not math.isclose(pinch_to_specimen, clear_length / 2.0, abs_tol=1.0e-12):
        raise ValueError("application pinch-to-specimen distance must be half the clear length")
    if configured_pinch[0] != 0.0 or configured_pinch[1] != 0.0:
        raise ValueError("application pinch must remain centred between the inward fingers")


def load_tool_model(args: argparse.Namespace) -> ResolvedToolModel:
    config_path = args.tool_model.resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if data.get("format_version") != 1:
        raise ValueError(f"Unsupported tool-model format: {data.get('format_version')!r}")

    profile_name = args.tool_profile or data.get("default_profile")
    profiles = data.get("profiles", {})
    if profile_name not in profiles:
        raise ValueError(
            f"Unknown tool profile {profile_name!r}; available: {sorted(profiles)}"
        )
    shared = data.get("shared", {})
    profile = profiles[profile_name]
    quick_changer_mode = profile.get("quick_changer_mode")
    if quick_changer_mode not in {"omitted", "standard_robot_side_nominal"}:
        raise ValueError(f"Unsupported quick_changer_mode: {quick_changer_mode!r}")

    if quick_changer_mode == "standard_robot_side_nominal":
        default_mount_xyz = profile.get("flange_to_qc_xyz_m", (0.0, 0.0, 0.0))
    else:
        default_mount_xyz = profile.get(
            "flange_to_2fg7_device_origin_xyz_m", (0.0, 0.0, 0.0)
        )
    mount_xyz = vector3(args.mount_xyz or default_mount_xyz, "mount_xyz")
    mount_rpy = vector3(
        args.mount_rpy or profile.get("mount_rpy_rad", (0.0, 0.0, 0.0)),
        "mount_rpy",
    )
    cad_origin_xyz = vector3(
        args.cad_origin_xyz
        or profile.get("device_origin_to_cad_origin_xyz_m", (0.0, 0.0, 0.0)),
        "cad_origin_xyz",
    )
    cad_origin_rpy = vector3(
        args.cad_origin_rpy
        or profile.get("device_origin_to_cad_origin_rpy_rad", (0.0, 0.0, 0.0)),
        "cad_origin_rpy",
    )
    pin_grasp_xyz = vector3(
        args.pin_grasp_tcp_xyz or profile.get("pin_grasp_tcp_xyz_m"),
        "pin_grasp_tcp_xyz",
    )
    if args.tcp_z is not None:
        if args.pin_grasp_tcp_xyz is not None:
            raise ValueError("Use either --tcp-z or --pin-grasp-tcp-xyz, not both")
        if not math.isfinite(args.tcp_z):
            raise ValueError("--tcp-z must be finite")
        pin_grasp_xyz = (pin_grasp_xyz[0], pin_grasp_xyz[1], args.tcp_z)
    pin_grasp_rpy = vector3(
        args.pin_grasp_tcp_rpy or profile.get("pin_grasp_tcp_rpy_rad", (0.0, 0.0, 0.0)),
        "pin_grasp_tcp_rpy",
    )
    validate_application_pin_baseline(profile, pin_grasp_xyz)
    finger_configuration = args.finger_configuration or shared.get("finger_configuration")
    if finger_configuration not in FINGER_TRANSFORMS:
        raise ValueError(f"Unsupported finger configuration: {finger_configuration!r}")

    return ResolvedToolModel(
        config_path=config_path,
        profile_name=profile_name,
        shared=shared,
        profile=profile,
        finger_configuration=finger_configuration,
        mount_xyz=mount_xyz,
        mount_rpy=mount_rpy,
        cad_origin_xyz=cad_origin_xyz,
        cad_origin_rpy=cad_origin_rpy,
        pin_grasp_xyz=pin_grasp_xyz,
        pin_grasp_rpy=pin_grasp_rpy,
    )


def resolve_output_paths(
    args: argparse.Namespace, model: ResolvedToolModel
) -> tuple[Path, Path]:
    profile_dir = DEFAULT_PROFILE_OUTPUT_ROOT / model.profile_name
    output = args.output or profile_dir / "tm5s_with_2fg7.urdf"
    metadata = args.metadata
    if metadata is None:
        metadata = output.with_name(f"{output.stem}_metadata.json")
    return output, metadata


def rotation_matrix_from_rpy(rpy: tuple[float, float, float]) -> tuple[tuple[float, ...], ...]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def rotate_vector(
    rotation: tuple[tuple[float, ...], ...], vector: tuple[float, float, float]
) -> tuple[float, float, float]:
    return tuple(
        sum(rotation[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )


def compose_transform(
    parent_xyz: tuple[float, float, float],
    parent_rpy: tuple[float, float, float],
    child_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    rotated = rotate_vector(rotation_matrix_from_rpy(parent_rpy), child_xyz)
    return tuple(parent_xyz[index] + rotated[index] for index in range(3))


def add_origin(parent: ET.Element, xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> ET.Element:
    return ET.SubElement(parent, "origin", {"xyz": fmt(xyz), "rpy": fmt(rpy)})


def add_material(root: ET.Element, name: str, rgba: tuple[float, float, float, float]) -> None:
    material = ET.SubElement(root, "material", {"name": name})
    ET.SubElement(material, "color", {"rgba": fmt(rgba)})


def add_inertial(
    link: ET.Element,
    mass_kg: float,
    inertia: tuple[float, float, float, float, float, float],
    *,
    xyz=(0.0, 0.0, 0.0),
) -> None:
    if mass_kg <= 0.0 or not math.isfinite(mass_kg):
        raise ValueError(f"Inertial mass must be positive and finite, got {mass_kg}")
    ixx, ixy, ixz, iyy, iyz, izz = inertia
    inertial = ET.SubElement(link, "inertial")
    add_origin(inertial, xyz)
    ET.SubElement(inertial, "mass", {"value": f"{mass_kg:.12g}"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": f"{ixx:.12g}",
            "ixy": f"{ixy:.12g}",
            "ixz": f"{ixz:.12g}",
            "iyy": f"{iyy:.12g}",
            "iyz": f"{iyz:.12g}",
            "izz": f"{izz:.12g}",
        },
    )


def uniform_box_inertia(
    mass_kg: float, dimensions_xyz_m: tuple[float, float, float]
) -> tuple[float, float, float, float, float, float]:
    x, y, z = dimensions_xyz_m
    return (
        mass_kg * (y * y + z * z) / 12.0,
        0.0,
        0.0,
        mass_kg * (x * x + z * z) / 12.0,
        0.0,
        mass_kg * (x * x + y * y) / 12.0,
    )


def uniform_cylinder_inertia(
    mass_kg: float, radius_m: float, height_m: float
) -> tuple[float, float, float, float, float, float]:
    transverse = mass_kg * (3.0 * radius_m * radius_m + height_m * height_m) / 12.0
    axial = 0.5 * mass_kg * radius_m * radius_m
    return (transverse, 0.0, 0.0, transverse, 0.0, axial)


def add_surrogate_inertial(
    link: ET.Element,
    mass_kg: float,
    *,
    xyz=(0.0, 0.0, 0.0),
) -> None:
    # The fingers stay kinematic/no-contact for now. This deliberately tiny,
    # explicit inertia prevents importers from inventing kilogram-scale defaults.
    diagonal = max(1.0e-12, mass_kg * 1.0e-6)
    add_inertial(
        link,
        mass_kg,
        (diagonal, 0.0, 0.0, diagonal, 0.0, diagonal),
        xyz=xyz,
    )


def mesh_uri(mesh_path: Path) -> str:
    return mesh_path.resolve().as_uri()


def add_mesh_geometry(parent: ET.Element, mesh_path: Path) -> None:
    geometry = ET.SubElement(parent, "geometry")
    ET.SubElement(
        geometry,
        "mesh",
        {
            "filename": mesh_uri(mesh_path),
            "scale": "0.001 0.001 0.001",
        },
    )


def add_mesh_visual(
    link: ET.Element,
    mesh_path: Path,
    *,
    xyz=(0.0, 0.0, 0.0),
    rpy=(0.0, 0.0, 0.0),
    material="onrobot_dark",
) -> None:
    visual = ET.SubElement(link, "visual")
    add_origin(visual, xyz, rpy)
    add_mesh_geometry(visual, mesh_path)
    ET.SubElement(visual, "material", {"name": material})


def add_mesh_collision(
    link: ET.Element,
    mesh_path: Path,
    *,
    xyz=(0.0, 0.0, 0.0),
    rpy=(0.0, 0.0, 0.0),
) -> None:
    collision = ET.SubElement(link, "collision")
    add_origin(collision, xyz, rpy)
    add_mesh_geometry(collision, mesh_path)


def add_cylinder_geometry(
    link: ET.Element,
    radius_m: float,
    length_m: float,
    *,
    xyz=(0.0, 0.0, 0.0),
    material: str | None = None,
    collision_radius_m: float | None = None,
) -> None:
    visual = ET.SubElement(link, "visual")
    add_origin(visual, xyz)
    visual_geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(
        visual_geometry,
        "cylinder",
        {"radius": f"{radius_m:.12g}", "length": f"{length_m:.12g}"},
    )
    if material:
        ET.SubElement(visual, "material", {"name": material})

    collision = ET.SubElement(link, "collision")
    add_origin(collision, xyz)
    collision_geometry = ET.SubElement(collision, "geometry")
    collision_radius = radius_m if collision_radius_m is None else collision_radius_m
    ET.SubElement(
        collision_geometry,
        "cylinder",
        {"radius": f"{collision_radius:.12g}", "length": f"{length_m:.12g}"},
    )


def add_fixed_joint(
    root: ET.Element,
    name: str,
    parent_link: str,
    child_link: str,
    *,
    xyz=(0.0, 0.0, 0.0),
    rpy=(0.0, 0.0, 0.0),
) -> None:
    joint = ET.SubElement(root, "joint", {"name": name, "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": parent_link})
    ET.SubElement(joint, "child", {"link": child_link})
    add_origin(joint, xyz, rpy)


def add_prismatic_joint(
    root: ET.Element,
    name: str,
    parent_link: str,
    child_link: str,
    *,
    xyz,
    axis,
    lower,
    upper,
    velocity,
    mimic: str | None = None,
) -> None:
    joint = ET.SubElement(root, "joint", {"name": name, "type": "prismatic"})
    ET.SubElement(joint, "parent", {"link": parent_link})
    ET.SubElement(joint, "child", {"link": child_link})
    add_origin(joint, xyz)
    ET.SubElement(joint, "axis", {"xyz": fmt(axis)})
    ET.SubElement(
        joint,
        "limit",
        {
            "lower": str(lower),
            "upper": str(upper),
            # 140 N is retained only as a non-authoritative visualization
            # placeholder; it is not interpreted as force per finger.
            "effort": "140",
            "velocity": str(velocity),
        },
    )
    if mimic:
        ET.SubElement(joint, "mimic", {"joint": mimic, "multiplier": "1.0"})


def add_empty_frame(
    root: ET.Element,
    name: str,
    parent_link: str,
    *,
    xyz=(0.0, 0.0, 0.0),
    rpy=(0.0, 0.0, 0.0),
) -> None:
    ET.SubElement(root, "link", {"name": name})
    add_fixed_joint(root, f"{parent_link}_to_{name}", parent_link, name, xyz=xyz, rpy=rpy)


def add_coordinate_frames(root: ET.Element, model: ResolvedToolModel) -> None:
    device_frames = model.shared["frames_from_2fg7_device_origin"]
    cad_frames = model.shared["frames_from_cad_origin"]
    nominal = device_frames["onrobot_nominal_tcp"]
    fingertip = cad_frames["finger_tip_plane"]
    add_empty_frame(
        root,
        "onrobot_nominal_tcp",
        "onrobot_2fg7_origin",
        xyz=vector3(nominal["xyz_m"], "onrobot_nominal_tcp.xyz_m"),
        rpy=vector3(nominal["rpy_rad"], "onrobot_nominal_tcp.rpy_rad"),
    )
    add_empty_frame(
        root,
        "finger_tip_plane",
        "onrobot_2fg7_base_link",
        xyz=vector3(fingertip["xyz_m"], "finger_tip_plane.xyz_m"),
        rpy=vector3(fingertip["rpy_rad"], "finger_tip_plane.rpy_rad"),
    )

    marker_mass_kg = float(model.shared["surrogate_link_mass_kg"])
    marker_inertia = 0.4 * marker_mass_kg * MARKER_RADIUS_M**2
    link = ET.SubElement(root, "link", {"name": "gripper_tcp"})
    add_inertial(
        link,
        marker_mass_kg,
        (marker_inertia, 0.0, 0.0, marker_inertia, 0.0, marker_inertia),
    )
    visual = ET.SubElement(link, "visual")
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "sphere", {"radius": str(MARKER_RADIUS_M)})
    ET.SubElement(visual, "material", {"name": "tcp_blue"})
    add_fixed_joint(
        root,
        "onrobot_2fg7_origin_to_gripper_tcp",
        "onrobot_2fg7_origin",
        "gripper_tcp",
        xyz=model.pin_grasp_xyz,
        rpy=model.pin_grasp_rpy,
    )
    # Keep the long-standing gripper_tcp name for compatibility while exposing
    # the application frame under an unambiguous semantic name.
    add_empty_frame(root, "pin_grasp_tcp", "gripper_tcp")


def add_fake_ros2_control(root: ET.Element) -> None:
    """Add a simple fake ros2_control block for MoveIt demo launches."""
    if root.find("ros2_control") is not None:
        return

    control = ET.SubElement(root, "ros2_control", {"name": "FakeSystem", "type": "system"})
    hardware = ET.SubElement(control, "hardware")
    ET.SubElement(hardware, "plugin").text = "mock_components/GenericSystem"

    for joint_name in TM5S_JOINTS:
        joint = ET.SubElement(control, "joint", {"name": joint_name})
        ET.SubElement(joint, "command_interface", {"name": "position"})
        state_position = ET.SubElement(joint, "state_interface", {"name": "position"})
        ET.SubElement(state_position, "param", {"name": "initial_value"}).text = "0.0"
        ET.SubElement(joint, "state_interface", {"name": "velocity"})


def normalize_mesh_filenames(root: ET.Element, urdf_dir: Path) -> None:
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename or "://" in filename:
            continue

        path = Path(filename)
        if not path.is_absolute():
            path = urdf_dir / path
        # Historical manifests may contain another workstation user's home.
        # Rebase only the suffix, without embedding that identity in source.
        if len(path.parts) >= 4 and path.parts[:2] == ("/", "home"):
            current_home_path = Path.home().joinpath(*path.parts[3:])
            if current_home_path.exists():
                path = current_home_path
        mesh.set("filename", mesh_uri(path))


def load_installed_tm5s_xacro() -> SourceRobot:
    try:
        from ament_index_python.packages import get_package_share_directory
        import xacro
    except Exception as exc:
        raise RuntimeError("ROS xacro package lookup is not available") from exc

    share_dir = Path(get_package_share_directory("tm_description"))
    xacro_path = share_dir / "xacro/tm5s.urdf.xacro"
    if not xacro_path.exists():
        raise FileNotFoundError(f"Installed TM5S xacro was not found: {xacro_path}")

    xml_doc = xacro.process_file(str(xacro_path))
    root = ET.fromstring(xml_doc.toxml())
    return SourceRobot(ET.ElementTree(root), str(xacro_path), xacro_path.parent)


def load_tm5s_source(tm5s_urdf: Path | None) -> SourceRobot:
    if tm5s_urdf is not None:
        path = tm5s_urdf.resolve()
        return SourceRobot(ET.parse(path), str(path), path.parent)

    try:
        return load_installed_tm5s_xacro()
    except Exception as exc:
        raise RuntimeError(
            "Could not load installed tm_description xacro. Source the native "
            "Jazzy workspace first, or pass --tm5s-urdf with a pre-expanded TM5S URDF."
        ) from exc


def validate_vendor_assets(finger_configuration: str) -> None:
    required = [MESH_DIR / "base_link.stl"]
    required.extend(MESH_DIR / spec["mesh"] for spec in FINGER_TRANSFORMS[finger_configuration].values())
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing vendored OnRobot mesh files:\n"
            + "\n".join(f"  {path}" for path in missing)
            + "\nCopy the assets into vendor/onrobot_2fg7/meshes first."
        )


def inverse_rotate_vector(
    rotation: tuple[tuple[float, ...], ...], vector: tuple[float, float, float]
) -> tuple[float, float, float]:
    return tuple(
        sum(rotation[row][column] * vector[row] for row in range(3))
        for column in range(3)
    )


def add_mount_chain(root: ET.Element, model: ResolvedToolModel) -> None:
    ET.SubElement(root, "link", {"name": "onrobot_2fg7_origin"})
    mode = model.profile["quick_changer_mode"]
    if mode == "omitted":
        add_fixed_joint(
            root,
            "flange_to_onrobot_2fg7_origin",
            "flange",
            "onrobot_2fg7_origin",
            xyz=model.mount_xyz,
            rpy=model.mount_rpy,
        )
        return

    quick_changer = model.shared["standard_quick_changer_robot_side"]
    mass_kg = float(quick_changer["mass_kg"])
    height_m = float(quick_changer["physical_body_height_m"])
    body_radius_m = 0.5 * float(quick_changer["body_diameter_m"])
    collision_radius_m = float(quick_changer["maximum_radial_reach_from_axis_m"])
    inertia_radius_m = float(quick_changer["inertia_proxy_radius_m"])
    cog_xyz_m = vector3(
        quick_changer["cog_xyz_m"],
        "standard_quick_changer_robot_side.cog_xyz_m",
    )
    qc_link = ET.SubElement(root, "link", {"name": "onrobot_qc_robot_side_link"})
    add_inertial(
        qc_link,
        mass_kg,
        uniform_cylinder_inertia(mass_kg, inertia_radius_m, height_m),
        xyz=cog_xyz_m,
    )
    add_cylinder_geometry(
        qc_link,
        body_radius_m,
        height_m,
        xyz=(0.0, 0.0, 0.5 * height_m),
        material="onrobot_qc_blue",
        collision_radius_m=collision_radius_m,
    )
    add_fixed_joint(
        root,
        "flange_to_onrobot_qc_robot_side",
        "flange",
        "onrobot_qc_robot_side_link",
        xyz=model.mount_xyz,
        rpy=model.mount_rpy,
    )
    add_fixed_joint(
        root,
        "onrobot_qc_robot_side_to_2fg7_origin",
        "onrobot_qc_robot_side_link",
        "onrobot_2fg7_origin",
        xyz=(0.0, 0.0, float(quick_changer["flange_to_tool_interface_z_m"])),
    )


def gripper_base_dynamics(
    model: ResolvedToolModel, args: argparse.Namespace
) -> tuple[float, tuple[float, float, float], tuple[float, float, float, float, float, float]]:
    dynamics = model.shared["device_dynamics"]
    total_mass = float(dynamics["mass_kg"])
    surrogate_mass = float(model.shared["surrogate_link_mass_kg"])
    base_mass = total_mass - 3.0 * surrogate_mass
    if base_mass <= 0.0:
        raise ValueError("Surrogate masses exceed the official 2FG7 mass")

    target_cog = vector3(dynamics["cog_xyz_m"], "device_dynamics.cog_xyz_m")
    cad_rotation = rotation_matrix_from_rpy(model.cad_origin_rpy)
    finger_positions: list[tuple[float, float, float]] = []
    for spec in FINGER_TRANSFORMS[model.finger_configuration].values():
        joint_xyz = tuple(
            base + axis * args.finger_position
            for base, axis in zip(spec["joint_xyz"], spec["axis"])
        )
        finger_in_cad = tuple(
            joint_xyz[index] + spec["inertial_xyz"][index] for index in range(3)
        )
        finger_in_device = compose_transform(
            model.cad_origin_xyz,
            model.cad_origin_rpy,
            finger_in_cad,
        )
        finger_positions.append(finger_in_device)

    remaining_moment = [total_mass * target_cog[index] for index in range(3)]
    for position in [*finger_positions, model.pin_grasp_xyz]:
        for index in range(3):
            remaining_moment[index] -= surrogate_mass * position[index]
    base_cog_in_device = tuple(component / base_mass for component in remaining_moment)
    base_cog_delta = tuple(
        base_cog_in_device[index] - model.cad_origin_xyz[index] for index in range(3)
    )
    base_cog_local = inverse_rotate_vector(cad_rotation, base_cog_delta)

    dimensions = vector3(
        dynamics["uniform_box_proxy_dimensions_xyz_m"],
        "device_dynamics.uniform_box_proxy_dimensions_xyz_m",
    )
    base_inertia = uniform_box_inertia(base_mass, dimensions)
    return base_mass, base_cog_local, base_inertia


def add_gripper(
    root: ET.Element, args: argparse.Namespace, model: ResolvedToolModel
) -> None:
    if root.find("./link[@name='flange']") is None:
        raise ValueError("TM5S URDF does not contain a 'flange' link.")
    if root.find("./link[@name='onrobot_2fg7_base_link']") is not None:
        raise ValueError("URDF already contains onrobot_2fg7_base_link.")
    root.set("name", "tm5s_with_2fg7")
    add_material(root, "onrobot_dark", (0.02, 0.02, 0.025, 1.0))
    add_material(root, "onrobot_silver", (0.65, 0.68, 0.70, 1.0))
    add_material(root, "onrobot_qc_blue", (0.02, 0.20, 0.34, 1.0))
    add_material(root, "tcp_blue", (0.0, 0.22, 1.0, 1.0))

    add_mount_chain(root, model)
    base_link = ET.SubElement(root, "link", {"name": "onrobot_2fg7_base_link"})
    base_mass, base_cog, base_inertia = gripper_base_dynamics(model, args)
    add_inertial(base_link, base_mass, base_inertia, xyz=base_cog)
    add_mesh_visual(base_link, MESH_DIR / "base_link.stl", material="onrobot_dark")
    add_mesh_collision(base_link, MESH_DIR / "base_link.stl")
    add_fixed_joint(
        root,
        "onrobot_2fg7_origin_to_base",
        "onrobot_2fg7_origin",
        "onrobot_2fg7_base_link",
        xyz=model.cad_origin_xyz,
        rpy=model.cad_origin_rpy,
    )

    left_joint_name = "onrobot_2fg7_left_finger_joint"
    surrogate_mass = float(model.shared["surrogate_link_mass_kg"])
    finger_speed = float(model.shared["finger_motion"]["maximum_per_finger_speed_m_s"])
    finger_travel = float(model.shared["finger_motion"]["per_finger_travel_m"])
    for side, spec in FINGER_TRANSFORMS[model.finger_configuration].items():
        link_name = f"onrobot_2fg7_{side}_finger_link"
        link = ET.SubElement(root, "link", {"name": link_name})
        add_surrogate_inertial(link, surrogate_mass, xyz=spec["inertial_xyz"])
        mesh_path = MESH_DIR / spec["mesh"]
        add_mesh_visual(link, mesh_path, xyz=spec["visual_xyz"], material="onrobot_silver")
        add_mesh_collision(link, mesh_path, xyz=spec["visual_xyz"])

        joint_xyz = tuple(
            base + axis * args.finger_position
            for base, axis in zip(spec["joint_xyz"], spec["axis"])
        )
        joint_name = f"onrobot_2fg7_{side}_finger_joint"
        if args.finger_joints == "fixed":
            add_fixed_joint(
                root,
                joint_name,
                "onrobot_2fg7_base_link",
                link_name,
                xyz=joint_xyz,
            )
        else:
            add_prismatic_joint(
                root,
                joint_name,
                "onrobot_2fg7_base_link",
                link_name,
                xyz=spec["joint_xyz"],
                axis=spec["axis"],
                lower=0.0,
                upper=finger_travel,
                velocity=finger_speed,
                mimic=left_joint_name if side == "right" else None,
            )

    add_coordinate_frames(root, model)
    add_fake_ros2_control(root)


def proxy_metadata(model: ResolvedToolModel) -> dict[str, Any]:
    dynamics = model.shared["device_dynamics"]
    gripper_mass = float(dynamics["mass_kg"])
    gripper_cog = vector3(dynamics["cog_xyz_m"], "device_dynamics.cog_xyz_m")
    gripper_inertia = uniform_box_inertia(
        gripper_mass,
        vector3(
            dynamics["uniform_box_proxy_dimensions_xyz_m"],
            "device_dynamics.uniform_box_proxy_dimensions_xyz_m",
        ),
    )
    mode = model.profile["quick_changer_mode"]
    qc_offset = 0.0
    qc_mass = 0.0
    qc_cog = (0.0, 0.0, 0.0)
    qc_inertia = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if mode == "standard_robot_side_nominal":
        quick_changer = model.shared["standard_quick_changer_robot_side"]
        qc_offset = float(quick_changer["flange_to_tool_interface_z_m"])
        qc_mass = float(quick_changer["mass_kg"])
        qc_height = float(quick_changer["physical_body_height_m"])
        qc_radius = float(quick_changer["inertia_proxy_radius_m"])
        qc_cog = vector3(
            quick_changer["cog_xyz_m"],
            "standard_quick_changer_robot_side.cog_xyz_m",
        )
        qc_inertia = uniform_cylinder_inertia(qc_mass, qc_radius, qc_height)

    total_mass = gripper_mass + qc_mass
    gripper_cog_from_mount = (
        gripper_cog[0],
        gripper_cog[1],
        qc_offset + gripper_cog[2],
    )
    aggregate_cog = tuple(
        (
            gripper_mass * gripper_cog_from_mount[index]
            + qc_mass * qc_cog[index]
        )
        / total_mass
        for index in range(3)
    )
    gripper_delta = tuple(
        gripper_cog_from_mount[index] - aggregate_cog[index]
        for index in range(3)
    )
    qc_delta = tuple(qc_cog[index] - aggregate_cog[index] for index in range(3))
    aggregate_inertia = (
        gripper_inertia[0]
        + gripper_mass * (gripper_delta[1] ** 2 + gripper_delta[2] ** 2)
        + qc_inertia[0]
        + qc_mass * (qc_delta[1] ** 2 + qc_delta[2] ** 2),
        -gripper_mass * gripper_delta[0] * gripper_delta[1]
        - qc_mass * qc_delta[0] * qc_delta[1],
        -gripper_mass * gripper_delta[0] * gripper_delta[2]
        - qc_mass * qc_delta[0] * qc_delta[2],
        gripper_inertia[3]
        + gripper_mass * (gripper_delta[0] ** 2 + gripper_delta[2] ** 2)
        + qc_inertia[3]
        + qc_mass * (qc_delta[0] ** 2 + qc_delta[2] ** 2),
        -gripper_mass * gripper_delta[1] * gripper_delta[2]
        - qc_mass * qc_delta[1] * qc_delta[2],
        gripper_inertia[5]
        + gripper_mass * (gripper_delta[0] ** 2 + gripper_delta[1] ** 2)
        + qc_inertia[5]
        + qc_mass * (qc_delta[0] ** 2 + qc_delta[1] ** 2),
    )
    return {
        "profile": "provisional_sim_only_bare_qc_2fg7" if qc_mass else "provisional_sim_only_bare_2fg7",
        "total_mass_kg": total_mass,
        "aggregate_cog_xyz_from_mount_m": list(aggregate_cog),
        "aggregate_inertia_at_cog_tool_axes_kg_m2": {
            "ixx": aggregate_inertia[0],
            "ixy": aggregate_inertia[1],
            "ixz": aggregate_inertia[2],
            "iyy": aggregate_inertia[3],
            "iyz": aggregate_inertia[4],
            "izz": aggregate_inertia[5],
        },
        "status": (
            "official_component_masses_and_cogs_with_uniform_box_cylinder_"
            "inertia_proxy"
        ),
    }


def build_metadata(
    args: argparse.Namespace,
    model: ResolvedToolModel,
    source_robot: SourceRobot,
) -> dict[str, Any]:
    output_path, _ = resolve_output_paths(args, model)
    mode = model.profile["quick_changer_mode"]
    if mode == "standard_robot_side_nominal":
        qc_offset = float(
            model.shared["standard_quick_changer_robot_side"][
                "flange_to_tool_interface_z_m"
            ]
        )
        flange_to_device_origin = compose_transform(
            model.mount_xyz, model.mount_rpy, (0.0, 0.0, qc_offset)
        )
    else:
        flange_to_device_origin = model.mount_xyz

    device_frames = model.shared["frames_from_2fg7_device_origin"]
    cad_frames = model.shared["frames_from_cad_origin"]
    nominal_tcp = device_frames["onrobot_nominal_tcp"]
    fingertip = cad_frames["finger_tip_plane"]
    fingertip_from_device = compose_transform(
        model.cad_origin_xyz,
        model.cad_origin_rpy,
        vector3(fingertip["xyz_m"], "finger_tip_plane"),
    )
    frame_transforms = {
        "onrobot_2fg7_origin": list(flange_to_device_origin),
        "onrobot_nominal_tcp": list(
            compose_transform(
                flange_to_device_origin,
                model.mount_rpy,
                vector3(nominal_tcp["xyz_m"], "nominal_tcp"),
            )
        ),
        "finger_tip_plane": list(
            compose_transform(
                flange_to_device_origin,
                model.mount_rpy,
                fingertip_from_device,
            )
        ),
        "pin_grasp_tcp": list(
            compose_transform(
                flange_to_device_origin,
                model.mount_rpy,
                model.pin_grasp_xyz,
            )
        ),
    }
    return {
        "generated_urdf": str(output_path.resolve()),
        "source_tm5s_urdf": source_robot.source,
        "tool_model_config": str(model.config_path),
        "tool_profile": model.profile_name,
        "profile_description": model.profile.get("description"),
        "onrobot_vendor_dir": str(VENDOR_DIR.resolve()),
        "gripper": model.shared["gripper"],
        "finger_configuration": model.finger_configuration,
        "finger_joints": args.finger_joints,
        "finger_position_m": args.finger_position,
        "finger_motion": model.shared["finger_motion"],
        "geometry_scope": model.shared.get("geometry_scope"),
        "quick_changer_mode": mode,
        "quick_changer_variant": model.profile.get("qc_variant"),
        "quick_changer_configuration": (
            model.shared["standard_quick_changer_robot_side"]
            if mode == "standard_robot_side_nominal"
            else None
        ),
        "adapter_k": model.shared["adapter_k"],
        "mount_xyz_m": list(model.mount_xyz),
        "mount_rpy_rad": list(model.mount_rpy),
        "device_origin_to_cad_origin_xyz_m": list(model.cad_origin_xyz),
        "device_origin_to_cad_origin_rpy_rad": list(model.cad_origin_rpy),
        "pin_grasp_tcp_xyz_from_device_origin_m": list(model.pin_grasp_xyz),
        "pin_grasp_tcp_rpy_from_device_origin_rad": list(model.pin_grasp_rpy),
        "application_pin_baseline": model.profile.get("application_pin_baseline"),
        "frame_xyz_from_flange_m": frame_transforms,
        "frame_status": {
            "mount": model.profile.get("mount_status"),
            "cad_registration": model.profile.get("cad_registration_status", "legacy_identity"),
            "onrobot_nominal_tcp": nominal_tcp["status"],
            "finger_tip_plane": fingertip["status"],
            "pin_grasp_tcp": model.profile.get("pin_grasp_status"),
        },
        "dynamics": proxy_metadata(model),
        "primary_application_tcp_link": "pin_grasp_tcp",
        "compatibility_tcp_link": "gripper_tcp",
        "moveit_planning_tip": "flange",
        "warnings": [
            "Simulation/dry-run model only; never copy proxy inertia into Watson's controller settings.",
            "Quick Changer and 2FG7 CoGs are vendor presets; proxy inertias remain simulation-only.",
            "Mount yaw remains physically unverified; the pin-grasp frame is a user-selected 10 mm CAD-relative baseline, not a calibrated TCP.",
            "The CAD fingertip plane is a mesh measurement, not the OnRobot nominal TCP.",
            "Finger links use tiny surrogate masses and are not a contact-dynamics model.",
        ],
    }


def main() -> int:
    args = build_parser().parse_args()
    model = load_tool_model(args)
    args.output, args.metadata = resolve_output_paths(args, model)
    validate_vendor_assets(model.finger_configuration)

    source_robot = load_tm5s_source(args.tm5s_urdf)
    tree = source_robot.tree
    root = tree.getroot()
    if source_robot.source_dir is not None:
        normalize_mesh_filenames(root, source_robot.source_dir)
    add_gripper(root, args, model)
    ET.indent(tree, space="  ")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output, encoding="utf-8", xml_declaration=True)

    metadata = build_metadata(args, model, source_robot)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
