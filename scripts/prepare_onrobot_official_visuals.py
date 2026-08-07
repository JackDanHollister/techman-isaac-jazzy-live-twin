#!/usr/bin/env python3
"""Build local, provenance-pinned visual meshes from official OnRobot STEP files.

The official STEP files and every derived mesh stay under ``generated/``, which
is git-ignored.  This script does not grant redistribution rights.  Run it with
the Isaac Sim 6.0 Python interpreter so the installed HOOPS STEP converter is
available.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import struct
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARENA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TOOL_MODEL = ARENA_DIR / "config/onrobot_2fg7_tool_model.json"
DEFAULT_OUTPUT_DIR = ARENA_DIR / "generated/onrobot_official_visuals"
BOUND_TOLERANCE_M = 5.0e-6
SOURCE_CACHE_NAMES = {
    "two_fg7": "2fg7_v1.step",
    "qc_robot_side": "qc_robot_side_v2_with_cable_holder.step",
}
OUTPUT_MESH_NAMES = {
    "two_fg7": "onrobot_2fg7_official_registered_m.stl",
    "qc_robot_side": "onrobot_qc_robot_side_v2_official_registered_m.stl",
}
REFERENCE_ONLY_STATUS = (
    "reference_only_not_consumed_by_robot_model_fixed_outwards_2fg7_pose_"
    "mismatches_watson_inwards_state"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-model", type=Path, default=DEFAULT_TOOL_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--2fg7-step", dest="two_fg7_step", type=Path, default=None)
    parser.add_argument("--qc-step", type=Path, default=None)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download both hash-pinned official STEP files into the ignored local cache.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Replace existing cached STEP files; requires --download.",
    )
    return parser


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_visual_spec(tool_model_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = json.loads(tool_model_path.read_text(encoding="utf-8"))
    spec = data.get("shared", {}).get("official_cad_visuals")
    if not isinstance(spec, dict):
        raise ValueError("Tool model has no shared.official_cad_visuals object")
    if spec.get("integration_status") != REFERENCE_ONLY_STATUS:
        raise ValueError("Official CAD visuals must retain their reference-only status")
    assets = spec.get("assets")
    if not isinstance(assets, dict) or set(assets) != set(SOURCE_CACHE_NAMES):
        raise ValueError("Official CAD visual spec must define 2FG7 and QC robot-side assets")
    for name, asset in assets.items():
        expected_hash = asset.get("sha256", "")
        if len(expected_hash) != 64:
            raise ValueError(f"{name} has an invalid SHA-256 value")
        validate_registration(asset.get("registration"), name)
    return data, spec


def validate_registration(registration: Any, name: str) -> None:
    if not isinstance(registration, dict):
        raise ValueError(f"{name} registration must be an object")
    origin = registration.get("source_origin_xyz_m")
    rows = registration.get("rotation_rows_source_to_link")
    if not isinstance(origin, list) or len(origin) != 3:
        raise ValueError(f"{name} source origin must have three values")
    if not isinstance(rows, list) or len(rows) != 3 or any(
        not isinstance(row, list) or len(row) != 3 for row in rows
    ):
        raise ValueError(f"{name} registration rotation must be 3 x 3")
    if not all(math.isfinite(float(value)) for value in origin):
        raise ValueError(f"{name} source origin contains a non-finite value")
    rotation = [[float(value) for value in row] for row in rows]
    for left in range(3):
        for right in range(3):
            dot = sum(rotation[left][index] * rotation[right][index] for index in range(3))
            target = 1.0 if left == right else 0.0
            if not math.isclose(dot, target, abs_tol=1.0e-12):
                raise ValueError(f"{name} registration rotation is not orthonormal")
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if not math.isclose(determinant, 1.0, abs_tol=1.0e-12):
        raise ValueError(f"{name} registration must be a proper rigid rotation")


def apply_registration(
    point_xyz_m: tuple[float, float, float], registration: dict[str, Any]
) -> tuple[float, float, float]:
    origin = [float(value) for value in registration["source_origin_xyz_m"]]
    delta = [float(point_xyz_m[index]) - origin[index] for index in range(3)]
    rows = registration["rotation_rows_source_to_link"]
    return tuple(
        sum(float(rows[row][column]) * delta[column] for column in range(3))
        for row in range(3)
    )


def validate_bounds(
    actual_minimum: tuple[float, float, float],
    actual_maximum: tuple[float, float, float],
    registration: dict[str, Any],
    name: str,
) -> None:
    expected = registration.get("expected_bounds_m", {})
    expected_minimum = expected.get("minimum")
    expected_maximum = expected.get("maximum")
    if not isinstance(expected_minimum, list) or not isinstance(expected_maximum, list):
        raise ValueError(f"{name} registration has no expected bounds")
    for label, actual, reference in (
        ("minimum", actual_minimum, expected_minimum),
        ("maximum", actual_maximum, expected_maximum),
    ):
        if len(reference) != 3:
            raise ValueError(f"{name} expected {label} bounds must have three values")
        for axis, (measured, wanted) in enumerate(zip(actual, reference)):
            if not math.isclose(measured, float(wanted), abs_tol=BOUND_TOLERANCE_M):
                raise ValueError(
                    f"{name} registered {label} axis {axis} is {measured:.9f} m; "
                    f"expected {float(wanted):.9f} +/- {BOUND_TOLERANCE_M:.1e} m"
                )


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "techman-digital-twin/1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_sources(
    args: argparse.Namespace, spec: dict[str, Any]
) -> dict[str, Path]:
    if args.force_download and not args.download:
        raise ValueError("--force-download requires --download")
    if args.download and (args.two_fg7_step is not None or args.qc_step is not None):
        raise ValueError("Use either --download or explicit STEP paths, not both")
    if (args.two_fg7_step is None) != (args.qc_step is None):
        raise ValueError("Provide both --2fg7-step and --qc-step")

    source_dir = args.output_dir.resolve() / "source"
    if args.two_fg7_step is not None:
        sources = {
            "two_fg7": args.two_fg7_step.resolve(),
            "qc_robot_side": args.qc_step.resolve(),
        }
    else:
        sources = {name: source_dir / filename for name, filename in SOURCE_CACHE_NAMES.items()}

    if args.download:
        for name, destination in sources.items():
            if args.force_download or not destination.exists():
                print(f"Downloading {name} -> {destination}")
                download_file(spec["assets"][name]["url"], destination)

    for name, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {name} STEP file: {path}\n"
                "Pass both explicit STEP paths or rerun with --download."
            )
        actual_hash = sha256_path(path)
        expected_hash = spec["assets"][name]["sha256"]
        if actual_hash != expected_hash:
            raise ValueError(
                f"{name} STEP SHA-256 mismatch: {actual_hash}; expected {expected_hash}"
            )
    return sources


def _normal(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    third: tuple[float, float, float],
) -> tuple[float, float, float]:
    ab = tuple(second[index] - first[index] for index in range(3))
    ac = tuple(third[index] - first[index] for index in range(3))
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    length = math.sqrt(sum(component * component for component in cross))
    if length <= 1.0e-20:
        return (0.0, 0.0, 0.0)
    return tuple(component / length for component in cross)


def export_registered_binary_stl(
    usd_path: Path,
    output_path: Path,
    registration: dict[str, Any],
    asset_name: str,
) -> dict[str, Any]:
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open converted USD: {usd_path}")
    metres_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if metres_per_unit <= 0.0:
        raise ValueError(f"Invalid USD metresPerUnit: {metres_per_unit}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    payload = bytearray()
    triangle_count = 0
    mesh_count = 0
    degenerate_triangle_count = 0
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_count += 1
        mesh = UsdGeom.Mesh(prim)
        local_points = mesh.GetPointsAttr().Get() or []
        counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
        indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
        local_to_world = xform_cache.GetLocalToWorldTransform(prim)
        points: list[tuple[float, float, float]] = []
        for point in local_points:
            world = local_to_world.Transform(Gf.Vec3d(*point))
            registered = apply_registration(
                tuple(float(component) * metres_per_unit for component in world),
                registration,
            )
            points.append(registered)
            for axis, value in enumerate(registered):
                minimum[axis] = min(minimum[axis], value)
                maximum[axis] = max(maximum[axis], value)

        cursor = 0
        for count in counts:
            if count < 3:
                cursor += count
                continue
            face = indices[cursor : cursor + count]
            cursor += count
            if len(face) != count:
                raise ValueError(f"Truncated face index data in {prim.GetPath()}")
            for offset in range(1, count - 1):
                vertices = (points[face[0]], points[face[offset]], points[face[offset + 1]])
                normal = _normal(*vertices)
                if normal == (0.0, 0.0, 0.0):
                    degenerate_triangle_count += 1
                payload.extend(
                    struct.pack(
                        "<12fH",
                        *normal,
                        *vertices[0],
                        *vertices[1],
                        *vertices[2],
                        0,
                    )
                )
                triangle_count += 1
        if cursor != len(indices):
            raise ValueError(f"Unused face indices in {prim.GetPath()}")

    if mesh_count == 0 or triangle_count == 0:
        raise ValueError(f"Converted {asset_name} USD contains no mesh triangles")
    if triangle_count > 0xFFFFFFFF:
        raise ValueError("Binary STL triangle count exceeds uint32 capacity")

    actual_minimum = tuple(minimum)
    actual_maximum = tuple(maximum)
    validate_bounds(actual_minimum, actual_maximum, registration, asset_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"OnRobot {asset_name} local visual; units=m; source hash pinned".encode("ascii")[:80]
    with output_path.open("wb") as stream:
        stream.write(header.ljust(80, b"\0"))
        stream.write(struct.pack("<I", triangle_count))
        stream.write(payload)

    return {
        "path": str(output_path.resolve()),
        "sha256": sha256_path(output_path),
        "bytes": output_path.stat().st_size,
        "mesh_count": mesh_count,
        "triangle_count": triangle_count,
        "degenerate_triangle_count": degenerate_triangle_count,
        "bounds_m": {
            "minimum": list(actual_minimum),
            "maximum": list(actual_maximum),
            "extents": [actual_maximum[index] - actual_minimum[index] for index in range(3)],
        },
        "units": "metres",
        "registration": registration,
    }


def convert_and_export(
    sources: dict[str, Path],
    output_dir: Path,
    spec: dict[str, Any],
    tool_model_path: Path,
) -> None:
    try:
        from isaacsim import SimulationApp
    except ImportError as exc:
        raise RuntimeError(
            "Run this script with the Isaac Sim 6.0 Python interpreter; "
            "the system Python has no STEP converter."
        ) from exc

    app = SimulationApp({"headless": True, "hide_ui": True})
    try:
        import omni.kit.app

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_manager.set_extension_enabled_immediate(
            "omni.kit.converter.hoops_core", True
        )
        app.update()
        import omni.converter.hoops
        from omni.kit.converter.hoops_core import HoopsOptions, get_instance

        converter = get_instance()
        if converter is None:
            raise RuntimeError("omni.kit.converter.hoops_core did not load")

        options = HoopsOptions()
        options.instancingStyle = omni.converter.hoops.InstancingStyle.eNone
        options.compositionStyle = omni.converter.hoops.CompositionStyle.eNone
        options.filterStyle = omni.converter.hoops.FilterStyle.eOmit
        options.upAxis = omni.converter.hoops.UpAxis.eFileDefault
        options.tessLOD = int(spec["converter"]["tessellation_lod"])
        options.useMaterials = False
        options.materialSelection = 0
        options.materialType = omni.converter.hoops.MaterialType.eNone
        options.bOptimize = False

        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="step_to_usd_", dir=output_dir) as temporary:
            temporary_dir = Path(temporary)
            usd_paths = {name: temporary_dir / f"{name}.usdc" for name in sources}

            async def run_conversions() -> None:
                for name in ("qc_robot_side", "two_fg7"):
                    output_url, status = await converter.create_converter_task(
                        str(sources[name]), str(usd_paths[name]), options.toArgs()
                    )
                    if status.error_code != 0 or not output_url:
                        raise RuntimeError(
                            f"{name} STEP conversion failed ({status.error_code}): "
                            f"{status.error_msg}"
                        )

            asyncio.run(run_conversions())
            meshes = {
                name: export_registered_binary_stl(
                    usd_paths[name],
                    output_dir / OUTPUT_MESH_NAMES[name],
                    spec["assets"][name]["registration"],
                    name,
                )
                for name in ("qc_robot_side", "two_fg7")
            }

        runtime = {
            "isaac_sim": "6.0.1",
            "hoops_core_extension_id": str(
                extension_manager.get_enabled_extension_id(
                    "omni.kit.converter.hoops_core"
                )
            ),
            "configured_backend": spec["converter"]["backend"],
        }
        source_manifest = {}
        for name, path in sources.items():
            asset = spec["assets"][name]
            source_manifest[name] = {
                "title": asset["title"],
                "url": asset["url"],
                "local_path": display_path(path),
                "sha256": sha256_path(path),
                "bytes": path.stat().st_size,
                "step_header_filename": asset["step_header_filename"],
            }
        for mesh in meshes.values():
            mesh["path"] = display_path(Path(mesh["path"]))

        manifest = {
            "format_version": 1,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "distribution_policy": spec["distribution_policy"],
            "integration_status": spec["integration_status"],
            "tool_model": display_path(tool_model_path),
            "tool_model_sha256": sha256_path(tool_model_path),
            "converter": {**spec["converter"], **runtime},
            "sources": source_manifest,
            "meshes": meshes,
            "warnings": [
                "Local visual assets only; no OnRobot CAD redistribution permission was identified.",
                "The registrations preserve vendor geometry and canonical axes but do not establish Watson's installed keyed yaw.",
                "The selected QC STEP is explicitly robot-side v2 with cable holder; Watson's installed QC revision is unverified.",
                "The official 2FG7 STEP has fixed outward fingers and is reference-only because Watson's archived state and current collision proxies are inward.",
                "These meshes are visual geometry, not reviewed collision or controller-dynamics models.",
            ],
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote {manifest_path}", flush=True)
        for name, mesh in meshes.items():
            print(
                f"{name}: {mesh['triangle_count']} triangles, {mesh['sha256']}, "
                f"bounds={mesh['bounds_m']}",
                flush=True,
            )
    finally:
        app.close()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ARENA_DIR))
    except ValueError:
        return str(path.resolve())


def main() -> int:
    args = build_parser().parse_args()
    args.tool_model = args.tool_model.resolve()
    args.output_dir = args.output_dir.resolve()
    _, spec = load_visual_spec(args.tool_model)
    sources = resolve_sources(args, spec)
    convert_and_export(sources, args.output_dir, spec, args.tool_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
