"""Point-cloud loading and visual-only USD authoring for workcell scans.

The geometry helpers in this module deliberately have no Isaac Sim dependency.
Only :func:`author_usd_points` imports ``pxr``, so scan preparation and tests can
run in a regular Python environment.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from typing import Any

import numpy as np


_PLY_SCALAR_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
    "int64": "<i8",
    "uint64": "<u8",
}


def _read_binary_little_endian_vertex_layout(
    stream: Any,
) -> tuple[int, np.dtype]:
    """Read a PLY header and return its vertex count and structured dtype."""
    if stream.readline() != b"ply\n":
        raise ValueError("not a PLY file: expected the 'ply' magic line")

    file_format: str | None = None
    elements: list[tuple[str, int]] = []
    current_element: str | None = None
    vertex_properties: list[tuple[str, str]] = []
    header_bytes = 4

    while True:
        raw_line = stream.readline()
        if not raw_line:
            raise ValueError("truncated PLY header: missing end_header")
        header_bytes += len(raw_line)
        if header_bytes > 1024 * 1024:
            raise ValueError("PLY header exceeds the 1 MiB safety limit")
        try:
            line = raw_line.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("PLY header is not ASCII") from exc

        if line == "end_header":
            break
        if not line or line.startswith(("comment ", "obj_info ")):
            continue

        parts = line.split()
        if parts[0] == "format":
            if len(parts) != 3:
                raise ValueError(f"invalid PLY format declaration: {line!r}")
            file_format = parts[1]
        elif parts[0] == "element":
            if len(parts) != 3:
                raise ValueError(f"invalid PLY element declaration: {line!r}")
            try:
                count = int(parts[2])
            except ValueError as exc:
                raise ValueError(f"invalid PLY element count: {parts[2]!r}") from exc
            if count < 0:
                raise ValueError("PLY element counts cannot be negative")
            current_element = parts[1]
            elements.append((current_element, count))
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) >= 2 and parts[1] == "list":
                raise ValueError("list-valued vertex properties are not supported")
            if len(parts) != 3:
                raise ValueError(f"invalid PLY vertex property: {line!r}")
            scalar_type, name = parts[1], parts[2]
            if scalar_type not in _PLY_SCALAR_DTYPES:
                raise ValueError(f"unsupported PLY scalar type: {scalar_type!r}")
            if any(existing_name == name for existing_name, _ in vertex_properties):
                raise ValueError(f"duplicate PLY vertex property: {name!r}")
            vertex_properties.append((name, scalar_type))

    if file_format != "binary_little_endian":
        raise ValueError(
            "expected a binary_little_endian PLY, "
            f"found {file_format or 'no format declaration'}"
        )
    vertex_elements = [element for element in elements if element[0] == "vertex"]
    if len(vertex_elements) != 1:
        raise ValueError("PLY must contain exactly one vertex element")
    if not elements or elements[0][0] != "vertex":
        raise ValueError("the vertex element must be the first PLY data element")

    required = {"x", "y", "z", "red", "green", "blue"}
    available = {name for name, _ in vertex_properties}
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"PLY vertex element is missing properties: {', '.join(missing)}")

    dtype = np.dtype(
        [(name, _PLY_SCALAR_DTYPES[scalar_type]) for name, scalar_type in vertex_properties]
    )
    return vertex_elements[0][1], dtype


def _validated_points(points: np.ndarray) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("points must contain only finite values")
    return result


def _validated_colors(colors: np.ndarray, point_count: int) -> np.ndarray:
    source = np.asarray(colors)
    if source.ndim != 2 or source.shape != (point_count, 3):
        raise ValueError(
            f"colors must have shape ({point_count}, 3), got {source.shape}"
        )
    if not np.issubdtype(source.dtype, np.number):
        raise ValueError("colors must be numeric RGB values")
    numeric = np.asarray(source, dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("colors must contain only finite values")
    if np.any(numeric < 0.0) or np.any(numeric > 255.0):
        raise ValueError("colors must be in the inclusive RGB range 0..255")
    return np.rint(numeric).astype(np.uint8)


def load_ply_xyzrgb(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load XYZ points and RGB colours from a binary little-endian PLY.

    Point coordinates are returned unchanged as ``float64``; callers can apply
    source-unit conversion and placement with :func:`transform_points`. Colours
    are returned as RGB ``uint8`` values in the range 0..255.
    """
    source_path = Path(path).expanduser()
    with source_path.open("rb") as stream:
        vertex_count, dtype = _read_binary_little_endian_vertex_layout(stream)
        records = np.fromfile(stream, dtype=dtype, count=vertex_count)

    if records.shape[0] != vertex_count:
        raise ValueError(
            f"truncated PLY vertex data: expected {vertex_count}, "
            f"read {records.shape[0]}"
        )
    if vertex_count == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)

    points = np.column_stack((records["x"], records["y"], records["z"])).astype(
        np.float64,
        copy=False,
    )
    colors = np.column_stack(
        (records["red"], records["green"], records["blue"])
    )
    points = _validated_points(points)
    colors = _validated_colors(colors, vertex_count)
    return points, colors


def transform_points(
    points: np.ndarray,
    scale: float,
    translation_xyz: tuple[float, float, float] | list[float] | np.ndarray,
    quaternion_xyzw: tuple[float, float, float, float] | list[float] | np.ndarray,
) -> np.ndarray:
    """Apply unit scale followed by an XYZW-quaternion rigid transform."""
    source = _validated_points(points)
    unit_scale = float(scale)
    if not np.isfinite(unit_scale) or unit_scale <= 0.0:
        raise ValueError("scale must be finite and greater than zero")

    translation = np.asarray(translation_xyz, dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("translation_xyz must contain three finite values")

    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion_xyzw must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("quaternion_xyzw must have non-zero magnitude")
    x, y, z, w = quaternion / norm
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return transform_points_matrix(
        source,
        scale=unit_scale,
        translation_xyz=translation,
        rotation_matrix=rotation,
    )


def transform_points_matrix(
    points: np.ndarray,
    scale: float,
    translation_xyz: tuple[float, float, float] | list[float] | np.ndarray,
    rotation_matrix: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...],
) -> np.ndarray:
    """Apply unit scale followed by a validated proper 3-D rigid transform."""

    source = _validated_points(points)
    unit_scale = float(scale)
    if not np.isfinite(unit_scale) or unit_scale <= 0.0:
        raise ValueError("scale must be finite and greater than zero")
    translation = np.asarray(translation_xyz, dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("translation_xyz must contain three finite values")
    rotation = np.asarray(rotation_matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation_matrix must contain a finite 3x3 matrix")
    determinant = float(np.linalg.det(rotation))
    orthogonality_error = float(np.linalg.norm(rotation @ rotation.T - np.eye(3)))
    if abs(determinant - 1.0) > 1.0e-5 or orthogonality_error > 1.0e-5:
        raise ValueError(
            "rotation_matrix must be a proper orthonormal rotation "
            f"(det={determinant}, error={orthogonality_error})"
        )
    return (source * unit_scale) @ rotation.T + translation


_REGISTRATION_GATE_SPECS = {
    "rotation_determinant_abs_error_max": (
        "rotation_determinant",
        "max",
        lambda value: abs(value - 1.0),
    ),
    "orthogonality_frobenius_error_max": (
        "orthogonality_frobenius_error",
        "max",
        float,
    ),
    "sift_3d_inliers_min": ("sift_3d_inliers", "min", float),
    "independent_ransac_median_mm_max": (
        "independent_ransac_median_mm",
        "max",
        float,
    ),
    "independent_ransac_p95_mm_max": (
        "independent_ransac_p95_mm",
        "max",
        float,
    ),
    "foam_nn_median_mm_max": ("foam_nn_median_mm", "max", float),
    "foam_fraction_lt_1mm_min": ("foam_fraction_lt_1mm", "min", float),
    "nonplane_nn_median_mm_max": ("nonplane_nn_median_mm", "max", float),
    "nonplane_fraction_lt_1mm_min": (
        "nonplane_fraction_lt_1mm",
        "min",
        float,
    ),
    "rim_nn_median_mm_max": ("rim_nn_median_mm", "max", float),
    "rim_fraction_lt_1mm_min": ("rim_fraction_lt_1mm", "min", float),
    "full_source_to_target_nn_median_mm_max": (
        "full_source_to_target_nn_median_mm",
        "max",
        float,
    ),
    "full_source_to_target_fraction_lt_1mm_min": (
        "full_source_to_target_fraction_lt_1mm",
        "min",
        float,
    ),
}


def registration_validation_failures(
    validation: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> list[str]:
    """Return deterministic failures for the frozen inter-capture evidence.

    These gates qualify the registration for visual fusion only. Passing them
    does not make either capture metrology-grade, collision-qualified, or
    registered to a robot or table frame.
    """

    failures: list[str] = []
    for threshold_name, (metric_name, comparison, conversion) in (
        _REGISTRATION_GATE_SPECS.items()
    ):
        if threshold_name not in thresholds:
            failures.append(f"missing threshold {threshold_name}")
            continue
        if metric_name not in validation:
            failures.append(f"missing validation metric {metric_name}")
            continue
        try:
            limit = float(thresholds[threshold_name])
            observed = conversion(float(validation[metric_name]))
        except (TypeError, ValueError, OverflowError):
            failures.append(
                f"non-numeric registration gate {threshold_name}/{metric_name}"
            )
            continue
        if not np.isfinite(limit) or not np.isfinite(observed):
            failures.append(
                f"non-finite registration gate {threshold_name}/{metric_name}"
            )
            continue
        failed = observed > limit if comparison == "max" else observed < limit
        if failed:
            operator = ">" if comparison == "max" else "<"
            failures.append(
                f"{metric_name}={observed:.12g} {operator} "
                f"{threshold_name}={limit:.12g}"
            )
    return failures


def drawer_top_rim_outline_points(
    *,
    center_xy_mm: tuple[float, float] | list[float] | np.ndarray,
    size_xy_mm: tuple[float, float] | list[float] | np.ndarray,
    short_axis_yaw_deg: float,
    rim_height_above_foam_mm: float,
    foam_plane_z_coefficients_mm: tuple[float, float, float]
    | list[float]
    | np.ndarray,
    secondary_to_primary_rotation: np.ndarray
    | list[list[float]]
    | tuple[tuple[float, ...], ...],
    secondary_to_primary_translation_mm: tuple[float, float, float]
    | list[float]
    | np.ndarray,
    scale_to_metres: float,
    primary_to_base_translation_m: tuple[float, float, float]
    | list[float]
    | np.ndarray,
    primary_to_base_quaternion_xyzw: tuple[float, float, float, float]
    | list[float]
    | np.ndarray,
) -> np.ndarray:
    """Build a closed outer-top-rim outline in the viewer's base frame.

    The audited rim lives in the 440 mm saved-local frame. It is first mapped
    into the primary 240 mm frame and then through the explicitly provisional
    scan-to-base placement. Only the observed top outline is returned: no
    underside, wall volume, collision proxy, or table relationship is implied.
    """

    center = np.asarray(center_xy_mm, dtype=np.float64)
    size = np.asarray(size_xy_mm, dtype=np.float64)
    plane = np.asarray(foam_plane_z_coefficients_mm, dtype=np.float64)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("center_xy_mm must contain two finite values")
    if size.shape != (2,) or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
        raise ValueError("size_xy_mm must contain two positive finite values")
    if plane.shape != (3,) or not np.all(np.isfinite(plane)):
        raise ValueError("foam_plane_z_coefficients_mm must contain three finite values")
    yaw_deg = float(short_axis_yaw_deg)
    rim_height = float(rim_height_above_foam_mm)
    if not np.isfinite(yaw_deg):
        raise ValueError("short_axis_yaw_deg must be finite")
    if not np.isfinite(rim_height) or rim_height <= 0.0:
        raise ValueError("rim_height_above_foam_mm must be positive and finite")

    yaw = np.deg2rad(yaw_deg)
    short_axis = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    long_axis = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float64)
    half_short, half_long = size / 2.0
    corners_xy = np.asarray(
        [
            center - half_short * short_axis - half_long * long_axis,
            center + half_short * short_axis - half_long * long_axis,
            center + half_short * short_axis + half_long * long_axis,
            center - half_short * short_axis + half_long * long_axis,
        ],
        dtype=np.float64,
    )
    corners_z = (
        plane[0] * corners_xy[:, 0]
        + plane[1] * corners_xy[:, 1]
        + plane[2]
        + rim_height
    )
    secondary_outline_mm = np.column_stack((corners_xy, corners_z))
    secondary_outline_mm = np.vstack(
        (secondary_outline_mm, secondary_outline_mm[0])
    )
    primary_outline_mm = transform_points_matrix(
        secondary_outline_mm,
        scale=1.0,
        translation_xyz=secondary_to_primary_translation_mm,
        rotation_matrix=secondary_to_primary_rotation,
    )
    return transform_points(
        primary_outline_mm,
        scale=scale_to_metres,
        translation_xyz=primary_to_base_translation_m,
        quaternion_xyzw=primary_to_base_quaternion_xyzw,
    )


def application_pin_guide_points(
    clear_start_xyz: tuple[float, float, float] | list[float] | np.ndarray,
    pinch_xyz: tuple[float, float, float] | list[float] | np.ndarray,
    specimen_near_xyz: tuple[float, float, float] | list[float] | np.ndarray,
    axis: tuple[float, float, float] | list[float] | np.ndarray,
    boundary_radius_m: float,
    segments: int,
) -> dict[str, np.ndarray]:
    """Build the local 10 mm pin guide and specimen-start boundary ring.

    The three axial points are supplied by the tool-profile metadata. The
    helper verifies that the pinch is exactly at the clear section's midpoint
    and that the section follows the declared pin axis before creating any
    viewer geometry.
    """

    start = np.asarray(clear_start_xyz, dtype=np.float64)
    pinch = np.asarray(pinch_xyz, dtype=np.float64)
    specimen = np.asarray(specimen_near_xyz, dtype=np.float64)
    pin_axis = np.asarray(axis, dtype=np.float64)
    for value, label in (
        (start, "clear_start_xyz"),
        (pinch, "pinch_xyz"),
        (specimen, "specimen_near_xyz"),
        (pin_axis, "axis"),
    ):
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{label} must contain three finite values")

    axis_norm = float(np.linalg.norm(pin_axis))
    if not np.isclose(axis_norm, 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("axis must be a unit vector")
    section = specimen - start
    section_length = float(np.linalg.norm(section))
    if section_length <= 0.0:
        raise ValueError("the clear pin section must have positive length")
    if not np.allclose(
        section,
        pin_axis * section_length,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("the clear pin section must follow the declared axis")
    if not np.allclose(
        pinch,
        (start + specimen) / 2.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("the pinch must be the clear pin section midpoint")

    radius = float(boundary_radius_m)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("boundary_radius_m must be positive and finite")
    if isinstance(segments, bool) or not isinstance(segments, (int, np.integer)):
        raise ValueError("segments must be an integer of at least eight")
    if segments < 8:
        raise ValueError("segments must be an integer of at least eight")

    # Choose the cardinal direction least parallel to the pin axis, then form
    # a stable orthonormal basis for the plane normal to that axis.
    reference = np.zeros(3, dtype=np.float64)
    reference[int(np.argmin(np.abs(pin_axis)))] = 1.0
    first_radius_axis = np.cross(pin_axis, reference)
    first_radius_axis /= np.linalg.norm(first_radius_axis)
    second_radius_axis = np.cross(pin_axis, first_radius_axis)
    angles = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    boundary = specimen + radius * (
        np.cos(angles)[:, None] * first_radius_axis
        + np.sin(angles)[:, None] * second_radius_axis
    )
    return {
        "bare_section": np.vstack((start, specimen)),
        "pinch": pinch.copy(),
        "specimen_boundary": boundary,
    }


def deterministic_voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size_m: float,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reproducible voxel centroids and mean RGB colours.

    Voxels and their members are sorted before reduction, making the result
    independent of input point order. If the voxel cloud exceeds ``max_points``,
    evenly spaced entries in the sorted voxel sequence are retained, including
    both ends of the sequence.
    """
    source_points = _validated_points(points)
    source_colors = _validated_colors(colors, source_points.shape[0])
    voxel_size = float(voxel_size_m)
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel_size_m must be finite and greater than zero")
    if max_points is not None:
        if isinstance(max_points, bool) or not isinstance(max_points, (int, np.integer)):
            raise ValueError("max_points must be a positive integer or None")
        if max_points <= 0:
            raise ValueError("max_points must be a positive integer or None")

    if source_points.shape[0] == 0:
        return source_points.copy(), source_colors.copy()

    voxel_coordinates = source_points / voxel_size
    int64_limit = float(np.iinfo(np.int64).max)
    if np.any(voxel_coordinates < -int64_limit) or np.any(voxel_coordinates >= int64_limit):
        raise ValueError("point coordinates are too large for the requested voxel size")
    voxel_keys = np.floor(voxel_coordinates).astype(np.int64)

    order = np.lexsort(
        (
            source_colors[:, 2],
            source_colors[:, 1],
            source_colors[:, 0],
            source_points[:, 2],
            source_points[:, 1],
            source_points[:, 0],
            voxel_keys[:, 2],
            voxel_keys[:, 1],
            voxel_keys[:, 0],
        )
    )
    sorted_keys = voxel_keys[order]
    sorted_points = source_points[order]
    sorted_colors = source_colors[order]
    starts = np.flatnonzero(
        np.concatenate(
            (
                np.array([True]),
                np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1),
            )
        )
    )
    counts = np.diff(np.append(starts, sorted_points.shape[0]))
    point_sums = np.add.reduceat(sorted_points, starts, axis=0)
    downsampled_points = point_sums / counts[:, None]

    color_sums = np.add.reduceat(sorted_colors.astype(np.uint64), starts, axis=0)
    downsampled_colors = (
        (color_sums + counts[:, None] // 2) // counts[:, None]
    ).astype(np.uint8)

    if max_points is None:
        return downsampled_points, downsampled_colors
    return deterministic_even_point_cap(
        downsampled_points,
        downsampled_colors,
        max_points=max_points,
    )


def deterministic_even_point_cap(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evenly retain entries from a deterministically ordered point cloud."""

    source_points = _validated_points(points)
    source_colors = _validated_colors(colors, source_points.shape[0])
    if isinstance(max_points, bool) or not isinstance(max_points, (int, np.integer)):
        raise ValueError("max_points must be a positive integer")
    if max_points <= 0:
        raise ValueError("max_points must be a positive integer")
    if source_points.shape[0] <= max_points:
        return source_points.copy(), source_colors.copy()
    if max_points == 1:
        selection = np.array([source_points.shape[0] // 2], dtype=np.int64)
    else:
        selection = (
            np.arange(max_points, dtype=np.int64)
            * (source_points.shape[0] - 1)
            // (max_points - 1)
        )
    return source_points[selection], source_colors[selection]


def author_usd_points(
    stage: Any,
    prim_path: str,
    points: np.ndarray,
    colors: np.ndarray,
    point_width_m: float,
) -> Any:
    """Author a visual-only ``UsdGeom.Points`` prim on an existing stage.

    This function must be called inside an environment that supplies Pixar USD,
    such as Isaac Sim. It intentionally adds no collision or rigid-body schema.
    """
    source_points = _validated_points(points)
    source_colors = _validated_colors(colors, source_points.shape[0])
    width = float(point_width_m)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("point_width_m must be finite and greater than zero")
    if not isinstance(prim_path, str) or not prim_path.startswith("/"):
        raise ValueError("prim_path must be an absolute USD prim path")

    try:
        from pxr import UsdGeom, Vt
    except ImportError as exc:
        raise RuntimeError("author_usd_points must run in an environment with pxr") from exc

    usd_points = UsdGeom.Points.Define(stage, prim_path)
    usd_points.GetPointsAttr().Set(
        Vt.Vec3fArray.FromNumpy(np.asarray(source_points, dtype=np.float32))
    )
    usd_points.GetWidthsAttr().Set([width])
    usd_points.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    display_colors = np.asarray(source_colors, dtype=np.float32) / 255.0
    display_color_primvar = usd_points.CreateDisplayColorPrimvar(
        UsdGeom.Tokens.vertex
    )
    display_color_primvar.Set(Vt.Vec3fArray.FromNumpy(display_colors))
    return usd_points


def author_usd_visual_polyline(
    stage: Any,
    prim_path: str,
    points: np.ndarray,
    color_rgb: tuple[int, int, int] | list[int] | np.ndarray,
    width_m: float,
    opacity: float,
    geometry_status: str,
    usd_purpose: str = "guide",
) -> Any:
    """Author a visual-only polyline with explicit non-collision metadata."""

    source_points = _validated_points(points)
    if source_points.shape[0] < 2:
        raise ValueError("a visual polyline requires at least two points")
    color = _validated_colors(np.asarray([color_rgb]), 1)[0]
    width = float(width_m)
    alpha = float(opacity)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("width_m must be finite and greater than zero")
    if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError("opacity must be finite and in the inclusive range 0..1")
    if not isinstance(prim_path, str) or not prim_path.startswith("/"):
        raise ValueError("prim_path must be an absolute USD prim path")
    if not isinstance(geometry_status, str) or not geometry_status:
        raise ValueError("geometry_status must be a non-empty string")
    allowed_purposes = {"default", "guide", "proxy", "render"}
    if usd_purpose not in allowed_purposes:
        raise ValueError(
            f"usd_purpose must be one of {sorted(allowed_purposes)}, got {usd_purpose!r}"
        )

    try:
        from pxr import Sdf, UsdGeom, Vt
    except ImportError as exc:
        raise RuntimeError(
            "author_usd_visual_polyline must run in an environment with pxr"
        ) from exc

    curve = UsdGeom.BasisCurves.Define(stage, prim_path)
    curve.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    curve.CreateWrapAttr().Set(UsdGeom.Tokens.nonperiodic)
    curve.CreateCurveVertexCountsAttr().Set([source_points.shape[0]])
    curve.CreatePointsAttr().Set(
        Vt.Vec3fArray.FromNumpy(np.asarray(source_points, dtype=np.float32))
    )
    curve.CreateWidthsAttr().Set([width])
    curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    curve.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(
        Vt.Vec3fArray.FromNumpy(
            np.asarray([color], dtype=np.float32) / 255.0
        )
    )
    curve.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set([alpha])
    purpose_token = (
        UsdGeom.Tokens.default_ if usd_purpose == "default" else usd_purpose
    )
    curve.CreatePurposeAttr().Set(purpose_token)
    prim = curve.GetPrim()
    prim.CreateAttribute(
        "magi:visualOnly", Sdf.ValueTypeNames.Bool, custom=True
    ).Set(True)
    prim.CreateAttribute(
        "magi:collisionQualified", Sdf.ValueTypeNames.Bool, custom=True
    ).Set(False)
    prim.CreateAttribute(
        "magi:geometryStatus", Sdf.ValueTypeNames.String, custom=True
    ).Set(geometry_status)
    return curve
