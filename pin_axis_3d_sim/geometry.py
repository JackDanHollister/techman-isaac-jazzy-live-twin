"""Small geometry helpers for 3D pin-axis estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


EPS = 1e-12


def normalize(v: np.ndarray, *, eps: float = EPS) -> np.ndarray:
    """Return a unit vector, raising if the vector is degenerate."""
    arr = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        raise ValueError("Cannot normalize a near-zero vector")
    return arr / norm


def safe_normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Normalize, returning fallback when the vector is degenerate."""
    arr = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < EPS:
        return normalize(fallback)
    return arr / norm


def angle_between_deg(a: np.ndarray, b: np.ndarray, *, unsigned_axis: bool = False) -> float:
    """Angle in degrees between vectors.

    If ``unsigned_axis`` is true, opposite directions are treated as the same axis.
    """
    aa = normalize(a)
    bb = normalize(b)
    dot = float(np.dot(aa, bb))
    if unsigned_axis:
        dot = abs(dot)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


@dataclass(frozen=True)
class Plane:
    origin: np.ndarray
    normal: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray

    def to_local(self, points: np.ndarray) -> np.ndarray:
        """Return ``[u, v, height]`` coordinates for points."""
        shifted = np.asarray(points, dtype=float) - self.origin
        return np.column_stack(
            (
                shifted @ self.u_axis,
                shifted @ self.v_axis,
                shifted @ self.normal,
            )
        )

    def from_local(self, local_points: np.ndarray) -> np.ndarray:
        """Map ``[u, v, height]`` coordinates into world coordinates."""
        local = np.asarray(local_points, dtype=float)
        return (
            self.origin
            + local[:, 0:1] * self.u_axis
            + local[:, 1:2] * self.v_axis
            + local[:, 2:3] * self.normal
        )


def make_plane_from_origin_normal(origin: np.ndarray, normal: np.ndarray) -> Plane:
    """Build a plane frame from an origin and normal."""
    n = normalize(normal)
    # Pick the least-aligned world axis as a stable helper.
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(helper, n))) > 0.85:
        helper = np.array([0.0, 1.0, 0.0])
    u = normalize(np.cross(helper, n))
    v = normalize(np.cross(n, u))
    return Plane(origin=np.asarray(origin, dtype=float), normal=n, u_axis=u, v_axis=v)


def fit_plane_svd(points: np.ndarray) -> Plane:
    """Least-squares plane fit using SVD."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        raise ValueError("Need at least 3 points to fit a plane")
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = normalize(vh[-1])
    if normal[2] < 0:
        normal = -normal
    return make_plane_from_origin_normal(centroid, normal)


def plane_distances(points: np.ndarray, origin: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Signed point-to-plane distances."""
    return (np.asarray(points, dtype=float) - origin) @ normal


def estimate_plane_ransac(
    points: np.ndarray,
    rng: np.random.Generator,
    *,
    iterations: int = 180,
    distance_threshold: float = 0.0016,
    max_points: int = 12000,
) -> tuple[Plane, np.ndarray]:
    """Robustly estimate the dominant plane and return inlier mask."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        raise ValueError("Need at least 3 points to estimate a plane")

    if pts.shape[0] > max_points:
        sample_idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        work = pts[sample_idx]
    else:
        sample_idx = np.arange(pts.shape[0])
        work = pts

    best_mask = None
    best_count = -1
    best_origin = None
    best_normal = None

    for _ in range(iterations):
        tri = work[rng.choice(work.shape[0], size=3, replace=False)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = float(np.linalg.norm(normal))
        if norm < EPS:
            continue
        normal = normal / norm
        if normal[2] < 0:
            normal = -normal
        distances = np.abs(plane_distances(work, tri[0], normal))
        mask = distances < distance_threshold
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask
            best_origin = tri[0]
            best_normal = normal

    if best_mask is None or best_count < 3 or best_origin is None or best_normal is None:
        plane = fit_plane_svd(pts)
        inliers = np.abs(plane_distances(pts, plane.origin, plane.normal)) < distance_threshold
        return plane, inliers

    # Refit from the sampled inliers, then compute the mask on the full cloud.
    refit = fit_plane_svd(work[best_mask])
    full_dist = np.abs(plane_distances(pts, refit.origin, refit.normal))
    full_mask = full_dist < distance_threshold
    if int(full_mask.sum()) >= 3:
        refit = fit_plane_svd(pts[full_mask])
        full_dist = np.abs(plane_distances(pts, refit.origin, refit.normal))
        full_mask = full_dist < distance_threshold
    return refit, full_mask


def line_distances(points: np.ndarray, point_on_line: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Perpendicular distances from points to a 3D line."""
    d = normalize(direction)
    shifted = np.asarray(points, dtype=float) - point_on_line
    projected = shifted @ d
    closest = point_on_line + projected[:, None] * d
    return np.linalg.norm(np.asarray(points, dtype=float) - closest, axis=1)


def fit_line_pca(points: np.ndarray, orient_hint: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fit a 3D line by PCA and optionally orient it toward ``orient_hint``."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        raise ValueError("Need at least 2 points to fit a line")
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = normalize(vh[0])
    if orient_hint is not None and float(np.dot(direction, orient_hint)) < 0:
        direction = -direction
    return centroid, direction


def line_endpoints_from_points(
    points: np.ndarray,
    point_on_line: np.ndarray,
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return min/max projected endpoints and projected length."""
    d = normalize(direction)
    t = (np.asarray(points, dtype=float) - point_on_line) @ d
    min_t = float(np.min(t))
    max_t = float(np.max(t))
    return point_on_line + min_t * d, point_on_line + max_t * d, max_t - min_t


def rotation_matrix_from_z_axis(z_axis: np.ndarray, x_hint: np.ndarray | None = None) -> np.ndarray:
    """Build a right-handed rotation matrix whose local +Z equals ``z_axis``.

    Columns are the local X/Y/Z axes expressed in the world frame.
    """
    z = normalize(z_axis)
    if x_hint is None:
        x_hint = np.array([1.0, 0.0, 0.0])
    x = np.asarray(x_hint, dtype=float)
    x = x - float(np.dot(x, z)) * z
    if float(np.linalg.norm(x)) < EPS:
        x = np.array([0.0, 1.0, 0.0])
        x = x - float(np.dot(x, z)) * z
    x = normalize(x)
    y = normalize(np.cross(z, x))
    x = normalize(np.cross(y, z))
    return np.column_stack((x, y, z))


def quaternion_xyzw_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion ``[x, y, z, w]``."""
    m = np.asarray(matrix, dtype=float)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return normalize(np.array([x, y, z, w], dtype=float))


def serializable_vec(v: np.ndarray, *, digits: int = 6) -> list[float]:
    """Round a vector for stable JSON output."""
    return [round(float(x), digits) for x in np.asarray(v, dtype=float).reshape(-1)]
