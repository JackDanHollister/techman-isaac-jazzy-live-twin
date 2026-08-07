"""Pin-axis detection from unlabelled 3D point clouds."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .geometry import (
    Plane,
    angle_between_deg,
    estimate_plane_ransac,
    fit_line_pca,
    line_distances,
    line_endpoints_from_points,
    normalize,
    serializable_vec,
)


@dataclass(frozen=True)
class DetectionConfig:
    plane_ransac_iterations: int = 90
    plane_distance_threshold: float = 0.0016
    min_height: float = 0.016
    max_height: float = 0.075
    cluster_radius: float = 0.014
    min_cluster_points: int = 45
    line_ransac_iterations: int = 90
    line_inlier_radius: float = 0.0026
    max_axis_angle_from_plane_normal_deg: float = 28.0
    min_axis_length: float = 0.024
    min_line_inliers: int = 35
    max_radial_rms: float = 0.0022
    duplicate_axis_distance: float = 0.012


@dataclass(frozen=True)
class PinAxisDetection:
    detection_id: int
    point_on_axis: np.ndarray
    axis_up: np.ndarray
    base: np.ndarray
    head: np.ndarray
    length: float
    inlier_count: int
    cluster_point_count: int
    radial_rms: float
    angle_from_plane_normal_deg: float
    score: float

    def to_dict(self) -> dict:
        return {
            "detection_id": self.detection_id,
            "point_on_axis": serializable_vec(self.point_on_axis),
            "axis_up": serializable_vec(self.axis_up),
            "base": serializable_vec(self.base),
            "head": serializable_vec(self.head),
            "length_m": round(float(self.length), 6),
            "inlier_count": int(self.inlier_count),
            "cluster_point_count": int(self.cluster_point_count),
            "radial_rms_m": round(float(self.radial_rms), 6),
            "angle_from_plane_normal_deg": round(float(self.angle_from_plane_normal_deg), 3),
            "score": round(float(self.score), 3),
        }


@dataclass(frozen=True)
class DetectionResult:
    plane: Plane
    plane_inlier_count: int
    config: DetectionConfig
    detections: list[PinAxisDetection]

    def to_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "plane": {
                "origin": serializable_vec(self.plane.origin),
                "normal": serializable_vec(self.plane.normal),
                "u_axis": serializable_vec(self.plane.u_axis),
                "v_axis": serializable_vec(self.plane.v_axis),
                "inlier_count": int(self.plane_inlier_count),
            },
            "detections": [d.to_dict() for d in self.detections],
        }


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def _cluster_xy(uv: np.ndarray, radius: float, min_points: int) -> list[np.ndarray]:
    """Fast connected components over occupied top-down grid cells.

    This deliberately avoids SciPy/sklearn imports so the tool starts reliably in
    the robot workstation environment. The synthetic tray keeps pins much farther
    apart than one grid cell, so cell-level connectivity is adequate for this
    prototype and much faster than point-pair radius checks.
    """
    if uv.shape[0] == 0:
        return []
    cell_size = radius
    cells: dict[tuple[int, int], list[int]] = {}
    grid = np.floor(uv / cell_size).astype(int)
    for idx, key in enumerate(map(tuple, grid)):
        cells.setdefault(key, []).append(idx)

    min_cell_points = max(4, min_points // 20)
    cells = {key: value for key, value in cells.items() if len(value) >= min_cell_points}
    if not cells:
        return []

    cell_keys = list(cells.keys())
    cell_index = {key: i for i, key in enumerate(cell_keys)}
    uf = _UnionFind(len(cell_keys))

    for key, idx in cell_index.items():
        gx, gy = key
        for nx in range(gx - 1, gx + 2):
            for ny in range(gy - 1, gy + 2):
                other_idx = cell_index.get((nx, ny))
                if other_idx is not None:
                    uf.union(idx, other_idx)

    groups: dict[int, list[int]] = {}
    for key, point_indices in cells.items():
        root = uf.find(cell_index[key])
        groups.setdefault(root, []).extend(point_indices)
    return [np.array(indices, dtype=int) for indices in groups.values() if len(indices) >= min_points]


def _fit_axis_ransac(
    cluster_points: np.ndarray,
    plane_normal: np.ndarray,
    rng: np.random.Generator,
    config: DetectionConfig,
) -> PinAxisDetection | None:
    if cluster_points.shape[0] < config.min_cluster_points:
        return None

    best_mask = None
    best_score = -1.0
    best_line = None

    n = cluster_points.shape[0]
    for _ in range(config.line_ransac_iterations):
        i, j = rng.choice(n, size=2, replace=False)
        delta = cluster_points[j] - cluster_points[i]
        if float(np.linalg.norm(delta)) < 0.004:
            continue
        direction = normalize(delta)
        if float(np.dot(direction, plane_normal)) < 0:
            direction = -direction
        angle = angle_between_deg(direction, plane_normal)
        if angle > config.max_axis_angle_from_plane_normal_deg:
            continue
        distances = line_distances(cluster_points, cluster_points[i], direction)
        mask = distances < config.line_inlier_radius
        inlier_count = int(mask.sum())
        if inlier_count < config.min_line_inliers:
            continue
        _, _, length = line_endpoints_from_points(cluster_points[mask], cluster_points[i], direction)
        if length < config.min_axis_length:
            continue
        # Prefer long, vertical, well-supported line hypotheses.
        score = inlier_count * length * max(0.2, math.cos(math.radians(angle)))
        if score > best_score:
            best_score = score
            best_mask = mask
            best_line = (cluster_points[i], direction)

    if best_mask is None or best_line is None:
        return None

    inliers = cluster_points[best_mask]
    point, direction = fit_line_pca(inliers, orient_hint=plane_normal)
    distances = line_distances(cluster_points, point, direction)
    mask = distances < config.line_inlier_radius
    inliers = cluster_points[mask]
    if inliers.shape[0] < config.min_line_inliers:
        return None

    point, direction = fit_line_pca(inliers, orient_hint=plane_normal)
    base, head, length = line_endpoints_from_points(inliers, point, direction)
    radial = line_distances(inliers, point, direction)
    radial_rms = float(np.sqrt(np.mean(radial * radial)))
    angle = angle_between_deg(direction, plane_normal)

    if length < config.min_axis_length:
        return None
    if radial_rms > config.max_radial_rms:
        return None
    if angle > config.max_axis_angle_from_plane_normal_deg:
        return None

    score = float(inliers.shape[0] * length / max(radial_rms, 1e-4))
    return PinAxisDetection(
        detection_id=-1,
        point_on_axis=point,
        axis_up=direction,
        base=base,
        head=head,
        length=float(length),
        inlier_count=int(inliers.shape[0]),
        cluster_point_count=int(cluster_points.shape[0]),
        radial_rms=radial_rms,
        angle_from_plane_normal_deg=float(angle),
        score=score,
    )


def _deduplicate_axes(
    detections: list[PinAxisDetection],
    *,
    distance_threshold: float,
) -> list[PinAxisDetection]:
    kept: list[PinAxisDetection] = []
    for det in sorted(detections, key=lambda d: d.score, reverse=True):
        duplicate = False
        for existing in kept:
            # Distance from new head to existing axis.
            dist = float(line_distances(det.head[None, :], existing.point_on_axis, existing.axis_up)[0])
            if dist < distance_threshold and angle_between_deg(det.axis_up, existing.axis_up, unsigned_axis=True) < 8.0:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)

    # Stable order: tray scan order by base x/y, not score.
    ordered = sorted(kept, key=lambda d: (float(d.base[0]), float(d.base[1])))
    return [
        PinAxisDetection(
            detection_id=i,
            point_on_axis=d.point_on_axis,
            axis_up=d.axis_up,
            base=d.base,
            head=d.head,
            length=d.length,
            inlier_count=d.inlier_count,
            cluster_point_count=d.cluster_point_count,
            radial_rms=d.radial_rms,
            angle_from_plane_normal_deg=d.angle_from_plane_normal_deg,
            score=d.score,
        )
        for i, d in enumerate(ordered)
    ]


def detect_pin_axes(
    points: np.ndarray,
    *,
    seed: int = 11,
    config: DetectionConfig | None = None,
) -> DetectionResult:
    """Detect likely pin axes from an unlabelled 3D point cloud."""
    cfg = config or DetectionConfig()
    rng = np.random.default_rng(seed)
    pts = np.asarray(points, dtype=float)

    plane, plane_inliers = estimate_plane_ransac(
        pts,
        rng,
        iterations=cfg.plane_ransac_iterations,
        distance_threshold=cfg.plane_distance_threshold,
    )
    local = plane.to_local(pts)
    height = local[:, 2]
    candidate_mask = (height > cfg.min_height) & (height < cfg.max_height)
    # Avoid using the fitted tray plane itself even if RANSAC tolerance is loose.
    candidate_mask &= ~plane_inliers
    candidate_points = pts[candidate_mask]
    candidate_uv = local[candidate_mask, :2]

    clusters = _cluster_xy(candidate_uv, cfg.cluster_radius, cfg.min_cluster_points)
    raw_detections: list[PinAxisDetection] = []
    for cluster_indices in clusters:
        cluster_points = candidate_points[cluster_indices]
        det = _fit_axis_ransac(cluster_points, plane.normal, rng, cfg)
        if det is not None:
            raw_detections.append(det)

    detections = _deduplicate_axes(raw_detections, distance_threshold=cfg.duplicate_axis_distance)
    return DetectionResult(
        plane=plane,
        plane_inlier_count=int(plane_inliers.sum()),
        config=cfg,
        detections=detections,
    )
