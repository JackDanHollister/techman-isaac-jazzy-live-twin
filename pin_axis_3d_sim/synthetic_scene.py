"""Synthetic point-cloud scenes for pin-axis alignment development."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .geometry import normalize, serializable_vec


LABEL_COLORS = {
    "foam": (36, 36, 38),
    "tray_wall": (180, 184, 188),
    "pin_shaft": (222, 222, 218),
    "pin_head": (235, 235, 230),
    "specimen": (110, 76, 42),
    "noise": (90, 90, 92),
}


@dataclass(frozen=True)
class SceneConfig:
    tray_size_x: float = 0.46
    tray_size_y: float = 0.31
    tray_center_x: float = 0.0
    tray_center_y: float = 0.0
    tray_center_z: float = 0.0
    foam_z: float = 0.0
    tray_margin: float = 0.035
    min_pin_spacing: float = 0.045
    foam_points: int = 9000
    tray_wall_points: int = 1200
    background_noise_points: int = 500
    pin_shaft_points: int = 160
    pin_head_points: int = 70
    specimen_points: int = 320
    pin_radius: float = 0.00045
    pin_head_radius: float = 0.0020
    pin_length_min: float = 0.038
    pin_length_max: float = 0.057
    max_pin_tilt_deg: float = 11.0
    scanner_noise_std: float = 0.00035
    foam_noise_std: float = 0.00025
    body_height_min: float = 0.010
    body_height_max: float = 0.020
    body_radius_x_min: float = 0.006
    body_radius_x_max: float = 0.015
    body_radius_y_min: float = 0.003
    body_radius_y_max: float = 0.008
    body_radius_z_min: float = 0.002
    body_radius_z_max: float = 0.005


@dataclass(frozen=True)
class PinTruth:
    pin_id: int
    base: np.ndarray
    axis_up: np.ndarray
    length: float
    head: np.ndarray
    specimen_center: np.ndarray
    specimen_radii: np.ndarray

    def to_dict(self) -> dict:
        return {
            "pin_id": self.pin_id,
            "base": serializable_vec(self.base),
            "axis_up": serializable_vec(self.axis_up),
            "length_m": round(float(self.length), 6),
            "head": serializable_vec(self.head),
            "specimen_center": serializable_vec(self.specimen_center),
            "specimen_radii": serializable_vec(self.specimen_radii),
        }


@dataclass(frozen=True)
class SyntheticScene:
    points: np.ndarray
    colors: np.ndarray
    labels: np.ndarray
    truth: list[PinTruth]
    config: SceneConfig

    def to_metadata(self) -> dict:
        return {
            "config": asdict(self.config),
            "point_count": int(self.points.shape[0]),
            "truth": [pin.to_dict() for pin in self.truth],
        }


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = normalize(axis)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(d, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    a = normalize(np.cross(d, helper))
    b = normalize(np.cross(d, a))
    return a, b


def _random_pin_axis(rng: np.random.Generator, max_tilt_deg: float) -> np.ndarray:
    tilt = math.radians(float(rng.uniform(0.0, max_tilt_deg)))
    yaw = float(rng.uniform(-math.pi, math.pi))
    lateral = math.sin(tilt)
    return normalize(np.array([lateral * math.cos(yaw), lateral * math.sin(yaw), math.cos(tilt)]))


def _sample_pin_positions(
    rng: np.random.Generator,
    count: int,
    config: SceneConfig,
) -> list[np.ndarray]:
    positions: list[np.ndarray] = []
    attempts = 0
    while len(positions) < count and attempts < count * 400:
        attempts += 1
        x = float(rng.uniform(-config.tray_size_x / 2 + config.tray_margin, config.tray_size_x / 2 - config.tray_margin))
        y = float(rng.uniform(-config.tray_size_y / 2 + config.tray_margin, config.tray_size_y / 2 - config.tray_margin))
        pos = np.array([x, y, config.foam_z], dtype=float)
        if all(float(np.linalg.norm(pos[:2] - other[:2])) >= config.min_pin_spacing for other in positions):
            positions.append(pos)
    if len(positions) < count:
        raise RuntimeError(f"Could only place {len(positions)} pins with requested spacing")
    return positions


def _sample_ellipsoid_surface(
    rng: np.random.Generator,
    center: np.ndarray,
    radii: np.ndarray,
    basis: np.ndarray,
    count: int,
    noise_std: float,
) -> np.ndarray:
    phi = rng.uniform(0.0, 2.0 * math.pi, size=count)
    costheta = rng.uniform(-1.0, 1.0, size=count)
    sintheta = np.sqrt(1.0 - costheta * costheta)
    unit = np.column_stack((sintheta * np.cos(phi), sintheta * np.sin(phi), costheta))
    local = unit * radii
    pts = center + local @ basis.T
    pts += rng.normal(0.0, noise_std, size=pts.shape)
    return pts


def generate_scene(
    *,
    seed: int = 7,
    pin_count: int = 12,
    config: SceneConfig | None = None,
) -> SyntheticScene:
    """Generate a synthetic scanner cloud in a robot-base-like frame."""
    cfg = config or SceneConfig()
    rng = np.random.default_rng(seed)

    points_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    truth: list[PinTruth] = []

    # Foam plane.
    foam_xy = np.column_stack(
        (
            rng.uniform(-cfg.tray_size_x / 2, cfg.tray_size_x / 2, size=cfg.foam_points),
            rng.uniform(-cfg.tray_size_y / 2, cfg.tray_size_y / 2, size=cfg.foam_points),
        )
    )
    foam_z = cfg.foam_z + rng.normal(0.0, cfg.foam_noise_std, size=cfg.foam_points)
    foam = np.column_stack((foam_xy, foam_z))
    points_parts.append(foam)
    color_parts.append(np.tile(LABEL_COLORS["foam"], (foam.shape[0], 1)))
    label_parts.append(np.full(foam.shape[0], "foam", dtype=object))

    # Simple tray wall points around the perimeter.
    side = rng.integers(0, 4, size=cfg.tray_wall_points)
    wall = np.zeros((cfg.tray_wall_points, 3), dtype=float)
    wall[:, 2] = cfg.foam_z + rng.uniform(0.0, 0.014, size=cfg.tray_wall_points)
    wall[:, 0] = rng.uniform(-cfg.tray_size_x / 2, cfg.tray_size_x / 2, size=cfg.tray_wall_points)
    wall[:, 1] = rng.uniform(-cfg.tray_size_y / 2, cfg.tray_size_y / 2, size=cfg.tray_wall_points)
    wall[side == 0, 0] = -cfg.tray_size_x / 2
    wall[side == 1, 0] = cfg.tray_size_x / 2
    wall[side == 2, 1] = -cfg.tray_size_y / 2
    wall[side == 3, 1] = cfg.tray_size_y / 2
    wall += rng.normal(0.0, cfg.scanner_noise_std, size=wall.shape)
    points_parts.append(wall)
    color_parts.append(np.tile(LABEL_COLORS["tray_wall"], (wall.shape[0], 1)))
    label_parts.append(np.full(wall.shape[0], "tray_wall", dtype=object))

    bases = _sample_pin_positions(rng, pin_count, cfg)
    for pin_id, base in enumerate(bases):
        axis = _random_pin_axis(rng, cfg.max_pin_tilt_deg)
        length = float(rng.uniform(cfg.pin_length_min, cfg.pin_length_max))
        head = base + axis * length
        radial_a, radial_b = _orthonormal_basis(axis)

        # Pin shaft surface.
        t = rng.uniform(0.0015, length, size=cfg.pin_shaft_points)
        theta = rng.uniform(0.0, 2.0 * math.pi, size=cfg.pin_shaft_points)
        shaft = (
            base
            + t[:, None] * axis
            + cfg.pin_radius * np.cos(theta)[:, None] * radial_a
            + cfg.pin_radius * np.sin(theta)[:, None] * radial_b
        )
        shaft += rng.normal(0.0, cfg.scanner_noise_std, size=shaft.shape)
        points_parts.append(shaft)
        color_parts.append(np.tile(LABEL_COLORS["pin_shaft"], (shaft.shape[0], 1)))
        label_parts.append(np.full(shaft.shape[0], "pin_shaft", dtype=object))

        # Pin head sphere.
        head_dirs = rng.normal(size=(cfg.pin_head_points, 3))
        head_dirs /= np.linalg.norm(head_dirs, axis=1)[:, None]
        head_pts = head + cfg.pin_head_radius * head_dirs
        head_pts += rng.normal(0.0, cfg.scanner_noise_std, size=head_pts.shape)
        points_parts.append(head_pts)
        color_parts.append(np.tile(LABEL_COLORS["pin_head"], (head_pts.shape[0], 1)))
        label_parts.append(np.full(head_pts.shape[0], "pin_head", dtype=object))

        # Simple insect body ellipsoid, intentionally wider than the shaft.
        body_h = float(rng.uniform(cfg.body_height_min, cfg.body_height_max))
        body_center = base + axis * body_h
        radii = np.array(
            [
                rng.uniform(cfg.body_radius_x_min, cfg.body_radius_x_max),
                rng.uniform(cfg.body_radius_y_min, cfg.body_radius_y_max),
                rng.uniform(cfg.body_radius_z_min, cfg.body_radius_z_max),
            ],
            dtype=float,
        )
        body_yaw = float(rng.uniform(-math.pi, math.pi))
        body_x = normalize(math.cos(body_yaw) * radial_a + math.sin(body_yaw) * radial_b)
        body_y = normalize(np.cross(axis, body_x))
        body_z = axis
        basis = np.column_stack((body_x, body_y, body_z))
        body = _sample_ellipsoid_surface(
            rng,
            body_center,
            radii,
            basis,
            cfg.specimen_points,
            cfg.scanner_noise_std,
        )
        points_parts.append(body)
        color_parts.append(np.tile(LABEL_COLORS["specimen"], (body.shape[0], 1)))
        label_parts.append(np.full(body.shape[0], "specimen", dtype=object))

        truth.append(
            PinTruth(
                pin_id=pin_id,
                base=base,
                axis_up=axis,
                length=length,
                head=head,
                specimen_center=body_center,
                specimen_radii=radii,
            )
        )

    # Sparse outliers: dust, scan speckles, and imperfect segmentation leftovers.
    noise = np.column_stack(
        (
            rng.uniform(-cfg.tray_size_x / 2, cfg.tray_size_x / 2, size=cfg.background_noise_points),
            rng.uniform(-cfg.tray_size_y / 2, cfg.tray_size_y / 2, size=cfg.background_noise_points),
            rng.uniform(-0.002, 0.07, size=cfg.background_noise_points),
        )
    )
    points_parts.append(noise)
    color_parts.append(np.tile(LABEL_COLORS["noise"], (noise.shape[0], 1)))
    label_parts.append(np.full(noise.shape[0], "noise", dtype=object))

    points = np.vstack(points_parts)
    colors = np.vstack(color_parts).astype(np.uint8)
    labels = np.concatenate(label_parts)
    offset = np.array([cfg.tray_center_x, cfg.tray_center_y, cfg.tray_center_z], dtype=float)
    if float(np.linalg.norm(offset)) > 0.0:
        points = points + offset
        truth = [
            PinTruth(
                pin_id=pin.pin_id,
                base=pin.base + offset,
                axis_up=pin.axis_up,
                length=pin.length,
                head=pin.head + offset,
                specimen_center=pin.specimen_center + offset,
                specimen_radii=pin.specimen_radii,
            )
            for pin in truth
        ]
    return SyntheticScene(points=points, colors=colors, labels=labels, truth=truth, config=cfg)
