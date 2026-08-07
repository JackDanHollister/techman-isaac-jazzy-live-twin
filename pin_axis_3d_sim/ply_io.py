"""ASCII PLY helpers for point-cloud debug outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_point_cloud_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Write an ASCII XYZRGB point cloud PLY."""
    pts = np.asarray(points, dtype=float)
    if colors is None:
        cols = np.full((pts.shape[0], 3), 220, dtype=np.uint8)
    else:
        cols = np.asarray(colors, dtype=np.uint8)
        if cols.shape[0] != pts.shape[0]:
            raise ValueError("colors length must match points length")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, cols):
            f.write(
                f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def sample_line_points(
    start: np.ndarray,
    end: np.ndarray,
    *,
    samples: int = 48,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, samples)
    return start[None, :] * (1.0 - t[:, None]) + end[None, :] * t[:, None]


def sampled_axes_cloud(
    axes: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]],
    *,
    samples_per_axis: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a point cloud from line segments with constant colours."""
    point_parts = []
    color_parts = []
    for start, end, color in axes:
        pts = sample_line_points(start, end, samples=samples_per_axis)
        point_parts.append(pts)
        color_parts.append(np.tile(color, (pts.shape[0], 1)))
    if not point_parts:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint8)
    return np.vstack(point_parts), np.vstack(color_parts).astype(np.uint8)
