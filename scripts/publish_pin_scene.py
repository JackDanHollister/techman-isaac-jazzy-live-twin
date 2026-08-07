#!/usr/bin/env python3
"""Publish the synthetic pin scene as PointCloud2 plus alignment markers."""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

from publish_alignment_markers import make_marker_array


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--cloud-ply", type=Path, default=None)
    parser.add_argument("--frame-id", default=None)
    parser.add_argument("--cloud-topic", default="/pin_axis_alignment/cloud")
    parser.add_argument("--marker-topic", default="/pin_axis_alignment/markers")
    parser.add_argument("--rate-hz", type=float, default=1.0)
    parser.add_argument(
        "--max-points",
        type=int,
        default=45000,
        help="Decimate the cloud if it has more points than this.",
    )
    parser.add_argument("--once", action="store_true")
    return parser


def load_ros():
    try:
        import rclpy
        from geometry_msgs.msg import Point
        from sensor_msgs.msg import PointCloud2, PointField
        from sensor_msgs_py import point_cloud2
        from std_msgs.msg import Header
        from visualization_msgs.msg import Marker, MarkerArray
    except Exception as exc:
        print(
            "ERROR: ROS2 scene dependencies are not importable. "
            "Source ROS first, e.g. `source /opt/ros/jazzy/setup.bash`.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return rclpy, Header, PointCloud2, PointField, point_cloud2, Point, Marker, MarkerArray


def read_ascii_xyzrgb_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="ascii") as f:
        line = f.readline().strip()
        if line != "ply":
            raise ValueError(f"{path} is not a PLY file")
        vertex_count = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path} has no end_header")
            stripped = line.strip()
            if stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[-1])
            if stripped == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"{path} has no vertex count")
        points = np.zeros((vertex_count, 3), dtype=np.float32)
        colors = np.zeros((vertex_count, 3), dtype=np.uint8)
        for idx in range(vertex_count):
            parts = f.readline().split()
            if len(parts) < 6:
                raise ValueError(f"{path} ended early at vertex {idx}")
            points[idx] = [float(parts[0]), float(parts[1]), float(parts[2])]
            colors[idx] = [int(parts[3]), int(parts[4]), int(parts[5])]
    return points, colors


def decimate(points: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices], colors[indices]


def normalize(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    length = float(np.linalg.norm(arr))
    if length <= 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return arr / length


def guide_tube_points(start, end, *, radius: float = 0.007, samples: int = 120) -> np.ndarray:
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    return np.linspace(start, end, samples, dtype=np.float32)


def append_cloud_alignment_guides(data: dict, points: np.ndarray, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    guide_points = []
    guide_colors = []

    for det in data["detection"]["detections"]:
        head = np.asarray(det["head"], dtype=np.float32)
        axis = normalize(det["axis_up"])
        start = head - 0.015 * axis
        end = head + 0.500 * axis
        tube = guide_tube_points(start, end, radius=0.007, samples=130)
        guide_points.append(tube)
        guide_colors.append(np.tile(np.array([[255, 20, 0]], dtype=np.uint8), (len(tube), 1)))

    for target in data["alignment"]["targets"]:
        pre = np.asarray(target["pregrasp_position"], dtype=np.float32)
        tool_z = normalize(target["tool_z_axis_robot"])
        start = pre - 0.350 * tool_z
        end = pre + 0.450 * tool_z
        tube = guide_tube_points(start, end, radius=0.008, samples=150)
        guide_points.append(tube)
        guide_colors.append(np.tile(np.array([[0, 80, 255]], dtype=np.uint8), (len(tube), 1)))

    if not guide_points:
        return points, colors
    return (
        np.vstack([points, *guide_points]).astype(np.float32),
        np.vstack([colors, *guide_colors]).astype(np.uint8),
    )


def rgb_float(color: np.ndarray) -> float:
    packed = (int(color[0]) << 16) | (int(color[1]) << 8) | int(color[2])
    return struct.unpack("f", struct.pack("I", packed))[0]


def make_cloud_msg(Header, PointField, point_cloud2, frame_id: str, points: np.ndarray, colors: np.ndarray):
    header = Header()
    header.frame_id = frame_id
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    cloud_points = [
        (float(point[0]), float(point[1]), float(point[2]), rgb_float(color))
        for point, color in zip(points, colors)
    ]
    return point_cloud2.create_cloud(header, fields, cloud_points)


def shutdown_quietly(rclpy, node) -> None:
    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


def main() -> int:
    args = build_parser().parse_args()
    data = json.loads(args.result_json.read_text(encoding="utf-8"))
    frame_id = args.frame_id or data["frames"]["target_frame"]
    cloud_path = args.cloud_ply or args.result_json.with_name("scene_cloud.ply")

    rclpy, Header, PointCloud2, PointField, point_cloud2, Point, Marker, MarkerArray = load_ros()
    points, colors = read_ascii_xyzrgb_ply(cloud_path)
    points, colors = decimate(points, colors, args.max_points)
    points, colors = append_cloud_alignment_guides(data, points, colors)

    rclpy.init(args=None)
    node = rclpy.create_node("pin_axis_scene_publisher")
    cloud_pub = node.create_publisher(PointCloud2, args.cloud_topic, 10)
    marker_pub = node.create_publisher(MarkerArray, args.marker_topic, 10)

    cloud_msg = make_cloud_msg(Header, PointField, point_cloud2, frame_id, points, colors)
    marker_array = make_marker_array(data, frame_id, Point, Marker, MarkerArray)

    def publish_once():
        now = node.get_clock().now().to_msg()
        cloud_msg.header.stamp = now
        for marker in marker_array.markers:
            marker.header.stamp = now
        cloud_pub.publish(cloud_msg)
        marker_pub.publish(marker_array)

    deadline = time.time() + 0.8
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    publish_once()
    print(
        f"Published {len(points)} cloud points on {args.cloud_topic} and "
        f"{len(marker_array.markers)} marker groups on {args.marker_topic} in frame {frame_id}"
    )
    if args.once:
        shutdown_quietly(rclpy, node)
        return 0

    period = 1.0 / max(args.rate_hz, 0.1)
    try:
        while rclpy.ok():
            publish_once()
            rclpy.spin_once(node, timeout_sec=period)
    except Exception as exc:
        if exc.__class__.__name__ != "ExternalShutdownException":
            raise
    finally:
        shutdown_quietly(rclpy, node)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
