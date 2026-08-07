#!/usr/bin/env python3
"""Publish detected pin axes and virtual gripper centerlines as RViz markers.

Run from a ROS2-sourced shell, for example:

    source /opt/ros/jazzy/setup.bash
    python3 scripts/publish_alignment_markers.py outputs/demo_seed7/result.json --once
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--frame-id", default=None)
    parser.add_argument("--topic", default="/pin_axis_alignment/markers")
    parser.add_argument("--rate-hz", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    return parser


def load_ros():
    try:
        import rclpy
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker, MarkerArray
    except Exception as exc:
        print(
            "ERROR: ROS2 marker dependencies are not importable. "
            "Source ROS first, e.g. `source /opt/ros/jazzy/setup.bash`.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return rclpy, Point, Marker, MarkerArray


def point_msg(Point, xyz):
    p = Point()
    p.x = float(xyz[0])
    p.y = float(xyz[1])
    p.z = float(xyz[2])
    return p


def set_color(marker, rgba):
    marker.color.r = float(rgba[0])
    marker.color.g = float(rgba[1])
    marker.color.b = float(rgba[2])
    marker.color.a = float(rgba[3])


def add_vec(a, b, scale: float):
    return [float(a[i]) + float(scale) * float(b[i]) for i in range(3)]


def make_arrow_marker(Marker, Point, frame_id: str, ns: str, marker_id: int, start, end, rgba):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = None
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.020
    marker.scale.y = 0.060
    marker.scale.z = 0.090
    set_color(marker, rgba)
    marker.points.append(point_msg(Point, start))
    marker.points.append(point_msg(Point, end))
    return marker


def make_marker_array(data: dict, frame_id: str, Point, Marker, MarkerArray):
    arr = MarkerArray()
    stamp_placeholder = None

    axes = Marker()
    axes.header.frame_id = frame_id
    axes.header.stamp = stamp_placeholder
    axes.ns = "detected_pin_axes"
    axes.id = 1
    axes.type = Marker.LINE_LIST
    axes.action = Marker.ADD
    axes.scale.x = 0.008
    set_color(axes, (1.0, 0.05, 0.02, 1.0))
    axes.pose.orientation.w = 1.0

    for det in data["detection"]["detections"]:
        base = det["base"]
        head = det["head"]
        axis = det["axis_up"]
        start = add_vec(base, axis, -0.005)
        end = add_vec(head, axis, 0.350)
        axes.points.append(point_msg(Point, start))
        axes.points.append(point_msg(Point, end))
        arr.markers.append(
            make_arrow_marker(
                Marker,
                Point,
                frame_id,
                "static_pin_axis_extension_arrows",
                1000 + int(det["detection_id"]),
                head,
                end,
                (1.0, 0.05, 0.02, 0.95),
            )
        )
    arr.markers.append(axes)

    centerlines = Marker()
    centerlines.header.frame_id = frame_id
    centerlines.header.stamp = stamp_placeholder
    centerlines.ns = "virtual_gripper_centerlines"
    centerlines.id = 2
    centerlines.type = Marker.LINE_LIST
    centerlines.action = Marker.ADD
    centerlines.scale.x = 0.007
    set_color(centerlines, (0.02, 0.28, 1.0, 1.0))
    centerlines.pose.orientation.w = 1.0

    pregrasp_points = Marker()
    pregrasp_points.header.frame_id = frame_id
    pregrasp_points.header.stamp = stamp_placeholder
    pregrasp_points.ns = "pregrasp_points"
    pregrasp_points.id = 3
    pregrasp_points.type = Marker.SPHERE_LIST
    pregrasp_points.action = Marker.ADD
    pregrasp_points.scale.x = 0.008
    pregrasp_points.scale.y = 0.008
    pregrasp_points.scale.z = 0.008
    set_color(pregrasp_points, (0.0, 0.9, 0.2, 1.0))
    pregrasp_points.pose.orientation.w = 1.0

    for target in data["alignment"]["targets"]:
        pre = target["pregrasp_position"]
        tool_z = target["tool_z_axis_robot"]
        start = add_vec(pre, tool_z, -0.250)
        end = add_vec(pre, tool_z, 0.350)
        centerlines.points.append(point_msg(Point, start))
        centerlines.points.append(point_msg(Point, end))
        pregrasp_points.points.append(point_msg(Point, pre))
    arr.markers.append(centerlines)
    arr.markers.append(pregrasp_points)
    return arr


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

    rclpy, Point, Marker, MarkerArray = load_ros()
    rclpy.init(args=None)
    node = rclpy.create_node("pin_axis_alignment_markers")
    publisher = node.create_publisher(MarkerArray, args.topic, 10)
    marker_array = make_marker_array(data, frame_id, Point, Marker, MarkerArray)

    def publish_once():
        now = node.get_clock().now().to_msg()
        for marker in marker_array.markers:
            marker.header.stamp = now
        publisher.publish(marker_array)

    # Give RViz/subscribers a moment to connect.
    deadline = time.time() + 0.5
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    publish_once()
    print(f"Published {len(marker_array.markers)} marker groups on {args.topic} in frame {frame_id}")
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
