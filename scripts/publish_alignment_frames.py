#!/usr/bin/env python3
"""Publish pin alignment target poses as TF frames for RViz.

Run from a ROS2-sourced shell, for example:

    source /opt/ros/jazzy/setup.bash
    /usr/bin/python3 scripts/publish_alignment_frames.py outputs/demo_seed7/result.json
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
    parser.add_argument(
        "--end-effector-link",
        choices=["gripper_tcp", "flange"],
        default="gripper_tcp",
        help="Publish virtual TCP target frames or converted TM flange target frames.",
    )
    parser.add_argument(
        "--flange-to-tcp-z",
        type=float,
        default=0.16225,
        help="Approximate flange/gripper-base to 2FG7 pinch-center offset in metres.",
    )
    parser.add_argument("--once", action="store_true")
    return parser


def load_ros():
    try:
        import rclpy
        from geometry_msgs.msg import TransformStamped
        from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
    except Exception as exc:
        print(
            "ERROR: ROS2 TF dependencies are not importable. "
            "Source ROS first, e.g. `source /opt/ros/jazzy/setup.bash`.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return rclpy, TransformStamped, StaticTransformBroadcaster


def converted_position(target: dict, key: str, end_effector_link: str, flange_to_tcp_z: float):
    pos_key = {
        "pregrasp": "pregrasp_position",
        "grasp": "grasp_position",
        "lift": "lift_position",
    }[key]
    position = list(target[pos_key])
    if end_effector_link == "flange":
        tool_z_axis = target["tool_z_axis_robot"]
        position = [
            position[i] - flange_to_tcp_z * tool_z_axis[i]
            for i in range(3)
        ]
    return position


def make_transform(TransformStamped, frame_id: str, child_frame: str, position, quaternion):
    msg = TransformStamped()
    msg.header.frame_id = frame_id
    msg.child_frame_id = child_frame
    msg.transform.translation.x = float(position[0])
    msg.transform.translation.y = float(position[1])
    msg.transform.translation.z = float(position[2])
    msg.transform.rotation.x = float(quaternion[0])
    msg.transform.rotation.y = float(quaternion[1])
    msg.transform.rotation.z = float(quaternion[2])
    msg.transform.rotation.w = float(quaternion[3])
    return msg


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

    rclpy, TransformStamped, StaticTransformBroadcaster = load_ros()
    rclpy.init(args=None)
    node = rclpy.create_node("pin_axis_alignment_frames")
    broadcaster = StaticTransformBroadcaster(node)

    transforms = []
    for target in data["alignment"]["targets"]:
        detection_id = int(target["detection_id"])
        quaternion = target["quaternion_xyzw"]
        for stage in ("pregrasp", "grasp", "lift"):
            position = converted_position(
                target,
                stage,
                args.end_effector_link,
                args.flange_to_tcp_z,
            )
            child = f"pin_{detection_id:02d}_{stage}_{args.end_effector_link}"
            transforms.append(make_transform(TransformStamped, frame_id, child, position, quaternion))

    now = node.get_clock().now().to_msg()
    for transform in transforms:
        transform.header.stamp = now
    broadcaster.sendTransform(transforms)

    print(
        f"Published {len(transforms)} static target TF frames as "
        f"{args.end_effector_link} in frame {frame_id}"
    )
    if args.once:
        deadline = time.time() + 0.5
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        shutdown_quietly(rclpy, node)
        return 0

    try:
        rclpy.spin(node)
    except Exception as exc:
        if exc.__class__.__name__ != "ExternalShutdownException":
            raise
    finally:
        shutdown_quietly(rclpy, node)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
