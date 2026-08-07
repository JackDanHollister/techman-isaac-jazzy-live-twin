#!/usr/bin/env python3
"""Step the fake TM5S arm through generated pin-alignment targets.

This node is for RViz/MoveIt demonstration only. It asks MoveIt for IK based on
the generated pin target poses, then sends the fake trajectory controller a
single-point joint trajectory for each stage.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import threading
import time
from math import sqrt
from pathlib import Path


JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets_json", type=Path)
    parser.add_argument("--group-name", default="tmr_arm")
    parser.add_argument("--ik-link-name", default="flange")
    parser.add_argument(
        "--motion-mode",
        choices=["joint_state", "trajectory"],
        default="joint_state",
        help="Use direct /joint_states replay for RViz demos, or a FollowJointTrajectory action.",
    )
    parser.add_argument("--controller-action", default="/tmr_arm_controller/follow_joint_trajectory")
    parser.add_argument("--compute-ik-service", default="/compute_ik")
    parser.add_argument("--frame-id", default=None)
    parser.add_argument("--marker-topic", default="/pin_axis_alignment/live_markers")
    parser.add_argument("--legacy-marker-topic", default="/pin_axis_alignment/markers")
    parser.add_argument("--live-cloud-topic", default="/pin_axis_alignment/live_cloud")
    parser.add_argument("--max-pins", type=int, default=13)
    parser.add_argument("--move-seconds", type=float, default=3.0)
    parser.add_argument("--settle-seconds", type=float, default=0.6)
    parser.add_argument("--motion-rate-hz", type=float, default=60.0)
    parser.add_argument("--ik-timeout", type=float, default=1.0)
    parser.add_argument(
        "--flange-to-tcp-z",
        type=float,
        default=0.16225,
        help="Approximate flange/gripper-base to 2FG7 pinch-center offset in metres.",
    )
    parser.add_argument(
        "--alignment-marker-length",
        type=float,
        default=0.65,
        help="Length of live pin/gripper alignment guide lines in RViz.",
    )
    parser.add_argument(
        "--alignment-hold-seconds",
        type=float,
        default=1.0,
        help="Extra pause after each reached target so the green aligned marker is visible.",
    )
    parser.add_argument(
        "--ready-joints",
        nargs=6,
        type=float,
        default=[0.0, 0.0, 1.5708, 0.0, 1.5708, 0.0],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Joint pose sent before the pin sequence so the fake arm visibly leaves zero pose.",
    )
    parser.add_argument(
        "--skip-ready",
        action="store_true",
        help="Do not move to the ready joint pose before the pin sequence.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["pregrasp", "grasp", "lift"],
        default=["pregrasp", "grasp", "lift"],
        help="Target stages to solve and move through for each pin.",
    )
    parser.add_argument(
        "--manual-step",
        action="store_true",
        help="Wait for Enter before each pin. Stage motion within a pin still runs continuously.",
    )
    parser.add_argument(
        "--auto-delay",
        type=float,
        default=1.0,
        help="Delay between pins when not using --manual-step.",
    )
    parser.add_argument(
        "--hold-open",
        action="store_true",
        help="Keep publishing the final joint state until interrupted.",
    )
    return parser


def load_ros():
    try:
        import rclpy
        from action_msgs.msg import GoalStatus
        from builtin_interfaces.msg import Duration
        from control_msgs.action import FollowJointTrajectory
        from geometry_msgs.msg import Point, PoseStamped
        from moveit_msgs.msg import MoveItErrorCodes, RobotState
        from moveit_msgs.srv import GetPositionIK
        from rclpy.action import ActionClient
        from sensor_msgs.msg import JointState, PointCloud2, PointField
        from sensor_msgs_py import point_cloud2
        from std_msgs.msg import Header
        from trajectory_msgs.msg import JointTrajectoryPoint
        from visualization_msgs.msg import Marker, MarkerArray
    except Exception as exc:
        print(
            "ERROR: ROS2/MoveIt demo dependencies are not importable. "
            "Source ROS first, e.g. `source /opt/ros/jazzy/setup.bash`.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return {
        "rclpy": rclpy,
        "GoalStatus": GoalStatus,
        "Duration": Duration,
        "FollowJointTrajectory": FollowJointTrajectory,
        "Header": Header,
        "Point": Point,
        "PointCloud2": PointCloud2,
        "PointField": PointField,
        "point_cloud2": point_cloud2,
        "PoseStamped": PoseStamped,
        "MoveItErrorCodes": MoveItErrorCodes,
        "RobotState": RobotState,
        "GetPositionIK": GetPositionIK,
        "ActionClient": ActionClient,
        "JointState": JointState,
        "JointTrajectoryPoint": JointTrajectoryPoint,
        "Marker": Marker,
        "MarkerArray": MarkerArray,
    }


def duration_msg(Duration, seconds: float):
    whole = int(seconds)
    return Duration(sec=whole, nanosec=int((seconds - whole) * 1_000_000_000))


def pose_msg(PoseStamped, frame_id: str, pose_data: dict):
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    pos = pose_data["position"]
    quat = pose_data["orientation"]
    msg.pose.position.x = float(pos["x"])
    msg.pose.position.y = float(pos["y"])
    msg.pose.position.z = float(pos["z"])
    msg.pose.orientation.x = float(quat["x"])
    msg.pose.orientation.y = float(quat["y"])
    msg.pose.orientation.z = float(quat["z"])
    msg.pose.orientation.w = float(quat["w"])
    return msg


def normalize(vec: list[float]) -> list[float]:
    length = sqrt(sum(float(value) * float(value) for value in vec))
    if length <= 1e-12:
        return [0.0, 0.0, 1.0]
    return [float(value) / length for value in vec]


def pose_position(pose_data: dict) -> list[float]:
    pos = pose_data["position"]
    return [float(pos["x"]), float(pos["y"]), float(pos["z"])]


def pose_z_axis(pose_data: dict) -> list[float]:
    quat = pose_data["orientation"]
    x = float(quat["x"])
    y = float(quat["y"])
    z = float(quat["z"])
    w = float(quat["w"])
    return normalize(
        [
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ]
    )


def add_vec(a: list[float], b: list[float], scale: float = 1.0) -> list[float]:
    return [float(a[i]) + float(scale) * float(b[i]) for i in range(3)]


def lerp_vec(a: list[float], b: list[float], alpha: float) -> list[float]:
    return [float(a[i]) * (1.0 - alpha) + float(b[i]) * alpha for i in range(3)]


def rgb_float(rgb: tuple[int, int, int]) -> float:
    packed = (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])
    return struct.unpack("f", struct.pack("I", packed))[0]


def guide_tube_points(start: list[float], end: list[float], *, radius: float, samples: int) -> list[list[float]]:
    points = []
    for idx in range(max(samples, 2)):
        alpha = idx / max(samples - 1, 1)
        points.append(lerp_vec(start, end, alpha))
    return points


def tcp_position_from_flange_pose(pose_data: dict, flange_to_tcp_z: float) -> list[float]:
    return add_vec(pose_position(pose_data), pose_z_axis(pose_data), flange_to_tcp_z)


class DemoPlayer:
    def __init__(self, args: argparse.Namespace, ros: dict):
        self.args = args
        self.ros = ros
        self.rclpy = ros["rclpy"]
        self.node = self.rclpy.create_node("pin_axis_alignment_demo_player")
        self.latest_positions = {name: 0.0 for name in JOINT_NAMES}
        self.position_lock = threading.Lock()
        self.have_joint_state = False
        self.stop_publishing = threading.Event()
        self.publisher_thread: threading.Thread | None = None
        self.last_stage_pose: dict | None = None
        self.last_live_marker_array = None
        self.last_live_cloud_msg = None
        self.node.create_subscription(
            ros["JointState"],
            "/joint_states",
            self.joint_state_callback,
            10,
        )
        self.joint_state_pub = self.node.create_publisher(ros["JointState"], "/joint_states", 10)
        self.marker_pub = self.node.create_publisher(ros["MarkerArray"], args.marker_topic, 10)
        self.legacy_marker_pub = self.node.create_publisher(ros["MarkerArray"], args.legacy_marker_topic, 10)
        self.live_cloud_pub = self.node.create_publisher(ros["PointCloud2"], args.live_cloud_topic, 10)
        self.ik_client = self.node.create_client(ros["GetPositionIK"], args.compute_ik_service)
        self.trajectory_client = None
        if args.motion_mode == "trajectory":
            self.trajectory_client = ros["ActionClient"](
                self.node,
                ros["FollowJointTrajectory"],
                args.controller_action,
            )

    def joint_state_callback(self, msg):
        name_to_position = dict(zip(msg.name, msg.position))
        updated = False
        with self.position_lock:
            for joint in JOINT_NAMES:
                if joint in name_to_position:
                    self.latest_positions[joint] = float(name_to_position[joint])
                    updated = True
        self.have_joint_state = self.have_joint_state or updated

    def wait_until_ready(self) -> None:
        print("Waiting for MoveIt /compute_ik service...")
        if not self.ik_client.wait_for_service(timeout_sec=30.0):
            raise RuntimeError("Timed out waiting for /compute_ik")
        if self.args.motion_mode == "trajectory":
            print("Waiting for fake trajectory controller action...")
            if self.trajectory_client is None or not self.trajectory_client.wait_for_server(timeout_sec=30.0):
                raise RuntimeError(f"Timed out waiting for {self.args.controller_action}")
        else:
            print("Using direct /joint_states replay for visible RViz motion.")

        deadline = time.time() + 8.0
        while time.time() < deadline and not self.have_joint_state:
            self.publish_joint_state(self.current_joint_positions())
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
        if self.have_joint_state:
            print("Received current joint state seed.")
        else:
            print("No joint state received yet; using zero-joint seed for the first IK request.")

    def current_joint_positions(self) -> list[float]:
        with self.position_lock:
            return [self.latest_positions[name] for name in JOINT_NAMES]

    def set_current_joint_positions(self, positions: list[float]) -> None:
        with self.position_lock:
            for joint, value in zip(JOINT_NAMES, positions):
                self.latest_positions[joint] = float(value)

    def publish_joint_state(self, positions: list[float]) -> None:
        msg = self.ros["JointState"]()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = [float(value) for value in positions]
        msg.velocity = [0.0] * len(JOINT_NAMES)
        self.joint_state_pub.publish(msg)

    def start_continuous_joint_state_publishing(self) -> None:
        if self.publisher_thread is not None:
            return

        def run():
            period = 1.0 / max(float(self.args.motion_rate_hz), 1.0)
            while not self.stop_publishing.is_set():
                self.publish_joint_state(self.current_joint_positions())
                time.sleep(period)

        self.publisher_thread = threading.Thread(target=run, name="joint_state_replay_publisher", daemon=True)
        self.publisher_thread.start()

    def replay_joint_state_motion(self, positions: list[float], progress_callback=None) -> bool:
        start = self.current_joint_positions()
        end = [float(value) for value in positions]
        duration = max(float(self.args.move_seconds), 0.05)
        rate_hz = max(float(self.args.motion_rate_hz), 1.0)
        steps = max(2, int(duration * rate_hz))
        start_time = time.monotonic()
        for idx in range(steps + 1):
            alpha = idx / steps
            # Smoothstep interpolation keeps starts/stops visually calmer.
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            current = [
                start[j] * (1.0 - smooth) + end[j] * smooth
                for j in range(len(JOINT_NAMES))
            ]
            self.publish_joint_state(current)
            self.set_current_joint_positions(current)
            if progress_callback is not None:
                progress_callback(smooth)
            self.rclpy.spin_once(self.node, timeout_sec=0.001)
            next_time = start_time + ((idx + 1) / rate_hz)
            sleep_for = min(next_time - time.monotonic(), duration - (time.monotonic() - start_time))
            if sleep_for > 0.0:
                time.sleep(sleep_for)
        return True

    def solve_ik(self, frame_id: str, pose_data: dict) -> list[float] | None:
        req = self.ros["GetPositionIK"].Request()
        req.ik_request.group_name = self.args.group_name
        req.ik_request.ik_link_name = self.args.ik_link_name
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout = duration_msg(self.ros["Duration"], self.args.ik_timeout)
        req.ik_request.pose_stamped = pose_msg(self.ros["PoseStamped"], frame_id, pose_data)

        state = self.ros["RobotState"]()
        state.joint_state.name = list(JOINT_NAMES)
        state.joint_state.position = self.current_joint_positions()
        req.ik_request.robot_state = state

        future = self.ik_client.call_async(req)
        self.rclpy.spin_until_future_complete(self.node, future, timeout_sec=self.args.ik_timeout + 2.0)
        if not future.done() or future.result() is None:
            return None

        response = future.result()
        if response.error_code.val != self.ros["MoveItErrorCodes"].SUCCESS:
            return None

        solution = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position))
        if not all(name in solution for name in JOINT_NAMES):
            return None
        return [float(solution[name]) for name in JOINT_NAMES]

    def send_joint_goal(self, positions: list[float], progress_callback=None) -> bool:
        if self.args.motion_mode == "joint_state":
            return self.replay_joint_state_motion(positions, progress_callback=progress_callback)
        if self.trajectory_client is None:
            return False
        goal = self.ros["FollowJointTrajectory"].Goal()
        goal.trajectory.joint_names = list(JOINT_NAMES)
        point = self.ros["JointTrajectoryPoint"]()
        point.positions = [float(value) for value in positions]
        point.velocities = [0.0] * len(JOINT_NAMES)
        point.time_from_start = duration_msg(self.ros["Duration"], self.args.move_seconds)
        goal.trajectory.points = [point]

        future = self.trajectory_client.send_goal_async(goal)
        self.rclpy.spin_until_future_complete(self.node, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            return False

        result_future = goal_handle.get_result_async()
        self.rclpy.spin_until_future_complete(
            self.node,
            result_future,
            timeout_sec=self.args.move_seconds + 5.0,
        )
        if not result_future.done() or result_future.result() is None:
            return False
        status = result_future.result().status
        if status != self.ros["GoalStatus"].STATUS_SUCCEEDED:
            print(f"  trajectory status was {status}")
            return False

        self.set_current_joint_positions(positions)
        return True

    def point_msg(self, xyz: list[float]):
        point = self.ros["Point"]()
        point.x = float(xyz[0])
        point.y = float(xyz[1])
        point.z = float(xyz[2])
        return point

    def set_marker_color(self, marker, rgba: tuple[float, float, float, float]) -> None:
        marker.color.r = float(rgba[0])
        marker.color.g = float(rgba[1])
        marker.color.b = float(rgba[2])
        marker.color.a = float(rgba[3])

    def make_line_marker(
        self,
        frame_id: str,
        ns: str,
        marker_id: int,
        points: list[list[float]],
        rgba: tuple[float, float, float, float],
        width: float,
    ):
        marker = self.ros["Marker"]()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = marker_id
        marker.type = marker.LINE_LIST
        marker.action = marker.ADD
        marker.scale.x = float(width)
        marker.pose.orientation.w = 1.0
        self.set_marker_color(marker, rgba)
        marker.points = [self.point_msg(point) for point in points]
        return marker

    def make_arrow_marker(
        self,
        frame_id: str,
        ns: str,
        marker_id: int,
        start: list[float],
        end: list[float],
        rgba: tuple[float, float, float, float],
        *,
        shaft_diameter: float,
        head_diameter: float,
        head_length: float,
    ):
        marker = self.ros["Marker"]()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = marker_id
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(shaft_diameter)
        marker.scale.y = float(head_diameter)
        marker.scale.z = float(head_length)
        self.set_marker_color(marker, rgba)
        marker.points = [self.point_msg(start), self.point_msg(end)]
        return marker

    def make_text_marker(
        self,
        frame_id: str,
        target: dict,
        xyz: list[float],
        text: str,
        rgba: tuple[float, float, float, float],
    ):
        marker = self.ros["Marker"]()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "live_alignment_label"
        marker.id = 102
        marker.type = marker.TEXT_VIEW_FACING
        marker.action = marker.ADD
        marker.pose.position.x = float(xyz[0])
        marker.pose.position.y = float(xyz[1])
        marker.pose.position.z = float(xyz[2])
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.035
        self.set_marker_color(marker, rgba)
        marker.text = text
        return marker

    def make_sphere_marker(
        self,
        frame_id: str,
        marker_id: int,
        xyz: list[float],
        rgba: tuple[float, float, float, float],
        diameter: float,
    ):
        marker = self.ros["Marker"]()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "live_alignment_points"
        marker.id = marker_id
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.pose.position.x = float(xyz[0])
        marker.pose.position.y = float(xyz[1])
        marker.pose.position.z = float(xyz[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(diameter)
        marker.scale.y = float(diameter)
        marker.scale.z = float(diameter)
        self.set_marker_color(marker, rgba)
        return marker

    def publish_live_alignment_cloud(
        self,
        frame_id: str,
        pin_start: list[float],
        pin_end: list[float],
        gripper_start: list[float],
        gripper_end: list[float],
        pin_rgb: tuple[int, int, int],
        gripper_rgb: tuple[int, int, int],
    ) -> None:
        header = self.ros["Header"]()
        header.frame_id = frame_id
        header.stamp = self.node.get_clock().now().to_msg()
        PointField = self.ros["PointField"]
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        cloud_points = []
        pin_color = rgb_float(pin_rgb)
        for point in guide_tube_points(pin_start, pin_end, radius=0.020, samples=150):
            cloud_points.append((float(point[0]), float(point[1]), float(point[2]), pin_color))

        gripper_color = rgb_float(gripper_rgb)
        for point in guide_tube_points(gripper_start, gripper_end, radius=0.024, samples=160):
            cloud_points.append((float(point[0]), float(point[1]), float(point[2]), gripper_color))

        msg = self.ros["point_cloud2"].create_cloud(header, fields, cloud_points)
        self.last_live_cloud_msg = msg
        self.live_cloud_pub.publish(msg)

    def publish_live_alignment_marker(
        self,
        frame_id: str,
        target: dict,
        stage: str,
        status: str,
        *,
        progress: float = 1.0,
        source_pose: dict | None = None,
    ) -> None:
        axis_up = normalize([float(value) for value in target["pin_axis_up"]])
        stage_pose = target[stage]
        grasp_tcp = tcp_position_from_flange_pose(target["grasp"], self.args.flange_to_tcp_z)
        target_tcp = tcp_position_from_flange_pose(stage_pose, self.args.flange_to_tcp_z)
        if source_pose is None:
            stage_tcp = target_tcp
        else:
            source_tcp = tcp_position_from_flange_pose(source_pose, self.args.flange_to_tcp_z)
            stage_tcp = lerp_vec(source_tcp, target_tcp, max(0.0, min(float(progress), 1.0)))
        tool_z = pose_z_axis(stage_pose)
        length = max(float(self.args.alignment_marker_length), 0.02)
        pin_extension_up = 1.20 * length
        pin_extension_down = 0.10 * length
        gripper_extension = 0.70 * length

        aligned = status == "aligned"
        pin_color = (0.0, 0.9, 0.15, 1.0) if aligned else (1.0, 0.45, 0.0, 1.0)
        gripper_color = (0.0, 0.9, 0.15, 1.0) if aligned else (0.0, 0.3, 1.0, 1.0)
        label_color = (0.0, 0.75, 0.12, 1.0) if aligned else (0.0, 0.2, 0.9, 1.0)
        pin_rgb = (0, 230, 40) if aligned else (255, 120, 0)
        gripper_rgb = (0, 230, 40) if aligned else (0, 80, 255)

        pin_start = add_vec(grasp_tcp, axis_up, -pin_extension_down)
        pin_end = add_vec(grasp_tcp, axis_up, pin_extension_up)
        gripper_start = add_vec(stage_tcp, tool_z, -gripper_extension)
        gripper_end = add_vec(stage_tcp, tool_z, gripper_extension)

        pin_axis_marker = self.make_line_marker(
            frame_id,
            "live_current_pin_axis",
            100,
            [pin_start, pin_end],
            pin_color,
            0.018,
        )
        gripper_marker = self.make_line_marker(
            frame_id,
            "live_gripper_centerline",
            101,
            [gripper_start, gripper_end],
            gripper_color,
            0.020,
        )
        pin_arrow = self.make_arrow_marker(
            frame_id,
            "live_current_pin_axis_arrow",
            105,
            grasp_tcp,
            pin_end,
            pin_color,
            shaft_diameter=0.026,
            head_diameter=0.075,
            head_length=0.11,
        )
        gripper_arrow_a = self.make_arrow_marker(
            frame_id,
            "live_gripper_centerline_arrow",
            106,
            stage_tcp,
            gripper_end,
            gripper_color,
            shaft_diameter=0.030,
            head_diameter=0.085,
            head_length=0.12,
        )
        gripper_arrow_b = self.make_arrow_marker(
            frame_id,
            "live_gripper_centerline_arrow",
            107,
            stage_tcp,
            gripper_start,
            gripper_color,
            shaft_diameter=0.030,
            head_diameter=0.085,
            head_length=0.12,
        )
        label = "ALIGNED" if aligned else "ALIGNING"
        label_marker = self.make_text_marker(
            frame_id,
            target,
            add_vec(grasp_tcp, axis_up, 0.95 * length),
            f"Pin {int(target['detection_id'])}: {label}",
            label_color,
        )
        target_point = self.make_sphere_marker(frame_id, 103, grasp_tcp, pin_color, 0.055)
        gripper_point = self.make_sphere_marker(frame_id, 104, stage_tcp, gripper_color, 0.050)
        arr = self.ros["MarkerArray"]()
        arr.markers = [
            pin_axis_marker,
            gripper_marker,
            pin_arrow,
            gripper_arrow_a,
            gripper_arrow_b,
            label_marker,
            target_point,
            gripper_point,
        ]
        self.last_live_marker_array = arr
        self.marker_pub.publish(arr)
        self.legacy_marker_pub.publish(arr)
        self.publish_live_alignment_cloud(
            frame_id,
            pin_start,
            pin_end,
            gripper_start,
            gripper_end,
            pin_rgb,
            gripper_rgb,
        )

    def republish_live_alignment_marker(self) -> None:
        if self.last_live_marker_array is None:
            return
        now = self.node.get_clock().now().to_msg()
        for marker in self.last_live_marker_array.markers:
            marker.header.stamp = now
        self.marker_pub.publish(self.last_live_marker_array)
        self.legacy_marker_pub.publish(self.last_live_marker_array)
        if self.last_live_cloud_msg is not None:
            self.last_live_cloud_msg.header.stamp = now
            self.live_cloud_pub.publish(self.last_live_cloud_msg)

    def destroy(self) -> None:
        self.stop_publishing.set()
        if self.publisher_thread is not None:
            self.publisher_thread.join(timeout=1.0)
        self.node.destroy_node()


def selected_poses(data: dict, max_pins: int) -> list[dict]:
    poses = list(data["poses"])
    poses.sort(key=lambda item: int(item["detection_id"]))
    if max_pins > 0:
        poses = poses[:max_pins]
    return poses


def main() -> int:
    args = build_parser().parse_args()
    data = json.loads(args.targets_json.read_text(encoding="utf-8"))
    frame_id = args.frame_id or data["frame_id"]
    poses = selected_poses(data, args.max_pins)
    if not poses:
        print("No target poses found.")
        return 1

    ros = load_ros()
    ros["rclpy"].init(args=None)
    player = DemoPlayer(args, ros)
    try:
        player.wait_until_ready()
        player.start_continuous_joint_state_publishing()
        if not args.skip_ready:
            print("Moving fake arm from zero pose to visible ready pose...")
            if not player.send_joint_goal(args.ready_joints):
                print("Ready-pose trajectory failed; continuing with current joint state.")
            time.sleep(max(args.settle_seconds, 0.0))
        print(
            f"Loaded {len(poses)} pin target sets from {args.targets_json} "
            f"in frame {frame_id}; IK link is {args.ik_link_name}."
        )
        for pin_index, target in enumerate(poses, start=1):
            detection_id = int(target["detection_id"])
            if args.manual_step:
                input(f"\nPress Enter to solve and move to pin {detection_id} ({pin_index}/{len(poses)})...")
            else:
                print(f"\nMoving to pin {detection_id} ({pin_index}/{len(poses)})...")
                time.sleep(max(args.auto_delay, 0.0))

            for stage in args.stages:
                print(f"  solving IK for {stage}...")
                source_pose = player.last_stage_pose
                moving_status = "aligned" if stage in {"grasp", "lift"} and source_pose is not None else "aligning"
                player.publish_live_alignment_marker(
                    frame_id,
                    target,
                    stage,
                    moving_status,
                    progress=0.0,
                    source_pose=source_pose,
                )
                solution = player.solve_ik(frame_id, target[stage])
                if solution is None:
                    print(f"  IK failed for pin {detection_id} {stage}; skipping remaining stages for this pin.")
                    break
                print("  IK solved; moving fake arm.")
                if not player.send_joint_goal(
                    solution,
                    progress_callback=lambda progress, source_pose=source_pose, target=target, stage=stage: (
                        player.publish_live_alignment_marker(
                            frame_id,
                            target,
                            stage,
                            moving_status,
                            progress=progress,
                            source_pose=source_pose,
                        )
                    ),
                ):
                    print(f"  trajectory failed for pin {detection_id} {stage}; stopping demo.")
                    return 1
                player.publish_live_alignment_marker(
                    frame_id,
                    target,
                    stage,
                    "aligned",
                    progress=1.0,
                    source_pose=source_pose,
                )
                player.last_stage_pose = target[stage]
                time.sleep(max(args.alignment_hold_seconds, 0.0))
                time.sleep(max(args.settle_seconds, 0.0))

        print("\nDemo sequence complete.")
        if args.hold_open:
            print("Holding final joint state for RViz inspection. Press Ctrl-C to stop.")
            try:
                while True:
                    player.republish_live_alignment_marker()
                    player.rclpy.spin_once(player.node, timeout_sec=0.2)
            except KeyboardInterrupt:
                print("\nStopping hold-open demo.")
        else:
            print("RViz/MoveIt can stay open for inspection.")
        return 0
    finally:
        player.destroy()
        if ros["rclpy"].ok():
            ros["rclpy"].shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
