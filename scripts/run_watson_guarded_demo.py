#!/usr/bin/env python3
"""Read, plan, and explicitly arm a tiny supervised motion on Watson.

This is intentionally separate from the fake pin-demo player. It never
publishes joint states and never sends a goal directly to the Techman
FollowJointTrajectory server. The default mode is read-only.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ARENA_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ARENA_DIR))

from pin_axis_3d_sim.watson_guard import (  # noqa: E402
    FIRST_MOTION_PROFILE,
    HealthSnapshot,
    J6_GUARD_PROFILES,
    J6_QUALIFICATION_PROFILE,
    J6_SHOWCASE_PROFILE,
    JOINT_NAMES,
    TrajectorySample,
    get_j6_guard_profile,
    health_failures,
    j6_profile_targets,
    motion_envelope_failures,
    validate_trajectory_samples,
    wrist_check_targets,
)
from pin_axis_3d_sim.controller_tool_state import (  # noqa: E402
    query_controller_tool_items,
)


RUNNER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
GUARD_SOURCE_SHA256 = hashlib.sha256(
    (ARENA_DIR / "pin_axis_3d_sim" / "watson_guard.py").read_bytes()
).hexdigest()
ARM_TOKEN = "MOVE_WATSON_SLOWLY"
J6_QUALIFICATION_ARM_TOKEN = "MOVE_WATSON_J6_QUALIFICATION"
J6_SHOWCASE_ARM_TOKEN = "MOVE_WATSON_J6_SHOWCASE"
MOVEIT_SUCCESS = 1
ACTION_STATUS_SUCCEEDED = 4
REPORT_SCHEMA_VERSION = 8
MAX_EXECUTE_AMPLITUDE_DEG = 0.9
J6_PLANNING_GOAL_TOLERANCE_RAD = 0.0001
MAX_PLANNED_GOAL_ERROR_RAD = 0.0002
ROBOT_IP = os.environ.get("TECHMAN_ROBOT_IP", "192.0.2.23")
ROBOT_INTERFACE = os.environ.get("TECHMAN_ROBOT_INTERFACE", "enp1s0")
ROBOT_SOURCE_IP = os.environ.get("TECHMAN_ROBOT_SOURCE_IP", "192.0.2.100")
# A locally administered placeholder deliberately fails the real neighbour
# check unless the site-specific value is supplied outside Git.
ROBOT_MAC = os.environ.get("TECHMAN_ROBOT_MAC", "02:00:00:00:00:23").lower()
LIVE_POSE_TOLERANCE_RAD = 0.003
GOAL_ACCEPTANCE_TIMEOUT_S = 10.0
EXECUTE_LOCK_PATH = Path(
    f"/tmp/watson-tm5s-{re.sub(r'[^A-Za-z0-9_.-]', '_', ROBOT_IP)}.execute.lock"
)
MAX_QUALIFICATION_REPORT_BYTES = 8 * 1024 * 1024
MAX_QUALIFICATION_REPORT_AGE_S = 2 * 60 * 60
QUALIFICATION_REPORT_NAME_RE = re.compile(r"^\d{8}T\d{6}Z_execute\.json$")
REPORT_DIGEST_FIELD = "report_payload_sha256"


class StopUnverifiedError(RuntimeError):
    """A motion command may still be active; the operator must use the E-stop."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["check", "plan", "execute"],
        default="check",
        help="check is read-only; plan creates no command; execute requires explicit arming",
    )
    parser.add_argument(
        "--motion-profile",
        choices=list(J6_GUARD_PROFILES),
        default=FIRST_MOTION_PROFILE,
        help="immutable J6 motion envelope; new profiles accept no custom motion limits",
    )
    parser.add_argument("--namespace", default="/watson")
    parser.add_argument("--group-name", default="tmr_arm")
    parser.add_argument("--planning-frame", default="base")
    parser.add_argument(
        "--amplitude-deg",
        type=float,
        default=0.9,
        help="requested J6 displacement; 0.9 degrees leaves margin under the hard 1-degree cap",
    )
    parser.add_argument("--velocity-scaling", type=float, default=0.01)
    parser.add_argument("--acceleration-scaling", type=float, default=0.01)
    parser.add_argument("--max-project-speed", type=int, default=5)
    parser.add_argument("--state-timeout", type=float, default=20.0)
    parser.add_argument("--service-timeout", type=float, default=20.0)
    parser.add_argument("--execution-timeout", type=float, default=90.0)
    parser.add_argument("--arm-token", default="")
    parser.add_argument(
        "--confirm-cell-clear",
        action="store_true",
        help="Required with --mode execute after checking the physical cell and E-stop access",
    )
    parser.add_argument(
        "--qualification-report",
        type=Path,
        default=None,
        help=(
            "absolute successful j6_qualification execute report; required only "
            "for j6_showcase execution"
        ),
    )
    parser.add_argument("--report", type=Path, default=None)
    return parser


def load_ros() -> dict[str, object]:
    try:
        import rclpy
        from action_msgs.msg import GoalStatus, GoalStatusArray
        from moveit_msgs.action import ExecuteTrajectory
        from moveit_msgs.msg import Constraints, DisplayTrajectory, JointConstraint, RobotState
        from moveit_msgs.srv import GetMotionPlan
        from rclpy.action import ActionClient
        from rclpy.action.graph import (
            get_action_client_names_and_types_by_node,
            get_action_server_names_and_types_by_node,
        )
        from rclpy.qos import qos_profile_action_status_default
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import JointState
        from tm_msgs.msg import FeedbackState
        from tm_msgs.srv import AskItem
    except Exception as exc:
        raise RuntimeError(
            "ROS 2 Jazzy and the Techman workspace must be sourced before running this script"
        ) from exc
    return {
        "rclpy": rclpy,
        "GoalStatus": GoalStatus,
        "GoalStatusArray": GoalStatusArray,
        "ExecuteTrajectory": ExecuteTrajectory,
        "Constraints": Constraints,
        "DisplayTrajectory": DisplayTrajectory,
        "JointConstraint": JointConstraint,
        "RobotState": RobotState,
        "GetMotionPlan": GetMotionPlan,
        "ActionClient": ActionClient,
        "get_action_server_names_and_types_by_node": (
            get_action_server_names_and_types_by_node
        ),
        "get_action_client_names_and_types_by_node": (
            get_action_client_names_and_types_by_node
        ),
        "qos_profile_action_status_default": qos_profile_action_status_default,
        "SignalHandlerOptions": SignalHandlerOptions,
        "JointState": JointState,
        "FeedbackState": FeedbackState,
        "AskItem": AskItem,
    }


def duration_seconds(duration) -> float:
    return float(duration.sec) + float(duration.nanosec) / 1_000_000_000.0


def report_payload_sha256(report: dict) -> str:
    payload = dict(report)
    payload.pop(REPORT_DIGEST_FIELD, None)
    canonical = json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def trajectory_report(trajectory) -> dict:
    joint_trajectory = trajectory.joint_trajectory
    multi_dof = trajectory.multi_dof_joint_trajectory
    payload = {
        "joint_names": list(joint_trajectory.joint_names),
        "multi_dof_joint_names": list(multi_dof.joint_names),
        "multi_dof_point_count": len(multi_dof.points),
        "points": [
            {
                "time_s": duration_seconds(point.time_from_start),
                "positions_rad": [float(value) for value in point.positions],
                "velocities_rad_s": [float(value) for value in point.velocities],
                "accelerations_rad_s2": [float(value) for value in point.accelerations],
            }
            for point in joint_trajectory.points
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def extract_trajectory_samples(trajectory) -> tuple[TrajectorySample, ...]:
    """Extract the exact controller-facing joint payload and reject extras."""

    multi_dof = trajectory.multi_dof_joint_trajectory
    if multi_dof.joint_names or multi_dof.points:
        raise RuntimeError(
            "RobotTrajectory contains a multi-DOF component; refusing execution"
        )
    joint_trajectory = trajectory.joint_trajectory
    names = list(joint_trajectory.joint_names)
    if names != list(JOINT_NAMES):
        raise RuntimeError(
            "trajectory joint order is not the exact Techman controller order: "
            f"{names}"
        )

    samples: list[TrajectorySample] = []
    for index, point in enumerate(joint_trajectory.points):
        if len(point.positions) != len(JOINT_NAMES):
            raise RuntimeError(
                f"trajectory point {index} does not contain six positions"
            )
        if point.velocities and len(point.velocities) != len(JOINT_NAMES):
            raise RuntimeError(
                f"trajectory point {index} does not contain six velocities"
            )
        if point.accelerations and len(point.accelerations) != len(JOINT_NAMES):
            raise RuntimeError(
                f"trajectory point {index} does not contain six accelerations"
            )
        if getattr(point, "effort", ()):
            raise RuntimeError(
                f"trajectory point {index} contains unexpected effort commands"
            )
        samples.append(
            TrajectorySample(
                positions=tuple(float(value) for value in point.positions),
                velocities=tuple(float(value) for value in point.velocities),
                accelerations=tuple(float(value) for value in point.accelerations),
                time_s=duration_seconds(point.time_from_start),
            )
        )
    return tuple(samples)


class WatsonGuardNode:
    def __init__(self, args: argparse.Namespace, ros: dict[str, object]):
        self.args = args
        self.ros = ros
        self.rclpy = ros["rclpy"]
        self.node = self.rclpy.create_node("watson_guarded_demo")
        namespace = "/" + args.namespace.strip("/")
        self.namespace = namespace
        self.feedback = None
        self.feedback_received_at = 0.0
        self.joint_positions: tuple[float, ...] | None = None
        self.joint_velocities: tuple[float, ...] | None = None
        self.joint_state_received_at = 0.0
        self.stop_requested = False
        self.stop_signal: int | None = None
        self.active_goal_handle = None
        self.active_result_future = None
        self.motion_command_sent = False
        self.execute_action_status = None
        self.controller_action_status = None
        self.node.create_subscription(
            ros["FeedbackState"],
            f"{namespace}/feedback_states",
            self._feedback_callback,
            20,
        )
        self.node.create_subscription(
            ros["JointState"],
            f"{namespace}/joint_states",
            self._joint_state_callback,
            20,
        )
        self.node.create_subscription(
            ros["GoalStatusArray"],
            f"{namespace}/execute_trajectory/_action/status",
            self._execute_status_callback,
            ros["qos_profile_action_status_default"],
        )
        self.node.create_subscription(
            ros["GoalStatusArray"],
            f"{namespace}/tmr_arm_controller/follow_joint_trajectory/_action/status",
            self._controller_status_callback,
            ros["qos_profile_action_status_default"],
        )
        self.plan_client = self.node.create_client(
            ros["GetMotionPlan"],
            f"{namespace}/plan_kinematic_path",
        )
        self.execute_client = ros["ActionClient"](
            self.node,
            ros["ExecuteTrajectory"],
            f"{namespace}/execute_trajectory",
        )
        self.display_pub = self.node.create_publisher(
            ros["DisplayTrajectory"],
            f"{namespace}/display_planned_path",
            10,
        )
        self.tool_settings_client = self.node.create_client(
            ros["AskItem"],
            f"{namespace}/ask_item",
        )

    def _feedback_callback(self, msg) -> None:
        self.feedback = msg
        self.feedback_received_at = time.monotonic()

    def _joint_state_callback(self, msg) -> None:
        positions = dict(zip(msg.name, msg.position))
        velocities = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
        if not all(name in positions for name in JOINT_NAMES):
            return
        self.joint_positions = tuple(float(positions[name]) for name in JOINT_NAMES)
        if all(name in velocities for name in JOINT_NAMES):
            self.joint_velocities = tuple(float(velocities[name]) for name in JOINT_NAMES)
        else:
            self.joint_velocities = ()
        self.joint_state_received_at = time.monotonic()

    def _execute_status_callback(self, msg) -> None:
        self.execute_action_status = msg

    def _controller_status_callback(self, msg) -> None:
        self.controller_action_status = msg

    def spin_until_state(self, timeout_s: float) -> HealthSnapshot:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.feedback is not None and self.joint_positions is not None:
                return self.snapshot()
        raise RuntimeError("timed out waiting for Watson feedback and joint states")

    def read_controller_tool_settings(self, timeout_s: float) -> dict[str, object]:
        """Read the active TMflow tool/base records without calling a write service."""

        return query_controller_tool_items(
            node=self.node,
            rclpy=self.rclpy,
            ask_item_type=self.ros["AskItem"],
            client=self.tool_settings_client,
            timeout_s=timeout_s,
        )

    def snapshot(self) -> HealthSnapshot:
        if self.feedback is None or self.joint_positions is None:
            raise RuntimeError("Watson state is incomplete")
        now = time.monotonic()
        feedback_velocities = tuple(float(value) for value in self.feedback.joint_vel)
        velocities = (
            feedback_velocities
            if len(feedback_velocities) == len(JOINT_NAMES)
            else self.joint_velocities or ()
        )
        return HealthSnapshot(
            is_svr_connected=bool(self.feedback.is_svr_connected),
            is_sct_connected=bool(self.feedback.is_sct_connected),
            tmsrv_cperr=int(self.feedback.tmsrv_cperr),
            tmscript_cperr=int(self.feedback.tmscript_cperr),
            tmsrv_dataerr=int(self.feedback.tmsrv_dataerr),
            tmscript_dataerr=int(self.feedback.tmscript_dataerr),
            is_data_table_correct=bool(self.feedback.is_data_table_correct),
            robot_link=bool(self.feedback.robot_link),
            robot_error=bool(self.feedback.robot_error),
            project_run=bool(self.feedback.project_run),
            project_pause=bool(self.feedback.project_pause),
            safetyguard_a=bool(self.feedback.safetyguard_a),
            e_stop=bool(self.feedback.e_stop),
            error_code=int(self.feedback.error_code),
            project_speed=int(self.feedback.project_speed),
            ma_mode=int(self.feedback.ma_mode),
            robot_light=int(self.feedback.robot_light),
            joint_positions=self.joint_positions,
            feedback_joint_positions=tuple(float(value) for value in self.feedback.joint_pos),
            joint_velocities=velocities,
            feedback_age_s=now - self.feedback_received_at,
            joint_state_age_s=now - self.joint_state_received_at,
        )

    def publisher_failures(self) -> list[str]:
        failures: list[str] = []
        for suffix in ("joint_states", "feedback_states"):
            topic = f"{self.namespace}/{suffix}"
            publishers = self.node.get_publishers_info_by_topic(topic)
            if len(publishers) != 1:
                failures.append(
                    f"expected exactly one publisher on {topic}, found {len(publishers)}"
                )
                continue
            publisher = publishers[0]
            if (
                publisher.node_name != "tm_driver_node"
                or publisher.node_namespace != self.namespace
            ):
                failures.append(
                    f"unexpected {topic} publisher "
                    f"{publisher.node_namespace}/{publisher.node_name}"
                )
        return failures

    def wait_for_expected_publishers(self, timeout_s: float = 3.0) -> None:
        """Allow DDS graph metadata to arrive, then require exact driver provenance."""

        deadline = time.monotonic() + timeout_s
        failures = ["publisher graph has not been inspected"]
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            failures = self.publisher_failures()
            if not failures:
                return
        raise RuntimeError("Watson publisher provenance failed: " + "; ".join(failures))

    def command_endpoint_failures(self, *, require_execute: bool) -> list[str]:
        """Require planning and execution endpoints to have exact local owners."""

        failures: list[str] = []
        service_owners: list[tuple[str, str, list[str]]] = []
        action_owners: list[tuple[str, str, list[str]]] = []
        action_clients: list[tuple[str, str, str, list[str]]] = []
        graph_errors: list[str] = []
        for node_name, node_namespace in self.node.get_node_names_and_namespaces():
            try:
                services = self.node.get_service_names_and_types_by_node(
                    node_name,
                    node_namespace,
                )
                for endpoint_name, endpoint_types in services:
                    if endpoint_name == f"{self.namespace}/plan_kinematic_path":
                        service_owners.append(
                            (node_name, node_namespace, list(endpoint_types))
                        )
            except Exception as exc:
                graph_errors.append(
                    f"could not inspect services for {node_namespace}/{node_name}: {exc}"
                )
            try:
                actions = self.ros[
                    "get_action_server_names_and_types_by_node"
                ](self.node, node_name, node_namespace)
                for endpoint_name, endpoint_types in actions:
                    if endpoint_name in {
                        f"{self.namespace}/execute_trajectory",
                        f"{self.namespace}/tmr_arm_controller/follow_joint_trajectory",
                    }:
                        action_owners.append(
                            (node_name, node_namespace, list(endpoint_types))
                        )
            except Exception as exc:
                graph_errors.append(
                    f"could not inspect actions for {node_namespace}/{node_name}: {exc}"
                )
            try:
                clients = self.ros[
                    "get_action_client_names_and_types_by_node"
                ](self.node, node_name, node_namespace)
                for endpoint_name, endpoint_types in clients:
                    if endpoint_name in {
                        f"{self.namespace}/execute_trajectory",
                        f"{self.namespace}/tmr_arm_controller/follow_joint_trajectory",
                        f"{self.namespace}/move_action",
                        f"{self.namespace}/sequence_move_group",
                    }:
                        action_clients.append(
                            (
                                endpoint_name,
                                node_name,
                                node_namespace,
                                list(endpoint_types),
                            )
                        )
            except Exception as exc:
                graph_errors.append(
                    f"could not inspect action clients for "
                    f"{node_namespace}/{node_name}: {exc}"
                )

        expected_service = [
            ("move_group", self.namespace, ["moveit_msgs/srv/GetMotionPlan"])
        ]
        if service_owners != expected_service:
            failures.append(
                f"unexpected {self.namespace}/plan_kinematic_path owners: {service_owners}"
            )
        if require_execute:
            expected_actions = sorted(
                [
                    (
                        "move_group",
                        self.namespace,
                        ["moveit_msgs/action/ExecuteTrajectory"],
                    ),
                    (
                        "tm_driver_node",
                        self.namespace,
                        ["control_msgs/action/FollowJointTrajectory"],
                    ),
                ]
            )
            if sorted(action_owners) != expected_actions:
                failures.append(
                    "unexpected Watson execution action owners: "
                    f"{sorted(action_owners)}"
                )
            expected_clients = sorted(
                [
                    (
                        f"{self.namespace}/execute_trajectory",
                        "watson_guarded_demo",
                        "/",
                        ["moveit_msgs/action/ExecuteTrajectory"],
                    ),
                    (
                        f"{self.namespace}/tmr_arm_controller/follow_joint_trajectory",
                        "moveit_simple_controller_manager",
                        self.namespace,
                        ["control_msgs/action/FollowJointTrajectory"],
                    ),
                ]
            )
            if sorted(action_clients) != expected_clients:
                failures.append(
                    "unexpected Watson execution action clients: "
                    f"{sorted(action_clients)}"
                )
        if graph_errors:
            failures.extend(graph_errors)
        return failures

    def wait_for_command_endpoints(
        self,
        *,
        require_execute: bool,
        timeout_s: float = 3.0,
    ) -> None:
        """Wait briefly for exact planning/action server provenance to be discoverable."""

        deadline = time.monotonic() + timeout_s
        failures = ["command endpoint graph has not been inspected"]
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            failures = self.command_endpoint_failures(require_execute=require_execute)
            if not failures:
                return
        raise RuntimeError("Watson command endpoint provenance failed: " + "; ".join(failures))

    def action_busy_failures(self) -> list[str]:
        """Reject any retained nonterminal MoveIt or controller action status."""

        terminal_statuses = {
            self.ros["GoalStatus"].STATUS_SUCCEEDED,
            self.ros["GoalStatus"].STATUS_CANCELED,
            self.ros["GoalStatus"].STATUS_ABORTED,
        }
        failures: list[str] = []
        for label, status_array in (
            ("MoveIt execute action", self.execute_action_status),
            ("Techman controller action", self.controller_action_status),
        ):
            # Jazzy action status is event-driven and transient-local. After the
            # settle spin below, None means this fresh server has no goal history.
            if status_array is None:
                continue
            nonterminal = [
                status.status
                for status in status_array.status_list
                if status.status not in terminal_statuses
            ]
            if nonterminal:
                failures.append(
                    f"{label} has nonterminal or unknown goal status values {nonterminal}"
                )
        return failures

    def settle_action_status_callbacks(self, duration_s: float = 0.25) -> None:
        """Allow transient-local action status history to reach this node."""

        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)

    def request_stop(self, signum: int, _frame) -> None:
        self.stop_requested = True
        self.stop_signal = signum

    def require_healthy(
        self,
        *,
        stationary: bool,
        require_auto_mode: bool,
        honor_stop_request: bool = True,
    ) -> HealthSnapshot:
        # Observe a full second and bound drift, rather than trusting one low-speed sample.
        deadline = time.monotonic() + 1.0
        last_snapshot = None
        first_feedback_positions = None
        while time.monotonic() < deadline:
            if honor_stop_request and self.stop_requested:
                raise RuntimeError(f"stop requested by signal {self.stop_signal}")
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            last_snapshot = self.snapshot()
            if first_feedback_positions is None:
                first_feedback_positions = last_snapshot.feedback_joint_positions
            failures = health_failures(
                last_snapshot,
                max_project_speed=self.args.max_project_speed,
                require_stationary=stationary,
                require_auto_mode=require_auto_mode,
            )
            failures.extend(self.publisher_failures())
            if stationary and first_feedback_positions is not None:
                position_drift = max(
                    abs(
                        last_snapshot.feedback_joint_positions[index]
                        - first_feedback_positions[index]
                    )
                    for index in range(len(JOINT_NAMES))
                )
                if position_drift > 0.001:
                    failures.append(
                        "robot position drifted during stationary proof "
                        f"({position_drift:.6f}rad > 0.001000rad)"
                    )
            if failures:
                raise RuntimeError("Watson health gate failed: " + "; ".join(failures))
        if last_snapshot is None:
            raise RuntimeError("no Watson state observed during health gate")
        return last_snapshot

    def plan_stage(
        self,
        *,
        stage_name: str,
        start_positions: tuple[float, ...],
        goal_positions: tuple[float, ...],
        hard_reference_start: tuple[float, ...],
    ) -> tuple[
        object,
        dict[str, float | int],
        tuple[TrajectorySample, ...],
    ]:
        endpoint_failures = self.command_endpoint_failures(require_execute=False)
        if endpoint_failures:
            raise RuntimeError(
                "Watson planning endpoint provenance failed: "
                + "; ".join(endpoint_failures)
            )
        if not self.plan_client.wait_for_service(timeout_sec=self.args.service_timeout):
            raise RuntimeError(f"{self.namespace}/plan_kinematic_path is unavailable")

        request = self.ros["GetMotionPlan"].Request()
        plan = request.motion_plan_request
        plan.workspace_parameters.header.frame_id = self.args.planning_frame
        plan.workspace_parameters.min_corner.x = -1.2
        plan.workspace_parameters.min_corner.y = -1.2
        plan.workspace_parameters.min_corner.z = -0.1
        plan.workspace_parameters.max_corner.x = 1.2
        plan.workspace_parameters.max_corner.y = 1.2
        plan.workspace_parameters.max_corner.z = 1.5
        plan.start_state = self.ros["RobotState"]()
        plan.start_state.joint_state.header.stamp = self.node.get_clock().now().to_msg()
        plan.start_state.joint_state.name = list(JOINT_NAMES)
        plan.start_state.joint_state.position = list(start_positions)
        plan.start_state.joint_state.velocity = [0.0] * len(JOINT_NAMES)
        plan.start_state.is_diff = False

        constraints = self.ros["Constraints"]()
        constraints.name = stage_name
        for joint_name, position in zip(JOINT_NAMES, goal_positions):
            joint_constraint = self.ros["JointConstraint"]()
            joint_constraint.joint_name = joint_name
            joint_constraint.position = float(position)
            tolerance = (
                J6_PLANNING_GOAL_TOLERANCE_RAD
                if joint_name == "joint_6"
                else 0.000001
            )
            joint_constraint.tolerance_above = tolerance
            joint_constraint.tolerance_below = tolerance
            joint_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_constraint)
        plan.goal_constraints = [constraints]
        plan.pipeline_id = "ompl"
        plan.group_name = self.args.group_name
        plan.num_planning_attempts = 5
        plan.allowed_planning_time = 5.0
        plan.max_velocity_scaling_factor = self.args.velocity_scaling
        plan.max_acceleration_scaling_factor = self.args.acceleration_scaling

        future = self.plan_client.call_async(request)
        self.rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=self.args.service_timeout,
        )
        if not future.done() or future.result() is None:
            raise RuntimeError(f"planning timed out for {stage_name}")
        response = future.result().motion_plan_response
        if response.error_code.val != MOVEIT_SUCCESS:
            raise RuntimeError(
                f"MoveIt planning failed for {stage_name}: {response.error_code.val} "
                f"{response.error_code.message}"
            )

        trajectory = response.trajectory
        planned_samples = extract_trajectory_samples(trajectory)
        guard_profile = get_j6_guard_profile(
            getattr(self.args, "motion_profile", FIRST_MOTION_PROFILE)
        )
        metrics = validate_trajectory_samples(
            planned_samples,
            expected_start=start_positions,
            expected_goal=goal_positions,
            hard_reference_start=hard_reference_start,
            max_goal_error_rad=MAX_PLANNED_GOAL_ERROR_RAD,
            max_excursion_rad=guard_profile.hard_excursion_rad,
            max_sample_step_rad=guard_profile.max_sample_step_rad,
            max_velocity_rad_s=guard_profile.max_planned_velocity_rad_s,
            max_acceleration_rad_s2=(
                guard_profile.max_planned_acceleration_rad_s2
            ),
            min_total_duration_s=guard_profile.min_duration_s,
            max_total_duration_s=guard_profile.max_duration_s,
            guard_profile=guard_profile.name,
        )

        display = self.ros["DisplayTrajectory"]()
        display.model_id = "tm5s"
        display.trajectory_start = response.trajectory_start
        display.trajectory = [trajectory]
        self.display_pub.publish(display)
        return trajectory, metrics, planned_samples

    def execute_stage(
        self,
        *,
        stage_name: str,
        trajectory,
        planned_samples: tuple[TrajectorySample, ...],
        expected_start,
        expected_goal,
        hard_reference_start,
    ) -> dict:
        guard_profile = get_j6_guard_profile(
            getattr(self.args, "motion_profile", FIRST_MOTION_PROFILE)
        )
        if not self.execute_client.wait_for_server(timeout_sec=self.args.service_timeout):
            raise RuntimeError(f"{self.namespace}/execute_trajectory is unavailable")
        endpoint_failures = self.command_endpoint_failures(require_execute=True)
        if endpoint_failures:
            raise RuntimeError(
                "Watson execution endpoint provenance failed: "
                + "; ".join(endpoint_failures)
            )
        self.settle_action_status_callbacks()
        busy_failures = self.action_busy_failures()
        if busy_failures:
            raise RuntimeError("Watson action server is not idle: " + "; ".join(busy_failures))
        stationary_snapshot = self.require_healthy(
            stationary=True,
            require_auto_mode=True,
        )
        validate_execute_network()
        endpoint_failures = self.command_endpoint_failures(require_execute=True)
        busy_failures = self.action_busy_failures()
        if endpoint_failures or busy_failures:
            raise RuntimeError(
                f"final command-path gate failed before {stage_name}: "
                + "; ".join(endpoint_failures + busy_failures)
            )

        # Snapshot again after the final route/action graph checks. It follows
        # the full stationary proof immediately, and it is the exact q/v pair
        # used to prove the controller's omitted-zero-point first PVT cubic.
        snapshot = self.snapshot()
        pre_send_health_failures = health_failures(
            snapshot,
            max_project_speed=self.args.max_project_speed,
            require_stationary=True,
            require_auto_mode=True,
        )
        pre_send_health_failures.extend(self.publisher_failures())
        stationary_drift = max(
            abs(
                snapshot.feedback_joint_positions[index]
                - stationary_snapshot.feedback_joint_positions[index]
            )
            for index in range(len(JOINT_NAMES))
        )
        if stationary_drift > 0.001:
            pre_send_health_failures.append(
                "robot moved after the stationary proof "
                f"({stationary_drift:.6f}rad > 0.001000rad)"
            )
        if pre_send_health_failures:
            raise RuntimeError(
                f"final health gate failed before {stage_name}: "
                + "; ".join(pre_send_health_failures)
            )
        start_error = max(
            abs(snapshot.feedback_joint_positions[index] - expected_start[index])
            for index in range(len(JOINT_NAMES))
        )
        if start_error > LIVE_POSE_TOLERANCE_RAD:
            raise RuntimeError(
                f"live start mismatch before {stage_name}: "
                f"{start_error:.6f}rad > {LIVE_POSE_TOLERANCE_RAD:.6f}rad"
            )
        physical_start_positions = snapshot.feedback_joint_positions
        physical_start_velocities = snapshot.joint_velocities
        outgoing_samples = extract_trajectory_samples(trajectory)
        if outgoing_samples != planned_samples:
            raise RuntimeError(
                f"controller-facing trajectory payload changed after planning {stage_name}"
            )
        execution_validation = validate_trajectory_samples(
            outgoing_samples,
            expected_start=expected_start,
            expected_goal=expected_goal,
            hard_reference_start=hard_reference_start,
            hard_travel_start=physical_start_positions,
            execution_start_positions=physical_start_positions,
            execution_start_velocities=physical_start_velocities,
            max_goal_error_rad=MAX_PLANNED_GOAL_ERROR_RAD,
            max_excursion_rad=guard_profile.hard_excursion_rad,
            max_sample_step_rad=guard_profile.max_sample_step_rad,
            max_velocity_rad_s=guard_profile.max_planned_velocity_rad_s,
            max_acceleration_rad_s2=(
                guard_profile.max_planned_acceleration_rad_s2
            ),
            min_total_duration_s=guard_profile.min_duration_s,
            max_total_duration_s=guard_profile.max_duration_s,
            guard_profile=guard_profile.name,
        )
        pre_send_failures = motion_envelope_failures(
            snapshot,
            expected_start=expected_start,
            expected_goal=expected_goal,
            hard_reference_start=hard_reference_start,
            hard_travel_start=physical_start_positions,
            max_live_velocity_rad_s=guard_profile.max_live_velocity_rad_s,
            guard_profile=guard_profile.name,
        )
        if pre_send_failures:
            raise RuntimeError(
                f"live motion envelope failed before {stage_name}: "
                + "; ".join(pre_send_failures)
            )
        require_fresh_showcase_gate_before_send(self.args, stage_name)

        goal = self.ros["ExecuteTrajectory"].Goal()
        goal.trajectory = trajectory
        goal.controller_names = ["tmr_arm_controller"]
        send_future = None
        goal_handle = None
        result_future = None
        send_attempted = False
        try:
            blocked_signals = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
            previous_signal_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK,
                blocked_signals,
            )
            try:
                if self.stop_requested:
                    raise RuntimeError(f"stop requested by signal {self.stop_signal}")
                send_attempted = True
                self.motion_command_sent = True
                send_future = self.execute_client.send_goal_async(goal)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_signal_mask)
            send_deadline = time.monotonic() + GOAL_ACCEPTANCE_TIMEOUT_S
            pre_accept_failures: list[str] = []
            if self.stop_requested:
                pre_accept_failures.append(
                    f"stop requested by signal {self.stop_signal}"
                )
            pre_accept_emergency_announced = False
            while not send_future.done() and time.monotonic() < send_deadline:
                self.rclpy.spin_once(self.node, timeout_sec=0.05)
                if self.stop_requested:
                    pre_accept_failures.append(f"stop requested by signal {self.stop_signal}")
                live_snapshot = self.snapshot()
                pre_accept_failures.extend(
                    health_failures(
                        live_snapshot,
                        max_project_speed=self.args.max_project_speed,
                        require_stationary=False,
                        require_auto_mode=True,
                    )
                )
                pre_accept_failures.extend(self.publisher_failures())
                pre_accept_failures.extend(
                    motion_envelope_failures(
                        live_snapshot,
                        expected_start=expected_start,
                        expected_goal=expected_goal,
                        hard_reference_start=hard_reference_start,
                        hard_travel_start=physical_start_positions,
                        max_live_velocity_rad_s=(
                            guard_profile.max_live_velocity_rad_s
                        ),
                        guard_profile=guard_profile.name,
                    )
                )
                if pre_accept_failures and not pre_accept_emergency_announced:
                    print(
                        "EMERGENCY: Watson health changed while goal acceptance is "
                        "unknown; use the physical E-stop immediately. "
                        + "; ".join(sorted(set(pre_accept_failures))),
                        file=sys.stderr,
                        flush=True,
                    )
                    pre_accept_emergency_announced = True
            if not send_future.done():
                raise StopUnverifiedError(
                    f"execution goal acceptance is unknown for {stage_name}; "
                    "use the physical E-stop"
                )
            goal_handle = send_future.result()
            if goal_handle is None:
                raise StopUnverifiedError(
                    f"execution goal acceptance returned no handle for {stage_name}; "
                    "use the physical E-stop"
                )
            if not goal_handle.accepted:
                raise RuntimeError(f"MoveIt rejected execution stage {stage_name}")

            result_future = goal_handle.get_result_async()
            self.active_goal_handle = goal_handle
            self.active_result_future = result_future
            if pre_accept_failures:
                raise RuntimeError(
                    f"health gate changed while accepting {stage_name}: "
                    + "; ".join(sorted(set(pre_accept_failures)))
                )
            deadline = time.monotonic() + self.args.execution_timeout
            while not result_future.done() and time.monotonic() < deadline:
                self.rclpy.spin_once(self.node, timeout_sec=0.05)
                live_snapshot = self.snapshot()
                live_failures = health_failures(
                    live_snapshot,
                    max_project_speed=self.args.max_project_speed,
                    require_stationary=False,
                    require_auto_mode=True,
                )
                live_failures.extend(self.publisher_failures())
                live_failures.extend(
                    motion_envelope_failures(
                        live_snapshot,
                        expected_start=expected_start,
                        expected_goal=expected_goal,
                        hard_reference_start=hard_reference_start,
                        hard_travel_start=physical_start_positions,
                        max_live_velocity_rad_s=(
                            guard_profile.max_live_velocity_rad_s
                        ),
                        guard_profile=guard_profile.name,
                    )
                )
                if self.stop_requested:
                    live_failures.append(f"stop requested by signal {self.stop_signal}")
                if live_failures:
                    raise RuntimeError(
                        f"health gate changed during {stage_name}: " + "; ".join(live_failures)
                    )
            if not result_future.done():
                raise RuntimeError(f"execution timed out for {stage_name}")
            wrapped_result = result_future.result()
            if (
                wrapped_result is None
                or wrapped_result.status != self.ros["GoalStatus"].STATUS_SUCCEEDED
                or wrapped_result.result.error_code.val != MOVEIT_SUCCESS
            ):
                code = None if wrapped_result is None else wrapped_result.result.error_code.val
                status = None if wrapped_result is None else wrapped_result.status
                raise RuntimeError(
                    f"MoveIt execution failed for {stage_name}: status={status}, "
                    f"error_code={code}"
                )

            stationary_failure = self.verify_stationary_after_motion()
            if stationary_failure is not None:
                raise StopUnverifiedError(
                    f"MoveIt reported success for {stage_name}, but {stationary_failure}; "
                    "use the physical E-stop"
                )
            final_snapshot = self.snapshot()
            final_health_failures = health_failures(
                final_snapshot,
                max_project_speed=self.args.max_project_speed,
                require_stationary=True,
                require_auto_mode=True,
            )
            final_health_failures.extend(self.publisher_failures())
            final_health_failures.extend(
                motion_envelope_failures(
                    final_snapshot,
                    expected_start=expected_start,
                    expected_goal=expected_goal,
                    hard_reference_start=hard_reference_start,
                    hard_travel_start=physical_start_positions,
                    max_live_velocity_rad_s=guard_profile.max_live_velocity_rad_s,
                    guard_profile=guard_profile.name,
                )
            )
            if final_health_failures:
                raise RuntimeError(
                    f"post-motion health gate failed after {stage_name}, with stop verified: "
                    + "; ".join(final_health_failures)
                )
            goal_error = max(
                abs(final_snapshot.feedback_joint_positions[index] - expected_goal[index])
                for index in range(len(JOINT_NAMES))
            )
            if goal_error > LIVE_POSE_TOLERANCE_RAD:
                raise RuntimeError(
                    f"live goal mismatch after {stage_name}: {goal_error:.6f}rad > "
                    f"{LIVE_POSE_TOLERANCE_RAD:.6f}rad; sustained stop was verified"
                )
            return {
                "stage": stage_name,
                "action_status": wrapped_result.status,
                "moveit_error_code": wrapped_result.result.error_code.val,
                "live_start_error_rad": start_error,
                "live_goal_error_rad": goal_error,
                "physical_start_positions_rad": list(physical_start_positions),
                "physical_start_velocities_rad_s": list(physical_start_velocities),
                "physical_start_feedback_age_s": snapshot.feedback_age_s,
                "physical_start_joint_state_age_s": snapshot.joint_state_age_s,
                "stationary_to_physical_start_drift_rad": stationary_drift,
                "post_motion_stationary_verified": True,
                "final_joint_positions_rad": list(final_snapshot.joint_positions),
                "final_feedback_positions_rad": list(
                    final_snapshot.feedback_joint_positions
                ),
                "final_joint_velocities_rad_s": list(
                    final_snapshot.joint_velocities
                ),
                "final_feedback_age_s": final_snapshot.feedback_age_s,
                "final_joint_state_age_s": final_snapshot.joint_state_age_s,
                "execution_revalidation_metrics": execution_validation,
            }
        except BaseException as exc:
            if not send_attempted:
                raise
            if goal_handle is None and send_future is not None and send_future.done():
                try:
                    goal_handle = send_future.result()
                except BaseException:
                    goal_handle = None
            if goal_handle is not None and goal_handle.accepted:
                if result_future is None:
                    try:
                        result_future = goal_handle.get_result_async()
                    except BaseException:
                        result_future = None
                cancellation_failures = self.cancel_execution(goal_handle, result_future)
                if cancellation_failures:
                    raise StopUnverifiedError(
                        f"{exc}; software cancellation was not fully verified: "
                        + "; ".join(cancellation_failures)
                        + "; use the physical E-stop"
                    ) from exc
                if isinstance(exc, StopUnverifiedError):
                    raise RuntimeError(
                        f"{exc}; software cancellation and sustained stop were verified"
                    ) from exc
                raise
            if goal_handle is not None and not goal_handle.accepted:
                raise
            if isinstance(exc, StopUnverifiedError):
                raise
            raise StopUnverifiedError(
                f"{exc}; execution goal acceptance is unknown for {stage_name}; "
                "use the physical E-stop"
            ) from exc
        finally:
            self.active_goal_handle = None
            self.active_result_future = None

    def cancel_execution(self, goal_handle, result_future) -> list[str]:
        """Verify cancel acknowledgement, terminal state, and a stationary robot."""

        failures: list[str] = []
        try:
            cancel_future = goal_handle.cancel_goal_async()
            self.rclpy.spin_until_future_complete(self.node, cancel_future, timeout_sec=3.0)
            if not cancel_future.done() or cancel_future.result() is None:
                failures.append("MoveIt did not acknowledge trajectory cancellation")
            elif (
                not cancel_future.result().goals_canceling
                and (result_future is None or not result_future.done())
            ):
                failures.append("MoveIt reported no active goal to cancel")
        except BaseException as exc:
            failures.append(f"trajectory cancellation raised {type(exc).__name__}: {exc}")

        try:
            if result_future is None:
                failures.append("no result future was available to verify terminal action state")
            else:
                terminal_deadline = time.monotonic() + 8.0
                while not result_future.done() and time.monotonic() < terminal_deadline:
                    self.rclpy.spin_once(self.node, timeout_sec=0.05)
                if not result_future.done() or result_future.result() is None:
                    failures.append(
                        "trajectory did not reach a terminal action state after cancellation"
                    )
                else:
                    terminal_status = result_future.result().status
                    allowed = {
                        self.ros["GoalStatus"].STATUS_CANCELED,
                        self.ros["GoalStatus"].STATUS_SUCCEEDED,
                        self.ros["GoalStatus"].STATUS_ABORTED,
                    }
                    if terminal_status not in allowed:
                        failures.append(f"trajectory ended with action status {terminal_status}")
        except BaseException as exc:
            failures.append(f"terminal action verification raised {type(exc).__name__}: {exc}")

        try:
            stationary_failure = self.verify_stationary_after_motion()
            if stationary_failure is not None:
                failures.append(stationary_failure)
        except BaseException as exc:
            failures.append(f"stationary verification raised {type(exc).__name__}: {exc}")
        return failures

    def verify_stationary_after_motion(self, timeout_s: float = 5.0) -> str | None:
        """Require a sustained one-second stationary window after a commanded motion."""

        deadline = time.monotonic() + timeout_s
        last_error = "fresh stationary feedback was not verified"
        while time.monotonic() < deadline:
            try:
                self.require_healthy(
                    stationary=True,
                    require_auto_mode=False,
                    honor_stop_request=False,
                )
                return None
            except RuntimeError as exc:
                last_error = str(exc)
        return f"sustained stationary feedback was not verified: {last_error}"

    def destroy(self) -> None:
        self.node.destroy_node()


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"qualification report {field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"qualification report {field} must be a finite number")
    return result


def _finite_joint_vector(value: object, field: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != len(JOINT_NAMES):
        raise ValueError(
            f"qualification report {field} must contain exactly six joint values"
        )
    return tuple(
        _finite_number(joint_value, f"{field}[{index}]")
        for index, joint_value in enumerate(value)
    )


def _require_close(actual: object, expected: float, field: str) -> float:
    value = _finite_number(actual, field)
    if abs(value - expected) > 1e-9:
        raise ValueError(
            f"qualification report {field} must equal {expected!r}, got {value!r}"
        )
    return value


def _require_joint_match(
    actual: object,
    expected: tuple[float, ...],
    field: str,
) -> tuple[float, ...]:
    values = _finite_joint_vector(actual, field)
    error = max(abs(values[index] - expected[index]) for index in range(len(values)))
    if error > 1e-9:
        raise ValueError(
            f"qualification report {field} does not match the expected J6 sequence"
        )
    return values


def _health_snapshot_from_report(value: object, field: str) -> HealthSnapshot:
    if not isinstance(value, dict):
        raise ValueError(f"qualification report {field} must be a health object")
    expected_fields = set(HealthSnapshot.__dataclass_fields__)
    if set(value) != expected_fields:
        raise ValueError(
            f"qualification report {field} does not match the health schema"
        )
    bool_fields = (
        "is_svr_connected",
        "is_sct_connected",
        "is_data_table_correct",
        "robot_link",
        "robot_error",
        "project_run",
        "project_pause",
        "safetyguard_a",
        "e_stop",
    )
    int_fields = (
        "tmsrv_cperr",
        "tmscript_cperr",
        "tmsrv_dataerr",
        "tmscript_dataerr",
        "error_code",
        "project_speed",
        "ma_mode",
        "robot_light",
    )
    parsed: dict[str, object] = {}
    for name in bool_fields:
        if not isinstance(value[name], bool):
            raise ValueError(
                f"qualification report {field}.{name} must be boolean"
            )
        parsed[name] = value[name]
    for name in int_fields:
        if isinstance(value[name], bool) or not isinstance(value[name], int):
            raise ValueError(
                f"qualification report {field}.{name} must be an integer"
            )
        parsed[name] = value[name]
    for name in (
        "joint_positions",
        "feedback_joint_positions",
        "joint_velocities",
    ):
        parsed[name] = _finite_joint_vector(value[name], f"{field}.{name}")
    for name in ("feedback_age_s", "joint_state_age_s"):
        parsed[name] = _finite_number(value[name], f"{field}.{name}")
    return HealthSnapshot(**parsed)


def _trajectory_samples_from_report(
    value: object,
    field: str,
) -> tuple[TrajectorySample, ...]:
    if not isinstance(value, dict):
        raise ValueError(f"qualification report {field} must be a trajectory object")
    required_fields = {
        "joint_names",
        "multi_dof_joint_names",
        "multi_dof_point_count",
        "points",
        "sha256",
    }
    if set(value) != required_fields:
        raise ValueError(
            f"qualification report {field} does not match the trajectory schema"
        )
    if value["joint_names"] != list(JOINT_NAMES):
        raise ValueError(
            f"qualification report {field} joint order is not the Techman order"
        )
    if value["multi_dof_joint_names"] != [] or value["multi_dof_point_count"] != 0:
        raise ValueError(
            f"qualification report {field} must not contain a multi-DOF trajectory"
        )
    stored_hash = value["sha256"]
    if not isinstance(stored_hash, str) or re.fullmatch(r"[0-9a-f]{64}", stored_hash) is None:
        raise ValueError(f"qualification report {field} has an invalid trajectory hash")
    hash_payload = dict(value)
    hash_payload.pop("sha256")
    canonical = json.dumps(
        hash_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != stored_hash:
        raise ValueError(f"qualification report {field} trajectory hash does not match")
    points = value["points"]
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError(
            f"qualification report {field} must contain at least two trajectory points"
        )
    samples: list[TrajectorySample] = []
    for index, point in enumerate(points):
        if not isinstance(point, dict) or set(point) != {
            "time_s",
            "positions_rad",
            "velocities_rad_s",
            "accelerations_rad_s2",
        }:
            raise ValueError(
                f"qualification report {field}.points[{index}] has invalid fields"
            )
        samples.append(
            TrajectorySample(
                positions=_finite_joint_vector(
                    point["positions_rad"],
                    f"{field}.points[{index}].positions_rad",
                ),
                velocities=_finite_joint_vector(
                    point["velocities_rad_s"],
                    f"{field}.points[{index}].velocities_rad_s",
                ),
                accelerations=_finite_joint_vector(
                    point["accelerations_rad_s2"],
                    f"{field}.points[{index}].accelerations_rad_s2",
                ),
                time_s=_finite_number(
                    point["time_s"],
                    f"{field}.points[{index}].time_s",
                ),
            )
        )
    return tuple(samples)


def _require_metrics_match(
    actual: object,
    expected: dict,
    field: str,
) -> None:
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError(
            f"qualification report {field} does not match recomputed metrics"
        )
    for name, expected_value in expected.items():
        actual_value = actual[name]
        if isinstance(expected_value, int):
            if isinstance(actual_value, bool) or actual_value != expected_value:
                raise ValueError(
                    f"qualification report {field}.{name} does not match recomputed metrics"
                )
        else:
            parsed = _finite_number(actual_value, f"{field}.{name}")
            if not math.isclose(
                parsed,
                float(expected_value),
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"qualification report {field}.{name} does not match recomputed metrics"
                )


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"qualification report contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"qualification report contains non-finite JSON value {value}")


def validate_qualification_report(
    path: Path,
    *,
    now: datetime | None = None,
) -> dict[str, object]:
    """Validate one trusted, recent, physically successful 6-degree report."""

    if not path.is_absolute():
        raise ValueError("--qualification-report must be an absolute path")
    report_dir = (ARENA_DIR / "outputs" / "watson_guarded_demo").resolve()
    try:
        supplied_parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"qualification report directory is unavailable: {exc}") from exc
    if supplied_parent != report_dir:
        raise ValueError(
            "--qualification-report must be a direct child of "
            f"{report_dir}"
        )
    if QUALIFICATION_REPORT_NAME_RE.fullmatch(path.name) is None:
        raise ValueError(
            "--qualification-report must use the guarded runner's "
            "YYYYMMDDTHHMMSSZ_execute.json filename"
        )

    directory_fd = -1
    report_fd = -1
    try:
        directory_fd = os.open(
            report_dir,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY,
        )
        report_fd = os.open(
            path.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        file_stat = os.fstat(report_fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("qualification report must be a regular file")
        if file_stat.st_uid != os.getuid():
            raise ValueError(
                f"qualification report must be owned by uid {os.getuid()}"
            )
        if file_stat.st_nlink != 1:
            raise ValueError("qualification report must have exactly one hard link")
        if file_stat.st_mode & 0o022:
            raise ValueError("qualification report must not be group/world writable")
        with os.fdopen(report_fd, "rb", closefd=True) as report_file:
            report_fd = -1
            raw = report_file.read(MAX_QUALIFICATION_REPORT_BYTES + 1)
    except OSError as exc:
        raise ValueError(f"could not securely read qualification report: {exc}") from exc
    finally:
        if report_fd >= 0:
            os.close(report_fd)
        if directory_fd >= 0:
            os.close(directory_fd)
    if len(raw) > MAX_QUALIFICATION_REPORT_BYTES:
        raise ValueError("qualification report exceeds the fixed 8 MiB size cap")
    try:
        report = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"qualification report is not strict JSON: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError("qualification report root must be a JSON object")
    stored_report_digest = report.get(REPORT_DIGEST_FIELD)
    if (
        not isinstance(stored_report_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", stored_report_digest) is None
        or stored_report_digest != report_payload_sha256(report)
    ):
        raise ValueError("qualification report payload digest does not match")

    exact_fields = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "mode": "execute",
        "namespace": "/watson",
        "robot_ip": ROBOT_IP,
        "robot_interface": ROBOT_INTERFACE,
        "robot_source_ip": ROBOT_SOURCE_IP,
        "robot_mac": ROBOT_MAC,
        "runner_source_sha256": RUNNER_SOURCE_SHA256,
        "guard_source_sha256": GUARD_SOURCE_SHA256,
        "motion_profile": J6_QUALIFICATION_PROFILE,
        "motion_pattern": "j6_toward_zero_then_return",
        "ros_automatic_discovery_range": "LOCALHOST",
        "publishes_joint_states": False,
        "direct_controller_goals": False,
        "commands_gripper": False,
        "queries_controller_tool_settings_read_only": True,
        "controller_tool_settings_promotion_passed": True,
        "motion_commanded": True,
        "status": "execution_passed_and_returned",
    }
    for field, expected in exact_fields.items():
        if report.get(field) != expected:
            raise ValueError(
                f"qualification report {field} must be {expected!r}, "
                f"got {report.get(field)!r}"
            )
    current_domain = os.environ.get("ROS_DOMAIN_ID")
    if current_domain is None or report.get("ros_domain_id") != current_domain:
        raise ValueError(
            "qualification report ROS domain must match the current explicit domain"
        )
    if report.get("health_failures") != []:
        raise ValueError("qualification report must contain no health failures")

    profile = get_j6_guard_profile(J6_QUALIFICATION_PROFILE)
    numeric_fields = {
        "amplitude_deg": profile.requested_amplitude_deg,
        "hard_max_excursion_deg": profile.hard_excursion_deg,
        "hard_max_excursion_rad": profile.hard_excursion_rad,
        "hard_max_planned_velocity_rad_s": profile.max_planned_velocity_rad_s,
        "hard_max_planned_acceleration_rad_s2": (
            profile.max_planned_acceleration_rad_s2
        ),
        "hard_max_live_velocity_rad_s": profile.max_live_velocity_rad_s,
        "hard_max_sample_step_rad": profile.max_sample_step_rad,
        "velocity_scaling": profile.velocity_scaling,
        "acceleration_scaling": profile.acceleration_scaling,
        "max_project_speed": float(profile.max_project_speed),
    }
    for field, expected in numeric_fields.items():
        _require_close(report.get(field), expected, field)
    duration_range = report.get("hard_duration_range_s")
    if not isinstance(duration_range, list) or len(duration_range) != 2:
        raise ValueError(
            "qualification report hard_duration_range_s must contain two values"
        )
    _require_close(duration_range[0], profile.min_duration_s, "hard_duration_range_s[0]")
    _require_close(duration_range[1], profile.max_duration_s, "hard_duration_range_s[1]")

    timestamp_text = report.get("timestamp_utc")
    if not isinstance(timestamp_text, str):
        raise ValueError("qualification report timestamp_utc must be an ISO timestamp")
    try:
        timestamp = datetime.fromisoformat(timestamp_text)
    except ValueError as exc:
        raise ValueError("qualification report timestamp_utc is invalid") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
        raise ValueError("qualification report timestamp_utc must be timezone-aware UTC")
    checked_at = now or datetime.now(timezone.utc)
    if checked_at.tzinfo is None:
        raise ValueError("qualification validation time must be timezone-aware")
    age_s = (checked_at.astimezone(timezone.utc) - timestamp).total_seconds()
    if age_s < 0.0:
        raise ValueError("qualification report timestamp is in the future")
    if age_s > MAX_QUALIFICATION_REPORT_AGE_S:
        raise ValueError(
            "qualification report is stale; rerun the supervised 6-degree "
            f"qualification within {MAX_QUALIFICATION_REPORT_AGE_S / 3600:.0f} hours"
        )

    initial_health = _health_snapshot_from_report(
        report.get("initial_health"),
        "initial_health",
    )
    initial_failures = health_failures(
        initial_health,
        max_project_speed=profile.max_project_speed,
        require_stationary=False,
        require_auto_mode=True,
    )
    if initial_failures:
        raise ValueError(
            "qualification report initial health is unsafe: "
            + "; ".join(initial_failures)
        )
    stable_health = _health_snapshot_from_report(
        report.get("stable_health"),
        "stable_health",
    )
    stable_failures = health_failures(
        stable_health,
        max_project_speed=profile.max_project_speed,
        require_stationary=True,
        require_auto_mode=True,
    )
    if stable_failures:
        raise ValueError(
            "qualification report stable health is unsafe: "
            + "; ".join(stable_failures)
        )

    hard_reference = _finite_joint_vector(
        report.get("hard_reference_start_rad"),
        "hard_reference_start_rad",
    )
    stable_reference_error = max(
        abs(stable_health.joint_positions[index] - hard_reference[index])
        for index in range(len(JOINT_NAMES))
    )
    stable_feedback_reference_error = max(
        abs(stable_health.feedback_joint_positions[index] - hard_reference[index])
        for index in range(len(JOINT_NAMES))
    )
    if max(stable_reference_error, stable_feedback_reference_error) > (
        LIVE_POSE_TOLERANCE_RAD
    ):
        raise ValueError(
            "qualification report stable health is not anchored to the hard reference"
        )
    expected_stages = j6_profile_targets(
        hard_reference,
        guard_profile=J6_QUALIFICATION_PROFILE,
    )
    expected_names = [stage_name for stage_name, _ in expected_stages]
    plans = report.get("plans")
    if not isinstance(plans, list) or len(plans) != len(expected_stages):
        raise ValueError("qualification report must contain exactly two plans")
    expected_start = hard_reference
    validated_plan_samples: list[tuple[TrajectorySample, ...]] = []
    for index, ((stage_name, expected_goal), plan) in enumerate(
        zip(expected_stages, plans, strict=True)
    ):
        if not isinstance(plan, dict) or plan.get("stage") != stage_name:
            raise ValueError(
                f"qualification report plan {index} must be stage {stage_name!r}"
            )
        _require_joint_match(
            plan.get("hard_reference_start_rad"),
            hard_reference,
            f"plans[{index}].hard_reference_start_rad",
        )
        _require_joint_match(
            plan.get("start_positions_rad"),
            expected_start,
            f"plans[{index}].start_positions_rad",
        )
        _require_joint_match(
            plan.get("goal_positions_rad"),
            expected_goal,
            f"plans[{index}].goal_positions_rad",
        )
        samples = _trajectory_samples_from_report(
            plan.get("trajectory"),
            f"plans[{index}].trajectory",
        )
        recomputed_metrics = validate_trajectory_samples(
            samples,
            expected_start=expected_start,
            expected_goal=expected_goal,
            hard_reference_start=hard_reference,
            max_goal_error_rad=MAX_PLANNED_GOAL_ERROR_RAD,
            max_excursion_rad=profile.hard_excursion_rad,
            max_sample_step_rad=profile.max_sample_step_rad,
            max_velocity_rad_s=profile.max_planned_velocity_rad_s,
            max_acceleration_rad_s2=profile.max_planned_acceleration_rad_s2,
            min_total_duration_s=profile.min_duration_s,
            max_total_duration_s=profile.max_duration_s,
            guard_profile=profile.name,
        )
        _require_metrics_match(
            plan.get("metrics"),
            recomputed_metrics,
            f"plans[{index}].metrics",
        )
        validated_plan_samples.append(samples)
        expected_start = expected_goal

    executions = report.get("execution")
    if not isinstance(executions, list) or len(executions) != len(expected_names):
        raise ValueError("qualification report must contain exactly two executions")
    max_live_goal_error = 0.0
    expected_start = hard_reference
    for index, execution in enumerate(executions):
        stage_name, expected_goal = expected_stages[index]
        if not isinstance(execution, dict) or execution.get("stage") != stage_name:
            raise ValueError(
                f"qualification report execution {index} must be stage {stage_name!r}"
            )
        if execution.get("action_status") != ACTION_STATUS_SUCCEEDED:
            raise ValueError(
                f"qualification report execution {index} did not succeed"
            )
        if execution.get("moveit_error_code") != MOVEIT_SUCCESS:
            raise ValueError(
                f"qualification report execution {index} has a MoveIt error"
            )
        live_start_error = _finite_number(
            execution.get("live_start_error_rad"),
            f"execution[{index}].live_start_error_rad",
        )
        if live_start_error < 0.0 or live_start_error > LIVE_POSE_TOLERANCE_RAD:
            raise ValueError(
                f"qualification report execution {index} live start error exceeds "
                f"{LIVE_POSE_TOLERANCE_RAD:.6f} rad"
            )
        live_goal_error = _finite_number(
            execution.get("live_goal_error_rad"),
            f"execution[{index}].live_goal_error_rad",
        )
        if live_goal_error < 0.0 or live_goal_error > LIVE_POSE_TOLERANCE_RAD:
            raise ValueError(
                f"qualification report execution {index} live goal error exceeds "
                f"{LIVE_POSE_TOLERANCE_RAD:.6f} rad"
            )
        physical_start = _finite_joint_vector(
            execution.get("physical_start_positions_rad"),
            f"execution[{index}].physical_start_positions_rad",
        )
        calculated_start_error = max(
            abs(physical_start[joint] - expected_start[joint])
            for joint in range(len(JOINT_NAMES))
        )
        if calculated_start_error > LIVE_POSE_TOLERANCE_RAD or not math.isclose(
            calculated_start_error,
            live_start_error,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"qualification report execution {index} physical start is not "
                "bound to the planned stage"
            )
        physical_start_velocities = _finite_joint_vector(
            execution.get("physical_start_velocities_rad_s"),
            f"execution[{index}].physical_start_velocities_rad_s",
        )
        if max(abs(value) for value in physical_start_velocities) > (
            profile.max_live_velocity_rad_s
        ):
            raise ValueError(
                f"qualification report execution {index} physical start velocity "
                "exceeds the qualification cap"
            )
        for field in (
            "physical_start_feedback_age_s",
            "physical_start_joint_state_age_s",
        ):
            message_age = _finite_number(
                execution.get(field),
                f"execution[{index}].{field}",
            )
            if message_age < 0.0 or message_age > 0.5:
                raise ValueError(
                    f"qualification report execution {index} contains stale feedback"
                )
        stationary_drift = _finite_number(
            execution.get("stationary_to_physical_start_drift_rad"),
            f"execution[{index}].stationary_to_physical_start_drift_rad",
        )
        if stationary_drift < 0.0 or stationary_drift > 0.001:
            raise ValueError(
                f"qualification report execution {index} stationary drift is unsafe"
            )
        if execution.get("post_motion_stationary_verified") is not True:
            raise ValueError(
                f"qualification report execution {index} lacks stationary-stop proof"
            )
        final_joint_positions = _finite_joint_vector(
            execution.get("final_joint_positions_rad"),
            f"execution[{index}].final_joint_positions_rad",
        )
        final_feedback_positions = _finite_joint_vector(
            execution.get("final_feedback_positions_rad"),
            f"execution[{index}].final_feedback_positions_rad",
        )
        calculated_goal_error = max(
            abs(final_feedback_positions[joint] - expected_goal[joint])
            for joint in range(len(JOINT_NAMES))
        )
        final_joint_goal_error = max(
            abs(final_joint_positions[joint] - expected_goal[joint])
            for joint in range(len(JOINT_NAMES))
        )
        final_source_delta = max(
            abs(final_joint_positions[joint] - final_feedback_positions[joint])
            for joint in range(len(JOINT_NAMES))
        )
        if (
            calculated_goal_error > LIVE_POSE_TOLERANCE_RAD
            or final_joint_goal_error > LIVE_POSE_TOLERANCE_RAD
            or final_source_delta > 0.005
            or not math.isclose(
                calculated_goal_error,
                live_goal_error,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"qualification report execution {index} final pose is not "
                "bound to the planned goal"
            )
        final_velocities = _finite_joint_vector(
            execution.get("final_joint_velocities_rad_s"),
            f"execution[{index}].final_joint_velocities_rad_s",
        )
        if max(abs(value) for value in final_velocities) > 0.01:
            raise ValueError(
                f"qualification report execution {index} final state is not stationary"
            )
        for field in ("final_feedback_age_s", "final_joint_state_age_s"):
            message_age = _finite_number(
                execution.get(field),
                f"execution[{index}].{field}",
            )
            if message_age < 0.0 or message_age > 0.5:
                raise ValueError(
                    f"qualification report execution {index} final feedback is stale"
                )
        recomputed_execution_metrics = validate_trajectory_samples(
            validated_plan_samples[index],
            expected_start=expected_start,
            expected_goal=expected_goal,
            hard_reference_start=hard_reference,
            hard_travel_start=physical_start,
            execution_start_positions=physical_start,
            execution_start_velocities=physical_start_velocities,
            max_goal_error_rad=MAX_PLANNED_GOAL_ERROR_RAD,
            max_excursion_rad=profile.hard_excursion_rad,
            max_sample_step_rad=profile.max_sample_step_rad,
            max_velocity_rad_s=profile.max_planned_velocity_rad_s,
            max_acceleration_rad_s2=profile.max_planned_acceleration_rad_s2,
            min_total_duration_s=profile.min_duration_s,
            max_total_duration_s=profile.max_duration_s,
            guard_profile=profile.name,
        )
        _require_metrics_match(
            execution.get("execution_revalidation_metrics"),
            recomputed_execution_metrics,
            f"execution[{index}].execution_revalidation_metrics",
        )
        max_live_goal_error = max(max_live_goal_error, live_goal_error)
        expected_start = expected_goal

    return {
        "required": True,
        "report_validation_passed": True,
        "live_pose_match_passed": None,
        "passed": False,
        "report_path": str(path),
        "report_sha256": hashlib.sha256(raw).hexdigest(),
        "report_timestamp_utc": timestamp.isoformat(),
        "report_age_s": age_s,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "report_status": report["status"],
        "report_motion_profile": report["motion_profile"],
        "execution_stages": expected_names,
        "max_live_goal_error_rad": max_live_goal_error,
        "hard_reference_start_rad": list(hard_reference),
    }


def require_fresh_showcase_gate_before_send(
    args: argparse.Namespace,
    stage_name: str,
    *,
    now: datetime | None = None,
) -> None:
    """Recheck evidence only before outbound showcase; never block its return."""

    if (
        getattr(args, "motion_profile", FIRST_MOTION_PROFILE)
        != J6_SHOWCASE_PROFILE
        or stage_name != "j6_showcase"
    ):
        return
    gate = getattr(args, "qualification_gate", None)
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise RuntimeError(
            "j6_showcase outbound send lacks a fully passed qualification gate"
        )
    timestamp_text = gate.get("report_timestamp_utc")
    if not isinstance(timestamp_text, str):
        raise RuntimeError("j6_showcase qualification timestamp is missing")
    try:
        timestamp = datetime.fromisoformat(timestamp_text)
    except ValueError as exc:
        raise RuntimeError("j6_showcase qualification timestamp is invalid") from exc
    checked_at = now or datetime.now(timezone.utc)
    age_s = (checked_at.astimezone(timezone.utc) - timestamp).total_seconds()
    gate["report_age_at_outbound_send_s"] = age_s
    if age_s < 0.0 or age_s > MAX_QUALIFICATION_REPORT_AGE_S:
        raise RuntimeError(
            "j6_showcase qualification evidence expired before outbound send; "
            "rerun the supervised 6-degree qualification"
        )


def validate_cli(args: argparse.Namespace) -> None:
    profile_name = getattr(args, "motion_profile", FIRST_MOTION_PROFILE)
    profile = get_j6_guard_profile(profile_name)
    args.motion_profile = profile.name
    args.qualification_gate = None
    if profile.name == FIRST_MOTION_PROFILE:
        if not 0.0 < args.amplitude_deg <= 1.0:
            raise ValueError("--amplitude-deg must be greater than 0 and at most 1")
        if not 0.0 < args.velocity_scaling <= 0.01:
            raise ValueError(
                "--velocity-scaling must be greater than 0 and at most 0.01"
            )
        if not 0.0 < args.acceleration_scaling <= 0.01:
            raise ValueError(
                "--acceleration-scaling must be greater than 0 and at most 0.01"
            )
        if not 1 <= args.max_project_speed <= 5:
            raise ValueError("--max-project-speed must be between 1 and 5")
    else:
        legacy_defaults = (0.9, 0.01, 0.01, 5)
        supplied = (
            args.amplitude_deg,
            args.velocity_scaling,
            args.acceleration_scaling,
            args.max_project_speed,
        )
        if supplied != legacy_defaults:
            raise ValueError(
                f"--motion-profile {profile.name} is immutable; omit custom "
                "--amplitude-deg, scaling, and --max-project-speed values"
            )
        args.amplitude_deg = profile.requested_amplitude_deg
        args.velocity_scaling = profile.velocity_scaling
        args.acceleration_scaling = profile.acceleration_scaling
        args.max_project_speed = profile.max_project_speed
    if args.mode == "execute":
        if (
            profile.name == FIRST_MOTION_PROFILE
            and args.amplitude_deg > MAX_EXECUTE_AMPLITUDE_DEG
        ):
            raise ValueError(
                "--mode execute is capped at --amplitude-deg 0.9 to retain "
                "margin under the hard one-degree physical envelope"
            )
        if "/" + args.namespace.strip("/") != "/watson":
            raise ValueError("--mode execute is locked to --namespace /watson")
        if args.group_name != "tmr_arm":
            raise ValueError("--mode execute is locked to --group-name tmr_arm")
        if args.planning_frame != "base":
            raise ValueError("--mode execute is locked to --planning-frame base")
        if os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST":
            raise ValueError(
                "--mode execute requires ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST"
            )
        domain_text = os.environ.get("ROS_DOMAIN_ID")
        try:
            domain_id = int(domain_text) if domain_text is not None else -1
        except ValueError as exc:
            raise ValueError("--mode execute requires an explicit integer ROS_DOMAIN_ID") from exc
        if not 0 <= domain_id <= 232:
            raise ValueError(
                "--mode execute requires an explicit ROS_DOMAIN_ID between 0 and 232"
            )
        required_token = {
            FIRST_MOTION_PROFILE: ARM_TOKEN,
            J6_QUALIFICATION_PROFILE: J6_QUALIFICATION_ARM_TOKEN,
            J6_SHOWCASE_PROFILE: J6_SHOWCASE_ARM_TOKEN,
        }[profile.name]
        if args.arm_token != required_token:
            raise ValueError(
                f"--mode execute with {profile.name} requires --arm-token "
                f"{required_token}"
            )
        if not args.confirm_cell_clear:
            raise ValueError("--mode execute requires --confirm-cell-clear")
    qualification_report = getattr(args, "qualification_report", None)
    showcase_execute = (
        args.mode == "execute" and profile.name == J6_SHOWCASE_PROFILE
    )
    if showcase_execute:
        if qualification_report is None:
            raise ValueError(
                "j6_showcase execution requires --qualification-report from a "
                "successful recent 6-degree execution"
            )
        args.qualification_gate = validate_qualification_report(
            qualification_report
        )
        output_report = getattr(args, "report", None)
        if output_report is not None and output_report.expanduser().resolve() == Path(
            args.qualification_gate["report_path"]
        ).resolve():
            raise ValueError(
                "--report cannot overwrite the consumed qualification report"
            )
    elif qualification_report is not None:
        raise ValueError(
            "--qualification-report is valid only for "
            "--mode execute --motion-profile j6_showcase"
        )


def validate_execute_network() -> None:
    """Prevent a direct Python invocation from bypassing the Watson route guard."""

    carrier_path = Path(f"/sys/class/net/{ROBOT_INTERFACE}/carrier")
    try:
        carrier = carrier_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(
            f"cannot read physical carrier for {ROBOT_INTERFACE}: {exc}"
        ) from exc
    if carrier != "1":
        raise ValueError(
            f"{ROBOT_INTERFACE} has no physical Ethernet carrier; check the robot link"
        )
    try:
        route_result = subprocess.run(
            ["ip", "-4", "route", "get", ROBOT_IP],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"could not verify Watson route: {exc}") from exc
    route = route_result.stdout.strip()
    if (
        route_result.returncode != 0
        or f"dev {ROBOT_INTERFACE}" not in route
        or f"src {ROBOT_SOURCE_IP}" not in route
    ):
        raise ValueError(
            "Watson is not routed over the dedicated robot link; "
            f"observed route: {route or route_result.stderr.strip() or 'none'}"
        )
    try:
        neighbour_result = subprocess.run(
            ["ip", "neigh", "show", ROBOT_IP, "dev", ROBOT_INTERFACE],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"could not verify Watson neighbour identity: {exc}") from exc
    neighbour = neighbour_result.stdout.strip().lower()
    if (
        neighbour_result.returncode != 0
        or f"lladdr {ROBOT_MAC}" not in neighbour
        or " failed" in neighbour
        or " incomplete" in neighbour
    ):
        raise ValueError(
            "Watson Ethernet identity does not match the commissioned robot; "
            f"observed neighbour: {neighbour or neighbour_result.stderr.strip() or 'none'}"
        )


def acquire_execute_lock(path: Path = EXECUTE_LOCK_PATH):
    """Hold an exclusive process lock for the complete execute-mode run."""

    try:
        fd = os.open(
            path,
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as exc:
        raise ValueError(f"could not open Watson execute lock {path}: {exc}") from exc
    try:
        stat = os.fstat(fd)
        if stat.st_uid != os.getuid():
            raise ValueError(
                f"Watson execute lock {path} is not owned by uid {os.getuid()}"
            )
        lock_file = os.fdopen(fd, "r+", encoding="utf-8")
        fd = -1
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.close()
        raise ValueError(
            "another Watson guarded execute process already holds the motion lock"
        ) from exc
    except (OSError, ValueError) as exc:
        if fd >= 0:
            os.close(fd)
        elif "lock_file" in locals():
            lock_file.close()
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"could not acquire Watson execute lock {path}: {exc}") from exc
    try:
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(
            f"pid={os.getpid()} started_utc={datetime.now(timezone.utc).isoformat()}\n"
        )
        lock_file.flush()
    except OSError as exc:
        lock_file.close()
        raise ValueError(f"could not record Watson execute lock owner: {exc}") from exc
    return lock_file


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report[REPORT_DIGEST_FIELD] = report_payload_sha256(report)
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", closefd=True) as report_file:
            fd = -1
            json.dump(report, report_file, indent=2)
            report_file.write("\n")
            report_file.flush()
            os.fsync(report_file.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    print(f"Report: {path}")


def write_report_best_effort(path: Path, report: dict) -> None:
    """Preserve operator-facing failure output even if evidence storage fails."""

    try:
        write_report(path, report)
    except BaseException as exc:
        print(
            f"REPORT WRITE FAILED: status={report.get('status')} path={path}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def main() -> int:
    args = build_parser().parse_args()
    execute_lock = None
    try:
        validate_cli(args)
        if args.mode == "execute":
            validate_execute_network()
            execute_lock = acquire_execute_lock()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = args.report or (
        ARENA_DIR / "outputs" / "watson_guarded_demo" / f"{timestamp}_{args.mode}.json"
    )
    guard_profile = get_j6_guard_profile(args.motion_profile)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "namespace": "/" + args.namespace.strip("/"),
        "robot_ip": ROBOT_IP,
        "robot_interface": ROBOT_INTERFACE,
        "robot_source_ip": ROBOT_SOURCE_IP,
        "robot_mac": ROBOT_MAC,
        "runner_source_sha256": RUNNER_SOURCE_SHA256,
        "guard_source_sha256": GUARD_SOURCE_SHA256,
        "motion_profile": guard_profile.name,
        "motion_pattern": "j6_toward_zero_then_return",
        "amplitude_deg": args.amplitude_deg,
        "hard_max_excursion_deg": guard_profile.hard_excursion_deg,
        "hard_max_excursion_rad": guard_profile.hard_excursion_rad,
        "hard_max_planned_velocity_rad_s": (
            guard_profile.max_planned_velocity_rad_s
        ),
        "hard_max_planned_acceleration_rad_s2": (
            guard_profile.max_planned_acceleration_rad_s2
        ),
        "hard_max_live_velocity_rad_s": guard_profile.max_live_velocity_rad_s,
        "hard_max_sample_step_rad": guard_profile.max_sample_step_rad,
        "hard_duration_range_s": [
            guard_profile.min_duration_s,
            guard_profile.max_duration_s,
        ],
        "velocity_scaling": args.velocity_scaling,
        "acceleration_scaling": args.acceleration_scaling,
        "max_project_speed": args.max_project_speed,
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
        "ros_automatic_discovery_range": os.environ.get(
            "ROS_AUTOMATIC_DISCOVERY_RANGE"
        ),
        "publishes_joint_states": False,
        "direct_controller_goals": False,
        "commands_gripper": False,
        "queries_controller_tool_settings_read_only": True,
        "real_tool_collision_geometry_in_moveit": False,
        "safeguard_a_field_available_in_default_table": False,
        "motion_commanded": False,
        "qualification_gate": args.qualification_gate,
        "execute_lock_path": str(EXECUTE_LOCK_PATH) if args.mode == "execute" else None,
        "status": "started",
    }

    ros = None
    try:
        ros = load_ros()
        ros["rclpy"].init(
            args=None,
            signal_handler_options=ros["SignalHandlerOptions"].NO,
        )
        guard = WatsonGuardNode(args, ros)
    except BaseException:
        if ros is not None and ros["rclpy"].ok():
            ros["rclpy"].shutdown()
        if execute_lock is not None:
            execute_lock.close()
        raise
    for stop_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(stop_signal, guard.request_stop)
    try:
        initial = guard.spin_until_state(args.state_timeout)
        guard.wait_for_expected_publishers()
        controller_tool_audit = guard.read_controller_tool_settings(
            args.service_timeout
        )
        report["controller_tool_audit"] = controller_tool_audit
        report["controller_tool_settings_promotion_passed"] = bool(
            controller_tool_audit["promotion_passed"]
        )
        if args.mode != "check":
            guard.wait_for_command_endpoints(
                require_execute=args.mode == "execute",
            )
        require_auto_mode = args.mode in {"check", "execute"}
        failures = health_failures(
            initial,
            max_project_speed=args.max_project_speed,
            require_auto_mode=require_auto_mode,
        )
        failures.extend(guard.publisher_failures())
        if args.mode == "execute" and not controller_tool_audit["promotion_passed"]:
            failures.extend(
                "controller tool commissioning: " + reason
                for reason in controller_tool_audit["promotion_failures"]
            )
        report["initial_health"] = asdict(initial)
        report["health_failures"] = failures
        if failures:
            raise RuntimeError("Watson health gate failed: " + "; ".join(failures))
        stable = guard.require_healthy(
            stationary=True,
            require_auto_mode=require_auto_mode,
        )
        report["stable_health"] = asdict(stable)
        print("Watson read-only health gate: PASS")
        if controller_tool_audit["promotion_passed"]:
            print("Controller tool commissioning gate: PASS")
        else:
            print(
                "Controller tool commissioning gate: BLOCKED - "
                + "; ".join(controller_tool_audit["promotion_failures"])
            )
        print(
            "Current joints [rad]: "
            + ", ".join(f"{value:.6f}" for value in stable.joint_positions)
        )
        if args.qualification_gate is not None:
            qualification_start = tuple(
                float(value)
                for value in args.qualification_gate["hard_reference_start_rad"]
            )
            joint_state_error = max(
                abs(stable.joint_positions[index] - qualification_start[index])
                for index in range(len(JOINT_NAMES))
            )
            feedback_error = max(
                abs(
                    stable.feedback_joint_positions[index]
                    - qualification_start[index]
                )
                for index in range(len(JOINT_NAMES))
            )
            args.qualification_gate["live_joint_state_start_error_rad"] = (
                joint_state_error
            )
            args.qualification_gate["live_feedback_start_error_rad"] = feedback_error
            if max(joint_state_error, feedback_error) > LIVE_POSE_TOLERANCE_RAD:
                args.qualification_gate["live_pose_match_passed"] = False
                raise RuntimeError(
                    "qualification evidence is not anchored to Watson's current "
                    f"returned pose ({max(joint_state_error, feedback_error):.6f}rad "
                    f"> {LIVE_POSE_TOLERANCE_RAD:.6f}rad)"
                )
            args.qualification_gate["live_pose_match_passed"] = True
            args.qualification_gate["passed"] = True
            print(
                "6-degree qualification evidence gate: PASS, current returned-pose "
                f"error {max(joint_state_error, feedback_error):.6f}rad"
            )
        if args.mode == "check":
            report["status"] = "check_passed"
            write_report(report_path, report)
            return 0

        hard_reference_start = stable.joint_positions
        report["hard_reference_start_rad"] = list(hard_reference_start)
        if guard_profile.name == FIRST_MOTION_PROFILE:
            stages = wrist_check_targets(
                hard_reference_start,
                amplitude_deg=args.amplitude_deg,
            )
        else:
            stages = j6_profile_targets(
                hard_reference_start,
                guard_profile=guard_profile.name,
            )
        planned = []
        plan_report = []
        stage_start = stable.joint_positions
        for stage_name, stage_goal in stages:
            trajectory, metrics, planned_samples = guard.plan_stage(
                stage_name=stage_name,
                start_positions=stage_start,
                goal_positions=stage_goal,
                hard_reference_start=hard_reference_start,
            )
            planned.append(
                (stage_name, stage_start, stage_goal, trajectory, planned_samples)
            )
            plan_report.append(
                {
                    "stage": stage_name,
                    "hard_reference_start_rad": list(hard_reference_start),
                    "start_positions_rad": list(stage_start),
                    "goal_positions_rad": list(stage_goal),
                    "metrics": metrics,
                    "trajectory": trajectory_report(trajectory),
                }
            )
            print(
                f"Plan {stage_name}: PASS, {metrics['sample_count']} samples, "
                f"{metrics['duration_s']:.3f}s, max velocity "
                f"{metrics['max_velocity_rad_s']:.6f}rad/s"
            )
            stage_start = stage_goal
        report["plans"] = plan_report
        if args.mode == "plan":
            report["status"] = "plan_passed_no_motion"
            write_report(report_path, report)
            return 0

        guard.require_healthy(stationary=True, require_auto_mode=True)
        print(
            "Execution is ARMED for a supervised "
            f"{args.amplitude_deg:.3f}-degree J6 motion and return "
            f"under the hard {guard_profile.hard_excursion_deg:.1f}-degree cap."
        )
        print("Keep the physical E-stop within immediate reach.")
        for remaining in (3, 2, 1):
            print(f"Starting in {remaining}...", flush=True)
            guard.require_healthy(stationary=True, require_auto_mode=True)

        execution_report = []
        for stage_name, stage_start, stage_goal, trajectory, planned_samples in planned:
            print(f"Executing {stage_name}...")
            execution_report.append(
                guard.execute_stage(
                    stage_name=stage_name,
                    trajectory=trajectory,
                    planned_samples=planned_samples,
                    expected_start=stage_start,
                    expected_goal=stage_goal,
                    hard_reference_start=hard_reference_start,
                )
            )
        report["execution"] = execution_report
        report["motion_commanded"] = guard.motion_command_sent
        report["status"] = "execution_passed_and_returned"
        write_report(report_path, report)
        return 0
    except StopUnverifiedError as exc:
        report["motion_commanded"] = guard.motion_command_sent
        report["status"] = "stop_unverified_use_estop"
        report["error"] = str(exc)
        print(f"EMERGENCY: {exc}", file=sys.stderr, flush=True)
        write_report_best_effort(report_path, report)
        return 3
    except (RuntimeError, ValueError) as exc:
        report["motion_commanded"] = guard.motion_command_sent
        report["status"] = "failed_closed"
        report["error"] = str(exc)
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        write_report_best_effort(report_path, report)
        return 1
    finally:
        try:
            guard.destroy()
        finally:
            try:
                if ros["rclpy"].ok():
                    ros["rclpy"].shutdown()
            finally:
                if execute_lock is not None:
                    execute_lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
