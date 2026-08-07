#!/usr/bin/env python3
"""Capture Watson's active TMflow TCP/base settings without commanding motion."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time


ARENA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA_DIR))

from pin_axis_3d_sim.controller_tool_state import query_controller_tool_items  # noqa: E402


SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="/watson")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--report", type=Path, default=None)
    return parser


def report_payload_sha256(report: dict) -> str:
    payload = dict(report)
    payload.pop("report_payload_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> int:
    args = build_parser().parse_args()
    if args.timeout <= 0.0:
        raise ValueError("--timeout must be positive")
    namespace = "/" + args.namespace.strip("/")
    timestamp = datetime.now(timezone.utc)
    report_path = args.report or (
        ARENA_DIR
        / "outputs/watson_controller_tool_audit"
        / f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}_read_only.json"
    )
    report_path = report_path.expanduser().resolve()
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite controller audit: {report_path}")

    try:
        import rclpy
        from rclpy.signals import SignalHandlerOptions
        from tm_msgs.msg import FeedbackState
        from tm_msgs.srv import AskItem
    except Exception as exc:
        raise RuntimeError(
            "Source ROS 2 Jazzy and the Techman workspace install/setup.bash first"
        ) from exc

    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("watson_controller_tool_audit")
    feedback = {"message": None}

    def feedback_callback(message) -> None:
        feedback["message"] = message

    node.create_subscription(
        FeedbackState,
        f"{namespace}/feedback_states",
        feedback_callback,
        20,
    )
    client = node.create_client(AskItem, f"{namespace}/ask_item")
    try:
        deadline = time.monotonic() + args.timeout
        while feedback["message"] is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        message = feedback["message"]
        if message is None:
            raise RuntimeError("timed out waiting for Watson feedback")
        if not message.is_svr_connected:
            raise RuntimeError("Watson Ethernet feedback connection is not healthy")

        audit = query_controller_tool_items(
            node=node,
            rclpy=rclpy,
            ask_item_type=AskItem,
            client=client,
            timeout_s=args.timeout,
        )
        tool0_pose = [float(value) for value in message.tool0_pose]
        tool_pose = [float(value) for value in message.tool_pose]
        maximum_pose_delta = (
            max(abs(first - second) for first, second in zip(tool0_pose, tool_pose))
            if len(tool0_pose) == len(tool_pose) and tool0_pose
            else None
        )
        report = {
            "format_version": 1,
            "timestamp_utc": timestamp.isoformat(),
            "status": (
                "captured_promotion_passed"
                if audit["promotion_passed"]
                else "captured_promotion_blocked"
            ),
            "mode": "read_only_tmflow_item_audit",
            "namespace": namespace,
            "robot_ip": os.environ.get("TECHMAN_ROBOT_IP", "192.0.2.23"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
            "ros_automatic_discovery_range": os.environ.get(
                "ROS_AUTOMATIC_DISCOVERY_RANGE"
            ),
            "script_sha256": SCRIPT_SHA256,
            "feedback": {
                "is_svr_connected": bool(message.is_svr_connected),
                "is_sct_connected": bool(message.is_sct_connected),
                "robot_link": bool(message.robot_link),
                "robot_error": bool(message.robot_error),
                "project_run": bool(message.project_run),
                "project_pause": bool(message.project_pause),
                "e_stop": bool(message.e_stop),
                "error_code": int(message.error_code),
                "tool0_pose": tool0_pose,
                "tool_pose": tool_pose,
                "maximum_tool_vs_tool0_pose_delta": maximum_pose_delta,
            },
            "controller_tool_audit": audit,
            "publishers_created": 0,
            "action_clients_created": 0,
            "write_services_called": [],
            "motion_commanded": False,
        }
        report["report_payload_sha256"] = report_payload_sha256(report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.chmod(report_path, 0o600)
        print(json.dumps(report, indent=2))
        print(f"Report: {report_path}")
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
