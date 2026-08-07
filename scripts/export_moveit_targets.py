#!/usr/bin/env python3
"""Export virtual gripper target poses from a pin-axis result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--target",
        choices=["pregrasp", "grasp", "lift", "all"],
        default="pregrasp",
        help="Which target pose to export per pin.",
    )
    parser.add_argument(
        "--end-effector-link",
        choices=["gripper_tcp", "flange"],
        default="gripper_tcp",
        help=(
            "Export poses for the virtual gripper TCP, or convert them to the "
            "TM flange link used by the current MoveIt group."
        ),
    )
    parser.add_argument(
        "--flange-to-tcp-z",
        type=float,
        default=0.16225,
        help="Approximate flange/gripper-base to 2FG7 pinch-center offset in metres.",
    )
    parser.add_argument(
        "--frame-id",
        default=None,
        help="Override the exported frame id. Use 'base' for the TM5S MoveIt demo.",
    )
    return parser


def pose_from_target(target: dict, key: str, *, end_effector_link: str, flange_to_tcp_z: float) -> dict:
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
    x, y, z = position
    qx, qy, qz, qw = target["quaternion_xyzw"]
    return {
        "position": {"x": x, "y": y, "z": z},
        "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
    }


def main() -> int:
    args = build_parser().parse_args()
    data = json.loads(args.result_json.read_text(encoding="utf-8"))
    targets = data["alignment"]["targets"]

    selected = ["pregrasp", "grasp", "lift"] if args.target == "all" else [args.target]
    export = {
        "source": str(args.result_json),
        "frame_id": args.frame_id or data["frames"]["target_frame"],
        "end_effector_link": args.end_effector_link,
        "tcp_model": "virtual_gripper_pinch_center",
        "flange_to_tcp_z_m": args.flange_to_tcp_z,
        "warning": (
            "Approximate dry-run poses. Measure the real flange->gripper_tcp "
            "transform before hardware use."
        ),
        "poses": [],
    }
    for target in targets:
        item = {"detection_id": target["detection_id"], "pin_axis_up": target["pin_axis_up"]}
        for name in selected:
            item[name] = pose_from_target(
                target,
                name,
                end_effector_link=args.end_effector_link,
                flange_to_tcp_z=args.flange_to_tcp_z,
            )
        export["poses"].append(item)

    default_name = f"moveit_targets_{args.end_effector_link}_{args.target}.json"
    out_path = args.output or args.result_json.with_name(default_name)
    out_path.write_text(json.dumps(export, indent=2), encoding="utf-8")
    print(f"Wrote {len(export['poses'])} target pose sets to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
