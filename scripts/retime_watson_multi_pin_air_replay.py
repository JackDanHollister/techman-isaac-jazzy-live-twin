#!/usr/bin/env python3
"""Create a private offline-only PVT candidate for the reviewed seven-pin plan."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ARENA_DIR = Path(__file__).resolve().parents[1]
if str(ARENA_DIR) not in sys.path:
    sys.path.insert(0, str(ARENA_DIR))

from pin_axis_3d_sim.watson_multi_pin_retime import (  # noqa: E402
    DEFAULT_REVIEWED_PLAN,
    build_retimed_artifact,
    write_private_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_REVIEWED_PLAN,
        help="Exact hash-pinned reviewed source plan; altered plans are rejected.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New mode-0600 JSON path; existing paths are never overwritten.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact = build_retimed_artifact(args.plan)
    output = write_private_artifact(args.output, artifact)
    metrics = artifact["metrics"]
    print(f"Status: {artifact['status']}")
    print(f"Reviewed stages: {metrics['stage_count']}")
    print(
        "Message points (including per-stage zero seeds): "
        f"{metrics['message_point_count_including_stage_zero_seeds']}"
    )
    print(
        "Source points removed by exact 25 ms filter: "
        f"{metrics['source_filter_skipped_points']}"
    )
    print(
        "Serialized wire endpoints: "
        f"{metrics['driver_transmitted_wire_endpoint_count']}"
    )
    print(
        "Peak internal wire acceleration-limit utilization: "
        f"{metrics['wire_internal_maximum_acceleration_limit_utilization']:.6f}"
    )
    print(
        "First wire cubics pending live q/v validation: "
        f"{metrics['wire_first_cubic_count_pending_live_validation']}"
    )
    print("ROS graph created: false")
    print("Controller message created: false")
    print("Motion commanded: false")
    print(f"Artifact: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
