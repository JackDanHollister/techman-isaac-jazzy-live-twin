#!/usr/bin/env python3
"""Run the synthetic 3D pin-axis alignment prototype."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pin_axis_3d_sim.alignment import AlignmentConfig, alignment_metadata, make_targets
from pin_axis_3d_sim.detection import DetectionConfig, detect_pin_axes
from pin_axis_3d_sim.evaluation import evaluate_detections, summarize_evaluation
from pin_axis_3d_sim.ply_io import sampled_axes_cloud, write_point_cloud_ply
from pin_axis_3d_sim.synthetic_scene import SceneConfig, generate_scene


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pins", type=int, default=12)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "demo")
    parser.add_argument("--noise-mm", type=float, default=0.35, help="scanner noise standard deviation in mm")
    parser.add_argument("--max-tilt-deg", type=float, default=11.0)
    parser.add_argument("--cluster-radius-mm", type=float, default=14.0)
    parser.add_argument("--line-inlier-radius-mm", type=float, default=2.6)
    parser.add_argument("--tray-center-x", type=float, default=0.50, help="Tray centre X in the output frame, metres")
    parser.add_argument("--tray-center-y", type=float, default=0.0, help="Tray centre Y in the output frame, metres")
    parser.add_argument("--tray-center-z", type=float, default=0.0, help="Tray centre Z in the output frame, metres")
    parser.add_argument(
        "--frame-id",
        default="base",
        help="Frame used for point-cloud and target outputs. Use 'base' for TM5S MoveIt/RViz.",
    )
    parser.add_argument("--no-ply", action="store_true", help="Skip PLY debug output")
    return parser


def write_report(path: Path, payload: dict) -> None:
    eval_summary = summarize_evaluation(payload["evaluation"])
    lines = [
        "# Pin Axis 3D Simulation Report",
        "",
        f"- Seed: `{payload['seed']}`",
        f"- Synthetic pins: `{payload['scene']['pin_count']}`",
        f"- Point count: `{payload['scene']['point_count']}`",
        f"- Detections: `{payload['detection']['detection_count']}`",
        f"- Evaluation: {eval_summary}",
        "",
        "## Output Files",
        "",
        "- `scene_cloud.ply`: synthetic scanner cloud.",
        "- `detected_axes.ply`: red detected pin-axis samples.",
        "- `gripper_centerlines.ply`: blue virtual gripper centerline samples.",
        "- `result.json`: full machine-readable output.",
        "",
        "## Alignment Convention",
        "",
        "The virtual gripper TCP is the pinch-center point. Tool local `+Z` points downward along the approach direction, opposite `pin_axis_up`.",
        "A real `flange -> gripper_tcp` transform must be added before hardware use.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    scene_config = SceneConfig(
        tray_center_x=args.tray_center_x,
        tray_center_y=args.tray_center_y,
        tray_center_z=args.tray_center_z,
        scanner_noise_std=args.noise_mm / 1000.0,
        max_pin_tilt_deg=args.max_tilt_deg,
    )
    detection_config = DetectionConfig(
        cluster_radius=args.cluster_radius_mm / 1000.0,
        line_inlier_radius=args.line_inlier_radius_mm / 1000.0,
    )
    alignment_config = AlignmentConfig()

    scene = generate_scene(seed=args.seed, pin_count=args.pins, config=scene_config)
    detection = detect_pin_axes(scene.points, seed=args.seed + 101, config=detection_config)
    targets = make_targets(detection.detections, detection.plane, config=alignment_config)
    evaluation = evaluate_detections(scene.truth, detection.detections)

    if not args.no_ply:
        write_point_cloud_ply(args.output / "scene_cloud.ply", scene.points, scene.colors)

        axes = []
        for det in detection.detections:
            start = det.base - 0.010 * det.axis_up
            end = det.head + 0.018 * det.axis_up
            axes.append((start, end, (245, 35, 28)))
        axis_points, axis_colors = sampled_axes_cloud(axes, samples_per_axis=72)
        write_point_cloud_ply(args.output / "detected_axes.ply", axis_points, axis_colors)

        centerlines = []
        for target in targets:
            start = target.pregrasp_position
            end = target.pregrasp_position + alignment_config.centerline_visual_length * target.tool_z_axis_robot
            centerlines.append((start, end, (30, 112, 255)))
        center_points, center_colors = sampled_axes_cloud(centerlines, samples_per_axis=72)
        write_point_cloud_ply(args.output / "gripper_centerlines.ply", center_points, center_colors)

    payload = {
        "seed": args.seed,
        "frames": {
            "point_cloud_frame": args.frame_id,
            "target_frame": args.frame_id,
            "scanner_optical_to_target_frame": "identity in synthetic demo",
            "tray_center_xyz": [args.tray_center_x, args.tray_center_y, args.tray_center_z],
        },
        "scene": {
            **scene.to_metadata(),
            "pin_count": int(args.pins),
        },
        "detection": {
            **detection.to_dict(),
            "detection_count": len(detection.detections),
        },
        "alignment": {
            **alignment_metadata(alignment_config),
            "targets": [target.to_dict() for target in targets],
        },
        "evaluation": evaluation,
    }

    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(args.output / "report.md", payload)

    print(f"Output: {args.output}")
    print(summarize_evaluation(evaluation))
    if evaluation["matched_count"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
