"""Synthetic-scene evaluation for detected pin axes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detection import PinAxisDetection
from .geometry import angle_between_deg, line_distances, serializable_vec
from .synthetic_scene import PinTruth


@dataclass(frozen=True)
class MatchResult:
    truth_id: int
    detection_id: int | None
    angular_error_deg: float | None
    axis_lateral_error_m: float | None
    head_error_m: float | None

    def to_dict(self) -> dict:
        return {
            "truth_id": int(self.truth_id),
            "detection_id": None if self.detection_id is None else int(self.detection_id),
            "angular_error_deg": None if self.angular_error_deg is None else round(float(self.angular_error_deg), 3),
            "axis_lateral_error_m": None if self.axis_lateral_error_m is None else round(float(self.axis_lateral_error_m), 6),
            "head_error_m": None if self.head_error_m is None else round(float(self.head_error_m), 6),
        }


def evaluate_detections(
    truth: list[PinTruth],
    detections: list[PinAxisDetection],
    *,
    max_match_distance: float = 0.020,
) -> dict:
    """Greedy match detections to truth and return error metrics."""
    remaining = set(range(len(detections)))
    matches: list[MatchResult] = []

    for t in truth:
        best_idx = None
        best_cost = float("inf")
        for idx in remaining:
            det = detections[idx]
            lateral = float(line_distances(t.head[None, :], det.point_on_axis, det.axis_up)[0])
            head_error = float(np.linalg.norm(t.head - det.head))
            angular = angle_between_deg(t.axis_up, det.axis_up, unsigned_axis=True)
            cost = lateral + 0.35 * head_error + 0.0004 * angular
            if cost < best_cost:
                best_idx = idx
                best_cost = cost

        if best_idx is None:
            matches.append(MatchResult(t.pin_id, None, None, None, None))
            continue

        det = detections[best_idx]
        lateral = float(line_distances(t.head[None, :], det.point_on_axis, det.axis_up)[0])
        head_error = float(np.linalg.norm(t.head - det.head))
        angular = angle_between_deg(t.axis_up, det.axis_up, unsigned_axis=True)
        if lateral > max_match_distance:
            matches.append(MatchResult(t.pin_id, None, None, None, None))
            continue

        remaining.remove(best_idx)
        matches.append(MatchResult(t.pin_id, det.detection_id, angular, lateral, head_error))

    matched = [m for m in matches if m.detection_id is not None]
    false_positive_ids = [detections[idx].detection_id for idx in sorted(remaining)]

    def mean_or_none(values: list[float]) -> float | None:
        return None if not values else float(np.mean(values))

    angular_values = [float(m.angular_error_deg) for m in matched if m.angular_error_deg is not None]
    lateral_values = [float(m.axis_lateral_error_m) for m in matched if m.axis_lateral_error_m is not None]
    head_values = [float(m.head_error_m) for m in matched if m.head_error_m is not None]

    return {
        "truth_count": int(len(truth)),
        "detection_count": int(len(detections)),
        "matched_count": int(len(matched)),
        "missed_count": int(len(truth) - len(matched)),
        "false_positive_count": int(len(false_positive_ids)),
        "false_positive_detection_ids": [int(x) for x in false_positive_ids],
        "mean_angular_error_deg": None if not angular_values else round(mean_or_none(angular_values), 3),
        "mean_axis_lateral_error_m": None if not lateral_values else round(mean_or_none(lateral_values), 6),
        "mean_head_error_m": None if not head_values else round(mean_or_none(head_values), 6),
        "matches": [m.to_dict() for m in matches],
    }


def summarize_evaluation(evaluation: dict) -> str:
    """Human-readable one-line summary."""
    return (
        f"matched {evaluation['matched_count']}/{evaluation['truth_count']} pins; "
        f"false positives {evaluation['false_positive_count']}; "
        f"mean angle {evaluation['mean_angular_error_deg']} deg; "
        f"mean lateral {evaluation['mean_axis_lateral_error_m']} m; "
        f"mean head {evaluation['mean_head_error_m']} m"
    )
