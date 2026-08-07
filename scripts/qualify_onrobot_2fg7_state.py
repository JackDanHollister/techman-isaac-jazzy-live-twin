#!/usr/bin/env python3
"""Qualify a passive OnRobot 2FG7 state snapshot for air-demo synchronization."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import sys
from typing import Any


ARENA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA_DIR))

from pin_axis_3d_sim.onrobot_state import (  # noqa: E402
    COMPUTE_BOX_ORIGIN,
    LIVE_CONFIRMATION,
    MAX_CAPTURE_BYTES,
    QualificationError,
    REPORT_DIGEST_FIELD,
    SOCKET_IO_PATH,
    build_qualification_report,
    canonical_digest,
    capture_live_read_only,
    capture_timestamp,
    decode_capture,
    write_private_report,
)


def _positive_finite(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--capture",
        type=Path,
        help="offline JSON or Socket.IO capture; opens no network connection",
    )
    source.add_argument(
        "--live-read-only",
        action="store_true",
        help="receive one passive message event from the fixed Compute Box route",
    )
    parser.add_argument(
        "--confirm-live-read-only",
        default=None,
        metavar="TOKEN",
        help=f"required only for live mode; exact token: {LIVE_CONFIRMATION}",
    )
    parser.add_argument(
        "--captured-at",
        default=None,
        help="trusted ISO-8601 capture time for an offline raw event",
    )
    parser.add_argument("--max-age-seconds", type=_positive_finite, default=5.0)
    parser.add_argument("--timeout-seconds", type=_positive_finite, default=5.0)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="new private JSON report path; existing files are never overwritten",
    )
    return parser


def _default_report_path(timestamp: datetime) -> Path:
    return (
        ARENA_DIR
        / "outputs/onrobot_2fg7_qualification"
        / f"{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}_read_only.json"
    )


def _read_offline_capture(path: Path) -> tuple[Any, str]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise QualificationError(f"offline capture is not a regular file: {resolved}")
    size = resolved.stat().st_size
    if size > MAX_CAPTURE_BYTES:
        raise QualificationError("capture exceeds the 2 MiB input limit")
    raw = resolved.read_bytes()
    return decode_capture(raw), hashlib.sha256(raw).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    timestamp = datetime.now(timezone.utc)
    report_path = (args.report or _default_report_path(timestamp)).expanduser()
    mode = "live_read_only_socketio" if args.live_read_only else "offline_capture"
    capture: Any = {}
    captured_at: datetime | str | None = args.captured_at
    capture_digest: str | None = None
    transport: dict[str, Any] | None = None
    acquisition_failure: str | None = None

    if args.live_read_only:
        if args.captured_at is not None:
            acquisition_failure = "--captured-at is only valid with --capture"
        else:
            if args.confirm_live_read_only == LIVE_CONFIRMATION:
                transport = {
                    "network_connection_attempted": True,
                    "network_connection_opened": None,
                    "origin": COMPUTE_BOX_ORIGIN,
                    "socket_path": SOCKET_IO_PATH,
                    "inbound_events_accepted": [],
                    "application_events_emitted": [],
                    "http_methods": [],
                }
            try:
                live = capture_live_read_only(
                    confirmation=args.confirm_live_read_only or "",
                    timeout_seconds=args.timeout_seconds,
                )
                capture = live.payload
                captured_at = live.received_at_utc
                capture_digest = canonical_digest(live.payload)
                transport = live.transport
            except (QualificationError, OSError, TimeoutError, ValueError) as exc:
                acquisition_failure = str(exc)
    else:
        if args.confirm_live_read_only is not None:
            acquisition_failure = (
                "--confirm-live-read-only is invalid in offline mode"
            )
        else:
            try:
                capture, capture_digest = _read_offline_capture(args.capture)
                if captured_at is None:
                    captured_at = capture_timestamp(capture)
            except (
                QualificationError,
                FileNotFoundError,
                OSError,
                ValueError,
            ) as exc:
                acquisition_failure = str(exc)

    report_time = datetime.now(timezone.utc)
    if acquisition_failure is None:
        report = build_qualification_report(
            capture,
            mode=mode,
            captured_at=captured_at,
            now=report_time,
            max_age_seconds=args.max_age_seconds,
            transport=transport,
            capture_digest=capture_digest,
        )
    else:
        report = build_qualification_report(
            {},
            mode=mode,
            captured_at=None,
            now=report_time,
            max_age_seconds=args.max_age_seconds,
            transport=transport,
            capture_digest=capture_digest
            or hashlib.sha256(b"capture-not-acquired").hexdigest(),
        )
        report["failures"] = [acquisition_failure]
        report["status"] = "blocked"
        report["ready_for_air_demo_sync"] = False
        report.pop(REPORT_DIGEST_FIELD, None)
        report[REPORT_DIGEST_FIELD] = canonical_digest(report)

    target = write_private_report(report_path, report)
    print(f"Status: {report['status']}")
    if report["failures"]:
        print("Blocked:")
        for failure in report["failures"]:
            print(f"- {failure}")
    else:
        state = report["state"]
        print(
            "2FG7:",
            f"device_id={state['device_id']}",
            f"external_width={state['external_width_mm']['current']:.3f}mm",
            f"busy={state['busy']}",
            f"grip_detected={state['grip_detected']}",
        )
    print(f"Private report: {target}")
    return 0 if report["ready_for_air_demo_sync"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
