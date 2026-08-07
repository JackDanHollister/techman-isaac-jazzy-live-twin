#!/usr/bin/env python3
"""Validate a blocked offline commissioning template; never contact Watson."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ARENA_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ARENA_DIR))

from pin_axis_3d_sim.controller_commissioning import (  # noqa: E402
    load_offline_commissioning_manifest,
    validate_offline_commissioning_manifest,
)


DEFAULT_TEMPLATE = (
    ARENA_DIR / "config" / "watson_controller_commissioning_offline_template.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="offline JSON template to read and validate",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest = load_offline_commissioning_manifest(args.manifest)
        result = validate_offline_commissioning_manifest(manifest)
    except (OSError, ValueError) as exc:
        print(f"OFFLINE TEMPLATE INVALID: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    print("OFFLINE TEMPLATE VALID; CONTROLLER CONFIGURATION NOT READY OR APPLIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
