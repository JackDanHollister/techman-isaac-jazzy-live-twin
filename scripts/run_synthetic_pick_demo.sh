#!/usr/bin/env bash
set -euo pipefail

ARENA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_CONFIG="${TASK_CONFIG:-$ARENA_DIR/config/synthetic_pick_task.yaml}"
SCENE_PYTHON="${SCENE_PYTHON:-python3}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_DIR:-$ARENA_DIR/outputs/synthetic_pick/${TIMESTAMP}}"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to reuse synthetic-pick output directory: $OUTPUT_DIR" >&2
  exit 1
fi
if [[ ! -f "$TASK_CONFIG" ]]; then
  echo "Missing synthetic-pick task config: $TASK_CONFIG" >&2
  exit 1
fi

read -r SEED PIN_COUNT TRAY_X TRAY_Y TRAY_Z FRAME_ID < <(
  "$SCENE_PYTHON" -c \
    'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); p=c["tray_center_xyz"]; print(c["seed"],c["pin_count"],*p,c["frame_id"])' \
    "$TASK_CONFIG"
)

cd "$ARENA_DIR"
"$SCENE_PYTHON" scripts/run_pin_axis_demo.py \
  --seed "$SEED" \
  --pins "$PIN_COUNT" \
  --tray-center-x "$TRAY_X" \
  --tray-center-y "$TRAY_Y" \
  --tray-center-z "$TRAY_Z" \
  --frame-id "$FRAME_ID" \
  --output "$OUTPUT_DIR"

./scripts/run_synthetic_pick_plan.sh \
  "$OUTPUT_DIR/result.json" \
  --task-config "$TASK_CONFIG" \
  --output "$OUTPUT_DIR/synthetic_pick_plan.json"

echo "Synthetic scene and plan: $OUTPUT_DIR"
exec ./scripts/run_isaac_synthetic_pick.sh \
  "$OUTPUT_DIR/synthetic_pick_plan.json" \
  --report "$OUTPUT_DIR/isaac_report.json" \
  "$@"
