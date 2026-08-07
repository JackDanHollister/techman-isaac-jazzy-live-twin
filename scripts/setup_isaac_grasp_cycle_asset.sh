#!/usr/bin/env bash
set -euo pipefail

ARENA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUMOTION_ENV="${CUMOTION_ENV:-$HOME/isaac-work/envs/cumotion-1.1}"
SOURCE_DIR="$ARENA_DIR/generated/tool_profiles/watson_qc_nominal/isaac_grasp_cycle"
SOURCE_URDF="$SOURCE_DIR/tm5s_with_2fg7_articulated.urdf"
SOURCE_METADATA="$SOURCE_DIR/tm5s_with_2fg7_articulated_metadata.json"
STAGED_DIR="$SOURCE_DIR/staged"

if [[ ! -x "$CUMOTION_ENV/bin/python" ]]; then
  echo "Missing cuMotion environment used for deterministic mesh staging: $CUMOTION_ENV" >&2
  exit 1
fi

set +u
source /opt/ros/jazzy/setup.bash
if [[ -f "$HOME/tm2_ws_apt/install/setup.bash" ]]; then
  source "$HOME/tm2_ws_apt/install/setup.bash"
elif [[ -f "$HOME/tm2_ws/install/setup.bash" ]]; then
  source "$HOME/tm2_ws/install/setup.bash"
else
  echo "Could not find a built Techman ROS workspace." >&2
  exit 2
fi
set -u

cd "$ARENA_DIR"
/usr/bin/python3 scripts/build_tm5s_2fg7_urdf.py \
  --tool-profile watson_qc_nominal \
  --finger-configuration inwards \
  --finger-joints prismatic \
  --finger-position 0.0 \
  --output "$SOURCE_URDF" \
  --metadata "$SOURCE_METADATA"

"$CUMOTION_ENV/bin/python" scripts/prepare_cumotion_assets.py \
  --asset-mode isaac_articulated \
  --input "$SOURCE_URDF" \
  --output-dir "$STAGED_DIR"

exec ./scripts/run_isaac_import.sh \
  --urdf "$STAGED_DIR/tm5s_with_2fg7.urdf" \
  --validation-profile watson_qc_articulated_2fg7 \
  --output-dir generated/isaac/6.0.1-watson-qc-10mm-grasp \
  --report outputs/isaac_sim/6.0.1/watson_qc_10mm_grasp_import_report.json
