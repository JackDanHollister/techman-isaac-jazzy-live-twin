#!/usr/bin/env bash
set -euo pipefail

ARENA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUMOTION_ENV="${CUMOTION_ENV:-$HOME/isaac-work/envs/cumotion-1.1}"
TOOL_PROFILE="${TOOL_PROFILE:-legacy_cad_dry_run}"
PROFILE_DIR="$ARENA_DIR/generated/tool_profiles/$TOOL_PROFILE"
URDF_PATH="$PROFILE_DIR/tm5s_with_2fg7.urdf"
METADATA_PATH="$PROFILE_DIR/tm5s_with_2fg7_metadata.json"
MODEL_DIR="$PROFILE_DIR/cumotion"

if [[ ! -x "$CUMOTION_ENV/bin/python" ]]; then
  echo "Missing cuMotion environment: $CUMOTION_ENV" >&2
  echo "Install standalone NVIDIA cuMotion 1.1.0 before running this setup." >&2
  exit 1
fi
if ! "$CUMOTION_ENV/bin/python" -c \
  "import cumotion; assert cumotion.__version__ == '1.1.0', cumotion.__version__"; then
  echo "Expected standalone cuMotion 1.1.0 in: $CUMOTION_ENV" >&2
  exit 2
fi

set +u
source /opt/ros/jazzy/setup.bash
if [[ -f "$HOME/tm2_ws_apt/install/setup.bash" ]]; then
  source "$HOME/tm2_ws_apt/install/setup.bash"
elif [[ -f "$HOME/tm2_ws/install/setup.bash" ]]; then
  source "$HOME/tm2_ws/install/setup.bash"
else
  echo "Could not find a built Techman ROS workspace." >&2
  exit 1
fi
set -u

cd "$ARENA_DIR"
/usr/bin/python3 scripts/build_tm5s_2fg7_urdf.py \
  --tool-profile "$TOOL_PROFILE" \
  --finger-configuration inwards \
  --finger-joints fixed \
  --output "$URDF_PATH" \
  --metadata "$METADATA_PATH"

"$CUMOTION_ENV/bin/python" scripts/prepare_cumotion_assets.py \
  --input "$URDF_PATH" \
  --output-dir "$MODEL_DIR"

check_urdf "$MODEL_DIR/tm5s_with_2fg7.urdf" >/dev/null
"$CUMOTION_ENV/bin/python" -c \
  "import cumotion,sys; r=cumotion.load_robot_from_file(sys.argv[1],sys.argv[2]); assert r.num_cspace_coords() == 6; print('cuMotion robot load: PASS')" \
  "$MODEL_DIR/tm5s_with_2fg7.xrdf" \
  "$MODEL_DIR/tm5s_with_2fg7.urdf"
