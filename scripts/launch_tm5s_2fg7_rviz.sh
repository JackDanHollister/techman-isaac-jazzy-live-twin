#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARENA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TOOL_PROFILE="${TOOL_PROFILE:-legacy_cad_dry_run}"
PROFILE_DIR="$ARENA_DIR/generated/tool_profiles/$TOOL_PROFILE"
URDF_PATH="$PROFILE_DIR/tm5s_with_2fg7.urdf"
METADATA_PATH="$PROFILE_DIR/tm5s_with_2fg7_metadata.json"
RESULT_JSON="${1:-$ARENA_DIR/outputs/demo_seed7/result.json}"
RVIZ_CONFIG="$ARENA_DIR/config/tm5s_2fg7_pin_demo.rviz"

source_setup() {
  set +u
  source "$1"
  set -u
}

source_setup /opt/ros/jazzy/setup.bash
if [ -f "$HOME/tm2_ws_apt/install/setup.bash" ]; then
  source_setup "$HOME/tm2_ws_apt/install/setup.bash"
elif [ -f "$HOME/tm2_ws/install/setup.bash" ]; then
  source_setup "$HOME/tm2_ws/install/setup.bash"
fi

unset GTK_PATH LOCPATH GDK_PIXBUF_MODULE_FILE GDK_PIXBUF_MODULEDIR LD_LIBRARY_PATH_SNAP SNAP SNAP_DATA SNAP_COMMON

python3 "$SCRIPT_DIR/build_tm5s_2fg7_urdf.py" \
  --tool-profile "$TOOL_PROFILE" \
  --finger-configuration inwards \
  --output "$URDF_PATH" \
  --metadata "$METADATA_PATH"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

ros2 run robot_state_publisher robot_state_publisher \
  --ros-args -p robot_description:="$(< "$URDF_PATH")" &
PIDS+=("$!")

if ros2 pkg executables joint_state_publisher_gui | grep -q "joint_state_publisher_gui"; then
  ros2 run joint_state_publisher_gui joint_state_publisher_gui &
  PIDS+=("$!")
else
  ros2 run joint_state_publisher joint_state_publisher &
  PIDS+=("$!")
fi

if [ -f "$RESULT_JSON" ]; then
  /usr/bin/python3 "$SCRIPT_DIR/publish_pin_scene.py" "$RESULT_JSON" \
    --cloud-ply "$(dirname "$RESULT_JSON")/scene_cloud.ply" \
    --frame-id base &
  PIDS+=("$!")
  /usr/bin/python3 "$SCRIPT_DIR/publish_alignment_frames.py" "$RESULT_JSON" \
    --end-effector-link flange \
    --frame-id base &
  PIDS+=("$!")
fi

rviz2 -d "$RVIZ_CONFIG"
