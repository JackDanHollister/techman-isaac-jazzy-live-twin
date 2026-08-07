#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARENA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SEED="${1:-$(date +%s)}"
PIN_COUNT="${PIN_COUNT:-13}"
MAX_PINS="${MAX_PINS:-13}"
TRAY_X="${TRAY_X:-0.50}"
TRAY_Y="${TRAY_Y:-0.00}"
TRAY_Z="${TRAY_Z:-0.00}"
USE_RVIZ="${USE_RVIZ:-1}"
KEEP_OPEN="${KEEP_OPEN:-1}"
TOOL_PROFILE="${TOOL_PROFILE:-legacy_cad_dry_run}"
OUTPUT_DIR="$ARENA_DIR/outputs/play_seed_${SEED}"
PROFILE_DIR="$ARENA_DIR/generated/tool_profiles/$TOOL_PROFILE"
URDF_PATH="$PROFILE_DIR/tm5s_with_2fg7.urdf"
METADATA_PATH="$PROFILE_DIR/tm5s_with_2fg7_metadata.json"
TARGETS_JSON="$OUTPUT_DIR/moveit_targets_flange_all.json"
RVIZ_CONFIG="$ARENA_DIR/config/tm5s_2fg7_moveit_pin_demo.rviz"
LAUNCH_LOG="$OUTPUT_DIR/moveit_rviz_launch.log"
LAUNCH_OWN_PROCESS_GROUP=0

if [ -z "${ROS_DOMAIN_ID:-}" ]; then
  export ROS_DOMAIN_ID="$((120 + RANDOM % 100))"
else
  export ROS_DOMAIN_ID
fi

source_setup() {
  set +u
  source "$1"
  set -u
}

ros_bool() {
  case "${1,,}" in
    1|true|yes|on) echo "true" ;;
    0|false|no|off) echo "false" ;;
    *) echo "$1" ;;
  esac
}

source_setup /opt/ros/jazzy/setup.bash
if [ -f "$HOME/tm2_ws_apt/install/setup.bash" ]; then
  source_setup "$HOME/tm2_ws_apt/install/setup.bash"
elif [ -f "$HOME/tm2_ws/install/setup.bash" ]; then
  source_setup "$HOME/tm2_ws/install/setup.bash"
fi

unset GTK_PATH LOCPATH GDK_PIXBUF_MODULE_FILE GDK_PIXBUF_MODULEDIR LD_LIBRARY_PATH_SNAP SNAP SNAP_DATA SNAP_COMMON

mkdir -p "$OUTPUT_DIR"

echo "Generating random pin scene: seed=$SEED pins=$PIN_COUNT tray=[$TRAY_X, $TRAY_Y, $TRAY_Z] m"
python3 "$SCRIPT_DIR/run_pin_axis_demo.py" \
  --seed "$SEED" \
  --pins "$PIN_COUNT" \
  --frame-id base \
  --tray-center-x "$TRAY_X" \
  --tray-center-y "$TRAY_Y" \
  --tray-center-z "$TRAY_Z" \
  --output "$OUTPUT_DIR"

python3 "$SCRIPT_DIR/make_html_viewer.py" "$OUTPUT_DIR" >/dev/null

python3 "$SCRIPT_DIR/build_tm5s_2fg7_urdf.py" \
  --tool-profile "$TOOL_PROFILE" \
  --finger-configuration inwards \
  --output "$URDF_PATH" \
  --metadata "$METADATA_PATH"

python3 "$SCRIPT_DIR/export_moveit_targets.py" "$OUTPUT_DIR/result.json" \
  --target all \
  --end-effector-link flange \
  --frame-id base \
  --output "$TARGETS_JSON"

LAUNCH_PID=""
cleanup() {
  if [ -n "$LAUNCH_PID" ]; then
    if [ "$LAUNCH_OWN_PROCESS_GROUP" = "1" ]; then
      kill -INT -- "-$LAUNCH_PID" >/dev/null 2>&1 || true
      sleep 1
      kill -TERM -- "-$LAUNCH_PID" >/dev/null 2>&1 || true
    else
      kill -INT "$LAUNCH_PID" >/dev/null 2>&1 || true
      sleep 1
      kill -TERM "$LAUNCH_PID" >/dev/null 2>&1 || true
    fi
    wait "$LAUNCH_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "Starting MoveIt/RViz demo. This publishes simulated joint states only; it does not command the real robot."
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
if command -v setsid >/dev/null 2>&1; then
  setsid ros2 launch "$ARENA_DIR/launch/tm5s_2fg7_moveit_demo.launch.py" \
    urdf_path:="$URDF_PATH" \
    result_json:="$OUTPUT_DIR/result.json" \
    rviz_config:="$RVIZ_CONFIG" \
    publish_alignment:=true \
    use_rviz:="$(ros_bool "$USE_RVIZ")" >"$LAUNCH_LOG" 2>&1 &
  LAUNCH_OWN_PROCESS_GROUP=1
else
  ros2 launch "$ARENA_DIR/launch/tm5s_2fg7_moveit_demo.launch.py" \
    urdf_path:="$URDF_PATH" \
    result_json:="$OUTPUT_DIR/result.json" \
    rviz_config:="$RVIZ_CONFIG" \
    publish_alignment:=true \
    use_rviz:="$(ros_bool "$USE_RVIZ")" >"$LAUNCH_LOG" 2>&1 &
fi
LAUNCH_PID="$!"

PLAYER_ARGS=(
  "$TARGETS_JSON"
  --max-pins "$MAX_PINS"
  --move-seconds "${MOVE_SECONDS:-3.0}"
  --settle-seconds "${SETTLE_SECONDS:-0.6}"
  --motion-rate-hz "${MOTION_RATE_HZ:-60}"
  --alignment-hold-seconds "${ALIGNMENT_HOLD_SECONDS:-1.0}"
  --alignment-marker-length "${ALIGNMENT_MARKER_LENGTH:-0.65}"
)

if [ "${AUTO_PLAY:-0}" != "1" ]; then
  PLAYER_ARGS+=(--manual-step)
fi
if [ "$KEEP_OPEN" = "1" ]; then
  PLAYER_ARGS+=(--hold-open)
fi

echo ""
echo "RViz will show:"
echo "  - synthetic scanner cloud"
echo "  - red detected pin axes"
echo "  - blue gripper centerlines"
echo "  - long orange/blue live alignment guides that turn green when aligned"
echo "  - a large live alignment point-cloud overlay that turns green before moving down"
echo "  - TM5S + OnRobot 2FG7 moving through IK-derived joint-state replay"
echo ""
echo "Targets: $TARGETS_JSON"
echo "HTML view: $OUTPUT_DIR/viewer.html"
echo "MoveIt/RViz log: $LAUNCH_LOG"
echo ""
echo "Play control:"
if [ "${AUTO_PLAY:-0}" = "1" ]; then
  echo "  AUTO_PLAY=1, so motion starts once MoveIt is ready."
else
  echo "  Press Enter in this terminal when prompted to move to the next pin."
fi
echo ""

set +e
/usr/bin/python3 -u "$SCRIPT_DIR/play_alignment_demo.py" "${PLAYER_ARGS[@]}"
PLAYER_STATUS="$?"
set -e

echo ""
if [ "$KEEP_OPEN" = "1" ]; then
  echo "Player exited with status $PLAYER_STATUS. RViz/MoveIt is still running for inspection."
  echo "Close RViz or press Ctrl-C here to stop the demo launch."
  wait "$LAUNCH_PID"
else
  echo "Player exited with status $PLAYER_STATUS."
  echo "Stopping demo launch because KEEP_OPEN=0."
fi
exit "$PLAYER_STATUS"
