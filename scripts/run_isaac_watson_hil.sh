#!/usr/bin/env bash
set -euo pipefail

ARENA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_ENV="${ISAAC_ENV:-$HOME/isaac-work/envs/isaac-sim-6.0}"
ROS_CORE="$ISAAC_ENV/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/jazzy"
HIL_SCRIPT="$ARENA_DIR/scripts/run_isaac_watson_hil.py"
CONFIG="$ARENA_DIR/config/isaac_multi_pin_verticalization.yaml"
ARTICULATED_USD="$ARENA_DIR/reference/seven_pin/isaac/tm5s_with_2fg7/tm5s_with_2fg7.usda"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEFAULT_REPORT="$ARENA_DIR/outputs/isaac_sim/6.0.1/${STAMP}_watson_hil.json"
LOCK_PATH="/tmp/watson-isaac-hil.lock"

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "Another Watson Isaac HIL window is already running." >&2
  exit 1
fi

if [[ ! -x "$ISAAC_ENV/bin/python" ]]; then
  echo "Missing Isaac Sim environment: $ISAAC_ENV" >&2
  exit 2
fi
if [[ ! -d "$ROS_CORE/rclpy" || ! -d "$ROS_CORE/lib" ]]; then
  echo "Isaac Sim's bundled Jazzy runtime is missing: $ROS_CORE" >&2
  exit 2
fi
if [[ ! -f "$HIL_SCRIPT" ]]; then
  echo "Missing Isaac/Watson HIL GUI: $HIL_SCRIPT" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Missing HIL source configuration: $CONFIG" >&2
  exit 2
fi
if [[ ! -f "$ARTICULATED_USD" ]]; then
  echo "Missing the bundled articulated Techman/QC/2FG7 USD: $ARTICULATED_USD" >&2
  exit 2
fi

accepted="${OMNI_KIT_ACCEPT_EULA:-}"
accepted="${accepted,,}"
if [[ "$accepted" != "y" && "$accepted" != "yes" && "$accepted" != "1" ]]; then
  marker_accepted=false
  while IFS= read -r marker; do
    if head -n 1 "$marker" | grep -Eiq '^(y|yes|1)$'; then
      marker_accepted=true
      break
    fi
  done < <(
    find "$ISAAC_ENV/lib" \
      -path '*/site-packages/isaacsim/kit/EULA_ACCEPTED' \
      -type f 2>/dev/null
  )
  if [[ "$marker_accepted" != true ]]; then
    echo "The NVIDIA Omniverse EULA must be accepted before starting Isaac Sim." >&2
    echo "After reviewing it, rerun with OMNI_KIT_ACCEPT_EULA=YES." >&2
    exit 3
  fi
fi

if ! "$ISAAC_ENV/bin/python" -c '
import importlib.metadata as metadata
import sys

expected = {
    "isaacsim": "6.0.1.0",
    "isaacsim-core": "6.0.1.0",
    "isaacsim-ros2": "6.0.1.0",
}
actual = {name: metadata.version(name) for name in expected}
if sys.version_info[:2] != (3, 12) or actual != expected:
    raise SystemExit(
        f"Expected Python 3.12 and {expected}; found Python {sys.version.split()[0]} and {actual}"
    )
'; then
  echo "Isaac Sim environment does not match the Watson HIL GUI: $ISAAC_ENV" >&2
  exit 3
fi

has_mode=false
has_report=false
for argument in "$@"; do
  case "$argument" in
    --mode|--mode=*)
      has_mode=true
      ;;
    --report|--report=*)
      has_report=true
      ;;
    --)
      break
      ;;
  esac
done

launch_args=()
if [[ "$has_mode" != true ]]; then
  launch_args+=(--mode preview)
fi
if [[ "$has_report" != true ]]; then
  launch_args+=(--report "$DEFAULT_REPORT")
fi
launch_args+=("$@")

export ROS_DISTRO=jazzy
export ROS_DOMAIN_ID=219
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONPATH="$ROS_CORE/rclpy:$ARENA_DIR"
export LD_LIBRARY_PATH="$ROS_CORE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_VERSION ROS_PYTHON_VERSION \
  CUDA_VISIBLE_DEVICES || true

echo "Starting the Isaac/Watson HIL GUI (default mode: preview)."
if [[ "$has_report" != true ]]; then
  echo "HIL report: $DEFAULT_REPORT"
fi
cd "$ARENA_DIR"
exec "$ISAAC_ENV/bin/python" "$HIL_SCRIPT" "${launch_args[@]}"
