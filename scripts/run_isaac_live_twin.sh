#!/usr/bin/env bash
set -euo pipefail

ARENA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_ENV="${ISAAC_ENV:-$HOME/isaac-work/envs/isaac-sim-6.0}"
ROS_CORE="$ISAAC_ENV/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/jazzy"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PREFLIGHT_REPORT="$ARENA_DIR/outputs/watson_guarded_demo/${STAMP}_isaac_live_preflight.json"
VIEWER_REPORT="$ARENA_DIR/outputs/isaac_sim/6.0.1/${STAMP}_live_twin.json"
LOCK_PATH="/tmp/watson-isaac-live-twin.lock"

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "Another Watson Isaac live twin is already running." >&2
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
if [[ ! -f "$ARENA_DIR/reference/seven_pin/isaac/tm5s_with_2fg7/tm5s_with_2fg7.usda" ]]; then
  echo "The bundled articulated TM5S USD is missing." >&2
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
  done < <(find "$ISAAC_ENV/lib" -path '*/site-packages/isaacsim/kit/EULA_ACCEPTED' -type f 2>/dev/null)
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
  echo "Isaac Sim environment does not match the live twin: $ISAAC_ENV" >&2
  exit 3
fi

(
  set +u
  source /opt/ros/jazzy/setup.bash
  source "$HOME/tm2_ws_apt/install/setup.bash"
  set -u
  export ROS_DOMAIN_ID=219
  export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
  execution_state="$(ros2 param get /watson/move_group allow_trajectory_execution)"
  if [[ "$execution_state" != *"False"* ]]; then
    echo "Refusing live-twin commissioning while MoveIt execution is enabled." >&2
    echo "Observed: $execution_state" >&2
    exit 4
  fi
)

"$ARENA_DIR/scripts/run_watson_guarded_demo.sh" \
  --mode check \
  --report "$PREFLIGHT_REPORT"

export ROS_DISTRO=jazzy
export ROS_DOMAIN_ID=219
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONPATH="$ROS_CORE/rclpy:$ARENA_DIR"
export LD_LIBRARY_PATH="$ROS_CORE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_VERSION ROS_PYTHON_VERSION \
  CUDA_VISIBLE_DEVICES || true

echo "Starting the read-only Watson twin in Isaac Sim."
echo "Preflight: $PREFLIGHT_REPORT"
echo "Viewer report: $VIEWER_REPORT"
cd "$ARENA_DIR"
exec "$ISAAC_ENV/bin/python" scripts/run_isaac_live_twin.py \
  --preflight-report "$PREFLIGHT_REPORT" \
  --report "$VIEWER_REPORT" \
  "$@"
