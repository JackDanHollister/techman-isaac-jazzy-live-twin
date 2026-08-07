#!/usr/bin/env bash
set -euo pipefail

ARENA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_ENV="${ISAAC_ENV:-$HOME/isaac-work/envs/isaac-sim-6.0}"
CONFIG="$ARENA_DIR/config/isaac_multi_pin_verticalization.yaml"
USD="$ARENA_DIR/generated/isaac/6.0.1-watson-qc-10mm-grasp/tm5s_with_2fg7/tm5s_with_2fg7.usda"

if [[ ! -x "$ISAAC_ENV/bin/python" ]]; then
  echo "Missing Isaac Sim environment: $ISAAC_ENV" >&2
  exit 1
fi

if ! "$ISAAC_ENV/bin/python" -c '
import importlib.metadata as metadata
import sys

expected = {
    "isaacsim": "6.0.1.0",
    "isaacsim-asset": "6.0.1.0",
    "isaacsim-core": "6.0.1.0",
}
actual = {name: metadata.version(name) for name in expected}
if sys.version_info[:2] != (3, 12) or actual != expected:
    raise SystemExit(
        f"Expected Python 3.12 and {expected}; found Python {sys.version.split()[0]} and {actual}"
    )
'; then
  echo "Isaac Sim environment does not match the multi-pin demo: $ISAAC_ENV" >&2
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

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing multi-pin config: $CONFIG" >&2
  exit 4
fi
if [[ ! -f "$USD" ]]; then
  echo "Missing articulated multi-pin USD: $USD" >&2
  echo "Run scripts/setup_isaac_grasp_cycle_asset.sh first." >&2
  exit 5
fi

unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION AMENT_PREFIX_PATH COLCON_PREFIX_PATH || true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ARENA_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$ARENA_DIR"
exec "$ISAAC_ENV/bin/python" scripts/run_isaac_multi_pin_verticalization.py "$@"
