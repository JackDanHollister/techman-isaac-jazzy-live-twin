#!/usr/bin/env bash
set -euo pipefail

ARENA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_ENV="${ISAAC_ENV:-$HOME/isaac-work/envs/isaac-sim-6.0}"

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
  echo "Isaac Sim environment does not match the visual demo: $ISAAC_ENV" >&2
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

if [[ ! -f "$ARENA_DIR/generated/isaac/6.0.1/tm5s_with_2fg7/tm5s_with_2fg7.usda" ]]; then
  echo "The imported TM5S USD is missing; run scripts/run_isaac_import.sh first." >&2
  exit 4
fi

unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION AMENT_PREFIX_PATH COLCON_PREFIX_PATH \
  CUDA_VISIBLE_DEVICES || true
cd "$ARENA_DIR"
exec "$ISAAC_ENV/bin/python" scripts/run_isaac_visual_demo.py "$@"
