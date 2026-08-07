#!/usr/bin/env bash
set -euo pipefail

ARENA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_ENV="${ISAAC_ENV:-$HOME/isaac-work/envs/isaac-sim-6.0}"

if [[ ! -x "$ISAAC_ENV/bin/python" ]]; then
  echo "Missing Isaac Sim environment: $ISAAC_ENV" >&2
  exit 1
fi
if ! "$ISAAC_ENV/bin/python" -c \
  "import importlib.metadata as m; assert m.version('isaacsim') == '6.0.1.0'"; then
  echo "Expected Isaac Sim 6.0.1 in: $ISAAC_ENV" >&2
  exit 2
fi

cd "$ARENA_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "$ISAAC_ENV/bin/python" scripts/run_isaac_synthetic_pick.py "$@"
