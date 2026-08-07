#!/usr/bin/env bash
set -euo pipefail

ARENA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUMOTION_ENV="${CUMOTION_ENV:-$HOME/isaac-work/envs/cumotion-1.1}"

if [[ ! -x "$CUMOTION_ENV/bin/python" ]]; then
  echo "Missing cuMotion environment: $CUMOTION_ENV" >&2
  exit 1
fi
if ! "$CUMOTION_ENV/bin/python" -c \
  "import cumotion; assert cumotion.__version__ == '1.1.0', cumotion.__version__"; then
  echo "Expected standalone cuMotion 1.1.0 in: $CUMOTION_ENV" >&2
  exit 2
fi

cd "$ARENA_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "$CUMOTION_ENV/bin/python" scripts/plan_synthetic_pick.py "$@"
