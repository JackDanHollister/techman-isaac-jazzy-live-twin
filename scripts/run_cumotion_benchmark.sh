#!/usr/bin/env bash
set -euo pipefail

ARENA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUMOTION_ENV="${CUMOTION_ENV:-$HOME/isaac-work/envs/cumotion-1.1}"
TOOL_PROFILE="${TOOL_PROFILE:-legacy_cad_dry_run}"
MODEL_DIR="${MODEL_DIR:-$ARENA_DIR/generated/tool_profiles/$TOOL_PROFILE/cumotion}"

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
if [[ ! -f "$MODEL_DIR/tm5s_with_2fg7.xrdf" ]]; then
  echo "Prepared profile assets are missing: $MODEL_DIR" >&2
  echo "Run TOOL_PROFILE=$TOOL_PROFILE scripts/setup_cumotion_benchmark.sh first." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "$CUMOTION_ENV/bin/python" scripts/run_cumotion_benchmark.py \
  --urdf "$MODEL_DIR/tm5s_with_2fg7.urdf" \
  --xrdf "$MODEL_DIR/tm5s_with_2fg7.xrdf" \
  "$@"
