#!/usr/bin/env bash
# Run inside the RLinf training container after the compact dataset is copied in.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${NERO_RIGHT_DATASET:?Set NERO_RIGHT_DATASET to the compact LeRobot dataset directory}"
: "${NERO_PI05_CHECKPOINT:?Set NERO_PI05_CHECKPOINT to the π0.5 checkpoint directory}"
test -f "${NERO_RIGHT_DATASET}/meta/info.json" || { echo "Missing LeRobot meta/info.json" >&2; exit 1; }
test -f "${NERO_RIGHT_DATASET}/norm_stats.json" || { echo "Missing norm_stats.json" >&2; exit 1; }
test -d "${NERO_PI05_CHECKPOINT}" || { echo "Checkpoint directory missing" >&2; exit 1; }

export EMBODIED_PATH="${ROOT}/examples/sft"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

exec python "${ROOT}/examples/sft/train_vla_sft.py" \
  --config-path "${EMBODIED_PATH}/config" \
  --config-name nero_right_rlt_stage1_sft_openpi_pi05 "$@"
