#!/usr/bin/env bash
# Host-side launcher for Nero right-arm absolute-action RLT Stage 1 on GPUs 3-4.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${RLINF_IMAGE:-rlinf/rlinf:agentic-rlinf0.4-maniskill_libero}"
DATASET_HOST="${NERO_RIGHT_DATASET_HOST:-/home/ubuntu/fjh/workbench/data/0929_red_on_green_right_lerobot_structural_pass_abs_wide20}"
CHECKPOINT_HOST="${NERO_PI05_CHECKPOINT_HOST:-/home/ubuntu/lxd_shared/pi_origin/pi05_base}"
LOGS_HOST="${ROOT}/logs"

test -f "${DATASET_HOST}/meta/info.json" || {
  echo "Missing dataset: ${DATASET_HOST}" >&2
  exit 1
}
test -f "${DATASET_HOST}/norm_stats.json" || {
  echo "Missing norm stats: ${DATASET_HOST}/norm_stats.json" >&2
  exit 1
}
test -f "${CHECKPOINT_HOST}/model.safetensors" || {
  echo "Missing checkpoint: ${CHECKPOINT_HOST}" >&2
  exit 1
}
mountpoint -q "${LOGS_HOST}" || {
  echo "Refusing to train: ${LOGS_HOST} is not the NAS mountpoint" >&2
  exit 1
}
docker image inspect "${IMAGE}" >/dev/null

echo "GPU 3-4 status before launch:"
nvidia-smi --id=3,4 \
  --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader
echo "YAML defaults: global_batch=32, per-GPU micro_batch=16, accumulation=1, absolute actions"
echo "Command-line overrides: $*"

tty_args=()
if [[ -t 0 && -t 1 ]]; then
  tty_args=(-it)
fi

exec docker run --rm "${tty_args[@]}" \
  --gpus all \
  --shm-size=20g \
  --network host \
  -v "${ROOT}:/workspace/RLinf" \
  -v "${DATASET_HOST}:/workspace/data:ro" \
  -v "${CHECKPOINT_HOST}:/workspace/pi05_base:ro" \
  -e NERO_RIGHT_DATASET=/workspace/data \
  -e NERO_PI05_CHECKPOINT=/workspace/pi05_base \
  "${IMAGE}" \
  bash -lc '
    source switch_env openpi
    cd /workspace/RLinf
    exec bash examples/sft/train_nero_right_rlt_stage1.sh "$@"
  ' -- "$@"
