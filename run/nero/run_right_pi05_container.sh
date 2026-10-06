#!/usr/bin/env bash
# Run the two-camera, right-arm Stage 1 checkpoint in the RLinf OpenPI image.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
CHECKPOINT_DIR="${NERO_RIGHT_CHECKPOINT_DIR:-/home/rcir/Data_Disk/fjh/checkpoints/nero_right_rlt_stage1_14000}"
IMAGE="${RLINF_NERO_IMAGE:-rlinf/rlinf:agentic-rlinf0.4-maniskill_libero}"

test -f "$CHECKPOINT_DIR/full_weights.pt"
test -f "$CHECKPOINT_DIR/norm_stats.json"
docker image inspect "$IMAGE" >/dev/null

tty_args=()
if [[ -t 0 && -t 1 ]]; then
  tty_args=(-it)
fi

exec docker run --rm "${tty_args[@]}" \
  --gpus all --network host --shm-size=20g \
  --mount "type=bind,src=${REPO_ROOT},dst=/workspace/RLinf,readonly" \
  --mount "type=bind,src=${CHECKPOINT_DIR},dst=/workspace/checkpoint,readonly" \
  --workdir /workspace/RLinf --entrypoint bash "$IMAGE" \
  -lc 'source switch_env openpi && export PYTHONPATH=/workspace/RLinf:${PYTHONPATH:-} && exec python run/nero/run_right_pi05_live.py "$@"' \
  bash "$@"
