#!/usr/bin/env bash
# Run the two-camera, right-arm Stage 1 checkpoint in the RLinf OpenPI image.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
#CHECKPOINT_DIR="${NERO_RIGHT_CHECKPOINT_DIR:-/home/rcir/Data_Disk/fjh/checkpoints/nero_right_rlt_stage1_14000}"
CHECKPOINT_DIR="${NERO_RIGHT_CHECKPOINT_DIR:-/home/rcir/Data_Disk/fjh/checkpoints/nero_right_rlt_stage1_20000}"
IMAGE="${RLINF_NERO_IMAGE:-rlinf/rlinf:agentic-rlinf0.4-maniskill_libero}"

test -f "$CHECKPOINT_DIR/full_weights.pt"
test -f "$CHECKPOINT_DIR/norm_stats.json"
docker image inspect "$IMAGE" >/dev/null

tty_args=()
if [[ -t 0 && -t 1 ]]; then
  tty_args=(-it)
fi

compile_cache_args=()
for arg in "$@"; do
  if [[ "$arg" == --compile-suffix || "$arg" == --compile-prefix ]]; then
    # Keep generated kernels across disposable Docker containers.  The
    # checkpoint remains read-only and the cache has its own writable mount.
    COMPILE_CACHE_DIR="${NERO_RIGHT_COMPILE_CACHE_DIR:-${CHECKPOINT_DIR}/torch_compile_cache}"
    mkdir -p "$COMPILE_CACHE_DIR/runtime_home"
    COMPILE_CACHE_DIR="$(cd "$COMPILE_CACHE_DIR" && pwd -P)"
    compile_cache_args=(
      --user "$(id -u):$(id -g)"
      --env "USER=$(id -un)"
      --env "LOGNAME=$(id -un)"
      --env HOME=/workspace/torch_compile_cache/runtime_home
      --mount "type=bind,src=${COMPILE_CACHE_DIR},dst=/workspace/torch_compile_cache"
      --env TORCHINDUCTOR_CACHE_DIR=/workspace/torch_compile_cache
      --env TORCHINDUCTOR_FX_GRAPH_CACHE=1
    )
    break
  fi
done

exec docker run --rm "${tty_args[@]}" \
  --gpus all --network host --shm-size=20g \
  --mount "type=bind,src=${REPO_ROOT},dst=/workspace/RLinf,readonly" \
  --mount "type=bind,src=${CHECKPOINT_DIR},dst=/workspace/checkpoint,readonly" \
  "${compile_cache_args[@]}" \
  --workdir /workspace/RLinf --entrypoint bash "$IMAGE" \
  -lc 'source switch_env openpi && set -e
    # The image bundles this tokenizer under the root home. A host-UID
    # compile container has a separate persistent home, so seed it locally.
    if [[ -n "${TORCHINDUCTOR_CACHE_DIR:-}" ]]; then
      bundled_tokenizer=/opt/assets/.cache/openpi/big_vision/paligemma_tokenizer.model
      cached_tokenizer="${HOME}/.cache/openpi/big_vision/paligemma_tokenizer.model"
      if [[ -r "$bundled_tokenizer" && ! -s "$cached_tokenizer" ]]; then
        mkdir -p "${cached_tokenizer%/*}"
        cp "$bundled_tokenizer" "${cached_tokenizer}.tmp.$$"
        mv -n "${cached_tokenizer}.tmp.$$" "$cached_tokenizer"
        rm -f "${cached_tokenizer}.tmp.$$"
      fi
    fi
    export PYTHONPATH=/workspace/RLinf:${PYTHONPATH:-}
    exec python run/nero/run_right_pi05_live.py "$@"' \
  bash "$@"
