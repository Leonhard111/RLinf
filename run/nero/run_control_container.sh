#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${RLINF_REPO_ROOT:-/home/agilex/fjh/workbench/RLinf}"
STATE_ROOT="${RLINF_NERO_STATE_ROOT:-/home/agilex/.local/share/rlinf-nero}"
IMAGE="${RLINF_NERO_IMAGE:-rlinf/rlinf:agentic-rlinf0.4-maniskill_libero-nero-control}"
NAME="${RLINF_NERO_CONTAINER_NAME:-rlinf-nero-control}"

mkdir -p "${STATE_ROOT}/logs" "${STATE_ROOT}/state"

exec docker run --rm -it \
  --name "${NAME}" \
  --network host \
  --shm-size 20g \
  --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=4g \
  --tmpfs /root:rw,nosuid,nodev,size=1g \
  --mount "type=bind,src=${REPO_ROOT},dst=/workspace/RLinf,readonly" \
  --mount "type=bind,src=${STATE_ROOT},dst=/workspace/nero-runtime" \
  --workdir /workspace/RLinf \
  --entrypoint bash \
  "${IMAGE}" \
  -lc 'source switch_env openpi && export PYTHONPATH=/workspace/RLinf:${PYTHONPATH:-} && exec bash'
