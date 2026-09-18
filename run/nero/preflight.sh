#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${RLINF_REPO_ROOT:-/home/agilex/fjh/workbench/RLinf}"
STATE_ROOT="${RLINF_NERO_STATE_ROOT:-/home/agilex/.local/share/rlinf-nero}"
IMAGE="${RLINF_NERO_IMAGE:-rlinf/rlinf:agentic-rlinf0.4-maniskill_libero-nero-control}"
RAY_HEAD_HOST="${RLINF_RAY_HEAD_HOST:-192.168.50.102}"
RAY_HEAD_PORT="${RLINF_RAY_HEAD_PORT:-6379}"
EXPECTED_RAY_VERSION="${RLINF_EXPECTED_RAY_VERSION:-2.56.1}"

failures=0
check() {
  local label="$1"
  shift
  if "$@"; then
    printf 'PASS  %s\n' "${label}"
  else
    printf 'FAIL  %s\n' "${label}" >&2
    failures=$((failures + 1))
  fi
}

check "RLinf checkout exists" test -d "${REPO_ROOT}/.git"
check "RLinf checkout is readable" test -r "${REPO_ROOT}/rlinf/__init__.py"
check "Docker daemon is accessible" docker info
check "RLinf control image exists" docker image inspect "${IMAGE}"
mkdir -p "${STATE_ROOT}/logs" "${STATE_ROOT}/state"
check "runtime directories are writable" test -w "${STATE_ROOT}/logs"
check "Ray head TCP port is reachable" timeout 3 bash -c "</dev/tcp/${RAY_HEAD_HOST}/${RAY_HEAD_PORT}"

container_check=(
  docker run --rm --network host --shm-size 20g
  --mount "type=bind,src=${REPO_ROOT},dst=/workspace/RLinf,readonly"
  --workdir /workspace/RLinf --entrypoint bash "${IMAGE}" -lc
  "source switch_env openpi && PYTHONPATH=/workspace/RLinf python -c 'import cv2,gymnasium,msgpack,numpy,ray,rlinf,torch,websockets,zmq; assert ray.__version__ == \"${EXPECTED_RAY_VERSION}\"; print(ray.__version__, torch.__version__)'"
)
check "container imports and Ray version" "${container_check[@]}"

if [[ -f /opt/ros/humble/setup.bash ]]; then
  check "host ROS2 Humble is installed" bash -lc 'source /opt/ros/humble/setup.bash && ros2 --help >/dev/null'
else
  printf 'FAIL  host ROS2 Humble is installed\n' >&2
  failures=$((failures + 1))
fi

for topic in /nero_left/control/joint_states /nero_right/control/joint_states; do
  info="$(bash -lc "source /opt/ros/humble/setup.bash >/dev/null 2>&1; timeout 3 ros2 topic info '${topic}' 2>/dev/null" || true)"
  publishers="$(sed -n 's/^Publisher count: //p' <<<"${info}")"
  if [[ -z "${publishers}" ]]; then
    printf 'WARN  ROS graph unavailable for %s (no publisher was created)\n' "${topic}"
  elif [[ "${publishers}" == "0" ]]; then
    printf 'PASS  %s has zero publishers\n' "${topic}"
  else
    printf 'FAIL  %s has %s publishers\n' "${topic}" "${publishers}" >&2
    failures=$((failures + 1))
  fi
done

printf 'repo_commit=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
printf 'image_id=%s\n' "$(docker image inspect --format '{{.Id}}' "${IMAGE}")"
printf 'failure_count=%d\n' "${failures}"
exit "${failures}"
