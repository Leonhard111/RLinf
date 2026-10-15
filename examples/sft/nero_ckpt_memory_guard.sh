#!/usr/bin/env bash
# Stop a named training container before physical or parent-cgroup memory is exhausted.
set -euo pipefail

container_name="${1:?usage: nero_ckpt_memory_guard.sh CONTAINER_NAME CSV_PATH}"
csv_path="${2:?usage: nero_ckpt_memory_guard.sh CONTAINER_NAME CSV_PATH}"
host_available_floor_kib="${HOST_AVAILABLE_FLOOR_KIB:-12582912}" # 12 GiB
parent_used_ceiling_bytes="${PARENT_USED_CEILING_BYTES:-88000000000}" # 88 GB of 96 GB

printf 'timestamp,host_available_kib,parent_used_bytes,container_used_bytes\n' > "${csv_path}"

wait_start="${SECONDS}"
while ! docker inspect -f '{{.State.Running}}' "${container_name}" 2>/dev/null | grep -qx true; do
  if (( SECONDS - wait_start >= 180 )); then
    echo "Container ${container_name} did not start within 180 seconds" >&2
    exit 1
  fi
  sleep 1
done

failures=0
while docker inspect -f '{{.State.Running}}' "${container_name}" 2>/dev/null | grep -qx true; do
  if ! metrics="$(docker exec "${container_name}" sh -c \
    "awk '/MemAvailable:/ {print \$2}' /proc/meminfo; cat /sys/fs/cgroup/memory.current" 2>/dev/null)"; then
    failures=$((failures + 1))
    if (( failures >= 3 )); then
      echo "Memory monitor failed three times; stopping ${container_name}" >&2
      docker kill "${container_name}" >/dev/null 2>&1 || true
      exit 2
    fi
    sleep 1
    continue
  fi
  failures=0
  host_available_kib="$(sed -n '1p' <<< "${metrics}")"
  container_used_bytes="$(sed -n '2p' <<< "${metrics}")"
  parent_used_bytes="$(cat /sys/fs/cgroup/memory.current)"
  printf '%(%s)T,%s,%s,%s\n' -1 "${host_available_kib}" \
    "${parent_used_bytes}" "${container_used_bytes}" >> "${csv_path}"

  if (( host_available_kib < host_available_floor_kib ||
        parent_used_bytes > parent_used_ceiling_bytes )); then
    echo "Memory safety floor reached; stopping ${container_name}" >&2
    echo "host_available_kib=${host_available_kib} parent_used_bytes=${parent_used_bytes} container_used_bytes=${container_used_bytes}" >&2
    docker kill "${container_name}" >/dev/null 2>&1 || true
    exit 2
  fi
  sleep 1
done
