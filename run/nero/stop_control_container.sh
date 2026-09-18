#!/usr/bin/env bash
set -euo pipefail

NAME="${RLINF_NERO_CONTAINER_NAME:-rlinf-nero-control}"
if docker container inspect "${NAME}" >/dev/null 2>&1; then
  docker stop --time 10 "${NAME}"
else
  echo "container ${NAME} is not running"
fi
