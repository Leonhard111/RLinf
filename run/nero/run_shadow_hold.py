#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Send measured-position hold chunks to a shadow-only Nero bridge."""

import argparse
import statistics
import time

import numpy as np

from rlinf.envs.realworld.nero.contract import ACTION_HORIZON, ControlMode
from rlinf.envs.realworld.nero.nero_client import NeroRobotClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    client = NeroRobotClient(args.endpoint, request_timeout_s=args.timeout)
    if client.capabilities.mode is not ControlMode.SHADOW:
        raise RuntimeError("shadow acceptance refuses a publish-mode bridge")
    timings: list[float] = []
    try:
        observation = client.reset()
        for _ in range(args.steps):
            actions = np.repeat(
                observation.puppet_state[None, :], ACTION_HORIZON, axis=0
            )
            started = time.perf_counter()
            observation = client.step(actions)
            timings.append((time.perf_counter() - started) * 1000.0)
    finally:
        client.close()
    print(
        "shadow hold passed: "
        f"steps={args.steps} median_ms={statistics.median(timings):.3f} "
        f"max_ms={max(timings):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
