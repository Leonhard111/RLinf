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

"""Run the phase-3 Nero dummy environment acceptance loop."""

import argparse
import time

import numpy as np

from rlinf.envs.realworld.nero.contract import ACTION_HORIZON
from rlinf.envs.realworld.nero.nero_env import NeroEnv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10_000)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    env = NeroEnv({
        "mode": "dummy",
        "image_height": 2,
        "image_width": 3,
        "max_num_steps": args.steps + 1,
    })
    state = np.asarray([0.0] * 7 + [0.05] + [0.0] * 7 + [0.05], dtype=np.float32)
    actions = np.repeat(state[None, :], ACTION_HORIZON, axis=0)
    started = time.perf_counter()
    try:
        env.reset()
        for _ in range(args.steps):
            env.step(actions)
    finally:
        env.close()
    elapsed = time.perf_counter() - started
    print(f"dummy acceptance passed: steps={args.steps} elapsed_s={elapsed:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
