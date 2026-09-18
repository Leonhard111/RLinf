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

"""Run the existing JAX π0.5 service over fixed observations, without ROS."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HOME = Path.home()
VLASTOP_ROOT = HOME / "pi05" / "vlastop"


@dataclass(frozen=True)
class Snapshot:
    """Minimal CameraSnapshot-compatible fixed sample."""

    images: tuple[np.ndarray, np.ndarray, np.ndarray]
    ages_ms: tuple[float, float, float]
    skew_ms: float


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("observations", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--output", type=Path, default=Path("baseline_pi05_actions.npz")
    )
    parser.add_argument("--log", type=Path, default=Path("baseline_pi05_dry_run.json"))
    args = parser.parse_args()

    manifest_path = args.observations / "observations_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompt = str(manifest["prompt"])
    sys.path.insert(0, str(VLASTOP_ROOT))
    from vlastop_robot.policy_client import Pi05PolicyClient

    client = Pi05PolicyClient(prompt, host=args.host, port=args.port)
    actions = []
    inference_ms = []
    total_ms = []
    try:
        for entry in manifest["samples"]:
            with np.load(
                args.observations / entry["path"], allow_pickle=False
            ) as sample:
                puppet = np.asarray(sample["puppet_state"], dtype=np.float32)
                snapshot = Snapshot(
                    images=(
                        sample["cam_high"],
                        sample["cam_right_wrist"],
                        sample["cam_left_wrist"],
                    ),
                    ages_ms=tuple(float(value) for value in entry["camera_ages_ms"]),
                    skew_ms=float(entry["camera_skew_ms"]),
                )
                result = client.infer(puppet, snapshot)
            if (
                result.actions.shape != (50, 16)
                or not np.isfinite(result.actions).all()
            ):
                raise RuntimeError("π0.5 returned an invalid action chunk")
            actions.append(result.actions)
            inference_ms.append(result.inference_ms)
            total_ms.append(result.total_ms)
    finally:
        client.close()

    action_array = np.asarray(actions, dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, actions=action_array)
    timings = {
        "schema_version": "nero-pi05-baseline/1.0.0",
        "created_at": time.strftime("%Y%m%dT%H%M%S%z"),
        "publisher_created": False,
        "sample_count": int(action_array.shape[0]),
        "action_shape_per_sample": [50, 16],
        "inference_ms": {
            "min": min(inference_ms),
            "median": statistics.median(inference_ms),
            "max": max(inference_ms),
        },
        "total_ms": {
            "min": min(total_ms),
            "median": statistics.median(total_ms),
            "max": max(total_ms),
        },
        "actions_path": str(args.output.resolve()),
    }
    args.log.write_text(json.dumps(timings, indent=2), encoding="utf-8")
    print(args.output)
    print(args.log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
