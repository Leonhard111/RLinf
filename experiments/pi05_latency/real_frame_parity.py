#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Compare native and compiled pi0.5 on saved two-camera Nero frames."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from experiments.pi05_latency.runtime import compile_and_warm_model
from rlinf.envs.realworld.nero.right_arm_adapter import right_model_observation
from rlinf.models.embodiment.openpi_rlinf import get_model
from run.nero.run_right_pi05_live import model_config, offline_observation


def load_observations(dataset: Path, frames: Path, norm_stats: Path) -> list[dict]:
    parquet = dataset / "data/chunk-000/episode_000000.parquet"
    states = pq.read_table(parquet, columns=["observation.state"]).column(0).to_pylist()
    front_paths = sorted(frames.glob("front_*.png"))
    wrist_paths = sorted(frames.glob("wrist_*.png"))
    if not front_paths or len(front_paths) != len(wrist_paths):
        raise ValueError("Expected matched saved front and wrist frame samples")
    observations = []
    for index, (front_path, wrist_path) in enumerate(zip(front_paths, wrist_paths)):
        sample = offline_observation(
            norm_stats, "Stack the red square on top of the green square."
        )
        sample.images = {
            "cam_high": np.asarray(Image.open(front_path).convert("RGB")),
            "cam_right_wrist": np.asarray(Image.open(wrist_path).convert("RGB")),
        }
        sample.puppet_state[8:] = np.asarray(states[index * 30], dtype=np.float32)
        observations.append(right_model_observation(sample))
    return observations


def run_action(
    model, model_input: dict, noise: torch.Tensor
) -> tuple[np.ndarray, float]:
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    with torch.inference_mode():
        action = model.predict_action_batch(model_input, noise=noise)[0]
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    return action.detach().float().cpu().numpy().copy(), elapsed_ms


def run_features(model, model_input: dict, seed: int) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    with torch.inference_mode():
        features = model.extract_rlt_obs(model_input)
    return {
        key: value.detach().float().cpu().numpy().copy()
        for key, value in features.items()
    }


def summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("/workspace/checkpoint/full_weights.pt")
    )
    parser.add_argument(
        "--norm-stats", type=Path, default=Path("/workspace/checkpoint/norm_stats.json")
    )
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/dataset"))
    parser.add_argument(
        "--frames",
        type=Path,
        default=Path("experiments/pi05_latency/results/real_frames"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    observations = load_observations(args.dataset, args.frames, args.norm_stats)
    model = (
        get_model(model_config(args.checkpoint, args.norm_stats, "bf16", 5))
        .to("cuda")
        .eval()
    )
    generator = torch.Generator(device="cuda").manual_seed(20261013)
    # Match deployment: sample noise under inference_mode, avoiding a separate
    # ADInplaceOrView dispatch-key specialization in the compiled suffix.
    with torch.inference_mode():
        noises = [
            torch.randn(
                (1, model.model.action_horizon, model.model.action_dim),
                device="cuda",
                generator=generator,
            )
            for _ in observations
        ]
    for _ in range(3):
        run_action(model, observations[0], noises[0])
    with torch.inference_mode():
        for _ in range(3):
            model.extract_rlt_obs(observations[0])
    native_actions = [
        run_action(model, obs, noise) for obs, noise in zip(observations, noises)
    ]
    native_features = [
        run_features(model, obs, 20261013 + index)
        for index, obs in enumerate(observations)
    ]

    compile_warmup_s = compile_and_warm_model(
        model, observations[0], compile_prefix=True
    )
    with torch.inference_mode():
        for _ in range(3):
            model.extract_rlt_obs(observations[0])
    compiled_actions = [
        run_action(model, obs, noise) for obs, noise in zip(observations, noises)
    ]
    compiled_features = [
        run_features(model, obs, 20261013 + index)
        for index, obs in enumerate(observations)
    ]

    action_diffs = [
        np.abs(compiled[0] - native[0])
        for native, compiled in zip(native_actions, compiled_actions)
    ]
    feature_diffs = [
        np.abs(compiled["z_rl"] - native["z_rl"])
        for native, compiled in zip(native_features, compiled_features)
    ]
    relative_feature_l2 = [
        float(
            np.linalg.norm(compiled["z_rl"] - native["z_rl"])
            / max(np.linalg.norm(native["z_rl"]), 1e-12)
        )
        for native, compiled in zip(native_features, compiled_features)
    ]
    result = {
        "samples": len(observations),
        "source": "saved Nero smoke-video episode 15, 1 fps frame samples; task prompt is the deployed red-on-green prompt",
        "native_model_ms": summary([item[1] for item in native_actions]),
        "native_model_ms_by_frame": [item[1] for item in native_actions],
        "compiled_model_ms": summary([item[1] for item in compiled_actions]),
        "compiled_model_ms_by_frame": [item[1] for item in compiled_actions],
        "compile_warmup_s": compile_warmup_s,
        "action_mean_abs": float(np.mean([item.mean() for item in action_diffs])),
        "action_max_abs": float(max(item.max() for item in action_diffs)),
        "action_joint_max_abs": float(
            max(item[..., :7].max() for item in action_diffs)
        ),
        "action_gripper_max_abs": float(
            max(item[..., 7:].max() for item in action_diffs)
        ),
        "z_rl_mean_abs": float(np.mean([item.mean() for item in feature_diffs])),
        "z_rl_max_abs": float(max(item.max() for item in feature_diffs)),
        "z_rl_relative_l2": summary(relative_feature_l2),
        "z_rl_native_rms": summary(
            [float(np.sqrt(np.mean(item["z_rl"] ** 2))) for item in native_features]
        ),
        "ref_chunk_max_abs": float(
            max(
                np.abs(compiled["ref_chunk"] - native["ref_chunk"]).max()
                for native, compiled in zip(native_features, compiled_features)
            )
        ),
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
