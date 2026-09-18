#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Strictly load and benchmark an OpenPI_RLinf π0.5 checkpoint for Nero."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from rlinf.envs.realworld.nero.contract import (
    ACTION_HORIZON,
    GRIPPER_RANGE_M,
    JOINT_LOWER_RAD,
    JOINT_UPPER_RAD,
    MODEL_ACTION_DIM,
    PHYSICAL_ACTION_DIM,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model_config(
    model_path: Path,
    norm_stats_path: Path,
    *,
    precision: str = "bf16",
    num_steps: int = 5,
):
    from omegaconf import OmegaConf

    return OmegaConf.create({
        "model_type": "openpi_rlinf",
        "model_path": str(model_path),
        "precision": precision,
        "pi05": True,
        "num_action_chunks": ACTION_HORIZON,
        "action_dim": PHYSICAL_ACTION_DIM,
        "use_proprio": True,
        "num_steps": int(num_steps),
        "add_value_head": False,
        "openpi_data": {"norm_stats_path": str(norm_stats_path)},
        "openpi": {
            "task": "eval",
            "config_name": "pi05_nero",
            "num_images_in_input": 3,
            "action_chunk": ACTION_HORIZON,
            "action_env_dim": PHYSICAL_ACTION_DIM,
            "model_action_dim": MODEL_ACTION_DIM,
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
            "max_token_len": 200,
            "discrete_state_input": True,
        },
    })


def make_observation(prompt: str, *, height: int = 480, width: int = 640) -> dict:
    return {
        "states": np.zeros((1, PHYSICAL_ACTION_DIM), dtype=np.float32),
        "main_images": np.zeros((1, height, width, 3), dtype=np.uint8),
        "wrist_images": np.zeros((1, 2, height, width, 3), dtype=np.uint8),
        "task_descriptions": [prompt],
    }


def validate_action_output(actions, *, check_hard_limits: bool = True) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
    except ImportError:
        pass
    result = np.asarray(actions, dtype=np.float32)
    expected = (1, ACTION_HORIZON, PHYSICAL_ACTION_DIM)
    if result.shape != expected:
        raise ValueError(f"expected action shape {expected}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("π0.5 produced NaN/Inf")
    if check_hard_limits:
        for offset in (0, 8):
            joints = result[..., offset : offset + 7]
            if np.any(joints < JOINT_LOWER_RAD) or np.any(joints > JOINT_UPPER_RAD):
                raise ValueError("π0.5 produced a joint outside Nero hard limits")
            gripper = result[..., offset + 7]
            if np.any(gripper < GRIPPER_RANGE_M[0]) or np.any(
                gripper > GRIPPER_RANGE_M[1]
            ):
                raise ValueError("π0.5 produced a gripper outside Nero hard limits")
    return result


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("pi05_nero_verify_manifest.json")
    )
    args = parser.parse_args()
    weights = args.model / "model.safetensors"
    for path in (weights, args.norm_stats):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if min(args.warmup, args.runs, args.num_steps) < 1:
        parser.error("warmup, runs and num-steps must be positive")

    import torch

    from rlinf.models.embodiment.openpi_rlinf import get_model

    cfg = build_model_config(
        args.model,
        args.norm_stats,
        precision=args.precision,
        num_steps=args.num_steps,
    )
    model = get_model(cfg).to(args.device).eval()  # strict checkpoint load
    if model.model.action_dim != MODEL_ACTION_DIM:
        raise ValueError(f"model action dim is {model.model.action_dim}, expected 32")
    if model.model.action_horizon != ACTION_HORIZON:
        raise ValueError(
            f"model horizon is {model.model.action_horizon}, expected {ACTION_HORIZON}"
        )
    observation = make_observation(args.prompt)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)

    def infer_once():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        started = time.perf_counter()
        actions, details = model.predict_action_batch(observation, rng=generator)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        physical = validate_action_output(actions)
        model_action = details["forward_inputs"]["model_action"].reshape(
            1, ACTION_HORIZON, MODEL_ACTION_DIM
        )
        if not torch.isfinite(model_action).all():
            raise ValueError("internal 50x32 model action contains NaN/Inf")
        return elapsed_ms, physical

    for _ in range(args.warmup):
        infer_once()
    latencies = []
    last_actions = None
    for _ in range(args.runs):
        latency, last_actions = infer_once()
        latencies.append(latency)
    ordered = sorted(latencies)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
        return float(ordered[index])

    manifest = {
        "schema_version": "nero-pi05-checkpoint/1.0.0",
        "git_commit": git_commit(),
        "model_path": str(args.model.resolve()),
        "model_sha256": sha256_file(weights),
        "norm_stats_path": str(args.norm_stats.resolve()),
        "norm_stats_sha256": sha256_file(args.norm_stats),
        "strict_load": True,
        "model_action_shape": [ACTION_HORIZON, MODEL_ACTION_DIM],
        "physical_action_shape": [ACTION_HORIZON, PHYSICAL_ACTION_DIM],
        "seed": args.seed,
        "prompt": args.prompt,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "latency_ms": {
            "min": min(latencies),
            "median": statistics.median(latencies),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": max(latencies),
        },
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(args.device))
            if args.device.startswith("cuda")
            else 0
        ),
        "action_min": np.min(last_actions, axis=(0, 1)).tolist(),
        "action_max": np.max(last_actions, axis=(0, 1)).tolist(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.manifest.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
