#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Offline probe of the frozen Stage 1 feature path consumed by RLT Stage 2."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pi05_latency.runtime import compile_and_warm_model
from rlinf.envs.realworld.nero.right_arm_adapter import right_model_observation
from rlinf.models.embodiment.openpi_rlinf import get_model
from run.nero.run_right_pi05_live import model_config, offline_observation


def measure(model, model_input, repeats: int) -> dict[str, float]:
    times = []
    with torch.inference_mode():
        for _ in range(repeats):
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            model.extract_rlt_obs(model_input)
            torch.cuda.synchronize()
            times.append((time.perf_counter_ns() - started) / 1e6)
    ordered = sorted(times)
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
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--suffix-only",
        action="store_true",
        help="preserve native visual/RLT features and compile only the action suffix",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    model = (
        get_model(model_config(args.checkpoint, args.norm_stats, "bf16", 5))
        .to("cuda")
        .eval()
    )
    observation = offline_observation(
        args.norm_stats, "Stack the red square on top of the green square."
    )
    model_input = right_model_observation(observation)
    with torch.inference_mode():
        for _ in range(3):
            model.extract_rlt_obs(model_input)
    native_ms = measure(model, model_input, args.repeats)
    torch.manual_seed(20261013)
    with torch.inference_mode():
        native = {
            key: value.detach().float().cpu().numpy().copy()
            for key, value in model.extract_rlt_obs(model_input).items()
        }

    cold_start_s = compile_and_warm_model(
        model, model_input, compile_prefix=not args.suffix_only
    )
    with torch.inference_mode():
        for _ in range(3):
            model.extract_rlt_obs(model_input)
    compiled_ms = measure(model, model_input, args.repeats)
    torch.manual_seed(20261013)
    with torch.inference_mode():
        compiled = {
            key: value.detach().float().cpu().numpy().copy()
            for key, value in model.extract_rlt_obs(model_input).items()
        }

    parity = {}
    for key in native:
        difference = np.abs(native[key] - compiled[key])
        parity[key] = {
            "shape": list(native[key].shape),
            "mean_abs": float(difference.mean()),
            "max_abs": float(difference.max()),
            "all_finite": bool(np.isfinite(compiled[key]).all()),
        }
    result = {
        "repeats": args.repeats,
        "mode": "suffix_only" if args.suffix_only else "prefix_suffix",
        "native_ms": native_ms,
        "compiled_ms": compiled_ms,
        "compile_warmup_s": cold_start_s,
        "parity": parity,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
