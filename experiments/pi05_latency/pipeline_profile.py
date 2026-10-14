#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Measure the full offline observation-to-native-action path (no robot)."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pi05_latency.runtime import compile_and_warm_model
from rlinf.envs.realworld.nero.right_arm_adapter import (
    right_actions_to_native,
    right_model_observation,
)
from rlinf.models.embodiment.openpi_rlinf import get_model
from run.nero.run_right_pi05_live import model_config, offline_observation


def summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "p99": ordered[int(0.99 * (len(ordered) - 1))],
        "max": ordered[-1],
        "over_100": sum(value > 100 for value in ordered),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("/workspace/checkpoint/full_weights.pt")
    )
    parser.add_argument(
        "--norm-stats", type=Path, default=Path("/workspace/checkpoint/norm_stats.json")
    )
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--compile-prefix", action="store_true")
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
    cold_start_s = compile_and_warm_model(
        model,
        right_model_observation(observation),
        compile_prefix=args.compile_prefix,
    )
    records = []
    collections = []
    current_index = -1

    def record_gc(phase, info):
        if phase == "stop":
            collections.append(
                {"index": current_index, "generation": info["generation"]}
            )

    gc.callbacks.append(record_gc)
    with torch.inference_mode():
        for index in range(args.repeats):
            current_index = index
            collection_count = len(collections)
            started = time.perf_counter_ns()
            thread_started = time.thread_time_ns()
            model_input = right_model_observation(observation)
            adapted = time.perf_counter_ns()
            torch.cuda.synchronize()
            model_started = time.perf_counter_ns()
            predicted, _ = model.predict_action_batch(model_input)
            torch.cuda.synchronize()
            modeled = time.perf_counter_ns()
            right = predicted[0].detach().cpu().numpy()
            native = right_actions_to_native(right, observation.puppet_state)
            if native.shape != (50, 16) or not np.isfinite(native).all():
                raise AssertionError("Invalid native action chunk")
            finished = time.perf_counter_ns()
            records.append(
                {
                    "index": index,
                    "thread_cpu_ms": (time.thread_time_ns() - thread_started) / 1e6,
                    "gc_collections": len(collections) - collection_count,
                    "adapter_ms": (adapted - started) / 1e6,
                    "model_ms": (modeled - model_started) / 1e6,
                    "conversion_ms": (finished - modeled) / 1e6,
                    "total_ms": (finished - started) / 1e6,
                }
            )
    gc.callbacks.remove(record_gc)
    result = {
        "count": len(records),
        "compile_prefix": args.compile_prefix,
        "cold_start_s": cold_start_s,
        "summary": {
            key: summary([row[key] for row in records])
            for key in ("adapter_ms", "model_ms", "conversion_ms", "total_ms")
        },
        "slowest": sorted(records, key=lambda row: row["total_ms"], reverse=True)[:20],
        "gc_collections": collections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"result": result, "records": records}, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
