#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Locate rare offline latency spikes in the visual prefix or action suffix."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from experiments.pi05_latency.runtime import compile_and_warm_model
from rlinf.envs.realworld.nero.right_arm_adapter import right_model_observation
from rlinf.models.embodiment.openpi_rlinf import get_model
from run.nero.run_right_pi05_live import model_config, offline_observation


def summarize(records: list[dict], key: str) -> dict[str, float]:
    values = sorted(row[key] for row in records)
    return {
        "median": statistics.median(values),
        "p99": values[int(0.99 * (len(values) - 1))],
        "max": values[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("/workspace/checkpoint/full_weights.pt")
    )
    parser.add_argument(
        "--norm-stats", type=Path, default=Path("/workspace/checkpoint/norm_stats.json")
    )
    parser.add_argument("--repeats", type=int, default=2500)
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
    cold_start_s = compile_and_warm_model(model, model_input)
    core = model.model
    original_prefix = core.build_prefix_cache
    compiled_suffix = core.run_suffix
    prefix_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    suffix_events = [
        [torch.cuda.Event(enable_timing=True) for _ in range(2)] for _ in range(5)
    ]
    suffix_index = 0

    def timed_prefix(*args, **kwargs):
        prefix_events[0].record()
        output = original_prefix(*args, **kwargs)
        prefix_events[1].record()
        return output

    def timed_suffix(*args, **kwargs):
        nonlocal suffix_index
        if suffix_index >= 5:
            raise AssertionError("Unexpected extra denoising step")
        start, end = suffix_events[suffix_index]
        suffix_index += 1
        start.record()
        output = compiled_suffix(*args, **kwargs)
        end.record()
        return output

    core.build_prefix_cache = timed_prefix
    core.run_suffix = timed_suffix
    records = []
    with torch.inference_mode():
        for index in range(args.repeats):
            suffix_index = 0
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            model.predict_action_batch(model_input)
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter_ns() - started) / 1e6
            if suffix_index != 5:
                raise AssertionError(
                    f"Expected five denoising steps, got {suffix_index}"
                )
            prefix_ms = prefix_events[0].elapsed_time(prefix_events[1])
            suffix_ms = [start.elapsed_time(end) for start, end in suffix_events]
            records.append(
                {
                    "index": index,
                    "wall_ms": wall_ms,
                    "prefix_ms": prefix_ms,
                    "suffix_ms": suffix_ms,
                    "suffix_total_ms": sum(suffix_ms),
                }
            )
    result = {
        "count": len(records),
        "cold_start_s": cold_start_s,
        "summary": {
            key: summarize(records, key)
            for key in ("wall_ms", "prefix_ms", "suffix_total_ms")
        },
        "slowest": sorted(records, key=lambda row: row["wall_ms"], reverse=True)[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"result": result, "records": records}, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
