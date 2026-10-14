#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Profile offline pi0.5 latency tails without a bridge or robot connection."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("/workspace/checkpoint/full_weights.pt")
    )
    parser.add_argument(
        "--norm-stats", type=Path, default=Path("/workspace/checkpoint/norm_stats.json")
    )
    parser.add_argument("--repeats", type=int, default=1000)
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
    records = []
    with torch.inference_mode():
        for index in range(args.repeats):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall_start = time.perf_counter_ns()
            thread_start = time.thread_time_ns()
            start_event.record()
            model.predict_action_batch(model_input)
            end_event.record()
            torch.cuda.synchronize()
            records.append(
                {
                    "index": index,
                    "wall_ms": (time.perf_counter_ns() - wall_start) / 1e6,
                    "thread_cpu_ms": (time.thread_time_ns() - thread_start) / 1e6,
                    "event_ms": start_event.elapsed_time(end_event),
                }
            )
    values = sorted(row["wall_ms"] for row in records)
    summary = {
        "count": len(records),
        "cold_start_s": cold_start_s,
        "median_ms": statistics.median(values),
        "p95_ms": values[int(0.95 * (len(values) - 1))],
        "p99_ms": values[int(0.99 * (len(values) - 1))],
        "max_ms": values[-1],
        "over_100_ms": sum(value > 100 for value in values),
        "slowest": sorted(records, key=lambda row: row["wall_ms"], reverse=True)[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
