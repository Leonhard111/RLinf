#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Offline latency and action-drift benchmark for the deployed Nero pi0.5 model.

This script never connects to the robot or ROS. It uses the same model factory,
checkpoint, normalization, and observation adapter as run_right_pi05_live.py.
"""

from __future__ import annotations

import argparse
import json
import time
from copy import copy
from pathlib import Path

import numpy as np
import torch

from experiments.pi05_latency.runtime import clone_cuda_graph_outputs
from rlinf.envs.realworld.nero.right_arm_adapter import right_model_observation
from rlinf.models.embodiment.openpi_rlinf import get_model
from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import preprocess_observation
from run.nero.run_right_pi05_live import model_config, offline_observation


def describe_ms(values: list[float]) -> dict[str, float]:
    """Summarize synchronized wall latency for a fixed-shape batch."""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def time_repeated(fn, *, warmup: int, repeats: int) -> tuple[dict, object]:
    """Run the callable with CUDA synchronization around each wall sample."""
    last = None
    with torch.inference_mode():
        for _ in range(warmup):
            last = fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            last = fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter_ns() - started) / 1e6)
    return describe_ms(times), last


def main() -> int:
    """Measure the deployed path and its major GPU stages on a static input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("/workspace/checkpoint/full_weights.pt")
    )
    parser.add_argument(
        "--norm-stats", type=Path, default=Path("/workspace/checkpoint/norm_stats.json")
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 3, 2, 1])
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--compile-suffix", action="store_true")
    parser.add_argument("--compile-prefix", action="store_true")
    parser.add_argument(
        "--prefix-cudagraph-safe",
        action="store_true",
        help="compile prefix with CUDA Graphs, mark iterations, and clone returned tensors",
    )
    parser.add_argument(
        "--sdpa", action="store_true", help="Use experimental fused attention"
    )
    parser.add_argument("--validate-dynamic", action="store_true")
    parser.add_argument("--dynamic-cases", type=int, default=4)
    parser.add_argument("--random-observation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.steps) < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("steps and repeats must be positive; warmup must be nonnegative")
    if args.dynamic_cases < 1:
        parser.error("--dynamic-cases must be positive")
    if args.prefix_cudagraph_safe and not args.compile_prefix:
        parser.error("--prefix-cudagraph-safe requires --compile-prefix")

    model = (
        get_model(model_config(args.checkpoint, args.norm_stats, "bf16", 5))
        .to("cuda")
        .eval()
    )
    core = model.model
    static = offline_observation(
        args.norm_stats, "Stack the red square on top of the green square."
    )
    if args.random_observation:
        image_rng = np.random.default_rng(20261012)
        static.images = {
            name: image_rng.integers(0, 256, size=image.shape, dtype=np.uint8)
            for name, image in static.images.items()
        }
        static.puppet_state = np.asarray(static.puppet_state).copy()
        static.puppet_state[8:15] += image_rng.uniform(-0.1, 0.1, size=7).astype(
            np.float32
        )
    env_obs = right_model_observation(static)
    generator = torch.Generator(device="cuda").manual_seed(20261012)
    noise = torch.randn(
        (1, core.action_horizon, core.action_dim), device="cuda", generator=generator
    )
    native_suffix = core.run_suffix
    native_prefix = core.build_prefix_cache
    native_reference = None
    if args.sdpa or args.compile_suffix or args.compile_prefix:
        with torch.inference_mode():
            native_reference = model.predict_action_batch(env_obs, noise=noise)[0]
            native_reference = native_reference.detach().float().cpu().numpy().copy()
    if args.sdpa:
        from attention_sdpa import install

        install()
    if args.compile_suffix:
        core.run_suffix = torch.compile(core.run_suffix, mode="reduce-overhead")
    if args.compile_prefix:
        if args.prefix_cudagraph_safe:
            compiled_prefix = torch.compile(
                core.build_prefix_cache, mode="reduce-overhead"
            )

            def safe_prefix(observation):
                torch.compiler.cudagraph_mark_step_begin()
                return clone_cuda_graph_outputs(compiled_prefix(observation))

            core.build_prefix_cache = safe_prefix
        else:
            core.build_prefix_cache = torch.compile(
                core.build_prefix_cache, mode="default"
            )
    result: dict = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "compile_suffix": args.compile_suffix,
        "compile_prefix": args.compile_prefix,
        "prefix_cudagraph_safe": args.prefix_cudagraph_safe,
        "sdpa": args.sdpa,
        "random_observation": args.random_observation,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "dynamic_cases": args.dynamic_cases,
        "checkpoint": str(args.checkpoint),
        "steps": {},
    }
    reference = None
    for step_count in args.steps:
        model.num_steps = step_count
        timing, pair = time_repeated(
            lambda: model.predict_action_batch(env_obs, noise=noise),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        actions = pair[0].detach().float().cpu().numpy()
        if step_count == 5:
            reference = actions.copy()
        drift = None
        if reference is not None:
            difference = np.abs(actions - reference)
            drift = {
                "mean_abs": float(difference.mean()),
                "p95_abs": float(np.percentile(difference, 95)),
                "max_abs": float(difference.max()),
            }
        result["steps"][str(step_count)] = {"full_ms": timing, "drift_vs_5": drift}
        if native_reference is not None:
            native_difference = np.abs(actions - native_reference)
            result["steps"][str(step_count)]["drift_vs_native_5"] = {
                "mean_abs": float(native_difference.mean()),
                "p95_abs": float(np.percentile(native_difference, 95)),
                "max_abs": float(native_difference.max()),
            }
        print(f"steps={step_count} full_ms={timing} drift_vs_5={drift}", flush=True)

    processed = model.input_transform(model._repack_env_obs(env_obs), transpose=False)
    observation = model._observation_dict_to_device(processed)
    prepared = preprocess_observation(observation, train=False)
    prefix_ms, prefix_result = time_repeated(
        lambda: core.build_prefix_cache(prepared),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    _, prefix_mask, kv_cache = prefix_result
    time_tensor = torch.full((1,), 0.5, device="cuda", dtype=torch.float32)
    suffix_ms, _ = time_repeated(
        lambda: core.run_suffix(prepared, noise, time_tensor, kv_cache, prefix_mask),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    result["stages_ms"] = {"prefix": prefix_ms, "one_suffix": suffix_ms}
    if args.validate_dynamic:
        if not args.compile_suffix or args.sdpa:
            parser.error("--validate-dynamic currently requires --compile-suffix only")
        model.num_steps = 5
        optimized_suffix = core.run_suffix
        optimized_prefix = core.build_prefix_cache
        comparisons = []
        previous_optimized = None
        for case in range(args.dynamic_cases):
            changed_noise = torch.randn(noise.shape, device="cuda", generator=generator)
            changed_static = copy(static)
            changed_static.images = dict(static.images)
            changed_static.puppet_state = np.asarray(static.puppet_state).copy()
            rng = np.random.default_rng(9000 + case)
            if case:
                changed_static.puppet_state[8:15] += rng.uniform(
                    -0.1, 0.1, size=7
                ).astype(np.float32)
            if case >= 2:
                changed_static.images = {
                    name: rng.integers(0, 256, size=image.shape, dtype=np.uint8)
                    for name, image in changed_static.images.items()
                }
            changed_obs = right_model_observation(changed_static)
            with torch.inference_mode():
                core.run_suffix = native_suffix
                core.build_prefix_cache = native_prefix
                native = model.predict_action_batch(changed_obs, noise=changed_noise)[0]
                native = native.detach().float().cpu().numpy().copy()
                core.run_suffix = optimized_suffix
                core.build_prefix_cache = optimized_prefix
                optimized = model.predict_action_batch(
                    changed_obs, noise=changed_noise
                )[0]
                optimized = optimized.detach().float().cpu().numpy().copy()
            difference = np.abs(optimized - native)
            if not np.isfinite(optimized).all():
                raise AssertionError(f"non-finite compiled actions in case {case}")
            changed = (
                None
                if previous_optimized is None
                else float(np.max(np.abs(optimized - previous_optimized)))
            )
            if changed is not None and changed == 0.0:
                raise AssertionError(f"compiled output did not change in case {case}")
            comparisons.append(
                {
                    "case": case,
                    "mean_abs": float(difference.mean()),
                    "p95_abs": float(np.percentile(difference, 95)),
                    "max_abs": float(difference.max()),
                    "max_change_from_previous": changed,
                }
            )
            previous_optimized = optimized
        core.run_suffix = optimized_suffix
        core.build_prefix_cache = optimized_prefix
        result["dynamic_parity"] = comparisons
        print(f"dynamic_parity={comparisons}", flush=True)
    result["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"stages_ms={result['stages_ms']}", flush=True)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
