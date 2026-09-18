#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Run RLinf PyTorch π0.5 on fixed Nero observations without ROS/CAN."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from verify_pi05_checkpoint import build_model_config, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("observations", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source_manifest_path = args.observations / "observations_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    entries = source_manifest.get("samples", [])
    if not entries:
        raise ValueError("observation manifest contains no samples")
    import torch

    from rlinf.models.embodiment.openpi_rlinf import get_model

    cfg = build_model_config(
        args.model,
        args.norm_stats,
        precision=args.precision,
        num_steps=args.num_steps,
    )
    model = get_model(cfg).to(args.device).eval()
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    all_actions: list[np.ndarray] = []
    latencies: list[float] = []
    for entry in entries:
        sample_path = args.observations / entry["path"]
        if sha256_file(sample_path) != entry["sha256"]:
            raise ValueError(f"sample hash mismatch: {sample_path}")
        with np.load(sample_path, allow_pickle=False) as sample:
            prompt = str(sample["prompt"].item())
            observation = {
                "states": np.asarray(sample["puppet_state"], dtype=np.float32)[None, :],
                "main_images": np.asarray(sample["cam_high"], dtype=np.uint8)[
                    None, ...
                ],
                # The order is explicit and matches pi05_nero; never sort keys.
                "wrist_images": np.stack(
                    (
                        np.asarray(sample["cam_left_wrist"], dtype=np.uint8),
                        np.asarray(sample["cam_right_wrist"], dtype=np.uint8),
                    ),
                    axis=0,
                )[None, ...],
                "task_descriptions": [prompt],
            }
        if observation["states"].shape != (1, 16):
            raise ValueError(f"invalid puppet state in {sample_path}")
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        started = time.perf_counter()
        actions, details = model.predict_action_batch(observation, rng=generator)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        latencies.append((time.perf_counter() - started) * 1000.0)
        physical = actions.detach().cpu().numpy().astype(np.float32)
        if physical.shape != (1, 50, 16) or not np.isfinite(physical).all():
            raise ValueError(f"invalid π0.5 output for {sample_path}: {physical.shape}")
        model_action = details["forward_inputs"]["model_action"].reshape(1, 50, 32)
        if not torch.isfinite(model_action).all():
            raise ValueError(f"non-finite internal action for {sample_path}")
        all_actions.append(physical[0])

    action_array = np.asarray(all_actions, dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, actions=action_array)
    manifest = {
        "schema_version": "nero-pi05-offline/1.0.0",
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "model_sha256": sha256_file(args.model / "model.safetensors"),
        "norm_stats_sha256": sha256_file(args.norm_stats),
        "seed": args.seed,
        "sample_count": len(all_actions),
        "action_shape_per_sample": [50, 16],
        "publisher_created": False,
        "latency_ms": {
            "min": min(latencies),
            "median": statistics.median(latencies),
            "max": max(latencies),
        },
        "action_min": np.min(action_array, axis=(0, 1)).tolist(),
        "action_max": np.max(action_array, axis=(0, 1)).tolist(),
        "action_mean": np.mean(action_array, axis=(0, 1)).tolist(),
        "action_std": np.std(action_array, axis=(0, 1)).tolist(),
        "actions_path": str(args.output.resolve()),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output.resolve())
    print(args.manifest.resolve())
    print(
        json.dumps(
            {
                "sample_count": manifest["sample_count"],
                "latency_ms": manifest["latency_ms"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
