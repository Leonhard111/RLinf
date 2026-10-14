#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Evaluate a right-arm Stage 1 checkpoint offline or via the native Nero bridge."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rlinf.envs.realworld.nero.contract import ControlMode
from rlinf.envs.realworld.nero.nero_client import NeroRobotClient
from rlinf.envs.realworld.nero.right_arm_adapter import (
    right_actions_to_native,
    right_model_observation,
)

DEFAULT_PROMPT = "Stack the red square on top of the green square."
TRAIN_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "examples/sft/config/nero_right_rlt_stage1_sft_openpi_pi05.yaml"
)


def model_config(checkpoint: Path, norm_stats: Path, precision: str, num_steps: int):
    """Reuse the exact Stage 1 model and RLT settings for evaluation."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(TRAIN_CONFIG)
    model = cfg.actor.model
    model.model_path = str(checkpoint)
    model.precision = precision
    model.num_steps = num_steps
    model.openpi.task = "eval"
    model.openpi_data.norm_stats_path = str(norm_stats)
    del model.openpi_data.repo_id
    model.openpi_data.wrist_dropout_prob = 0.0
    model.openpi_data.state_dropout_prob = 0.0
    return OmegaConf.create(OmegaConf.to_container(model, resolve=True))


def offline_observation(norm_stats: Path, prompt: str) -> SimpleNamespace:
    """Build an inert sample from the training-state mean for a model smoke test."""
    stats = json.loads(norm_stats.read_text(encoding="utf-8"))["norm_stats"]
    right_state = np.asarray(stats["state"]["mean"], dtype=np.float32)
    if right_state.shape != (8,):
        raise ValueError("The selected normalization file has no right-arm 8D state")
    puppet = np.concatenate((np.zeros(8, dtype=np.float32), right_state))
    black = np.zeros((224, 224, 3), dtype=np.uint8)
    return SimpleNamespace(
        puppet_state=puppet,
        images={"cam_high": black, "cam_right_wrist": black},
        prompt=prompt,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("/workspace/checkpoint/full_weights.pt")
    )
    parser.add_argument(
        "--norm-stats", type=Path, default=Path("/workspace/checkpoint/norm_stats.json")
    )
    parser.add_argument(
        "--mode", choices=("offline", "shadow", "publish"), default="offline"
    )
    parser.add_argument("--endpoint", default="tcp://192.168.50.108:5555")
    parser.add_argument("--bridge-request-timeout-s", type=float, default=180.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--max-generations", type=int, default=1, help="0 runs until Ctrl+C"
    )
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--compile-suffix",
        action="store_true",
        help="experiment: compile the pi0.5 action-expert suffix and warm it before inference",
    )
    parser.add_argument(
        "--compile-prefix",
        action="store_true",
        help="experiment: also compile the visual prefix; requires --compile-suffix",
    )
    args = parser.parse_args()
    if (
        args.max_generations < 0
        or args.num_steps < 1
        or args.bridge_request_timeout_s <= 0
        or not args.prompt.strip()
    ):
        parser.error(
            "max-generations must be nonnegative; num-steps, bridge timeout, "
            "and prompt must be valid"
        )
    if args.compile_prefix and not args.compile_suffix:
        parser.error("--compile-prefix requires --compile-suffix")
    for path in (args.checkpoint, args.norm_stats):
        if not path.is_file():
            parser.error(f"Missing file: {path}")

    import torch

    from rlinf.models.embodiment.openpi_rlinf import get_model

    model = (
        get_model(
            model_config(
                args.checkpoint, args.norm_stats, args.precision, args.num_steps
            )
        )
        .to("cuda")
        .eval()
    )
    if args.compile_suffix:
        from experiments.pi05_latency.runtime import compile_and_warm_model

        warmup_observation = offline_observation(args.norm_stats, args.prompt)
        cold_start_s = compile_and_warm_model(
            model,
            right_model_observation(warmup_observation),
            compile_prefix=args.compile_prefix,
        )
        label = "compiled_prefix_suffix" if args.compile_prefix else "compiled_suffix"
        print(f"{label}_warmup_s={cold_start_s:.1f}", flush=True)
    client = None
    try:
        if args.mode == "offline":
            observation = offline_observation(args.norm_stats, args.prompt)
        else:
            client = NeroRobotClient(
                args.endpoint, request_timeout_s=args.bridge_request_timeout_s
            )
            expected = (
                ControlMode.SHADOW if args.mode == "shadow" else ControlMode.PUBLISH
            )
            if client.capabilities.mode is not expected:
                raise RuntimeError(
                    f"Bridge mode is {client.capabilities.mode.value}; expected {expected.value}"
                )
            if client.capabilities.control_owner != "pi05-native":
                raise RuntimeError("Expected the native pi05 bridge on AgileX")
            observation = client.reset()

        count = 0
        while args.max_generations == 0 or count < args.max_generations:
            if observation.prompt.strip() != args.prompt.strip():
                raise ValueError("Bridge prompt differs from the trained task prompt")
            model_input = right_model_observation(observation)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                predicted, _ = model.predict_action_batch(model_input)
            torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - started) * 1000.0
            right = predicted[0].detach().cpu().numpy()
            native = right_actions_to_native(right, observation.puppet_state)
            right_state = np.asarray(observation.puppet_state, dtype=np.float32)[8:]
            max_joint_delta = float(np.max(np.abs(right[:, :7] - right_state[:7])))
            print(
                f"generation={count} mode={args.mode} inference_ms={inference_ms:.1f} "
                f"action_shape={native.shape} right_peak_delta_rad={max_joint_delta:.3f}",
                flush=True,
            )
            count += 1
            if client is not None:
                observation = client.step(native, representation="absolute")
    except KeyboardInterrupt:
        print("Stopped by operator", flush=True)
    finally:
        if client is not None:
            client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
