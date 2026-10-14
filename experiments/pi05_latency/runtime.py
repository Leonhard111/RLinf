# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Opt-in low-latency inference path for the Nero pi0.5 action expert."""

from __future__ import annotations

import time
from typing import Any

import torch


def clone_cuda_graph_outputs(value: Any) -> Any:
    """Copy tensors returned from a CUDA Graph into ordinary tensor storage."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone_cuda_graph_outputs(item) for item in value)
    if isinstance(value, list):
        return [clone_cuda_graph_outputs(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_cuda_graph_outputs(item) for key, item in value.items()}
    return value


def compile_and_warm_model(
    model: Any, model_input: dict[str, Any], *, compile_prefix: bool = False
) -> float:
    """Compile suffix and optional prefix, warm fixed shapes, return cold seconds.

    This changes only the current eval model instance. Weights, five-step
    sampling, normalization, and action conversion remain unchanged.
    """
    started = time.perf_counter()
    model.model.run_suffix = torch.compile(
        model.model.run_suffix, mode="reduce-overhead"
    )
    if compile_prefix:
        compiled_prefix = torch.compile(
            model.model.build_prefix_cache, mode="reduce-overhead"
        )

        def safe_prefix(observation: Any) -> Any:
            torch.compiler.cudagraph_mark_step_begin()
            return clone_cuda_graph_outputs(compiled_prefix(observation))

        model.model.build_prefix_cache = safe_prefix
    with torch.inference_mode():
        for _ in range(3):
            model.predict_action_batch(model_input)
    torch.cuda.synchronize()
    return time.perf_counter() - started
