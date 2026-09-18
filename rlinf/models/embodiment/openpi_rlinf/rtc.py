# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Small RTC primitives for the RLinf π0.5 eval sampler.

The guidance equations follow Physical Intelligence's Kinetix reference and
LeRobot's 1→0 flow-time adaptation.  This file is deliberately independent of
the robot transport and controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

RTC_PREFIX_SCHEDULES = ("exp", "linear", "ones", "zeros")
RTC_CONTEXT_VECTOR_DIM = 9


@dataclass(frozen=True)
class RTCModelContext:
    enabled: bool
    response_generation_id: int
    previous_generation_id: int
    request_policy_step: int
    execution_horizon_steps: int
    predicted_delay_steps: int
    policy_hz: float
    prefix_attention_schedule: str
    max_guidance_weight: float

    @classmethod
    def from_tensor(cls, value: torch.Tensor | None) -> "RTCModelContext | None":
        if value is None:
            return None
        tensor = torch.as_tensor(value).detach().cpu()
        if tensor.ndim == 2:
            if tensor.shape[0] != 1:
                raise ValueError("Nero RTC currently supports exactly one robot")
            tensor = tensor[0]
        if tensor.shape != (RTC_CONTEXT_VECTOR_DIM,):
            raise ValueError(
                f"RTC context must have shape {(RTC_CONTEXT_VECTOR_DIM,)}, "
                f"got {tuple(tensor.shape)}"
            )
        fields = tensor.tolist()
        schedule_index = int(round(fields[7]))
        if not 0 <= schedule_index < len(RTC_PREFIX_SCHEDULES):
            raise ValueError("invalid RTC prefix schedule code")
        context = cls(
            enabled=bool(round(fields[0])),
            response_generation_id=int(round(fields[1])),
            previous_generation_id=int(round(fields[2])),
            request_policy_step=int(round(fields[3])),
            execution_horizon_steps=int(round(fields[4])),
            predicted_delay_steps=int(round(fields[5])),
            policy_hz=float(fields[6]),
            prefix_attention_schedule=RTC_PREFIX_SCHEDULES[schedule_index],
            max_guidance_weight=float(fields[8]),
        )
        if context.response_generation_id < 0:
            return None
        if (
            not math.isfinite(context.max_guidance_weight)
            or context.max_guidance_weight <= 0
        ):
            raise ValueError("RTC max guidance weight must be finite and positive")
        return context


def prefix_weights(
    start: int,
    end: int,
    total: int,
    schedule: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the paper's [fixed→blended→free] temporal prefix weights."""

    if total < 1 or not 0 <= start <= total or not 0 <= end <= total:
        raise ValueError("invalid RTC prefix weight interval")
    start = min(start, end)
    indices = torch.arange(total, device=device, dtype=dtype)
    if schedule == "ones":
        weights = torch.ones(total, device=device, dtype=dtype)
    elif schedule == "zeros":
        weights = (indices < start).to(dtype=dtype)
    elif schedule in {"linear", "exp"}:
        weights = torch.clamp(
            (start - 1 - indices) / (end - start + 1) + 1,
            min=0.0,
            max=1.0,
        )
        if schedule == "exp":
            weights = weights * torch.expm1(weights) / (math.e - 1)
    else:
        raise ValueError(f"unsupported RTC prefix schedule: {schedule}")
    return torch.where(indices >= end, torch.zeros_like(weights), weights)


def align_previous_chunk(
    previous: torch.Tensor,
    execution_horizon_steps: int,
) -> torch.Tensor:
    """Shift an old normalized model chunk to the new observation's time origin."""

    if previous.ndim != 3:
        raise ValueError("previous RTC model chunk must be [B,H,D]")
    horizon = previous.shape[1]
    if not 0 <= execution_horizon_steps <= horizon:
        raise ValueError("RTC execution horizon is outside the model chunk")
    tail = previous[:, execution_horizon_steps:]
    padding = torch.zeros(
        previous.shape[0],
        execution_horizon_steps,
        previous.shape[2],
        device=previous.device,
        dtype=previous.dtype,
    )
    return torch.cat((tail, padding), dim=1)


def guided_velocity(
    x_t: torch.Tensor,
    previous_aligned: torch.Tensor,
    *,
    inference_delay_steps: int,
    prefix_attention_horizon: int,
    time: float,
    schedule: str,
    max_guidance_weight: float,
    denoise_fn,
) -> torch.Tensor:
    """Apply one VJP-based RTC correction to a 1→0 flow-matching velocity."""

    if previous_aligned.shape != x_t.shape:
        raise ValueError("aligned previous RTC chunk must match the sampled chunk")
    weights = prefix_weights(
        inference_delay_steps,
        prefix_attention_horizon,
        x_t.shape[1],
        schedule,
        device=x_t.device,
        dtype=x_t.dtype,
    )[None, :, None]

    # Each denoising step is an independent VJP. Detaching here prevents the
    # autograd graph from growing across the Euler loop.
    with torch.enable_grad():
        x_leaf = x_t.detach().requires_grad_(True)
        base_velocity = denoise_fn(x_leaf)
        estimated_action = x_leaf - float(time) * base_velocity
        error = (previous_aligned - estimated_action) * weights
        correction = torch.autograd.grad(
            estimated_action,
            x_leaf,
            grad_outputs=error.detach(),
            retain_graph=False,
            create_graph=False,
        )[0]

    tau = 1.0 - float(time)
    denominator = max((1.0 - tau) ** 2, torch.finfo(x_t.dtype).eps)
    inv_r2 = ((1.0 - tau) ** 2 + tau**2) / denominator
    if tau <= 0.0:
        c = max_guidance_weight
    else:
        c = (1.0 - tau) / tau
    weight = min(c * inv_r2, max_guidance_weight)
    return (base_velocity - weight * correction).detach()
