# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Adapt the native dual-arm Nero exchange to the trained right-arm policy."""

from __future__ import annotations

from typing import Any

import numpy as np

from .contract import ACTION_HORIZON, PHYSICAL_ACTION_DIM

RIGHT_ARM_DIM = 8


def right_model_observation(observation: Any) -> dict[str, Any]:
    """Select the trained right arm and two RGB views from a native snapshot."""
    puppet = np.asarray(observation.puppet_state, dtype=np.float32)
    if puppet.shape != (PHYSICAL_ACTION_DIM,) or not np.isfinite(puppet).all():
        raise ValueError("Nero puppet state must be a finite 16-vector")

    def rgb(name: str) -> np.ndarray:
        image = np.asarray(observation.images[name])
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{name} must be an HWC uint8 BGR image")
        # The bridge sends OpenCV BGR; the training LeRobot videos were RGB.
        return np.ascontiguousarray(image[..., ::-1])

    high = rgb("cam_high")
    wrist = rgb("cam_right_wrist")
    return {
        "states": puppet[RIGHT_ARM_DIM:][None, :].copy(),
        "main_images": high[None, ...],
        "wrist_images": wrist[None, None, ...],
        "task_descriptions": [str(observation.prompt)],
    }


def right_actions_to_native(actions: Any, puppet_state: Any) -> np.ndarray:
    """Keep the observed left arm fixed while the policy commands the right."""
    right = np.asarray(actions, dtype=np.float32)
    puppet = np.asarray(puppet_state, dtype=np.float32)
    if right.shape != (ACTION_HORIZON, RIGHT_ARM_DIM):
        raise ValueError(f"Right-arm actions must have shape ({ACTION_HORIZON}, 8)")
    if puppet.shape != (PHYSICAL_ACTION_DIM,):
        raise ValueError("Nero puppet state must have shape (16,)")
    if not np.isfinite(right).all() or not np.isfinite(puppet).all():
        raise ValueError("Nero state and actions must be finite")
    left_hold = np.broadcast_to(puppet[:RIGHT_ARM_DIM], right.shape)
    return np.ascontiguousarray(np.concatenate((left_hold, right), axis=1))
