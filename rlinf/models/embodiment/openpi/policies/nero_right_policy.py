# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""OpenPI transforms for the right-only Nero (two RGB cameras, eight joints)."""

from __future__ import annotations

import dataclasses

import numpy as np
from openpi import transforms

from rlinf.models.embodiment.openpi.policies.nero_policy import (
    ACTION_HORIZON,
    _convert_image,
)

RIGHT_DIM = 8


@dataclasses.dataclass(frozen=True)
class NeroRightInputs(transforms.DataTransformFn):
    """Map the compact LeRobot record to the two-camera OpenPI contract."""

    def __call__(self, data: dict) -> dict:
        images = data["images"]
        expected = {"cam_high", "cam_right_wrist"}
        if set(images) != expected:
            raise ValueError(
                f"Expected cameras {sorted(expected)}, got {sorted(images)}"
            )
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (RIGHT_DIM,) or not np.isfinite(state).all():
            raise ValueError(
                f"Right-arm state must be finite [{RIGHT_DIM}], got {state.shape}"
            )
        result = {
            "image": {
                "base_0_rgb": _convert_image(images["cam_high"]),
                "right_wrist_0_rgb": _convert_image(images["cam_right_wrist"]),
            },
            "image_mask": {"base_0_rgb": np.True_, "right_wrist_0_rgb": np.True_},
            "state": state,
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if (
                actions.ndim != 2
                or actions.shape[-1] != RIGHT_DIM
                or not np.isfinite(actions).all()
            ):
                raise ValueError("Right-arm actions must be finite [horizon, 8]")
            result["actions"] = actions.copy()
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class NeroRightOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if (
            actions.ndim != 2
            or actions.shape[0] != ACTION_HORIZON
            or actions.shape[-1] < RIGHT_DIM
        ):
            raise ValueError(
                f"Right-arm model actions must have shape [{ACTION_HORIZON}, >=8]"
            )
        physical = np.asarray(actions[:, :RIGHT_DIM], dtype=np.float32)
        if not np.isfinite(physical).all():
            raise ValueError("Right-arm model actions contain NaN/Inf")
        return {"actions": physical}


@dataclasses.dataclass(frozen=True)
class NeroRightTrainingDropout(transforms.DataTransformFn):
    """Independent, per-sample dropout before π0.5 tokenizes discrete state.

    The transform is attached only to the right-arm SFT data config. Rollout
    observations have no ``actions`` key, so inference is unaffected.
    """

    wrist_probability: float = 0.0
    state_probability: float = 0.0

    def __post_init__(self) -> None:
        for probability in (self.wrist_probability, self.state_probability):
            if not 0.0 <= probability < 1.0:
                raise ValueError("Dropout probabilities must be in [0, 1)")

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        # A separate draw for each modality. Never mutate the dataset's arrays.
        if np.random.random() < self.wrist_probability:
            data = {
                **data,
                "image": dict(data["image"]),
                "image_mask": dict(data["image_mask"]),
            }
            data["image"]["right_wrist_0_rgb"] = np.zeros_like(
                data["image"]["right_wrist_0_rgb"]
            )
            data["image_mask"]["right_wrist_0_rgb"] = np.False_
        if np.random.random() < self.state_probability:
            data = {**data, "state": np.zeros_like(data["state"])}
        return data
