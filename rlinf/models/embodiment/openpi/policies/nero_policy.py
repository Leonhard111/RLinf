# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""OpenPI input/output transforms for the 16D AgileX Nero embodiment."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import numpy as np
from openpi import transforms

PHYSICAL_ACTION_DIM = 16
ACTION_HORIZON = 50


def _convert_image(image: np.ndarray) -> np.ndarray:
    result = np.asarray(image)
    if np.issubdtype(result.dtype, np.floating):
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    if result.ndim == 3 and result.shape[0] == 3:
        result = np.moveaxis(result, 0, -1)
    return result


def _decode_nero(data: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if "observation/state" in data:
        if "observation/image" not in data or "observation/wrist_image" not in data:
            raise KeyError("Nero rollout requires observation/image and wrist_image")
        wrist_images = _convert_image(data["observation/wrist_image"])
        if wrist_images.ndim != 4 or wrist_images.shape[0] != 2:
            raise ValueError(
                "observation/wrist_image must have shape [2, H, W, 3] in left/right order"
            )
        images = {
            "cam_high": _convert_image(data["observation/image"]),
            "cam_left_wrist": wrist_images[0],
            "cam_right_wrist": wrist_images[1],
        }
        state = np.asarray(data["observation/state"], dtype=np.float32)
    else:
        if "images" not in data or "state" not in data:
            raise KeyError("Nero sample requires images and state")
        images = {name: _convert_image(value) for name, value in data["images"].items()}
        state = np.asarray(data["state"], dtype=np.float32)
    if state.shape != (PHYSICAL_ACTION_DIM,):
        raise ValueError(
            f"Nero state must have shape ({PHYSICAL_ACTION_DIM},), got {state.shape}"
        )
    return images, state


@dataclasses.dataclass(frozen=True)
class NeroInputs(transforms.DataTransformFn):
    """Map three named Nero cameras and 16D proprioception to OpenPI keys."""

    adapt_to_pi: bool = False
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    )

    def __call__(self, data: dict) -> dict:
        if self.adapt_to_pi:
            raise ValueError("Nero requires adapt_to_pi=False")
        images, state = _decode_nero(data)
        missing = [name for name in self.EXPECTED_CAMERAS if name not in images]
        unknown = sorted(set(images) - set(self.EXPECTED_CAMERAS))
        if missing or unknown:
            raise ValueError(f"Nero cameras missing={missing}, unknown={unknown}")
        result = {
            "image": {
                "base_0_rgb": images["cam_high"],
                "left_wrist_0_rgb": images["cam_left_wrist"],
                "right_wrist_0_rgb": images["cam_right_wrist"],
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": state,
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != PHYSICAL_ACTION_DIM:
                raise ValueError("Nero training actions must have shape [horizon, 16]")
            result["actions"] = actions.copy()
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class NeroOutputs(transforms.DataTransformFn):
    """Extract the physical 16D prefix from the model's padded 32D action."""

    adapt_to_pi: bool = False

    def __call__(self, data: dict) -> dict:
        if self.adapt_to_pi:
            raise ValueError("Nero requires adapt_to_pi=False")
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[0] != ACTION_HORIZON:
            raise ValueError(
                f"Nero model actions must have shape [{ACTION_HORIZON}, D], got {actions.shape}"
            )
        if actions.shape[-1] < PHYSICAL_ACTION_DIM:
            raise ValueError("Nero model action width is smaller than 16")
        physical = np.asarray(actions[:, :PHYSICAL_ACTION_DIM], dtype=np.float32)
        if not np.isfinite(physical).all():
            raise ValueError("Nero model actions contain NaN/Inf")
        return {"actions": physical}
