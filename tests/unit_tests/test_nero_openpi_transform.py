# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Hardware-free tests for the Nero OpenPI transforms and env repack."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from openpi import transforms
from openpi.shared.normalize import NormStats

from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.policies.nero_policy import (
    NeroInputs,
    NeroOutputs,
)


def sample() -> dict:
    return {
        "observation/image": np.full((4, 5, 3), 1, dtype=np.uint8),
        "observation/wrist_image": np.stack((
            np.full((4, 5, 3), 2, dtype=np.uint8),
            np.full((4, 5, 3), 3, dtype=np.uint8),
        )),
        "observation/state": np.arange(16, dtype=np.float32),
        "prompt": "test nero",
    }


def test_pi05_nero_config_contract() -> None:
    config = get_openpi_config("pi05_nero")
    assert config.model.pi05
    assert config.model.action_horizon == 50
    assert config.model.discrete_state_input
    assert config.data.adapt_to_pi is False
    assert config.data.extra_delta_transform
    assert config.data.assets.asset_id == "local/nero_aloha16"


def test_nero_inputs_preserve_explicit_left_right_order() -> None:
    result = NeroInputs()(sample())
    assert result["state"].shape == (16,)
    assert tuple(result["image"]) == (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    assert np.all(result["image"]["left_wrist_0_rgb"] == 2)
    assert np.all(result["image"]["right_wrist_0_rgb"] == 3)


def test_nero_inputs_reject_aloha_14d_state() -> None:
    invalid = sample()
    invalid["observation/state"] = np.zeros(14, dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        NeroInputs()(invalid)


def test_nero_outputs_extract_exact_50_by_16_prefix() -> None:
    padded = np.arange(50 * 32, dtype=np.float32).reshape(50, 32)
    result = NeroOutputs()({"actions": padded})["actions"]
    assert result.shape == (50, 16)
    np.testing.assert_array_equal(result, padded[:, :16])


def test_delta_absolute_and_normalize_round_trips() -> None:
    mask = np.asarray([True] * 7 + [False] + [True] * 7 + [False])
    state = np.linspace(-0.5, 0.5, 16, dtype=np.float32)
    absolute = np.tile(state, (50, 1))
    absolute[:, [7, 15]] = 0.05
    delta_data = transforms.DeltaActions(mask)({
        "state": state.copy(),
        "actions": absolute.copy(),
    })
    restored = transforms.AbsoluteActions(mask)({
        "state": state.copy(),
        "actions": delta_data["actions"].copy(),
    })
    np.testing.assert_allclose(restored["actions"], absolute, atol=1e-6)

    stats = {
        "state": NormStats(mean=np.linspace(-1, 1, 16), std=np.linspace(1, 2, 16)),
        "actions": NormStats(mean=np.linspace(-1, 1, 16), std=np.linspace(1, 2, 16)),
    }
    values = {"state": state.copy(), "actions": absolute.copy()}
    normalized = transforms.Normalize(stats)(copy.deepcopy(values))
    unnormalized = transforms.Unnormalize(stats)(normalized)
    np.testing.assert_allclose(unnormalized["state"], values["state"], atol=1e-6)
    np.testing.assert_allclose(unnormalized["actions"], values["actions"], atol=1e-6)


def test_realworld_wrapper_selects_only_puppet_and_explicit_wrists() -> None:
    wrapper = RealWorldEnv.__new__(RealWorldEnv)
    wrapper.main_image_key = "cam_high"
    wrapper.model_state_keys = ("puppet",)
    wrapper.wrist_image_keys = ("cam_left_wrist", "cam_right_wrist")
    wrapper.task_descriptions = ["test nero"]
    raw = {
        "state": {
            "master": np.ones((1, 16), dtype=np.float32),
            "puppet": np.zeros((1, 16), dtype=np.float32),
            "risk": np.ones((1, 5), dtype=np.float32),
        },
        "frames": {
            "cam_high": np.full((1, 4, 5, 3), 1, dtype=np.uint8),
            "cam_right_wrist": np.full((1, 4, 5, 3), 3, dtype=np.uint8),
            "cam_left_wrist": np.full((1, 4, 5, 3), 2, dtype=np.uint8),
        },
    }
    result = wrapper._wrap_obs(raw)
    assert tuple(result["states"].shape) == (1, 16)
    assert tuple(result["wrist_images"].shape) == (1, 2, 4, 5, 3)
    assert np.all(result["wrist_images"][0, 0].numpy() == 2)
    assert np.all(result["wrist_images"][0, 1].numpy() == 3)
    assert "extra_view_images" not in result
