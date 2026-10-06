# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Right-arm deployment mapping never sends model output to the left arm."""

from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.envs.realworld.nero.right_arm_adapter import (
    right_actions_to_native,
    right_model_observation,
)


def test_right_only_observation_and_native_action_mapping():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[..., :] = [10, 20, 30]  # bridge BGR
    puppet = np.arange(16, dtype=np.float32)
    observation = SimpleNamespace(
        puppet_state=puppet,
        images={"cam_high": image, "cam_right_wrist": image, "cam_left_wrist": image},
        prompt="Stack the red square on top of the green square.",
    )
    model_input = right_model_observation(observation)
    assert model_input["states"].shape == (1, 8)
    np.testing.assert_array_equal(model_input["states"][0], puppet[8:])
    assert model_input["wrist_images"].shape == (1, 1, 4, 5, 3)
    np.testing.assert_array_equal(model_input["main_images"][0, 0, 0], [30, 20, 10])

    right = np.tile(np.arange(8, dtype=np.float32), (50, 1))
    native = right_actions_to_native(right, puppet)
    assert native.shape == (50, 16)
    np.testing.assert_array_equal(native[:, :8], np.tile(puppet[:8], (50, 1)))
    np.testing.assert_array_equal(native[:, 8:], right)
    with pytest.raises(ValueError, match="shape"):
        right_actions_to_native(right[:-1], puppet)


def test_right_openpi_input_accepts_live_observation():
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.policies.nero_right_policy import (
        NeroRightInputs,
    )

    sample = {
        "observation/image": np.zeros((4, 5, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((1, 4, 5, 3), dtype=np.uint8),
        "observation/extra_view_image": np.zeros((1, 4, 5, 3), dtype=np.uint8),
        "observation/state": np.arange(8, dtype=np.float32),
        "prompt": "task",
    }
    transformed = NeroRightInputs()(sample)
    assert set(transformed["image"]) == {"base_0_rgb", "right_wrist_0_rgb"}
    np.testing.assert_array_equal(transformed["state"], np.arange(8))
    with pytest.raises(ValueError, match="wrist image"):
        NeroRightInputs()(
            {
                **sample,
                "observation/wrist_image": np.zeros((2, 4, 5, 3), dtype=np.uint8),
            }
        )
