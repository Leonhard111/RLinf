# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Focused tests for right-arm capture conversion and OpenPI adapters."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from toolkits.lerobot.prepare_nero_right_dataset import (
    align_episode,
    audit_episode,
    resize_with_pad,
    write_norm_stats,
)


def test_resize_preserves_aspect_ratio_and_black_pad():
    source = np.full((40, 80, 3), 200, dtype=np.uint8)
    resized = resize_with_pad(source, 224)
    assert resized.shape == (224, 224, 3)
    assert np.all(resized[:56] == 0)
    assert np.all(resized[56:168] == 200)
    assert np.all(resized[168:] == 0)


def test_alignment_and_stats_on_timestamped_capture():
    with TemporaryDirectory() as root:
        episode = Path(root) / "episode15"
        for stream in (
            "camera/color/front",
            "camera/color/right",
            "arm/jointState/puppetRight",
            "arm/jointState/masterRight",
        ):
            (episode / stream).mkdir(parents=True)
        for index in range(60):
            t = 1000 + index / 30
            image = np.full((40, 80, 3), 80 + index, dtype=np.uint8)
            assert cv2.imwrite(
                str(episode / "camera/color/front" / f"{t:.6f}.jpg"), image
            )
            assert cv2.imwrite(
                str(episode / "camera/color/right" / f"{t + 0.01:.6f}.jpg"), image
            )
            state = np.zeros(8, dtype=float)
            action = np.zeros(8, dtype=float)
            action[0] = index * 0.01
            for name, vector in (("puppetRight", state), ("masterRight", action)):
                (
                    episode / "arm/jointState" / name / f"{t + 0.002:.6f}.json"
                ).write_text(json.dumps({"position": vector.tolist()}))
        frames = align_episode(episode)
        assert len(frames) >= 59
        report = audit_episode(episode, 30, 50, 0.06, 0.025)
        assert report["quality"] != "reject"
        states = np.zeros((60, 8), dtype=np.float32)
        actions = np.zeros((60, 8), dtype=np.float32)
        actions[:, 0] = np.arange(60) * 0.01
        write_norm_stats(Path(root), [(states, actions)])
        stats = json.loads((Path(root) / "norm_stats.json").read_text())["norm_stats"]
        assert stats["state"]["mean"] == [0.0] * 8
        assert len(stats["actions"]["q01"]) == 8
        hidden_bad_image = (
            episode / "camera/color/right" / f"{1000 + 20 / 30 + 0.01:.6f}.jpg"
        )
        hidden_bad_image.write_bytes(b"corrupt JPEG")
        assert audit_episode(episode, 30, 50, 0.06, 0.025)["quality"] != "reject"
        assert (
            audit_episode(episode, 30, 50, 0.06, 0.025, deep=True)["quality"]
            == "reject"
        )


def test_right_input_and_independent_training_dropout():
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.policies.nero_right_policy import (
        NeroRightInputs,
        NeroRightTrainingDropout,
    )

    sample = {
        "images": {
            "cam_high": np.ones((16, 16, 3), dtype=np.uint8),
            "cam_right_wrist": np.ones((16, 16, 3), dtype=np.uint8),
        },
        "state": np.ones(8, dtype=np.float32),
        "actions": np.ones((50, 8), dtype=np.float32),
        "prompt": "stack two items",
    }
    original = NeroRightInputs()(sample)
    dropout = NeroRightTrainingDropout(0.2, 0.3)
    with patch("numpy.random.random", side_effect=[0.1, 0.9]):
        wrist_only = dropout(original)
    assert not wrist_only["image_mask"]["right_wrist_0_rgb"]
    assert wrist_only["image_mask"]["base_0_rgb"]
    np.testing.assert_array_equal(wrist_only["state"], original["state"])
    with patch("numpy.random.random", side_effect=[0.9, 0.1]):
        state_only = dropout(original)
    assert state_only["image_mask"]["right_wrist_0_rgb"]
    assert np.all(state_only["state"] == 0)
    np.testing.assert_array_equal(state_only["actions"], original["actions"])
    inference = {k: v for k, v in original.items() if k != "actions"}
    with patch(
        "numpy.random.random",
        side_effect=AssertionError("Inference should not draw dropout"),
    ):
        assert dropout(inference) is inference


def test_two_camera_model_preprocessing():
    pytest.importorskip("openpi")
    import torch

    from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import (
        Observation,
        preprocess_observation,
    )

    obs = Observation(
        images={
            "base_0_rgb": torch.zeros(2, 224, 224, 3),
            "right_wrist_0_rgb": torch.zeros(2, 224, 224, 3),
        },
        image_masks={
            "base_0_rgb": torch.ones(2, dtype=torch.bool),
            "right_wrist_0_rgb": torch.ones(2, dtype=torch.bool),
        },
        state=torch.zeros(2, 32),
    )
    result = preprocess_observation(obs, train=False)
    assert set(result.images) == {"base_0_rgb", "right_wrist_0_rgb"}
    legacy = dataclasses.replace(
        obs,
        images={**obs.images, "left_wrist_0_rgb": torch.zeros(2, 224, 224, 3)},
        image_masks={
            **obs.image_masks,
            "left_wrist_0_rgb": torch.ones(2, dtype=torch.bool),
        },
    )
    assert len(preprocess_observation(legacy, train=False).images) == 3
    with pytest.raises(ValueError, match="Unsupported image keys"):
        preprocess_observation(
            dataclasses.replace(obs, images={"base_0_rgb": obs.images["base_0_rgb"]})
        )
