# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Tests for Nero right-arm absolute-action statistics and transforms."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from toolkits.lerobot.recompute_nero_right_norm_stats import (
    compute_norm_stats,
    main,
)


def _dataset(root: Path) -> Path:
    data_dir = root / "data/chunk-000"
    meta_dir = root / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir()
    state = [[1.0] * 7 + [0.1], [2.0] * 7 + [0.2]]
    action = [[3.0] * 7 + [0.3], [4.0] * 7 + [0.4]]
    vector_type = pa.list_(pa.float32(), 8)
    table = pa.table(
        {
            "observation.state": pa.array(state, type=vector_type),
            "action": pa.array(action, type=vector_type),
        }
    )
    pq.write_table(table, data_dir / "episode_000000.parquet")
    info = {
        "robot_type": "nero_right8",
        "total_episodes": 1,
        "total_frames": 2,
        "features": {
            "observation.state": {"shape": [8]},
            "action": {"shape": [8]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return root


def test_absolute_and_delta_stats_share_state_and_gripper(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    absolute, episodes, frames = compute_norm_stats(
        dataset, action_space="absolute", horizon=3
    )
    delta, _, _ = compute_norm_stats(dataset, action_space="delta", horizon=3)
    absolute_stats = absolute["norm_stats"]
    delta_stats = delta["norm_stats"]

    assert (episodes, frames) == (1, 2)
    assert absolute_stats["state"] == delta_stats["state"]
    np.testing.assert_allclose(absolute_stats["actions"]["mean"][:7], [23 / 6] * 7)
    np.testing.assert_allclose(delta_stats["actions"]["mean"][:7], [14 / 6] * 7)
    for key in ("mean", "std", "q01", "q99"):
        assert absolute_stats["actions"][key][-1] == pytest.approx(
            delta_stats["actions"][key][-1]
        )


def test_cli_writes_readable_stats_and_refuses_overwrite(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset")
    output = tmp_path / "profile/norm_stats.json"
    argv = [
        "recompute_nero_right_norm_stats.py",
        "--dataset",
        str(dataset),
        "--output",
        str(output),
        "--action-space",
        "absolute",
        "--horizon",
        "3",
    ]
    with patch("sys.argv", argv):
        main()
    assert stat.S_IMODE(output.stat().st_mode) == 0o644
    assert json.loads(output.read_text())["norm_stats"]["actions"]["mean"][0] == (
        pytest.approx(23 / 6)
    )
    with patch("sys.argv", argv), pytest.raises(FileExistsError):
        main()


def test_action_config_selects_matching_input_and_output_transforms() -> None:
    from openpi import transforms

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    for use_delta in (True, False):
        config = get_openpi_config(
            "pi05_nero_right",
            data_kwargs={"use_delta_joint_actions": use_delta},
        )
        data = config.data.create(config.assets_dirs, config.model)
        input_types = tuple(map(type, data.data_transforms.inputs))
        output_types = tuple(map(type, data.data_transforms.outputs))
        assert (transforms.DeltaActions in input_types) is use_delta
        assert (transforms.AbsoluteActions in output_types) is use_delta

        state = np.arange(8, dtype=np.float32)
        actions = np.tile(state + 0.5, (50, 1))
        raw = {
            "images": {
                "cam_high": np.zeros((16, 16, 3), dtype=np.uint8),
                "cam_right_wrist": np.zeros((16, 16, 3), dtype=np.uint8),
            },
            "state": state,
            "actions": actions,
        }
        transformed = raw
        for transform in data.data_transforms.inputs:
            transformed = transform(transformed)
        expected = actions.copy()
        if use_delta:
            expected[:, :7] -= state[:7]
        np.testing.assert_allclose(transformed["actions"], expected)
        restored = {"actions": transformed["actions"], "state": state}
        for transform in data.data_transforms.outputs:
            restored = transform(restored)
        np.testing.assert_allclose(restored["actions"], actions)
