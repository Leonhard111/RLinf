# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from toolkits.lerobot.convert_nero_capture_to_lerobot import (
    CAMERA_DATASETS,
    discover_episode_paths,
    inspect_episode,
    iter_episode_frames,
)


def _write_episode(root: Path, *, prompt: str = "stack the blocks") -> Path:
    directory = root / "episode3"
    directory.mkdir()
    image_paths: dict[str, list[bytes]] = {}
    for camera_name, source_key in CAMERA_DATASETS.items():
        camera = source_key.rsplit("/", 1)[-1]
        relative_paths: list[bytes] = []
        for frame_index in range(3):
            relative = Path("camera") / "color" / camera / f"{frame_index}.jpg"
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            bgr = np.full((12, 16, 3), frame_index + len(camera_name), dtype=np.uint8)
            assert cv2.imwrite(str(path), bgr)
            relative_paths.append(str(relative).encode())
        image_paths[source_key] = relative_paths

    with h5py.File(directory / "episode3.hdf5", "w") as episode:
        episode.create_dataset("size", data=3)
        episode.create_dataset("timestamp", data=np.arange(3) / 30.0)
        for role, offset in (("puppet", 0.0), ("master", 1.0)):
            for side, side_offset in (("Left", 0.0), ("Right", 10.0)):
                values = np.arange(24, dtype=np.float64).reshape(3, 8)
                episode.create_dataset(
                    f"arm/jointStatePosition/{role}{side}",
                    data=values + offset + side_offset,
                )
        text_dtype = h5py.string_dtype(encoding="utf-8")
        episode.create_dataset(
            "instructions/full_instructions/text",
            data=np.asarray([prompt], dtype=object),
            dtype=text_dtype,
        )
        for source_key, values in image_paths.items():
            episode.create_dataset(source_key, data=values)
    return directory


def test_inspect_and_iterate_standard_16d_frames(tmp_path: Path) -> None:
    directory = _write_episode(tmp_path)
    inspection = inspect_episode(
        directory,
        prompt_override=None,
        action_source="master",
        expected_fps=30.0,
    )

    frames = list(
        iter_episode_frames(
            inspection,
            action_source="master",
            image_width=8,
            image_height=6,
        )
    )

    assert inspection.source_episode == 3
    assert inspection.prompt == "stack the blocks"
    assert len(frames) == 3
    assert frames[0]["observation.state"].shape == (16,)
    assert frames[0]["action"].shape == (16,)
    assert frames[0]["action"][0] == pytest.approx(1.0)
    assert frames[0]["action"][8] == pytest.approx(11.0)
    for camera_name in CAMERA_DATASETS:
        image = frames[0][f"observation.images.{camera_name}"]
        assert image.shape == (6, 8, 3)
        assert image.dtype == np.uint8


def test_null_instruction_requires_prompt_override(tmp_path: Path) -> None:
    directory = _write_episode(tmp_path, prompt="null")
    with pytest.raises(ValueError, match="--prompt"):
        inspect_episode(
            directory,
            prompt_override=None,
            action_source="master",
            expected_fps=30.0,
        )

    inspection = inspect_episode(
        directory,
        prompt_override="override task",
        action_source="master",
        expected_fps=30.0,
    )
    assert inspection.prompt == "override task"


def test_discovery_reports_incomplete_episode(tmp_path: Path) -> None:
    _write_episode(tmp_path)
    (tmp_path / "episode4").mkdir()

    usable, skipped = discover_episode_paths(tmp_path)

    assert [_path.name for _path in usable] == ["episode3"]
    assert skipped == [
        {
            "source_episode": 4,
            "path": str((tmp_path / "episode4").resolve()),
            "reason": "missing packed file episode4.hdf5",
        }
    ]
    with pytest.raises(FileNotFoundError, match="episode4"):
        discover_episode_paths(tmp_path, strict=True)
