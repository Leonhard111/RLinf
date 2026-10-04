#!/usr/bin/env python3
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

"""Convert synchronized Nero capture episodes to a LeRobot v2.1 dataset.

The AgileX capture packer stores one HDF5 file per episode. Each HDF5 row is an
approximately 30 Hz synchronized observation and contains paths to the three
JPEG images plus leader/follower joint samples. This converter writes one 16D
absolute action per row. OpenPI's LeRobot loader is responsible for assembling
the future 50-step action horizon during training.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterator

import cv2
import h5py
import numpy as np

LOG = logging.getLogger("convert_nero_capture_to_lerobot")

CAMERA_DATASETS = {
    "cam_high": "camera/color/front",
    "cam_left_wrist": "camera/color/left",
    "cam_right_wrist": "camera/color/right",
}
ARM_ORDER = ("Left", "Right")
JOINT_NAMES = tuple(
    f"{side}_{name}"
    for side in ("left", "right")
    for name in (
        "joint_0",
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "gripper",
    )
)
PHYSICAL_DIM = 16
NULL_PROMPTS = {"", "null", "none", "nil", "n/a"}


@dataclasses.dataclass(frozen=True)
class EpisodeInspection:
    """Validated metadata for one packed Nero episode."""

    source_episode: int
    directory: Path
    hdf5_path: Path
    frame_count: int
    prompt: str
    measured_fps: float


def _episode_index(path: Path) -> int:
    suffix = path.name.removeprefix("episode")
    if path.name == f"episode{suffix}" and suffix.isdigit():
        return int(suffix)
    raise ValueError(f"Not an episode directory: {path}")


def discover_episode_paths(
    source: Path,
    requested: set[int] | None = None,
    *,
    strict: bool = False,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Discover numerically ordered packed episodes under ``source``.

    Args:
        source: Capture root containing ``episode<N>`` directories.
        requested: Optional explicit source episode indices.
        strict: Reject incomplete episode directories instead of reporting them.

    Returns:
        A pair of usable episode directories and skipped-entry reports.
    """
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)

    candidates: list[Path] = []
    for path in source.iterdir():
        if not path.is_dir() or not path.name.startswith("episode"):
            continue
        try:
            index = _episode_index(path)
        except ValueError:
            continue
        if requested is None or index in requested:
            candidates.append(path)
    candidates.sort(key=_episode_index)

    found = {_episode_index(path) for path in candidates}
    if requested is not None:
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(
                f"Requested source episodes do not exist: {missing}"
            )

    usable: list[Path] = []
    skipped: list[dict[str, Any]] = []
    for directory in candidates:
        packed = directory / f"{directory.name}.hdf5"
        if packed.is_file():
            usable.append(directory)
            continue
        skipped.append(
            {
                "source_episode": _episode_index(directory),
                "path": str(directory),
                "reason": f"missing packed file {packed.name}",
            }
        )

    if strict and skipped:
        raise FileNotFoundError(
            "Incomplete episode directories found: "
            + ", ".join(f"episode{x['source_episode']}" for x in skipped)
        )
    if not usable:
        raise FileNotFoundError(
            f"No packed episode<N>/episode<N>.hdf5 files in {source}"
        )
    return usable, skipped


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _stored_prompt(hdf5_file: h5py.File) -> str | None:
    key = "instructions/full_instructions/text"
    if key not in hdf5_file:
        return None
    prompts = [
        _decode_text(value).strip() for value in np.asarray(hdf5_file[key]).reshape(-1)
    ]
    prompts = [prompt for prompt in prompts if prompt.lower() not in NULL_PROMPTS]
    unique = list(dict.fromkeys(prompts))
    if not unique:
        return None
    if len(unique) != 1:
        raise ValueError(f"Expected one full-episode instruction, got {unique}")
    return unique[0]


def _position_key(role: str, side: str) -> str:
    return f"arm/jointStatePosition/{role}{side}"


def inspect_episode(
    directory: Path,
    *,
    prompt_override: str | None,
    action_source: str,
    expected_fps: float,
) -> EpisodeInspection:
    """Validate one packed episode without decoding every image."""
    directory = directory.expanduser().resolve()
    source_episode = _episode_index(directory)
    hdf5_path = directory / f"episode{source_episode}.hdf5"
    if action_source not in {"master", "puppet"}:
        raise ValueError(f"Unsupported action source: {action_source}")

    required = ["timestamp", "size", *CAMERA_DATASETS.values()]
    required.extend(_position_key("puppet", side) for side in ARM_ORDER)
    required.extend(_position_key(action_source, side) for side in ARM_ORDER)

    with h5py.File(hdf5_path, "r") as episode:
        missing = [key for key in required if key not in episode]
        if missing:
            raise KeyError(f"{hdf5_path} is missing datasets: {missing}")

        frame_count = int(np.asarray(episode["size"]).item())
        if frame_count <= 1:
            raise ValueError(f"{hdf5_path} has too few frames: {frame_count}")
        bad_lengths = {
            key: int(episode[key].shape[0])
            for key in required
            if key != "size" and episode[key].shape[0] != frame_count
        }
        if bad_lengths:
            raise ValueError(
                f"{hdf5_path} size={frame_count}, inconsistent datasets={bad_lengths}"
            )

        timestamps = np.asarray(episode["timestamp"], dtype=np.float64)
        deltas = np.diff(timestamps)
        if not np.isfinite(timestamps).all() or np.any(deltas <= 0):
            raise ValueError(f"{hdf5_path} timestamps are not finite and increasing")
        measured_fps = float(1.0 / np.median(deltas))
        if abs(measured_fps - expected_fps) > max(2.0, expected_fps * 0.1):
            raise ValueError(
                f"{hdf5_path} measured fps {measured_fps:.3f} is not close to "
                f"requested {expected_fps:.3f}"
            )

        for role in ("puppet", action_source):
            for side in ARM_ORDER:
                key = _position_key(role, side)
                values = np.asarray(episode[key], dtype=np.float64)
                if values.shape != (frame_count, 8):
                    raise ValueError(f"{hdf5_path}:{key} has shape {values.shape}")
                if not np.isfinite(values).all():
                    raise ValueError(f"{hdf5_path}:{key} contains NaN/Inf")

        stored_prompt = _stored_prompt(episode)

    prompt = (prompt_override or stored_prompt or "").strip()
    if prompt.lower() in NULL_PROMPTS:
        raise ValueError(
            f"{hdf5_path} has no usable instruction; pass a non-empty --prompt"
        )

    return EpisodeInspection(
        source_episode=source_episode,
        directory=directory,
        hdf5_path=hdf5_path,
        frame_count=frame_count,
        prompt=prompt,
        measured_fps=measured_fps,
    )


def _resolve_image_path(directory: Path, stored_path: Any) -> Path:
    text = _decode_text(stored_path).strip()
    path = Path(text)
    candidate = path.resolve() if path.is_absolute() else (directory / path).resolve()
    root = directory.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Image path escapes episode directory: {text}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def decode_image(
    directory: Path,
    stored_path: Any,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Read one capture JPEG and return a contiguous RGB uint8 image."""
    path = _resolve_image_path(directory, stored_path)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"Could not decode RGB image: {path}")
    if bgr.shape[:2] != (height, width):
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _vector16(episode: h5py.File, role: str, frame_index: int) -> np.ndarray:
    values = np.concatenate(
        [
            np.asarray(episode[_position_key(role, side)][frame_index])
            for side in ARM_ORDER
        ]
    )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (PHYSICAL_DIM,) or not np.isfinite(result).all():
        raise ValueError(
            f"Invalid {role} position at frame {frame_index}: shape={result.shape}"
        )
    return result


def iter_episode_frames(
    inspection: EpisodeInspection,
    *,
    action_source: str,
    image_width: int,
    image_height: int,
    max_frames: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield LeRobot frame dictionaries for one inspected episode."""
    frame_count = inspection.frame_count
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        frame_count = min(frame_count, max_frames)

    with h5py.File(inspection.hdf5_path, "r") as episode:
        for frame_index in range(frame_count):
            frame: dict[str, Any] = {
                "observation.state": _vector16(episode, "puppet", frame_index),
                "action": _vector16(episode, action_source, frame_index),
                "task": inspection.prompt,
            }
            for camera_name, source_key in CAMERA_DATASETS.items():
                frame[f"observation.images.{camera_name}"] = decode_image(
                    inspection.directory,
                    episode[source_key][frame_index],
                    width=image_width,
                    height=image_height,
                )
            yield frame


def _create_lerobot_dataset(
    *,
    output: Path,
    repo_id: str,
    fps: int,
    image_width: int,
    image_height: int,
    image_writer_threads: int,
) -> Any:
    try:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ModuleNotFoundError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "lerobot is required for conversion. Activate the AgileX lerobot "
            "environment before running this command."
        ) from exc

    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (PHYSICAL_DIM,),
            "names": list(JOINT_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (PHYSICAL_DIM,),
            "names": list(JOINT_NAMES),
        },
    }
    for camera_name in CAMERA_DATASETS:
        features[f"observation.images.{camera_name}"] = {
            "dtype": "image",
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channel"],
        }

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=output,
        robot_type="nero_aloha16",
        fps=fps,
        features=features,
        use_videos=True,
        image_writer_threads=image_writer_threads,
        image_writer_processes=0,
    )


def _add_frame(dataset: Any, frame: dict[str, Any]) -> None:
    # LeRobot 0.3.x exposes add_frame(frame, task, ...), while older releases
    # and 0.4+ read task from frame["task"]. Inspecting the signature avoids
    # importing the full RLinf package in AgileX's lightweight lerobot env.
    payload = dict(frame)
    task = payload.pop("task")
    try:
        takes_task = "task" in inspect.signature(type(dataset).add_frame).parameters
    except (TypeError, ValueError):
        takes_task = False
    if takes_task:
        dataset.add_frame(payload, task=task)
    else:
        payload["task"] = task
        dataset.add_frame(payload)


def _write_report(output: Path, report: dict[str, Any]) -> None:
    path = output / "conversion_report.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--repo-id",
        help="LeRobot metadata id. Defaults to local/<output directory name>.",
    )
    parser.add_argument(
        "--prompt",
        help="Override the capture instruction for every converted episode.",
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        type=int,
        help="Optional source episode indices, for example --episodes 0 1 5.",
    )
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument(
        "--max-frames-per-episode",
        type=int,
        help="Debug/smoke-test truncation; omit for production conversion.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument(
        "--action-source",
        choices=("master", "puppet"),
        default="master",
        help="Use leader positions by default, matching the existing Nero pipeline.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when any selected episode directory lacks its packed HDF5.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate metadata and sample images without writing a dataset.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.image_width <= 0 or args.image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if args.limit_episodes is not None and args.limit_episodes <= 0:
        raise ValueError("--limit-episodes must be positive")

    requested = set(args.episodes) if args.episodes else None
    directories, skipped = discover_episode_paths(
        args.source, requested, strict=args.strict
    )
    if args.limit_episodes is not None:
        directories = directories[: args.limit_episodes]

    inspections: list[EpisodeInspection] = []
    for directory in directories:
        inspection = inspect_episode(
            directory,
            prompt_override=args.prompt,
            action_source=args.action_source,
            expected_fps=float(args.fps),
        )
        # Decode representative images now so validate-only catches broken paths.
        with h5py.File(inspection.hdf5_path, "r") as episode:
            for source_key in CAMERA_DATASETS.values():
                decode_image(
                    inspection.directory,
                    episode[source_key][0],
                    width=args.image_width,
                    height=args.image_height,
                )
        inspections.append(inspection)
        LOG.info(
            "validated episode%d frames=%d measured_fps=%.3f",
            inspection.source_episode,
            inspection.frame_count,
            inspection.measured_fps,
        )

    if skipped:
        LOG.warning(
            "skipping %d incomplete episode directories: %s", len(skipped), skipped
        )
    if args.validate_only:
        LOG.info(
            "validation complete: episodes=%d frames=%d",
            len(inspections),
            sum(item.frame_count for item in inspections),
        )
        return 0

    if args.output is None:
        raise ValueError("--output is required unless --validate-only is used")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Choose a new directory; this tool "
            "never overwrites datasets."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    repo_id = args.repo_id or f"local/{output.name}"
    dataset = _create_lerobot_dataset(
        output=output,
        repo_id=repo_id,
        fps=args.fps,
        image_width=args.image_width,
        image_height=args.image_height,
        image_writer_threads=args.image_writer_threads,
    )

    report: dict[str, Any] = {
        "source": str(args.source.expanduser().resolve()),
        "output": str(output),
        "repo_id": repo_id,
        "fps": args.fps,
        "image_size": [args.image_height, args.image_width, 3],
        "state": "puppetLeft.position + puppetRight.position",
        "action": f"{args.action_source}Left.position + {args.action_source}Right.position",
        "action_contract": (
            "one absolute 16D action per 30Hz row; OpenPI constructs the future "
            "action horizon through LeRobot delta_timestamps"
        ),
        "joint_order": list(JOINT_NAMES),
        "skipped": skipped,
        "episodes": [],
    }

    total_frames = 0
    for output_episode, inspection in enumerate(inspections):
        written = 0
        for frame in iter_episode_frames(
            inspection,
            action_source=args.action_source,
            image_width=args.image_width,
            image_height=args.image_height,
            max_frames=args.max_frames_per_episode,
        ):
            _add_frame(dataset, frame)
            written += 1
        dataset.save_episode()
        total_frames += written
        report["episodes"].append(
            {
                "output_episode": output_episode,
                "source_episode": inspection.source_episode,
                "frames": written,
                "source_frames": inspection.frame_count,
                "prompt": inspection.prompt,
                "measured_fps": inspection.measured_fps,
                "truncated": written != inspection.frame_count,
            }
        )
        _write_report(output, report)
        LOG.info(
            "converted %d/%d: source=episode%d output=episode%d frames=%d",
            output_episode + 1,
            len(inspections),
            inspection.source_episode,
            output_episode,
            written,
        )

    LOG.info(
        "conversion complete: episodes=%d frames=%d", len(inspections), total_frames
    )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(main())
