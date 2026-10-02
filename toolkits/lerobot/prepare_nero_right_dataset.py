# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Audit and convert timestamped Nero captures into compact right-arm LeRobot data.

Raw captures are never modified. Run ``audit`` first, review its CSV/contact
sheets, mark only successful episodes as ``accept``, then run ``convert``.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import inspect
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

LOG = logging.getLogger("nero_right_dataset")
STREAMS = {
    "front": "camera/color/front",
    "wrist": "camera/color/right",
    "state": "arm/jointState/puppetRight",
    "action": "arm/jointState/masterRight",
}
JOINT_NAMES = [f"right_joint_{i}" for i in range(1, 8)] + ["right_gripper"]


@dataclass(frozen=True)
class TimedFile:
    time: float
    path: Path


@dataclass(frozen=True)
class AlignedFrame:
    time: float
    front: Path
    wrist: Path
    state: Path
    action: Path


def _episode_number(path: Path) -> int:
    match = re.fullmatch(r"episode(\d+)", path.name)
    if match is None:
        raise ValueError(f"Not an episode directory: {path}")
    return int(match.group(1))


def episode_dirs(source: Path, start: int, stop: int | None) -> list[Path]:
    episodes = []
    for path in source.iterdir():
        if path.is_dir() and re.fullmatch(r"episode\d+", path.name):
            number = _episode_number(path)
            if number >= start and (stop is None or number <= stop):
                episodes.append(path)
    return sorted(episodes, key=_episode_number)


def timed_files(directory: Path, suffix: str) -> list[TimedFile]:
    files = []
    if not directory.is_dir():
        return files
    for path in directory.iterdir():
        if path.suffix.lower() != suffix:
            continue
        try:
            timestamp = float(path.stem)
        except ValueError:
            continue
        if np.isfinite(timestamp):
            files.append(TimedFile(timestamp, path))
    return sorted(files, key=lambda item: item.time)


def _closest(
    stream: list[TimedFile], times: list[float], timestamp: float, max_skew: float
) -> Path | None:
    index = bisect.bisect_left(times, timestamp)
    choices = stream[max(0, index - 1) : min(len(stream), index + 1)]
    if not choices:
        return None
    closest = min(choices, key=lambda item: abs(item.time - timestamp))
    return closest.path if abs(closest.time - timestamp) <= max_skew else None


def align_episode(
    episode: Path,
    *,
    fps: int = 30,
    max_camera_skew: float = 0.06,
    max_joint_skew: float = 0.025,
) -> list[AlignedFrame]:
    streams = {
        name: timed_files(
            episode / directory, ".jpg" if name in ("front", "wrist") else ".json"
        )
        for name, directory in STREAMS.items()
    }
    if any(not files for files in streams.values()):
        return []
    times = {name: [item.time for item in files] for name, files in streams.items()}
    start = max(files[0].time for files in streams.values())
    end = min(files[-1].time for files in streams.values())
    frames = []
    if end <= start:
        return frames
    for index in range(int((end - start) * fps) + 1):
        timestamp = start + index / fps
        front = _closest(streams["front"], times["front"], timestamp, max_camera_skew)
        wrist = _closest(streams["wrist"], times["wrist"], timestamp, max_camera_skew)
        state = _closest(streams["state"], times["state"], timestamp, max_joint_skew)
        action = _closest(streams["action"], times["action"], timestamp, max_joint_skew)
        if all(path is not None for path in (front, wrist, state, action)):
            frames.append(AlignedFrame(timestamp, front, wrist, state, action))
    return frames


def read_joint(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    vector = np.asarray(data["position"], dtype=np.float32)
    if vector.shape != (8,) or not np.isfinite(vector).all():
        raise ValueError(f"Invalid 8D joint position: {path}")
    return vector


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def resize_with_pad(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Aspect-preserving bilinear resize + black center pad, like OpenPI."""
    height, width = image.shape[:2]
    ratio = max(width / size, height / size)
    new_h, new_w = int(height / ratio), int(width / ratio)
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[top : top + new_h, left : left + new_w] = scaled
    return canvas


def _sample_indices(length: int) -> list[int]:
    return sorted({0, length // 2, length - 1})


def _visual_metrics(path: Path) -> tuple[float, float]:
    gray = cv2.cvtColor(read_image(path), cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())


def audit_episode(
    episode: Path,
    fps: int,
    min_frames: int,
    max_camera_skew: float,
    max_joint_skew: float,
    deep: bool = False,
) -> dict:
    name = episode.name
    frames = align_episode(
        episode, fps=fps, max_camera_skew=max_camera_skew, max_joint_skew=max_joint_skew
    )
    flags = []
    hard_fail = False
    if len(frames) < min_frames:
        flags.append("too_few_aligned_frames")
        hard_fail = True
    median_fps = 0.0
    if len(frames) >= 2:
        deltas = np.diff([item.time for item in frames])
        if np.any(deltas <= 0):
            flags.append("nonmonotonic_time")
            hard_fail = True
        else:
            median_fps = float(1.0 / np.median(deltas))
            if not 0.9 * fps <= median_fps <= 1.1 * fps:
                flags.append("fps_out_of_range")
                hard_fail = True
            if np.max(deltas) > 2 / fps + 0.002:
                flags.append("aligned_gap_over_two_steps")
                hard_fail = True
    if frames:
        front_reuse = 1 - len({frame.front for frame in frames}) / len(frames)
        wrist_reuse = 1 - len({frame.wrist for frame in frames}) / len(frames)
        if front_reuse > 0.2 or wrist_reuse > 0.2:
            flags.append("camera_reuse_over_20_percent")
            hard_fail = True
    if frames:
        try:
            indices = _sample_indices(len(frames))
            states = np.stack([read_joint(frames[i].state) for i in indices])
            actions = np.stack([read_joint(frames[i].action) for i in indices])
            motion = float(np.max(np.ptp(actions[:, :7], axis=0)))
            if motion < 0.05:
                flags.append("low_joint_motion_review")
            if float(np.max(np.abs(actions[:, :7] - states[:, :7]))) > 0.5:
                flags.append("large_master_puppet_gap_review")
            for index in indices:
                for camera_name, camera in (
                    ("front", frames[index].front),
                    ("wrist", frames[index].wrist),
                ):
                    brightness, blur = _visual_metrics(camera)
                    if brightness < 25 or brightness > 235:
                        flags.append("exposure_review")
                    # Wrist views contain large textureless table regions even
                    # when optically sharp; use a separate conservative floor.
                    if blur < (20 if camera_name == "front" else 3):
                        flags.append(f"{camera_name}_blur_review")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            flags.append(f"decode_error:{type(exc).__name__}")
            hard_fail = True
        if deep and not hard_fail:
            try:
                for path in {
                    item for frame in frames for item in (frame.front, frame.wrist)
                }:
                    read_image(path)
                for path in {
                    item for frame in frames for item in (frame.state, frame.action)
                }:
                    read_joint(path)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                flags.append(f"deep_decode_error:{type(exc).__name__}")
                hard_fail = True
    flags = sorted(set(flags))
    return {
        "episode": name,
        "frames": len(frames),
        "duration_s": round(frames[-1].time - frames[0].time, 3)
        if len(frames) >= 2
        else 0,
        "median_fps": round(median_fps, 2),
        "front_reuse_pct": round(100 * front_reuse, 1) if frames else 0,
        "wrist_reuse_pct": round(100 * wrist_reuse, 1) if frames else 0,
        "quality": "reject" if hard_fail else "review" if flags else "pass",
        "flags": ";".join(flags),
        # Task success cannot be inferred from these telemetry checks.
        "decision": "reject" if hard_fail else "",
    }


def contact_sheet(frames: list[AlignedFrame], output: Path, size: int = 224) -> None:
    if not frames:
        return
    columns = []
    for index in _sample_indices(len(frames)):
        frame = frames[index]
        front = resize_with_pad(read_image(frame.front), size)
        wrist = resize_with_pad(read_image(frame.wrist), size)
        columns.append(np.vstack((front, wrist)))
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), np.hstack(columns)):
        raise OSError(f"Failed to write {output}")


def _dataset(output: Path, repo_id: str, fps: int, size: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        key: {"dtype": "float32", "shape": (8,), "names": JOINT_NAMES}
        for key in ("observation.state", "action")
    }
    for camera in ("cam_high", "cam_right_wrist"):
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (size, size, 3),
            "names": ["height", "width", "channel"],
        }
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=output,
        robot_type="nero_right8",
        fps=fps,
        features=features,
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
    )


def _add_frame(dataset, payload: dict, task: str) -> None:
    if "task" in inspect.signature(type(dataset).add_frame).parameters:
        dataset.add_frame(payload, task=task)
    else:
        dataset.add_frame({**payload, "task": task})


def _stats(array: np.ndarray) -> dict:
    return {
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "q01": np.quantile(array, 0.01, axis=0).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).tolist(),
    }


def write_norm_stats(
    output: Path, episodes: list[tuple[np.ndarray, np.ndarray]], horizon: int = 50
) -> None:
    states = np.concatenate([item[0] for item in episodes], axis=0)
    delta_chunks = []
    for state, action in episodes:
        indices = np.minimum(
            np.arange(len(action))[:, None] + np.arange(horizon)[None, :],
            len(action) - 1,
        )
        chunk = action[indices].copy()
        chunk[..., :7] -= state[:, None, :7]
        delta_chunks.append(chunk.reshape(-1, 8))
    actions = np.concatenate(delta_chunks, axis=0)
    stats = {"norm_stats": {"state": _stats(states), "actions": _stats(actions)}}
    (output / "norm_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )


def convert(args: argparse.Namespace) -> None:
    output = args.output.expanduser().resolve()
    source = args.source.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output exists; refusing to overwrite: {output}")
    if output == source or source in output.parents:
        raise ValueError("Output must be outside the raw capture directory")
    with args.selection.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    accepted = [row for row in rows if row["decision"].strip().lower() == "accept"]
    if not accepted:
        raise ValueError(
            "Selection CSV has no accepted episodes; review contact sheets first"
        )
    if len({row["episode"] for row in accepted}) != len(accepted):
        raise ValueError("Selection CSV contains duplicate accepted episodes")
    episodes = []
    for row in accepted:
        episode = source / row["episode"]
        if (
            episode.resolve().parent != source
            or not episode.is_dir()
            or _episode_number(episode) < args.start_episode
        ):
            raise ValueError(f"Invalid accepted episode: {episode}")
        quality = audit_episode(
            episode,
            args.fps,
            args.min_frames,
            args.max_camera_skew,
            args.max_joint_skew,
            deep=True,
        )
        if quality["quality"] == "reject":
            raise ValueError(
                f"Accepted episode fails structural audit: {episode}: {quality['flags']}"
            )
        frames = align_episode(
            episode,
            fps=args.fps,
            max_camera_skew=args.max_camera_skew,
            max_joint_skew=args.max_joint_skew,
        )
        if len(frames) < args.min_frames:
            raise ValueError(f"Accepted episode has too few aligned frames: {episode}")
        episodes.append((episode, frames))
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = _dataset(output, args.repo_id, args.fps, args.size)
    stats_episodes = []
    manifest = []
    for index, (episode, frames) in enumerate(episodes):
        states, actions = [], []
        for frame in frames:
            state, action = read_joint(frame.state), read_joint(frame.action)
            payload = {
                "observation.state": state,
                "action": action,
                "observation.images.cam_high": cv2.cvtColor(
                    resize_with_pad(read_image(frame.front), args.size),
                    cv2.COLOR_BGR2RGB,
                ),
                "observation.images.cam_right_wrist": cv2.cvtColor(
                    resize_with_pad(read_image(frame.wrist), args.size),
                    cv2.COLOR_BGR2RGB,
                ),
            }
            _add_frame(dataset, payload, args.task)
            states.append(state)
            actions.append(action)
        dataset.save_episode()
        stats_episodes.append((np.stack(states), np.stack(actions)))
        manifest.append(
            {"source": episode.name, "output_episode": index, "frames": len(frames)}
        )
        LOG.info(
            "converted %s -> episode%d (%d frames)", episode.name, index, len(frames)
        )
    write_norm_stats(output, stats_episodes)
    (output / "conversion_manifest.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "repo_id": args.repo_id,
                "fps": args.fps,
                "size": args.size,
                "task": args.task,
                "episodes": manifest,
                "observation": "puppetRight.position",
                "action": "masterRight.position",
                "alignment": "uniform 30Hz timeline; nearest front, right wrist, puppetRight, masterRight",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "preview", "convert"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--source", type=Path, required=True)
        cmd.add_argument("--start-episode", type=int, default=15)
        cmd.add_argument("--fps", type=int, default=30)
        cmd.add_argument("--min-frames", type=int, default=50)
        cmd.add_argument("--max-camera-skew", type=float, default=0.06)
        cmd.add_argument("--max-joint-skew", type=float, default=0.025)
        cmd.add_argument("--size", type=int, default=224)
    sub.choices["audit"].add_argument("--stop-episode", type=int)
    sub.choices["audit"].add_argument("--report-dir", type=Path, required=True)
    sub.choices["audit"].add_argument("--contact-sheets", action="store_true")
    sub.choices["audit"].add_argument(
        "--deep", action="store_true", help="Decode every selected image/joint file"
    )
    sub.choices["preview"].add_argument("--episode", type=int, required=True)
    sub.choices["preview"].add_argument("--output", type=Path, required=True)
    sub.choices["convert"].add_argument("--selection", type=Path, required=True)
    sub.choices["convert"].add_argument("--output", type=Path, required=True)
    sub.choices["convert"].add_argument(
        "--repo-id", default="local/nero_right_stack2item"
    )
    sub.choices["convert"].add_argument("--task", required=True)
    args = parser.parse_args()
    if (
        args.fps <= 0
        or args.size <= 0
        or args.min_frames <= 0
        or args.start_episode < 0
    ):
        parser.error(
            "fps, size, and min-frames must be positive; start-episode must be nonnegative"
        )
    if args.max_camera_skew < 0 or args.max_joint_skew < 0:
        parser.error("Maximum skew values must be nonnegative")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "audit":
        report = args.report_dir.expanduser().resolve()
        report.mkdir(parents=True, exist_ok=True)
        selection = report / "selection.csv"
        if selection.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing review decisions: {selection}"
            )
        rows = []
        for episode in episode_dirs(args.source, args.start_episode, args.stop_episode):
            row = audit_episode(
                episode,
                args.fps,
                args.min_frames,
                args.max_camera_skew,
                args.max_joint_skew,
                deep=args.deep,
            )
            rows.append(row)
            LOG.info("%s: %s %s", episode.name, row["quality"], row["flags"])
            if args.contact_sheets and row["frames"]:
                contact_sheet(
                    align_episode(
                        episode,
                        fps=args.fps,
                        max_camera_skew=args.max_camera_skew,
                        max_joint_skew=args.max_joint_skew,
                    ),
                    report / "contact_sheets" / f"{episode.name}.jpg",
                    args.size,
                )
        if not rows:
            raise ValueError("No episodes in requested range")
        with selection.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        LOG.info("Wrote %d audit rows to %s", len(rows), selection)
    elif args.command == "preview":
        episode = args.source / f"episode{args.episode}"
        frames = align_episode(
            episode,
            fps=args.fps,
            max_camera_skew=args.max_camera_skew,
            max_joint_skew=args.max_joint_skew,
        )
        if not frames:
            raise ValueError(f"No aligned frames: {episode}")
        contact_sheet(frames, args.output, args.size)
        LOG.info(
            "Saved front (top) + right wrist (bottom), 3 time points: %s", args.output
        )
    else:
        convert(args)


if __name__ == "__main__":
    main()
