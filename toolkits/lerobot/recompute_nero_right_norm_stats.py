#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Recompute Nero right-arm OpenPI stats from an existing LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _stats(values: np.ndarray) -> dict[str, list[float]]:
    """Summarize each physical state or action dimension."""
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def compute_norm_stats(
    dataset: Path, *, action_space: str, horizon: int
) -> tuple[dict, int, int]:
    """Compute pooled 8-D stats over terminal-padded action chunks.

    The delta mode reproduces prepare_nero_right_dataset.py. Absolute mode
    leaves the first seven joint targets unchanged; the gripper is absolute
    in both modes.
    """
    if action_space not in {"delta", "absolute"}:
        raise ValueError("action_space must be 'delta' or 'absolute'")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    dataset = dataset.expanduser().resolve()
    with (dataset / "meta/info.json").open(encoding="utf-8") as stream:
        info = json.load(stream)
    if info.get("robot_type") != "nero_right8":
        raise ValueError("expected a Nero right-arm LeRobot dataset")
    for feature in ("observation.state", "action"):
        if info.get("features", {}).get(feature, {}).get("shape") != [8]:
            raise ValueError(f"expected 8-D {feature}")

    episode_paths = sorted((dataset / "data").rglob("*.parquet"))
    if not episode_paths or len(episode_paths) != info.get("total_episodes"):
        raise ValueError("Parquet episode count does not match meta/info.json")

    states: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    for path in episode_paths:
        table = pq.read_table(path, columns=["observation.state", "action"])
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if (
            state.ndim != 2
            or state.shape[1:] != (8,)
            or action.shape != state.shape
            or not np.isfinite(state).all()
            or not np.isfinite(action).all()
        ):
            raise ValueError(f"invalid state/action data in {path}")

        indices = np.minimum(
            np.arange(len(action))[:, None] + np.arange(horizon)[None, :],
            len(action) - 1,
        )
        chunk = action[indices].copy()
        if action_space == "delta":
            chunk[..., :7] -= state[:, None, :7]
        states.append(state)
        action_chunks.append(chunk.reshape(-1, 8))

    all_states = np.concatenate(states, axis=0)
    if len(all_states) != info.get("total_frames"):
        raise ValueError("Parquet frame count does not match meta/info.json")
    all_actions = np.concatenate(action_chunks, axis=0)
    result = {
        "norm_stats": {
            "state": _stats(all_states),
            "actions": _stats(all_actions),
        }
    }
    return result, len(episode_paths), len(all_states)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-space", choices=("delta", "absolute"), required=True)
    parser.add_argument("--horizon", type=int, default=50)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    result, episodes, frames = compute_norm_stats(
        args.dataset, action_space=args.action_space, horizon=args.horizon
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(result, stream, indent=2)
        stream.write("\n")
    try:
        os.chmod(temporary, 0o644)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "dataset": str(args.dataset.expanduser().resolve()),
                "output": str(output),
                "action_space": args.action_space,
                "horizon": args.horizon,
                "episodes": episodes,
                "frames": frames,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
