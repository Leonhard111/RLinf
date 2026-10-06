#!/usr/bin/env python3
"""Create a conservative, traceable widened OpenPI norm-stats profile."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _widen_block(
    block: dict[str, Any], *, scale: float, preserve_last_dim: bool
) -> dict[str, Any]:
    low = [float(value) for value in block["q01"]]
    high = [float(value) for value in block["q99"]]
    if len(low) != len(high) or not low:
        raise ValueError("q01 and q99 must be non-empty arrays of equal length")

    changed_dimensions = len(low) - int(preserve_last_dim)
    for index in range(changed_dimensions):
        if high[index] <= low[index]:
            raise ValueError(f"non-positive quantile span at dimension {index}")
        midpoint = (low[index] + high[index]) / 2.0
        half_span = (high[index] - low[index]) * scale / 2.0
        low[index] = midpoint - half_span
        high[index] = midpoint + half_span

    widened = copy.deepcopy(block)
    widened["q01"] = low
    widened["q99"] = high
    if "std" in widened:
        std = [float(value) for value in widened["std"]]
        if len(std) != len(low):
            raise ValueError("std and quantile dimensions do not match")
        for index in range(changed_dimensions):
            std[index] *= scale
        widened["std"] = std
    return widened


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Symmetrically expand state/action q01-q99 spans and std values "
            "without modifying the source file."
        )
    )
    parser.add_argument("input", type=Path, help="source norm_stats.json")
    parser.add_argument("output", type=Path, help="new norm_stats.json")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.20,
        help="span/std multiplier; 1.20 means 10%% extra margin on each side",
    )
    parser.add_argument(
        "--include-last-dim",
        action="store_true",
        help="also widen the final gripper dimension (preserved by default)",
    )
    parser.add_argument("--force", action="store_true", help="replace output")
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.scale < 1.0:
        raise ValueError("--scale must be >= 1.0")
    if source == output:
        raise ValueError("input and output must be different files")
    if output.exists() and not args.force:
        raise FileExistsError(f"output already exists: {output}")

    with source.open(encoding="utf-8") as stream:
        original = json.load(stream)
    stats = original.get("norm_stats")
    if not isinstance(stats, dict):
        raise ValueError("expected top-level 'norm_stats' object")

    adjusted = copy.deepcopy(original)
    for key in ("state", "actions"):
        if key not in stats:
            raise KeyError(f"missing norm_stats.{key}")
        adjusted["norm_stats"][key] = _widen_block(
            stats[key],
            scale=args.scale,
            preserve_last_dim=not args.include_last_dim,
        )

    _atomic_json_dump(output, adjusted)
    manifest = {
        "method": "symmetric q01-q99 span and std expansion about each midpoint",
        "scale": args.scale,
        "margin_each_side_fraction_of_original_span": (args.scale - 1.0) / 2.0,
        "modified_dimensions": [0, 1, 2, 3, 4, 5, 6]
        if not args.include_last_dim
        else list(range(8)),
        "preserved_gripper_dimension": not args.include_last_dim,
        "source": str(source),
        "source_sha256": _sha256(source),
        "output": str(output),
        "output_sha256": _sha256(output),
    }
    _atomic_json_dump(output.parent / "norm_stats_adjustment.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
