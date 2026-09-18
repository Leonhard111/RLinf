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

"""Aggregate one day's RLinf/pi05 JSONL latency records across processes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path, help="directory containing latency JSONL")
    parser.add_argument(
        "--date",
        default=datetime.now().astimezone().strftime("%Y%m%d"),
        help="date prefix in YYYYMMDD form (default: today)",
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help="skip the first N records in each process file (for cold start)",
    )
    parser.add_argument("--output", type=Path, help="summary output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.skip_first < 0:
        raise SystemExit("--skip-first must be non-negative")
    files = sorted(args.log_dir.expanduser().glob(f"{args.date}_*.jsonl"))
    values: dict[str, list[float]] = defaultdict(list)
    file_records = 0
    used_records = 0
    bad_records = 0

    for path in files:
        skipped = 0
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                file_records += 1
                try:
                    row = json.loads(line)
                    component = str(row["component"])
                    metrics = row["metrics_ms"]
                    if not isinstance(metrics, dict):
                        raise TypeError("metrics_ms is not a mapping")
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    bad_records += 1
                    continue
                if skipped < args.skip_first:
                    skipped += 1
                    continue
                used_records += 1
                for metric, value in metrics.items():
                    number = float(value)
                    if math.isfinite(number) and number >= 0.0:
                        values[f"{component}.{metric}"].append(number)

    lines = [
        f"date: {args.date}",
        f"source_dir: {args.log_dir.expanduser().resolve()}",
        f"files: {len(files)}",
        f"records_total: {file_records}",
        f"records_used: {used_records}",
        f"records_invalid: {bad_records}",
        f"skip_first_per_file: {args.skip_first}",
        "",
        "component.metric_ms | count | mean | std | min | p50 | p90 | p95 | max",
    ]
    for name in sorted(values):
        samples = values[name]
        std = statistics.pstdev(samples) if len(samples) > 1 else 0.0
        lines.append(
            f"{name} | {len(samples)} | {statistics.fmean(samples):.3f} | "
            f"{std:.3f} | {min(samples):.3f} | {_percentile(samples, 0.50):.3f} | "
            f"{_percentile(samples, 0.90):.3f} | "
            f"{_percentile(samples, 0.95):.3f} | {max(samples):.3f}"
        )

    result = "\n".join(lines) + "\n"
    output = args.output or (
        args.log_dir.expanduser() / f"{args.date}_all_components_summary.log"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result, encoding="utf-8")
    print(result, end="")
    print(f"summary_written: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
