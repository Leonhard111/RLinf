# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Low-overhead, process-local latency telemetry for real-robot runs.

The measured thread only enqueues immutable records.  JSONL writes and running
statistics are handled by a daemon thread so profiling does not add synchronous
disk I/O to the inference/control critical path.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import queue
import socket
import statistics
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

_TRUE_VALUES = {"1", "true", "yes", "on"}
_STOP = object()
_LOGGERS: dict[str, "LatencyLogger"] = {}
_LOGGERS_LOCK = threading.Lock()


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return math.nan
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


class LatencyLogger:
    """Asynchronously persist latency samples and per-metric statistics."""

    def __init__(
        self,
        component: str,
        *,
        enabled: bool | None = None,
        log_dir: str | os.PathLike[str] | None = None,
        summary_every: int | None = None,
    ) -> None:
        self.component = str(component)
        self.enabled = (
            _env_enabled("RLINF_LATENCY_LOG") if enabled is None else bool(enabled)
        )
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self._sequence = 0
        self._sequence_lock = threading.Lock()
        self._closed = False
        self._queue: queue.SimpleQueue[object] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None

        configured_dir = log_dir or os.environ.get(
            "RLINF_LATENCY_LOG_DIR", "logs/latency"
        )
        self.log_dir = Path(configured_dir).expanduser()
        configured_summary_every = summary_every or int(
            os.environ.get("RLINF_LATENCY_SUMMARY_EVERY", "10")
        )
        self.summary_every = max(1, configured_summary_every)

        date = datetime.now().astimezone().strftime("%Y%m%d")
        stem = "_".join((
            date,
            _safe_name(self.component),
            _safe_name(self.hostname),
            str(self.pid),
        ))
        self.detail_path = self.log_dir / f"{stem}.jsonl"
        self.summary_path = self.log_dir / f"{stem}_summary.log"

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(
                target=self._writer_loop,
                name=f"latency-log-{_safe_name(self.component)}",
                daemon=True,
            )
            self._thread.start()
            atexit.register(self.close)

    def next_trace_id(self) -> str:
        """Return a process-unique trace id without touching the filesystem."""

        with self._sequence_lock:
            self._sequence += 1
            sequence = self._sequence
        return f"{self.hostname}-{self.pid}-{sequence}"

    def record(
        self,
        metrics_ms: Mapping[str, float],
        *,
        trace_id: str | int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Enqueue one successful sample; invalid/non-finite metrics are rejected."""

        if not self.enabled or self._closed:
            return
        normalized_metrics = {
            str(key): float(value) for key, value in metrics_ms.items()
        }
        if not normalized_metrics or not all(
            math.isfinite(value) and value >= 0.0
            for value in normalized_metrics.values()
        ):
            raise ValueError(
                "latency metrics must be non-empty, finite, and non-negative"
            )
        record = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "component": self.component,
            "host": self.hostname,
            "pid": self.pid,
            "trace_id": str(trace_id) if trace_id is not None else self.next_trace_id(),
            "metrics_ms": normalized_metrics,
            "metadata": dict(metadata or {}),
        }
        self._queue.put(record)

    def close(self) -> None:
        if not self.enabled or self._closed:
            return
        self._closed = True
        self._queue.put(_STOP)
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)

    def _writer_loop(self) -> None:
        values: dict[str, list[float]] = defaultdict(list)
        sample_count = 0
        with self.detail_path.open("a", encoding="utf-8", buffering=1) as detail:
            while True:
                item = self._queue.get()
                if item is _STOP:
                    break
                if not isinstance(item, dict):
                    continue
                detail.write(
                    json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                )
                sample_count += 1
                for name, value in item["metrics_ms"].items():
                    values[name].append(float(value))
                if sample_count % self.summary_every == 0:
                    self._write_summary(values, sample_count)
            detail.flush()
        self._write_summary(values, sample_count)

    def _write_summary(
        self, values: Mapping[str, list[float]], sample_count: int
    ) -> None:
        lines = [
            f"component: {self.component}",
            f"host: {self.hostname}",
            f"pid: {self.pid}",
            f"updated_at: {datetime.now().astimezone().isoformat(timespec='seconds')}",
            f"records: {sample_count}",
            "",
            "metric_ms | count | mean | std | min | p50 | p90 | p95 | max",
        ]
        for name in sorted(values):
            metric_values = values[name]
            std = statistics.pstdev(metric_values) if len(metric_values) > 1 else 0.0
            lines.append(
                f"{name} | {len(metric_values)} | "
                f"{statistics.fmean(metric_values):.3f} | {std:.3f} | "
                f"{min(metric_values):.3f} | {_percentile(metric_values, 0.50):.3f} | "
                f"{_percentile(metric_values, 0.90):.3f} | "
                f"{_percentile(metric_values, 0.95):.3f} | {max(metric_values):.3f}"
            )
        self.summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_latency_logger(component: str) -> LatencyLogger:
    """Return one logger per component and process."""

    key = str(component)
    with _LOGGERS_LOCK:
        logger = _LOGGERS.get(key)
        if logger is None or logger.pid != os.getpid():
            logger = LatencyLogger(key)
            _LOGGERS[key] = logger
        return logger
