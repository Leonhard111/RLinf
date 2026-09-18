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

import json

from rlinf.utils.latency_logger import LatencyLogger


def test_latency_logger_writes_samples_and_statistics(tmp_path):
    logger = LatencyLogger("unit_test", enabled=True, log_dir=tmp_path, summary_every=2)
    logger.record(
        {"preprocess": 1.0, "total": 5.0},
        trace_id=10,
        metadata={"batch_size": 1},
    )
    logger.record(
        {"preprocess": 3.0, "total": 7.0},
        trace_id=11,
        metadata={"batch_size": 1},
    )
    logger.close()

    rows = [json.loads(line) for line in logger.detail_path.read_text().splitlines()]
    assert [row["trace_id"] for row in rows] == ["10", "11"]
    assert rows[0]["metrics_ms"]["preprocess"] == 1.0

    summary = logger.summary_path.read_text()
    assert "records: 2" in summary
    assert "preprocess | 2 | 2.000 | 1.000 | 1.000 | 2.000" in summary
    assert "total | 2 | 6.000 | 1.000 | 5.000 | 6.000" in summary


def test_latency_logger_is_a_noop_when_disabled(tmp_path):
    logger = LatencyLogger("disabled", enabled=False, log_dir=tmp_path)
    logger.record({"total": 1.0})
    logger.close()
    assert not list(tmp_path.iterdir())
