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

"""Pure software tests for the AgileX Nero contract."""

import numpy as np
import pytest

from rlinf.envs.realworld.nero.contract import (
    ACTION_HORIZON,
    CAMERA_NAMES,
    CONTRACT,
    ControlMode,
    NeroActionChunk,
    NeroErrorCode,
    NeroObservation,
    NeroRiskState,
    NeroStepRecord,
    absolute_to_delta,
    delta_to_absolute,
    pad_physical_action,
    unpad_model_action,
)


def _state() -> np.ndarray:
    return np.asarray([0.0] * 7 + [0.05] + [0.0] * 7 + [0.05], dtype=np.float32)


def _observation(now_ns: int = 2_000_000_000) -> NeroObservation:
    return NeroObservation(
        sequence_id=3,
        monotonic_timestamp_ns=now_ns - 20_000_000,
        images={
            name: np.full((8, 10, 3), index, dtype=np.uint8)
            for index, name in enumerate(CAMERA_NAMES)
        },
        image_timestamps_ns={
            name: now_ns - (index + 1) * 10_000_000
            for index, name in enumerate(CAMERA_NAMES)
        },
        puppet_state=_state(),
        master_state=_state(),
        prompt="put the object in the tray",
        risk_state=NeroRiskState(0, 0, True, 0, 0.01),
    )


def test_observation_payload_round_trip() -> None:
    observation = _observation()
    restored = NeroObservation.from_payload(observation.to_payload())
    assert restored.sequence_id == observation.sequence_id
    assert restored.prompt == observation.prompt
    np.testing.assert_array_equal(restored.puppet_state, observation.puppet_state)
    for name in CAMERA_NAMES:
        np.testing.assert_array_equal(restored.images[name], observation.images[name])


def test_action_and_step_payload_round_trip() -> None:
    actions = np.repeat(_state()[None, :], ACTION_HORIZON, axis=0)
    chunk = NeroActionChunk(4, 2_000_000_000, actions)
    restored = NeroActionChunk.from_payload(chunk.to_payload())
    np.testing.assert_array_equal(restored.actions, chunk.actions)
    record = NeroStepRecord(
        2, 2_000_000_001, _state(), _state(), True, ControlMode.INTERVENTION
    )
    restored_record = NeroStepRecord.from_payload(record.to_payload())
    assert restored_record.sequence_id == record.sequence_id
    assert restored_record.mode is record.mode
    assert restored_record.intervened is True
    np.testing.assert_array_equal(restored_record.policy_action, record.policy_action)
    np.testing.assert_array_equal(
        restored_record.executed_action, record.executed_action
    )


def test_padding_round_trip_and_nonzero_padding_rejected() -> None:
    physical = np.arange(32, dtype=np.float32).reshape(2, 16)
    model = pad_physical_action(physical)
    assert model.shape == (2, 32)
    np.testing.assert_array_equal(unpad_model_action(model), physical)
    model[0, 20] = 1.0
    with pytest.raises(ValueError, match="padding"):
        unpad_model_action(model, require_zero_padding=True)
    np.testing.assert_array_equal(unpad_model_action(model), physical)


def test_delta_absolute_round_trip() -> None:
    reference = _state()
    absolute = reference.copy()
    absolute[[0, 8]] += np.float32(0.01)
    absolute[[7, 15]] = np.float32(0.08)
    delta = absolute_to_delta(absolute, reference)
    restored = delta_to_absolute(delta, reference)
    np.testing.assert_allclose(restored, absolute, atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    "bad_state", [np.zeros(14, dtype=np.float32), np.full(16, np.nan, dtype=np.float32)]
)
def test_bad_state_rejected(bad_state: np.ndarray) -> None:
    with pytest.raises(ValueError):
        NeroObservation(
            0,
            1,
            {name: np.zeros((2, 2, 3), dtype=np.uint8) for name in CAMERA_NAMES},
            dict.fromkeys(CAMERA_NAMES, 1),
            bad_state,
            _state(),
            "prompt",
            NeroRiskState(0, 0, True, 0, 0.0),
        )


def test_wrong_camera_order_and_image_layout_rejected() -> None:
    observation = _observation()
    reversed_images = dict(reversed(list(observation.images.items())))
    with pytest.raises(ValueError, match="camera order"):
        NeroObservation(
            0,
            1,
            reversed_images,
            observation.image_timestamps_ns,
            _state(),
            _state(),
            "prompt",
            observation.risk_state,
        )
    payload = observation.to_payload()
    payload["images"][CAMERA_NAMES[0]]["dtype"] = "float32"
    with pytest.raises(ValueError, match="dtype"):
        NeroObservation.from_payload(payload)


def test_stale_and_skewed_observation_rejected() -> None:
    observation = _observation()
    with pytest.raises(ValueError, match="observation age"):
        observation.validate_freshness(3_000_000_000)
    skewed = _observation()
    object.__setattr__(
        skewed,
        "image_timestamps_ns",
        {
            CAMERA_NAMES[0]: 1_900_000_000,
            CAMERA_NAMES[1]: 1_700_000_000,
            CAMERA_NAMES[2]: 1_900_000_000,
        },
    )
    with pytest.raises(ValueError, match="skew"):
        skewed.validate_freshness(2_000_000_000, max_camera_age_s=1.0)


def test_bad_action_shapes_and_values_rejected_but_range_is_transport_transparent() -> (
    None
):
    with pytest.raises(ValueError, match="shape"):
        NeroActionChunk(0, 1, np.zeros((50, 14), dtype=np.float32))
    invalid = np.repeat(_state()[None, :], 2, axis=0)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        NeroActionChunk(0, 1, invalid)
    invalid = np.repeat(_state()[None, :], 2, axis=0)
    invalid[0, 7] = 0.2
    chunk = NeroActionChunk(0, 1, invalid)
    assert chunk.actions[0, 7] == pytest.approx(0.2)


def test_source_timing_does_not_use_the_consumer_monotonic_clock() -> None:
    observation = _observation()
    observation.validate_source_timing(source_now_ns=2_000_000_000)
    with pytest.raises(ValueError, match="observation age"):
        observation.validate_freshness(3_000_000_000)


def test_protocol_mismatch_rejected() -> None:
    with pytest.raises(RuntimeError, match="mismatch"):
        CONTRACT.assert_compatible("nero-contract/2.0.0", publishing=True)


def test_unknown_error_code_rejected() -> None:
    with pytest.raises(ValueError):
        NeroStepRecord(0, 1, _state(), _state(), False, ControlMode.HOLD, "unknown")
    record = NeroStepRecord(
        0,
        1,
        _state(),
        _state(),
        False,
        ControlMode.HOLD,
        NeroErrorCode.HEARTBEAT_TIMEOUT,
    )
    assert (
        NeroStepRecord.from_payload(record.to_payload()).error_code
        is NeroErrorCode.HEARTBEAT_TIMEOUT
    )
