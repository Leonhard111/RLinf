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

"""Pure-software tests for the phase-3 Nero bridge client and Gym env."""

import threading
import time

import numpy as np

from rlinf.envs.realworld.nero.bridge_server import NeroBridgeServer, PolicyExchange
from rlinf.envs.realworld.nero.contract import (
    ACTION_HORIZON,
    CAMERA_NAMES,
    ControlMode,
    NeroObservation,
    NeroRiskState,
)
from rlinf.envs.realworld.nero.nero_client import NeroRobotClient
from rlinf.envs.realworld.nero.nero_env import NeroEnv


def _state() -> np.ndarray:
    return np.asarray([0.0] * 7 + [0.05] + [0.0] * 7 + [0.05], dtype=np.float32)


def _observation(sequence_id: int) -> NeroObservation:
    now_ns = time.monotonic_ns()
    return NeroObservation(
        sequence_id=sequence_id,
        monotonic_timestamp_ns=now_ns,
        images={name: np.zeros((2, 3, 3), dtype=np.uint8) for name in CAMERA_NAMES},
        image_timestamps_ns=dict.fromkeys(CAMERA_NAMES, now_ns),
        puppet_state=_state(),
        master_state=_state(),
        prompt="test Nero",
        risk_state=NeroRiskState(0, 0, True, 0, 0.0),
    )


def test_tcp_client_round_trip_preserves_action_chunk() -> None:
    exchange = PolicyExchange()
    server = NeroBridgeServer(
        "tcp://127.0.0.1:0",
        exchange,
        mode=ControlMode.SHADOW,
        control_config_sha256="a" * 64,
        wait_timeout_s=0.05,
    )
    server.start()
    expected = np.repeat(_state()[None, :], ACTION_HORIZON, axis=0)
    received: list[np.ndarray] = []

    def produce() -> None:
        first = exchange.offer_observation(_observation(0), timeout_s=2.0)
        received.append(first.actions)
        try:
            exchange.offer_observation(_observation(1), timeout_s=2.0)
        except Exception:
            pass

    producer = threading.Thread(target=produce)
    producer.start()
    client = NeroRobotClient(server.endpoint, request_timeout_s=2.0)
    try:
        assert client.capabilities.control_owner == "vlastop"
        assert server.client_activity_age_s() is not None
        assert client.reset().sequence_id == 0
        assert client.step(expected).sequence_id == 1
        np.testing.assert_array_equal(received[0], expected)
    finally:
        client.close()
        exchange.close()
        server.close()
        producer.join(timeout=2.0)
    assert not producer.is_alive()


def test_tcp_client_accepts_native_pi05_control_owner() -> None:
    exchange = PolicyExchange()
    server = NeroBridgeServer(
        "tcp://127.0.0.1:0",
        exchange,
        mode=ControlMode.SHADOW,
        control_config_sha256="c" * 64,
        control_owner="pi05-native",
        wait_timeout_s=0.05,
    )
    server.start()
    client = NeroRobotClient(server.endpoint, request_timeout_s=0.5)
    try:
        assert client.capabilities.control_owner == "pi05-native"
    finally:
        client.close()
        exchange.close()
        server.close()


def test_dummy_env_action_space_does_not_preempt_native_topp_projection() -> None:
    env = NeroEnv({"mode": "dummy", "image_height": 2, "image_width": 3})
    try:
        out_of_range = _state()
        out_of_range[[0, 7]] = (9.0, 0.2)
        assert env.action_space.contains(out_of_range)
    finally:
        env.close()


def test_independent_heartbeat_survives_inference_pause_and_stops_on_close() -> None:
    exchange = PolicyExchange()
    server = NeroBridgeServer(
        "tcp://127.0.0.1:0",
        exchange,
        mode=ControlMode.SHADOW,
        control_config_sha256="b" * 64,
        wait_timeout_s=0.05,
    )
    server.start()
    client = NeroRobotClient(
        server.endpoint,
        request_timeout_s=0.5,
        connect_timeout_s=0.2,
        heartbeat_interval_s=0.05,
        heartbeat_timeout_s=0.1,
    )
    try:
        # This is longer than the host bridge's 0.75 s guard and represents a
        # slow model inference. Independent status requests must keep it fresh.
        time.sleep(2.0)
        age = server.client_activity_age_s()
        assert client.heartbeat_alive
        assert age is not None and age < 0.30
        assert client.last_heartbeat_error is None
    finally:
        client.close()
    time.sleep(0.80)
    age = server.client_activity_age_s()
    assert age is not None and age >= 0.75
    exchange.close()
    server.close()


def test_dummy_env_adapts_rlinf_steps_to_vlastop_action_chunks() -> None:
    env = NeroEnv({
        "mode": "dummy",
        "image_height": 2,
        "image_width": 3,
        "max_num_steps": 20_000,
    })
    action = _state()
    try:
        observation, info = env.reset()
        assert env.observation_space.contains(observation)
        assert info["sequence_id"] == 0
        env.on_action_chunk_begin()
        for step_index in range(ACTION_HORIZON):
            observation, reward, terminated, truncated, info = env.step(action)
            assert reward == 0.0
            assert not terminated
            assert not truncated
            expected_sequence = int(step_index == ACTION_HORIZON - 1)
            assert info["sequence_id"] == expected_sequence
        assert env.observation_space.contains(observation)
        assert info["sequence_id"] == 1
    finally:
        env.close()


def test_dummy_env_can_submit_native_chunk_before_expanding_steps() -> None:
    env = NeroEnv({
        "mode": "dummy",
        "image_height": 2,
        "image_width": 3,
        "max_num_steps": 20_000,
    })
    action_chunk = np.repeat(_state()[None, :], ACTION_HORIZON, axis=0)
    try:
        _, info = env.reset()
        assert info["sequence_id"] == 0
        env.on_action_chunk_begin()
        env.submit_action_chunk(action_chunk)

        for step_index, action in enumerate(action_chunk):
            _, reward, terminated, truncated, info = env.step(action)
            assert reward == 0.0
            assert not terminated
            assert not truncated
            expected_sequence = int(step_index == ACTION_HORIZON - 1)
            assert info["sequence_id"] == expected_sequence

        assert info["sequence_id"] == 1
    finally:
        env.close()


def test_native_chunk_submission_rejects_changed_followup_action() -> None:
    env = NeroEnv({"mode": "dummy", "image_height": 2, "image_width": 3})
    action_chunk = np.repeat(_state()[None, :], ACTION_HORIZON, axis=0)
    try:
        env.reset()
        env.on_action_chunk_begin()
        env.submit_action_chunk(action_chunk)
        changed = action_chunk[0].copy()
        changed[0] += 0.01
        with np.testing.assert_raises(ValueError):
            env.step(changed)
    finally:
        env.close()


def test_dummy_env_rejects_chunk_action_and_nan() -> None:
    env = NeroEnv({"mode": "dummy", "image_height": 2, "image_width": 3})
    try:
        env.reset()
        with np.testing.assert_raises(ValueError):
            env.step(np.repeat(_state()[None, :], ACTION_HORIZON, axis=0))
        invalid = _state()
        invalid[0] = np.nan
        with np.testing.assert_raises(ValueError):
            env.step(invalid)
    finally:
        env.close()
