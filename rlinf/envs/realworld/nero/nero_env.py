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

"""Gym environment for an AgileX Nero controlled by a host policy bridge."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from rlinf.utils.latency_logger import get_latency_logger

from .contract import (
    ACTION_HORIZON,
    CAMERA_NAMES,
    PHYSICAL_ACTION_DIM,
    RTC_CONTEXT_VECTOR_DIM,
    ControlMode,
    NeroObservation,
    NeroRiskState,
)
from .nero_client import NeroRobotClient


@dataclass
class NeroRobotConfig:
    """Hydra/Gym configuration for :class:`NeroEnv`."""

    mode: str = "dummy"
    bridge_endpoint: str = "tcp://127.0.0.1:5555"
    bridge_connect_timeout_s: float = 2.0
    bridge_request_timeout_s: float = 10.0
    heartbeat_interval_s: float = 0.1
    heartbeat_timeout_s: float = 0.2
    task_description: str = "Nero manipulation task"
    max_num_steps: int = 300
    action_representation: str = "absolute"
    image_height: int = 480
    image_width: int = 640
    max_observation_age_s: float = 0.25
    max_camera_age_s: float = 0.25
    max_camera_skew_s: float = 0.10
    dummy_seed: int = 0

    def __post_init__(self) -> None:
        self.mode = ControlMode(self.mode).value if self.mode != "dummy" else "dummy"
        if self.action_representation not in {"absolute", "delta"}:
            raise ValueError("action_representation must be absolute or delta")
        if self.max_num_steps < 1 or min(self.image_height, self.image_width) < 1:
            raise ValueError("max_num_steps and image dimensions must be positive")
        if min(self.heartbeat_interval_s, self.heartbeat_timeout_s) <= 0:
            raise ValueError(
                "heartbeat_interval_s and heartbeat_timeout_s must be positive"
            )


class NeroEnv(gym.Env):
    """One physical Nero environment; hardware ownership remains on AgileX."""

    metadata = {"render_modes": []}
    CONFIG_CLS = NeroRobotConfig

    def __init__(
        self,
        override_cfg: dict[str, Any] | None = None,
        worker_info: object | None = None,
        hardware_info: object | None = None,
        env_idx: int = 0,
        env_cfg: object | None = None,
        *,
        client_factory: Callable[..., NeroRobotClient] = NeroRobotClient,
    ) -> None:
        del worker_info, hardware_info, env_cfg
        if env_idx != 0:
            raise ValueError("NeroEnv supports exactly one hardware environment")
        self.config = self.CONFIG_CLS(**(override_cfg or {}))
        self._task_description = self.config.task_description
        self._rng = np.random.default_rng(self.config.dummy_seed)
        self._num_steps = 0
        self._dummy_sequence = 0
        self._closed = False
        self._client: NeroRobotClient | None = None
        self._last_observation: NeroObservation | None = None
        self._pending_actions: list[np.ndarray] = []
        self._submitted_chunk: tuple[np.ndarray, NeroObservation] | None = None
        self._latency_logger = get_latency_logger("nero_env")
        if self.config.mode != "dummy":
            self._client = client_factory(
                self.config.bridge_endpoint,
                connect_timeout_s=self.config.bridge_connect_timeout_s,
                request_timeout_s=self.config.bridge_request_timeout_s,
                heartbeat_interval_s=self.config.heartbeat_interval_s,
                heartbeat_timeout_s=self.config.heartbeat_timeout_s,
            )
            expected_mode = ControlMode(self.config.mode)
            if self._client.capabilities.mode is not expected_mode:
                self._client.close()
                raise RuntimeError(
                    "Nero env/bridge mode mismatch: "
                    f"env={expected_mode.value}, bridge={self._client.capabilities.mode.value}"
                )

        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(PHYSICAL_ACTION_DIM,),
            dtype=np.float32,
        )
        image_space = gym.spaces.Box(
            0,
            255,
            shape=(self.config.image_height, self.config.image_width, 3),
            dtype=np.uint8,
        )
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Dict({
                "master": gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(PHYSICAL_ACTION_DIM,),
                    dtype=np.float32,
                ),
                "puppet": gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(PHYSICAL_ACTION_DIM,),
                    dtype=np.float32,
                ),
                "risk": gym.spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32),
            }),
            "frames": gym.spaces.Dict(dict.fromkeys(CAMERA_NAMES, image_space)),
            "rtc": gym.spaces.Box(
                -np.inf,
                np.inf,
                shape=(RTC_CONTEXT_VECTOR_DIM,),
                dtype=np.float64,
            ),
        })

    @property
    def task_description(self) -> str:
        return self._task_description

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del options
        super().reset(seed=seed)
        if self._closed:
            raise RuntimeError("NeroEnv is closed")
        self._num_steps = 0
        self._pending_actions.clear()
        self._submitted_chunk = None
        observation = (
            self._dummy_observation() if self._client is None else self._client.reset()
        )
        self._last_observation = observation
        return self._as_gym_observation(observation), self._info(observation)

    def on_action_chunk_begin(self) -> None:
        """Start collecting one RLinf chunk for the native policy adapter.

        RLinf invokes Gym ``step`` once per physical action, whereas the existing
        native policy interface consumes one exact 50x16 generation. Buffering
        only at this boundary preserves both APIs without changing robot control.
        """

        if self._pending_actions:
            raise RuntimeError(
                "new action chunk started before the previous Nero chunk completed"
            )
        if self._submitted_chunk is not None:
            raise RuntimeError(
                "new action chunk started before the submitted Nero chunk completed"
            )

    def submit_action_chunk(self, actions: np.ndarray) -> None:
        """Submit one native 50x16 chunk before RLinf expands its step results.

        The native bridge consumes the entire policy generation at once.  RLinf's
        generic real-world adapter historically called ``step`` fifty times and
        submitted the chunk only on the final call, needlessly putting 49 image
        copies and tensor conversions on the observation-to-action critical path.
        This method moves only the bridge submission forward; the following
        per-action ``step`` calls still reproduce the same observations, metrics,
        and episode accounting.
        """

        started = time.perf_counter()
        if self._closed:
            raise RuntimeError("NeroEnv is closed")
        if self._last_observation is None:
            raise RuntimeError("NeroEnv.reset() must be called before step()")
        if self._pending_actions or self._submitted_chunk is not None:
            raise RuntimeError("a Nero action chunk is already in progress")

        action_chunk = np.asarray(actions, dtype=np.float32)
        expected_shape = (ACTION_HORIZON, PHYSICAL_ACTION_DIM)
        if action_chunk.shape != expected_shape:
            raise ValueError(
                f"Nero action chunk must have shape {expected_shape}, "
                f"got {action_chunk.shape}"
            )
        if not np.isfinite(action_chunk).all():
            raise ValueError("Nero action chunk contains NaN/Inf")
        action_chunk = np.ascontiguousarray(action_chunk)
        validated = time.perf_counter()

        if self._client is None:
            observation = self._dummy_observation()
        else:
            observation = self._client.step(
                action_chunk,
                representation=self.config.action_representation,
            )
        client_finished = time.perf_counter()
        if self._client is not None:
            observation.validate_source_timing(
                max_camera_age_s=self.config.max_camera_age_s,
                max_camera_skew_s=self.config.max_camera_skew_s,
            )
        timing_validated = time.perf_counter()
        self._submitted_chunk = (action_chunk.copy(), observation)
        finished = time.perf_counter()
        self._latency_logger.record(
            {
                "validate_action": (validated - started) * 1000.0,
                "client_step": (client_finished - validated) * 1000.0,
                "validate_observation_timing": (timing_validated - client_finished)
                * 1000.0,
                "store_submitted_chunk": (finished - timing_validated) * 1000.0,
                "submit_chunk_total": (finished - started) * 1000.0,
            },
            trace_id=observation.sequence_id,
            metadata={
                "mode": self.config.mode,
                "action_shape": list(action_chunk.shape),
            },
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("NeroEnv is closed")
        actions = np.asarray(action, dtype=np.float32)
        if actions.shape != (PHYSICAL_ACTION_DIM,):
            raise ValueError(f"NeroEnv action must have shape {(PHYSICAL_ACTION_DIM,)}")
        if not np.isfinite(actions).all():
            raise ValueError("NeroEnv action contains NaN/Inf")
        if len(self._pending_actions) >= ACTION_HORIZON:
            raise RuntimeError("Nero action chunk overflow")
        if self._submitted_chunk is not None:
            submitted_actions, _ = self._submitted_chunk
            action_index = len(self._pending_actions)
            if not np.array_equal(actions, submitted_actions[action_index]):
                raise ValueError(
                    "Nero action changed after native chunk submission at "
                    f"index {action_index}"
                )
        self._pending_actions.append(actions.copy())

        if len(self._pending_actions) == ACTION_HORIZON:
            action_chunk = np.stack(self._pending_actions, axis=0)
            self._pending_actions.clear()
            if self._submitted_chunk is not None:
                submitted_actions, observation = self._submitted_chunk
                self._submitted_chunk = None
                if not np.array_equal(action_chunk, submitted_actions):
                    raise ValueError("Nero buffered chunk differs from submitted chunk")
            elif self._client is None:
                observation = self._dummy_observation()
            else:
                observation = self._client.step(
                    action_chunk,
                    representation=self.config.action_representation,
                )
                observation.validate_source_timing(
                    max_camera_age_s=self.config.max_camera_age_s,
                    max_camera_skew_s=self.config.max_camera_skew_s,
                )
            self._last_observation = observation
        else:
            observation = self._last_observation
            if observation is None:
                raise RuntimeError("NeroEnv.reset() must be called before step()")
        self._num_steps += 1
        truncated = self._num_steps >= self.config.max_num_steps
        return (
            self._as_gym_observation(observation),
            0.0,
            False,
            truncated,
            self._info(observation),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._client is not None:
            self._client.close()

    def _dummy_observation(self) -> NeroObservation:
        now_ns = time.monotonic_ns()
        sequence_id = self._dummy_sequence
        self._dummy_sequence += 1
        state = np.asarray([0.0] * 7 + [0.05] + [0.0] * 7 + [0.05], dtype=np.float32)
        images = {
            name: self._rng.integers(
                0,
                256,
                size=(self.config.image_height, self.config.image_width, 3),
                dtype=np.uint8,
            )
            for name in CAMERA_NAMES
        }
        return NeroObservation(
            sequence_id=sequence_id,
            monotonic_timestamp_ns=now_ns,
            images=images,
            image_timestamps_ns=dict.fromkeys(CAMERA_NAMES, now_ns),
            puppet_state=state,
            master_state=state,
            prompt=self.config.task_description,
            risk_state=NeroRiskState(0, 0, True, 0, 0.0),
        )

    @staticmethod
    def _as_gym_observation(observation: NeroObservation) -> dict[str, Any]:
        risk = observation.risk_state
        rtc_vector = (
            np.asarray([0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            if observation.rtc_context is None
            else observation.rtc_context.to_vector()
        )
        return {
            "state": {
                # NeroObservation owns immutable snapshot arrays. Returning the
                # views avoids copying three images for every one of the 50 Gym
                # expansion steps in a native action chunk.
                "master": observation.master_state,
                "puppet": observation.puppet_state,
                "risk": np.asarray(
                    [
                        risk.risk_level,
                        risk.raw_risk_level,
                        float(risk.measurement_valid),
                        risk.fault_code,
                        risk.data_age_s,
                    ],
                    dtype=np.float32,
                ),
            },
            "frames": {name: observation.images[name] for name in CAMERA_NAMES},
            "rtc": rtc_vector,
        }

    def _info(self, observation: NeroObservation) -> dict[str, Any]:
        return {
            "sequence_id": observation.sequence_id,
            "control_mode": self.config.mode,
            "intervene_action": None,
        }
