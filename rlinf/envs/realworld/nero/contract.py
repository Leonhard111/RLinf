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

"""Versioned, hardware-independent data contract for dual-arm AgileX Nero.

Images use uint8 HWC BGR layout, matching the native ``run_robot.sh`` Orbbec
path. Joint values are radians and gripper values are metres. Monotonic
timestamps are integer nanoseconds from the producer host and must never be
compared directly with another host's monotonic clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

PROTOCOL_VERSION = "nero-contract/1.0.0"
PHYSICAL_ACTION_DIM = 16
MODEL_ACTION_DIM = 32
ACTION_HORIZON = 50
ARM_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "gripper",
)
PHYSICAL_ACTION_NAMES = tuple(
    f"{side}_{name}" for side in ("left", "right") for name in ARM_JOINT_NAMES
)
MODEL_ACTION_NAMES = PHYSICAL_ACTION_NAMES + tuple(
    f"padding_{index:02d}" for index in range(MODEL_ACTION_DIM - PHYSICAL_ACTION_DIM)
)
DELTA_MASK_PHYSICAL = np.asarray(
    (True,) * 7 + (False,) + (True,) * 7 + (False,), dtype=np.bool_
)
DELTA_MASK_MODEL = np.concatenate((
    DELTA_MASK_PHYSICAL,
    np.zeros(MODEL_ACTION_DIM - PHYSICAL_ACTION_DIM, dtype=np.bool_),
))
CAMERA_NAMES = ("cam_high", "cam_right_wrist", "cam_left_wrist")
RTC_PREFIX_SCHEDULES = ("exp", "linear", "ones", "zeros")
RTC_CONTEXT_VECTOR_DIM = 9
CAMERA_SERIALS = {
    "cam_high": "CC1N16201TR",
    "cam_right_wrist": "CC1N16201YL",
    "cam_left_wrist": "CC1N16200RD",
}
ROS_TOPICS = {
    "left_puppet_state": "/nero_left/puppet/joint_states",
    "right_puppet_state": "/nero_right/puppet/joint_states",
    "left_master_state": "/nero_left/master/joint_states",
    "right_master_state": "/nero_right/master/joint_states",
    "left_control": "/nero_left/control/joint_states",
    "right_control": "/nero_right/control/joint_states",
    "risk_state": "/looklook/risk_state",
}
JOINT_LOWER_RAD = np.asarray(
    (-2.705261, -1.745330, -2.757621, -1.012291, -2.757621, -0.733039, -1.570797),
    dtype=np.float32,
)
JOINT_UPPER_RAD = np.asarray(
    (2.705261, 1.745330, 2.757621, 2.146755, 2.757621, 0.959932, 1.570797),
    dtype=np.float32,
)
GRIPPER_RANGE_M = (0.0, 0.1)


class ControlMode(str, Enum):
    """Bridge execution modes.  Only PUBLISH may create control publishers."""

    SHADOW = "shadow"
    HOLD = "hold"
    PUBLISH = "publish"
    INTERVENTION = "intervention"
    STOP = "stop"


class NeroErrorCode(str, Enum):
    """Stable bridge/environment error codes carried across the protocol."""

    PROTOCOL_MISMATCH = "protocol_mismatch"
    STALE_OBSERVATION = "stale_observation"
    INVALID_ACTION = "invalid_action"
    PUBLISHER_CONFLICT = "publisher_conflict"
    RISK_UNSAFE = "risk_unsafe"
    HEARTBEAT_TIMEOUT = "heartbeat_timeout"
    BRIDGE_FAULT = "bridge_fault"


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(result)


def _validate_sequence_id(value: int) -> int:
    result = int(value)
    if result < 0:
        raise ValueError("sequence_id must be non-negative")
    return result


def _validate_timestamp(value: int, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive monotonic timestamp")
    return result


def validate_absolute_physical_action(action: Any) -> np.ndarray:
    """Validate a 16D absolute action against Nero's configured hard limits."""

    result = _array(action, (PHYSICAL_ACTION_DIM,), "absolute action")
    for offset in (0, 8):
        joints = result[offset : offset + 7]
        if np.any(joints < JOINT_LOWER_RAD) or np.any(joints > JOINT_UPPER_RAD):
            raise ValueError("absolute action contains an out-of-range joint")
        gripper = float(result[offset + 7])
        if not GRIPPER_RANGE_M[0] <= gripper <= GRIPPER_RANGE_M[1]:
            raise ValueError("absolute action contains an out-of-range gripper")
    return result


def pad_physical_action(action: Any) -> np.ndarray:
    """Pad one or more 16D physical actions to the 32D π0.5 model width."""

    result = np.asarray(action, dtype=np.float32)
    if result.ndim < 1 or result.shape[-1] != PHYSICAL_ACTION_DIM:
        raise ValueError(
            f"physical action last dimension must be {PHYSICAL_ACTION_DIM}"
        )
    if not np.isfinite(result).all():
        raise ValueError("physical action must contain only finite values")
    padding = np.zeros(
        result.shape[:-1] + (MODEL_ACTION_DIM - PHYSICAL_ACTION_DIM,), dtype=np.float32
    )
    return np.ascontiguousarray(np.concatenate((result, padding), axis=-1))


def unpad_model_action(
    action: Any, *, require_zero_padding: bool = False
) -> np.ndarray:
    """Extract the 16D physical prefix from one or more 32D model actions.

    Model inference may produce arbitrary values in unused dimensions, so zero
    padding is optional.  Enable it when validating padded dataset targets.
    """

    result = np.asarray(action, dtype=np.float32)
    if result.ndim < 1 or result.shape[-1] != MODEL_ACTION_DIM:
        raise ValueError(f"model action last dimension must be {MODEL_ACTION_DIM}")
    if not np.isfinite(result).all():
        raise ValueError("model action must contain only finite values")
    if require_zero_padding and not np.allclose(
        result[..., PHYSICAL_ACTION_DIM:], 0.0, atol=1e-7
    ):
        raise ValueError("model action padding must be zero")
    return np.ascontiguousarray(result[..., :PHYSICAL_ACTION_DIM])


def delta_to_absolute(delta: Any, reference: Any) -> np.ndarray:
    """Apply joint deltas while preserving absolute gripper commands."""

    delta_array = _array(delta, (PHYSICAL_ACTION_DIM,), "delta")
    reference_array = _array(reference, (PHYSICAL_ACTION_DIM,), "reference")
    absolute = delta_array.copy()
    absolute[DELTA_MASK_PHYSICAL] += reference_array[DELTA_MASK_PHYSICAL]
    return absolute


def absolute_to_delta(absolute: Any, reference: Any) -> np.ndarray:
    """Convert joint targets to deltas while preserving absolute grippers."""

    absolute_array = _array(absolute, (PHYSICAL_ACTION_DIM,), "absolute")
    reference_array = _array(reference, (PHYSICAL_ACTION_DIM,), "reference")
    delta = absolute_array.copy()
    delta[DELTA_MASK_PHYSICAL] -= reference_array[DELTA_MASK_PHYSICAL]
    return delta


def _image_to_payload(image: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": "uint8",
        "shape": list(image.shape),
        "data": image.tobytes(order="C"),
    }


def _image_from_payload(payload: Mapping[str, Any], name: str) -> np.ndarray:
    if payload.get("dtype") != "uint8":
        raise ValueError(f"{name} image dtype must be uint8")
    shape = tuple(int(value) for value in payload["shape"])
    if len(shape) != 3 or shape[2] != 3 or min(shape) <= 0:
        raise ValueError(f"{name} image shape must be positive HWC with 3 channels")
    data = payload["data"]
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError(f"{name} image data must be bytes")
    expected = math.prod(shape)
    if len(data) != expected:
        raise ValueError(
            f"{name} image byte length is {len(data)}, expected {expected}"
        )
    return np.frombuffer(data, dtype=np.uint8).reshape(shape).copy()


@dataclass(frozen=True)
class NeroRiskState:
    """Snapshot of the LookLook risk gate."""

    risk_level: int
    raw_risk_level: int
    measurement_valid: bool
    fault_code: int
    data_age_s: float

    def __post_init__(self) -> None:
        if self.risk_level < 0 or self.raw_risk_level < 0 or self.fault_code < 0:
            raise ValueError("risk levels and fault_code must be non-negative")
        if not math.isfinite(self.data_age_s) or self.data_age_s < 0:
            raise ValueError("risk data_age_s must be finite and non-negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "raw_risk_level": self.raw_risk_level,
            "measurement_valid": self.measurement_valid,
            "fault_code": self.fault_code,
            "data_age_s": self.data_age_s,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NeroRiskState":
        return cls(
            risk_level=int(payload["risk_level"]),
            raw_risk_level=int(payload["raw_risk_level"]),
            measurement_valid=bool(payload["measurement_valid"]),
            fault_code=int(payload["fault_code"]),
            data_age_s=float(payload["data_age_s"]),
        )


@dataclass(frozen=True)
class NeroRTCContext:
    """Integer RTC timing generated entirely in the AgileX clock domain."""

    enabled: bool
    response_generation_id: int
    previous_generation_id: int
    request_policy_step: int
    execution_horizon_steps: int
    predicted_delay_steps: int
    policy_hz: float
    prefix_attention_schedule: str = "exp"
    max_guidance_weight: float = 5.0

    def __post_init__(self) -> None:
        if self.response_generation_id < 0:
            raise ValueError("RTC response_generation_id must be non-negative")
        if self.previous_generation_id < -1:
            raise ValueError("RTC previous_generation_id must be >= -1")
        if self.request_policy_step < 0:
            raise ValueError("RTC request_policy_step must be non-negative")
        if not 0 <= self.execution_horizon_steps < ACTION_HORIZON:
            raise ValueError("RTC execution_horizon_steps must be in [0, 49]")
        if not 0 <= self.predicted_delay_steps < ACTION_HORIZON:
            raise ValueError("RTC predicted_delay_steps must be in [0, 49]")
        if not math.isfinite(self.policy_hz) or self.policy_hz <= 0:
            raise ValueError("RTC policy_hz must be finite and positive")
        if self.prefix_attention_schedule not in RTC_PREFIX_SCHEDULES:
            raise ValueError("unsupported RTC prefix attention schedule")
        if not math.isfinite(self.max_guidance_weight) or self.max_guidance_weight <= 0:
            raise ValueError("RTC max_guidance_weight must be finite and positive")
        if self.enabled:
            if self.previous_generation_id < 0:
                raise ValueError("enabled RTC context requires a previous generation")
            if self.execution_horizon_steps < 1:
                raise ValueError(
                    "enabled RTC context requires a positive execution horizon"
                )
            if (
                self.predicted_delay_steps
                > ACTION_HORIZON - self.execution_horizon_steps
            ):
                raise ValueError("RTC delay/execution overlap is infeasible")

    def to_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "response_generation_id": self.response_generation_id,
            "previous_generation_id": self.previous_generation_id,
            "request_policy_step": self.request_policy_step,
            "execution_horizon_steps": self.execution_horizon_steps,
            "predicted_delay_steps": self.predicted_delay_steps,
            "policy_hz": self.policy_hz,
            "prefix_attention_schedule": self.prefix_attention_schedule,
            "max_guidance_weight": self.max_guidance_weight,
        }

    def to_vector(self) -> np.ndarray:
        """Stable numeric form that survives Gym vectorization and P2P tensors."""

        return np.asarray(
            [
                float(self.enabled),
                self.response_generation_id,
                self.previous_generation_id,
                self.request_policy_step,
                self.execution_horizon_steps,
                self.predicted_delay_steps,
                self.policy_hz,
                RTC_PREFIX_SCHEDULES.index(self.prefix_attention_schedule),
                self.max_guidance_weight,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NeroRTCContext":
        return cls(
            enabled=bool(payload["enabled"]),
            response_generation_id=int(payload["response_generation_id"]),
            previous_generation_id=int(payload["previous_generation_id"]),
            request_policy_step=int(payload["request_policy_step"]),
            execution_horizon_steps=int(payload["execution_horizon_steps"]),
            predicted_delay_steps=int(payload["predicted_delay_steps"]),
            policy_hz=float(payload["policy_hz"]),
            prefix_attention_schedule=str(payload["prefix_attention_schedule"]),
            max_guidance_weight=float(payload["max_guidance_weight"]),
        )


@dataclass(frozen=True)
class NeroObservation:
    """One synchronized three-camera, puppet, master, and risk observation."""

    sequence_id: int
    monotonic_timestamp_ns: int
    images: Mapping[str, np.ndarray]
    image_timestamps_ns: Mapping[str, int]
    puppet_state: np.ndarray
    master_state: np.ndarray
    prompt: str
    risk_state: NeroRiskState
    rtc_context: NeroRTCContext | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence_id", _validate_sequence_id(self.sequence_id))
        object.__setattr__(
            self,
            "monotonic_timestamp_ns",
            _validate_timestamp(self.monotonic_timestamp_ns, "observation timestamp"),
        )
        if (
            tuple(self.images) != CAMERA_NAMES
            or tuple(self.image_timestamps_ns) != CAMERA_NAMES
        ):
            raise ValueError(
                f"images and timestamps must use exact camera order {CAMERA_NAMES}"
            )
        normalized_images: dict[str, np.ndarray] = {}
        normalized_timestamps: dict[str, int] = {}
        for name in CAMERA_NAMES:
            image = np.asarray(self.images[name])
            if (
                image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[2] != 3
                or min(image.shape) <= 0
            ):
                raise ValueError(f"{name} must be a non-empty uint8 HWC BGR image")
            normalized_images[name] = np.ascontiguousarray(image)
            normalized_timestamps[name] = _validate_timestamp(
                self.image_timestamps_ns[name], f"{name} timestamp"
            )
        object.__setattr__(self, "images", normalized_images)
        object.__setattr__(self, "image_timestamps_ns", normalized_timestamps)
        object.__setattr__(
            self,
            "puppet_state",
            _array(self.puppet_state, (PHYSICAL_ACTION_DIM,), "puppet_state"),
        )
        object.__setattr__(
            self,
            "master_state",
            _array(self.master_state, (PHYSICAL_ACTION_DIM,), "master_state"),
        )
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(self.risk_state, NeroRiskState):
            raise ValueError("risk_state must be NeroRiskState")
        if self.rtc_context is not None and not isinstance(
            self.rtc_context, NeroRTCContext
        ):
            raise ValueError("rtc_context must be NeroRTCContext or None")

    def validate_freshness(
        self,
        now_ns: int,
        *,
        max_observation_age_s: float = 0.25,
        max_camera_age_s: float = 0.25,
        max_camera_skew_s: float = 0.10,
    ) -> None:
        """Reject stale, future-dated, or excessively skewed observations."""

        now = _validate_timestamp(now_ns, "now")
        observation_age = (now - self.monotonic_timestamp_ns) / 1e9
        if observation_age < 0 or observation_age > max_observation_age_s:
            raise ValueError(f"observation age out of range: {observation_age:.6f}s")
        self.validate_source_timing(
            max_camera_age_s=max_camera_age_s,
            max_camera_skew_s=max_camera_skew_s,
            source_now_ns=now,
        )

    def validate_source_timing(
        self,
        *,
        max_camera_age_s: float = 0.25,
        max_camera_skew_s: float = 0.10,
        source_now_ns: int | None = None,
    ) -> None:
        """Validate camera timing entirely in the producer clock domain.

        ``source_now_ns`` defaults to this observation's producer timestamp.
        This form is safe on the RLinf inference host because it never compares
        the AgileX monotonic clock with the 4090 machine's monotonic clock.
        Network staleness remains bounded by the bridge request timeout.
        """

        now = (
            self.monotonic_timestamp_ns
            if source_now_ns is None
            else _validate_timestamp(source_now_ns, "source now")
        )
        timestamps = tuple(self.image_timestamps_ns[name] for name in CAMERA_NAMES)
        ages = tuple((now - timestamp) / 1e9 for timestamp in timestamps)
        if any(age < 0 or age > max_camera_age_s for age in ages):
            raise ValueError(f"camera ages out of range: {ages}")
        skew = (max(timestamps) - min(timestamps)) / 1e9
        if skew > max_camera_skew_s:
            raise ValueError(f"camera skew out of range: {skew:.6f}s")

    def to_payload(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "sequence_id": self.sequence_id,
            "monotonic_timestamp_ns": self.monotonic_timestamp_ns,
            "images": {
                name: _image_to_payload(self.images[name]) for name in CAMERA_NAMES
            },
            "image_timestamps_ns": dict(self.image_timestamps_ns),
            "puppet_state": self.puppet_state.tolist(),
            "master_state": self.master_state.tolist(),
            "prompt": self.prompt,
            "risk_state": self.risk_state.to_payload(),
            "rtc_context": (
                None if self.rtc_context is None else self.rtc_context.to_payload()
            ),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NeroObservation":
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Nero protocol version mismatch")
        image_payloads = payload["images"]
        if tuple(image_payloads) != CAMERA_NAMES:
            raise ValueError(
                f"image payload must use exact camera order {CAMERA_NAMES}"
            )
        return cls(
            sequence_id=int(payload["sequence_id"]),
            monotonic_timestamp_ns=int(payload["monotonic_timestamp_ns"]),
            images={
                name: _image_from_payload(image_payloads[name], name)
                for name in CAMERA_NAMES
            },
            image_timestamps_ns={
                name: int(payload["image_timestamps_ns"][name]) for name in CAMERA_NAMES
            },
            puppet_state=np.asarray(payload["puppet_state"], dtype=np.float32),
            master_state=np.asarray(payload["master_state"], dtype=np.float32),
            prompt=str(payload["prompt"]),
            risk_state=NeroRiskState.from_payload(payload["risk_state"]),
            rtc_context=(
                None
                if payload.get("rtc_context") is None
                else NeroRTCContext.from_payload(payload["rtc_context"])
            ),
        )


@dataclass(frozen=True)
class NeroActionChunk:
    """A policy generation containing at most 50 physical Nero actions.

    The transport checks shape, horizon and finiteness only. Absolute Nero
    limits are enforced by the native ``RobotwinToppPlanner``, exactly as in
    ``run_robot.sh``; rejecting targets here would change native semantics.
    """

    generation_id: int
    monotonic_timestamp_ns: int
    actions: np.ndarray
    representation: str = "absolute"

    def __post_init__(self) -> None:
        if int(self.generation_id) < 0:
            raise ValueError("generation_id must be non-negative")
        object.__setattr__(self, "generation_id", int(self.generation_id))
        object.__setattr__(
            self,
            "monotonic_timestamp_ns",
            _validate_timestamp(self.monotonic_timestamp_ns, "action timestamp"),
        )
        actions = np.asarray(self.actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != PHYSICAL_ACTION_DIM:
            raise ValueError(f"actions must have shape [H, {PHYSICAL_ACTION_DIM}]")
        if actions.shape[0] < 1 or actions.shape[0] > ACTION_HORIZON:
            raise ValueError(f"action horizon must be in [1, {ACTION_HORIZON}]")
        if not np.isfinite(actions).all():
            raise ValueError("actions must contain only finite values")
        if self.representation not in {"absolute", "delta"}:
            raise ValueError("representation must be 'absolute' or 'delta'")
        object.__setattr__(self, "actions", np.ascontiguousarray(actions))

    def to_payload(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "generation_id": self.generation_id,
            "monotonic_timestamp_ns": self.monotonic_timestamp_ns,
            "representation": self.representation,
            "shape": list(self.actions.shape),
            "dtype": "float32",
            "data": self.actions.tobytes(order="C"),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NeroActionChunk":
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Nero protocol version mismatch")
        if payload.get("dtype") != "float32":
            raise ValueError("action dtype must be float32")
        shape = tuple(int(value) for value in payload["shape"])
        if len(shape) != 2 or shape[1] != PHYSICAL_ACTION_DIM:
            raise ValueError("invalid action payload shape")
        data = payload["data"]
        if (
            not isinstance(data, (bytes, bytearray, memoryview))
            or len(data) != math.prod(shape) * 4
        ):
            raise ValueError("invalid action payload byte length")
        actions = np.frombuffer(data, dtype=np.float32).reshape(shape).copy()
        return cls(
            generation_id=int(payload["generation_id"]),
            monotonic_timestamp_ns=int(payload["monotonic_timestamp_ns"]),
            actions=actions,
            representation=str(payload["representation"]),
        )


@dataclass(frozen=True)
class NeroStepRecord:
    """Auditable result of arbitration between policy and human control."""

    sequence_id: int
    monotonic_timestamp_ns: int
    policy_action: np.ndarray
    executed_action: np.ndarray
    intervened: bool
    mode: ControlMode
    error_code: NeroErrorCode | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence_id", _validate_sequence_id(self.sequence_id))
        object.__setattr__(
            self,
            "monotonic_timestamp_ns",
            _validate_timestamp(self.monotonic_timestamp_ns, "step timestamp"),
        )
        object.__setattr__(
            self,
            "policy_action",
            _array(self.policy_action, (PHYSICAL_ACTION_DIM,), "policy_action"),
        )
        object.__setattr__(
            self,
            "executed_action",
            _array(self.executed_action, (PHYSICAL_ACTION_DIM,), "executed_action"),
        )
        if not isinstance(self.mode, ControlMode):
            object.__setattr__(self, "mode", ControlMode(self.mode))
        if self.mode is ControlMode.INTERVENTION and not self.intervened:
            raise ValueError("intervention mode requires intervened=True")
        if self.error_code is not None and not isinstance(
            self.error_code, NeroErrorCode
        ):
            object.__setattr__(self, "error_code", NeroErrorCode(self.error_code))

    def to_payload(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "sequence_id": self.sequence_id,
            "monotonic_timestamp_ns": self.monotonic_timestamp_ns,
            "policy_action": self.policy_action.tolist(),
            "executed_action": self.executed_action.tolist(),
            "intervened": bool(self.intervened),
            "mode": self.mode.value,
            "error_code": None if self.error_code is None else self.error_code.value,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NeroStepRecord":
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Nero protocol version mismatch")
        return cls(
            sequence_id=int(payload["sequence_id"]),
            monotonic_timestamp_ns=int(payload["monotonic_timestamp_ns"]),
            policy_action=np.asarray(payload["policy_action"], dtype=np.float32),
            executed_action=np.asarray(payload["executed_action"], dtype=np.float32),
            intervened=bool(payload["intervened"]),
            mode=ControlMode(str(payload["mode"])),
            error_code=(
                None
                if payload.get("error_code") is None
                else NeroErrorCode(str(payload["error_code"]))
            ),
        )


@dataclass(frozen=True)
class NeroContract:
    """Single source of truth shared by data, model, environment, and bridge."""

    protocol_version: str = PROTOCOL_VERSION
    physical_action_dim: int = PHYSICAL_ACTION_DIM
    model_action_dim: int = MODEL_ACTION_DIM
    action_horizon: int = ACTION_HORIZON
    action_names: Sequence[str] = PHYSICAL_ACTION_NAMES
    camera_names: Sequence[str] = CAMERA_NAMES
    ros_topics: Mapping[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.ros_topics is None:
            object.__setattr__(self, "ros_topics", dict(ROS_TOPICS))

    def assert_compatible(self, peer_version: str, *, publishing: bool = False) -> None:
        """Reject all version mismatches; identify publish-mode failures clearly."""

        if peer_version != self.protocol_version:
            context = "publish" if publishing else "shadow"
            raise RuntimeError(
                f"Nero protocol mismatch in {context} mode: local={self.protocol_version}, peer={peer_version}"
            )


CONTRACT = NeroContract()
