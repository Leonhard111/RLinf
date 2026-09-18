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

"""RLinf-side client for the host-resident Nero bridge."""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from rlinf.utils.latency_logger import get_latency_logger

from .contract import (
    ACTION_HORIZON,
    PHYSICAL_ACTION_DIM,
    PROTOCOL_VERSION,
    ControlMode,
    NeroActionChunk,
    NeroObservation,
)
from .transport import NeroTransportError, TcpEndpoint, recv_message, send_message


class NeroBridgeError(RuntimeError):
    """The bridge rejected a request."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = str(code)
        self.message = str(message)


@dataclass(frozen=True)
class NeroBridgeCapabilities:
    """Control-relevant capabilities reported by the host bridge."""

    mode: ControlMode
    action_horizon: int
    physical_action_dim: int
    control_owner: str
    control_config_sha256: str


class NeroRobotClient:
    """Synchronous client with bounded connect/request deadlines."""

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5555",
        *,
        request_timeout_s: float = 10.0,
        connect_timeout_s: float = 2.0,
        retry_interval_s: float = 0.05,
        heartbeat_interval_s: float = 0.1,
        heartbeat_timeout_s: float = 0.2,
    ) -> None:
        self.endpoint = TcpEndpoint.parse(endpoint)
        self.request_timeout_s = float(request_timeout_s)
        self.connect_timeout_s = float(connect_timeout_s)
        self.retry_interval_s = float(retry_interval_s)
        self.heartbeat_interval_s = float(heartbeat_interval_s)
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        if (
            min(
                self.request_timeout_s,
                self.connect_timeout_s,
                self.retry_interval_s,
                self.heartbeat_interval_s,
                self.heartbeat_timeout_s,
            )
            <= 0
        ):
            raise ValueError("Nero client timeouts must be positive")
        self.last_observation: NeroObservation | None = None
        self._closed = False
        self._heartbeat_stop = threading.Event()
        self._heartbeat_lock = threading.Lock()
        self._last_heartbeat_error: str | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._latency_logger = get_latency_logger("nero_client")
        self.capabilities = self.hello()
        self._start_heartbeat()

    def hello(self) -> NeroBridgeCapabilities:
        response = self._request({"type": "hello"})
        capabilities = NeroBridgeCapabilities(
            mode=ControlMode(response["mode"]),
            action_horizon=int(response["action_horizon"]),
            physical_action_dim=int(response["physical_action_dim"]),
            control_owner=str(response["control_owner"]),
            control_config_sha256=str(response["control_config_sha256"]),
        )
        if (
            capabilities.action_horizon != ACTION_HORIZON
            or capabilities.physical_action_dim != PHYSICAL_ACTION_DIM
            or capabilities.control_owner not in {"vlastop", "pi05-native"}
        ):
            raise NeroBridgeError("protocol_mismatch", str(capabilities))
        return capabilities

    def reset(self) -> NeroObservation:
        started = time.perf_counter()
        after = (
            -1 if self.last_observation is None else self.last_observation.sequence_id
        )
        self.last_observation = self._wait_observation(after)
        self._latency_logger.record(
            {"reset_wait_observation": (time.perf_counter() - started) * 1000.0},
            trace_id=self.last_observation.sequence_id,
        )
        return self.last_observation

    def step(
        self,
        actions: np.ndarray,
        *,
        representation: str = "absolute",
    ) -> NeroObservation:
        if self.last_observation is None:
            raise RuntimeError("NeroRobotClient.reset() must be called before step()")
        step_started = time.perf_counter()
        sequence_id = self.last_observation.sequence_id
        action = NeroActionChunk(
            generation_id=sequence_id,
            monotonic_timestamp_ns=time.monotonic_ns(),
            actions=np.asarray(actions, dtype=np.float32),
            representation=representation,
        )
        action_ready = time.perf_counter()
        self._request({
            "type": "submit_action",
            "sequence_id": sequence_id,
            "action": action.to_payload(),
        })
        submit_finished = time.perf_counter()
        self.last_observation = self._wait_observation(sequence_id)
        observation_ready = time.perf_counter()
        self._latency_logger.record(
            {
                "action_pack": (action_ready - step_started) * 1000.0,
                "submit_action_rtt": (submit_finished - action_ready) * 1000.0,
                "wait_next_observation": (observation_ready - submit_finished) * 1000.0,
                "step_total": (observation_ready - step_started) * 1000.0,
            },
            trace_id=sequence_id,
            metadata={
                "next_sequence_id": self.last_observation.sequence_id,
                "action_shape": list(action.actions.shape),
            },
        )
        return self.last_observation

    def status(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        return self._request({"type": "status"}, timeout_s=timeout_s)

    @property
    def heartbeat_alive(self) -> bool:
        thread = self._heartbeat_thread
        return thread is not None and thread.is_alive()

    @property
    def last_heartbeat_error(self) -> str | None:
        with self._heartbeat_lock:
            return self._last_heartbeat_error

    def _start_heartbeat(self) -> None:
        if self._heartbeat_thread is not None:
            raise RuntimeError("Nero heartbeat is already started")
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="nero-client-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self.heartbeat_interval_s):
            try:
                self.status(timeout_s=self.heartbeat_timeout_s)
            except Exception as error:  # bridge watchdog remains authoritative
                with self._heartbeat_lock:
                    self._last_heartbeat_error = f"{type(error).__name__}: {error}"
            else:
                with self._heartbeat_lock:
                    self._last_heartbeat_error = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._heartbeat_stop.set()
        thread, self._heartbeat_thread = self._heartbeat_thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(
                timeout=max(
                    1.0,
                    self.connect_timeout_s + self.heartbeat_timeout_s + 0.5,
                )
            )
            if thread.is_alive():
                raise RuntimeError("Nero heartbeat thread did not stop cleanly")
        self.last_observation = None

    def _wait_observation(self, after_sequence_id: int) -> NeroObservation:
        deadline = time.monotonic() + self.request_timeout_s
        while True:
            try:
                response = self._request(
                    {
                        "type": "next_observation",
                        "after_sequence_id": int(after_sequence_id),
                    },
                    timeout_s=min(1.0, max(0.05, deadline - time.monotonic())),
                )
            except NeroBridgeError as error:
                if error.code != "not_ready":
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "timed out waiting for a Nero observation"
                    ) from error
                time.sleep(self.retry_interval_s)
                continue
            return NeroObservation.from_payload(response["observation"])

    def _request(
        self, request: Mapping[str, Any], *, timeout_s: float | None = None
    ) -> dict[str, Any]:
        request_type = str(request.get("type", "unknown"))
        log_request = not (
            request_type == "status"
            and threading.current_thread().name == "nero-client-heartbeat"
        )
        request_started = time.perf_counter()
        value = {
            "protocol_version": PROTOCOL_VERSION,
            **dict(request),
        }
        request_timeout = (
            self.request_timeout_s if timeout_s is None else float(timeout_s)
        )
        if request_timeout <= 0:
            raise ValueError("request timeout must be positive")
        try:
            with socket.create_connection(
                (self.endpoint.host, self.endpoint.port),
                timeout=min(self.connect_timeout_s, request_timeout),
            ) as connection:
                connected = time.perf_counter()
                connection.settimeout(request_timeout)
                send_message(connection, value)
                sent = time.perf_counter()
                response = recv_message(connection)
                received = time.perf_counter()
        except (OSError, TimeoutError, NeroTransportError) as error:
            raise NeroTransportError(
                f"Nero bridge request {request.get('type')} failed: {error}"
            ) from error
        if response.get("protocol_version") != PROTOCOL_VERSION:
            raise NeroBridgeError(
                "protocol_mismatch", "bridge response version mismatch"
            )
        if not bool(response.get("ok")):
            raise NeroBridgeError(
                str(response.get("error_code", "bridge_fault")),
                str(response.get("message", "unknown bridge error")),
            )
        if log_request:
            self._latency_logger.record(
                {
                    f"{request_type}.connect": (connected - request_started) * 1000.0,
                    f"{request_type}.serialize_send": (sent - connected) * 1000.0,
                    f"{request_type}.recv_deserialize": (received - sent) * 1000.0,
                    f"{request_type}.request_total": (received - request_started)
                    * 1000.0,
                },
                trace_id=request.get("sequence_id", request.get("after_sequence_id")),
                metadata={
                    "request_type": request_type,
                    "response_type": str(response.get("type", "unknown")),
                },
            )
        return response
