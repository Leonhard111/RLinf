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

"""Host-side rendezvous server for a remote Nero policy."""

from __future__ import annotations

import socketserver
import threading
import time
from typing import Any, Mapping

from rlinf.utils.latency_logger import get_latency_logger

from .contract import (
    ACTION_HORIZON,
    PHYSICAL_ACTION_DIM,
    PROTOCOL_VERSION,
    ControlMode,
    NeroActionChunk,
    NeroObservation,
)
from .transport import TcpEndpoint, recv_message, send_message


class BridgeClosedError(RuntimeError):
    """The policy exchange has been closed."""


class BridgeNotReadyError(TimeoutError):
    """No newer observation became available before the bounded deadline."""


class PolicyExchange:
    """Thread-safe one-observation/one-action rendezvous.

    The host inference worker offers an observation and waits for one
    correlated action chunk. The TCP server exposes that rendezvous to the
    lightweight RLinf environment worker.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._observation: NeroObservation | None = None
        self._actions: dict[int, NeroActionChunk] = {}
        self._closed = False
        self._latency_logger = get_latency_logger("policy_exchange")

    def offer_observation(
        self, observation: NeroObservation, *, timeout_s: float
    ) -> NeroActionChunk:
        started = time.perf_counter()
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            if self._closed:
                raise BridgeClosedError("policy exchange is closed")
            if (
                self._observation is not None
                and self._observation.sequence_id >= observation.sequence_id
            ):
                raise RuntimeError("observation sequence must be strictly increasing")
            self._observation = observation
            self._condition.notify_all()
            while observation.sequence_id not in self._actions:
                if self._closed:
                    raise BridgeClosedError("policy exchange is closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"remote action timeout for sequence {observation.sequence_id}"
                    )
                self._condition.wait(remaining)
            action = self._actions.pop(observation.sequence_id)
        finished = time.perf_counter()
        self._latency_logger.record(
            {"observation_to_action": (finished - started) * 1000.0},
            trace_id=observation.sequence_id,
            metadata={"action_shape": list(action.actions.shape)},
        )
        return action

    def wait_observation(
        self, *, after_sequence_id: int, timeout_s: float
    ) -> NeroObservation:
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while self._observation is None or self._observation.sequence_id <= int(
                after_sequence_id
            ):
                if self._closed:
                    raise BridgeClosedError("policy exchange is closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BridgeNotReadyError("no newer Nero observation")
                self._condition.wait(remaining)
            return self._observation

    def submit_action(self, sequence_id: int, action: NeroActionChunk) -> None:
        expected = int(sequence_id)
        with self._condition:
            if self._closed:
                raise BridgeClosedError("policy exchange is closed")
            if self._observation is None:
                raise BridgeNotReadyError("there is no pending observation")
            if self._observation.sequence_id != expected:
                raise ValueError(
                    "action sequence does not match the pending observation: "
                    f"expected {self._observation.sequence_id}, got {expected}"
                )
            if action.generation_id != expected:
                raise ValueError(
                    "action generation does not match its observation sequence"
                )
            if expected in self._actions:
                raise ValueError(f"duplicate action for sequence {expected}")
            self._actions[expected] = action
            self._condition.notify_all()

    def status(self) -> dict[str, Any]:
        with self._condition:
            return {
                "closed": self._closed,
                "latest_sequence_id": (
                    None if self._observation is None else self._observation.sequence_id
                ),
                "pending_action_sequences": sorted(self._actions),
            }

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._actions.clear()
            self._condition.notify_all()


class _ThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class NeroBridgeServer:
    """A one-request-per-connection framed TCP bridge server."""

    def __init__(
        self,
        endpoint: str,
        exchange: PolicyExchange,
        *,
        mode: ControlMode = ControlMode.SHADOW,
        control_config_sha256: str,
        control_owner: str = "vlastop",
        wait_timeout_s: float = 0.25,
    ) -> None:
        parsed = TcpEndpoint.parse(endpoint)
        self.exchange = exchange
        self.mode = ControlMode(mode)
        self.control_config_sha256 = str(control_config_sha256)
        self.control_owner = str(control_owner).strip()
        self.wait_timeout_s = float(wait_timeout_s)
        if not self.control_owner:
            raise ValueError("control_owner must not be empty")
        if self.wait_timeout_s <= 0:
            raise ValueError("wait_timeout_s must be positive")
        self._activity_lock = threading.Lock()
        self._last_client_activity_ns: int | None = None
        self._latency_logger = get_latency_logger("nero_bridge_server")
        owner = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                started = time.perf_counter()
                request: Mapping[str, Any] = {}
                try:
                    request = recv_message(self.request)
                    received = time.perf_counter()
                    response = owner._handle(request)
                    handled = time.perf_counter()
                except Exception as error:  # defensive protocol boundary
                    received = time.perf_counter()
                    handled = received
                    response = owner._error("bridge_fault", str(error))
                send_message(self.request, response)
                sent = time.perf_counter()
                request_type = str(request.get("type", "unknown"))
                owner._latency_logger.record(
                    {
                        f"{request_type}.recv_deserialize": (received - started)
                        * 1000.0,
                        f"{request_type}.handle": (handled - received) * 1000.0,
                        f"{request_type}.serialize_send": (sent - handled) * 1000.0,
                        f"{request_type}.server_total": (sent - started) * 1000.0,
                    },
                    trace_id=request.get(
                        "sequence_id", request.get("after_sequence_id")
                    ),
                    metadata={
                        "request_type": request_type,
                        "response_type": str(response.get("type", "unknown")),
                        "ok": bool(response.get("ok")),
                    },
                )

        self._server = _ThreadingServer((parsed.host, parsed.port), Handler)
        actual_host, actual_port = self._server.server_address[:2]
        self.endpoint = TcpEndpoint(str(actual_host), int(actual_port)).uri()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Nero bridge server is already started")
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="nero-bridge-tcp",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def _handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if request.get("protocol_version") != PROTOCOL_VERSION:
            return self._error("protocol_mismatch", "Nero protocol version mismatch")
        self._mark_client_activity()
        request_type = str(request.get("type", ""))
        if request_type == "hello":
            return self._ok(
                "hello",
                mode=self.mode.value,
                action_horizon=ACTION_HORIZON,
                physical_action_dim=PHYSICAL_ACTION_DIM,
                control_owner=self.control_owner,
                control_config_sha256=self.control_config_sha256,
            )
        if request_type == "status":
            return self._ok("status", mode=self.mode.value, **self.exchange.status())
        if request_type == "next_observation":
            try:
                observation = self.exchange.wait_observation(
                    after_sequence_id=int(request.get("after_sequence_id", -1)),
                    timeout_s=self.wait_timeout_s,
                )
            except BridgeNotReadyError as error:
                return self._error("not_ready", str(error))
            return self._ok("observation", observation=observation.to_payload())
        if request_type == "submit_action":
            try:
                action = NeroActionChunk.from_payload(request["action"])
                self.exchange.submit_action(int(request["sequence_id"]), action)
            except (BridgeNotReadyError, KeyError, TypeError, ValueError) as error:
                return self._error("invalid_action", str(error))
            return self._ok("action_accepted", sequence_id=action.generation_id)
        return self._error("bridge_fault", f"unknown bridge request: {request_type}")

    def client_activity_age_s(self, now_ns: int | None = None) -> float | None:
        """Return seconds since the last valid client request, if any."""

        with self._activity_lock:
            last_activity_ns = self._last_client_activity_ns
        if last_activity_ns is None:
            return None
        timestamp_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        return max(0.0, (timestamp_ns - last_activity_ns) / 1e9)

    def _mark_client_activity(self) -> None:
        with self._activity_lock:
            self._last_client_activity_ns = time.monotonic_ns()

    @staticmethod
    def _ok(response_type: str, **values: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "type": response_type,
            "protocol_version": PROTOCOL_VERSION,
            **values,
        }

    @staticmethod
    def _error(code: str, message: str) -> dict[str, Any]:
        return {
            "ok": False,
            "type": "error",
            "protocol_version": PROTOCOL_VERSION,
            "error_code": code,
            "message": message,
        }
