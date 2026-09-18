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

"""Small framed-msgpack transport shared by the Nero bridge and client."""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from typing import Any, Mapping

import msgpack

MAX_FRAME_BYTES = 64 * 1024 * 1024
_HEADER = struct.Struct("!Q")


class NeroTransportError(RuntimeError):
    """The bridge transport failed or returned a malformed frame."""


@dataclass(frozen=True)
class TcpEndpoint:
    """A validated TCP endpoint."""

    host: str
    port: int

    @classmethod
    def parse(cls, value: str) -> "TcpEndpoint":
        text = str(value).strip()
        if text.startswith("tcp://"):
            text = text[6:]
        host, separator, port_text = text.rpartition(":")
        if not separator or not host:
            raise ValueError("endpoint must be tcp://HOST:PORT")
        port = int(port_text)
        if not 0 <= port <= 65535:
            raise ValueError("endpoint port must be in [0, 65535]")
        return cls(host, port)

    def uri(self) -> str:
        return f"tcp://{self.host}:{self.port}"


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise NeroTransportError("bridge connection closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(connection: socket.socket) -> dict[str, Any]:
    """Receive one bounded msgpack mapping."""

    length = _HEADER.unpack(_recv_exact(connection, _HEADER.size))[0]
    if length < 1 or length > MAX_FRAME_BYTES:
        raise NeroTransportError(f"invalid bridge frame length: {length}")
    try:
        value = msgpack.unpackb(_recv_exact(connection, length), raw=False)
    except (ValueError, msgpack.UnpackException) as error:
        raise NeroTransportError(f"invalid bridge msgpack: {error}") from error
    if not isinstance(value, dict):
        raise NeroTransportError("bridge frame must contain a mapping")
    return value


def send_message(connection: socket.socket, value: Mapping[str, Any]) -> None:
    """Send one bounded msgpack mapping."""

    try:
        payload = msgpack.packb(dict(value), use_bin_type=True)
    except (TypeError, ValueError) as error:
        raise NeroTransportError(
            f"bridge message is not serializable: {error}"
        ) from error
    if not payload or len(payload) > MAX_FRAME_BYTES:
        raise NeroTransportError(f"bridge frame is too large: {len(payload)}")
    connection.sendall(_HEADER.pack(len(payload)) + payload)
