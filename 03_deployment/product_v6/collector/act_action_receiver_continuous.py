#!/usr/bin/env python3
"""Continuous one-way localhost ACT action receiver for Product v5."""

from __future__ import annotations

import json
from typing import Any

import zmq

from act_action_receiver import (
    ACTION_ENDPOINT,
    ACTION_SCHEMA,
    ActionPacketError,
    decode_action_packet,
    validate_endpoint,
)


def decode_continuous_action_packet(payload: bytes, *, allow_mock: bool = False) -> dict[str, Any]:
    """Reuse the verified one-chunk decoder, adding only the chunk index."""
    try:
        raw = json.loads(payload.decode("utf-8"))
        chunk_index = int(raw["chunk_index"])
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ActionPacketError("continuous action packet chunk_index is invalid") from exc
    if chunk_index < 0:
        raise ActionPacketError("continuous action packet chunk_index must be non-negative")
    packet = decode_action_packet(payload, allow_mock=allow_mock)
    packet["chunk_index"] = chunk_index
    return packet


class ContinuousActionChunkReceiver:
    """Persistent PULL/bind socket. Product never sends an ACT reply."""

    def __init__(self, endpoint: str = ACTION_ENDPOINT) -> None:
        validate_endpoint(endpoint)
        self._socket = zmq.Context.instance().socket(zmq.PULL)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 3)
        self._socket.bind(endpoint)

    def receive(self, timeout_ms: int, *, allow_mock: bool = False) -> dict[str, Any] | None:
        if timeout_ms < 0:
            raise ValueError("timeout_ms must be non-negative")
        if self._socket.poll(timeout_ms, zmq.POLLIN) == 0:
            return None
        return decode_continuous_action_packet(self._socket.recv(), allow_mock=allow_mock)

    def close(self) -> None:
        self._socket.close(linger=0)


__all__ = [
    "ACTION_ENDPOINT",
    "ACTION_SCHEMA",
    "ActionPacketError",
    "ContinuousActionChunkReceiver",
    "decode_continuous_action_packet",
]
