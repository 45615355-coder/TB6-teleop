#!/usr/bin/env python3
"""One-way localhost ACT action-chunk receiver for Product v5.

This module owns no hardware. It creates exactly one PULL socket, receives one
packet, validates it, and never sends data back to the ACT process.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import zmq


ACTION_ENDPOINT = "tcp://127.0.0.1:5558"
ACTION_SCHEMA = "tb6_r5_act_action_chunk_v1"
ACTION_SHAPE = (100, 7)
EXPECTED_CHECKPOINT = "080000"
MOCK_CHECKPOINT = "MOCK_SAFE_ACTION_FOR_DRY_RUN_ONLY"


class ActionPacketError(RuntimeError):
    """Malformed or unauthorized one-chunk action packet."""


def validate_endpoint(endpoint: str) -> str:
    if endpoint != ACTION_ENDPOINT:
        raise ActionPacketError(f"Action endpoint must be localhost-only {ACTION_ENDPOINT}, got {endpoint!r}")
    return endpoint


def decode_action_packet(payload: bytes, *, allow_mock: bool = False) -> dict[str, Any]:
    try:
        packet = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise ActionPacketError(f"Invalid action packet JSON: {exc}") from exc
    if not isinstance(packet, dict) or packet.get("schema") != ACTION_SCHEMA:
        raise ActionPacketError("Unsupported action packet schema")
    checkpoint = packet.get("checkpoint")
    allowed = {EXPECTED_CHECKPOINT}
    if allow_mock:
        allowed.add(MOCK_CHECKPOINT)
    if checkpoint not in allowed:
        raise ActionPacketError(f"Unexpected checkpoint: {checkpoint!r}")
    if packet.get("action_shape") != list(ACTION_SHAPE):
        raise ActionPacketError(f"action_shape must be {list(ACTION_SHAPE)}")
    action = np.asarray(packet.get("action"), dtype=np.float64)
    if action.shape != ACTION_SHAPE:
        raise ActionPacketError(f"action must have shape {ACTION_SHAPE}, got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise ActionPacketError("action contains NaN/Inf")
    try:
        sequence = int(packet["source_observation_sequence"])
        timestamp_ns = int(packet["source_observation_timestamp_ns"])
        inference_ms = float(packet["inference_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ActionPacketError("source sequence/timestamp/inference_ms are invalid") from exc
    if sequence < 0 or timestamp_ns <= 0 or not np.isfinite(inference_ms) or inference_ms < 0.0:
        raise ActionPacketError("source sequence/timestamp/inference_ms are out of range")
    return {
        "schema": ACTION_SCHEMA,
        "checkpoint": checkpoint,
        "source_observation_sequence": sequence,
        "source_observation_timestamp_ns": timestamp_ns,
        "inference_ms": inference_ms,
        "action": action,
    }


class ActionChunkReceiver:
    """Single-channel PULL/bind receiver; there is no Product-to-ACT reply."""

    def __init__(self, endpoint: str = ACTION_ENDPOINT) -> None:
        validate_endpoint(endpoint)
        self._socket = zmq.Context.instance().socket(zmq.PULL)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.bind(endpoint)

    def receive(self, timeout_ms: int, *, allow_mock: bool = False) -> dict[str, Any] | None:
        if timeout_ms < 0:
            raise ValueError("timeout_ms must be non-negative")
        if self._socket.poll(timeout_ms, zmq.POLLIN) == 0:
            return None
        packet = decode_action_packet(self._socket.recv(), allow_mock=allow_mock)
        # One-chunk control consumes exactly one packet. Any queued extra action
        # packets are deliberately never read or executed.
        return packet

    def close(self) -> None:
        self._socket.close(linger=0)
