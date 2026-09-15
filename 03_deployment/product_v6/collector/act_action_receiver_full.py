#!/usr/bin/env python3
"""Validated one-way localhost receiver for selected full-deployment ACT actions."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import zmq


ACTION_ENDPOINT = "tcp://127.0.0.1:5558"
ACTION_SCHEMA = "tb6_r5_act_stream_action_v1"
ACTION_SHAPE = (7,)
EXPECTED_CHECKPOINT = "120000"
MOCK_CHECKPOINT = "MOCK_SAFE_ACTION_FOR_DRY_RUN_ONLY"
TEMPORAL_ENSEMBLE_COEFF = 0.01


class FullActionPacketError(RuntimeError):
    pass


def validate_endpoint(endpoint: str) -> str:
    if endpoint != ACTION_ENDPOINT:
        raise FullActionPacketError(
            f"Action endpoint must be localhost-only {ACTION_ENDPOINT}, got {endpoint!r}"
        )
    return endpoint


def decode_full_action_packet(payload: bytes, *, allow_mock: bool = False) -> dict[str, Any]:
    try:
        packet = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise FullActionPacketError(f"Invalid action packet JSON: {exc}") from exc
    if not isinstance(packet, dict) or packet.get("schema") != ACTION_SCHEMA:
        raise FullActionPacketError("Unsupported full action packet schema")
    checkpoint = packet.get("checkpoint")
    allowed = {EXPECTED_CHECKPOINT}
    if allow_mock:
        allowed.add(MOCK_CHECKPOINT)
    if checkpoint not in allowed:
        raise FullActionPacketError(f"Unexpected checkpoint: {checkpoint!r}")
    if packet.get("action_shape") != list(ACTION_SHAPE):
        raise FullActionPacketError(f"action_shape must be {list(ACTION_SHAPE)}")
    action = np.asarray(packet.get("action"), dtype=np.float64)
    if action.shape != ACTION_SHAPE or not np.all(np.isfinite(action)):
        raise FullActionPacketError("selected action must be finite [7]")
    try:
        step_index = int(packet["step_index"])
        sequence = int(packet["source_observation_sequence"])
        timestamp_ns = int(packet["source_observation_timestamp_ns"])
        observation_age_ms = float(packet["observation_age_ms_at_inference"])
        inference_ms = float(packet["inference_ms"])
        temporal_coeff = float(packet["temporal_ensemble_coeff"])
        speed_scale = float(packet["speed_scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FullActionPacketError("full action metadata is invalid") from exc
    finite_nonnegative = (observation_age_ms, inference_ms)
    if (
        step_index < 0
        or sequence < 0
        or timestamp_ns <= 0
        or not all(np.isfinite(value) and value >= 0.0 for value in finite_nonnegative)
        or not np.isfinite(temporal_coeff)
        or abs(temporal_coeff - TEMPORAL_ENSEMBLE_COEFF) > 1e-12
        or not np.isfinite(speed_scale)
        or not 0.0 < speed_scale <= 1.0
    ):
        raise FullActionPacketError("full action metadata is out of range")
    return {
        "schema": ACTION_SCHEMA,
        "checkpoint": checkpoint,
        "step_index": step_index,
        "source_observation_sequence": sequence,
        "source_observation_timestamp_ns": timestamp_ns,
        "observation_age_ms_at_inference": observation_age_ms,
        "inference_ms": inference_ms,
        "temporal_ensemble_coeff": temporal_coeff,
        "speed_scale": speed_scale,
        "action": action,
    }


class FullActionReceiver:
    """Persistent PULL/bind receiver enforcing ordered steps and observations."""

    def __init__(self, endpoint: str = ACTION_ENDPOINT) -> None:
        validate_endpoint(endpoint)
        self._socket = zmq.Context.instance().socket(zmq.PULL)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 2)
        self._socket.bind(endpoint)
        self._previous_step: int | None = None
        self._previous_sequence: int | None = None

    def receive(self, timeout_ms: int, *, allow_mock: bool = False) -> dict[str, Any] | None:
        if timeout_ms < 0:
            raise ValueError("timeout_ms must be non-negative")
        if self._socket.poll(timeout_ms, zmq.POLLIN) == 0:
            return None
        packet = decode_full_action_packet(self._socket.recv(), allow_mock=allow_mock)
        expected_step = 0 if self._previous_step is None else self._previous_step + 1
        if packet["step_index"] != expected_step:
            raise FullActionPacketError(
                f"step_index must be {expected_step}, got {packet['step_index']}"
            )
        if (
            self._previous_sequence is not None
            and packet["source_observation_sequence"] <= self._previous_sequence
        ):
            raise FullActionPacketError("source observation sequence must strictly increase")
        self._previous_step = packet["step_index"]
        self._previous_sequence = packet["source_observation_sequence"]
        return packet

    def close(self) -> None:
        self._socket.close(linger=0)


__all__ = [
    "ACTION_ENDPOINT",
    "ACTION_SCHEMA",
    "EXPECTED_CHECKPOINT",
    "FullActionPacketError",
    "FullActionReceiver",
    "decode_full_action_packet",
]
