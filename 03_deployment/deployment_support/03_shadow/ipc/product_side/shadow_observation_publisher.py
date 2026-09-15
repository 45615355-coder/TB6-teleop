#!/usr/bin/env python3
"""One-way, localhost-only observation PUB for TB6-R5 ACT Shadow.

This module owns no hardware. In Product mode it receives references to already
running Product objects and reads only their completed caches/snapshots. It
contains no robot, RPC, JogAny, or OmniPicker control call.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import zmq


ENDPOINT = "tcp://127.0.0.1:5557"
TOPIC = b"tb6_r5.shadow.observation.v1"
SCHEMA = "tb6_r5_shadow_observation_v1"
VIEW_ORDER = ("front", "top", "wrist")
IMAGE_SHAPE = (480, 640, 3)


class ShadowPublishError(RuntimeError):
    """Invalid observation cache/snapshot; never a robot-control error."""


def _validate_endpoint(endpoint: str) -> str:
    if endpoint != ENDPOINT:
        raise ShadowPublishError(f"Shadow endpoint must be localhost-only {ENDPOINT}, got {endpoint!r}")
    return endpoint


def _finite_vector(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ShadowPublishError(f"{name} must be finite with shape {shape}, got {array.shape}")
    return array


def _finite_scalar(value: Any, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ShadowPublishError(f"{name} is not numeric") from exc
    if not math.isfinite(scalar):
        raise ShadowPublishError(f"{name} is not finite")
    return scalar


def _validate_rgb_uint8(image: Any, view: str) -> np.ndarray:
    array = np.asarray(image)
    if array.shape != IMAGE_SHAPE:
        raise ShadowPublishError(f"{view} image shape must be {IMAGE_SHAPE}, got {array.shape}")
    if array.dtype != np.uint8:
        raise ShadowPublishError(f"{view} image dtype must be uint8, got {array.dtype}")
    return np.ascontiguousarray(array)


class ShadowObservationPublisher:
    """Latest-only PUB that reads existing Product camera/gripper/topic state."""

    def __init__(
        self,
        *,
        camera_serial_dict: Mapping[str, str],
        endpoint: str = ENDPOINT,
        publish_hz: float = 10.0,
    ) -> None:
        _validate_endpoint(endpoint)
        if publish_hz <= 0.0:
            raise ValueError("publish_hz must be positive")
        if tuple(camera_serial_dict.keys()) != VIEW_ORDER:
            raise ValueError(f"camera_serial_dict keys must be exactly {VIEW_ORDER}")
        self._serials = {view: str(camera_serial_dict[view]) for view in VIEW_ORDER}
        self._period_s = 1.0 / float(publish_hz)
        self._next_publish_s = 0.0
        self._sequence = 0
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.bind(endpoint)

    def close(self) -> None:
        self._socket.close(linger=0)

    def publish_if_due(
        self,
        *,
        camera: Any,
        topic: Any,
        gripper_snapshot: Mapping[str, Any],
        q_command: Any | None,
    ) -> bool:
        """Read only completed caches/snapshots; return False when no publish is due."""
        now_s = time.monotonic()
        if now_s < self._next_publish_s:
            return False
        self._next_publish_s = now_s + self._period_s
        if camera is None or not callable(getattr(camera, "get_frames", None)):
            raise ShadowPublishError("existing camera cache is unavailable")
        q_actual = _finite_vector(getattr(topic, "q_actual_rad", None), (6,), "q_actual")
        gripper_actual = _finite_scalar(gripper_snapshot.get("actual_pos"), "gripper_actual_pos")
        raw_frames = camera.get_frames()
        images: dict[str, np.ndarray] = {}
        for view in VIEW_ORDER:
            frame = raw_frames.get(self._serials[view])
            if not isinstance(frame, Mapping) or "color" not in frame:
                raise ShadowPublishError(f"existing {view} camera cache is incomplete")
            images[view] = _validate_rgb_uint8(frame["color"], view)

        q_command_list: list[float] | None = None
        if q_command is not None:
            q_command_list = _finite_vector(q_command, (6,), "q_command").tolist()
        gripper_command = gripper_snapshot.get("target_pos")
        if gripper_command is not None:
            gripper_command = _finite_scalar(gripper_command, "gripper_command_pos")

        header = {
            "schema": SCHEMA,
            "sequence": self._sequence,
            "monotonic_timestamp_ns": time.monotonic_ns(),
            "q_actual": q_actual.tolist(),
            "gripper_actual_pos": gripper_actual,
            "q_command": q_command_list,
            "gripper_command_pos": gripper_command,
            "camera_keys": [f"observation.images.{view}" for view in VIEW_ORDER],
            "image_shape": list(IMAGE_SHAPE),
            "image_dtype": "uint8",
            "image_color_space": "RGB",
        }
        payload = [
            TOPIC,
            json.dumps(header, separators=(",", ":")).encode("utf-8"),
            *(images[view].tobytes(order="C") for view in VIEW_ORDER),
        ]
        try:
            self._socket.send_multipart(payload, flags=zmq.NOBLOCK)
        except zmq.Again:
            return False
        self._sequence += 1
        return True


def _fake_images(sequence: int) -> dict[str, np.ndarray]:
    images = {view: np.zeros(IMAGE_SHAPE, dtype=np.uint8) for view in VIEW_ORDER}
    images["front"][:, :, 0] = (20 + sequence) % 255
    images["top"][:, :, 1] = (80 + sequence) % 255
    images["wrist"][:, :, 2] = (140 + sequence) % 255
    return images


def run_fake(endpoint: str, count: int, publish_hz: float, startup_delay_s: float) -> int:
    _validate_endpoint(endpoint)
    if count <= 0 or publish_hz <= 0.0 or startup_delay_s < 0.0:
        raise ValueError("fake count/hz/delay must be valid")
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.bind(endpoint)
    try:
        print(f"[FAKE-PUB] endpoint={endpoint} count={count} hz={publish_hz}", flush=True)
        time.sleep(startup_delay_s)
        period_s = 1.0 / publish_hz
        for sequence in range(count):
            images = _fake_images(sequence)
            header = {
                "schema": SCHEMA,
                "sequence": sequence,
                "monotonic_timestamp_ns": time.monotonic_ns(),
                "q_actual": [0.01 * index for index in range(6)],
                "gripper_actual_pos": 50.0,
                "q_command": [0.01 * index for index in range(6)],
                "gripper_command_pos": 50.0,
                "camera_keys": [f"observation.images.{view}" for view in VIEW_ORDER],
                "image_shape": list(IMAGE_SHAPE),
                "image_dtype": "uint8",
                "image_color_space": "RGB",
            }
            socket.send_multipart(
                [
                    TOPIC,
                    json.dumps(header, separators=(",", ":")).encode("utf-8"),
                    *(images[view].tobytes(order="C") for view in VIEW_ORDER),
                ]
            )
            time.sleep(period_s)
        print("[FAKE-PUB] complete", flush=True)
    finally:
        socket.close(linger=0)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Local-only fake observation PUB for Shadow IPC testing.")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--fake-count", type=int, default=100)
    parser.add_argument("--fake-hz", type=float, default=10.0)
    parser.add_argument("--startup-delay-s", type=float, default=1.0)
    args = parser.parse_args()
    if not args.fake:
        parser.error("This CLI supports --fake only. Product teleop imports the publisher class directly.")
    return run_fake(args.endpoint, args.fake_count, args.fake_hz, args.startup_delay_s)


if __name__ == "__main__":
    raise SystemExit(main())
