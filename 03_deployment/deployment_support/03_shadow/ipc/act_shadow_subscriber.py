#!/usr/bin/env python3
"""One-shot, observation-only ACT Shadow subscriber.

This process owns one ZMQ SUB socket only. It never binds/sends a return socket,
opens no robot hardware, and imports no Product control/hardware modules.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault(
    "HF_DATASETS_CACHE",
    str(Path(tempfile.gettempdir()) / "tb6_act_offline_evaluation_hf_cache"),
)

import numpy as np
import zmq


ENDPOINT = "tcp://127.0.0.1:5557"
TOPIC = b"tb6_r5.shadow.observation.v1"
SCHEMA = "tb6_r5_shadow_observation_v1"
VIEW_ORDER = ("front", "top", "wrist")
IMAGE_SHAPE = (480, 640, 3)
CHECKPOINT = Path(
    "/home/robot/data_collect/training_results/pick_place_object_v2/act/v1/train/"
    "20260901_172324/checkpoints/120000/pretrained_model"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/robot/teleop_act/03_deployment/logs/shadow/dynamic"
)
FORBIDDEN_IMPORT_ROOTS = {"jogany_io", "rpc", "gripper_io", "omnipicker_interface", "pyrealsense2", "topic"}


class ShadowProtocolError(RuntimeError):
    """Malformed or unsafe observation packet."""


def assert_no_forbidden_imports() -> None:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=str(Path(__file__)))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0].lower() for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0].lower())
    forbidden = sorted(imported & FORBIDDEN_IMPORT_ROOTS)
    if forbidden:
        raise RuntimeError(f"ACT Shadow source contains forbidden imports: {forbidden}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACT 080000 one-shot local Shadow SUB.")
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--validate-only", action="store_true", help="Validate IPC observation only; do not load ACT.")
    return parser.parse_args()


def validate_endpoint(endpoint: str) -> None:
    if endpoint != ENDPOINT:
        raise ShadowProtocolError(f"Only localhost endpoint {ENDPOINT} is allowed, got {endpoint!r}")


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ShadowProtocolError(f"{name} must be finite with shape {shape}, got {array.shape}")
    return array


def _finite_scalar(value: Any, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ShadowProtocolError(f"{name} is not numeric") from exc
    if not np.isfinite(scalar):
        raise ShadowProtocolError(f"{name} is not finite")
    return scalar


def decode_packet(parts: list[bytes]) -> dict[str, Any]:
    if len(parts) != 5 or parts[0] != TOPIC:
        raise ShadowProtocolError(f"Expected 5-part {TOPIC!r} observation packet")
    try:
        header = json.loads(parts[1].decode("utf-8"))
    except Exception as exc:
        raise ShadowProtocolError(f"Invalid JSON header: {exc}") from exc
    if not isinstance(header, dict) or header.get("schema") != SCHEMA:
        raise ShadowProtocolError("Unsupported observation schema")
    if header.get("camera_keys") != [f"observation.images.{view}" for view in VIEW_ORDER]:
        raise ShadowProtocolError("Camera keys must be front/top/wrist in exact training order")
    if header.get("image_shape") != list(IMAGE_SHAPE):
        raise ShadowProtocolError(f"Image shape must be {IMAGE_SHAPE}")
    if header.get("image_dtype") != "uint8" or header.get("image_color_space") != "RGB":
        raise ShadowProtocolError("Images must be RGB uint8")
    try:
        sequence = int(header["sequence"])
        monotonic_timestamp_ns = int(header["monotonic_timestamp_ns"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ShadowProtocolError("sequence/monotonic_timestamp_ns are invalid") from exc
    if sequence < 0 or monotonic_timestamp_ns <= 0:
        raise ShadowProtocolError("sequence/timestamp must be non-negative/positive")
    q_actual = _finite_array(header.get("q_actual"), (6,), "q_actual")
    gripper_actual = _finite_scalar(header.get("gripper_actual_pos"), "gripper_actual_pos")
    images: dict[str, np.ndarray] = {}
    expected_bytes = int(np.prod(IMAGE_SHAPE))
    for view, payload in zip(VIEW_ORDER, parts[2:], strict=True):
        if len(payload) != expected_bytes:
            raise ShadowProtocolError(f"{view} image bytes={len(payload)}, expected={expected_bytes}")
        image = np.frombuffer(payload, dtype=np.uint8).reshape(IMAGE_SHAPE)
        images[view] = image.copy()
    return {
        "sequence": sequence,
        "monotonic_timestamp_ns": monotonic_timestamp_ns,
        "q_actual": q_actual,
        "gripper_actual_pos": gripper_actual,
        "q_command": header.get("q_command"),
        "gripper_command_pos": header.get("gripper_command_pos"),
        "images": images,
    }


def receive_latest(endpoint: str, timeout_s: float) -> dict[str, Any]:
    if timeout_s <= 0.0:
        raise ValueError("timeout-s must be positive")
    validate_endpoint(endpoint)
    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.SUBSCRIBE, TOPIC)
    socket.connect(endpoint)
    try:
        deadline = time.monotonic() + timeout_s
        latest: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            if socket.poll(remaining_ms) == 0:
                break
            latest = decode_packet(socket.recv_multipart())
            while True:
                try:
                    latest = decode_packet(socket.recv_multipart(flags=zmq.NOBLOCK))
                except zmq.Again:
                    return latest
        if latest is None:
            raise ShadowProtocolError(f"Timed out waiting for observation at {endpoint}")
        return latest
    finally:
        socket.close(linger=0)


def import_evaluation_module() -> Any:
    path = Path(__file__).resolve().parents[2] / "02_evaluate" / "evaluate_act_checkpoints_v2.py"
    spec = importlib.util.spec_from_file_location("tb6_shadow_evaluate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluation module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_policy_observation(packet: dict[str, Any], torch: Any) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "observation.state": torch.from_numpy(
            np.concatenate((packet["q_actual"], np.asarray([packet["gripper_actual_pos"]], dtype=np.float32)))
        ).to(dtype=torch.float32),
    }
    if tuple(observation["observation.state"].shape) != (7,):
        raise ShadowProtocolError("observation.state must be 7D")
    for view in VIEW_ORDER:
        image = packet["images"][view]
        tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().to(dtype=torch.float32).div_(255.0)
        if tuple(tensor.shape) != (3, 480, 640):
            raise ShadowProtocolError(f"{view} tensor shape is invalid: {tuple(tensor.shape)}")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ShadowProtocolError(f"{view} tensor contains NaN/Inf")
        observation[f"observation.images.{view}"] = tensor
    return observation


def write_prediction_csv(path: Path, prediction: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"])
        for step, action in enumerate(prediction):
            writer.writerow([step, *(f"{float(value):.10f}" for value in action)])


def run() -> Path:
    assert_no_forbidden_imports()
    args = parse_args()
    packet = receive_latest(args.endpoint, args.timeout_s)
    output_dir = args.output_root.expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    packet_summary = {
        "sequence": packet["sequence"],
        "monotonic_timestamp_ns": packet["monotonic_timestamp_ns"],
        "q_actual": packet["q_actual"].tolist(),
        "gripper_actual_pos": packet["gripper_actual_pos"],
        "image_shape": list(IMAGE_SHAPE),
        "image_color_space": "RGB",
        "validate_only": args.validate_only,
    }

    if args.validate_only:
        (output_dir / "observation_validation.json").write_text(
            json.dumps(packet_summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[PASS] Fake/IPC observation validated: {output_dir}")
        print(json.dumps(packet_summary, indent=2))
        return output_dir

    import torch
    import lerobot

    evalmod = import_evaluation_module()
    if lerobot.__version__ != evalmod.EXPECTED_LEROBOT_VERSION:
        raise RuntimeError(f"LeRobot must be {evalmod.EXPECTED_LEROBOT_VERSION}, got {lerobot.__version__}")
    evalmod.validate_checkpoint(CHECKPOINT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, preprocessor, postprocessor, runtime_config = evalmod.load_policy_and_processors(CHECKPOINT, device)
    if (
        runtime_config.chunk_size != 100
        or runtime_config.n_action_steps != 100
        or runtime_config.n_obs_steps != 1
        or runtime_config.temporal_ensemble_coeff is not None
    ):
        raise RuntimeError("Checkpoint runtime ACT config is not the required 100/100/1/no-ensemble")
    observation = make_policy_observation(packet, torch)
    processed = preprocessor(observation)
    start_ns = time.perf_counter_ns()
    with torch.inference_mode():
        prediction_tensor = policy.predict_action_chunk(processed)
    inference_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    evalmod.require_finite_tensor("normalized Shadow prediction", prediction_tensor)
    prediction_tensor = postprocessor(prediction_tensor)
    evalmod.require_finite_tensor("postprocessed Shadow prediction", prediction_tensor)
    if tuple(prediction_tensor.shape) != (1, 100, 7):
        raise RuntimeError(f"Prediction shape must be (1,100,7), got {tuple(prediction_tensor.shape)}")
    prediction = prediction_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Postprocessed prediction contains NaN/Inf")
    q_actual = packet["q_actual"].astype(np.float32, copy=False)
    arm_prediction = prediction[:, :6]
    arm_delta = np.abs(np.diff(arm_prediction, axis=0))
    summary = {
        **packet_summary,
        "lerobot_version": lerobot.__version__,
        "torch_version": torch.__version__,
        "device": str(device),
        "prediction_shape": list(prediction.shape),
        "first_predicted_action": prediction[0].tolist(),
        "joint_min": arm_prediction.min(axis=0).tolist(),
        "joint_max": arm_prediction.max(axis=0).tolist(),
        "joint_max_adjacent_delta": arm_delta.max(axis=0).tolist(),
        "joint_max_deviation_from_current_q_actual": np.abs(arm_prediction - q_actual[None, :]).max(axis=0).tolist(),
        "gripper_min": float(prediction[:, 6].min()),
        "gripper_max": float(prediction[:, 6].max()),
        "prediction_has_nan_or_inf": bool(not np.isfinite(prediction).all()),
        "predict_action_chunk_ms": inference_ms,
    }
    np.savez_compressed(
        output_dir / "prediction.npz",
        prediction_action=prediction,
        q_actual=q_actual,
        gripper_actual_pos=np.asarray(packet["gripper_actual_pos"], dtype=np.float32),
    )
    write_prediction_csv(output_dir / "prediction.csv", prediction)
    (output_dir / "shadow_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("[SHADOW] ACT observation-only prediction complete")
    print(json.dumps(summary, indent=2))
    print(f"[OUTPUT] {output_dir}")
    return output_dir


def main() -> int:
    try:
        run()
    except Exception as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
