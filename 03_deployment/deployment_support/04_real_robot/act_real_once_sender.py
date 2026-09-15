#!/usr/bin/env python3
"""Infer one ACT chunk and send it once to Product v5 over localhost PUSH."""

from __future__ import annotations

import argparse
import ast
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


SHADOW_IPC_ROOT = Path(__file__).resolve().parents[1] / "03_shadow" / "ipc"
sys.path.insert(0, str(SHADOW_IPC_ROOT))
import act_shadow_subscriber as shadow  # noqa: E402


ACTION_ENDPOINT = "tcp://127.0.0.1:5558"
ACTION_SCHEMA = "tb6_r5_act_action_chunk_v1"
MOCK_CHECKPOINT = "MOCK_SAFE_ACTION_FOR_DRY_RUN_ONLY"
DEFAULT_OUTPUT_ROOT = Path(
    "/home/robot/teleop_act/03_deployment/logs/pick_place_object_v2/once_sender"
)


def assert_no_forbidden_imports() -> None:
    shadow.assert_no_forbidden_imports()
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=str(Path(__file__)))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0].lower() for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0].lower())
    forbidden = sorted(imported & shadow.FORBIDDEN_IMPORT_ROOTS)
    if forbidden:
        raise RuntimeError(f"ACT real sender source contains forbidden imports: {forbidden}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACT 080000 one-chunk localhost action sender.")
    parser.add_argument("--observation-endpoint", default=shadow.ENDPOINT)
    parser.add_argument("--action-endpoint", default=ACTION_ENDPOINT)
    parser.add_argument("--observation-timeout-s", type=float, default=30.0)
    parser.add_argument("--action-send-timeout-ms", type=int, default=5000)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mock-safe-action", action="store_true",
                        help="Dry-run fixture only; Product real mode rejects this packet.")
    return parser.parse_args()


def validate_action_endpoint(endpoint: str) -> None:
    if endpoint != ACTION_ENDPOINT:
        raise RuntimeError(f"Action PUSH endpoint must be localhost-only {ACTION_ENDPOINT}, got {endpoint!r}")


def infer_real_action(packet: dict[str, Any]) -> tuple[np.ndarray, float, str, dict[str, str]]:
    import torch
    import lerobot

    evaluation = shadow.import_evaluation_module()
    if lerobot.__version__ != evaluation.EXPECTED_LEROBOT_VERSION:
        raise RuntimeError(f"LeRobot must be {evaluation.EXPECTED_LEROBOT_VERSION}, got {lerobot.__version__}")
    evaluation.validate_checkpoint(shadow.CHECKPOINT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, preprocessor, postprocessor, runtime_config = evaluation.load_policy_and_processors(shadow.CHECKPOINT, device)
    if (
        runtime_config.chunk_size != 100
        or runtime_config.n_action_steps != 100
        or runtime_config.n_obs_steps != 1
        or runtime_config.temporal_ensemble_coeff is not None
    ):
        raise RuntimeError("Checkpoint runtime ACT config is not the required 100/100/1/no-ensemble")
    policy.eval()
    observation = shadow.make_policy_observation(packet, torch)
    processed = preprocessor(observation)
    started_ns = time.perf_counter_ns()
    with torch.inference_mode():
        prediction_tensor = policy.predict_action_chunk(processed)
    inference_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    evaluation.require_finite_tensor("normalized ACT real-once prediction", prediction_tensor)
    prediction_tensor = postprocessor(prediction_tensor)
    evaluation.require_finite_tensor("postprocessed ACT real-once prediction", prediction_tensor)
    if tuple(prediction_tensor.shape) != (1, 100, 7):
        raise RuntimeError(f"Prediction shape must be (1,100,7), got {tuple(prediction_tensor.shape)}")
    action = prediction_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    if action.shape != (100, 7) or not np.all(np.isfinite(action)):
        raise RuntimeError("ACT real-once prediction must be finite [100,7]")
    versions = {"lerobot_version": lerobot.__version__, "torch_version": torch.__version__, "device": str(device)}
    versions["checkpoint_path"] = str(shadow.CHECKPOINT)
    return action, inference_ms, "080000", versions


def make_mock_action(packet: dict[str, Any]) -> tuple[np.ndarray, float, str, dict[str, str]]:
    """Create a smooth protocol fixture anchored to the fake observation."""
    q_actual = np.asarray(packet["q_actual"], dtype=np.float32)
    fractions = np.linspace(0.01, 1.0, 100, dtype=np.float32)[:, None]
    direction = np.asarray((1.0, -1.0, 1.0, -1.0, 1.0, -1.0), dtype=np.float32)
    arm = q_actual[None, :] + fractions * (0.05 * direction[None, :])
    gripper = np.full((100, 1), float(packet["gripper_actual_pos"]), dtype=np.float32)
    return np.concatenate((arm, gripper), axis=1), 0.0, MOCK_CHECKPOINT, {
        "lerobot_version": "not_loaded_mock_mode",
        "torch_version": "not_loaded_mock_mode",
        "device": "none_mock_mode",
    }


def send_action_packet(endpoint: str, packet: dict[str, Any], timeout_ms: int) -> None:
    validate_action_endpoint(endpoint)
    if timeout_ms <= 0:
        raise ValueError("--action-send-timeout-ms must be positive")
    socket = zmq.Context.instance().socket(zmq.PUSH)
    socket.setsockopt(zmq.LINGER, 1000)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.connect(endpoint)
    try:
        socket.send(json.dumps(packet, separators=(",", ":")).encode("utf-8"))
    finally:
        socket.close()


def run() -> Path:
    assert_no_forbidden_imports()
    args = parse_args()
    shadow.validate_endpoint(args.observation_endpoint)
    validate_action_endpoint(args.action_endpoint)
    observation = shadow.receive_latest(args.observation_endpoint, args.observation_timeout_s)
    if args.mock_safe_action:
        action, inference_ms, checkpoint, versions = make_mock_action(observation)
    else:
        action, inference_ms, checkpoint, versions = infer_real_action(observation)
    if action.shape != (100, 7) or not np.all(np.isfinite(action)):
        raise RuntimeError("Refusing to send: action must be finite [100,7]")
    action_packet = {
        "schema": ACTION_SCHEMA,
        "checkpoint": checkpoint,
        "source_observation_sequence": int(observation["sequence"]),
        "source_observation_timestamp_ns": int(observation["monotonic_timestamp_ns"]),
        "inference_ms": float(inference_ms),
        "action_shape": [100, 7],
        "action": action.tolist(),
    }
    output_dir = args.output_root.expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output_dir / "action_chunk.npz",
        action=action,
        source_observation_sequence=np.asarray(observation["sequence"], dtype=np.int64),
        source_observation_timestamp_ns=np.asarray(observation["monotonic_timestamp_ns"], dtype=np.int64),
    )
    (output_dir / "action_packet.json").write_text(json.dumps(action_packet, indent=2) + "\n", encoding="utf-8")
    summary = {
        **versions,
        "checkpoint": checkpoint,
        "mock_safe_action": args.mock_safe_action,
        "source_observation_sequence": observation["sequence"],
        "source_observation_timestamp_ns": observation["monotonic_timestamp_ns"],
        "inference_ms": inference_ms,
        "action_shape": list(action.shape),
        "finite": bool(np.all(np.isfinite(action))),
        "first_action": action[0].tolist(),
        "gripper_min": float(action[:, 6].min()),
        "gripper_max": float(action[:, 6].max()),
        "action_channel": "PUSH/connect once",
    }
    (output_dir / "sender_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    send_action_packet(args.action_endpoint, action_packet, args.action_send_timeout_ms)
    print(
        f"[ACT-REAL-SENDER] SENT ONCE | seq={observation['sequence']} | "
        f"shape={action.shape} | finite=PASS | infer={inference_ms:.1f}ms",
        flush=True,
    )
    print("[ACT-REAL-SENDER] EXIT | no more action packets will be sent", flush=True)
    print(f"[OUTPUT] {output_dir}", flush=True)
    return output_dir


def main() -> int:
    try:
        run()
    except Exception as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
