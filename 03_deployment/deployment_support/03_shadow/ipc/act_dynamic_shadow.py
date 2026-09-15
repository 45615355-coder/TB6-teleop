#!/usr/bin/env python3
"""Continuous, observation-only ACT Shadow subscriber.

This process owns exactly one localhost ZMQ SUB socket. It receives Product v5
observations, predicts ACT action chunks, and writes records only. It never
creates a return/control socket and imports no Product hardware modules.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
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

import act_shadow_subscriber as oneshot


DEFAULT_OUTPUT_ROOT = Path(
    "/home/robot/teleop_act/03_deployment/logs/shadow/dynamic_shadow"
)
FORBIDDEN_IMPORT_ROOTS = oneshot.FORBIDDEN_IMPORT_ROOTS


class DynamicShadowError(RuntimeError):
    """A dynamic Shadow failure that must stop inference safely."""


def assert_no_forbidden_imports() -> None:
    """Reuse the one-shot guard and check this new source file as well."""
    oneshot.assert_no_forbidden_imports()
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=str(Path(__file__)))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0].lower() for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0].lower())
    forbidden = sorted(imported & FORBIDDEN_IMPORT_ROOTS)
    if forbidden:
        raise RuntimeError(f"Dynamic Shadow source contains forbidden imports: {forbidden}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACT 080000 continuous local Shadow SUB.")
    parser.add_argument("--endpoint", default=oneshot.ENDPOINT)
    parser.add_argument("--inference-period-s", type=float, default=2.0)
    parser.add_argument("--packet-timeout-s", type=float, default=10.0)
    parser.add_argument("--max-inferences", type=int, default=None,
                        help="Test-only bounded run; omitted means run until Ctrl+C.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    oneshot.validate_endpoint(args.endpoint)
    if args.inference_period_s <= 0.0:
        raise ValueError("--inference-period-s must be positive")
    if args.packet_timeout_s <= 0.0:
        raise ValueError("--packet-timeout-s must be positive")
    if args.max_inferences is not None and args.max_inferences <= 0:
        raise ValueError("--max-inferences must be positive when supplied")


def receive_latest_from_socket(socket: zmq.Socket, timeout_s: float) -> dict[str, Any] | None:
    """Wait for one packet then drain all queued packets, retaining only the newest."""
    deadline = time.monotonic() + timeout_s
    latest: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        remaining_ms = max(1, int((deadline - time.monotonic()) * 1000.0))
        if socket.poll(remaining_ms) == 0:
            break
        latest = oneshot.decode_packet(socket.recv_multipart())
        while True:
            try:
                latest = oneshot.decode_packet(socket.recv_multipart(flags=zmq.NOBLOCK))
            except zmq.Again:
                return latest
    return latest


def open_subscriber(endpoint: str) -> zmq.Socket:
    """Create the only communication endpoint: a localhost SUB socket."""
    socket = zmq.Context.instance().socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.SUBSCRIBE, oneshot.TOPIC)
    socket.connect(endpoint)
    return socket


def optional_command_values(packet: dict[str, Any]) -> tuple[np.ndarray, float]:
    q_command = packet.get("q_command")
    gripper_command = packet.get("gripper_command_pos")
    q_value = np.full(6, np.nan, dtype=np.float32)
    gripper_value = float("nan")
    if q_command is not None:
        q_value = oneshot._finite_array(q_command, (6,), "q_command")
    if gripper_command is not None:
        gripper_value = oneshot._finite_scalar(gripper_command, "gripper_command_pos")
    return q_value, gripper_value


@dataclass
class Cycle:
    inference_index: int
    observation_sequence: int
    observation_timestamp_ns: int
    q_actual: np.ndarray
    gripper_actual: float
    q_command: np.ndarray
    gripper_command: float
    prediction: np.ndarray
    inference_ms: float

    def csv_row(self) -> dict[str, Any]:
        arm = self.prediction[:, :6]
        adjacent_delta = np.abs(np.diff(arm, axis=0)).max(axis=0)
        max_deviation = np.abs(arm - self.q_actual[None, :]).max(axis=0)
        first_delta = np.abs(arm[0] - self.q_actual)
        row: dict[str, Any] = {
            "inference_index": self.inference_index,
            "observation_sequence": self.observation_sequence,
            "observation_timestamp_monotonic_ns": self.observation_timestamp_ns,
            "gripper_actual": self.gripper_actual,
            "gripper_command_pos": self.gripper_command,
            "inference_ms": self.inference_ms,
            "prediction_has_nan_or_inf": False,
            "first_delta_max_deg": float(np.rad2deg(first_delta.max())),
            "gripper_min": float(self.prediction[:, 6].min()),
            "gripper_max": float(self.prediction[:, 6].max()),
        }
        for index in range(6):
            label = index + 1
            row[f"q_actual_{label}"] = float(self.q_actual[index])
            row[f"q_command_{label}"] = float(self.q_command[index])
            row[f"first_action_joint_{label}"] = float(self.prediction[0, index])
            row[f"joint_{label}_min"] = float(arm[:, index].min())
            row[f"joint_{label}_max"] = float(arm[:, index].max())
            row[f"joint_{label}_max_adjacent_delta"] = float(adjacent_delta[index])
            row[f"joint_{label}_max_deviation_from_current_q_actual"] = float(max_deviation[index])
        row["first_action_gripper"] = float(self.prediction[0, 6])
        return row


@dataclass
class RunState:
    started_wall_time: str
    lerobot_version: str
    torch_version: str
    device: str
    inference_period_s: float
    checkpoint: str
    cycles: list[Cycle] = field(default_factory=list)
    status: str = "running"
    error: str | None = None


def csv_fieldnames() -> list[str]:
    fields = ["inference_index", "observation_sequence", "observation_timestamp_monotonic_ns"]
    fields += [f"q_actual_{index}" for index in range(1, 7)]
    fields += ["gripper_actual"]
    fields += [f"q_command_{index}" for index in range(1, 7)]
    fields += ["gripper_command_pos"]
    fields += [f"first_action_joint_{index}" for index in range(1, 7)]
    fields += ["first_action_gripper"]
    for index in range(1, 7):
        fields += [
            f"joint_{index}_min",
            f"joint_{index}_max",
            f"joint_{index}_max_adjacent_delta",
            f"joint_{index}_max_deviation_from_current_q_actual",
        ]
    return fields + [
        "gripper_min",
        "gripper_max",
        "prediction_has_nan_or_inf",
        "inference_ms",
        "first_delta_max_deg",
    ]


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_cycles_csv(path: Path, cycles: list[Cycle]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fieldnames())
        writer.writeheader()
        for cycle in cycles:
            writer.writerow(cycle.csv_row())
    temporary.replace(path)


def write_predictions_npz(path: Path, cycles: list[Cycle]) -> None:
    if cycles:
        prediction = np.stack([cycle.prediction for cycle in cycles], axis=0)
        q_actual = np.stack([cycle.q_actual for cycle in cycles], axis=0)
        q_command = np.stack([cycle.q_command for cycle in cycles], axis=0)
        gripper_actual = np.asarray([cycle.gripper_actual for cycle in cycles], dtype=np.float32)
        gripper_command = np.asarray([cycle.gripper_command for cycle in cycles], dtype=np.float32)
        inference_index = np.asarray([cycle.inference_index for cycle in cycles], dtype=np.int64)
        observation_sequence = np.asarray([cycle.observation_sequence for cycle in cycles], dtype=np.int64)
        observation_timestamp_ns = np.asarray([cycle.observation_timestamp_ns for cycle in cycles], dtype=np.int64)
    else:
        prediction = np.empty((0, 100, 7), dtype=np.float32)
        q_actual = np.empty((0, 6), dtype=np.float32)
        q_command = np.empty((0, 6), dtype=np.float32)
        gripper_actual = np.empty((0,), dtype=np.float32)
        gripper_command = np.empty((0,), dtype=np.float32)
        inference_index = observation_sequence = observation_timestamp_ns = np.empty((0,), dtype=np.int64)
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".npz", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(
            stream,
            prediction_action=prediction,
            q_actual=q_actual,
            gripper_actual_pos=gripper_actual,
            q_command=q_command,
            gripper_command_pos=gripper_command,
            inference_index=inference_index,
            observation_sequence=observation_sequence,
            observation_timestamp_monotonic_ns=observation_timestamp_ns,
        )
    temporary.replace(path)


def make_summary(state: RunState, stopped_wall_time: str) -> dict[str, Any]:
    latest = state.cycles[-1] if state.cycles else None
    return {
        "status": state.status,
        "error": state.error,
        "started_wall_time": state.started_wall_time,
        "stopped_wall_time": stopped_wall_time,
        "checkpoint": state.checkpoint,
        "endpoint": oneshot.ENDPOINT,
        "socket_type": "SUB only",
        "lerobot_version": state.lerobot_version,
        "torch_version": state.torch_version,
        "device": state.device,
        "inference_period_s": state.inference_period_s,
        "num_inferences": len(state.cycles),
        "prediction_shape_per_cycle": [100, 7],
        "latest_observation_sequence": None if latest is None else latest.observation_sequence,
        "latest_observation_timestamp_monotonic_ns": None if latest is None else latest.observation_timestamp_ns,
    }


def persist(output_dir: Path, state: RunState) -> None:
    write_cycles_csv(output_dir / "dynamic_shadow_cycles.csv", state.cycles)
    write_predictions_npz(output_dir / "predictions.npz", state.cycles)
    atomic_write_text(
        output_dir / "dynamic_shadow_summary.json",
        json.dumps(make_summary(state, datetime.now().isoformat(timespec="seconds")), indent=2) + "\n",
    )


def load_runtime() -> tuple[Any, Any, Any, Any, Any]:
    import torch
    import lerobot

    evaluation = oneshot.import_evaluation_module()
    if lerobot.__version__ != evaluation.EXPECTED_LEROBOT_VERSION:
        raise DynamicShadowError(f"LeRobot must be {evaluation.EXPECTED_LEROBOT_VERSION}, got {lerobot.__version__}")
    evaluation.validate_checkpoint(oneshot.CHECKPOINT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, preprocessor, postprocessor, runtime_config = evaluation.load_policy_and_processors(oneshot.CHECKPOINT, device)
    if (
        runtime_config.chunk_size != 100
        or runtime_config.n_action_steps != 100
        or runtime_config.n_obs_steps != 1
        or runtime_config.temporal_ensemble_coeff is not None
    ):
        raise DynamicShadowError("Checkpoint runtime ACT config is not the required 100/100/1/no-ensemble")
    policy.eval()
    return torch, lerobot, evaluation, (policy, preprocessor, postprocessor), device


def infer(packet: dict[str, Any], runtime: tuple[Any, Any, Any, Any, Any]) -> tuple[np.ndarray, float]:
    torch, _lerobot, evaluation, policy_bundle, _device = runtime
    policy, preprocessor, postprocessor = policy_bundle
    observation = oneshot.make_policy_observation(packet, torch)
    processed = preprocessor(observation)
    start_ns = time.perf_counter_ns()
    with torch.inference_mode():
        prediction_tensor = policy.predict_action_chunk(processed)
    inference_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    evaluation.require_finite_tensor("normalized dynamic Shadow prediction", prediction_tensor)
    prediction_tensor = postprocessor(prediction_tensor)
    evaluation.require_finite_tensor("postprocessed dynamic Shadow prediction", prediction_tensor)
    if tuple(prediction_tensor.shape) != (1, 100, 7):
        raise DynamicShadowError(f"Prediction shape must be (1,100,7), got {tuple(prediction_tensor.shape)}")
    prediction = prediction_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(prediction).all():
        raise DynamicShadowError("Postprocessed prediction contains NaN/Inf")
    return prediction, inference_ms


def run() -> Path:
    assert_no_forbidden_imports()
    args = parse_args()
    validate_args(args)
    output_dir = args.output_root.expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    runtime = load_runtime()
    torch, lerobot, _evaluation, _policy_bundle, device = runtime
    state = RunState(
        started_wall_time=datetime.now().isoformat(timespec="seconds"),
        lerobot_version=lerobot.__version__,
        torch_version=torch.__version__,
        device=str(device),
        inference_period_s=args.inference_period_s,
        checkpoint=str(oneshot.CHECKPOINT),
    )
    socket = open_subscriber(args.endpoint)
    next_due = time.monotonic()
    try:
        while True:
            sleep_s = next_due - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            # This call drains the SUB queue immediately before each inference.
            packet = receive_latest_from_socket(socket, args.packet_timeout_s)
            if packet is None:
                next_due = time.monotonic() + args.inference_period_s
                continue
            q_command, gripper_command = optional_command_values(packet)
            prediction, inference_ms = infer(packet, runtime)
            cycle = Cycle(
                inference_index=len(state.cycles),
                observation_sequence=packet["sequence"],
                observation_timestamp_ns=packet["monotonic_timestamp_ns"],
                q_actual=packet["q_actual"].astype(np.float32, copy=False),
                gripper_actual=float(packet["gripper_actual_pos"]),
                q_command=q_command,
                gripper_command=gripper_command,
                prediction=prediction,
                inference_ms=inference_ms,
            )
            state.cycles.append(cycle)
            persist(output_dir, state)
            print(
                f"[SHADOW {cycle.inference_index:03d}] seq={cycle.observation_sequence} | "
                f"infer={cycle.inference_ms:.0f}ms | first_delta_max={cycle.csv_row()['first_delta_max_deg']:.2f}deg | "
                f"grip={cycle.prediction[0, 6]:.1f} | finite=PASS",
                flush=True,
            )
            if args.max_inferences is not None and len(state.cycles) >= args.max_inferences:
                state.status = "completed"
                break
            next_due = time.monotonic() + args.inference_period_s
    except KeyboardInterrupt:
        state.status = "interrupted"
        print("[SHADOW] Ctrl+C received; saving complete cycles and exiting.", flush=True)
    except Exception as exc:
        state.status = "failed"
        state.error = f"{type(exc).__name__}: {exc}"
        print(f"[FAIL] {state.error}", file=sys.stderr, flush=True)
        raise
    finally:
        socket.close(linger=0)
        persist(output_dir, state)
    print(f"[OUTPUT] {output_dir}")
    return output_dir


def main() -> int:
    try:
        run()
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
