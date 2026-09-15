#!/usr/bin/env python3
"""Infinite fresh-observation ACT sender with official temporal ensembling."""

from __future__ import annotations

import argparse
import ast
import json
import math
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


TRAINING_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SHADOW_IPC_ROOT = TRAINING_PACKAGE_ROOT / "03_shadow" / "ipc"
REAL_ONCE_ROOT = TRAINING_PACKAGE_ROOT / "04_real_robot"
sys.path.insert(0, str(SHADOW_IPC_ROOT))
sys.path.insert(0, str(REAL_ONCE_ROOT))
import act_real_once_sender as once_sender  # noqa: E402
import act_shadow_subscriber as shadow  # noqa: E402


ACTION_ENDPOINT = once_sender.ACTION_ENDPOINT
ACTION_SCHEMA = "tb6_r5_act_stream_action_v1"
REAL_CHECKPOINT = "080000"
MOCK_CHECKPOINT = once_sender.MOCK_CHECKPOINT
TEMPORAL_ENSEMBLE_COEFF = 0.01
CHUNK_SIZE = 100
ACTION_DIM = 7
MODEL_HZ = 50.0
DEFAULT_SPEED_SCALE = 0.03
MAX_PLANNING_LEAD_S = 0.20
PLANNING_LEAD_STEP_FRACTION = 0.8
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zzw/projects/TB6_R5_ACT/training_results/pick_place_object_v1/act/v1/"
    "real_robot/full_sender"
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
        raise RuntimeError(f"ACT full sender source contains forbidden imports: {forbidden}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACT 080000 infinite full-deployment sender.")
    parser.add_argument("--observation-endpoint", default=shadow.ENDPOINT)
    parser.add_argument("--action-endpoint", default=ACTION_ENDPOINT)
    parser.add_argument("--observation-timeout-s", type=float, default=30.0)
    parser.add_argument("--action-send-timeout-ms", type=int, default=5000)
    parser.add_argument("--act-speed-scale", type=float, default=DEFAULT_SPEED_SCALE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--save-raw-predictions", action="store_true")
    parser.add_argument("--mock-safe-action", action="store_true")
    parser.add_argument(
        "--mock-steps",
        type=int,
        default=None,
        help="Dry-run-only finite stop; forbidden with checkpoint 080000.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    shadow.validate_endpoint(args.observation_endpoint)
    if args.action_endpoint != ACTION_ENDPOINT:
        raise ValueError(f"action endpoint must be localhost-only {ACTION_ENDPOINT}")
    if args.observation_timeout_s <= 0.0 or args.action_send_timeout_ms <= 0:
        raise ValueError("observation/action timeouts must be positive")
    if not 0.0 < args.act_speed_scale <= 1.0:
        raise ValueError("--act-speed-scale must be in (0,1]")
    if args.mock_safe_action:
        if args.mock_steps is None or args.mock_steps <= 0:
            raise ValueError("mock mode requires a positive --mock-steps")
    elif args.mock_steps is not None:
        raise ValueError("--mock-steps is forbidden in real mode")


def cumulative_host_ticks(model_steps: int, speed_scale: float) -> int:
    if model_steps < 0 or not 0.0 < speed_scale <= 1.0:
        raise ValueError("invalid model_steps/speed_scale")
    return math.ceil(model_steps / speed_scale - 1e-12)


def model_step_duration_s(step_index: int, speed_scale: float) -> float:
    """Return the real host duration assigned to one model step."""
    if step_index < 0:
        raise ValueError("step_index must be non-negative")
    return host_ticks_for_model_step(step_index, speed_scale) / MODEL_HZ


def host_ticks_for_model_step(step_index: int, speed_scale: float) -> int:
    if step_index < 0:
        raise ValueError("step_index must be non-negative")
    return cumulative_host_ticks(step_index + 1, speed_scale) - cumulative_host_ticks(
        step_index, speed_scale
    )


def planning_lead_s_for_model_step(step_index: int, speed_scale: float) -> float:
    """Leave a per-step execution margin without imposing a speed-scale ceiling."""
    return min(
        MAX_PLANNING_LEAD_S,
        PLANNING_LEAD_STEP_FRACTION * model_step_duration_s(step_index, speed_scale),
    )


class LatestObservationSubscriber:
    def __init__(self, endpoint: str) -> None:
        shadow.validate_endpoint(endpoint)
        self._socket = zmq.Context.instance().socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.SUBSCRIBE, shadow.TOPIC)
        self._socket.connect(endpoint)

    def receive_fresh(
        self, timeout_s: float, previous_sequence: int | None, cutoff_ns: int
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000.0))
            if self._socket.poll(remaining_ms, zmq.POLLIN) == 0:
                break
            latest = shadow.decode_packet(self._socket.recv_multipart())
            while True:
                try:
                    latest = shadow.decode_packet(self._socket.recv_multipart(flags=zmq.NOBLOCK))
                except zmq.Again:
                    break
            sequence_is_newer = previous_sequence is None or latest["sequence"] > previous_sequence
            if sequence_is_newer and latest["monotonic_timestamp_ns"] >= cutoff_ns:
                return latest
        raise shadow.ShadowProtocolError("Timed out waiting for a fresh full-deployment observation")

    def close(self) -> None:
        self._socket.close(linger=0)


class ActionPusher:
    def __init__(self, endpoint: str, timeout_ms: int) -> None:
        if endpoint != ACTION_ENDPOINT:
            raise ValueError(f"action endpoint must be localhost-only {ACTION_ENDPOINT}")
        self._socket = zmq.Context.instance().socket(zmq.PUSH)
        self._socket.setsockopt(zmq.LINGER, 1000)
        self._socket.setsockopt(zmq.SNDHWM, 2)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.connect(endpoint)

    def send(self, packet: dict[str, Any]) -> None:
        self._socket.send(json.dumps(packet, separators=(",", ":")).encode("utf-8"))

    def close(self) -> None:
        self._socket.close(linger=1000)


class RealActPredictor:
    checkpoint = REAL_CHECKPOINT

    def __init__(self) -> None:
        import lerobot
        import torch
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

        evaluation = shadow.import_evaluation_module()
        if lerobot.__version__ != "0.6.1":
            raise RuntimeError(f"LeRobot must be 0.6.1, got {lerobot.__version__}")
        evaluation.validate_checkpoint(shadow.CHECKPOINT)
        self._torch = torch
        self._evaluation = evaluation
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._policy, self._preprocessor, self._postprocessor, runtime = (
            evaluation.load_policy_and_processors(shadow.CHECKPOINT, self._device)
        )
        if (
            runtime.chunk_size != CHUNK_SIZE
            or runtime.n_action_steps != CHUNK_SIZE
            or runtime.n_obs_steps != 1
            or runtime.temporal_ensemble_coeff is not None
        ):
            raise RuntimeError("Checkpoint runtime ACT config must remain 100/100/1/no-ensemble")
        self._policy.eval()
        self._ensembler = ACTTemporalEnsembler(TEMPORAL_ENSEMBLE_COEFF, CHUNK_SIZE)
        self._prediction_count = 0
        self.versions = {
            "lerobot_version": lerobot.__version__,
            "torch_version": torch.__version__,
            "device": str(self._device),
            "checkpoint_path": str(shadow.CHECKPOINT),
        }

    def warmup(self, packet: dict[str, Any]) -> float:
        """Run one discarded forward pass without touching temporal-ensemble state."""
        observation = shadow.make_policy_observation(packet, self._torch)
        processed = self._preprocessor(observation)
        started_ns = time.perf_counter_ns()
        with self._torch.inference_mode():
            normalized_chunk = self._policy.predict_action_chunk(processed)
        warmup_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        self._evaluation.require_finite_tensor("normalized ACT full warmup prediction", normalized_chunk)
        if tuple(normalized_chunk.shape) != (1, CHUNK_SIZE, ACTION_DIM):
            raise RuntimeError(
                f"normalized warmup prediction must be (1,{CHUNK_SIZE},{ACTION_DIM}), "
                f"got {tuple(normalized_chunk.shape)}"
            )
        return warmup_ms

    def predict(self, packet: dict[str, Any], step_index: int) -> tuple[np.ndarray, np.ndarray, float]:
        observation = shadow.make_policy_observation(packet, self._torch)
        processed = self._preprocessor(observation)
        started_ns = time.perf_counter_ns()
        with self._torch.inference_mode():
            normalized_chunk = self._policy.predict_action_chunk(processed)
        inference_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        self._evaluation.require_finite_tensor("normalized ACT full prediction", normalized_chunk)
        if tuple(normalized_chunk.shape) != (1, CHUNK_SIZE, ACTION_DIM):
            raise RuntimeError(
                f"normalized prediction must be (1,{CHUNK_SIZE},{ACTION_DIM}), "
                f"got {tuple(normalized_chunk.shape)}"
            )
        normalized_selected = self._ensembler.update(normalized_chunk)
        self._evaluation.require_finite_tensor("normalized ACT ensembled action", normalized_selected)
        if tuple(normalized_selected.shape) != (1, ACTION_DIM):
            raise RuntimeError(f"ensembled action must be (1,{ACTION_DIM})")
        if self._prediction_count == 0 and not self._torch.equal(
            normalized_selected, normalized_chunk[:, 0]
        ):
            raise RuntimeError("first temporal ensemble output differs from prediction action[0]")
        physical_selected = self._postprocessor(normalized_selected)
        self._evaluation.require_finite_tensor("postprocessed ACT selected action", physical_selected)
        if tuple(physical_selected.shape) != (1, ACTION_DIM):
            raise RuntimeError(f"postprocessed selected action must be (1,{ACTION_DIM})")
        raw = normalized_chunk.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        selected = physical_selected.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        if raw.shape != (CHUNK_SIZE, ACTION_DIM) or selected.shape != (ACTION_DIM,):
            raise RuntimeError("ACT full prediction shapes are invalid")
        if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(selected)):
            raise RuntimeError("ACT full prediction contains NaN/Inf")
        self._prediction_count += 1
        return raw, selected, inference_ms


class MockPredictor:
    checkpoint = MOCK_CHECKPOINT

    def __init__(self) -> None:
        import torch
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

        self._torch = torch
        self._ensembler = ACTTemporalEnsembler(TEMPORAL_ENSEMBLE_COEFF, CHUNK_SIZE)
        self._prediction_count = 0
        self.versions = {
            "lerobot_version": "0.6.1_mock_identity",
            "torch_version": torch.__version__,
            "device": "cpu_mock_mode",
            "checkpoint_path": "none_mock_mode",
        }

    def warmup(self, packet: dict[str, Any]) -> float:
        """Dry-run counterpart: validate a discarded chunk, without ensemble update."""
        chunk, _, _, _ = once_sender.make_mock_action(packet)
        if chunk.shape != (CHUNK_SIZE, ACTION_DIM) or not np.all(np.isfinite(chunk)):
            raise RuntimeError("mock ACT full warmup prediction is invalid")
        return 0.0

    def predict(self, packet: dict[str, Any], step_index: int) -> tuple[np.ndarray, np.ndarray, float]:
        chunk, _, _, _ = once_sender.make_mock_action(packet)
        mock_gripper_target = (100.0, 0.0, 100.0)[step_index % 3]
        chunk[:, 6] = mock_gripper_target
        tensor = self._torch.as_tensor(chunk, dtype=self._torch.float32).unsqueeze(0)
        selected_tensor = self._ensembler.update(tensor)
        if self._prediction_count == 0 and not self._torch.equal(selected_tensor, tensor[:, 0]):
            raise RuntimeError("mock first temporal ensemble output mismatch")
        raw = tensor.squeeze(0).numpy().astype(np.float32, copy=False)
        selected = selected_tensor.squeeze(0).numpy().astype(np.float32, copy=False)
        if raw.shape != (CHUNK_SIZE, ACTION_DIM) or selected.shape != (ACTION_DIM,):
            raise RuntimeError("mock ACT full prediction shapes are invalid")
        if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(selected)):
            raise RuntimeError("mock ACT full prediction contains NaN/Inf")
        self._prediction_count += 1
        return raw, selected, 0.0


def run() -> Path:
    assert_no_forbidden_imports()
    args = parse_args()
    validate_args(args)
    output_dir = args.output_root.expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = output_dir / "raw_predictions"
    if args.save_raw_predictions:
        raw_dir.mkdir()
    predictor = MockPredictor() if args.mock_safe_action else RealActPredictor()
    subscriber = LatestObservationSubscriber(args.observation_endpoint)
    pusher = ActionPusher(args.action_endpoint, args.action_send_timeout_ms)
    print(
        f"[ACT-FULL-SENDER] READY | checkpoint={predictor.checkpoint} | "
        f"speed_scale={args.act_speed_scale} | temporal_ensemble={TEMPORAL_ENSEMBLE_COEFF}",
        flush=True,
    )
    previous_sequence: int | None = None
    schedule_origin_s: float | None = None
    sent_steps = 0
    status = "running"
    warmup_ms: float | None = None
    try:
        with (output_dir / "selected_actions.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
            # Consume one frame solely to pay the first-forward cost. It is never
            # ensembled or sent, so formal step 0 must use the following fresh frame.
            warmup_cutoff_ns = time.monotonic_ns()
            warmup_observation = subscriber.receive_fresh(
                args.observation_timeout_s, previous_sequence, warmup_cutoff_ns
            )
            warmup_ms = predictor.warmup(warmup_observation)
            previous_sequence = int(warmup_observation["sequence"])
            print(
                f"[ACT-FULL-SENDER] WARMUP | seq={previous_sequence} "
                f"warmup={warmup_ms:.1f}ms | result_discarded=true",
                flush=True,
            )
            while args.mock_steps is None or sent_steps < args.mock_steps:
                step_duration_s = model_step_duration_s(sent_steps, args.act_speed_scale)
                planning_lead_s = planning_lead_s_for_model_step(
                    sent_steps, args.act_speed_scale
                )
                if schedule_origin_s is None:
                    planned_execution_boundary_s = None
                    planning_started_s = time.monotonic()
                else:
                    planned_execution_boundary_s = schedule_origin_s + (
                        cumulative_host_ticks(sent_steps, args.act_speed_scale) / MODEL_HZ
                    )
                    planning_deadline_s = planned_execution_boundary_s - planning_lead_s
                    remaining = planning_deadline_s - time.monotonic()
                    if remaining > 0.0:
                        time.sleep(remaining)
                    planning_started_s = time.monotonic()
                cutoff_ns = time.monotonic_ns()
                observation = subscriber.receive_fresh(
                    args.observation_timeout_s, previous_sequence, cutoff_ns
                )
                inference_started_s = time.monotonic()
                observation_age_ms = (
                    time.monotonic_ns() - int(observation["monotonic_timestamp_ns"])
                ) / 1_000_000.0
                raw, selected, inference_ms = predictor.predict(observation, sent_steps)
                packet = {
                    "schema": ACTION_SCHEMA,
                    "checkpoint": predictor.checkpoint,
                    "step_index": sent_steps,
                    "source_observation_sequence": int(observation["sequence"]),
                    "source_observation_timestamp_ns": int(observation["monotonic_timestamp_ns"]),
                    "observation_age_ms_at_inference": float(observation_age_ms),
                    "inference_ms": float(inference_ms),
                        "temporal_ensemble_coeff": TEMPORAL_ENSEMBLE_COEFF,
                        "speed_scale": args.act_speed_scale,
                        "model_step_duration_s": float(step_duration_s),
                        "planning_lead_s": float(planning_lead_s),
                        "action_shape": [ACTION_DIM],
                    "action": selected.tolist(),
                }
                pusher.send(packet)
                action_sent_s = time.monotonic()
                if schedule_origin_s is None:
                    # Product begins executing only after receiving step 0. Sending
                    # completion is therefore the best sender-side execution origin.
                    schedule_origin_s = action_sent_s
                    planned_execution_boundary_s = schedule_origin_s
                ready_lead_ms = (planned_execution_boundary_s - action_sent_s) * 1000.0
                packet.update(
                    {
                        "planned_execution_boundary_s": float(planned_execution_boundary_s),
                        "planning_started_s": float(planning_started_s),
                        "action_sent_s": float(action_sent_s),
                        "ready_lead_ms": float(ready_lead_ms),
                    }
                )
                stream.write(json.dumps(packet) + "\n")
                if args.save_raw_predictions:
                    np.savez_compressed(
                        raw_dir / f"step_{sent_steps:08d}.npz",
                        normalized_prediction=raw,
                        selected_action=selected,
                    )
                previous_sequence = int(observation["sequence"])
                sent_steps += 1
                print(
                    f"[ACT-FULL-SENDER] step={sent_steps - 1} seq={previous_sequence} "
                    f"obs_age={observation_age_ms:.1f}ms infer={inference_ms:.1f}ms "
                    f"step_duration={step_duration_s:.3f}s "
                    f"planning_lead={planning_lead_s:.3f}s ready_lead={ready_lead_ms:.1f}ms",
                    flush=True,
                )
            status = "mock_complete"
    except KeyboardInterrupt:
        status = "operator_ctrl_c"
        print("[ACT-FULL-SENDER] STOP | operator Ctrl+C", flush=True)
    finally:
        pusher.close()
        subscriber.close()
        summary = {
            **predictor.versions,
            "status": status,
            "checkpoint": predictor.checkpoint,
            "sent_steps": sent_steps,
            "speed_scale": args.act_speed_scale,
            "temporal_ensemble_coeff": TEMPORAL_ENSEMBLE_COEFF,
            "chunk_size": CHUNK_SIZE,
            "action_shape": [ACTION_DIM],
            "real_mode_has_no_step_limit": not args.mock_safe_action,
            "max_planning_lead_s": MAX_PLANNING_LEAD_S,
            "planning_lead_step_fraction": PLANNING_LEAD_STEP_FRACTION,
            "warmup_ms": warmup_ms,
        }
        (output_dir / "sender_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print(f"[ACT-FULL-SENDER] EXIT | steps={sent_steps} | output={output_dir}", flush=True)
    return output_dir


def main() -> int:
    try:
        run()
    except Exception as exc:
        print(f"[ACT-FULL-SENDER] FAIL | {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
