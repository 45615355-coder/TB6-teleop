#!/usr/bin/env python3
"""Minimal J6 JogAnyJ3 velocity-drop diagnostic; no ACT or peripherals."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import config
from act_control_once import ActAbort, initialize_verified_transport
from jogany_io import build_active_payload
from topic_io import TopicIO


PHASES = (
    ("phase1", 15, 0.02, 0.08),
    ("phase2", 25, 0.001, 0.004),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TB6-R5 J6 JogAnyJ3 velocity-drop probe.")
    parser.add_argument("--ip", default=config.ROBOT_IP)
    parser.add_argument("--connect-timeout-ms", type=int, default=1000)
    parser.add_argument("--execute-probe", action="store_true")
    parser.add_argument("--confirm-robot-ready", action="store_true")
    parser.add_argument("--confirm-emergency-stop-ready", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.connect_timeout_ms <= 0:
        raise ValueError("--connect-timeout-ms must be positive")
    if not args.execute_probe:
        raise ValueError("real motion requires --execute-probe")
    missing = [
        name
        for name in ("confirm_robot_ready", "confirm_emergency_stop_ready")
        if not getattr(args, name)
    ]
    if missing:
        raise ValueError("missing real-motion confirmations: " + ", ".join(missing))


def topic_record(timestamp_ns: int, phase: str, q_command: np.ndarray, topic: Any) -> dict[str, Any]:
    return {
        "timestamp_ns": int(timestamp_ns),
        "phase": phase,
        "q_command": q_command.tolist(),
        "q_target": np.asarray(topic.q_target_rad, dtype=np.float64).tolist(),
        "q_actual": np.asarray(topic.q_actual_rad, dtype=np.float64).tolist(),
        "dq_actual": np.asarray(topic.dq_actual_rad_s, dtype=np.float64).tolist(),
    }


def run(args: argparse.Namespace) -> Path:
    from teleop import _wait_start_topic, robot_side_stop_observed, send_shutdown_release, topic_safe

    reader = TopicIO()
    transport = None
    released = False
    robot_side_stop = False
    log_path = config.LOG_DIRECTORY / f"jogany_act_velocity_drop_probe_v1_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    try:
        reader.start(args.ip)
        initial_topic = _wait_start_topic(reader)
        q_previous = np.asarray(initial_topic.q_actual_rad, dtype=np.float64).copy()
        transport = initialize_verified_transport(q_previous, args.ip, args.connect_timeout_ms)
        config.LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
        with log_path.open("x", encoding="utf-8") as stream:
            deadline = time.perf_counter() + config.DT
            for phase, cycles, velocity_j6, expected_acc_dec_j6 in PHASES:
                for _ in range(cycles):
                    remaining = deadline - time.perf_counter()
                    if remaining > 0.0:
                        time.sleep(remaining)
                    topic = reader.latest()
                    safety = topic_safe(topic)
                    if safety:
                        raise ActAbort(
                            f"Topic safety failed during {phase}: {safety}",
                            robot_side_stop=robot_side_stop_observed(topic),
                        )
                    if transport.fatal_reason:
                        raise ActAbort(f"transport fatal during {phase}: {transport.fatal_reason}")
                    q_command = q_previous.copy()
                    q_command[5] = q_previous[5] + velocity_j6 * config.DT
                    payload = build_active_payload(q_command, q_previous)
                    expected = np.zeros(6, dtype=np.float64)
                    expected[5] = expected_acc_dec_j6
                    if not (
			np.allclose(payload.joint_acc, expected, atol=1e-12, rtol=0.0)
			and np.allclose(payload.joint_dec, expected, atol=1e-12, rtol=0.0)
                    ):
                        raise RuntimeError(f"unexpected JogAnyJ3 acc/dec in {phase}")
                    stream.write(json.dumps(topic_record(time.monotonic_ns(), phase, q_command, topic)) + "\n")
                    stream.flush()
                    result = transport.submit_main(payload.text, payload.joint_pos, payload.joint_vel)
                    if not result.submitted or transport.fatal_reason:
                        raise ActAbort(
                            f"submission failed during {phase}: {transport.fatal_reason or result.callback_state}"
                        )
                    q_previous = q_command
                    deadline += config.DT
                    if time.perf_counter() >= deadline:
                        deadline = time.perf_counter() + config.DT
        release_count = send_shutdown_release(transport, reader)
        released = release_count > 0
        if not released:
            raise ActAbort("verified release/deceleration submitted zero frames")
        return log_path
    except Exception as exc:
        robot_side_stop = isinstance(exc, ActAbort) and exc.robot_side_stop
        raise
    finally:
        if transport is not None and not released and not robot_side_stop and not transport.fatal_reason:
            send_shutdown_release(transport, reader)
        if transport is not None:
            transport.terminate_once()
        reader.close()


def main() -> int:
    try:
        args = parse_args()
        validate_args(args)
        log_path = run(args)
    except Exception as exc:
        print(f"[JOGANY-ACT-VELOCITY-DROP-PROBE] ABORT | {type(exc).__name__}: {exc}", flush=True)
        return 1
    print(f"[JOGANY-ACT-VELOCITY-DROP-PROBE] COMPLETE | log={log_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
