#!/usr/bin/env python3
"""Product v5 one-chunk ACT arm executor.

Real mode owns Product hardware and executes at most one validated arm chunk.
Dry-run mode owns only a localhost PULL socket and performs no hardware import,
connection, or initialization.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import config
from act_action_receiver import ACTION_ENDPOINT, ActionChunkReceiver
from jogany_io import build_active_payload


DEFAULT_SPEED_SCALE = 0.2
ACTION_STEPS = 100
ARM_DIM = 6
JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))


class ActAbort(RuntimeError):
    def __init__(self, reason: str, *, robot_side_stop: bool = False) -> None:
        super().__init__(reason)
        self.robot_side_stop = bool(robot_side_stop)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Product v5 ACT real-arm one-chunk executor.")
    parser.add_argument("--act-speed-scale", type=float, default=DEFAULT_SPEED_SCALE)
    parser.add_argument("--action-endpoint", default=ACTION_ENDPOINT)
    parser.add_argument("--action-timeout-s", type=float, default=60.0)
    parser.add_argument("--connect-timeout-ms", type=int, default=1000)
    parser.add_argument("--ip", default=config.ROBOT_IP)
    parser.add_argument("--execute-act-motion", action="store_true")
    parser.add_argument("--confirm-arm-home", action="store_true")
    parser.add_argument("--confirm-gripper-open", action="store_true")
    parser.add_argument("--confirm-web-stop-ready", action="store_true")
    parser.add_argument("--confirm-emergency-stop-ready", action="store_true")
    parser.add_argument("--dry-run-action-ipc", action="store_true")
    parser.add_argument("--dry-run-current-q", type=float, nargs=6,
                        default=(0.0, 0.01, 0.02, 0.03, 0.04, 0.05))
    parser.add_argument("--dry-run-output", type=Path,
                        default=Path("/tmp/tb6_act_control_once_dry_run.json"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.act_speed_scale <= 1.0:
        raise ValueError("--act-speed-scale must be in (0, 1]")
    if args.action_timeout_s <= 0.0:
        raise ValueError("--action-timeout-s must be positive")
    if args.dry_run_action_ipc:
        if args.execute_act_motion:
            raise ValueError("dry-run and real motion are mutually exclusive")
        return
    if not args.execute_act_motion:
        raise ValueError("real mode requires --execute-act-motion")
    missing = [
        name for name in (
            "confirm_arm_home",
            "confirm_gripper_open",
            "confirm_web_stop_ready",
            "confirm_emergency_stop_ready",
        ) if not getattr(args, name)
    ]
    if missing:
        raise ValueError("missing real-motion confirmations: " + ", ".join(missing))


def load_existing_joint_limits() -> tuple[np.ndarray, np.ndarray]:
    """Read the six established TB6 limits from Product v5's existing URDF."""
    root = ET.parse(config.URDF_PATH).getroot()
    by_name = {joint.get("name"): joint for joint in root.findall("joint")}
    lower, upper = [], []
    for name in JOINT_NAMES:
        joint = by_name.get(name)
        limit = None if joint is None else joint.find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            raise RuntimeError(f"Existing URDF has no finite position limits for {name}")
        lower.append(float(limit.get("lower")))
        upper.append(float(limit.get("upper")))
    lower_array, upper_array = np.asarray(lower), np.asarray(upper)
    if not np.all(np.isfinite(lower_array)) or not np.all(np.isfinite(upper_array)):
        raise RuntimeError("Existing URDF joint limits contain NaN/Inf")
    return lower_array, upper_array


def validate_joint_positions(q_values: Any) -> None:
    q = np.asarray(q_values, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != ARM_DIM or not np.all(np.isfinite(q)):
        raise ActAbort(f"arm trajectory must be finite [N,{ARM_DIM}], got {q.shape}")
    lower, upper = load_existing_joint_limits()
    if np.any(q < lower[None, :]) or np.any(q > upper[None, :]):
        indices = np.argwhere((q < lower[None, :]) | (q > upper[None, :]))
        first = indices[0]
        raise ActAbort(
            f"joint position violates existing URDF limit at path_index={int(first[0])}, "
            f"joint={int(first[1]) + 1}, value={q[tuple(first)]:.10f}, "
            f"range=[{lower[int(first[1])]:.10f},{upper[int(first[1])]:.10f}]"
        )


def resample_arm_chunk(command_anchor: Any, action: Any, speed_scale: float) -> np.ndarray:
    current = np.asarray(command_anchor, dtype=np.float64)
    chunk = np.asarray(action, dtype=np.float64)
    if current.shape != (ARM_DIM,) or not np.all(np.isfinite(current)):
        raise ActAbort("command anchor must contain six finite values")
    if chunk.shape != (ACTION_STEPS, 7) or not np.all(np.isfinite(chunk)):
        raise ActAbort(f"ACT action must be finite [{ACTION_STEPS},7]")
    if not 0.0 < speed_scale <= 1.0:
        raise ActAbort("speed_scale must be in (0,1]")
    nodes = np.vstack((current[None, :], chunk[:, :ARM_DIM]))
    command_count = math.ceil(ACTION_STEPS / speed_scale - 1e-12)
    commands = np.empty((command_count, ARM_DIM), dtype=np.float64)
    for step in range(1, command_count + 1):
        phase = min(float(ACTION_STEPS), step * speed_scale)
        left = int(math.floor(phase))
        if left >= ACTION_STEPS:
            commands[step - 1] = nodes[ACTION_STEPS]
        else:
            alpha = phase - left
            commands[step - 1] = (1.0 - alpha) * nodes[left] + alpha * nodes[left + 1]
    if not np.all(np.isfinite(commands)) or not np.array_equal(commands[-1], chunk[-1, :ARM_DIM]):
        raise ActAbort("time-resampled trajectory is invalid or does not reach action[99]")
    return commands


def prevalidate_payloads(commands: np.ndarray, previous_command: Any) -> list[Any]:
    """Use the existing 50 Hz Product payload validator for every command."""
    previous = np.asarray(previous_command, dtype=np.float64)
    validate_joint_positions(np.vstack((previous[None, :], commands)))
    payloads = []
    for index, command in enumerate(commands):
        try:
            payload = build_active_payload(command, previous)
        except Exception as exc:
            raise ActAbort(f"existing 50Hz payload validator rejected command {index}: {exc}") from exc
        payloads.append(payload)
        previous = command
    return payloads


def dry_run(args: argparse.Namespace) -> int:
    receiver = ActionChunkReceiver(args.action_endpoint)
    try:
        print("[ACT-REAL/DRY-RUN] WAITING FOR ONE MOCK ACTION CHUNK", flush=True)
        packet = receiver.receive(int(args.action_timeout_s * 1000.0), allow_mock=True)
        if packet is None:
            raise ActAbort("timed out waiting for fake action packet")
    finally:
        receiver.close()
    command_anchor = np.asarray(args.dry_run_current_q, dtype=np.float64)
    commands = resample_arm_chunk(command_anchor, packet["action"], args.act_speed_scale)
    payloads = prevalidate_payloads(commands, command_anchor)
    full_path = np.vstack((command_anchor[None, :], commands))
    first_command_velocity = (commands[0] - command_anchor) / config.DT
    report = {
        "mode": "NO_HARDWARE_DRY_RUN",
        "source_observation_sequence": packet["source_observation_sequence"],
        "speed_scale": args.act_speed_scale,
        "control_hz": config.CONTROL_HZ,
        "model_action_steps": ACTION_STEPS,
        "execution_command_count": len(payloads),
        "full_path_point_count_including_current_anchor": len(full_path),
        "estimated_execution_duration_s": len(payloads) * config.DT,
        "anchor_semantics": "previous accepted q_command",
        "command_anchor": command_anchor.tolist(),
        "act_first_action": packet["action"][0].tolist(),
        "first_execution_command": commands[0].tolist(),
        "first_command_velocity_rad_s": first_command_velocity.tolist(),
        "last_execution_command": commands[-1].tolist(),
        "action_last_arm": packet["action"][-1, :6].tolist(),
        "finite": bool(np.all(np.isfinite(full_path))),
        "max_50hz_adjacent_delta_rad": np.abs(np.diff(full_path, axis=0)).max(axis=0).tolist(),
        "last_reaches_action_99": bool(np.array_equal(commands[-1], packet["action"][-1, :6])),
    }
    args.dry_run_output.parent.mkdir(parents=True, exist_ok=True)
    args.dry_run_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print("[ACT-REAL/DRY-RUN] PASS", flush=True)
    return 0


def initialize_verified_transport(q_initial: np.ndarray, ip: str, timeout_ms: int) -> Any:
    from jogany_io import create_transport, strict_hold

    transport = create_transport(ip, timeout_ms)
    warm = transport.submit_warmup_hold(strict_hold(q_initial).text, q_initial)
    if not warm.submitted:
        raise ActAbort("seq1 warmup strict hold submission failed")
    deadline = time.monotonic() + 2.5
    while transport.callback_state(1) == "PENDING" and time.monotonic() < deadline:
        time.sleep(0.002)
    if transport.callback_state(1) == "PENDING":
        transport.quarantine_warmup_timeout()
    verify = transport.submit_verification_hold(strict_hold(q_initial).text, q_initial)
    if not verify.submitted or not transport.wait_for_verification(2.0):
        raise ActAbort(transport.fatal_reason or "seq2 verification strict hold failed")
    return transport


def execute_payloads(payloads: list[Any], transport: Any, reader: Any) -> int:
    from teleop import robot_side_stop_observed, topic_safe

    deadline = time.perf_counter() + config.DT
    submitted = 0
    for index, payload in enumerate(payloads):
        remaining = deadline - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)
        topic = reader.latest()
        safety = topic_safe(topic)
        if safety:
            raise ActAbort(
                f"Topic safety failed before execution command {index}: {safety}",
                robot_side_stop=robot_side_stop_observed(topic),
            )
        if transport.fatal_reason:
            raise ActAbort(f"transport fatal before command {index}: {transport.fatal_reason}")
        try:
            result = transport.submit_main(payload.text, payload.joint_pos, payload.joint_vel)
        except Exception as exc:
            raise ActAbort(f"command {index} submission exception: {exc}") from exc
        if not result.submitted or transport.fatal_reason:
            raise ActAbort(f"command {index} submission failed: {transport.fatal_reason or result.callback_state}")
        submitted += 1
        deadline += config.DT
        if time.perf_counter() >= deadline:
            deadline = time.perf_counter() + config.DT
    return submitted


def real_run(args: argparse.Namespace) -> int:
    from gripper_io import GripperIO
    from jogany_io import build_release_payload
    from shadow_observation_publisher import ShadowObservationPublisher, ShadowPublishError
    from teleop import (
        _start_recording_peripherals,
        _wait_start_topic,
        home_max_delta_deg,
        robot_side_stop_observed,
        send_shutdown_release,
        topic_safe,
    )
    from topic_io import TopicIO

    reader = TopicIO()
    receiver = ActionChunkReceiver(args.action_endpoint)
    transport = gripper = recorder = publisher = None
    recording_stop = capture_thread = preview_thread = None
    released = False
    robot_side_stop = False
    log_record: dict[str, Any] = {"status": "starting", "speed_scale": args.act_speed_scale}
    try:
        reader.start(args.ip)
        initial_topic = _wait_start_topic(reader)
        q_initial = np.asarray(initial_topic.q_actual_rad, dtype=np.float64)
        home_error_deg = home_max_delta_deg(q_initial)
        if home_error_deg > config.HOME_MAX_DELTA_DEG:
            raise ActAbort(
                f"robot is too far from training HOME: {home_error_deg:.3f}deg > existing "
                f"HOME_MAX_DELTA_DEG={config.HOME_MAX_DELTA_DEG:.3f}deg"
            )
        transport = initialize_verified_transport(q_initial, args.ip, args.connect_timeout_ms)
        gripper = GripperIO(
            vel=config.GRIPPER_VEL,
            force=config.GRIPPER_FORCE,
            acc=config.GRIPPER_ACC,
            dec=config.GRIPPER_DEC,
        )
        gripper.start()
        print("[ACT-REAL] OmniPicker is Product-owned; ACT gripper actions will be ignored.", flush=True)
        recorder, recording_stop, capture_thread, preview_thread = _start_recording_peripherals()
        publisher = ShadowObservationPublisher(
            camera_serial_dict=config.CAM_SERIAL_DICT,
            endpoint="tcp://127.0.0.1:5557",
            publish_hz=10.0,
        )
        print("[ACT-REAL] WAITING FOR ONE ACTION CHUNK", flush=True)
        started_wait = time.monotonic()
        deadline = time.perf_counter() + config.DT
        packet = None
        latest_topic = initial_topic
        while packet is None:
            if time.monotonic() - started_wait > args.action_timeout_s:
                raise ActAbort("timed out waiting for one ACT action chunk")
            remaining = deadline - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            latest_topic = reader.latest()
            safety = topic_safe(latest_topic)
            if safety:
                raise ActAbort(
                    f"Topic safety failed while waiting: {safety}",
                    robot_side_stop=robot_side_stop_observed(latest_topic),
                )
            packet = receiver.receive(0, allow_mock=False)
            if packet is not None:
                break
            try:
                publisher.publish_if_due(
                    camera=recorder.camera,
                    topic=latest_topic,
                    gripper_snapshot=gripper.get_snapshot(),
                    q_command=transport.last_accepted_command_q,
                )
            except ShadowPublishError:
                pass
            hold_payload = build_release_payload(transport.last_accepted_command_q)
            result = transport.submit_main(hold_payload.text, hold_payload.joint_pos, hold_payload.joint_vel)
            if not result.submitted or transport.fatal_reason:
                raise ActAbort(f"waiting hold submission failed: {transport.fatal_reason or result.callback_state}")
            deadline += config.DT
            if time.perf_counter() >= deadline:
                deadline = time.perf_counter() + config.DT

        receiver.close()
        receiver = None
        publisher.close()
        publisher = None
        latest_topic = reader.latest()
        safety = topic_safe(latest_topic, stationary=True)
        if safety:
            raise ActAbort(
                f"Topic safety failed before trajectory validation: {safety}",
                robot_side_stop=robot_side_stop_observed(latest_topic),
            )
        current_q_actual = np.asarray(latest_topic.q_actual_rad, dtype=np.float64)
        command_anchor = np.asarray(transport.last_accepted_command_q, dtype=np.float64)
        anchor_actual_gap = np.abs(command_anchor - current_q_actual)
        commands = resample_arm_chunk(command_anchor, packet["action"], args.act_speed_scale)
        first_command_velocity = (commands[0] - command_anchor) / config.DT
        log_record.update({
            "status": "trajectory_generated",
            "source_observation_sequence": packet["source_observation_sequence"],
            "source_observation_timestamp_ns": packet["source_observation_timestamp_ns"],
            "inference_ms": packet["inference_ms"],
            "current_q_actual": current_q_actual.tolist(),
            "command_anchor": command_anchor.tolist(),
            "command_anchor_q_actual_max_abs_rad": float(anchor_actual_gap.max()),
            "command_anchor_q_actual_max_abs_deg": float(np.rad2deg(anchor_actual_gap.max())),
            "prediction_first_action": packet["action"][0].tolist(),
            "first_resampled_command": commands[0].tolist(),
            "first_command_velocity_rad_s": first_command_velocity.tolist(),
            "prediction_gripper_min": float(packet["action"][:, 6].min()),
            "prediction_gripper_max": float(packet["action"][:, 6].max()),
            "execution_command_count": len(commands),
            "estimated_execution_duration_s": len(commands) * config.DT,
        })
        print(
            f"[ACT-REAL] command-anchor/q-actual gap="
            f"{log_record['command_anchor_q_actual_max_abs_deg']:.6f}deg",
            flush=True,
        )
        payloads = prevalidate_payloads(commands, command_anchor)
        log_record["status"] = "validated"
        submitted = execute_payloads(payloads, transport, reader)
        release_count = send_shutdown_release(transport, reader)
        released = release_count > 0
        if not released:
            raise ActAbort("verified release/deceleration submitted zero frames after chunk")
        log_record.update({"status": "complete", "submitted_motion_commands": submitted,
                           "release_frames": release_count})
        print("[ACT-REAL] ONE CHUNK COMPLETE", flush=True)
        print("[ACT-REAL] NO MORE ACT COMMANDS WILL BE EXECUTED", flush=True)
        return 0
    except KeyboardInterrupt:
        print("[ACT-REAL] ABORT | operator Ctrl+C", flush=True)
        log_record.update({"status": "aborted", "reason": "operator Ctrl+C"})
        return 1
    except Exception as exc:
        robot_side_stop = isinstance(exc, ActAbort) and exc.robot_side_stop
        print(f"[ACT-REAL] ABORT | {type(exc).__name__}: {exc}", flush=True)
        log_record.update({"status": "aborted", "reason": f"{type(exc).__name__}: {exc}"})
        return 1
    finally:
        if transport is not None and not released and not robot_side_stop and not transport.fatal_reason:
            try:
                if not topic_safe(reader.latest()):
                    release_count = send_shutdown_release(transport, reader)
                    log_record["abort_release_frames"] = release_count
            except Exception as exc:
                log_record["abort_release_error"] = f"{type(exc).__name__}: {exc}"
        if transport is not None:
            transport.terminate_once()
        if receiver is not None:
            receiver.close()
        if publisher is not None:
            publisher.close()
        if recording_stop is not None:
            recording_stop.set()
        if preview_thread is not None:
            preview_thread.join(1.0)
        if capture_thread is not None:
            capture_thread.join(2.0)
        if gripper is not None:
            gripper.shutdown()
        reader.close()
        config.LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
        log_path = config.LOG_DIRECTORY / f"act_real_once_{datetime.now():%Y%m%d_%H%M%S}.json"
        log_path.write_text(json.dumps(log_record, indent=2) + "\n", encoding="utf-8")
        print(f"[ACT-REAL] log={log_path}", flush=True)


def main() -> int:
    try:
        args = parse_args()
        validate_args(args)
        return dry_run(args) if args.dry_run_action_ipc else real_run(args)
    except Exception as exc:
        print(f"[ACT-REAL] ABORT | {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
