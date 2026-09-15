#!/usr/bin/env python3
"""Product v5 continuous multi-chunk ACT arm-only executor.

Real mode is the sole hardware owner.  It executes only action[0:10, :6] from
each of at most ten validated [100,7] chunks.  Dry-run mode creates only the
localhost action PULL socket and never initializes hardware.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import act_control_once as once
import config
from act_action_receiver_continuous import ACTION_ENDPOINT, ContinuousActionChunkReceiver
from jogany_io import _payload, build_active_payload, build_release_payload


DEFAULT_SPEED_SCALE = 0.05
DEFAULT_MODEL_STEPS_PER_REPLAN = 10
DEFAULT_MAX_CHUNKS = 10
MAX_REAL_CHUNKS = 10
ACTION_STEPS = 100
ARM_DIM = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Product v5 continuous ACT real-arm executor.")
    parser.add_argument("--act-speed-scale", type=float, default=DEFAULT_SPEED_SCALE)
    parser.add_argument("--model-steps-per-replan", type=int, default=DEFAULT_MODEL_STEPS_PER_REPLAN)
    parser.add_argument("--max-chunks", type=int, default=DEFAULT_MAX_CHUNKS)
    parser.add_argument("--action-endpoint", default=ACTION_ENDPOINT)
    parser.add_argument("--action-timeout-s", type=float, default=5.0)
    parser.add_argument("--connect-timeout-ms", type=int, default=1000)
    parser.add_argument("--ip", default=config.ROBOT_IP)
    parser.add_argument("--execute-act-motion", action="store_true")
    parser.add_argument("--confirm-arm-home", action="store_true")
    parser.add_argument("--confirm-gripper-open", action="store_true")
    parser.add_argument("--confirm-web-stop-ready", action="store_true")
    parser.add_argument("--confirm-emergency-stop-ready", action="store_true")
    parser.add_argument("--dry-run-action-ipc", action="store_true")
    parser.add_argument("--dry-run-realtime", action="store_true")
    parser.add_argument(
        "--dry-run-current-q",
        type=float,
        nargs=6,
        default=(0.0, 0.01, 0.02, 0.03, 0.04, 0.05),
    )
    parser.add_argument(
        "--dry-run-output",
        type=Path,
        default=Path("/tmp/tb6_act_control_continuous_dry_run.json"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.act_speed_scale <= 1.0:
        raise ValueError("--act-speed-scale must be in (0,1]")
    if not 1 <= args.model_steps_per_replan <= ACTION_STEPS:
        raise ValueError(f"--model-steps-per-replan must be in [1,{ACTION_STEPS}]")
    if not 1 <= args.max_chunks <= MAX_REAL_CHUNKS:
        raise ValueError(f"--max-chunks must be in [1,{MAX_REAL_CHUNKS}]; an eleventh chunk is forbidden")
    if args.action_timeout_s <= 0.0:
        raise ValueError("--action-timeout-s must be positive")
    if args.dry_run_action_ipc:
        if args.execute_act_motion:
            raise ValueError("dry-run and real motion are mutually exclusive")
        return
    if not args.execute_act_motion:
        raise ValueError("real mode requires --execute-act-motion")
    missing = [
        name
        for name in (
            "confirm_arm_home",
            "confirm_gripper_open",
            "confirm_web_stop_ready",
            "confirm_emergency_stop_ready",
        )
        if not getattr(args, name)
    ]
    if missing:
        raise ValueError("missing real-motion confirmations: " + ", ".join(missing))


def resample_arm_prefix(command_anchor: Any, action: Any, model_steps: int, speed_scale: float) -> np.ndarray:
    """Time-resample anchor -> action[0] -> ... -> action[model_steps-1]."""
    anchor = np.asarray(command_anchor, dtype=np.float64)
    chunk = np.asarray(action, dtype=np.float64)
    if anchor.shape != (ARM_DIM,) or not np.all(np.isfinite(anchor)):
        raise once.ActAbort("command anchor must contain six finite values")
    if chunk.shape != (ACTION_STEPS, 7) or not np.all(np.isfinite(chunk)):
        raise once.ActAbort(f"ACT action must be finite [{ACTION_STEPS},7]")
    if not 1 <= model_steps <= ACTION_STEPS:
        raise once.ActAbort(f"model_steps must be in [1,{ACTION_STEPS}]")
    if not 0.0 < speed_scale <= 1.0:
        raise once.ActAbort("speed_scale must be in (0,1]")
    prefix = chunk[:model_steps, :ARM_DIM]
    nodes = np.vstack((anchor[None, :], prefix))
    command_count = math.ceil(model_steps / speed_scale - 1e-12)
    commands = np.empty((command_count, ARM_DIM), dtype=np.float64)
    for step in range(1, command_count + 1):
        phase = min(float(model_steps), step * speed_scale)
        left = int(math.floor(phase))
        if left >= model_steps:
            commands[step - 1] = nodes[model_steps]
        else:
            alpha = phase - left
            commands[step - 1] = (1.0 - alpha) * nodes[left] + alpha * nodes[left + 1]
    if not np.all(np.isfinite(commands)) or not np.array_equal(commands[-1], prefix[-1]):
        raise once.ActAbort(
            f"time-resampled prefix is invalid or does not reach action[{model_steps - 1}]"
        )
    return commands


def validate_packet_order(packet: dict[str, Any], expected_chunk: int, previous_sequence: int | None) -> bool:
    if previous_sequence is not None and packet["source_observation_sequence"] <= previous_sequence:
        return False
    if packet["chunk_index"] != expected_chunk:
        raise once.ActAbort(
            f"action chunk_index must be {expected_chunk}, got {packet['chunk_index']}"
        )
    return True


def make_chunk_record(
    packet: dict[str, Any],
    current_q_actual: np.ndarray,
    command_anchor: np.ndarray,
    commands: np.ndarray,
    model_steps: int,
) -> dict[str, Any]:
    anchor_gap = np.abs(command_anchor - current_q_actual)
    full_path = np.vstack((command_anchor[None, :], commands))
    velocity = np.diff(full_path, axis=0) / config.DT
    first_velocity = velocity[0]
    return {
        "chunk_index": packet["chunk_index"],
        "source_observation_sequence": packet["source_observation_sequence"],
        "source_observation_timestamp_ns": packet["source_observation_timestamp_ns"],
        "inference_ms": packet["inference_ms"],
        "current_q_actual": current_q_actual.tolist(),
        "command_anchor": command_anchor.tolist(),
        "command_anchor_q_actual_max_abs_rad": float(anchor_gap.max()),
        "command_anchor_q_actual_max_abs_deg": float(np.rad2deg(anchor_gap.max())),
        "prediction_first_action": packet["action"][0].tolist(),
        "prediction_action_9": packet["action"][model_steps - 1].tolist(),
        "first_resampled_command": commands[0].tolist(),
        "first_command_velocity_rad_s": first_velocity.tolist(),
        "execution_model_action_first_index": 0,
        "execution_model_action_last_index": model_steps - 1,
        "ignored_model_action_count": ACTION_STEPS - model_steps,
        "execution_command_count": len(commands),
        "estimated_execution_duration_s": len(commands) * config.DT,
        "max_actual_command_velocity_rad_s_by_joint": np.abs(velocity).max(axis=0).tolist(),
        "max_actual_command_velocity_rad_s": float(np.abs(velocity).max()),
        "last_execution_command": commands[-1].tolist(),
        "last_reaches_selected_action": bool(
            np.array_equal(commands[-1], packet["action"][model_steps - 1, :ARM_DIM])
        ),
        "action_after_prefix_not_executed": True,
        "prediction_gripper_min": float(packet["action"][:, 6].min()),
        "prediction_gripper_max": float(packet["action"][:, 6].max()),
        "submitted_command_count": 0,
        "status": "validated",
    }


def apply_chunk_deceleration_envelope(
    payloads: list[Any], current_dq_actual: Any
) -> tuple[list[Any], np.ndarray, np.ndarray]:
    if not payloads:
        raise once.ActAbort("chunk payloads must not be empty")
    dq_actual = np.asarray(current_dq_actual, dtype=np.float64)
    if dq_actual.shape != (ARM_DIM,) or not np.all(np.isfinite(dq_actual)):
        raise once.ActAbort("chunk-start dq_actual must contain six finite values")
    payload_velocity = np.stack(
        [np.asarray(payload.joint_vel, dtype=np.float64) for payload in payloads], axis=0
    )
    chunk_peak_velocity = np.maximum(np.abs(dq_actual), np.abs(payload_velocity).max(axis=0))
    chunk_deceleration = config.PAYLOAD_ACCELERATION_SCALE * chunk_peak_velocity
    enveloped = [
        _payload(
            payload.joint_pos,
            payload.joint_vel,
            payload.joint_acc,
            chunk_deceleration,
        )
        for payload in payloads
    ]
    return enveloped, chunk_peak_velocity, chunk_deceleration


def dry_run(args: argparse.Namespace) -> int:
    receiver = ContinuousActionChunkReceiver(args.action_endpoint)
    command_anchor = np.asarray(args.dry_run_current_q, dtype=np.float64)
    previous_sequence: int | None = None
    chunks: list[dict[str, Any]] = []
    total_commands = 0
    try:
        for expected_chunk in range(args.max_chunks):
            deadline = time.monotonic() + args.action_timeout_s
            packet = None
            while packet is None:
                remaining_ms = max(0, int((deadline - time.monotonic()) * 1000.0))
                if remaining_ms == 0:
                    raise once.ActAbort(f"timed out waiting for action chunk {expected_chunk}")
                candidate = receiver.receive(remaining_ms, allow_mock=True)
                if candidate is None:
                    raise once.ActAbort(f"timed out waiting for action chunk {expected_chunk}")
                if validate_packet_order(candidate, expected_chunk, previous_sequence):
                    packet = candidate
            current_q_actual = command_anchor.copy()
            commands = resample_arm_prefix(
                command_anchor, packet["action"], args.model_steps_per_replan, args.act_speed_scale
            )
            payloads = once.prevalidate_payloads(commands, command_anchor)
            payloads, chunk_peak_velocity, chunk_deceleration = apply_chunk_deceleration_envelope(
                payloads, np.zeros(ARM_DIM, dtype=np.float64)
            )
            record = make_chunk_record(
                packet, current_q_actual, command_anchor, commands, args.model_steps_per_replan
            )
            record["chunk_peak_velocity_rad_s_by_joint"] = chunk_peak_velocity.tolist()
            record["chunk_deceleration_rad_s2_by_joint"] = chunk_deceleration.tolist()
            record["submitted_command_count"] = len(payloads)
            record["status"] = "complete"
            chunks.append(record)
            total_commands += len(payloads)
            previous_sequence = packet["source_observation_sequence"]
            command_anchor = commands[-1].copy()
            if args.dry_run_realtime:
                time.sleep(len(payloads) * config.DT)
            print(
                f"[ACT-CONTINUOUS/DRY-RUN] chunk {expected_chunk + 1}/{args.max_chunks} complete | "
                f"seq={previous_sequence} | commands={len(payloads)}",
                flush=True,
            )
    finally:
        receiver.close()

    report = {
        "mode": "NO_HARDWARE_DRY_RUN",
        "requested_max_chunks": args.max_chunks,
        "completed_chunks": len(chunks),
        "model_steps_per_replan": args.model_steps_per_replan,
        "speed_scale": args.act_speed_scale,
        "control_hz": config.CONTROL_HZ,
        "total_submitted_commands": total_commands,
        "estimated_total_execution_duration_s": total_commands * config.DT,
        "dry_run_realtime": bool(args.dry_run_realtime),
        "chunk_anchor_chain_pass": all(
            np.array_equal(
                np.asarray(chunks[index]["command_anchor"]),
                np.asarray(chunks[index - 1]["last_execution_command"]),
            )
            for index in range(1, len(chunks))
        ),
        "all_prefixes_end_at_action_9": all(chunk["last_reaches_selected_action"] for chunk in chunks),
        "all_actions_10_to_99_not_executed": all(chunk["action_after_prefix_not_executed"] for chunk in chunks),
        "existing_0_2_rad_s_validator": "PASS",
        "chunks": chunks,
        "status": "complete",
    }
    args.dry_run_output.parent.mkdir(parents=True, exist_ok=True)
    args.dry_run_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print("[ACT-CONTINUOUS/DRY-RUN] PASS", flush=True)
    return 0


def publish_observation(publisher: Any, recorder: Any, topic: Any, gripper: Any, transport: Any) -> None:
    from shadow_observation_publisher import ShadowPublishError

    try:
        publisher.publish_if_due(
            camera=recorder.camera,
            topic=topic,
            gripper_snapshot=gripper.get_snapshot(),
            q_command=transport.last_accepted_command_q,
        )
    except ShadowPublishError:
        # Existing caches can be temporarily incomplete during startup.
        pass


def topic_snapshot_record(topic: Any) -> dict[str, Any]:
    return {
        "monotonic_timestamp_ns": time.monotonic_ns(),
        "topic_global_count": int(topic.global_count),
        "topic_q_target": np.asarray(topic.q_target_rad, dtype=np.float64).tolist(),
        "topic_q_actual": np.asarray(topic.q_actual_rad, dtype=np.float64).tolist(),
        "topic_dq_actual": np.asarray(topic.dq_actual_rad_s, dtype=np.float64).tolist(),
    }


def submit_wait_hold_endpoint(transport: Any) -> None:
    anchor = np.asarray(transport.last_accepted_command_q, dtype=np.float64)
    payload = build_release_payload(anchor)
    result = transport.submit_main(payload.text, payload.joint_pos, payload.joint_vel)
    if not result.submitted or transport.fatal_reason:
        raise once.ActAbort(
            f"wait hold submission failed: {transport.fatal_reason or result.callback_state}"
        )


def wait_for_fresh_packet(
    receiver: ContinuousActionChunkReceiver,
    expected_chunk: int,
    previous_sequence: int | None,
    timeout_s: float,
    transport: Any,
    reader: Any,
    publisher: Any,
    recorder: Any,
    gripper: Any,
) -> tuple[dict[str, Any], Any, int]:
    from teleop import robot_side_stop_observed, topic_safe

    started = time.monotonic()
    deadline = time.perf_counter() + config.DT
    latest_topic = reader.latest()
    maintenance_submissions = 0
    while True:
        if time.monotonic() - started > timeout_s:
            raise once.ActAbort(f"timed out waiting for fresh action chunk {expected_chunk}")
        remaining = deadline - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)
        latest_topic = reader.latest()
        safety = topic_safe(latest_topic)
        if safety:
            raise once.ActAbort(
                f"Topic safety failed while waiting for chunk {expected_chunk}: {safety}",
                robot_side_stop=robot_side_stop_observed(latest_topic),
            )
        if transport.fatal_reason:
            raise once.ActAbort(f"transport fatal while waiting: {transport.fatal_reason}")
        packet = receiver.receive(0, allow_mock=False)
        if packet is not None and validate_packet_order(packet, expected_chunk, previous_sequence):
            return packet, latest_topic, maintenance_submissions
        publish_observation(publisher, recorder, latest_topic, gripper, transport)
        submit_wait_hold_endpoint(transport)
        maintenance_submissions += 1
        deadline += config.DT
        if time.perf_counter() >= deadline:
            deadline = time.perf_counter() + config.DT


def execute_payloads_with_observation(
    payloads: list[Any],
    chunk_index: int,
    execution_trace: list[dict[str, Any]],
    transport: Any,
    reader: Any,
    publisher: Any,
    recorder: Any,
    gripper: Any,
) -> int:
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
            raise once.ActAbort(
                f"Topic safety failed before execution command {index}: {safety}",
                robot_side_stop=robot_side_stop_observed(topic),
            )
        if transport.fatal_reason:
            raise once.ActAbort(f"transport fatal before command {index}: {transport.fatal_reason}")
        execution_trace.append(
            {
                **topic_snapshot_record(topic),
                "trace_point": "topic_before_submit",
                "chunk_index": chunk_index,
                "command_index": index,
                "payload_joint_pos": np.asarray(payload.joint_pos, dtype=np.float64).tolist(),
                "payload_joint_vel": np.asarray(payload.joint_vel, dtype=np.float64).tolist(),
                "payload_joint_acc": np.asarray(payload.joint_acc, dtype=np.float64).tolist(),
                "payload_joint_dec": np.asarray(payload.joint_dec, dtype=np.float64).tolist(),
                "transport_last_accepted_command_q": np.asarray(
                    transport.last_accepted_command_q, dtype=np.float64
                ).tolist(),
            }
        )
        publish_observation(publisher, recorder, topic, gripper, transport)
        try:
            result = transport.submit_main(payload.text, payload.joint_pos, payload.joint_vel)
        except Exception as exc:
            raise once.ActAbort(f"command {index} submission exception: {exc}") from exc
        if not result.submitted or transport.fatal_reason:
            raise once.ActAbort(
                f"command {index} submission failed: {transport.fatal_reason or result.callback_state}"
            )
        submitted += 1
        deadline += config.DT
        if time.perf_counter() >= deadline:
            deadline = time.perf_counter() + config.DT
    return submitted


def real_run(args: argparse.Namespace) -> int:
    from gripper_io import GripperIO
    from shadow_observation_publisher import ShadowObservationPublisher
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
    receiver = ContinuousActionChunkReceiver(args.action_endpoint)
    transport = gripper = recorder = publisher = None
    recording_stop = capture_thread = preview_thread = None
    released = False
    robot_side_stop = False
    log_record: dict[str, Any] = {
        "status": "starting",
        "requested_max_chunks": args.max_chunks,
        "completed_chunks": 0,
        "model_steps_per_replan": args.model_steps_per_replan,
        "speed_scale": args.act_speed_scale,
        "total_submitted_commands": 0,
        "active_wait_maintenance_submissions": 0,
        "chunks": [],
    }
    try:
        reader.start(args.ip)
        initial_topic = _wait_start_topic(reader)
        q_initial = np.asarray(initial_topic.q_actual_rad, dtype=np.float64)
        home_error_deg = home_max_delta_deg(q_initial)
        if home_error_deg > config.HOME_MAX_DELTA_DEG:
            raise once.ActAbort(
                f"robot is too far from training HOME: {home_error_deg:.3f}deg > existing "
                f"HOME_MAX_DELTA_DEG={config.HOME_MAX_DELTA_DEG:.3f}deg"
            )
        transport = once.initialize_verified_transport(q_initial, args.ip, args.connect_timeout_ms)
        gripper = GripperIO(
            vel=config.GRIPPER_VEL,
            force=config.GRIPPER_FORCE,
            acc=config.GRIPPER_ACC,
            dec=config.GRIPPER_DEC,
        )
        gripper.start()
        print("[ACT-CONTINUOUS] OmniPicker is Product-owned; ACT gripper actions are ignored.", flush=True)
        recorder, recording_stop, capture_thread, preview_thread = _start_recording_peripherals()
        publisher = ShadowObservationPublisher(
            camera_serial_dict=config.CAM_SERIAL_DICT,
            endpoint="tcp://127.0.0.1:5557",
            publish_hz=10.0,
        )
        previous_sequence: int | None = None
        for expected_chunk in range(args.max_chunks):
            packet, latest_topic, maintenance = wait_for_fresh_packet(
                receiver,
                expected_chunk,
                previous_sequence,
                args.action_timeout_s,
                transport,
                reader,
                publisher,
                recorder,
                gripper,
            )
            log_record["active_wait_maintenance_submissions"] += maintenance
            latest_topic = reader.latest()
            safety = topic_safe(latest_topic)
            if safety:
                raise once.ActAbort(
                    f"Topic safety failed before chunk {expected_chunk} validation: {safety}",
                    robot_side_stop=robot_side_stop_observed(latest_topic),
                )
            current_q_actual = np.asarray(latest_topic.q_actual_rad, dtype=np.float64)
            command_anchor = np.asarray(transport.last_accepted_command_q, dtype=np.float64)
            commands = resample_arm_prefix(
                command_anchor,
                packet["action"],
                args.model_steps_per_replan,
                args.act_speed_scale,
            )
            chunk_record = make_chunk_record(
                packet, current_q_actual, command_anchor, commands, args.model_steps_per_replan
            )
            log_record["chunks"].append(chunk_record)
            payloads = once.prevalidate_payloads(commands, command_anchor)
            payloads, chunk_peak_velocity, chunk_deceleration = apply_chunk_deceleration_envelope(
                payloads, latest_topic.dq_actual_rad_s
            )
            chunk_record["chunk_peak_velocity_rad_s_by_joint"] = chunk_peak_velocity.tolist()
            chunk_record["chunk_deceleration_rad_s2_by_joint"] = chunk_deceleration.tolist()
            chunk_record["execution_trace"] = []
            submitted = execute_payloads_with_observation(
                payloads,
                expected_chunk,
                chunk_record["execution_trace"],
                transport,
                reader,
                publisher,
                recorder,
                gripper,
            )
            chunk_record["topic_after_execution"] = topic_snapshot_record(reader.latest())
            chunk_record["submitted_command_count"] = submitted
            chunk_record["status"] = "complete"
            log_record["completed_chunks"] += 1
            log_record["total_submitted_commands"] += submitted
            previous_sequence = packet["source_observation_sequence"]
            print(
                f"[ACT-CONTINUOUS] chunk {expected_chunk + 1}/{args.max_chunks} complete | "
                f"seq={previous_sequence} | commands={submitted}",
                flush=True,
            )

        release_count = send_shutdown_release(transport, reader)
        released = release_count > 0
        if not released:
            raise once.ActAbort("verified release/deceleration submitted zero frames after final chunk")
        log_record.update({"status": "complete", "release_frames": release_count})
        print("[ACT-CONTINUOUS] TEST COMPLETE", flush=True)
        print(
            f"[ACT-CONTINUOUS] {log_record['completed_chunks']}/{args.max_chunks} CHUNKS EXECUTED",
            flush=True,
        )
        print("[ACT-CONTINUOUS] NO MORE ACT COMMANDS WILL BE EXECUTED", flush=True)
        return 0
    except KeyboardInterrupt:
        print("[ACT-CONTINUOUS] ABORT | operator Ctrl+C", flush=True)
        log_record.update({"status": "aborted", "reason": "operator Ctrl+C"})
        return 1
    except Exception as exc:
        robot_side_stop = isinstance(exc, once.ActAbort) and exc.robot_side_stop
        print(f"[ACT-CONTINUOUS] ABORT | {type(exc).__name__}: {exc}", flush=True)
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
        log_path = config.LOG_DIRECTORY / f"act_real_continuous_{datetime.now():%Y%m%d_%H%M%S}.json"
        log_path.write_text(json.dumps(log_record, indent=2) + "\n", encoding="utf-8")
        print(f"[ACT-CONTINUOUS] log={log_path}", flush=True)


def main() -> int:
    try:
        args = parse_args()
        validate_args(args)
        return dry_run(args) if args.dry_run_action_ipc else real_run(args)
    except Exception as exc:
        print(f"[ACT-CONTINUOUS] ABORT | {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
