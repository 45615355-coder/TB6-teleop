#!/usr/bin/env python3
"""Product v5 infinite full ACT owner for TB6-R5, OmniPicker, and observations."""

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
from act_action_receiver_full import ACTION_ENDPOINT, FullActionReceiver
from jogany_io import _payload, build_release_payload


DEFAULT_SPEED_SCALE = 0.03
MODEL_HZ = 50.0
ARM_DIM = 6
GRIPPER_MIN_ACCEPTED = -5.0
GRIPPER_MAX_ACCEPTED = 105.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Product v5 infinite full ACT executor.")
    parser.add_argument("--act-speed-scale", type=float, default=DEFAULT_SPEED_SCALE)
    parser.add_argument("--action-endpoint", default=ACTION_ENDPOINT)
    parser.add_argument("--action-timeout-s", type=float, default=5.0)
    parser.add_argument("--gripper-ready-timeout-s", type=float, default=7.0)
    parser.add_argument("--connect-timeout-ms", type=int, default=1000)
    parser.add_argument("--ip", default=config.ROBOT_IP)
    parser.add_argument("--execute-act-motion", action="store_true")
    parser.add_argument("--confirm-arm-home", action="store_true")
    parser.add_argument("--confirm-gripper-open", action="store_true")
    parser.add_argument("--confirm-web-stop-ready", action="store_true")
    parser.add_argument("--confirm-emergency-stop-ready", action="store_true")
    parser.add_argument("--dry-run-action-ipc", action="store_true")
    parser.add_argument("--dry-run-steps", type=int, default=None)
    parser.add_argument(
        "--dry-run-current-q",
        type=float,
        nargs=6,
        default=(0.0, 0.01, 0.02, 0.03, 0.04, 0.05),
    )
    parser.add_argument("--dry-run-gripper-pos", type=float, default=100.0)
    parser.add_argument(
        "--dry-run-output",
        type=Path,
        default=Path("/tmp/tb6_act_control_full_dry_run.json"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.action_endpoint != ACTION_ENDPOINT:
        raise ValueError(f"action endpoint must be localhost-only {ACTION_ENDPOINT}")
    if not 0.0 < args.act_speed_scale <= 1.0:
        raise ValueError("--act-speed-scale must be in (0,1]")
    if args.action_timeout_s <= 0.0 or args.gripper_ready_timeout_s <= 0.0:
        raise ValueError("action/gripper timeouts must be positive")
    if args.connect_timeout_ms <= 0:
        raise ValueError("--connect-timeout-ms must be positive")
    if args.dry_run_action_ipc:
        if args.execute_act_motion:
            raise ValueError("dry-run and real motion are mutually exclusive")
        if args.dry_run_steps is None or args.dry_run_steps <= 0:
            raise ValueError("dry-run requires a positive --dry-run-steps")
        return
    if args.dry_run_steps is not None:
        raise ValueError("--dry-run-steps is forbidden in real mode")
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


def cumulative_host_ticks(model_steps: int, speed_scale: float) -> int:
    if model_steps < 0 or not 0.0 < speed_scale <= 1.0:
        raise once.ActAbort("invalid model_steps/speed_scale")
    return math.ceil(model_steps / speed_scale - 1e-12)


def host_ticks_for_model_step(step_index: int, speed_scale: float) -> int:
    if step_index < 0:
        raise once.ActAbort("step_index must be non-negative")
    return cumulative_host_ticks(step_index + 1, speed_scale) - cumulative_host_ticks(
        step_index, speed_scale
    )


def validate_gripper_target(value: Any) -> float:
    try:
        target = float(value)
    except (TypeError, ValueError) as exc:
        raise once.ActAbort("ACT gripper position is not numeric") from exc
    if not math.isfinite(target):
        raise once.ActAbort("ACT gripper position is not finite")
    if not GRIPPER_MIN_ACCEPTED <= target <= GRIPPER_MAX_ACCEPTED:
        raise once.ActAbort(
            f"ACT gripper position {target:.6f} is outside "
            f"[{GRIPPER_MIN_ACCEPTED},{GRIPPER_MAX_ACCEPTED}]"
        )
    return min(100.0, max(0.0, target))


def build_model_step_commands(
    q_anchor: Any,
    target_q: Any,
    previous_gripper_pos: float,
    target_gripper_pos: float,
    step_index: int,
    speed_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor = np.asarray(q_anchor, dtype=np.float64)
    target = np.asarray(target_q, dtype=np.float64)
    if anchor.shape != (ARM_DIM,) or target.shape != (ARM_DIM,):
        raise once.ActAbort("arm anchor/target must each contain six values")
    if not np.all(np.isfinite(anchor)) or not np.all(np.isfinite(target)):
        raise once.ActAbort("arm anchor/target contains NaN/Inf")
    ticks = host_ticks_for_model_step(step_index, speed_scale)
    if ticks <= 0:
        raise once.ActAbort("model step produced no host commands")
    alpha = np.arange(1, ticks + 1, dtype=np.float64) / float(ticks)
    commands = anchor[None, :] + alpha[:, None] * (target - anchor)[None, :]
    gripper_commands = float(previous_gripper_pos) + alpha * (
        float(target_gripper_pos) - float(previous_gripper_pos)
    )
    commands[-1] = target
    gripper_commands[-1] = float(target_gripper_pos)
    if (
        not np.all(np.isfinite(commands))
        or not np.all(np.isfinite(gripper_commands))
        or not np.array_equal(commands[-1], target)
        or gripper_commands[-1] != float(target_gripper_pos)
    ):
        raise once.ActAbort("model-step interpolation is invalid")
    return commands, gripper_commands, alpha


def apply_segment_deceleration_envelope(
    payloads: list[Any], current_dq_actual: Any
) -> tuple[list[Any], np.ndarray, np.ndarray]:
    if not payloads:
        raise once.ActAbort("segment payloads must not be empty")
    dq_actual = np.asarray(current_dq_actual, dtype=np.float64)
    if dq_actual.shape != (ARM_DIM,) or not np.all(np.isfinite(dq_actual)):
        raise once.ActAbort("segment-start dq_actual must contain six finite values")
    payload_velocity = np.stack(
        [np.asarray(payload.joint_vel, dtype=np.float64) for payload in payloads], axis=0
    )
    peak_velocity = np.maximum(np.abs(dq_actual), np.abs(payload_velocity).max(axis=0))
    fixed_deceleration = config.PAYLOAD_ACCELERATION_SCALE * peak_velocity
    enveloped = [
        _payload(
            payload.joint_pos,
            payload.joint_vel,
            payload.joint_acc,
            fixed_deceleration,
        )
        for payload in payloads
    ]
    return enveloped, peak_velocity, fixed_deceleration


def gripper_health_reason(snapshot: dict[str, Any], *, require_ready: bool = False) -> str:
    fault = snapshot.get("fault")
    if fault not in (None, 0):
        return f"OmniPicker fault={fault}"
    error = snapshot.get("error")
    if error:
        return f"OmniPicker worker error: {error}"
    if require_ready and not bool(snapshot.get("ready")):
        return "OmniPicker is not READY"
    # state=2 (STALL) remains valid task feedback and is not a global arm fault.
    return ""


def command_gripper_position(gripper: Any, position: float) -> None:
    if not 0.0 <= position <= 100.0 or not math.isfinite(position):
        raise once.ActAbort("interpolated gripper position is invalid")
    # Reuse GripperIO's public API only. ACT action[6] remains an absolute
    # gripper position, not a Trigger; this adapter performs the conversion.
    trigger = 1.0 - position / 100.0
    if not gripper.set_trigger(trigger, xr_fresh=True):
        raise once.ActAbort("GripperIO rejected an ACT gripper position")


def publish_observation(
    publisher: Any, recorder: Any, topic: Any, gripper: Any, transport: Any
) -> None:
    from shadow_observation_publisher import ShadowPublishError

    try:
        publisher.publish_if_due(
            camera=recorder.camera,
            topic=topic,
            gripper_snapshot=gripper.get_snapshot(),
            q_command=transport.last_accepted_command_q,
        )
    except ShadowPublishError:
        pass


def next_deadline(deadline: float) -> float:
    candidate = deadline + config.DT
    return candidate if time.perf_counter() < candidate else time.perf_counter() + config.DT


def start_camera_video_recorder(camera: Any, run_stamp: str) -> Any:
    """Best-effort startup for the non-critical automatic camera video."""
    if not config.CAMERA_VIDEO_ENABLED:
        return None
    try:
        from camera_video_recorder import CameraVideoRecorder

        video_recorder = CameraVideoRecorder(
            camera=camera,
            camera_serial_dict=config.CAM_SERIAL_DICT,
            fps=config.CAMERA_VIDEO_FPS,
            panel_width=config.CAMERA_VIDEO_PANEL_WIDTH,
            panel_height=config.CAMERA_VIDEO_PANEL_HEIGHT,
            output_directory=config.CAMERA_VIDEO_DIR,
            run_stamp=run_stamp,
        )
        video_recorder.start()
        return video_recorder
    except Exception as error:
        print(
            f"[VIDEO] WARNING | startup failed: {type(error).__name__}: {error}",
            flush=True,
        )
        return None


def dry_run(args: argparse.Namespace) -> int:
    receiver = FullActionReceiver(args.action_endpoint)
    q_anchor = np.asarray(args.dry_run_current_q, dtype=np.float64)
    previous_gripper = validate_gripper_target(args.dry_run_gripper_pos)
    trace: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    host_command_index = 0
    try:
        for expected_step in range(args.dry_run_steps):
            packet = receiver.receive(int(args.action_timeout_s * 1000.0), allow_mock=True)
            if packet is None:
                raise once.ActAbort(f"timed out waiting for mock action step {expected_step}")
            if abs(packet["speed_scale"] - args.act_speed_scale) > 1e-12:
                raise once.ActAbort("sender/Product speed_scale mismatch")
            action = packet["action"]
            target_gripper = validate_gripper_target(action[6])
            commands, gripper_commands, alphas = build_model_step_commands(
                q_anchor,
                action[:ARM_DIM],
                previous_gripper,
                target_gripper,
                packet["step_index"],
                args.act_speed_scale,
            )
            original_payloads = once.prevalidate_payloads(commands, q_anchor)
            payloads, peak_velocity, fixed_deceleration = apply_segment_deceleration_envelope(
                original_payloads, np.zeros(ARM_DIM, dtype=np.float64)
            )
            if not all(
                np.array_equal(original.joint_pos, modified.joint_pos)
                and np.array_equal(original.joint_vel, modified.joint_vel)
                and np.array_equal(original.joint_acc, modified.joint_acc)
                for original, modified in zip(original_payloads, payloads, strict=True)
            ):
                raise once.ActAbort("deceleration envelope changed pos/vel/acc")
            for local_index, (payload, gripper_command, alpha) in enumerate(
                zip(payloads, gripper_commands, alphas, strict=True)
            ):
                trace.append(
                    {
                        "model_step_index": packet["step_index"],
                        "host_command_index": host_command_index,
                        "model_phase": packet["step_index"] + float(alpha),
                        "alpha": float(alpha),
                        "q_command": np.asarray(payload.joint_pos).tolist(),
                        "payload_joint_vel": np.asarray(payload.joint_vel).tolist(),
                        "payload_joint_acc": np.asarray(payload.joint_acc).tolist(),
                        "payload_joint_dec": np.asarray(payload.joint_dec).tolist(),
                        "q_target": np.asarray(payload.joint_pos).tolist(),
                        "q_actual": np.asarray(payload.joint_pos).tolist(),
                        "dq_actual": np.asarray(payload.joint_vel).tolist(),
                        "act_gripper_target": target_gripper,
                        "interpolated_gripper_command": float(gripper_command),
                        "gripper_target_pos": float(gripper_command),
                        "gripper_actual_pos": float(gripper_command),
                        "gripper_state": 0,
                        "gripper_fault": 0,
                    }
                )
                host_command_index += 1
            steps.append(
                {
                    "step_index": packet["step_index"],
                    "host_command_count": len(payloads),
                    "chunk_peak_velocity_rad_s_by_joint": peak_velocity.tolist(),
                    "chunk_deceleration_rad_s2_by_joint": fixed_deceleration.tolist(),
                    "target_gripper_pos": target_gripper,
                }
            )
            q_anchor = commands[-1].copy()
            previous_gripper = target_gripper
    finally:
        receiver.close()

    report = {
        "mode": "NO_HARDWARE_FULL_DRY_RUN",
        "status": "complete",
        "completed_model_steps": len(steps),
        "speed_scale": args.act_speed_scale,
        "total_host_commands": host_command_index,
        "expected_total_host_commands": cumulative_host_ticks(len(steps), args.act_speed_scale),
        "release_hold_commands_between_ready_steps": 0,
        "all_arm_commands_finite": bool(
            all(np.all(np.isfinite(item["q_command"])) for item in trace)
        ),
        "steps": steps,
        "trace": trace,
    }
    args.dry_run_output.parent.mkdir(parents=True, exist_ok=True)
    args.dry_run_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"[ACT-FULL/DRY-RUN] PASS | steps={len(steps)} host_commands={host_command_index}",
        flush=True,
    )
    return 0


def wait_for_gripper_ready(
    gripper: Any,
    timeout_s: float,
    transport: Any,
    reader: Any,
    publisher: Any,
    recorder: Any,
    deadline: float,
) -> float:
    from teleop import robot_side_stop_observed, topic_safe

    started = time.monotonic()
    while time.monotonic() - started <= timeout_s:
        remaining = deadline - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)
        topic = reader.latest()
        safety = topic_safe(topic, stationary=True)
        if safety:
            raise once.ActAbort(
                f"Topic safety failed while waiting for OmniPicker READY: {safety}",
                robot_side_stop=robot_side_stop_observed(topic),
            )
        snapshot = gripper.get_snapshot()
        health = gripper_health_reason(snapshot)
        if health:
            raise once.ActAbort(health)
        if snapshot.get("ready"):
            publish_observation(publisher, recorder, topic, gripper, transport)
            return next_deadline(deadline)
        hold = build_release_payload(transport.last_accepted_command_q)
        result = transport.submit_main(hold.text, hold.joint_pos, hold.joint_vel)
        if not result.submitted or transport.fatal_reason:
            raise once.ActAbort(
                f"OmniPicker READY wait hold failed: {transport.fatal_reason or result.callback_state}"
            )
        deadline = next_deadline(deadline)
    raise once.ActAbort("timed out waiting for OmniPicker INITIAL_OPEN / READY")


def wait_for_action(
    receiver: FullActionReceiver,
    timeout_s: float,
    transport: Any,
    reader: Any,
    publisher: Any,
    recorder: Any,
    gripper: Any,
    deadline: float,
) -> tuple[dict[str, Any], Any, float, int]:
    from teleop import robot_side_stop_observed, topic_safe

    packet = receiver.receive(0, allow_mock=False)
    if packet is not None:
        topic = reader.latest()
        safety = topic_safe(topic)
        if safety:
            raise once.ActAbort(
                f"Topic safety failed before ready ACT action: {safety}",
                robot_side_stop=robot_side_stop_observed(topic),
            )
        health = gripper_health_reason(gripper.get_snapshot(), require_ready=True)
        if health:
            raise once.ActAbort(health)
        if transport.fatal_reason:
            raise once.ActAbort(f"transport fatal before ready ACT action: {transport.fatal_reason}")
        return packet, topic, deadline, 0
    started = time.monotonic()
    release_holds = 0
    while time.monotonic() - started <= timeout_s:
        remaining = deadline - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)
        topic = reader.latest()
        safety = topic_safe(topic)
        if safety:
            raise once.ActAbort(
                f"Topic safety failed while waiting for ACT action: {safety}",
                robot_side_stop=robot_side_stop_observed(topic),
            )
        health = gripper_health_reason(gripper.get_snapshot(), require_ready=True)
        if health:
            raise once.ActAbort(health)
        if transport.fatal_reason:
            raise once.ActAbort(f"transport fatal while waiting for ACT action: {transport.fatal_reason}")
        publish_observation(publisher, recorder, topic, gripper, transport)
        packet = receiver.receive(0, allow_mock=False)
        if packet is not None:
            return packet, topic, deadline, release_holds
        hold = build_release_payload(transport.last_accepted_command_q)
        result = transport.submit_main(hold.text, hold.joint_pos, hold.joint_vel)
        if not result.submitted or transport.fatal_reason:
            raise once.ActAbort(
                f"ACT action wait hold failed: {transport.fatal_reason or result.callback_state}"
            )
        release_holds += 1
        deadline = next_deadline(deadline)
    raise once.ActAbort("timed out waiting for the next ACT action")


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
    receiver = FullActionReceiver(args.action_endpoint)
    transport = gripper = recorder = publisher = video_recorder = None
    recording_stop = capture_thread = preview_thread = None
    trace_stream = None
    released = False
    robot_side_stop = False
    status = "starting"
    completed_steps = 0
    host_command_index = 0
    wait_release_holds = 0
    config.LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = config.LOG_DIRECTORY / f"act_real_full_{stamp}.jsonl"
    summary_path = config.LOG_DIRECTORY / f"act_real_full_{stamp}_summary.json"
    try:
        reader.start(args.ip)
        initial_topic = _wait_start_topic(reader)
        q_initial = np.asarray(initial_topic.q_actual_rad, dtype=np.float64)
        home_error_deg = home_max_delta_deg(q_initial)
        if home_error_deg > config.HOME_MAX_DELTA_DEG:
            raise once.ActAbort(
                f"robot is too far from training HOME: {home_error_deg:.3f}deg > "
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
        recorder, recording_stop, capture_thread, preview_thread = _start_recording_peripherals()
        video_recorder = start_camera_video_recorder(recorder.camera, stamp)
        publisher = ShadowObservationPublisher(
            camera_serial_dict=config.CAM_SERIAL_DICT,
            endpoint="tcp://127.0.0.1:5557",
            publish_hz=20.0,
        )
        trace_stream = trace_path.open("x", encoding="utf-8", buffering=1)
        deadline = time.perf_counter() + config.DT
        deadline = wait_for_gripper_ready(
            gripper,
            args.gripper_ready_timeout_s,
            transport,
            reader,
            publisher,
            recorder,
            deadline,
        )
        initial_gripper = gripper.get_snapshot().get("actual_pos")
        if initial_gripper is None or not 0.0 <= float(initial_gripper) <= 100.0:
            raise once.ActAbort("OmniPicker READY has no valid actual_pos")
        previous_gripper_command = float(initial_gripper)
        status = "running"
        print("[ACT-FULL] READY | waiting for first valid 7D action", flush=True)
        while True:
            packet, segment_topic, deadline, holds = wait_for_action(
                receiver,
                args.action_timeout_s,
                transport,
                reader,
                publisher,
                recorder,
                gripper,
                deadline,
            )
            wait_release_holds += holds
            if abs(packet["speed_scale"] - args.act_speed_scale) > 1e-12:
                raise once.ActAbort("sender/Product speed_scale mismatch")
            action = packet["action"]
            target_gripper = validate_gripper_target(action[6])
            q_anchor = np.asarray(transport.last_accepted_command_q, dtype=np.float64)
            commands, gripper_commands, alphas = build_model_step_commands(
                q_anchor,
                action[:ARM_DIM],
                previous_gripper_command,
                target_gripper,
                packet["step_index"],
                args.act_speed_scale,
            )
            payloads = once.prevalidate_payloads(commands, q_anchor)
            payloads, _, _ = apply_segment_deceleration_envelope(
                payloads, segment_topic.dq_actual_rad_s
            )
            for payload, gripper_command, alpha in zip(
                payloads, gripper_commands, alphas, strict=True
            ):
                remaining = deadline - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
                topic = reader.latest()
                safety = topic_safe(topic)
                if safety:
                    raise once.ActAbort(
                        f"Topic safety failed during full ACT execution: {safety}",
                        robot_side_stop=robot_side_stop_observed(topic),
                    )
                snapshot = gripper.get_snapshot()
                health = gripper_health_reason(snapshot, require_ready=True)
                if health:
                    raise once.ActAbort(health)
                if transport.fatal_reason:
                    raise once.ActAbort(f"transport fatal during full ACT: {transport.fatal_reason}")
                publish_observation(publisher, recorder, topic, gripper, transport)
                command_gripper_position(gripper, float(gripper_command))
                trace = {
                    "monotonic_timestamp_ns": time.monotonic_ns(),
                    "model_step_index": packet["step_index"],
                    "host_command_index": host_command_index,
                    "model_phase": packet["step_index"] + float(alpha),
                    "alpha": float(alpha),
                    "q_command": np.asarray(payload.joint_pos).tolist(),
                    "payload_joint_vel": np.asarray(payload.joint_vel).tolist(),
                    "payload_joint_acc": np.asarray(payload.joint_acc).tolist(),
                    "payload_joint_dec": np.asarray(payload.joint_dec).tolist(),
                    "q_target": np.asarray(topic.q_target_rad).tolist(),
                    "q_actual": np.asarray(topic.q_actual_rad).tolist(),
                    "dq_actual": np.asarray(topic.dq_actual_rad_s).tolist(),
                    "act_gripper_target": target_gripper,
                    "interpolated_gripper_command": float(gripper_command),
                    "gripper_target_pos": snapshot.get("target_pos"),
                    "gripper_actual_pos": snapshot.get("actual_pos"),
                    "gripper_state": snapshot.get("state"),
                    "gripper_fault": snapshot.get("fault"),
                }
                trace_stream.write(json.dumps(trace) + "\n")
                try:
                    result = transport.submit_main(
                        payload.text, payload.joint_pos, payload.joint_vel
                    )
                except Exception as exc:
                    raise once.ActAbort(f"full ACT command submission exception: {exc}") from exc
                if not result.submitted or transport.fatal_reason:
                    raise once.ActAbort(
                        f"full ACT command failed: {transport.fatal_reason or result.callback_state}"
                    )
                host_command_index += 1
                deadline = next_deadline(deadline)
            previous_gripper_command = target_gripper
            completed_steps += 1
            print(
                f"[ACT-FULL] model_step={packet['step_index']} complete | "
                f"host_commands={len(payloads)}",
                flush=True,
            )
    except KeyboardInterrupt:
        status = "operator_ctrl_c"
        print("[ACT-FULL] STOP | operator Ctrl+C", flush=True)
        return 0
    except Exception as exc:
        robot_side_stop = isinstance(exc, once.ActAbort) and exc.robot_side_stop
        status = f"aborted: {type(exc).__name__}: {exc}"
        print(f"[ACT-FULL] ABORT | {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        if video_recorder is not None:
            video_recorder.request_stop()
        if transport is not None and not released and not robot_side_stop and not transport.fatal_reason:
            try:
                if not topic_safe(reader.latest()):
                    released = send_shutdown_release(transport, reader) > 0
            except Exception:
                pass
        if transport is not None:
            transport.terminate_once()
        receiver.close()
        if publisher is not None:
            publisher.close()
        if trace_stream is not None:
            trace_stream.close()
        if video_recorder is not None:
            try:
                video_recorder.stop()
            except Exception as error:
                print(
                    f"[VIDEO] WARNING | shutdown failed: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
        if recording_stop is not None:
            recording_stop.set()
        if preview_thread is not None:
            preview_thread.join(1.0)
        if capture_thread is not None:
            capture_thread.join(2.0)
        if gripper is not None:
            gripper.shutdown()
        reader.close()
        summary = {
            "status": status,
            "speed_scale": args.act_speed_scale,
            "completed_model_steps": completed_steps,
            "total_host_commands": host_command_index,
            "wait_release_hold_commands": wait_release_holds,
            "trace_path": str(trace_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[ACT-FULL] summary={summary_path}", flush=True)


def main() -> int:
    try:
        args = parse_args()
        validate_args(args)
        return dry_run(args) if args.dry_run_action_ipc else real_run(args)
    except Exception as exc:
        print(f"[ACT-FULL] ABORT | {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
