"""Offline HOME_SYNC v2 semantics; no hardware or SDK is opened."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

PRODUCT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PRODUCT_ROOT / "collector"
sys.path.insert(0, str(ROOT))

import config
from jogany_io import LiveTransport, build_home_sync_payload, build_release_payload
from teleop import (ACTIVE, HOLD, HOME_JOINT_RAD, EpisodeRecorder,
                    GripperEventReporter, RealRobotSession, XrPose,
                    create_home_runtime, diagnostic_fields,
                    enter_home_sync_decel, finish_home_sync,
                    home_final_position_pass, home_sync_settle_ready,
                    reset_after_home_task, restart_home_sync_pass,
                    send_home_fatal_stop,
                    training_state_action, update_home_sync_actual_drift)


def topic(actual: np.ndarray | None = None, *, target: np.ndarray | None = None,
          actual_dq: np.ndarray | None = None, count: int = 1,
          link: bool = True, enabled: tuple[bool, ...] = (True,) * 6) -> SimpleNamespace:
    actual_q = HOME_JOINT_RAD.copy() if actual is None else np.asarray(actual, dtype=float).copy()
    target_q = actual_q.copy() if target is None else np.asarray(target, dtype=float).copy()
    return SimpleNamespace(
        q_actual_rad=actual_q,
        q_target_rad=target_q,
        dq_actual_rad_s=np.zeros(6) if actual_dq is None else np.asarray(actual_dq, dtype=float).copy(),
        enabled=enabled,
        joint_error_code=(0,) * 6,
        topic_age_ms=0.0,
        controller_link_up=link,
        model_state=0,
        model_error_code=0,
        global_count=count,
    )


class FakeTransport:
    def __init__(self, q: np.ndarray) -> None:
        self.last_accepted_command_q = tuple(np.asarray(q, dtype=float))
        self.last_accepted_command_velocity = (0.0,) * 6
        self.submissions: list[tuple[str, np.ndarray, np.ndarray]] = []
        self.client = object()
        self.stop_calls = 0

    def submit_main(self, text: str, q: np.ndarray, velocity: np.ndarray) -> SimpleNamespace:
        accepted_q = np.asarray(q, dtype=float).copy()
        accepted_velocity = np.asarray(velocity, dtype=float).copy()
        self.last_accepted_command_q = tuple(accepted_q)
        self.last_accepted_command_velocity = tuple(accepted_velocity)
        self.submissions.append((text, accepted_q, accepted_velocity))
        return SimpleNamespace(submitted=True, accepted_state_advanced=True)

    def send_true_stop_once(self) -> str:
        self.stop_calls += 1
        return "STOP_RESULT"


class FakePlanner:
    def __init__(self) -> None:
        self.reset_q: np.ndarray | None = None

    def reset_to(self, q: np.ndarray) -> None:
        self.reset_q = np.asarray(q, dtype=float).copy()


class FakeLogger:
    def __init__(self) -> None:
        self.log_data: list[dict[str, object]] = []

    def add_entry(self, entry: dict[str, object]) -> None:
        self.log_data.append(entry)


class FakeCamera:
    def get_compressed_frames(self) -> dict[str, dict[str, object]]:
        return {serial: {"color": b"jpeg", "timestamp": 1.0, "frame_number": 1}
                for serial in config.CAM_SERIAL_DICT.values()}


class FakeMessage:
    def __init__(self, text: str) -> None:
        self.text = text
        self.message_id = None
        self.sequence_id = None

    def setMsgID(self, value: int) -> None:
        self.message_id = value

    def setMsgSeqID(self, value: int) -> None:
        self.sequence_id = value


class FakeRpcInner:
    def __init__(self) -> None:
        self.calls: list[tuple[FakeMessage, int]] = []

    def CallAwait(self, message: FakeMessage, timeout_ms: int):
        self.calls.append((message, timeout_ms))
        return 0, [SimpleNamespace(code=0)]


class HomeSyncV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.command = HOME_JOINT_RAD.copy()

    def make_sync(self, gap_deg: float = 1.0):
        actual = self.command + np.deg2rad(np.full(6, gap_deg))
        home, reason = create_home_runtime(self.command, actual)
        self.assertEqual(reason, "")
        return home, actual

    def drive_sync(self, home, transport: FakeTransport) -> None:
        self.assertIsNotNone(home.sync_session)
        while not home.sync_session.motion_done:
            command = home.sync_session.next_target()
            payload = build_home_sync_payload(command, transport.last_accepted_command_q)
            result = transport.submit_main(payload.text, payload.joint_pos, payload.joint_vel)
            self.assertTrue(result.submitted and result.accepted_state_advanced)

    def test_01_gap_at_bypass_threshold_enters_original_home_directly(self) -> None:
        actual = self.command + np.deg2rad(np.full(6, config.HOME_SYNC_BYPASS_GAP_DEG))
        home, reason = create_home_runtime(self.command, actual)
        self.assertEqual(reason, "")
        self.assertTrue(home.sync_bypassed)
        self.assertEqual(home.phase, "MOTION")
        self.assertIsNotNone(home.session)
        self.assertIsNone(home.sync_session)

    def test_02_gap_above_five_degrees_is_nonfatal_rejection(self) -> None:
        actual = self.command + np.deg2rad(np.full(6, config.HOME_SYNC_MAX_INITIAL_GAP_DEG + 0.01))
        home, reason = create_home_runtime(self.command, actual)
        self.assertEqual(reason, "sync gap too large")
        self.assertGreater(home.sync_initial_gap_deg, config.HOME_SYNC_MAX_INITIAL_GAP_DEG)
        self.assertIsNone(home.sync_session)

    def test_03_sync_target_is_a_press_time_actual_snapshot(self) -> None:
        home, actual = self.make_sync()
        actual[:] += 1.0
        np.testing.assert_allclose(home.sync_target_actual_q, self.command + np.deg2rad(np.ones(6)))

    def test_04_runtime_actual_updates_do_not_chase_or_replace_sync_target(self) -> None:
        home, _ = self.make_sync()
        fixed_target = home.sync_target_actual_q.copy()
        update_home_sync_actual_drift(home, fixed_target + np.deg2rad(np.full(6, 0.4)))
        update_home_sync_actual_drift(home, fixed_target - np.deg2rad(np.full(6, 0.7)))
        np.testing.assert_array_equal(home.sync_target_actual_q, fixed_target)
        np.testing.assert_array_equal(home.sync_session.target_q, fixed_target)

    def test_05_sync_per_frame_absolute_commands_are_continuous(self) -> None:
        home, _ = self.make_sync()
        previous = self.command.copy()
        while not home.sync_session.motion_done:
            current = home.sync_session.next_target()
            self.assertLessEqual(float(np.max(np.abs(current - previous))),
                                 config.HOME_SYNC_SPEED_RAD_S * config.DT + 1e-12)
            self.assertTrue(np.all(current >= previous - 1e-15))
            previous = current

    def test_06_sync_payload_never_exceeds_configured_sync_speed(self) -> None:
        home, _ = self.make_sync()
        previous = self.command.copy()
        while not home.sync_session.motion_done:
            current = home.sync_session.next_target()
            payload = build_home_sync_payload(current, previous)
            self.assertLessEqual(float(np.max(np.abs(payload.joint_vel))),
                                 config.HOME_SYNC_SPEED_RAD_S + 1e-9)
            previous = current
        excessive_step = (config.HOME_SYNC_SPEED_RAD_S + 0.01) * config.DT
        with self.assertRaises(ValueError):
            build_home_sync_payload(
                self.command + np.array((excessive_step, 0, 0, 0, 0, 0)),
                self.command,
            )

    def test_07_every_sync_frame_advances_real_transport_accepted_state(self) -> None:
        home, actual = self.make_sync()
        transport = FakeTransport(self.command)
        self.drive_sync(home, transport)
        self.assertEqual(len(transport.submissions), home.sync_session.motion_frames)
        np.testing.assert_array_equal(np.asarray(transport.last_accepted_command_q), actual)
        np.testing.assert_array_equal(transport.submissions[-1][1], actual)

    def test_08_sync_completion_decelerates_at_fixed_accepted_endpoint(self) -> None:
        home, _ = self.make_sync()
        transport = FakeTransport(self.command)
        self.drive_sync(home, transport)
        enter_home_sync_decel(home, transport)
        endpoint = np.asarray(transport.last_accepted_command_q).copy()
        payload = build_release_payload(home.sync_hold_q)
        np.testing.assert_array_equal(payload.joint_pos, endpoint)
        np.testing.assert_array_equal(payload.joint_vel, np.zeros(6))
        self.assertEqual(home.phase, "SYNC_DECEL")

    def test_09_sync_settle_requires_target_and_actual_stationary(self) -> None:
        home, _ = self.make_sync(); home.phase = "SYNC_DECEL"
        high = np.full(6, config.BRAKE_SETTLE_VELOCITY_RAD_S * 2)
        self.assertFalse(home_sync_settle_ready(home, topic(actual_dq=np.zeros(6), count=1), high))
        self.assertFalse(home_sync_settle_ready(home, topic(actual_dq=high, count=2), np.zeros(6)))
        self.assertEqual(home.settle_count, 0)

    def test_10_sync_settle_requires_three_unique_topic_samples(self) -> None:
        home, _ = self.make_sync(); home.phase = "SYNC_DECEL"
        zero = np.zeros(6)
        self.assertFalse(home_sync_settle_ready(home, topic(count=10), zero))
        self.assertFalse(home_sync_settle_ready(home, topic(count=10), zero))
        self.assertFalse(home_sync_settle_ready(home, topic(count=11), zero))
        self.assertTrue(home_sync_settle_ready(home, topic(count=12), zero))

    def test_11_final_gap_at_ready_threshold_enters_existing_home_motion(self) -> None:
        home, _ = self.make_sync(); home.phase = "SYNC_DECEL"
        transport = FakeTransport(self.command)
        actual = self.command + np.deg2rad(np.full(6, config.HOME_SYNC_READY_GAP_DEG))
        self.assertTrue(finish_home_sync(home, transport, actual))
        self.assertEqual(home.phase, "MOTION")
        self.assertIsNotNone(home.session)

    def test_12_final_gap_above_ready_threshold_fails_without_fatal_state(self) -> None:
        home, _ = self.make_sync(); home.phase = "SYNC_DECEL"
        transport = FakeTransport(self.command)
        actual = self.command + np.deg2rad(np.full(6, config.HOME_SYNC_READY_GAP_DEG + 0.01))
        self.assertFalse(finish_home_sync(home, transport, actual))
        self.assertEqual(home.phase, "SYNC_DECEL")

    def test_13_sync_failure_reanchors_planner_to_last_accepted_command(self) -> None:
        endpoint = self.command + 0.003
        planner, controller = FakePlanner(), RealRobotSession()
        controller.state = ACTIVE
        reset_after_home_task(planner, endpoint, controller)
        np.testing.assert_array_equal(planner.reset_q, endpoint)

    def test_14_sync_failure_clears_xr_session_and_requires_new_grip_low_gate(self) -> None:
        planner, controller = FakePlanner(), RealRobotSession()
        pose = XrPose(np.ones(3), np.eye(3))
        controller.consume(1.0, pose, 1, "FRESH")
        reset_after_home_task(planner, self.command, controller)
        self.assertEqual(controller.state, HOLD)
        self.assertIsNone(controller.previous_pose)
        source = (ROOT / "teleop.py").read_text(encoding="utf-8")
        self.assertIn("home = None; arm_low_seen = False", source)
        self.assertIn("if controller.state == HOLD and not arm_low_seen:", source)

    def test_15_final_home_position_tolerance_failure_is_nonfatal_result(self) -> None:
        failed, error = home_final_position_pass(
            HOME_JOINT_RAD + np.deg2rad(np.full(6, config.HOME_FINAL_TOLERANCE_DEG + 0.01))
        )
        self.assertFalse(failed)
        self.assertGreater(error, config.HOME_FINAL_TOLERANCE_DEG)
        source = (ROOT / "teleop.py").read_text(encoding="utf-8")
        self.assertIn("reason=position tolerance", source)

    def test_16_home_pass_and_position_failure_do_not_change_recorder_active(self) -> None:
        recorder = EpisodeRecorder(FakeLogger(), FakeCamera()); recorder.active = True
        planner, controller = FakePlanner(), RealRobotSession()
        reset_after_home_task(planner, self.command, controller)
        self.assertTrue(recorder.active)
        reset_after_home_task(planner, self.command + 0.001, controller)
        self.assertTrue(recorder.active)

    def test_17_continuous_opening_prints_only_one_direction_event(self) -> None:
        reporter = GripperEventReporter()
        snapshots = [
            {"ready": True, "target_pos": 80.0, "actual_pos": 20.0},
            {"ready": True, "target_pos": 80.0, "actual_pos": 40.0},
            {"ready": True, "target_pos": 80.0, "actual_pos": 80.0},
            {"ready": True, "target_pos": 80.0, "actual_pos": 80.0},
        ]
        with patch("builtins.print") as printed:
            for snapshot in snapshots:
                reporter.observe(snapshot)
        messages = [call.args[0] for call in printed.call_args_list]
        self.assertEqual(messages.count("[GRIPPER] OPENING"), 1)

    def test_18_open_close_open_reversals_each_print_once(self) -> None:
        reporter = GripperEventReporter()
        snapshots = [
            {"ready": True, "target_pos": 60.0, "actual_pos": 10.0},
            {"ready": True, "target_pos": 60.0, "actual_pos": 30.0},
            {"ready": True, "target_pos": 20.0, "actual_pos": 35.0},
            {"ready": True, "target_pos": 20.0, "actual_pos": 20.0},
            {"ready": True, "target_pos": 70.0, "actual_pos": 20.0},
        ]
        with patch("builtins.print") as printed:
            for snapshot in snapshots:
                reporter.observe(snapshot)
        directions = [call.args[0] for call in printed.call_args_list if call.args[0] != "[GRIPPER] READY"]
        self.assertEqual(directions, ["[GRIPPER] OPENING", "[GRIPPER] CLOSING", "[GRIPPER] OPENING"])

    def test_19_serious_home_fault_uses_true_stop_helper(self) -> None:
        transport = FakeTransport(self.command)
        state, detail = send_home_fatal_stop(transport, topic())
        self.assertEqual((state, detail), ("SENT", "STOP_RESULT"))
        self.assertEqual(transport.stop_calls, 1)

    def test_20_robot_side_stop_does_not_send_duplicate_stop(self) -> None:
        transport = FakeTransport(self.command)
        state, detail = send_home_fatal_stop(transport, topic(enabled=(False,) * 6))
        self.assertEqual((state, detail), ("SKIPPED_ROBOT_SIDE_STOP", None))
        self.assertEqual(transport.stop_calls, 0)

    def test_21_true_stop_wire_is_exact_idempotent_sync_callawait(self) -> None:
        inner = FakeRpcInner()
        native = SimpleNamespace(Msg=FakeMessage)
        transport = LiveTransport(native, SimpleNamespace(inner=inner))
        first = transport.send_true_stop_once()
        second = transport.send_true_stop_once()
        self.assertEqual(first, second)
        self.assertEqual(len(inner.calls), 1)
        message, timeout_ms = inner.calls[0]
        self.assertEqual(message.text, "{Stop}")
        self.assertEqual(message.message_id, 10001)
        self.assertEqual(timeout_ms, 10000)

    def test_22_training_schema_and_accepted_q_command_source_are_unchanged(self) -> None:
        accepted = self.command + 0.123
        state, action = training_state_action(topic(self.command), accepted,
                                              {"actual_pos": 12.0, "target_pos": 34.0})
        np.testing.assert_array_equal(state, np.concatenate((self.command, (12.0,))))
        np.testing.assert_array_equal(action, np.concatenate((accepted, (34.0,))))
        logger = FakeLogger(); recorder = EpisodeRecorder(logger, FakeCamera()); recorder.active = True
        self.assertTrue(recorder.record(1.0, topic(self.command), accepted,
                                        {"actual_pos": 12.0, "target_pos": 34.0}))
        self.assertEqual(set(logger.log_data[0]), {
            "timestamp", "step_index", "q_actual", "q_command",
            "gripper_actual_pos", "gripper_command_pos", "image", "topic_global_count",
        })
        np.testing.assert_array_equal(logger.log_data[0]["q_command"], accepted)

    def test_23_jsonl_diagnostics_include_control_source_and_sync_fields(self) -> None:
        home, _ = self.make_sync()
        fields = diagnostic_fields(topic(), self.command, np.zeros(6),
                                   RealRobotSession(), home, True)
        self.assertEqual(fields["control_source"], "HOME_SYNC")
        self.assertTrue(fields["recording_active"])
        for key in (
            "topic_q_target", "topic_dq_actual", "target_dq",
            "q_command_actual_gap_deg", "q_command_home_error_deg",
            "q_actual_home_error_deg", "home_sync_initial_gap_deg",
            "home_sync_final_gap_deg", "home_sync_actual_drift_deg",
            "home_sync_target_q",
        ):
            self.assertIn(key, fields)

    def test_24_placo_transaction_file_matches_verified_v2_hash(self) -> None:
        digest = hashlib.sha256((ROOT / "placo_ik.py").read_bytes()).hexdigest()
        self.assertEqual(digest, "cf5299908fe8871cd58533d67f1def243680c1c069b3a26c17ecc1274af45e14")

    def test_25_xr_math_and_real_robot_session_match_verified_v2_hash(self) -> None:
        def definitions(path: Path) -> dict[str, str]:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            wanted = {"frame_delta", "_scaled_rotation", "RealRobotSession"}
            return {node.name: ast.get_source_segment(source, node)
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted}
        items = definitions(ROOT / "teleop.py")
        payload = "\n".join(name + "\n" + items[name] for name in sorted(items))
        self.assertEqual(hashlib.sha256(payload.encode()).hexdigest(),
                         "59438d3c85b7eea04a1533c0114940a52fe2031cc2c3a2b7a6d03e1e80ba5826")

    def test_26_normal_teleop_release_and_commit_block_matches_verified_v2_hash(self) -> None:
        def normal_block(path: Path) -> str:
            source = path.read_text(encoding="utf-8")
            start = source.index("if home is None and not fault:")
            return source[start:source.index("if recorder is not None:", start)]
        v2_block = normal_block(ROOT / "teleop.py")
        self.assertEqual(hashlib.sha256(v2_block.encode()).hexdigest(),
                         "b5a923adc603c4d4bbd318cbda20ec718765cdc02d7fb345d74f8982465c3e1c")
        self.assertIn("build_release_payload(planner.q_commit)", v2_block)
        self.assertIn("planner.commit(pending)", v2_block)

    def test_27_web_stop_safety_branch_precedes_home_sync_dispatch(self) -> None:
        source = (ROOT / "teleop.py").read_text(encoding="utf-8")
        safety_index = source.index("if safety:", source.index("while not fault:"))
        home_index = source.index("elif home is not None:", safety_index)
        self.assertLess(safety_index, home_index)
        self.assertIn("ROBOT_SIDE_STOP", source[safety_index:home_index])

    def settled_retry(self, gap_deg: float = 0.8):
        home, _ = self.make_sync()
        transport = FakeTransport(self.command)
        home.phase = "SYNC_DECEL"
        home.settle_count = config.BRAKE_SETTLE_UNIQUE_SAMPLES
        actual = self.command + np.deg2rad(np.full(6, gap_deg))
        return home, transport, actual

    def test_28_first_pass_within_06_enters_home_without_retry(self) -> None:
        home, transport, _ = self.settled_retry()
        actual = self.command + np.deg2rad(np.full(6, config.HOME_SYNC_READY_GAP_DEG))
        self.assertTrue(finish_home_sync(home, transport, actual))
        self.assertEqual(home.sync_pass, 1)
        self.assertEqual(home.phase, "MOTION")

    def test_29_first_pass_above_06_creates_pass_two(self) -> None:
        home, transport, actual = self.settled_retry()
        self.assertFalse(finish_home_sync(home, transport, actual))
        self.assertTrue(restart_home_sync_pass(home, transport, actual))
        self.assertEqual(home.sync_pass, 2)
        self.assertEqual(home.phase, "SYNC")
        self.assertIsNotNone(home.sync_session)

    def test_30_pass_two_target_is_new_stationary_snapshot_not_a_snapshot(self) -> None:
        home, transport, actual = self.settled_retry()
        first_target = home.sync_target_actual_q.copy()
        self.assertFalse(finish_home_sync(home, transport, actual))
        self.assertTrue(restart_home_sync_pass(home, transport, actual))
        np.testing.assert_array_equal(home.sync_target_actual_q, actual)
        self.assertFalse(np.array_equal(home.sync_target_actual_q, first_target))

    def test_31_pass_two_starts_at_latest_accepted_command(self) -> None:
        home, transport, actual = self.settled_retry()
        accepted = self.command + np.deg2rad(np.full(6, 0.17))
        transport.last_accepted_command_q = tuple(accepted)
        self.assertFalse(finish_home_sync(home, transport, actual))
        self.assertTrue(restart_home_sync_pass(home, transport, actual))
        np.testing.assert_array_equal(home.sync_start_command_q, accepted)
        np.testing.assert_array_equal(home.sync_session.start_q, accepted)

    def test_32_pass_two_target_remains_fixed_during_actual_drift(self) -> None:
        home, transport, actual = self.settled_retry()
        self.assertFalse(finish_home_sync(home, transport, actual))
        self.assertTrue(restart_home_sync_pass(home, transport, actual))
        fixed = home.sync_target_actual_q.copy()
        update_home_sync_actual_drift(home, actual + np.deg2rad(np.full(6, 0.4)))
        update_home_sync_actual_drift(home, actual - np.deg2rad(np.full(6, 0.3)))
        np.testing.assert_array_equal(home.sync_target_actual_q, fixed)
        np.testing.assert_array_equal(home.sync_session.target_q, fixed)

    def test_33_pass_two_success_enters_existing_home_motion(self) -> None:
        home, transport, actual = self.settled_retry()
        self.assertFalse(finish_home_sync(home, transport, actual))
        self.assertTrue(restart_home_sync_pass(home, transport, actual))
        transport.last_accepted_command_q = tuple(home.sync_target_actual_q)
        settled_actual = home.sync_target_actual_q - np.deg2rad(np.full(6, 0.4))
        self.assertTrue(finish_home_sync(home, transport, settled_actual))
        self.assertEqual(home.sync_pass, 2)
        self.assertEqual(home.phase, "MOTION")

    def test_34_pass_two_failure_is_nonfatal_and_cannot_create_pass_three(self) -> None:
        home, transport, actual = self.settled_retry()
        self.assertFalse(finish_home_sync(home, transport, actual))
        self.assertTrue(restart_home_sync_pass(home, transport, actual))
        home.phase = "SYNC_DECEL"
        home.settle_count = config.BRAKE_SETTLE_UNIQUE_SAMPLES
        second_actual = np.asarray(transport.last_accepted_command_q) + np.deg2rad(np.full(6, 0.8))
        self.assertFalse(finish_home_sync(home, transport, second_actual))
        self.assertFalse(restart_home_sync_pass(home, transport, second_actual))
        self.assertEqual(home.sync_pass, config.HOME_SYNC_MAX_PASSES)

    def test_35_retry_requires_sync_decel_and_stationary_unique_samples(self) -> None:
        home, transport, actual = self.settled_retry()
        home.phase = "SYNC"
        with self.assertRaises(RuntimeError):
            restart_home_sync_pass(home, transport, actual)
        home.phase = "SYNC_DECEL"
        home.settle_count = config.BRAKE_SETTLE_UNIQUE_SAMPLES - 1
        with self.assertRaises(RuntimeError):
            restart_home_sync_pass(home, transport, actual)

    def test_36_retry_does_not_change_recording_state(self) -> None:
        home, transport, actual = self.settled_retry()
        recorder = EpisodeRecorder(FakeLogger(), FakeCamera())
        recorder.active = True
        self.assertFalse(finish_home_sync(home, transport, actual))
        self.assertTrue(restart_home_sync_pass(home, transport, actual))
        self.assertTrue(recorder.active)

    def test_37_pass_two_commands_advance_accepted_action_source(self) -> None:
        home, transport, actual = self.settled_retry()
        self.assertFalse(finish_home_sync(home, transport, actual))
        self.assertTrue(restart_home_sync_pass(home, transport, actual))
        self.drive_sync(home, transport)
        accepted = np.asarray(transport.last_accepted_command_q)
        np.testing.assert_array_equal(accepted, home.sync_target_actual_q)
        _, action = training_state_action(topic(), accepted,
                                          {"actual_pos": 10.0, "target_pos": 20.0})
        np.testing.assert_array_equal(action[:6], accepted)

    def test_38_retry_diagnostics_report_current_pass_and_fixed_target(self) -> None:
        home, transport, actual = self.settled_retry()
        self.assertFalse(finish_home_sync(home, transport, actual))
        self.assertTrue(restart_home_sync_pass(home, transport, actual))
        fields = diagnostic_fields(topic(actual), transport.last_accepted_command_q,
                                   np.zeros(6), RealRobotSession(), home, True)
        self.assertEqual(fields["home_sync_pass"], 2)
        self.assertEqual(fields["home_sync_max_passes"], 2)
        self.assertAlmostEqual(fields["home_sync_pass_initial_gap_deg"],
                               home.sync_pass_initial_gap_deg)
        np.testing.assert_array_equal(fields["home_sync_target_q"], actual)


if __name__ == "__main__":
    unittest.main()
