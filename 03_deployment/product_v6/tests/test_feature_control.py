"""Offline feature-integration semantics; no test opens XR, RPC, CAN, or cameras."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

PRODUCT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PRODUCT_ROOT / "collector"
sys.path.insert(0, str(ROOT))

import config
from data_logger import DataLogger
from gripper_io import GripperIO, READY_KEEPALIVE_S, _ready_send_flags, trigger_to_position
from home_motion import HomeMotionSession
from jogany_io import build_home_payload, strict_hold
from teleop import (ACTIVE, HOLD, HOME_JOINT_RAD, EpisodeRecorder, HomeRuntime,
                    RealRobotSession, XrPose, _scaled_rotation,
                    begin_or_continue_home_brake, enter_home_settle,
                    home_settle_ready, home_start_allowed,
                    reset_planner_after_home_pass, settle_update,
                    training_state_action, update_target_velocity)


def safe_topic(q: np.ndarray | None = None, *, moving: bool = False, global_count: int = 1) -> SimpleNamespace:
    q = HOME_JOINT_RAD if q is None else np.asarray(q, dtype=float)
    return SimpleNamespace(
        q_actual_rad=q.copy(), q_target_rad=q.copy(),
        dq_actual_rad_s=np.full(6, config.ROBOT_STATIONARY_LIMIT_RAD_S * 2 if moving else 0.0),
        enabled=(True,) * 6, joint_error_code=(0,) * 6, topic_age_ms=0.0,
        controller_link_up=True, model_state=0, model_error_code=0,
        global_count=global_count,
    )


class FakeTransport:
    def __init__(self, q: np.ndarray) -> None:
        self.last_accepted_command_q = tuple(q)
        self.submissions: list[np.ndarray] = []

    def submit_main(self, _text: str, q: np.ndarray, _velocity: np.ndarray) -> SimpleNamespace:
        accepted = np.asarray(q, dtype=float).copy()
        self.last_accepted_command_q = tuple(accepted)
        self.submissions.append(accepted)
        return SimpleNamespace(submitted=True)


class FakeCamera:
    def get_compressed_frames(self) -> dict[str, dict[str, object]]:
        return {
            serial: {"color": b"jpeg", "timestamp": 1.0, "frame_number": 1}
            for serial in config.CAM_SERIAL_DICT.values()
        }


class FeatureIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(execute_motion=True)
        self.transport = FakeTransport(HOME_JOINT_RAD)

    def test_home_gate_rejects_active_arm_and_grip_high(self) -> None:
        topic = safe_topic()
        self.assertFalse(home_start_allowed(self.args, self.transport, ACTIVE, False, 0.0, topic))
        self.assertFalse(home_start_allowed(self.args, self.transport, HOLD, False, 1.0, topic))

    def test_home_gate_rejects_moving_or_too_far_robot(self) -> None:
        self.assertFalse(home_start_allowed(self.args, self.transport, HOLD, False, 0.0, safe_topic(moving=True)))
        far = HOME_JOINT_RAD + np.deg2rad(np.full(6, config.HOME_MAX_DELTA_DEG + 0.01))
        self.assertFalse(home_start_allowed(self.args, self.transport, HOLD, False, 0.0, safe_topic(far)))

    def test_home_gate_allows_stationary_hold_low_grip_within_configured_distance(self) -> None:
        near = HOME_JOINT_RAD + np.deg2rad(np.full(6, config.HOME_MAX_DELTA_DEG - 1e-6))
        self.assertTrue(home_start_allowed(self.args, self.transport, HOLD, False, config.GRIP_THRESHOLD, safe_topic(near)))

    def test_home_starts_from_last_accepted_transport_endpoint_and_updates_it(self) -> None:
        start = HOME_JOINT_RAD + np.deg2rad(np.full(6, 5.0))
        transport = FakeTransport(start)
        session = HomeMotionSession(transport.last_accepted_command_q, HOME_JOINT_RAD,
                                    config.HOME_SPEED_RAD_S, config.HOME_RAMP_DURATION_S, config.CONTROL_HZ)
        while not session.motion_done:
            command = session.next_target()
            payload = build_home_payload(command, transport.last_accepted_command_q)
            self.assertTrue(transport.submit_main(payload.text, command, payload.joint_vel).submitted)
        self.assertTrue(np.array_equal(transport.submissions[-1], HOME_JOINT_RAD))
        self.assertTrue(np.array_equal(np.asarray(transport.last_accepted_command_q), HOME_JOINT_RAD))

    def test_home_uses_same_transport_and_never_moveabsj(self) -> None:
        source = (ROOT / "teleop.py").read_text(encoding="utf-8")
        self.assertIn("transport.submit_main", source)
        self.assertNotIn("MoveAbsJ", source)
        self.assertNotIn("MoveJ", source)

    def test_home_pass_clears_arm_session_and_requires_low_before_next_grip(self) -> None:
        source = (ROOT / "teleop.py").read_text(encoding="utf-8")
        self.assertIn("reset_planner_after_home_pass(planner, home, controller)", source)
        self.assertIn("home = None; arm_low_seen = False", source)
        self.assertIn("if controller.state == HOLD and not arm_low_seen:", source)

    def test_trigger_absolute_mapping_and_latest_target_reverse(self) -> None:
        self.assertEqual(trigger_to_position(0.0), 100.0)
        self.assertEqual(trigger_to_position(1.0), 0.0)
        self.assertEqual(trigger_to_position(0.35), 65.0)
        gripper = GripperIO(vel=1.0, force=1.0, acc=1.0, dec=1.0)
        self.assertTrue(gripper.set_trigger(0.2))
        self.assertAlmostEqual(gripper._latest_target, 80.0)
        self.assertTrue(gripper.set_trigger(0.8))
        self.assertAlmostEqual(gripper._latest_target, 20.0)
        self.assertTrue(gripper.set_trigger(0.1))
        self.assertAlmostEqual(gripper._latest_target, 90.0)

    def test_omnipicker_ready_keepalive_and_real_actual_command_are_distinct(self) -> None:
        self.assertEqual(READY_KEEPALIVE_S, 0.25)
        self.assertEqual(_ready_send_flags(42.0, 42.0, 1.0, 1.25), (False, True, True))
        topic = safe_topic(np.arange(6, dtype=float))
        state, action = training_state_action(topic, np.arange(10, 16, dtype=float), {"actual_pos": 17.0, "target_pos": 83.0})
        np.testing.assert_array_equal(state, np.array((0., 1., 2., 3., 4., 5., 17.)))
        np.testing.assert_array_equal(action, np.array((10., 11., 12., 13., 14., 15., 83.)))

    def test_recording_b_start_stop_save_and_right_click_discard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            logger = DataLogger(directory); recorder = EpisodeRecorder(logger, FakeCamera())
            self.assertEqual(recorder.buttons(True, False)[0], "START")
            self.assertTrue(recorder.active)
            self.assertEqual(recorder.buttons(True, False)[0], "")
            self.assertEqual(recorder.buttons(False, False)[0], "")
            self.assertEqual(recorder.buttons(True, False)[0], "STOP+SAVE")
            self.assertFalse(recorder.active)
            self.assertEqual(recorder.buttons(False, False)[0], "")
            self.assertEqual(recorder.buttons(True, False)[0], "START")
            self.assertEqual(recorder.buttons(False, True)[0], "DISCARD")
            self.assertFalse(recorder.active)

    def test_recording_directory_is_04_local_and_action_is_absolute_last_command(self) -> None:
        self.assertEqual(config.RAW_DATA_DIR, PRODUCT_ROOT / "data_teleop" / "pick_place_object_v1" / "raw_data")
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeRecorder(DataLogger(directory), FakeCamera())
            recorder.buttons(True, False)
            topic = safe_topic(np.zeros(6))
            accepted = np.full(6, 0.123)
            self.assertTrue(recorder.record(1.0, topic, accepted, {"actual_pos": 22.0, "target_pos": 77.0}))
            entry = recorder.logger.log_data[0]
            np.testing.assert_array_equal(entry["q_command"], accepted)
            self.assertEqual(entry["gripper_command_pos"], 77.0)
            self.assertEqual(entry["topic_global_count"], topic.global_count)

    def test_camera_is_single_owner_and_preview_isolated_from_arm_fault(self) -> None:
        preview = (ROOT / "camera_preview.py").read_text(encoding="utf-8")
        realsense = (ROOT / "realsense.py").read_text(encoding="utf-8")
        teleop = (ROOT / "teleop.py").read_text(encoding="utf-8")
        self.assertNotIn("pyrealsense2", preview)
        self.assertNotIn("RealSenseCameraInterface", preview)
        self.assertIn("threading.Thread(target=camera_loop", teleop)
        self.assertEqual(realsense.count("rs.pipeline()"), 1)
        self.assertIn("[PREVIEW] stopped:", preview)

    def test_home_runtime_has_only_home_specific_brake_phase(self) -> None:
        runtime = HomeRuntime(HomeMotionSession(HOME_JOINT_RAD, HOME_JOINT_RAD,
                                                config.HOME_SPEED_RAD_S, config.HOME_RAMP_DURATION_S, config.CONTROL_HZ))
        self.assertEqual(runtime.phase, "MOTION")
        source = (ROOT / "teleop.py").read_text(encoding="utf-8")
        self.assertIn('phase == "BRAKE"', source)
        self.assertNotIn('HOLD, ACTIVE, BRAKE', source)

    def test_regrip_anchor_stays_zero_after_hold(self) -> None:
        session = RealRobotSession()
        pose = XrPose(np.array((1., 0., 0.)), np.eye(3))
        session.consume(1.0, pose, 1, "FRESH")
        session.consume(0.0, pose, 1, "FRESH")
        anchor = session.consume(1.0, XrPose(np.array((9., 0., 0.)), np.eye(3)), 2, "FRESH")
        np.testing.assert_array_equal(anchor.delta_position, np.zeros(3))


class RegressionFixTests(unittest.TestCase):
    @staticmethod
    def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
        x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        c, s, d = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
        return np.array(((c+x*x*d, x*y*d-z*s, x*z*d+y*s),
                         (y*x*d+z*s, c+y*y*d, y*z*d-x*s),
                         (z*x*d-y*s, z*y*d+x*s, c+z*z*d)))

    def make_home(self) -> HomeRuntime:
        return HomeRuntime(HomeMotionSession(HOME_JOINT_RAD, HOME_JOINT_RAD,
                                             config.HOME_SPEED_RAD_S,
                                             config.HOME_RAMP_DURATION_S,
                                             config.CONTROL_HZ))

    def test_scaled_positive_x_rotation_has_correct_rodrigues_signs(self) -> None:
        scaled_angle = math.radians(10.0) * config.ROTATION_SCALE
        result = _scaled_rotation(self.axis_rotation(np.array((1., 0., 0.)), math.radians(10.0)))
        self.assertAlmostEqual(result[1, 2], -math.sin(scaled_angle), places=12)
        self.assertAlmostEqual(result[2, 1], math.sin(scaled_angle), places=12)

    def test_arbitrary_axis_rotation_scales_angle_and_remains_so3(self) -> None:
        input_angle = math.radians(37.0)
        result = _scaled_rotation(self.axis_rotation(np.array((1., -2., 3.)), input_angle))
        output_angle = math.acos(float(np.clip((np.trace(result) - 1.0) * 0.5, -1.0, 1.0)))
        self.assertAlmostEqual(output_angle, input_angle * config.ROTATION_SCALE, places=12)
        np.testing.assert_allclose(result.T @ result, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(result)), 1.0, places=12)

    def test_target_velocity_uses_unique_global_count_and_product_period(self) -> None:
        first = safe_topic(np.zeros(6), global_count=10)
        previous_q, previous_count, latest, latest_ns = update_target_velocity(None, None, None, None, first, 100)
        second = safe_topic(np.full(6, 0.002), global_count=12)
        _, count, velocity, velocity_ns = update_target_velocity(previous_q, previous_count, latest, latest_ns, second, 200)
        np.testing.assert_allclose(velocity, np.full(6, 0.5))
        self.assertEqual(count, 12)
        self.assertEqual(velocity_ns, 200)

    def test_home_motion_end_has_no_fixed_cycle_completion(self) -> None:
        source = (ROOT / "teleop.py").read_text(encoding="utf-8")
        self.assertNotIn("home_brake_cycles", source)
        self.assertIn("fresh_target_velocity(latest_target_dq, latest_target_ns, now_ns)", source)
        self.assertIn("begin_or_continue_home_brake(home, transport, target_dq", source)

    def test_high_target_velocity_continues_home_brake_segments(self) -> None:
        home = self.make_home(); transport = FakeTransport(HOME_JOINT_RAD)
        begin_or_continue_home_brake(home, transport, np.full(6, 0.05), 1_000_000_000, "HOME_END")
        first_brake, total_start = home.brake, home.brake_started_ns
        self.assertEqual(home.phase, "BRAKE")
        transport.last_accepted_command_q = tuple(HOME_JOINT_RAD + 0.001)
        begin_or_continue_home_brake(home, transport, np.full(6, 0.03), 1_100_000_000, "HOME_BRAKE_CONTINUE")
        self.assertEqual(home.phase, "BRAKE")
        self.assertIsNot(home.brake, first_brake)
        self.assertEqual(home.brake_started_ns, total_start)

    def test_low_target_velocity_freezes_last_accepted_home_hold_q(self) -> None:
        home = self.make_home(); transport = FakeTransport(HOME_JOINT_RAD + 0.004)
        enter_home_settle(home, transport, 2_000_000_000)
        self.assertEqual(home.phase, "SETTLE")
        np.testing.assert_array_equal(home.hold_q, np.asarray(transport.last_accepted_command_q))
        payload = strict_hold(home.hold_q)
        np.testing.assert_array_equal(payload.joint_pos, home.hold_q)
        np.testing.assert_array_equal(payload.joint_vel, np.zeros(6))

    def test_settle_requires_target_and_actual_velocity_and_unique_samples(self) -> None:
        low, high = np.zeros(6), np.full(6, config.BRAKE_SETTLE_VELOCITY_RAD_S * 2)
        last, count, target_ok, actual_ok = settle_update(None, 10, 0, high, low)
        self.assertTrue(actual_ok and not target_ok); self.assertEqual(count, 0)
        last, count, target_ok, actual_ok = settle_update(last, 11, count, low, high)
        self.assertTrue(target_ok and not actual_ok); self.assertEqual(count, 0)
        last, count, _, _ = settle_update(last, 12, count, low, low)
        self.assertEqual(count, 1)
        _, duplicate_count, _, _ = settle_update(last, 12, count, low, low)
        self.assertEqual(duplicate_count, 1)

    def test_home_settle_needs_hold_time_and_three_unique_topic_samples(self) -> None:
        home = self.make_home(); transport = FakeTransport(HOME_JOINT_RAD)
        enter_home_settle(home, transport, 1_000_000_000)
        for count in (1, 2):
            self.assertFalse(home_settle_ready(home, safe_topic(global_count=count), np.zeros(6),
                                               1_000_000_000 + int(1.1e9)))
        self.assertTrue(home_settle_ready(home, safe_topic(global_count=3), np.zeros(6), 2_100_000_000))

    def test_home_pass_resets_planner_to_home_hold_q_not_motion_endpoint(self) -> None:
        home = self.make_home(); home.hold_q = HOME_JOINT_RAD + 0.007
        planner = SimpleNamespace(reset_to=lambda q: setattr(planner, "reset_q", np.asarray(q).copy()))
        controller = RealRobotSession(); controller.state = ACTIVE
        reset_planner_after_home_pass(planner, home, controller)
        np.testing.assert_array_equal(planner.reset_q, home.hold_q)
        self.assertEqual(controller.state, HOLD)
        self.assertIsNone(controller.previous_pose)

    def test_normal_release_path_does_not_call_home_brake_helpers(self) -> None:
        source = (ROOT / "teleop.py").read_text(encoding="utf-8")
        normal = source[source.index("if home is None and not fault:"):source.index("if recorder is not None:")]
        self.assertIn("build_release_payload(planner.q_commit)", normal)
        self.assertNotIn("build_brake(", normal)
        self.assertNotIn("sample_brake(", normal)


if __name__ == "__main__":
    unittest.main()
