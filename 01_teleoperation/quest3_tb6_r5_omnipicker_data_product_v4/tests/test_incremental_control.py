from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

PRODUCT_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_ROOT = PRODUCT_ROOT / "collector"
sys.path.insert(0, str(COLLECTOR_ROOT / "vendor" / "topic"))
sys.path.insert(0, str(COLLECTOR_ROOT))

import config
from jogany_io import LiveTransport, build_active_payload, build_release_payload
from placo_ik import IncrementalRealRobotPlacoPlanner
from teleop import (ACTIVE, FRESH, HOLD, STALE, LOCAL_OPERATOR_EXIT, ROBOT_SIDE_STOP,
                    RealRobotSession, XrPose, shutdown_release_cycles,
                    shutdown_release_if_local_and_healthy)
from vendor.topic.platform_loader import get_topic_module

SEED = np.deg2rad(np.array((90.0, -90.0, 90.0, -90.0, -90.0, 0.0)))


def healthy_topic() -> SimpleNamespace:
    return SimpleNamespace(
        q_actual_rad=np.zeros(6), q_target_rad=np.zeros(6), dq_actual_rad_s=np.zeros(6),
        enabled=(True,) * 6, joint_error_code=(0,) * 6, topic_age_ms=0.0,
        controller_link_up=True, model_state=0, model_error_code=0,
    )


class FakeReader:
    def __init__(self, topic: SimpleNamespace) -> None: self.topic = topic
    def latest(self) -> SimpleNamespace: return self.topic


class FakeTransport:
    def __init__(self, endpoint: np.ndarray, *, fatal: str = "") -> None:
        self.last_accepted_command_q = tuple(endpoint)
        self.fatal_reason = fatal
        self.submissions: list[tuple[str, np.ndarray, np.ndarray]] = []
    def submit_main(self, text: str, position: np.ndarray, velocity: np.ndarray) -> SimpleNamespace:
        self.submissions.append((text, position.copy(), velocity.copy()))
        return SimpleNamespace(submitted=True)


class TestRealRobotV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None: cls.planner = IncrementalRealRobotPlacoPlanner(SEED)
    def setUp(self) -> None: self.planner.reset_to(SEED)

    def test_frozen_03_speed_config_is_the_sole_placo_limit(self) -> None:
        self.assertEqual(config.JOINT_SPEED_LIMIT_RAD_S, 0.2)
        self.assertEqual(config.MAX_ALLOWED_JOINT_SPEED_RAD_S, 0.25)
        pending = self.planner.propose(np.zeros(3), np.eye(3))
        self.assertIsNotNone(pending, self.planner.last_failure)
        self.assertLessEqual(float(np.max(np.abs(pending["joint_velocity"]))), config.JOINT_SPEED_LIMIT_RAD_S + 1e-9)

    def test_no_external_acceptance_layer_and_command_is_candidate(self) -> None:
        self.assertFalse(hasattr(self.planner, "_accept"))
        pending = self.planner.propose(np.zeros(3), np.eye(3))
        np.testing.assert_array_equal(pending["q_command"], pending["q_candidate"])
        self.assertNotIn("q_accepted", pending)

    def test_zero_delta_does_not_reuse_old_target(self) -> None:
        self.planner.commit(self.planner.propose(np.zeros(3), np.eye(3)))
        before = self.planner.q_commit.copy()
        second = self.planner.propose(np.zeros(3), np.eye(3))
        np.testing.assert_allclose(second["q_command"], before, atol=1e-8)

    def test_release_payload_is_fixed_endpoint_zero_velocity_and_configured_deceleration(self) -> None:
        payload = build_release_payload(SEED)
        np.testing.assert_allclose(payload.joint_pos, SEED)
        np.testing.assert_array_equal(payload.joint_vel, np.zeros(6))
        np.testing.assert_array_equal(payload.joint_acc, np.zeros(6))
        np.testing.assert_allclose(payload.joint_dec, np.full(6, config.RELEASE_DECEL_RAD_S2))

    def test_active_payload_validates_without_clamping(self) -> None:
        with self.assertRaises(ValueError): build_active_payload(SEED + np.array((0.01, 0, 0, 0, 0, 0)), SEED)

    def test_regrip_first_frame_is_anchor_with_no_motion_increment(self) -> None:
        session = RealRobotSession(); pose = XrPose(np.array((1., 0., 0.)), np.eye(3))
        first = session.consume(1.0, pose, 1, FRESH)
        release = session.consume(0.0, pose, 1, FRESH)
        regrip = session.consume(1.0, XrPose(np.array((9., 0., 0.)), np.eye(3)), 2, FRESH)
        self.assertEqual(first.transition, "HOLD → ACTIVE")
        self.assertEqual(release.transition, "ACTIVE → HOLD (RELEASE)")
        self.assertTrue(regrip.state_changed and regrip.xr_new_frame)
        np.testing.assert_array_equal(regrip.delta_position, np.zeros(3))

    def test_regrip_second_frame_can_immediately_produce_reverse_intent(self) -> None:
        session = RealRobotSession(); session.consume(1.0, XrPose(np.array((1., 0., 0.)), np.eye(3)), 1, FRESH)
        session.consume(0.0, None, 1, FRESH)
        session.consume(1.0, XrPose(np.array((3., 0., 0.)), np.eye(3)), 2, FRESH)
        second = session.consume(1.0, XrPose(np.array((2., 0., 0.)), np.eye(3)), 3, FRESH)
        self.assertEqual(session.state, ACTIVE)
        self.assertTrue(second.xr_new_frame and not second.state_changed)
        self.assertNotEqual(float(np.linalg.norm(second.delta_position)), 0.0)

    def test_stale_drops_session_without_brake_state(self) -> None:
        session = RealRobotSession(); session.consume(1.0, XrPose(np.array((1., 0., 0.)), np.eye(3)), 1, FRESH)
        stale = session.consume(1.0, None, 1, STALE)
        self.assertEqual(stale.transition, "ACTIVE → HOLD (XR_STALE)")
        self.assertEqual(session.state, HOLD)
        self.assertIsNone(session.previous_pose)

    def test_commit_uses_command_not_topic_actual(self) -> None:
        pending = self.planner.propose(np.zeros(3), np.eye(3)); self.planner.commit(pending)
        np.testing.assert_array_equal(self.planner.q_commit, pending["q_command"])

    def test_vendored_topic_module_loads_without_starting_subscriber(self) -> None:
        module = get_topic_module()
        self.assertTrue(hasattr(module, "start_subscriber"))

    def test_shutdown_cycle_count_covers_theoretical_controller_deceleration(self) -> None:
        self.assertEqual(shutdown_release_cycles(), 16)
        self.assertGreaterEqual(
            shutdown_release_cycles() * config.DT,
            config.JOINT_SPEED_LIMIT_RAD_S / config.RELEASE_DECEL_RAD_S2,
        )

    def test_local_exit_releases_last_accepted_endpoint_with_zero_velocity_and_configured_deceleration(self) -> None:
        endpoint = SEED + 0.001
        transport, reader = FakeTransport(endpoint), FakeReader(healthy_topic())
        with patch("teleop.time.sleep"):
            sent = shutdown_release_if_local_and_healthy(LOCAL_OPERATOR_EXIT, transport, reader)
        self.assertEqual(sent, shutdown_release_cycles())
        self.assertEqual(len(transport.submissions), shutdown_release_cycles())
        for _, position, velocity in transport.submissions:
            np.testing.assert_allclose(position, endpoint)
            np.testing.assert_array_equal(velocity, np.zeros(6))
        self.assertIn("--joint_acc={0,0,0,0,0,0}", transport.submissions[0][0])
        self.assertIn("--joint_dec={0.8,0.8,0.8,0.8,0.8,0.8}", transport.submissions[0][0])

    def test_robot_side_stop_submits_no_shutdown_release_payload(self) -> None:
        transport, topic = FakeTransport(SEED), healthy_topic()
        topic.controller_link_up = False
        with patch("teleop.time.sleep"):
            sent = shutdown_release_if_local_and_healthy(ROBOT_SIDE_STOP, transport, FakeReader(topic))
        self.assertEqual(sent, 0)
        self.assertEqual(transport.submissions, [])

    def test_rpc_fatal_submits_no_shutdown_release_payload(self) -> None:
        transport, reader = FakeTransport(SEED, fatal="callback failure"), FakeReader(healthy_topic())
        with patch("teleop.time.sleep"):
            sent = shutdown_release_if_local_and_healthy(LOCAL_OPERATOR_EXIT, transport, reader)
        self.assertEqual(sent, 0)
        self.assertEqual(transport.submissions, [])

    def test_transport_termination_is_local_cleanup_not_stop_rpc(self) -> None:
        native = SimpleNamespace(Msg=lambda *_: self.fail("termination must not create an RPC message"))
        client = object()
        transport = LiveTransport(native, client)
        self.assertIsNone(transport.terminate_once())
        self.assertIsNone(transport.client)
        self.assertFalse(transport.rpc_stream_ready)
