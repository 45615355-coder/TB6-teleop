"""Offline product-v4 HOME trajectory, gate, BRAKE, and completion checks."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np


PRODUCT_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = PRODUCT_ROOT / "collector"
sys.path.insert(0, str(COLLECTOR))

import config  # noqa: E402
from home_motion import HomeMotionSession  # noqa: E402
from jogany_io import build_home_payload  # noqa: E402
from teleop import (ACTIVE, HOLD, HOME_JOINT_RAD, HomeRuntime,
                    begin_or_continue_home_brake, button_rising,
                    create_home_runtime, home_final_position_pass,
                    home_start_allowed)  # noqa: E402


class FakeTransport:
    def __init__(self, q_start: np.ndarray) -> None:
        self.last_accepted_command_q = tuple(q_start)
        self.submissions: list[np.ndarray] = []

    def submit_main(self, _text: str, q_command: np.ndarray, _velocity: np.ndarray) -> SimpleNamespace:
        accepted = np.asarray(q_command, dtype=float).copy()
        self.last_accepted_command_q = tuple(accepted)
        self.submissions.append(accepted)
        return SimpleNamespace(submitted=True)


def safe_topic(q_actual: np.ndarray, *, moving: bool = False) -> SimpleNamespace:
    q = np.asarray(q_actual, dtype=float)
    return SimpleNamespace(
        q_actual_rad=q.copy(),
        q_target_rad=q.copy(),
        dq_actual_rad_s=np.full(6, 0.001 if moving else 0.0),
        topic_age_ms=0.0,
        controller_link_up=True,
        enabled=(True,) * 6,
        model_state=0,
        model_error_code=0,
        joint_error_code=(0,) * 6,
        global_count=1,
    )


class ProductV4HomeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(execute_motion=True)

    def test_0_5_10_15_degree_home_trajectories_respect_configured_speed(self) -> None:
        for delta_deg in (0.0, 5.0, 10.0, 15.0):
            session = HomeMotionSession(
                HOME_JOINT_RAD + np.deg2rad(np.full(6, delta_deg)),
                HOME_JOINT_RAD,
                config.HOME_SPEED_RAD_S,
                config.HOME_RAMP_DURATION_S,
                config.CONTROL_HZ,
            )
            targets = [session.next_target() for _ in range(session.motion_frames)]
            self.assertTrue(session.motion_done)
            np.testing.assert_array_equal(targets[-1], HOME_JOINT_RAD)
            self.assertLessEqual(session.planned_peak_velocity_rad_s,
                                 config.HOME_SPEED_RAD_S + 1e-12)

    def test_existing_home_motion_uses_one_transport_and_updates_accepted_command(self) -> None:
        start = HOME_JOINT_RAD + np.deg2rad(np.full(6, 5.0))
        session = HomeMotionSession(start, HOME_JOINT_RAD, config.HOME_SPEED_RAD_S,
                                    config.HOME_RAMP_DURATION_S, config.CONTROL_HZ)
        transport = FakeTransport(start)
        while not session.motion_done:
            command = session.next_target()
            payload = build_home_payload(command, transport.last_accepted_command_q)
            self.assertTrue(transport.submit_main(payload.text, payload.joint_pos,
                                                  payload.joint_vel).submitted)
        np.testing.assert_array_equal(np.asarray(transport.last_accepted_command_q),
                                      HOME_JOINT_RAD)

    def test_a_is_rising_edge_only(self) -> None:
        previous = False
        edges = []
        for current in (False, True, True, True, False, True):
            edges.append(button_rising(current, previous))
            previous = current
        self.assertEqual(edges, [False, True, False, False, False, True])

    def test_home_gate_requires_hold_low_grip_stationary_and_near_home(self) -> None:
        transport = FakeTransport(HOME_JOINT_RAD)
        healthy = safe_topic(HOME_JOINT_RAD)
        self.assertTrue(home_start_allowed(self.args, transport, HOLD, False,
                                           config.GRIP_THRESHOLD, healthy))
        self.assertFalse(home_start_allowed(self.args, transport, ACTIVE, False, 0.0, healthy))
        self.assertFalse(home_start_allowed(self.args, transport, HOLD, False, 1.0, healthy))
        self.assertFalse(home_start_allowed(self.args, transport, HOLD, False, 0.0,
                                            safe_topic(HOME_JOINT_RAD, moving=True)))
        far = HOME_JOINT_RAD + np.deg2rad(np.full(6, config.HOME_MAX_DELTA_DEG + 0.01))
        self.assertFalse(home_start_allowed(self.args, transport, HOLD, False, 0.0,
                                            safe_topic(far)))

    def test_home_sync_bypass_and_reject_thresholds_wrap_existing_home(self) -> None:
        bypass, reason = create_home_runtime(
            HOME_JOINT_RAD,
            HOME_JOINT_RAD + np.deg2rad(np.full(6, config.HOME_SYNC_BYPASS_GAP_DEG)),
        )
        self.assertEqual(reason, "")
        self.assertTrue(bypass.sync_bypassed)
        self.assertEqual(bypass.phase, "MOTION")
        rejected, reason = create_home_runtime(
            HOME_JOINT_RAD,
            HOME_JOINT_RAD + np.deg2rad(np.full(6, config.HOME_SYNC_MAX_INITIAL_GAP_DEG + 0.01)),
        )
        self.assertEqual(reason, "sync gap too large")
        self.assertIsNone(rejected.sync_session)

    def test_original_home_brake_uses_mature_segment_continuation(self) -> None:
        home = HomeRuntime(HomeMotionSession(
            HOME_JOINT_RAD, HOME_JOINT_RAD, config.HOME_SPEED_RAD_S,
            config.HOME_RAMP_DURATION_S, config.CONTROL_HZ,
        ))
        transport = FakeTransport(HOME_JOINT_RAD)
        begin_or_continue_home_brake(home, transport, np.full(6, 0.05),
                                     1_000_000_000, "HOME_END")
        self.assertEqual(home.phase, "BRAKE")
        first_start = home.brake_started_ns
        begin_or_continue_home_brake(home, transport, np.full(6, 0.03),
                                     1_100_000_000, "HOME_BRAKE_CONTINUE")
        self.assertEqual(home.phase, "BRAKE")
        self.assertEqual(home.brake_started_ns, first_start)

    def test_final_position_tolerance_failure_is_nonfatal_boolean(self) -> None:
        passed, error = home_final_position_pass(
            HOME_JOINT_RAD + np.deg2rad(np.full(6, config.HOME_FINAL_TOLERANCE_DEG + 0.01))
        )
        self.assertFalse(passed)
        self.assertGreater(error, config.HOME_FINAL_TOLERANCE_DEG)

    def test_normal_release_and_home_brake_remain_separate(self) -> None:
        source = (COLLECTOR / "teleop.py").read_text(encoding="utf-8")
        normal = source[source.index("if home is None and not fault:"):
                        source.index("if recorder is not None:")]
        self.assertIn("build_release_payload(planner.q_commit)", normal)
        self.assertNotIn("build_brake(", normal)
        self.assertIn('elif phase == "BRAKE":', source)
        self.assertIn("sample_brake(home.brake, now_ns)", source)


if __name__ == "__main__":
    unittest.main()
