"""Offline regression tests; no Quest SDK initialization or real-robot access."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from placo_ik import IncrementalMujocoPlacoPlanner
from teleop_mujoco import ACTIVE, FRESH, HOLD, MujocoTeleopSession, XrPose, get_joint_qpos_indices


class IncrementalBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.q_initial = np.deg2rad(config.INITIAL_JOINTS_DEG)

    def test_01_mujoco_model_load_and_required_names(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(config.MUJOCO_URDF_PATH))
        self.assertEqual(len(get_joint_qpos_indices(model)), 6)
        self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_Link"), 0)

    def test_02_placo_startup_and_fk_are_finite(self) -> None:
        planner = IncrementalMujocoPlacoPlanner(self.q_initial)
        self.assertTrue(np.all(np.isfinite(planner.q_commit)))
        self.assertTrue(np.all(np.isfinite(planner.current_tcp())))

    def test_03_zero_delta_does_not_drift(self) -> None:
        planner = IncrementalMujocoPlacoPlanner(self.q_initial)
        pending = planner.propose(np.zeros(3), np.eye(3))
        self.assertIsNotNone(pending, planner.last_failure)
        assert pending is not None
        np.testing.assert_allclose(pending["q_candidate"], planner.q_commit, atol=1e-8)

    def test_04_grip_first_frame_is_anchor(self) -> None:
        session = MujocoTeleopSession()
        first = session.consume(1.0, XrPose(np.array((1.0, 0.0, 0.0)), np.eye(3)), 1, FRESH)
        self.assertEqual(session.state, ACTIVE)
        self.assertTrue(first.state_changed)
        np.testing.assert_array_equal(first.delta_position, np.zeros(3))

    def test_05_strict_no_backlog_after_rejected_frame(self) -> None:
        session = MujocoTeleopSession()
        planner = IncrementalMujocoPlacoPlanner(self.q_initial)
        anchor = XrPose(np.zeros(3), np.eye(3))
        moved = XrPose(np.array((0.05, 0.0, 0.0)), np.eye(3))
        session.consume(1.0, anchor, 1, FRESH)
        rejected = session.consume(1.0, moved, 2, FRESH)
        self.assertTrue(rejected.xr_new_frame)
        # Force an actual PlaCo transaction failure after the session has consumed
        # the XR frame.  Its motion must not reappear on the next frame.
        with patch.object(planner, "_solve_once", side_effect=RuntimeError("forced solver failure")):
            self.assertIsNone(planner.propose(rejected.delta_position, rejected.delta_rotation))
        np.testing.assert_array_equal(planner.q_commit, self.q_initial)
        zero_next = session.consume(1.0, moved, 3, FRESH)
        np.testing.assert_allclose(zero_next.delta_position, np.zeros(3), atol=1e-12)
        self.assertAlmostEqual(zero_next.delta_rotation_deg, 0.0)

    def test_06_release_regrip_does_not_follow_old_target(self) -> None:
        session = MujocoTeleopSession()
        session.consume(1.0, XrPose(np.array((1.0, 0.0, 0.0)), np.eye(3)), 1, FRESH)
        release = session.consume(0.0, None, 1, FRESH)
        regrip = session.consume(1.0, XrPose(np.array((9.0, 0.0, 0.0)), np.eye(3)), 2, FRESH)
        next_frame = session.consume(1.0, XrPose(np.array((9.01, 0.0, 0.0)), np.eye(3)), 3, FRESH)
        self.assertEqual(release.transition, "ACTIVE -> HOLD (RELEASE)")
        self.assertEqual(session.state, ACTIVE)
        np.testing.assert_array_equal(regrip.delta_position, np.zeros(3))
        self.assertGreater(float(np.linalg.norm(next_frame.delta_position)), 0.0)


if __name__ == "__main__":
    unittest.main()
