"""Offline tests for the Product v4 Pro wrist-pivot input correction."""

from __future__ import annotations

import json
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


PRODUCT_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_ROOT = PRODUCT_ROOT / "collector"
sys.path.insert(0, str(COLLECTOR_ROOT))

import config  # noqa: E402
from teleop import FRESH, RealRobotSession, XrPose, parse_args, run  # noqa: E402
from wrist_pivot import (WristPivotProfileError, apply_wrist_pivot,
                         load_wrist_pivot_profile)  # noqa: E402


EXPECTED_OFFSET = np.array((
    0.050824526646232444,
    -0.031242958885661656,
    0.08492180130925026,
))


class WristPivotTests(unittest.TestCase):
    def test_product_local_valid_profile_loads_expected_offset(self) -> None:
        offset = load_wrist_pivot_profile(
            config.WRIST_PIVOT_PROFILE, config.WRIST_PIVOT_MAX_OFFSET_M
        )
        np.testing.assert_array_equal(offset, EXPECTED_OFFSET)
        self.assertLessEqual(float(np.linalg.norm(offset)), config.WRIST_PIVOT_MAX_OFFSET_M)
        self.assertEqual(config.WRIST_PIVOT_PROFILE, PRODUCT_ROOT / "calibration" / "current_user.json")

    def test_known_rotation_math_is_position_plus_rotated_offset(self) -> None:
        position = np.array((0.4, -0.2, 0.8))
        rotation = np.array(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)))
        offset = np.array((0.1, 0.2, 0.3))
        corrected, _ = apply_wrist_pivot(position, rotation, offset, gain=1.0)
        np.testing.assert_allclose(corrected, position + rotation @ offset, atol=1e-12)

    def test_rotation_is_returned_untouched(self) -> None:
        rotation = np.array(((0., 0., 1.), (1., 0., 0.), (0., 1., 0.)))
        _, corrected_rotation = apply_wrist_pivot(np.zeros(3), rotation, EXPECTED_OFFSET)
        self.assertIs(corrected_rotation, rotation)
        np.testing.assert_array_equal(corrected_rotation, rotation)

    def test_constant_orientation_preserves_full_translation_delta(self) -> None:
        rotation = np.eye(3)
        first, _ = apply_wrist_pivot(np.zeros(3), rotation, EXPECTED_OFFSET)
        second, _ = apply_wrist_pivot(np.array((0.10, 0., 0.)), rotation, EXPECTED_OFFSET)
        np.testing.assert_allclose(second - first, np.array((0.10, 0., 0.)), atol=1e-12)

    def test_regrip_first_corrected_frame_is_zero_delta_anchor(self) -> None:
        session = RealRobotSession()
        old_position, old_rotation = apply_wrist_pivot(np.zeros(3), np.eye(3), EXPECTED_OFFSET)
        session.consume(1.0, XrPose(old_position, old_rotation), 1, FRESH)
        session.consume(0.0, XrPose(old_position, old_rotation), 2, FRESH)

        angle = np.deg2rad(90.0)
        new_rotation = np.array(((np.cos(angle), -np.sin(angle), 0.),
                                 (np.sin(angle), np.cos(angle), 0.),
                                 (0., 0., 1.)))
        new_position, unchanged_rotation = apply_wrist_pivot(
            np.array((10., -4., 2.)), new_rotation, EXPECTED_OFFSET
        )
        event = session.consume(1.0, XrPose(new_position, unchanged_rotation), 3, FRESH)
        np.testing.assert_array_equal(event.delta_position, np.zeros(3))
        np.testing.assert_array_equal(event.delta_rotation, np.eye(3))
        self.assertTrue(event.xr_new_frame)
        self.assertTrue(event.state_changed)

    def test_missing_and_invalid_profiles_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            with self.assertRaises(WristPivotProfileError):
                load_wrist_pivot_profile(path, 0.20)

            invalid_profiles = (
                "{bad json",
                json.dumps({"valid": False, "offset_m": [0., 0., 0.]}),
                json.dumps({"valid": True, "offset_m": [0., 0.]}),
                json.dumps({"valid": True, "offset_m": [0.21, 0., 0.]}),
                '{"valid": true, "offset_m": [NaN, 0.0, 0.0]}',
            )
            for content in invalid_profiles:
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(WristPivotProfileError):
                        load_wrist_pivot_profile(path, 0.20)

    def test_enabled_missing_profile_fails_before_hardware_import_or_start(self) -> None:
        args = parse_args((
            "--execute-motion",
            "--confirm-controller-ready",
            "--confirm-robot-enabled",
            "--confirm-robot-stationary",
        ))
        missing = PRODUCT_ROOT / "calibration" / "does_not_exist.json"
        with patch.object(config, "WRIST_PIVOT_PROFILE", missing):
            with self.assertRaises(WristPivotProfileError):
                run(args)

    def test_raw_validation_and_freshness_precede_control_pose_correction(self) -> None:
        source = inspect.getsource(run)
        validation = source.index("raw_pose = validate_xr_pose(raw)")
        freshness = source.index("monitor.observe(stamp, started, raw_pose is not None)")
        correction = source.index("position, rotation = apply_wrist_pivot(")
        consumption = source.index("controller.consume(grip, pose, stamp, xr_status)")
        self.assertLess(validation, freshness)
        self.assertLess(freshness, correction)
        self.assertLess(correction, consumption)


if __name__ == "__main__":
    unittest.main()
