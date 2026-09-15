"""Synthetic offline regression tests for wrist-pivot calibration and profiles."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wrist_pivot import (WristPivotProfileError, apply_wrist_pivot, corrected_wrist_pose,
                         load_wrist_pivot_profile, solve_wrist_pivot)
import config
import calibrate_wrist_pivot as calibration
from teleop_mujoco import active_wrist_pivot_offset


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


class WristPivotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.offset = np.array((0.037, -0.033, 0.062))
        cls.center = np.array((0.3, -0.2, 0.5))
        cls.rotations = np.asarray([
            np.eye(3), rotation_x(0.5), rotation_y(-0.7), rotation_z(0.9),
            rotation_z(-0.4) @ rotation_y(0.6), rotation_x(-0.8) @ rotation_z(0.3),
        ])
        cls.positions = np.asarray([cls.center - rotation @ cls.offset for rotation in cls.rotations])

    def test_a_known_offset_is_recovered(self) -> None:
        result = solve_wrist_pivot(self.positions, self.rotations, max_offset_m=0.20, max_rms_error_m=0.015)
        self.assertTrue(result["valid"])
        self.assertEqual(result["rank"], 3)
        np.testing.assert_allclose(result["offset_m"], self.offset, atol=1e-10)

    def test_b_correction_keeps_synthetic_pivot_fixed(self) -> None:
        pivots = np.asarray([apply_wrist_pivot(position, rotation, self.offset) for position, rotation in zip(self.positions, self.rotations)])
        np.testing.assert_allclose(pivots, np.repeat(self.center[None, :], len(pivots), axis=0), atol=1e-10)

    def test_c_correction_does_not_change_rotation(self) -> None:
        position, rotation = self.positions[1], self.rotations[1]
        corrected_position, corrected_rotation = corrected_wrist_pose(position, rotation, self.offset)
        self.assertFalse(np.array_equal(corrected_position, position))
        np.testing.assert_array_equal(corrected_rotation, rotation)
        self.assertIsNot(corrected_rotation, rotation)

    def test_d_rank_deficient_calibration_is_invalid(self) -> None:
        positions = np.array(((0.0, 0.0, 0.0), (0.001, 0.0, 0.0), (0.002, 0.0, 0.0)))
        rotations = np.repeat(np.eye(3)[None, :, :], len(positions), axis=0)
        result = solve_wrist_pivot(positions, rotations, max_offset_m=0.20, max_rms_error_m=0.015)
        self.assertFalse(result["valid"])
        self.assertLess(result["rank"], 3)

    def test_e_profile_loading_accepts_valid_and_rejects_missing_or_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            valid_path = directory / "valid.json"
            valid_path.write_text(json.dumps({"valid": True, "offset_m": self.offset.tolist()}), encoding="utf-8")
            loaded = load_wrist_pivot_profile(valid_path, max_offset_m=0.20)
            np.testing.assert_array_equal(loaded["offset_m"], self.offset)
            with patch.object(config, "WRIST_PIVOT_PROFILE", valid_path), patch.object(config, "WRIST_PIVOT_ENABLED", True):
                np.testing.assert_array_equal(active_wrist_pivot_offset(), self.offset)
            with self.assertRaises(WristPivotProfileError):
                load_wrist_pivot_profile(directory / "missing.json", max_offset_m=0.20)
            with patch.object(config, "WRIST_PIVOT_PROFILE", directory / "missing.json"), patch.object(config, "WRIST_PIVOT_ENABLED", True):
                with self.assertRaisesRegex(RuntimeError, "enabled but unavailable"):
                    active_wrist_pivot_offset()
            invalid_path = directory / "invalid.json"
            invalid_path.write_text(json.dumps({"valid": False, "offset_m": self.offset.tolist()}), encoding="utf-8")
            with self.assertRaises(WristPivotProfileError):
                load_wrist_pivot_profile(invalid_path, max_offset_m=0.20)

    def test_f_invalid_calibration_never_overwrites_current_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            current = Path(temporary_directory) / "current_user.json"
            current.write_text('{"valid": true, "marker": "keep"}\n', encoding="utf-8")
            invalid = {"valid": False, "offset_m": [0.0, 0.0, 0.0], "offset_norm_m": 0.0,
                       "rms_error_m": 1.0, "samples_total": 90, "samples_yaw": 30,
                       "samples_pitch": 30, "samples_roll": 30, "rank": 0, "condition": None}
            with patch.object(config, "WRIST_PIVOT_PROFILE", current):
                backup, profile = calibration.save_result(invalid)
            self.assertTrue(backup.is_file())
            self.assertIsNone(profile)
            self.assertEqual(json.loads(current.read_text(encoding="utf-8"))["marker"], "keep")

    def test_g_capture_records_each_quest_timestamp_once(self) -> None:
        rotation = np.eye(3)
        samples = [
            (np.array((100.0, 0.0, 0.0)), rotation, 1.0, 100),
            (np.array((100.0, 0.0, 0.0)), rotation, 1.0, 100),
            (np.array((100.0, 0.0, 0.0)), rotation, 1.0, 100),
            (np.array((101.0, 0.0, 0.0)), rotation, 1.0, 101),
            (np.array((101.0, 0.0, 0.0)), rotation, 1.0, 101),
            (np.array((102.0, 0.0, 0.0)), rotation, 1.0, 102),
            # Release must end capture even though this repeats timestamp 102.
            (np.array((102.0, 0.0, 0.0)), rotation, 0.0, 102),
        ]
        with patch.object(calibration, "read_pose", side_effect=samples), \
                patch.object(calibration.time, "sleep"), \
                patch.object(config, "WRIST_PIVOT_MIN_SAMPLES_PER_DIRECTION", 3):
            positions, rotations = calibration.capture_direction("YAW", 1, calibration.FreshMonitor())
        self.assertEqual(len(positions), 3)
        self.assertEqual(len(rotations), 3)
        self.assertEqual([position[0] for position in positions], [100.0, 101.0, 102.0])


if __name__ == "__main__":
    unittest.main()
