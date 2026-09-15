"""Pure-offline checks for ACTIVE-session wrist rotation analysis."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analyze_wrist_rotation as analyzer


def record(timestamp: float, transition: str = "", position: list[float] | None = None,
           tcp: list[float] | None = None, rotation: list[list[float]] | None = None,
           corrected: list[float] | None = None) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "transition": transition,
        "xr_position": position,
        "xr_corrected_position": position if corrected is None else corrected,
        "mujoco_tcp_position": tcp,
        "xr_rotation": rotation,
    }


class WristRotationAnalysisTests(unittest.TestCase):
    def test_sessions_have_independent_origins_and_rotation_angle(self) -> None:
        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        quarter_turn_z = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        entries = [
            record(0.0),
            record(1.0, "HOLD -> ACTIVE", [1.0, 2.0, 3.0], [0.1, 0.2, 0.3], identity),
            record(2.0, "", [1.01, 2.02, 3.0], [0.101, 0.2, 0.3], quarter_turn_z),
            record(3.0, "ACTIVE -> HOLD (RELEASE)"),
            record(10.0, "HOLD -> ACTIVE", [9.0, 9.0, 9.0], [0.5, 0.5, 0.5], identity),
            record(12.0, "", [9.003, 9.0, 9.0], [0.5, 0.502, 0.5], identity),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "synthetic.jsonl"
            log_path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
            sessions, skipped = analyzer.read_active_sessions(log_path)
        self.assertEqual(skipped, 0)
        self.assertEqual([len(session) for session in sessions], [2, 2])
        first_rows, first = analyzer.analyze_session(1, sessions[0], "yaw")
        second_rows, second = analyzer.analyze_session(2, sessions[1], "yaw")
        self.assertAlmostEqual(first["quest_raw"]["x_range_mm"], 10.0)
        self.assertAlmostEqual(first["quest_raw"]["y_range_mm"], 20.0)
        self.assertAlmostEqual(first["quest_corrected"]["x_range_mm"], 10.0)
        self.assertAlmostEqual(first["rotation"]["max_relative_rotation_deg"], 90.0)
        self.assertAlmostEqual(second["quest_raw"]["x_range_mm"], 3.0)
        overall = analyzer.overall_summary([first, second], skipped)
        self.assertEqual(overall["sessions"], 2)
        self.assertAlmostEqual(overall["total_active_duration_s"], 3.0)
        self.assertAlmostEqual(overall["quest_raw_max_across_sessions"]["x_range_mm"], 10.0)
        self.assertTrue(math.isfinite(overall["mujoco_tcp_max_across_sessions"]["max_3d_drift_mm"]))
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            log_path = root / "synthetic.jsonl"
            log_path.write_text("{}\n", encoding="utf-8")
            with patch.object(analyzer, "__file__", str(root / "analyze_wrist_rotation.py")):
                csv_path, summary_path, _ = analyzer.write_outputs(
                    log_path, "yaw", first_rows + second_rows, [first, second], skipped
                )
            self.assertEqual(csv_path.parent, root / "analysis")
            self.assertTrue(csv_path.is_file())
            self.assertTrue(summary_path.is_file())
            self.assertIn("xr_drift_3d_mm", csv_path.read_text(encoding="utf-8").splitlines()[0])


if __name__ == "__main__":
    unittest.main()
