"""Pure offline checks for READY same-target OmniPicker keepalive gating."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


COLLECTOR = Path(__file__).resolve().parents[1] / "collector"
sys.path.insert(0, str(COLLECTOR))

import gripper_io  # noqa: E402


class ReadyKeepaliveTests(unittest.TestCase):
    def test_unchanged_target_is_not_resent_before_keepalive(self) -> None:
        changed, send_ready, keepalive_due = gripper_io._ready_send_flags(
            42.0, 42.0, 10.0, 10.24
        )
        self.assertFalse(changed)
        self.assertTrue(send_ready)
        self.assertFalse(keepalive_due)

    def test_unchanged_target_is_resent_at_keepalive_interval(self) -> None:
        changed, send_ready, keepalive_due = gripper_io._ready_send_flags(
            42.0, 42.0, 10.0, 10.25
        )
        self.assertFalse(changed)
        self.assertTrue(send_ready)
        self.assertTrue(keepalive_due)

    def test_keepalive_restarts_its_interval_after_send(self) -> None:
        changed, send_ready, keepalive_due = gripper_io._ready_send_flags(
            42.0, 42.0, 10.25, 10.26
        )
        self.assertFalse(changed)
        self.assertFalse(keepalive_due)
        self.assertFalse(send_ready)

    def test_changed_target_uses_existing_max_send_rate_not_keepalive_rate(self) -> None:
        changed, send_ready, keepalive_due = gripper_io._ready_send_flags(
            43.0, 42.0, 10.0, 10.05
        )
        self.assertTrue(changed)
        self.assertTrue(send_ready)
        self.assertFalse(keepalive_due)

    def test_ready_command_uses_exact_latest_target(self) -> None:
        source = (COLLECTOR / "gripper_io.py").read_text(encoding="utf-8")
        self.assertIn("if send_ready and (changed or keepalive_due):", source)
        self.assertIn("driver.control(Pos=target, Vel=self._vel, For=self._force, Acc=self._acc, Dec=self._dec)", source)

    def test_latched_fault_has_no_ready_keepalive_target(self) -> None:
        gripper = gripper_io.GripperIO(vel=1.0, force=1.0, acc=1.0, dec=1.0)
        with gripper._lock:
            gripper._latest_target = 42.0
            gripper._fault_latched = 7
        self.assertIsNone(gripper._next_command_target())
        self.assertEqual(
            gripper_io._ready_send_flags(gripper._next_command_target(), 42.0, 0.0, 1.0),
            (False, True, False),
        )

    def test_initial_open_constants_and_resend_path_are_unchanged(self) -> None:
        source = (COLLECTOR / "gripper_io.py").read_text(encoding="utf-8")
        self.assertEqual(gripper_io.INITIAL_OPEN_POSITION, 100.0)
        self.assertEqual(gripper_io.INITIAL_OPEN_RESEND_S, 0.20)
        self.assertIn("driver.control(Pos=INITIAL_OPEN_POSITION", source)


if __name__ == "__main__":
    unittest.main()
