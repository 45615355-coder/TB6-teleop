"""Non-blocking OmniPicker I/O for the Quest3/PlaCo product.

This module deliberately owns only the USB-CAN gripper connection.  Quest SDK
access remains in ``teleop.py`` so the gripper worker never creates a second
XR client or blocks the 50 Hz arm-control loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import threading
import time
from typing import Any, Callable


# These are hardware/transport details, rather than user tuning parameters.
CAN_ID = 0x01
DEVICE_TYPE = 3
DEVICE_INDEX = 0
CHANNEL = 0
BAUD = 0x1400

POSITION_DEADBAND_PERCENT = 1.0
MAX_SEND_HZ = 20.0
READY_KEEPALIVE_S = 0.25
FEEDBACK_TIMEOUT_MS = 20
IDLE_WAIT_S = 0.01
INITIAL_OPEN_POSITION = 100.0
INITIAL_OPEN_READY_POSITION = 98.0
INITIAL_OPEN_RESEND_S = 0.20
INITIAL_OPEN_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class GripperSnapshot:
    """Latest worker-owned gripper state; values are percentages when present."""

    target_pos: float | None = None
    actual_pos: float | None = None
    actual_vel: float | None = None
    actual_force: float | None = None
    state: int | None = None
    fault: int | None = None
    ready: bool = False
    error: str | None = None
    sent_at_s: float | None = None
    feedback_at_s: float | None = None


def trigger_to_position(trigger: float) -> float:
    """Map a validated Quest Trigger value to OmniPicker absolute position."""
    return 100.0 * (1.0 - trigger)


def _ready_send_flags(
    target: float | None,
    last_sent_pos: float | None,
    last_sent_s: float,
    now: float,
) -> tuple[bool, bool, bool]:
    """Return READY changed/rate/keepalive gates without communicating with CAN."""
    changed = target is not None and (
        last_sent_pos is None
        or abs(target - last_sent_pos) >= POSITION_DEADBAND_PERCENT
    )
    send_ready = now - last_sent_s >= 1.0 / MAX_SEND_HZ
    keepalive_due = (
        target is not None
        and last_sent_pos is not None
        and not changed
        and now - last_sent_s >= READY_KEEPALIVE_S
    )
    return changed, send_ready, keepalive_due


class GripperIO:
    """INITIAL_OPEN then latest-target-wins OmniPicker worker.

    ``set_trigger`` only validates and stores one latest target under a lock.
    CAN send/receive calls run exclusively in the daemon worker thread.
    """

    def __init__(
        self,
        *,
        vel: float,
        force: float,
        acc: float,
        dec: float,
        driver_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._vel, self._force, self._acc, self._dec = vel, force, acc, dec
        self._driver_factory = driver_factory or self._default_driver_factory
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._shutdown_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_target: float | None = None
        self._fault_latched: int | None = None
        self._initialization_failed = False
        self._snapshot = GripperSnapshot()
        self._last_invalid_warning_s = float("-inf")

    @staticmethod
    def _default_driver_factory() -> Any:
        # Keep the USB-CAN dynamic-library load and device open out of import
        # time so offline tests and product module smoke imports need no CAN.
        from OmniPicker_Interface import OmniPicker_Interface

        return OmniPicker_Interface(
            Can_ID=CAN_ID,
            device_type=DEVICE_TYPE,
            device_index=DEVICE_INDEX,
            channel=CHANNEL,
            baud=BAUD,
        )

    def start(self) -> None:
        """Start I/O asynchronously. Calling this more than once is harmless."""
        with self._lock:
            if self._thread is not None:
                return
            self._stop.clear()
            self._shutdown_requested.clear()
            self._thread = threading.Thread(target=self._run, name="omnipicker-io", daemon=True)
            self._thread.start()

    def set_trigger(self, trigger_value: Any, *, xr_fresh: bool = True) -> bool:
        """Publish one fresh Trigger target without sending CAN from this caller.

        Values in [-0.05, 1.05] are accepted then clamped.  Larger or non-finite
        values leave the previous valid target untouched, avoiding a spurious
        open/close command from malformed XR data.
        """
        if not xr_fresh:
            return False
        try:
            trigger = float(trigger_value)
        except (TypeError, ValueError):
            self._warn_invalid("not numeric")
            return False
        if not math.isfinite(trigger):
            self._warn_invalid("not finite")
            return False
        if not -0.05 <= trigger <= 1.05:
            self._warn_invalid(f"outside accepted range: {trigger:.3f}")
            return False
        target = trigger_to_position(min(1.0, max(0.0, trigger)))
        with self._lock:
            if self._fault_latched is not None or self._initialization_failed:
                return False
            self._latest_target = target
        self._wake.set()
        return True

    def get_snapshot(self) -> dict[str, float | int | str | None]:
        """Return a copy of the latest state without communicating with CAN."""
        with self._lock:
            return asdict(self._snapshot)

    def shutdown(self, timeout_s: float = 1.0) -> None:
        """Request a worker-owned best-effort current-position hold, then close."""
        # The caller only requests shutdown and waits.  Every CAN call below is
        # performed by _run's worker thread, never by teleop's 50 Hz thread.
        self._shutdown_requested.set()
        self._stop.set()
        self._wake.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_s)
        with self._lock:
            self._thread = None

    def _warn_invalid(self, reason: str) -> None:
        now = time.monotonic()
        with self._lock:
            if now - self._last_invalid_warning_s < 1.0:
                return
            self._last_invalid_warning_s = now
        print(f"[GRIPPER] ignored invalid Trigger ({reason})", flush=True)

    def _update_snapshot(self, **changes: Any) -> None:
        with self._lock:
            self._snapshot = GripperSnapshot(**(asdict(self._snapshot) | changes))

    def _latest(self) -> float | None:
        with self._lock:
            return self._latest_target

    def _next_command_target(self) -> float | None:
        """Return no normal motion target once a device fault has latched."""
        with self._lock:
            return None if self._fault_latched is not None else self._latest_target

    def _fault(self) -> int | None:
        with self._lock:
            return self._fault_latched

    def _fail_initial_open(self, reason: str) -> None:
        with self._lock:
            self._initialization_failed = True
        self._update_snapshot(ready=False, error=f"INITIAL_OPEN failed: {reason}")

    def _initial_open_ready(self, actual_pos: float | None) -> bool:
        """Accept the verified near-open feedback tolerance, not strict 100%."""
        with self._lock:
            return (
                self._fault_latched is None
                and actual_pos is not None
                and actual_pos >= INITIAL_OPEN_READY_POSITION
                and self._snapshot.state == 0
            )

    def _initial_open(self, driver: Any) -> tuple[float, float] | None:
        """Open to 100% in the worker before releasing normal Trigger control."""
        deadline = time.monotonic() + INITIAL_OPEN_TIMEOUT_S
        last_sent_s = float("-inf")
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= deadline:
                self._fail_initial_open("timeout")
                print("[GRIPPER] INITIAL_OPEN failed: timeout", flush=True)
                return None
            if now - last_sent_s >= INITIAL_OPEN_RESEND_S:
                driver.control(Pos=INITIAL_OPEN_POSITION, Vel=self._vel, For=self._force, Acc=self._acc, Dec=self._dec)
                last_sent_s = now
                self._update_snapshot(target_pos=INITIAL_OPEN_POSITION, sent_at_s=now, ready=False, error=None)

            actual_pos = self._record_feedback(driver.read_feedback(timeout_ms=FEEDBACK_TIMEOUT_MS), time.monotonic())
            if self._fault() is not None:
                self._fail_initial_open("device fault latched")
                print("[GRIPPER] INITIAL_OPEN failed: device fault latched", flush=True)
                return None
            if self._initial_open_ready(actual_pos):
                self._update_snapshot(ready=True, error=None)
                return INITIAL_OPEN_POSITION, last_sent_s
            self._wake.wait(IDLE_WAIT_S)
            self._wake.clear()
        return None

    def _record_feedback(self, feedback: Any, feedback_at_s: float) -> float | None:
        """Store valid feedback and return its usable actual position, if any."""
        if feedback is None:
            return None
        try:
            fault, state, actual_pos, actual_vel, actual_force = feedback
            fault = int(fault)
            state = int(state)
            newly_latched = False
            with self._lock:
                if fault != 0 and self._fault_latched is None:
                    self._fault_latched, newly_latched = fault, True
                latched_fault = self._fault_latched
            if newly_latched:
                print(f"[GRIPPER] FAULT latched: fault={fault}", flush=True)
            actual_pos = float(actual_pos)
            actual_vel = float(actual_vel)
            actual_force = float(actual_force)
            if not all(math.isfinite(value) for value in (actual_pos, actual_vel, actual_force)):
                if latched_fault is not None:
                    self._update_snapshot(fault=latched_fault, state=state, error=f"OmniPicker fault={latched_fault} (latched)")
                return None
            self._update_snapshot(
                fault=fault if latched_fault is None else latched_fault, state=state, actual_pos=actual_pos,
                actual_vel=actual_vel, actual_force=actual_force,
                feedback_at_s=feedback_at_s,
                error=None if latched_fault is None else f"OmniPicker fault={latched_fault} (latched)",
            )
            return actual_pos if 0.0 <= actual_pos <= 100.0 else None
        except (TypeError, ValueError, OverflowError):
            return None

    def _best_effort_hold(self, driver: Any) -> None:
        """Hold a freshly-read actual position without issuing an extreme fallback."""
        actual_pos = None
        try:
            actual_pos = self._record_feedback(driver.read_feedback(timeout_ms=FEEDBACK_TIMEOUT_MS), time.monotonic())
        except Exception as error:
            print(f"[GRIPPER] shutdown feedback read failed: {type(error).__name__}: {error}", flush=True)

        # OmniPicker can be quiet unless it receives a command. Re-send only
        # its latest requested target to solicit fresh feedback; never invent a
        # full-open or full-close target during shutdown.
        target = self._latest()
        if target is not None:
            try:
                driver.control(Pos=target, Vel=self._vel, For=self._force, Acc=self._acc, Dec=self._dec)
                self._update_snapshot(target_pos=target, sent_at_s=time.monotonic())
                actual_pos = self._record_feedback(driver.read_feedback(timeout_ms=FEEDBACK_TIMEOUT_MS), time.monotonic())
            except Exception as error:
                print(f"[GRIPPER] shutdown feedback refresh failed: {type(error).__name__}: {error}", flush=True)

        if actual_pos is None:
            print("[GRIPPER] shutdown hold skipped: no valid actual_pos", flush=True)
            return
        try:
            driver.control(Pos=actual_pos, Vel=self._vel, For=self._force, Acc=self._acc, Dec=self._dec)
            self._update_snapshot(target_pos=actual_pos, sent_at_s=time.monotonic())
        except Exception as error:
            print(f"[GRIPPER] shutdown hold send failed: {type(error).__name__}: {error}", flush=True)

    def _run(self) -> None:
        driver: Any = None
        last_sent_pos: float | None = None
        last_sent_s = float("-inf")
        phase = "INITIAL_OPEN"
        try:
            driver = self._driver_factory()
            initial_open = self._initial_open(driver)
            if initial_open is None:
                return
            last_sent_pos, last_sent_s = initial_open
            phase = "READY"
            while not self._stop.is_set():
                target = self._next_command_target()
                now = time.monotonic()
                changed, send_ready, keepalive_due = _ready_send_flags(
                    target, last_sent_pos, last_sent_s, now
                )
                if send_ready and (changed or keepalive_due):
                    # A single absolute target lets the gripper controller run
                    # autonomously; re-sending it also solicits fresh feedback.
                    driver.control(Pos=target, Vel=self._vel, For=self._force, Acc=self._acc, Dec=self._dec)
                    last_sent_pos, last_sent_s = target, now
                    self._update_snapshot(target_pos=target, sent_at_s=now, error=None)

                # This potentially blocking read is intentionally worker-only.
                # state=2 (STALL) remains recorded feedback, never an arm stop.
                self._record_feedback(driver.read_feedback(timeout_ms=FEEDBACK_TIMEOUT_MS), time.monotonic())
                self._wake.wait(IDLE_WAIT_S)
                self._wake.clear()
        except Exception as error:
            # Do not replace the last target or send a fallback position here.
            if phase == "INITIAL_OPEN":
                self._fail_initial_open(f"{type(error).__name__}: {error}")
                print(f"[GRIPPER] INITIAL_OPEN failed: {type(error).__name__}: {error}", flush=True)
            else:
                self._update_snapshot(error=f"{type(error).__name__}: {error}")
                print(f"[GRIPPER] I/O worker stopped: {type(error).__name__}: {error}", flush=True)
        finally:
            if driver is not None:
                try:
                    if self._shutdown_requested.is_set():
                        self._best_effort_hold(driver)
                    driver.shutdown()
                except Exception as error:
                    self._update_snapshot(error=f"shutdown {type(error).__name__}: {error}")
