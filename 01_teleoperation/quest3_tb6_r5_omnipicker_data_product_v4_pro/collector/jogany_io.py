"""JogAny payloads: validate PlaCo velocity, never clip or rescale it."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Iterable

import numpy as np

import config
from vendor.rpc.rpc_transport import CallbackRecord, SubmitResult, VerifiedJogAnyTransport, load_vendor_rpc_modules


def _six(values: Iterable[object], name: str) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=float)
    if result.shape != (6,) or not np.all(np.isfinite(result)): raise ValueError(f"{name} must contain six finite values")
    return result


def _text(values: Iterable[object]) -> str:
    return "{" + ",".join("0" if value == 0.0 else format(float(value), ".10g") for value in _six(values, "payload")) + "}"


@dataclass(frozen=True)
class Payload:
    joint_pos: np.ndarray; joint_vel: np.ndarray; joint_acc: np.ndarray; joint_dec: np.ndarray; text: str


def _payload(q: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray, deceleration: np.ndarray) -> Payload:
    text = ("{JogAnyJ3 " f"--joint_pos={_text(q)} --joint_vel={_text(velocity)} "
            f"--joint_acc={_text(acceleration)} --joint_dec={_text(deceleration)} --last_count=75}}")
    return Payload(q, velocity, acceleration, deceleration, text)


def build_active_payload(q_command: Iterable[object], q_previous: Iterable[object]) -> Payload:
    q, previous = _six(q_command, "q_command"), _six(q_previous, "q_previous")
    velocity = (q - previous) / config.DT
    if float(np.max(np.abs(velocity))) > config.JOINT_SPEED_LIMIT_RAD_S + 1e-9:
        raise ValueError("payload velocity exceeds PlaCo limit; no clamp is applied")
    acceleration = config.PAYLOAD_ACCELERATION_SCALE * np.abs(velocity)
    return _payload(q, velocity, acceleration, acceleration.copy())


def build_home_payload(q_command: Iterable[object], q_previous: Iterable[object]) -> Payload:
    """HOME-only joint-space command; this is not an ACTIVE teleop limiter."""
    q, previous = _six(q_command, "q_command"), _six(q_previous, "q_previous")
    velocity = (q - previous) / config.DT
    if float(np.max(np.abs(velocity))) > config.HOME_SPEED_RAD_S + 1e-9:
        raise ValueError("HOME trajectory exceeds HOME_SPEED_RAD_S")
    acceleration = config.PAYLOAD_ACCELERATION_SCALE * np.abs(velocity)
    return _payload(q, velocity, acceleration, acceleration.copy())


def build_home_sync_payload(q_command: Iterable[object], q_previous: Iterable[object]) -> Payload:
    """HOME_SYNC-only absolute command validation; never clip or rescale."""
    q, previous = _six(q_command, "q_command"), _six(q_previous, "q_previous")
    velocity = (q - previous) / config.DT
    if float(np.max(np.abs(velocity))) > config.HOME_SYNC_SPEED_RAD_S + 1e-9:
        raise ValueError("HOME_SYNC trajectory exceeds HOME_SYNC_SPEED_RAD_S")
    acceleration = config.PAYLOAD_ACCELERATION_SCALE * np.abs(velocity)
    return _payload(q, velocity, acceleration, acceleration.copy())


@dataclass
class BrakeSession:
    """HOME-only host trajectory segment copied from the mature product semantics."""
    reason: str
    q_current: np.ndarray
    v_release: np.ndarray
    brake_acc: np.ndarray
    brake_time_s: float
    start_ns: int
    last_alpha: float | None = None


def build_brake(q0: Iterable[object], v0: Iterable[object], reason: str, now_ns: int) -> BrakeSession | None:
    q, velocity = _six(q0, "q0"), _six(v0, "v0")
    peak = float(np.max(np.abs(velocity)))
    if peak <= 1e-12:
        return None
    brake_time = min(peak / config.BRAKE_DECEL_REFERENCE_RAD_S2, config.BRAKE_SEGMENT_TIME_LIMIT_S)
    return BrakeSession(str(reason), q.copy(), velocity.copy(), np.abs(velocity) / brake_time, brake_time, int(now_ns))


def sample_brake(session: BrakeSession, now_ns: int) -> dict[str, Any]:
    elapsed = max(0.0, (int(now_ns) - session.start_ns) / 1_000_000_000.0) + config.DT
    alpha = max(0.0, 1.0 - elapsed / session.brake_time_s)
    if session.last_alpha is not None and alpha > session.last_alpha + 1e-15:
        raise RuntimeError("brake alpha regressed")
    velocity = alpha * session.v_release
    session.q_current = session.q_current + velocity * config.DT
    session.last_alpha = alpha
    return {"q_brake": session.q_current.copy(), "v_brake": velocity,
            "brake_alpha": alpha, "brake_terminal": alpha == 0.0}


def brake_payload(q: Iterable[object], velocity: Iterable[object], acceleration: Iterable[object]) -> Payload:
    q_values = _six(q, "q")
    velocity_values = _six(velocity, "velocity")
    acceleration_values = _six(acceleration, "acceleration")
    if np.any(acceleration_values < 0.0) or float(np.max(np.abs(velocity_values))) > config.JOINT_SPEED_LIMIT_RAD_S + 1e-12:
        raise ValueError("invalid HOME brake payload")
    return _payload(q_values, velocity_values, acceleration_values, acceleration_values.copy())


def build_release_payload(q_hold: Iterable[object]) -> Payload:
    q = _six(q_hold, "q_hold")
    zero, deceleration = np.zeros(6), np.full(6, config.RELEASE_DECEL_RAD_S2)
    return _payload(q, zero, zero, deceleration)


def strict_hold(q_hold: Iterable[object]) -> Payload:
    q = _six(q_hold, "q_hold"); zero = np.zeros(6)
    return _payload(q, zero, zero, zero)


class LiveTransport(VerifiedJogAnyTransport):
    """Verified seq1/seq2 adapter with local-only, idempotent transport cleanup."""
    def submit_warmup_hold(self, payload_text: str, q_hold: Any) -> SubmitResult:
        if payload_text != strict_hold(q_hold).text or self._next_sequence_id != 1: raise ValueError("seq1 must be strict hold")
        sequence = self._allocate_sequence(); record, event = CallbackRecord(sequence, "FIRST_FRAME_WARMUP"), threading.Event()
        with self._lock: self._records[sequence], self._events[sequence] = record, event
        message = self.native.Msg(payload_text); message.setMsgID(10001); message.setMsgSeqID(sequence)
        try: submitted = self.client.inner.CallAsync(message, 2000, lambda status, responses: self._callback(sequence, status, responses))
        except Exception as error: submitted, record.error = False, f"CallAsync exception: {type(error).__name__}: {error}"
        if type(submitted) is not bool or not submitted:
            with self._lock:
                record.submitted, record.state = False, "SUBMIT_FAILED"; self._latch_fatal_locked(record.error or f"CallAsync returned {submitted!r}")
            return SubmitResult(sequence, record.phase, False, record.state, self.outstanding, False)
        with self._lock:
            record.submitted = True; self.last_accepted_command_q, self.last_accepted_command_velocity = tuple(q_hold), (0.0,) * 6
        return SubmitResult(sequence, record.phase, True, record.state, self.outstanding, True)

    def quarantine_warmup_timeout(self) -> None:
        with self._lock:
            record = self._records[1]
            if record.callback_received: raise RuntimeError("seq1 callback already received")
            record.state, record.error = "QUARANTINED_TIMEOUT", "known seq1 strict-hold callback unresolved"

    def _callback(self, sequence_id: int, status: Any, responses: Any) -> None:
        with self._lock:
            record = self._records.get(sequence_id)
            if record is not None and sequence_id == 1 and record.state == "QUARANTINED_TIMEOUT": record.state = "LATE_QUARANTINED"; return
        super()._callback(sequence_id, status, responses)

    def terminate_once(self) -> None:
        """Release this process's RPC client without issuing a robot-side Stop command."""
        with self._lock:
            if getattr(self, "_transport_terminated", False):
                return None
            self._transport_terminated = True
            self.rpc_stream_ready = False
            # The vendored wrapper exposes no explicit close method. Releasing its only
            # reference closes the local client handle; it does not serialize an RPC.
            self.client = None
        return None

    def send_true_stop_once(self) -> tuple[int, tuple[int, ...]] | None:
        """HOME-fatal only: reuse the verified synchronous ``{Stop}`` wire once."""
        return super().terminate_once()


def create_transport(ip: str, timeout_ms: int) -> LiveTransport:
    native, wrapper = load_vendor_rpc_modules()
    client = wrapper.RpcClient(ip, connect_timeout_ms=timeout_ms)
    if not client.is_connected(): raise RuntimeError("RPC client not connected: " + client.error_info())
    return LiveTransport(native, client)
