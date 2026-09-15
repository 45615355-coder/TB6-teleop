"""Verified JogAnyJ3 text-RPC transport with seq1/seq2 isolation."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import threading
import time
from types import ModuleType
from typing import Any


RPC_MESSAGE_ID = 10001
ASYNC_RPC_TIMEOUT_MS = 40000
STOP_RPC_TIMEOUT_MS = 10000
MAX_OUTSTANDING = 8
VENDOR_ROOT = Path(__file__).resolve().parent
RPC_SO_PATH = VENDOR_ROOT / "lib" / "linux" / "x86" / "2004" / "rpc.so"
RPC_WRAPPER_PATH = VENDOR_ROOT / "rpc_client.py"
RPC_WRAPPER_MODULE_NAME = "_tb6_probe_verified_rpc_client"
STOP_TEXT = "{" + "Stop" + "}"


@dataclass
class CallbackRecord:
    sequence_id: int
    phase: str
    submitted: bool | None = None
    callback_received: bool = False
    communication_status: int | None = None
    response_codes: tuple[int, ...] = ()
    state: str = "PENDING"
    error: str = ""


@dataclass(frozen=True)
class SubmitResult:
    sequence_id: int
    phase: str
    submitted: bool
    callback_state: str
    outstanding: int
    accepted_state_advanced: bool


def load_vendor_rpc_modules() -> tuple[ModuleType, ModuleType]:
    """Load the two audited vendor modules without creating a client."""
    expected = RPC_SO_PATH.resolve(strict=True)
    native = sys.modules.get("rpc")
    if native is not None:
        if Path(str(getattr(native, "__file__", ""))).resolve(strict=True) != expected:
            raise ImportError("an unexpected rpc module is already loaded")
    else:
        spec = importlib.util.spec_from_file_location("rpc", expected)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load native RPC module: {expected}")
        native = importlib.util.module_from_spec(spec)
        sys.modules["rpc"] = native
        try:
            spec.loader.exec_module(native)
        except BaseException:
            sys.modules.pop("rpc", None)
            raise
    wrapper_spec = importlib.util.spec_from_file_location(
        RPC_WRAPPER_MODULE_NAME, RPC_WRAPPER_PATH.resolve(strict=True)
    )
    if wrapper_spec is None or wrapper_spec.loader is None:
        raise ImportError(f"cannot load RPC wrapper: {RPC_WRAPPER_PATH}")
    wrapper = importlib.util.module_from_spec(wrapper_spec)
    sys.modules[RPC_WRAPPER_MODULE_NAME] = wrapper
    wrapper_spec.loader.exec_module(wrapper)
    return native, wrapper


class VerifiedJogAnyTransport:
    """Only text wrapping, async submission, callbacks, and one-shot termination."""

    def __init__(self, native_module: Any, client: Any) -> None:
        self.native = native_module
        self.client = client
        self._lock = threading.Lock()
        self._events: dict[int, threading.Event] = {}
        self._records: dict[int, CallbackRecord] = {}
        self._next_sequence_id = 1
        self.rpc_stream_ready = False
        self.fatal_reason = ""
        self.last_accepted_command_q: tuple[float, ...] | None = None
        self.last_accepted_command_velocity: tuple[float, ...] | None = None
        self.stop_sent = False
        self.stop_result: tuple[int, tuple[int, ...]] | None = None

    @property
    def outstanding(self) -> int:
        with self._lock:
            return sum(
                record.submitted is True
                and not record.callback_received
                and record.sequence_id >= 3
                for record in self._records.values()
            )

    def submit_warmup_hold(self, payload_text: str, q_hold: Any) -> SubmitResult:
        if self._next_sequence_id != 1:
            raise RuntimeError("seq1 warmup has already been attempted")
        return self._submit(payload_text, "FIRST_FRAME_WARMUP", q_hold, (0.0,) * 6)

    def submit_verification_hold(self, payload_text: str, q_hold: Any) -> SubmitResult:
        if self._next_sequence_id != 2:
            raise RuntimeError("seq2 verification is not the next frame")
        return self._submit(payload_text, "VERIFICATION_FRAME", q_hold, (0.0,) * 6)

    def wait_for_verification(self, timeout_s: float) -> bool:
        event = self._events.get(2)
        if event is None or not event.wait(float(timeout_s)):
            self._latch_fatal("seq2 verification callback timeout")
            return False
        with self._lock:
            record = self._records[2]
            valid = record.state == "SUCCESS"
            self.rpc_stream_ready = valid
            if not valid:
                self._latch_fatal_locked(record.error or "seq2 verification failed")
            return valid

    def submit_main(
        self, payload_text: str, q_command: Any, joint_velocity: Any
    ) -> SubmitResult:
        if not self.rpc_stream_ready:
            raise RuntimeError("seq2 verification has not established stream readiness")
        if self._next_sequence_id < 3:
            raise RuntimeError("main motion sequence must be >= 3")
        if self.fatal_reason:
            raise RuntimeError(self.fatal_reason)
        if self.outstanding >= MAX_OUTSTANDING:
            raise RuntimeError(f"outstanding limit reached: {MAX_OUTSTANDING}")
        return self._submit(payload_text, "MAIN_STREAM", q_command, joint_velocity)

    def callback_state(self, sequence_id: int) -> str:
        with self._lock:
            record = self._records.get(sequence_id)
            return "UNKNOWN" if record is None else record.state

    def wait_until_below_limit(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + float(timeout_s)
        while self.outstanding >= MAX_OUTSTANDING and time.monotonic() < deadline:
            time.sleep(0.002)
        return self.outstanding < MAX_OUTSTANDING and not self.fatal_reason

    def wait_for_main_drain(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + float(timeout_s)
        while self.outstanding > 0 and time.monotonic() < deadline:
            time.sleep(0.002)
        return self.outstanding == 0 and not self.fatal_reason

    def terminate_once(self) -> tuple[int, tuple[int, ...]] | None:
        """Historical stream_core_v4 strategy: one synchronous text Stop."""
        if self.stop_sent:
            return self.stop_result
        self.stop_sent = True
        message = self.native.Msg(STOP_TEXT)
        message.setMsgID(RPC_MESSAGE_ID)
        message.setMsgSeqID(self._allocate_sequence())
        status, responses = self.client.inner.CallAwait(message, STOP_RPC_TIMEOUT_MS)
        codes = () if responses is None else tuple(int(item.code) for item in responses)
        self.stop_result = (int(status), codes)
        return self.stop_result

    def _submit(
        self, payload_text: str, phase: str, q_command: Any, joint_velocity: Any
    ) -> SubmitResult:
        sequence_id = self._allocate_sequence()
        record = CallbackRecord(sequence_id, phase)
        event = threading.Event()
        with self._lock:
            self._records[sequence_id] = record
            self._events[sequence_id] = event
        message = self.native.Msg(str(payload_text))
        message.setMsgID(RPC_MESSAGE_ID)
        message.setMsgSeqID(sequence_id)
        callback = lambda status, responses, seq=sequence_id: self._callback(
            seq, status, responses
        )
        try:
            submitted = self.client.inner.CallAsync(
                message, ASYNC_RPC_TIMEOUT_MS, callback
            )
        except Exception as error:
            with self._lock:
                record.submitted = False
                record.state = "SUBMIT_FAILED"
                record.error = f"CallAsync exception: {type(error).__name__}: {error}"
                self._latch_fatal_locked(record.error)
                callback_state = record.state
            return SubmitResult(sequence_id, phase, False, callback_state, self.outstanding, False)
        if type(submitted) is not bool or not submitted:
            with self._lock:
                record.submitted = False
                record.state = "SUBMIT_FAILED"
                record.error = f"CallAsync returned {submitted!r} for seq {sequence_id}"
                self._latch_fatal_locked(record.error)
                callback_state = record.state
            return SubmitResult(sequence_id, phase, False, callback_state, self.outstanding, False)
        with self._lock:
            record.submitted = True
            # Accepted command state advances only after explicit successful submission.
            self.last_accepted_command_q = tuple(float(v) for v in q_command)
            self.last_accepted_command_velocity = tuple(
                float(v) for v in joint_velocity
            )
            callback_state = record.state
        return SubmitResult(sequence_id, phase, True, callback_state, self.outstanding, True)

    def _callback(self, sequence_id: int, status: Any, responses: Any) -> None:
        with self._lock:
            record = self._records.get(sequence_id)
            if record is None or record.callback_received:
                self._latch_fatal_locked(f"invalid callback for seq {sequence_id}")
                return
            record.callback_received = True
            try:
                record.communication_status = int(status)
                response_tuple = () if responses is None else tuple(responses)
                record.response_codes = tuple(int(item.code) for item in response_tuple)
                error = ""
                if record.communication_status != 0:
                    error = f"communication status {record.communication_status}"
                elif not record.response_codes:
                    error = "empty response"
                elif any(code != 0 for code in record.response_codes):
                    error = f"non-zero response codes {record.response_codes}"
            except Exception as exc:
                error = f"callback parse error: {type(exc).__name__}: {exc}"
            if error:
                record.error = f"seq {sequence_id}: {error}"
                if sequence_id == 1:
                    record.state = "QUARANTINED"
                else:
                    record.state = "ERROR"
                    self._latch_fatal_locked(record.error)
            else:
                record.state = "SUCCESS"
            self._events[sequence_id].set()

    def _allocate_sequence(self) -> int:
        with self._lock:
            result = self._next_sequence_id
            self._next_sequence_id += 1
            return result

    def _latch_fatal(self, reason: str) -> None:
        with self._lock:
            self._latch_fatal_locked(reason)

    def _latch_fatal_locked(self, reason: str) -> None:
        if not self.fatal_reason:
            self.fatal_reason = str(reason)
