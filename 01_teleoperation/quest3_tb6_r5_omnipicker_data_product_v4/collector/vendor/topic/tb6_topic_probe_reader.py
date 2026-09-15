"""Extended read-only TB6 Topic snapshot used by the real-motion probe."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TB6ProbeTopicSnapshot:
    q_actual_rad: np.ndarray
    q_target_rad: np.ndarray
    dq_actual_rad_s: np.ndarray
    tcp_pe321: np.ndarray
    topic_age_ms: float
    header_timestamp: int
    global_count: int
    controller_link_up: bool
    model_state: int
    model_error_code: int
    model_error_msg: str
    enabled: tuple[bool, ...]
    joint_mode: tuple[int, ...]
    joint_error_code: tuple[int, ...]


class TB6ProbeTopicReader:
    """Latest-only subscriber; this module exposes no robot-control operation."""

    def __init__(self) -> None:
        self._module: Any | None = None
        self._last_key: tuple[int, int] | None = None
        self._last_change_ns: int | None = None
        self._latest: TB6ProbeTopicSnapshot | None = None

    def start(self, ip: str) -> None:
        if self._module is not None:
            raise RuntimeError("TB6ProbeTopicReader is already started")
        from .platform_loader import get_topic_module

        self._module = get_topic_module()
        self._module.start_subscriber(str(ip))

    def get_latest_snapshot(self) -> TB6ProbeTopicSnapshot | None:
        if self._module is None:
            raise RuntimeError("TB6ProbeTopicReader.start(ip) must be called first")
        state = self._module.get_system_state()
        if not state.has_rt():
            return self._latest_with_updated_age()
        return self._extract(state.get_rt(), time.monotonic_ns())

    def close(self) -> None:
        self._module = None
        self._latest = None
        self._last_key = None
        self._last_change_ns = None

    def _latest_with_updated_age(self) -> TB6ProbeTopicSnapshot | None:
        if self._latest is None or self._last_change_ns is None:
            return None
        return self._replace_age(self._latest, time.monotonic_ns())

    def _extract(self, rt: Any, now_ns: int) -> TB6ProbeTopicSnapshot:
        model_index = self._select_six_joint_model(rt)
        model = rt.models[model_index]
        start, count = int(model.joint_start_idx), int(model.joint_count)
        if count != 6 or start < 0 or start + count > len(rt.models_joints):
            raise ValueError("Topic RT joint layout is not a valid six-joint model")
        if model_index >= len(rt.models_info):
            raise ValueError("Topic RT model_info entry is unavailable")
        joints = rt.models_joints[start : start + count]
        info = rt.models_info[model_index]
        q = np.asarray([float(joint.position) for joint in joints], dtype=float)
        q_target = np.asarray([float(joint.target_position) for joint in joints], dtype=float)
        dq = np.asarray([float(joint.velocity) for joint in joints], dtype=float)
        tcp = np.asarray(info.ee_pe321, dtype=float)
        if q.shape != (6,) or q_target.shape != (6,) or dq.shape != (6,) or tcp.shape != (6,):
            raise ValueError("Topic q/target/dq/tcp must each contain six values")
        if not all(np.all(np.isfinite(values)) for values in (q, q_target, dq, tcp)):
            raise ValueError("Topic q/target/dq/tcp contains a non-finite value")
        header = int(rt.header_timestamp)
        count_value = int(rt.controller.global_count)
        key = (header, count_value)
        if self._last_change_ns is None or key != self._last_key:
            self._last_change_ns = now_ns
        self._last_key = key
        assert self._last_change_ns is not None
        snapshot = TB6ProbeTopicSnapshot(
            q_actual_rad=q,
            q_target_rad=q_target,
            dq_actual_rad_s=dq,
            tcp_pe321=tcp,
            topic_age_ms=(now_ns - self._last_change_ns) / 1_000_000.0,
            header_timestamp=header,
            global_count=count_value,
            controller_link_up=bool(rt.controller.is_link_up),
            model_state=int(info.model_state),
            model_error_code=int(info.error_code),
            model_error_msg=str(info.error_msg),
            enabled=tuple(bool(joint.is_enabled) for joint in joints),
            joint_mode=tuple(int(joint.mode) for joint in joints),
            joint_error_code=tuple(int(joint.error_code) for joint in joints),
        )
        self._latest = snapshot
        return snapshot

    def _replace_age(
        self, snapshot: TB6ProbeTopicSnapshot, now_ns: int
    ) -> TB6ProbeTopicSnapshot:
        assert self._last_change_ns is not None
        return TB6ProbeTopicSnapshot(
            **{
                **snapshot.__dict__,
                "topic_age_ms": (now_ns - self._last_change_ns) / 1_000_000.0,
            }
        )

    @staticmethod
    def _select_six_joint_model(rt: Any) -> int:
        for index, model in enumerate(rt.models):
            if int(model.joint_count) == 6:
                return index
        raise ValueError("Topic RT contains no six-joint robot model")
