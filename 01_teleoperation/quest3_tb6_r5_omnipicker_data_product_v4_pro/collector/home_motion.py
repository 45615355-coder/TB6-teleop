"""Pure joint-space HOME trajectory planning for the teleoperation main loop."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

import numpy as np


def _six(values: Iterable[object], name: str) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=float)
    if result.shape != (6,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain six finite values")
    return result


@dataclass
class HomeMotionSession:
    """A finite, shared-alpha trapezoid sampled by one existing control loop."""

    start_q: np.ndarray
    target_q: np.ndarray
    motion_duration_s: float
    motion_frames: int
    ramp_duration_s: float
    control_hz: float
    current_frame: int = 0
    planned_peak_velocity_rad_s: float = field(init=False)

    def __init__(
        self,
        start_q: Iterable[object],
        target_q: Iterable[object],
        speed_rad_s: float,
        ramp_duration_s: float,
        control_hz: float,
    ) -> None:
        self.start_q = _six(start_q, "start_q")
        self.target_q = _six(target_q, "target_q")
        speed = float(speed_rad_s)
        self.ramp_duration_s = float(ramp_duration_s)
        self.control_hz = float(control_hz)
        if not all(math.isfinite(value) and value > 0.0 for value in (
            speed, self.ramp_duration_s, self.control_hz,
        )):
            raise ValueError("speed, ramp_duration_s, and control_hz must be finite and > 0")

        distance = float(np.max(np.abs(self.target_q - self.start_q)))
        requested_duration = max(
            2.0 * self.ramp_duration_s,
            distance / speed + self.ramp_duration_s,
        )
        self.motion_frames = math.ceil(requested_duration * self.control_hz)
        self.motion_duration_s = self.motion_frames / self.control_hz
        self.current_frame = 0
        self.planned_peak_velocity_rad_s = self._check_discrete_velocity()
        if self.planned_peak_velocity_rad_s > speed + 1e-12:
            raise ValueError(
                "HOME trajectory discrete velocity exceeds requested speed: "
                f"{self.planned_peak_velocity_rad_s} > {speed}"
            )

    @property
    def motion_done(self) -> bool:
        return self.current_frame >= self.motion_frames

    def next_target(self) -> np.ndarray:
        """Return the next 50 Hz target; completed sessions stay exactly at HOME."""
        if self.motion_done:
            return self.target_q.copy()
        self.current_frame += 1
        if self.motion_done:
            return self.target_q.copy()
        return self.start_q + self._alpha(self.current_frame / self.control_hz) * (
            self.target_q - self.start_q
        )

    def _alpha(self, elapsed_s: float) -> float:
        duration = self.motion_duration_s
        ramp = self.ramp_duration_s
        acceleration = 1.0 / (ramp * (duration - ramp))
        if elapsed_s <= ramp:
            alpha = 0.5 * acceleration * elapsed_s * elapsed_s
        elif elapsed_s < duration - ramp:
            alpha = (elapsed_s - 0.5 * ramp) / (duration - ramp)
        else:
            remaining = duration - elapsed_s
            alpha = 1.0 - 0.5 * acceleration * remaining * remaining
        return max(0.0, min(1.0, alpha))

    def _check_discrete_velocity(self) -> float:
        previous = self.start_q
        peak = 0.0
        for frame in range(1, self.motion_frames + 1):
            target = (
                self.target_q
                if frame == self.motion_frames
                else self.start_q + self._alpha(frame / self.control_hz) * (
                    self.target_q - self.start_q
                )
            )
            peak = max(peak, float(np.max(np.abs(target - previous))) * self.control_hz)
            previous = target
        return peak
