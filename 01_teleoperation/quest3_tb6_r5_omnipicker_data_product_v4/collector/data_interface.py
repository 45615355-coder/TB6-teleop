"""Non-blocking extension point for future teleoperation data collection."""

from __future__ import annotations

from typing import Any


class NoOpDataSink:
    def record(self, sample: dict[str, Any]) -> None:
        del sample


class JsonlDataSink:
    """Optional small diagnostic sink; callers own opening/flushing policy."""

    def __init__(self, write_line: Any) -> None:
        self._write_line = write_line

    def record(self, sample: dict[str, Any]) -> None:
        self._write_line(sample)
