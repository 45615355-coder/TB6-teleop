"""Read-only latest-only Topic adapter."""
from typing import Any
from vendor.topic.tb6_topic_probe_reader import TB6ProbeTopicReader

class TopicIO:
    def __init__(self, reader: Any | None = None) -> None: self._reader = TB6ProbeTopicReader() if reader is None else reader
    def start(self, ip: str) -> None: self._reader.start(ip)
    def latest(self) -> Any: return self._reader.get_latest_snapshot()
    def close(self) -> None: self._reader.close()
