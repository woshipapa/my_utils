from __future__ import annotations

import json
import os
import threading
from collections import deque
from pathlib import Path
from typing import Deque, Iterable, List

from .metrics_types import MetricEvent


class MetricsStore:
    """
    Unified event storage:
    - in-memory ring buffer for fast reads
    - append-only JSONL file for persistence
    """

    def __init__(
        self,
        output_dir: str = "metrics",
        *,
        file_name: str = "metrics_events.jsonl",
        max_memory_events: int = 500_000,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.output_dir / file_name
        self._lock = threading.Lock()
        self._max_memory_events = max(1, int(max_memory_events))
        self._buffer: Deque[MetricEvent] = deque(maxlen=self._max_memory_events)
        self._buffer_overflowed = False

    def write_events(self, events: Iterable[MetricEvent]) -> int:
        event_list = list(events)
        if not event_list:
            return 0

        lines = []
        with self._lock:
            for evt in event_list:
                if len(self._buffer) >= self._max_memory_events:
                    self._buffer_overflowed = True
                self._buffer.append(evt)
                lines.append(json.dumps(evt.to_dict(), ensure_ascii=False))

            with self.file_path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(lines))
                handle.write("\n")

        return len(event_list)

    def read_all_events(self, *, prefer_disk: bool = True) -> List[MetricEvent]:
        if prefer_disk and self.file_path.exists():
            return self.read_events_file(str(self.file_path))

        with self._lock:
            return list(self._buffer)

    def clear(self, *, clear_disk: bool = False) -> None:
        with self._lock:
            self._buffer.clear()
            self._buffer_overflowed = False
            if clear_disk and self.file_path.exists():
                os.remove(self.file_path)

    def has_buffer_overflowed(self) -> bool:
        with self._lock:
            return bool(self._buffer_overflowed)

    @staticmethod
    def read_events_file(path: str) -> List[MetricEvent]:
        file_path = Path(path)
        if not file_path.exists():
            return []
        result: List[MetricEvent] = []
        with file_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    result.append(MetricEvent.from_dict(json.loads(line)))
                except Exception:
                    continue
        return result
