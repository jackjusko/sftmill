"""Periodic stdout progress for long-running sftmill commands."""

from __future__ import annotations

import sys
import threading
import time


class ProgressReporter:
    """Print a single-line status every ``interval`` seconds (stderr by default)."""

    def __init__(self, label: str, *, interval: float = 30.0, stream=None):
        self.label = label
        self.interval = max(0.0, float(interval))
        self._stream = stream or sys.stderr
        self._lock = threading.Lock()
        self._fields: dict[str, int | str | float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = time.monotonic()

    def set(self, **fields: int | str | float) -> None:
        with self._lock:
            self._fields.update(fields)

    def bump(self, key: str, delta: int = 1) -> None:
        with self._lock:
            current = self._fields.get(key, 0)
            if not isinstance(current, (int, float)):
                current = 0
            self._fields[key] = int(current) + delta

    def start(self) -> None:
        self._started_at = time.monotonic()
        if self.interval <= 0:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.emit()

    def emit(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._started_at
            parts = [f"elapsed={elapsed:.0f}s"]
            for key in sorted(self._fields):
                parts.append(f"{key}={self._fields[key]}")
        line = f"[sftmill {self.label}] " + " ".join(parts)
        print(line, file=self._stream, flush=True)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.emit()
