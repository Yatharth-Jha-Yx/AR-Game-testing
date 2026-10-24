"""A one-slot "newest wins" hand-off between threads.

The capture thread always overwrites the slot; the inference thread always
takes the newest frame. If inference is slower than the camera, old frames
are dropped instead of queuing up (queues add latency, drops don't).
"""

from __future__ import annotations

import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class LatestSlot(Generic[T]):
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: T | None = None
        self._version = 0
        self._taken_version = 0
        self._closed = False
        self.dropped = 0

    def put(self, item: T) -> None:
        with self._cond:
            if self._item is not None and self._taken_version != self._version:
                self.dropped += 1
            self._item = item
            self._version += 1
            self._cond.notify_all()

    def get(self, timeout: float | None = None) -> T | None:
        """Block until an item newer than the last one taken is available."""
        with self._cond:
            if not self._cond.wait_for(
                lambda: self._closed or self._version != self._taken_version, timeout
            ):
                return None
            if self._closed and self._version == self._taken_version:
                return None
            self._taken_version = self._version
            return self._item

    def peek(self) -> T | None:
        with self._cond:
            return self._item

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
