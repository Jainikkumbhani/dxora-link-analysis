"""Small in-memory helpers: TTL cache (avoid paying twice for the same URL) and rate limiter."""

import time
from collections import OrderedDict, defaultdict, deque
from typing import Any, Deque, Dict, Generic, Optional, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    """Least-recently-used cache whose entries expire after ``ttl_seconds``."""

    def __init__(self, ttl_seconds: int, max_entries: int) -> None:
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._data: "OrderedDict[str, tuple]" = OrderedDict()

    def get(self, key: str) -> Optional[T]:
        item = self._data.get(key)
        if item is None:
            return None
        expires_at, value = item
        if expires_at < time.monotonic():
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: T) -> None:
        if self.ttl <= 0 or self.max_entries <= 0:
            return
        self._data[key] = (time.monotonic() + self.ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()


class RateLimiter:
    """Sliding one-minute window per client key."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: Dict[Any, Deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= self.per_minute:
            return False
        hits.append(now)
        return True
