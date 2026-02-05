import time
from typing import Generic, TypeVar

K = TypeVar("K")


class KeyedBackoff(Generic[K]):
    """Tracks exponential backoff state per key."""

    def __init__(self, base: float = 0.5, cap: float = 10.0):
        self._base = base
        self._cap = cap
        self._attempts: dict[K, int] = {}
        self._last_time: dict[K, float] = {}

    def should_proceed(self, key: K) -> bool:
        """Returns True if enough time has elapsed since last attempt."""
        now = time.monotonic()
        last = self._last_time.get(key, 0.0)
        attempts = self._attempts.get(key, 0)
        delay = min(self._cap, self._base * (2.0**attempts))
        return now - last >= delay

    def record_attempt(self, key: K) -> None:
        """Record that an attempt was made for this key."""
        self._last_time[key] = time.monotonic()
        self._attempts[key] = self._attempts.get(key, 0) + 1

    def reset(self, key: K) -> None:
        """Reset backoff state for a key (e.g., on success)."""
        self._attempts.pop(key, None)
        self._last_time.pop(key, None)
