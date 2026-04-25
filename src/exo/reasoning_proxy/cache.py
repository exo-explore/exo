import threading


class ReasoningCache:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, content_hash: str) -> str | None:
        with self._lock:
            return self._store.get(content_hash)

    def put(self, content_hash: str, reasoning: str) -> None:
        if not reasoning:
            return
        with self._lock:
            self._store[content_hash] = reasoning

    def size(self) -> int:
        with self._lock:
            return len(self._store)
