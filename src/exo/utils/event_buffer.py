from loguru import logger


class OrderedBuffer[T]:
    """
    A buffer that resequences events to ensure their ordering is preserved.
    Currently this buffer doesn't raise any errors if an event is lost
    This buffer is NOT thread safe, and is designed to only be polled from one
    source at a time.
    """

    _MAX_BUFFERED = 10000

    def __init__(self):
        self.store: dict[int, T] = {}
        self.next_idx_to_release: int = 0

    def ingest(self, idx: int, t: T):
        """Ingest a sequence into the buffer"""
        logger.trace(f"Ingested event {t}")
        if idx < self.next_idx_to_release:
            return
        if idx in self.store:
            if self.store[idx] != t:
                raise RuntimeError(
                    f"Received different messages with identical index {idx}, probable race condition"
                )
            return
        self.store[idx] = t
        # Prevent unbounded growth from gaps that never fill
        if len(self.store) > self._MAX_BUFFERED:
            # Skip ahead to the lowest buffered index to unblock the drain
            min_idx = min(self.store)
            if min_idx > self.next_idx_to_release:
                logger.warning(
                    f"OrderedBuffer: skipping from {self.next_idx_to_release} to {min_idx} "
                    f"({min_idx - self.next_idx_to_release} lost events) to prevent unbounded growth"
                )
                self.next_idx_to_release = min_idx

    def drain(self) -> list[T]:
        """Drain all available events from the buffer"""
        ret: list[T] = []
        while self.next_idx_to_release in self.store:
            idx = self.next_idx_to_release
            event = self.store.pop(idx)
            ret.append(event)
            self.next_idx_to_release += 1
        logger.trace(f"Releasing event {ret}")
        return ret

    def drain_indexed(self) -> list[tuple[int, T]]:
        """Drain all available events from the buffer"""
        ret: list[tuple[int, T]] = []
        while self.next_idx_to_release in self.store:
            idx = self.next_idx_to_release
            event = self.store.pop(idx)
            ret.append((idx, event))
            self.next_idx_to_release += 1
        logger.trace(f"Releasing event {ret}")
        return ret


class MultiSourceBuffer[SourceId, T]:
    """
    A buffer that resequences events to ensure their ordering is preserved.
    Tracks events with multiple sources
    """

    def __init__(self):
        self.stores: dict[SourceId, OrderedBuffer[T]] = {}

    def ingest(self, idx: int, t: T, source: SourceId):
        if source not in self.stores:
            self.stores[source] = OrderedBuffer()
        buffer = self.stores[source]
        buffer.ingest(idx, t)

    def drain(self) -> list[T]:
        ret: list[T] = []
        for store in self.stores.values():
            ret.extend(store.drain())
        return ret
