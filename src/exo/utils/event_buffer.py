from loguru import logger


class OrderedBuffer[T]:
    """
    A buffer that resequences events to ensure their ordering is preserved.
    Currently this buffer doesn't raise any errors if an event is lost
    This buffer is NOT thread safe, and is designed to only be polled from one
    source at a time.
    """

    def __init__(self, start_idx: int = 0):
        self.store: dict[int, T] = {}
        self.next_idx_to_release: int = start_idx

    def ingest(self, idx: int, t: T):
        """Ingest a sequence into the buffer"""
        logger.trace(f"Ingested event {t}")
        if idx < self.next_idx_to_release:
            return
        if idx in self.store:
            return
        self.store[idx] = t

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
            # Seed the per-source buffer to start at the first observed index for that source.
            self.stores[source] = OrderedBuffer(start_idx=idx)
        buffer = self.stores[source]
        # Handle per-source sequence reset (e.g., worker restart resetting its local index to 0).
        # If we observe idx == 0 from an existing source with a higher expected index,
        # reset that source's buffer to accept the new sequence.
        if idx == 0 and buffer.next_idx_to_release > 0:
            self.stores[source] = OrderedBuffer(start_idx=0)
            buffer = self.stores[source]
        buffer.ingest(idx, t)

    def drain(self) -> list[T]:
        ret: list[T] = []
        for store in self.stores.values():
            ret.extend(store.drain())
        return ret
