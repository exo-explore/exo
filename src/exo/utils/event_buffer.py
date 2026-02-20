from loguru import logger


class OrderedBuffer[T]:
    """
    A buffer that resequences events to ensure their ordering is preserved.
    Currently this buffer doesn't raise any errors if an event is lost
    This buffer is NOT thread safe, and is designed to only be polled from one
    source at a time.
    """

    def __init__(self):
        self.store: dict[int, T] = {}
        self.next_idx_to_release: int = 0

    def ingest(self, idx: int, t: T):
        """Ingest a sequence into the buffer"""
        logger.trace(f"Ingested event {t}")
        if idx < self.next_idx_to_release:
            logger.warning(
                f"DROPPING event with idx={idx} because next_idx_to_release={self.next_idx_to_release} "
                f"(event is {self.next_idx_to_release - idx} behind). Event: {str(t)[:200]}"
            )
            return
        if idx in self.store:
            logger.debug(
                f"Duplicate event idx={idx} (next_idx_to_release={self.next_idx_to_release}). "
                f"Event: {str(t)[:200]}"
            )
            assert self.store[idx] == t, (
                "Received different messages with identical indices, probable race condition"
            )
            return
        gap = idx - self.next_idx_to_release
        if gap > 0 and idx not in self.store:
            logger.info(
                f"Buffering event idx={idx}, waiting for idx={self.next_idx_to_release} "
                f"({gap} events ahead, {len(self.store)} buffered). Event: {str(t)[:200]}"
            )
        else:
            logger.debug(
                f"Ingesting event idx={idx} (next_idx_to_release={self.next_idx_to_release}, "
                f"{len(self.store)} buffered). Event: {str(t)[:200]}"
            )
        self.store[idx] = t

    def drain(self) -> list[T]:
        """Drain all available events from the buffer"""
        ret: list[T] = []
        while self.next_idx_to_release in self.store:
            idx = self.next_idx_to_release
            event = self.store.pop(idx)
            ret.append(event)
            self.next_idx_to_release += 1
        if ret:
            logger.debug(
                f"Drained {len(ret)} events (indices {self.next_idx_to_release - len(ret)}"
                f"-{self.next_idx_to_release - 1}), {len(self.store)} still buffered"
            )
        if self.store:
            held_indices = sorted(self.store.keys())
            logger.warning(
                f"Buffer has {len(self.store)} events stuck waiting for idx={self.next_idx_to_release}. "
                f"Held indices: {held_indices[:20]}{'...' if len(held_indices) > 20 else ''}"
            )
        return ret

    def drain_indexed(self) -> list[tuple[int, T]]:
        """Drain all available events from the buffer"""
        ret: list[tuple[int, T]] = []
        while self.next_idx_to_release in self.store:
            idx = self.next_idx_to_release
            event = self.store.pop(idx)
            ret.append((idx, event))
            self.next_idx_to_release += 1
        if ret:
            logger.debug(
                f"Drained (indexed) {len(ret)} events (indices {ret[0][0]}-{ret[-1][0]}), "
                f"{len(self.store)} still buffered"
            )
        if self.store:
            held_indices = sorted(self.store.keys())
            logger.warning(
                f"Buffer has {len(self.store)} events stuck waiting for idx={self.next_idx_to_release}. "
                f"Held indices: {held_indices[:20]}{'...' if len(held_indices) > 20 else ''}"
            )
        return ret


class MultiSourceBuffer[SourceId, T]:
    """
    A buffer that resequences events to ensure their ordering is preserved.
    Tracks events with multiple sources
    """

    def __init__(self):
        self.stores: dict[SourceId, OrderedBuffer[T]] = {}

    def ingest(self, idx: int, t: T, source: SourceId):
        is_new_source = source not in self.stores
        if is_new_source:
            self.stores[source] = OrderedBuffer()
            logger.info(f"New source registered: {source}")
        buffer = self.stores[source]
        logger.debug(
            f"MultiSourceBuffer ingesting from source={source} idx={idx} "
            f"(source next_idx_to_release={buffer.next_idx_to_release}). Event: {str(t)[:200]}"
        )
        buffer.ingest(idx, t)

    def drain(self) -> list[T]:
        ret: list[T] = []
        for store in self.stores.values():
            ret.extend(store.drain())
        return ret
