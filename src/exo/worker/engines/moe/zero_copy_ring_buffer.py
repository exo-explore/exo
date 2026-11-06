import threading

import numpy as np
from numpy.typing import NDArray


class ZeroCopyTensorRingBuffer:
    """Pre-allocated circular ring buffer for zero-copy tensor payload transfer.

    Avoids runtime allocation and garbage collection overhead by reusing fixed memory slots.
    """

    def __init__(self, capacity: int = 64, slot_shape: tuple[int, ...] = (128, 4096), dtype: np.dtype = np.float32) -> None:
        if capacity <= 0:
            raise ValueError(f"Ring buffer capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._slot_shape = slot_shape
        self._dtype = dtype

        # Pre-allocate contiguous tensor slots in host memory
        self._slots: list[NDArray] = [
            np.zeros(slot_shape, dtype=dtype) for _ in range(capacity)
        ]
        self._metadata_slots: list[dict | None] = [None] * capacity

        self._head: int = 0  # Write index
        self._tail: int = 0  # Read index
        self._count: int = 0
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def is_full(self) -> bool:
        with self._lock:
            return self._count == self._capacity

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return self._count == 0

    def write(self, tensor_data: NDArray, metadata: dict | None = None, timeout: float | None = 2.0) -> int:
        """Writes tensor into the next available slot via in-place zero-copy copyto.

        Returns:
            The slot index written to.
        """
        with self._not_full:
            if self._count == self._capacity:
                signaled = self._not_full.wait(timeout=timeout)
                if not signaled or self._count == self._capacity:
                    raise TimeoutError("Ring buffer write timed out: buffer is full")

            slot_idx = self._head
            slot = self._slots[slot_idx]

            # In-place write to pre-allocated buffer
            np.copyto(slot[: tensor_data.shape[0], : tensor_data.shape[1]], tensor_data)
            self._metadata_slots[slot_idx] = metadata

            self._head = (self._head + 1) % self._capacity
            self._count += 1
            self._not_empty.notify()
            return slot_idx

    def read(self, timeout: float | None = 2.0) -> tuple[NDArray, dict | None]:
        """Reads a memory view slice of the tensor at the tail without copying.

        Returns:
            Tuple of (slot_view, metadata).
        """
        with self._not_empty:
            if self._count == 0:
                signaled = self._not_empty.wait(timeout=timeout)
                if not signaled or self._count == 0:
                    raise TimeoutError("Ring buffer read timed out: buffer is empty")

            slot_idx = self._tail
            slot = self._slots[slot_idx]
            meta = self._metadata_slots[slot_idx]

            # Return a view without duplicating buffer memory
            view = slot.view()

            self._tail = (self._tail + 1) % self._capacity
            self._count -= 1
            self._not_full.notify()
            return view, meta
