import concurrent.futures
import time
from collections.abc import Callable
from typing import Any

from exo.shared.types.paged_kv import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_LOOKAHEAD_BLOCKS,
    DEFAULT_SINK_BLOCKS,
    PagedKVBlock,
    PrefetchRequest,
)


class SpeculativePrefetchPredictor:
    """Predicts historical and future KV page blocks based on token locality and attention sink patterns."""

    def __init__(
        self,
        block_size: int = DEFAULT_BLOCK_SIZE,
        lookahead_blocks: int = DEFAULT_LOOKAHEAD_BLOCKS,
        sink_blocks: int = DEFAULT_SINK_BLOCKS,
    ) -> None:
        self.block_size = block_size
        self.lookahead_blocks = lookahead_blocks
        self.sink_blocks = sink_blocks

    def predict_candidate_blocks(
        self,
        current_token_index: int,
        total_tokens_in_sequence: int | None = None,
        custom_heavy_hitters: list[int] | None = None,
    ) -> list[int]:
        """Predict which logical block indices should be actively prefetched into fast memory."""
        current_block = current_token_index // self.block_size
        candidates: set[int] = set()

        # 1. Attention Sink Blocks (First N blocks containing high-attention initial tokens)
        for sink_idx in range(self.sink_blocks):
            candidates.add(sink_idx)

        # 2. Sequential Lookahead Blocks (Next blocks in decoding trajectory)
        for offset in range(1, self.lookahead_blocks + 1):
            target_block = current_block + offset
            if total_tokens_in_sequence is None:
                candidates.add(target_block)
            else:
                max_block = (max(0, total_tokens_in_sequence - 1)) // self.block_size
                if target_block <= max_block + self.lookahead_blocks:
                    candidates.add(target_block)

        # 3. Custom / Heavy-Hitter Attention Clusters (e.g. from H2O/Radix attention)
        if custom_heavy_hitters:
            for hh_block in custom_heavy_hitters:
                candidates.add(hh_block)

        # Remove the currently active block itself from candidate prefetch (it's already active)
        candidates.discard(current_block)

        return sorted(candidates)


class DoubleBufferPipeline:
    """Implements asynchronous double-buffering pipeline to overlap computation with KV memory I/O."""

    def __init__(
        self,
        storage_fetch_callback: Callable[[str, int], PagedKVBlock | None] | None = None,
        max_workers: int = 2,
    ) -> None:
        self.storage_fetch_callback = storage_fetch_callback
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.active_buffer: dict[str, PagedKVBlock] = {}
        self.prefetch_buffer: dict[str, PagedKVBlock] = {}
        self.pending_futures: dict[str, concurrent.futures.Future[PagedKVBlock | None]] = {}

    def start_speculative_prefetch(
        self,
        request: PrefetchRequest,
    ) -> None:
        """Asynchronously trigger prefetching for target logical block indices."""
        if not self.storage_fetch_callback:
            return

        for block_idx in request.target_logical_block_indices:
            key = f"{request.sequence_id}:{block_idx}"
            # Skip if already in active, prefetch buffer, or in-flight
            if key in self.active_buffer or key in self.prefetch_buffer or key in self.pending_futures:
                continue

            future = self.executor.submit(self.storage_fetch_callback, request.sequence_id, block_idx)
            self.pending_futures[key] = future

    def complete_in_flight_prefetches(self, timeout_sec: float = 1.0) -> int:
        """Wait for in-flight background transfers to complete and move them to prefetch_buffer."""
        completed_count = 0
        keys_to_remove: list[str] = []

        for key, future in list(self.pending_futures.items()):
            try:
                block = future.result(timeout=timeout_sec)
                if block is not None:
                    self.prefetch_buffer[key] = block
                    completed_count += 1
                keys_to_remove.append(key)
            except concurrent.futures.TimeoutError:
                # Still running in background
                continue
            except (concurrent.futures.CancelledError, TimeoutError, OSError, RuntimeError):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            self.pending_futures.pop(key, None)

        return completed_count

    def promote_prefetch_to_active(self, sequence_id: str, logical_index: int) -> PagedKVBlock | None:
        """Promote a prefetched block to active status immediately (zero-wait handover)."""
        key = f"{sequence_id}:{logical_index}"

        # 1. Check if already in active
        if key in self.active_buffer:
            return self.active_buffer[key]

        # 2. Check if in prefetch buffer
        if key in self.prefetch_buffer:
            block = self.prefetch_buffer.pop(key)
            self.active_buffer[key] = block
            return block

        # 3. Check if in-flight future
        if key in self.pending_futures:
            try:
                block = self.pending_futures[key].result(timeout=2.0)
                self.pending_futures.pop(key, None)
                if block is not None:
                    self.active_buffer[key] = block
                    return block
            except (concurrent.futures.CancelledError, TimeoutError, OSError, RuntimeError):
                self.pending_futures.pop(key, None)

        return None

    def execute_with_compute_overlap(
        self,
        compute_fn: Callable[..., Any],
        compute_args: tuple[Any, ...],
        request: PrefetchRequest,
    ) -> tuple[Any, float, float]:
        """Runs compute_fn on main thread while pipeline concurrently prefetches candidate blocks.

        Returns: (compute_result, compute_time_ms, masked_io_time_ms)
        """
        io_start = time.perf_counter()
        self.start_speculative_prefetch(request)

        compute_start = time.perf_counter()
        result = compute_fn(*compute_args)
        compute_end = time.perf_counter()
        compute_time_ms = (compute_end - compute_start) * 1000.0

        # Sync prefetch completions
        self.complete_in_flight_prefetches(timeout_sec=0.5)
        io_end = time.perf_counter()
        total_io_time_ms = (io_end - io_start) * 1000.0

        # Masked time is the overlapping duration where compute hid I/O latency
        masked_io_time_ms = min(compute_time_ms, total_io_time_ms)
        return result, compute_time_ms, masked_io_time_ms

    def close(self) -> None:
        """Shutdown thread pool executor."""
        self.executor.shutdown(wait=False)
