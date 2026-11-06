import time
from typing import Any

from exo.shared.types.paged_kv import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_LOOKAHEAD_BLOCKS,
    DEFAULT_SINK_BLOCKS,
    BlockTable,
    PagedKVBlock,
    PrefetchRequest,
    PrefetchTelemetry,
)
from exo.worker.engines.paged_attention.prefetch_manager import (
    DoubleBufferPipeline,
    SpeculativePrefetchPredictor,
)


class PagedCacheEngine:
    """Manages virtual paged KV memory, dynamic block tables, and speculative double-buffered prefetching."""

    def __init__(
        self,
        block_size: int = DEFAULT_BLOCK_SIZE,
        lookahead_blocks: int = DEFAULT_LOOKAHEAD_BLOCKS,
        sink_blocks: int = DEFAULT_SINK_BLOCKS,
        simulate_tier2_latency_sec: float = 0.005,
    ) -> None:
        self.block_size = block_size
        self.simulate_tier2_latency_sec = simulate_tier2_latency_sec

        self.block_tables: dict[str, BlockTable] = {}
        self.fast_pool: dict[str, PagedKVBlock] = {}
        self.secondary_storage: dict[str, PagedKVBlock] = {}

        self.predictor = SpeculativePrefetchPredictor(
            block_size=block_size,
            lookahead_blocks=lookahead_blocks,
            sink_blocks=sink_blocks,
        )
        self.pipeline = DoubleBufferPipeline(storage_fetch_callback=self._fetch_from_secondary_storage)

        self._telemetry_prefetches = 0
        self._telemetry_hits = 0
        self._telemetry_misses = 0
        self._telemetry_io_masked_ms = 0.0
        self._telemetry_total_io_ms = 0.0

    def _fetch_from_secondary_storage(self, sequence_id: str, block_idx: int) -> PagedKVBlock | None:
        """Internal callback to fetch a block from slower tier (e.g. SSD / network)."""
        table = self.block_tables.get(sequence_id)
        if not table:
            return None
        block_id = table.get_block_id(block_idx)
        if not block_id or block_id not in self.secondary_storage:
            return None

        # Simulate I/O transfer time
        if self.simulate_tier2_latency_sec > 0:
            time.sleep(self.simulate_tier2_latency_sec)

        block = self.secondary_storage[block_id]
        # Promoted to fast tier representation
        upgraded_block = PagedKVBlock(
            block_id=block.block_id,
            sequence_id=block.sequence_id,
            logical_block_index=block.logical_block_index,
            block_size=block.block_size,
            num_heads=block.num_heads,
            head_dim=block.head_dim,
            device_location="local_ram",
            is_pinned=block.is_pinned,
            access_count=block.access_count + 1,
            last_accessed_timestamp=time.time(),
            keys_data=block.keys_data,
            values_data=block.values_data,
        )
        return upgraded_block

    def get_or_create_block_table(self, sequence_id: str) -> BlockTable:
        """Retrieve existing or initialize a new BlockTable for a sequence."""
        if sequence_id not in self.block_tables:
            self.block_tables[sequence_id] = BlockTable(
                sequence_id=sequence_id,
                block_size=self.block_size,
                logical_to_physical={},
            )
        return self.block_tables[sequence_id]

    def allocate_block(
        self,
        sequence_id: str,
        logical_index: int,
        num_heads: int = 32,
        head_dim: int = 128,
        device_location: str = "local_ram",
    ) -> PagedKVBlock:
        """Allocates a new physical KV block and binds it to the logical index in BlockTable."""
        table = self.get_or_create_block_table(sequence_id)
        block_id = f"{sequence_id}_b{logical_index}_{int(time.time() * 1000) % 1000000}"

        block = PagedKVBlock(
            block_id=block_id,
            sequence_id=sequence_id,
            logical_block_index=logical_index,
            block_size=self.block_size,
            num_heads=num_heads,
            head_dim=head_dim,
            device_location=device_location,
            last_accessed_timestamp=time.time(),
        )

        if device_location == "local_ram" or device_location == "local_vram":
            self.fast_pool[block_id] = block
        else:
            self.secondary_storage[block_id] = block

        # Update table mapping
        new_mapping = dict(table.logical_to_physical)
        new_mapping[logical_index] = block_id
        self.block_tables[sequence_id] = BlockTable(
            sequence_id=sequence_id,
            block_size=self.block_size,
            logical_to_physical=new_mapping,
        )
        return block

    def append_kv_token(
        self,
        sequence_id: str,
        token_index: int,
        key_vector: tuple[float, ...],
        value_vector: tuple[float, ...],
        num_heads: int = 32,
        head_dim: int = 128,
    ) -> PagedKVBlock:
        """Appends a single token's KV projection into the appropriate logical block."""
        table = self.get_or_create_block_table(sequence_id)
        logical_index = token_index // self.block_size

        block_id = table.get_block_id(logical_index)
        target_block = None

        if block_id:
            target_block = self.fast_pool.get(block_id) or self.secondary_storage.get(block_id)

        if not target_block:
            target_block = self.allocate_block(
                sequence_id=sequence_id,
                logical_index=logical_index,
                num_heads=num_heads,
                head_dim=head_dim,
            )
            block_id = target_block.block_id

        # Update block with appended token data
        new_keys = tuple(list(target_block.keys_data) + [key_vector])
        new_values = tuple(list(target_block.values_data) + [value_vector])

        updated_block = PagedKVBlock(
            block_id=target_block.block_id,
            sequence_id=target_block.sequence_id,
            logical_block_index=target_block.logical_block_index,
            block_size=target_block.block_size,
            num_heads=target_block.num_heads,
            head_dim=target_block.head_dim,
            device_location=target_block.device_location,
            is_pinned=target_block.is_pinned,
            access_count=target_block.access_count + 1,
            last_accessed_timestamp=time.time(),
            keys_data=new_keys,
            values_data=new_values,
        )

        if updated_block.device_location in ("local_ram", "local_vram"):
            self.fast_pool[block_id] = updated_block
        else:
            self.secondary_storage[block_id] = updated_block

        return updated_block

    def evict_block_to_secondary_tier(
        self,
        sequence_id: str,
        logical_index: int,
        target_location: str = "nvme_ssd",
    ) -> bool:
        """Evicts a block from local fast memory to slower secondary storage (e.g. SSD / Remote Node)."""
        table = self.block_tables.get(sequence_id)
        if not table:
            return False
        block_id = table.get_block_id(logical_index)
        if not block_id or block_id not in self.fast_pool:
            return False

        fast_block = self.fast_pool.pop(block_id)
        demoted_block = PagedKVBlock(
            block_id=fast_block.block_id,
            sequence_id=fast_block.sequence_id,
            logical_block_index=fast_block.logical_block_index,
            block_size=fast_block.block_size,
            num_heads=fast_block.num_heads,
            head_dim=fast_block.head_dim,
            device_location=target_location,
            is_pinned=fast_block.is_pinned,
            access_count=fast_block.access_count,
            last_accessed_timestamp=time.time(),
            keys_data=fast_block.keys_data,
            values_data=fast_block.values_data,
        )
        self.secondary_storage[block_id] = demoted_block
        return True

    def read_block(
        self,
        sequence_id: str,
        logical_index: int,
    ) -> tuple[PagedKVBlock | None, bool]:
        """Reads a block. Returns (block, is_cache_hit)."""
        # 1. Check pipeline prefetch/active buffer
        pipelined = self.pipeline.promote_prefetch_to_active(sequence_id, logical_index)
        if pipelined:
            self._telemetry_hits += 1
            self.fast_pool[pipelined.block_id] = pipelined
            return pipelined, True

        table = self.block_tables.get(sequence_id)
        if not table:
            self._telemetry_misses += 1
            return None, False

        block_id = table.get_block_id(logical_index)
        if not block_id:
            self._telemetry_misses += 1
            return None, False

        # 2. Check local fast pool
        if block_id in self.fast_pool:
            self._telemetry_hits += 1
            return self.fast_pool[block_id], True

        # 3. Cache miss: load synchronously from secondary storage
        self._telemetry_misses += 1
        start_io = time.perf_counter()
        block = self._fetch_from_secondary_storage(sequence_id, logical_index)
        io_elapsed_ms = (time.perf_counter() - start_io) * 1000.0
        self._telemetry_total_io_ms += io_elapsed_ms

        if block:
            self.fast_pool[block.block_id] = block
        return block, False

    def step_forward_with_prefetch(
        self,
        sequence_id: str,
        current_token_index: int,
        compute_fn: Any = None,
        compute_args: tuple[Any, ...] = (),
    ) -> tuple[PagedKVBlock | None, PrefetchTelemetry]:
        """Executes current forward step while concurrently prefetching candidate blocks."""
        current_logical_block = current_token_index // self.block_size
        candidate_blocks = self.predictor.predict_candidate_blocks(
            current_token_index=current_token_index,
        )

        prefetch_req = PrefetchRequest(
            request_id=f"req_{sequence_id}_{current_token_index}",
            sequence_id=sequence_id,
            target_logical_block_indices=tuple(candidate_blocks),
        )

        self._telemetry_prefetches += len(candidate_blocks)

        def default_compute_workload() -> None:
            # Simulate forward compute layer time (e.g. 8ms)
            time.sleep(0.008)

        worker_compute = compute_fn if compute_fn else default_compute_workload
        _, _compute_time_ms, masked_io_ms = self.pipeline.execute_with_compute_overlap(
            compute_fn=worker_compute,
            compute_args=compute_args,
            request=prefetch_req,
        )

        self._telemetry_io_masked_ms += masked_io_ms
        self._telemetry_total_io_ms += masked_io_ms

        block, _ = self.read_block(sequence_id, current_logical_block)
        return block, self.get_telemetry()

    def get_telemetry(self) -> PrefetchTelemetry:
        return PrefetchTelemetry(
            total_prefetches=self._telemetry_prefetches,
            cache_hits=self._telemetry_hits,
            cache_misses=self._telemetry_misses,
            io_time_masked_ms=self._telemetry_io_masked_ms,
            total_io_latency_ms=self._telemetry_total_io_ms,
        )

    def close(self) -> None:
        self.pipeline.close()
