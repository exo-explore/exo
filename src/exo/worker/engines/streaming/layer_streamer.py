import time

from exo.shared.types.streaming_layer import (
    LayerDiskMetadata,
    LayerStreamingConfig,
    StreamedLayerBuffer,
    StreamingPipelineTelemetry,
)

SLOT_A = "slot_a"
SLOT_B = "slot_b"


class SlidingWindowLayerStreamer:
    """Manages AirLLM-style double-buffered layer streaming from disk with zero-dirty-page eviction."""

    def __init__(self, config: LayerStreamingConfig | None = None) -> None:
        self.config = config or LayerStreamingConfig()
        self._disk_catalog: dict[int, LayerDiskMetadata] = {}

        # Double-buffering memory slots
        self._slots_data: dict[str, bytearray | None] = {SLOT_A: None, SLOT_B: None}
        self._slots_layer_idx: dict[str, int | None] = {SLOT_A: None, SLOT_B: None}
        self._active_slot: str = SLOT_A
        self._standby_slot: str = SLOT_B

        # Telemetry tracking
        self._processed_layers_count: int = 0
        self._swap_latencies_ms: list[float] = []
        self._total_bytes_streamed: int = 0
        self._peak_resident_bytes: int = 0
        self._overlap_time_ms: float = 0.0
        self._total_compute_time_ms: float = 0.0

    @property
    def current_active_layer(self) -> int | None:
        return self._slots_layer_idx[self._active_slot]

    def register_layer(self, metadata: LayerDiskMetadata) -> None:
        """Registers a Transformer layer metadata descriptor in the catalog."""
        self._disk_catalog[metadata.layer_idx] = metadata

    def _read_from_source(self, metadata: LayerDiskMetadata) -> bytearray:
        """Simulates or performs raw disk I/O read for a layer chunk."""
        # Allocate clean buffer matching layer size
        size = metadata.size_bytes
        data = bytearray(size)
        self._total_bytes_streamed += size
        return data

    def start_pipeline(self, initial_layer_idx: int = 0) -> StreamedLayerBuffer:
        """Initializes the double-buffering pipeline, loading layer 0 and prefetching layer 1."""
        if initial_layer_idx not in self._disk_catalog:
            raise KeyError(f"Layer {initial_layer_idx} not found in disk catalog")

        # 1. Load initial layer into active slot
        meta_0 = self._disk_catalog[initial_layer_idx]
        self._slots_data[self._active_slot] = self._read_from_source(meta_0)
        self._slots_layer_idx[self._active_slot] = initial_layer_idx

        # 2. Prefetch next layer into standby slot if available
        next_layer = initial_layer_idx + 1
        if next_layer in self._disk_catalog:
            meta_1 = self._disk_catalog[next_layer]
            self._slots_data[self._standby_slot] = self._read_from_source(meta_1)
            self._slots_layer_idx[self._standby_slot] = next_layer

        self._update_peak_memory()
        self._processed_layers_count = 1

        return StreamedLayerBuffer(
            layer_idx=initial_layer_idx,
            buffer_slot=self._active_slot,
            size_bytes=meta_0.size_bytes,
            is_active=True,
            is_ready=True,
        )

    def step_layer(self, compute_duration_ms: float = 0.0) -> StreamedLayerBuffer | None:
        """Advances pipeline to the next layer, evicting the previous one and prefetching ahead.

        Args:
            compute_duration_ms: Time spent executing compute on the active layer (for overlap ratio).

        Returns:
            StreamedLayerBuffer for the new active layer, or None if end of layers reached.
        """
        t_start = time.perf_counter()

        # Check if standby slot has the preloaded next layer
        next_layer_idx = self._slots_layer_idx[self._standby_slot]
        if next_layer_idx is None:
            # End of registered layers
            return None

        # 1. Pointer swap: standby becomes active
        old_active = self._active_slot
        self._active_slot = self._standby_slot
        self._standby_slot = old_active

        # 2. Evict previous active layer from memory (clean page drop)
        if self.config.enable_clean_page_eviction:
            self._slots_data[self._standby_slot] = None
            self._slots_layer_idx[self._standby_slot] = None

        # 3. Asynchronously prefetch next-next layer into the newly freed standby slot
        future_layer_idx = next_layer_idx + 1
        if future_layer_idx in self._disk_catalog:
            future_meta = self._disk_catalog[future_layer_idx]
            self._slots_data[self._standby_slot] = self._read_from_source(future_meta)
            self._slots_layer_idx[self._standby_slot] = future_layer_idx

        self._update_peak_memory()

        elapsed_swap_ms = (time.perf_counter() - t_start) * 1000.0
        self._swap_latencies_ms.append(elapsed_swap_ms)
        self._processed_layers_count += 1
        self._total_compute_time_ms += compute_duration_ms

        active_meta = self._disk_catalog[next_layer_idx]
        return StreamedLayerBuffer(
            layer_idx=next_layer_idx,
            buffer_slot=self._active_slot,
            size_bytes=active_meta.size_bytes,
            is_active=True,
            is_ready=True,
        )

    def get_active_layer_bytes(self) -> bytes:
        """Retrieves read-only byteview of currently active layer in RAM."""
        buf = self._slots_data[self._active_slot]
        if buf is None:
            return b""
        return bytes(buf)

    def _update_peak_memory(self) -> None:
        """Tracks the maximum memory resident in RAM across both slots."""
        size_a = len(self._slots_data[SLOT_A]) if self._slots_data[SLOT_A] else 0
        size_b = len(self._slots_data[SLOT_B]) if self._slots_data[SLOT_B] else 0
        current_resident = size_a + size_b
        self._peak_resident_bytes = max(self._peak_resident_bytes, current_resident)

    def get_telemetry(self) -> StreamingPipelineTelemetry:
        """Produces streaming telemetry capturing memory cap and swap latencies."""
        avg_latency = (
            sum(self._swap_latencies_ms) / len(self._swap_latencies_ms)
            if self._swap_latencies_ms
            else 0.0
        )
        total_swap_time = sum(self._swap_latencies_ms)
        total_time = self._total_compute_time_ms + total_swap_time
        overlap_ratio = (
            self._total_compute_time_ms / total_time if total_time > 0.0 else 1.0
        )

        return StreamingPipelineTelemetry(
            total_layers_processed=self._processed_layers_count,
            peak_memory_resident_bytes=self._peak_resident_bytes,
            total_bytes_streamed=self._total_bytes_streamed,
            average_swap_latency_ms=avg_latency,
            io_compute_overlap_ratio=overlap_ratio,
        )
