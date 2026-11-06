from typing import Literal

from exo.utils.pydantic_ext import FrozenModel

StreamingMode = Literal["double_buffered", "mmap_direct"]
DEFAULT_BUFFER_SIZE_BYTES: int = 512 * 1024 * 1024  # 512 MB default layer cap


class LayerStreamingConfig(FrozenModel):
    """Configuration for AirLLM-style sliding-window layer streaming from disk."""

    mode: StreamingMode = "double_buffered"
    max_buffer_bytes: int = DEFAULT_BUFFER_SIZE_BYTES
    prefetch_ahead_count: int = 1
    enable_clean_page_eviction: bool = True


class LayerDiskMetadata(FrozenModel):
    """Location and size descriptor for a Transformer layer stored on disk."""

    layer_idx: int
    file_path: str
    offset_bytes: int = 0
    size_bytes: int = 0
    sha256_checksum: str = ""


class StreamedLayerBuffer(FrozenModel):
    """Status descriptor of an in-memory double-buffering slot."""

    layer_idx: int
    buffer_slot: str
    size_bytes: int
    is_active: bool = False
    is_ready: bool = False


class StreamingPipelineTelemetry(FrozenModel):
    """Telemetry report capturing memory residency, swap latency, and I/O overlap ratio."""

    total_layers_processed: int
    peak_memory_resident_bytes: int
    total_bytes_streamed: int
    average_swap_latency_ms: float
    io_compute_overlap_ratio: float
