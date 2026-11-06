from exo.utils.pydantic_ext import FrozenModel


class ANEShapeBucket(FrozenModel):
    """Static shape specification for pre-compiled Apple Neural Engine compute graphs."""

    bucket_id: str
    batch_size: int = 1
    chunk_size: int = 64
    hidden_dim: int = 4096
    compiled_model_path: str = ""

    @property
    def max_tokens(self) -> int:
        return self.batch_size * self.chunk_size


class IOSurfaceBufferDescriptor(FrozenModel):
    """Memory descriptor for Zero-Copy DMA page mapping between host memory and ANE."""

    surface_id: str
    width: int
    height: int
    bytes_per_row: int
    allocation_size_bytes: int
    is_dma_mapped: bool = True
    alignment_bytes: int = 64


class ANETelemetry(FrozenModel):
    """Execution telemetry capturing ANE bucket alignment, DMA zero-copy latency, and power metrics."""

    execution_id: str
    bucket: ANEShapeBucket
    input_tokens: int
    padded_tokens: int
    dma_transfer_time_saved_ms: float
    inference_latency_ms: float
    estimated_power_watts: float = 2.4

    @property
    def padding_overhead_ratio(self) -> float:
        if self.padded_tokens <= 0:
            return 0.0
        return max(0.0, (self.padded_tokens - self.input_tokens) / self.padded_tokens)

    @property
    def tokens_per_second(self) -> float:
        if self.inference_latency_ms <= 0:
            return 0.0
        return (self.input_tokens / (self.inference_latency_ms * 1e-3))
