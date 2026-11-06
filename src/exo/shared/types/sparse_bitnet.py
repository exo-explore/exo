from typing import Literal

from exo.utils.pydantic_ext import FrozenModel

SparseBitNetFormat = Literal["bitmask_sign", "rle_compressed"]


class SparseBitNetTensor(FrozenModel):
    """Encapsulates a sparse ternary weight matrix compressed with bitmask and sign stream."""

    tensor_name: str
    shape: tuple[int, int]
    total_elements: int
    non_zero_count: int
    bitmask_bytes: bytes
    sign_stream_bytes: bytes
    scale: float = 1.0
    format: SparseBitNetFormat = "bitmask_sign"

    @property
    def sparsity_ratio(self) -> float:
        """Percentage of weights that are zero (0.0 to 1.0)."""
        if self.total_elements <= 0:
            return 0.0
        return max(0.0, 1.0 - (self.non_zero_count / self.total_elements))

    @property
    def total_bytes(self) -> int:
        return len(self.bitmask_bytes) + len(self.sign_stream_bytes)

    @property
    def effective_bits_per_param(self) -> float:
        """Effective bits allocated per weight parameter."""
        if self.total_elements <= 0:
            return 0.0
        return (self.total_bytes * 8.0) / self.total_elements


class SparseCompressionTelemetry(FrozenModel):
    """Telemetry capturing compression efficiency and memory savings for BitNet."""

    tensor_name: str
    uncompressed_fp16_bytes: int
    compressed_bytes: int
    effective_bits_per_param: float
    sparsity_ratio: float

    @property
    def compression_ratio_vs_fp16(self) -> float:
        if self.compressed_bytes <= 0:
            return 0.0
        return self.uncompressed_fp16_bytes / self.compressed_bytes
