from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field

from exo.utils.pydantic_ext import FrozenModel

ActivationPrecision = Literal["int4", "fp16", "delta_int4"]

BYTES_PER_FP16_ELEMENT = 2


class CompressedActivationPayload(FrozenModel):
    """Wire payload for extreme cross-node activation tensor compression."""

    sequence_id: str
    step: int
    num_tokens: int
    hidden_dim: int
    precision: ActivationPrecision = "int4"
    # Per-token quantization scales and minimum values (length == num_tokens)
    scale_factors: Sequence[float]
    min_values: Sequence[float]
    # Packed byte stream where each uint8 byte holds two 4-bit nibbles
    packed_quantized_bytes: bytes
    is_delta_encoded: bool = False
    metadata: Mapping[str, str] = Field(default_factory=dict)

    @property
    def original_fp16_size_in_bytes(self) -> int:
        """Returns the uncompressed FP16 tensor size in bytes."""
        return self.num_tokens * self.hidden_dim * BYTES_PER_FP16_ELEMENT

    @property
    def compressed_payload_size_in_bytes(self) -> int:
        """Returns the actual compressed byte stream length."""
        # packed bytes + per-token scale (4 bytes float) + min_value (4 bytes float)
        header_overhead = self.num_tokens * (4 + 4)
        return len(self.packed_quantized_bytes) + header_overhead

    @property
    def compression_ratio(self) -> float:
        """Calculates size reduction ratio compared to uncompressed FP16."""
        if self.original_fp16_size_in_bytes == 0:
            return 1.0
        return self.compressed_payload_size_in_bytes / self.original_fp16_size_in_bytes
