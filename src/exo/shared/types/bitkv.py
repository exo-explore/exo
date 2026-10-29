from typing import Literal

from exo.utils.pydantic_ext import FrozenModel

BitKVBits = Literal[2, 4]
BYTES_PER_FP16: int = 2
KV_TENSORS_COUNT: int = 2  # Key and Value tensors


class BitKVConfig(FrozenModel):
    """Configuration for KIVI asymmetric 2-bit KV Cache quantization."""

    bits: BitKVBits = 2
    window_size: int = 32
    group_size: int = 32
    enable_per_channel_key: bool = True
    enable_per_token_value: bool = True


class QuantizedKVTensors(FrozenModel):
    """Compressed Key and Value cache tensors adhering to asymmetric KIVI distribution."""

    layer_idx: int
    num_heads: int
    head_dim: int
    quantized_seq_len: int
    recent_seq_len: int
    key_packed_bytes: bytes
    key_scales_bytes: bytes
    key_zeros_bytes: bytes
    value_packed_bytes: bytes
    value_scales_bytes: bytes
    value_zeros_bytes: bytes
    recent_keys_bytes: bytes
    recent_values_bytes: bytes

    @property
    def total_seq_len(self) -> int:
        return self.quantized_seq_len + self.recent_seq_len

    @property
    def total_compressed_bytes(self) -> int:
        return (
            len(self.key_packed_bytes)
            + len(self.key_scales_bytes)
            + len(self.key_zeros_bytes)
            + len(self.value_packed_bytes)
            + len(self.value_scales_bytes)
            + len(self.value_zeros_bytes)
            + len(self.recent_keys_bytes)
            + len(self.recent_values_bytes)
        )

    @property
    def uncompressed_fp16_bytes(self) -> int:
        total_elements = self.total_seq_len * self.num_heads * self.head_dim * KV_TENSORS_COUNT
        return total_elements * BYTES_PER_FP16

    @property
    def compression_ratio(self) -> float:
        compressed = self.total_compressed_bytes
        if compressed <= 0:
            return 0.0
        return self.uncompressed_fp16_bytes / compressed

    @property
    def effective_bits_per_param(self) -> float:
        total_elements = self.total_seq_len * self.num_heads * self.head_dim * KV_TENSORS_COUNT
        if total_elements <= 0:
            return 0.0
        return (self.total_compressed_bytes * 8.0) / total_elements


class BitKVTelemetry(FrozenModel):
    """Telemetry report capturing KV Cache compression performance for long context."""

    layer_idx: int
    total_tokens: int
    uncompressed_fp16_bytes: int
    compressed_bytes: int
    compression_ratio: float
    effective_bits_per_param: float
