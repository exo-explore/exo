import numpy as np
from numpy.typing import NDArray

from exo.shared.types.bitkv import (
    BitKVConfig,
    BitKVTelemetry,
    QuantizedKVTensors,
)

QUANT_LEVELS_2BIT: float = 3.0
EPSILON: float = 1e-7
VALUES_PER_BYTE_2BIT: int = 4
MASK_2BIT: int = 0x03


class BitKVQuantizer:
    """Core 2-bit asymmetric vector quantizer for Key (per-channel) and Value (per-token)."""

    @staticmethod
    def pack_2bit(uint8_arr: NDArray[np.uint8]) -> bytes:
        """Packs a 1D array of 2-bit integers (0..3) into compact bytes (4 values/byte)."""
        flat = uint8_arr.ravel()
        n = flat.size
        pad_size = (VALUES_PER_BYTE_2BIT - (n % VALUES_PER_BYTE_2BIT)) % VALUES_PER_BYTE_2BIT
        if pad_size > 0:
            flat = np.pad(flat, (0, pad_size), mode="constant", constant_values=0)

        reshaped = flat.reshape(-1, VALUES_PER_BYTE_2BIT)
        packed = (
            (reshaped[:, 0] & MASK_2BIT)
            | ((reshaped[:, 1] & MASK_2BIT) << 2)
            | ((reshaped[:, 2] & MASK_2BIT) << 4)
            | ((reshaped[:, 3] & MASK_2BIT) << 6)
        ).astype(np.uint8)
        return packed.tobytes()

    @staticmethod
    def unpack_2bit(packed_bytes: bytes, target_size: int) -> NDArray[np.uint8]:
        """Unpacks compact bytes back into an array of 2-bit integers (0..3)."""
        if not packed_bytes or target_size == 0:
            return np.empty(0, dtype=np.uint8)

        packed = np.frombuffer(packed_bytes, dtype=np.uint8)
        q0 = packed & MASK_2BIT
        q1 = (packed >> 2) & MASK_2BIT
        q2 = (packed >> 4) & MASK_2BIT
        q3 = (packed >> 6) & MASK_2BIT

        unpacked = np.column_stack([q0, q1, q2, q3]).ravel()
        return unpacked[:target_size]

    @classmethod
    def quantize_key_per_channel(
        cls, key_tensor: NDArray[np.floating]
    ) -> tuple[bytes, bytes, bytes]:
        """Quantizes Key tensor using per-channel statistics (across token dimension).

        Args:
            key_tensor: Tensor of shape (seq_len, num_heads, head_dim).

        Returns:
            Tuple of (packed_2bit_bytes, scales_float32_bytes, zeros_float32_bytes).
        """
        arr = np.asarray(key_tensor, dtype=np.float32)
        if arr.size == 0:
            return b"", b"", b""

        # Compute min/max per head and channel across the sequence dimension (axis=0)
        min_vals = np.min(arr, axis=0, keepdims=True)
        max_vals = np.max(arr, axis=0, keepdims=True)
        scales = np.maximum((max_vals - min_vals) / QUANT_LEVELS_2BIT, EPSILON)

        quantized = np.clip(np.round((arr - min_vals) / scales), 0, QUANT_LEVELS_2BIT).astype(np.uint8)
        packed_bytes = cls.pack_2bit(quantized)

        return (
            packed_bytes,
            scales.astype(np.float32).tobytes(),
            min_vals.astype(np.float32).tobytes(),
        )

    @classmethod
    def dequantize_key(
        cls,
        packed_bytes: bytes,
        scales_bytes: bytes,
        zeros_bytes: bytes,
        shape: tuple[int, int, int],
    ) -> NDArray[np.float32]:
        """Reconstructs float32 Key tensor from 2-bit per-channel representation."""
        seq_len, num_heads, head_dim = shape
        target_size = seq_len * num_heads * head_dim
        if target_size == 0 or not packed_bytes:
            return np.zeros(shape, dtype=np.float32)

        unpacked = cls.unpack_2bit(packed_bytes, target_size).astype(np.float32).reshape(shape)
        scales = np.frombuffer(scales_bytes, dtype=np.float32).reshape(1, num_heads, head_dim)
        min_vals = np.frombuffer(zeros_bytes, dtype=np.float32).reshape(1, num_heads, head_dim)

        return unpacked * scales + min_vals

    @classmethod
    def quantize_value_per_token(
        cls, value_tensor: NDArray[np.floating]
    ) -> tuple[bytes, bytes, bytes]:
        """Quantizes Value tensor using per-token statistics (across channel dimension).

        Args:
            value_tensor: Tensor of shape (seq_len, num_heads, head_dim).

        Returns:
            Tuple of (packed_2bit_bytes, scales_float32_bytes, zeros_float32_bytes).
        """
        arr = np.asarray(value_tensor, dtype=np.float32)
        if arr.size == 0:
            return b"", b"", b""

        # Compute min/max per token and head across head_dim (axis=2)
        min_vals = np.min(arr, axis=2, keepdims=True)
        max_vals = np.max(arr, axis=2, keepdims=True)
        scales = np.maximum((max_vals - min_vals) / QUANT_LEVELS_2BIT, EPSILON)

        quantized = np.clip(np.round((arr - min_vals) / scales), 0, QUANT_LEVELS_2BIT).astype(np.uint8)
        packed_bytes = cls.pack_2bit(quantized)

        return (
            packed_bytes,
            scales.astype(np.float32).tobytes(),
            min_vals.astype(np.float32).tobytes(),
        )

    @classmethod
    def dequantize_value(
        cls,
        packed_bytes: bytes,
        scales_bytes: bytes,
        zeros_bytes: bytes,
        shape: tuple[int, int, int],
    ) -> NDArray[np.float32]:
        """Reconstructs float32 Value tensor from 2-bit per-token representation."""
        seq_len, num_heads, head_dim = shape
        target_size = seq_len * num_heads * head_dim
        if target_size == 0 or not packed_bytes:
            return np.zeros(shape, dtype=np.float32)

        unpacked = cls.unpack_2bit(packed_bytes, target_size).astype(np.float32).reshape(shape)
        scales = np.frombuffer(scales_bytes, dtype=np.float32).reshape(seq_len, num_heads, 1)
        min_vals = np.frombuffer(zeros_bytes, dtype=np.float32).reshape(seq_len, num_heads, 1)

        return unpacked * scales + min_vals


class BitKVCacheManager:
    """Manages asymmetric 2-bit quantized KV cache with high-precision sliding window."""

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        config: BitKVConfig | None = None,
    ) -> None:
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.config = config or BitKVConfig()

        # Storage for fully quantized historical tokens
        self._quantized_seq_len: int = 0
        self._key_packed_bytes: bytes = b""
        self._key_scales_bytes: bytes = b""
        self._key_zeros_bytes: bytes = b""
        self._value_packed_bytes: bytes = b""
        self._value_scales_bytes: bytes = b""
        self._value_zeros_bytes: bytes = b""

        # Recent high-precision sliding window buffer
        self._recent_keys: list[NDArray[np.float32]] = []
        self._recent_values: list[NDArray[np.float32]] = []

    @property
    def total_tokens(self) -> int:
        return self._quantized_seq_len + len(self._recent_keys)

    def append_token_kv(
        self, key: NDArray[np.floating], value: NDArray[np.floating]
    ) -> None:
        """Appends a single token's KV tensors of shape (num_heads, head_dim) or (1, num_heads, head_dim)."""
        k = np.asarray(key, dtype=np.float32)
        v = np.asarray(value, dtype=np.float32)
        if k.ndim == 3:
            k = k.squeeze(0)
        if v.ndim == 3:
            v = v.squeeze(0)

        self._recent_keys.append(k)
        self._recent_values.append(v)

        # When recent window overflows, archive older tokens into 2-bit storage
        if len(self._recent_keys) > self.config.window_size:
            self._compress_sliding_window()

    def _compress_sliding_window(self) -> None:
        """Compresses tokens that have shifted beyond the recent window."""
        excess_count = len(self._recent_keys) - self.config.window_size
        if excess_count <= 0:
            return

        # Extract tokens to compress
        tokens_to_compress_k = np.stack(self._recent_keys[:excess_count], axis=0)  # (excess, heads, dim)
        tokens_to_compress_v = np.stack(self._recent_values[:excess_count], axis=0)

        # Retain only the active sliding window
        self._recent_keys = self._recent_keys[excess_count:]
        self._recent_values = self._recent_values[excess_count:]

        # Merge with existing historical quantized tokens
        if self._quantized_seq_len > 0:
            existing_k = BitKVQuantizer.dequantize_key(
                self._key_packed_bytes,
                self._key_scales_bytes,
                self._key_zeros_bytes,
                (self._quantized_seq_len, self.num_heads, self.head_dim),
            )
            existing_v = BitKVQuantizer.dequantize_value(
                self._value_packed_bytes,
                self._value_scales_bytes,
                self._value_zeros_bytes,
                (self._quantized_seq_len, self.num_heads, self.head_dim),
            )
            all_k = np.concatenate([existing_k, tokens_to_compress_k], axis=0)
            all_v = np.concatenate([existing_v, tokens_to_compress_v], axis=0)
        else:
            all_k = tokens_to_compress_k
            all_v = tokens_to_compress_v

        new_quantized_len = all_k.shape[0]

        # Re-quantize combined historical tokens
        (
            self._key_packed_bytes,
            self._key_scales_bytes,
            self._key_zeros_bytes,
        ) = BitKVQuantizer.quantize_key_per_channel(all_k)

        (
            self._value_packed_bytes,
            self._value_scales_bytes,
            self._value_zeros_bytes,
        ) = BitKVQuantizer.quantize_value_per_token(all_v)

        self._quantized_seq_len = new_quantized_len

    def get_full_kv(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Reconstructs the full sequence Key and Value matrices for multi-head attention."""
        if self.total_tokens == 0:
            empty = np.empty((0, self.num_heads, self.head_dim), dtype=np.float32)
            return empty, empty

        # 1. Dequantize historical pool
        if self._quantized_seq_len > 0:
            hist_k = BitKVQuantizer.dequantize_key(
                self._key_packed_bytes,
                self._key_scales_bytes,
                self._key_zeros_bytes,
                (self._quantized_seq_len, self.num_heads, self.head_dim),
            )
            hist_v = BitKVQuantizer.dequantize_value(
                self._value_packed_bytes,
                self._value_scales_bytes,
                self._value_zeros_bytes,
                (self._quantized_seq_len, self.num_heads, self.head_dim),
            )
        else:
            hist_k = np.empty((0, self.num_heads, self.head_dim), dtype=np.float32)
            hist_v = np.empty((0, self.num_heads, self.head_dim), dtype=np.float32)

        # 2. Get recent window tokens
        if self._recent_keys:
            recent_k = np.stack(self._recent_keys, axis=0)
            recent_v = np.stack(self._recent_values, axis=0)
        else:
            recent_k = np.empty((0, self.num_heads, self.head_dim), dtype=np.float32)
            recent_v = np.empty((0, self.num_heads, self.head_dim), dtype=np.float32)

        if hist_k.size == 0:
            return recent_k, recent_v
        if recent_k.size == 0:
            return hist_k, hist_v

        full_k = np.concatenate([hist_k, recent_k], axis=0)
        full_v = np.concatenate([hist_v, recent_v], axis=0)
        return full_k, full_v

    def snapshot_tensors(self) -> QuantizedKVTensors:
        """Captures an immutable snapshot of current quantized and recent window state."""
        recent_k_bytes = (
            np.stack(self._recent_keys, axis=0).astype(np.float16).tobytes()
            if self._recent_keys
            else b""
        )
        recent_v_bytes = (
            np.stack(self._recent_values, axis=0).astype(np.float16).tobytes()
            if self._recent_values
            else b""
        )

        return QuantizedKVTensors(
            layer_idx=self.layer_idx,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            quantized_seq_len=self._quantized_seq_len,
            recent_seq_len=len(self._recent_keys),
            key_packed_bytes=self._key_packed_bytes,
            key_scales_bytes=self._key_scales_bytes,
            key_zeros_bytes=self._key_zeros_bytes,
            value_packed_bytes=self._value_packed_bytes,
            value_scales_bytes=self._value_scales_bytes,
            value_zeros_bytes=self._value_zeros_bytes,
            recent_keys_bytes=recent_k_bytes,
            recent_values_bytes=recent_v_bytes,
        )

    def get_telemetry(self) -> BitKVTelemetry:
        """Produces compression telemetry for current layer."""
        snapshot = self.snapshot_tensors()
        return BitKVTelemetry(
            layer_idx=self.layer_idx,
            total_tokens=self.total_tokens,
            uncompressed_fp16_bytes=snapshot.uncompressed_fp16_bytes,
            compressed_bytes=snapshot.total_compressed_bytes,
            compression_ratio=snapshot.compression_ratio,
            effective_bits_per_param=snapshot.effective_bits_per_param,
        )
