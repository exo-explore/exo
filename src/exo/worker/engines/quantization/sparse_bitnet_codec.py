import numpy as np
from numpy.typing import NDArray

from exo.shared.types.sparse_bitnet import (
    SparseBitNetTensor,
    SparseCompressionTelemetry,
)


class SparseBitNetCodec:
    """Encodes and decodes ternary BitNet weights into ultra-compact sparse 0.8-bit format."""

    @classmethod
    def compress_ternary(
        cls,
        weights: NDArray[np.floating | np.integer],
        tensor_name: str = "bitnet_weight",
        scale: float = 1.0,
    ) -> tuple[SparseBitNetTensor, SparseCompressionTelemetry]:
        """Compresses ternary weights {-1, 0, +1} into 1-bit non-zero bitmask and 1-bit sign stream.

        Returns:
            Tuple of (SparseBitNetTensor, SparseCompressionTelemetry).
        """
        arr = np.asarray(weights, dtype=np.float32)
        shape = (int(arr.shape[0]), int(arr.shape[1])) if arr.ndim == 2 else (1, int(arr.size))
        flat = arr.flatten()
        total_elements = len(flat)

        # 1. Non-zero boolean bitmask
        non_zero_mask = (flat != 0.0)
        non_zero_count = int(np.count_nonzero(non_zero_mask))
        bitmask_bytes = np.packbits(non_zero_mask).tobytes()

        # 2. Sign stream for non-zero entries: +1 -> 1, -1 -> 0
        if non_zero_count > 0:
            non_zero_vals = flat[non_zero_mask]
            signs = (non_zero_vals > 0.0).astype(np.uint8)
            sign_stream_bytes = np.packbits(signs).tobytes()
        else:
            sign_stream_bytes = b""

        tensor = SparseBitNetTensor(
            tensor_name=tensor_name,
            shape=shape,
            total_elements=total_elements,
            non_zero_count=non_zero_count,
            bitmask_bytes=bitmask_bytes,
            sign_stream_bytes=sign_stream_bytes,
            scale=scale,
            format="bitmask_sign",
        )

        uncompressed_fp16 = total_elements * 2
        telemetry = SparseCompressionTelemetry(
            tensor_name=tensor_name,
            uncompressed_fp16_bytes=uncompressed_fp16,
            compressed_bytes=tensor.total_bytes,
            effective_bits_per_param=tensor.effective_bits_per_param,
            sparsity_ratio=tensor.sparsity_ratio,
        )

        return tensor, telemetry

    @classmethod
    def decompress_to_ternary(cls, sparse_tensor: SparseBitNetTensor) -> NDArray[np.float32]:
        """Decompresses bitmask and sign stream back into uncompressed ternary matrix."""
        total = sparse_tensor.total_elements
        flat_result = np.zeros(total, dtype=np.float32)

        if sparse_tensor.non_zero_count == 0:
            return flat_result.reshape(sparse_tensor.shape)

        # Unpack non-zero mask
        raw_mask_bytes = np.frombuffer(sparse_tensor.bitmask_bytes, dtype=np.uint8)
        mask_bits = np.unpackbits(raw_mask_bytes)[:total].astype(bool)

        # Unpack sign stream
        raw_sign_bytes = np.frombuffer(sparse_tensor.sign_stream_bytes, dtype=np.uint8)
        sign_bits = np.unpackbits(raw_sign_bytes)[: sparse_tensor.non_zero_count]

        # Convert sign bits: 1 -> +1.0, 0 -> -1.0
        sign_values = np.where(sign_bits == 1, 1.0, -1.0).astype(np.float32)

        # Reconstruct scaled values
        flat_result[mask_bits] = sign_values * sparse_tensor.scale
        return flat_result.reshape(sparse_tensor.shape)
