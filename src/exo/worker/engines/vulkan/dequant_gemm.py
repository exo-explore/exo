import numpy as np
from numpy.typing import NDArray


class VulkanDequantGemmEngine:
    """Fused Dequant-GEMM execution engine modeling Vulkan SPIR-V compute kernels.

    Unpacks 2-bit and 4-bit weights on-the-fly in registers to avoid VRAM writeback bottlenecks.
    """

    @staticmethod
    def pack_4bit_weights(int4_weights: NDArray[np.integer]) -> NDArray[np.uint8]:
        """Packs two signed/unsigned INT4 weights (range [-8, 7] or [0, 15]) into one uint8 byte.

        Assumes input shape is (K, N) where K is even.
        """
        arr = np.asarray(int4_weights, dtype=np.uint8) & 0x0F
        k, _ = arr.shape
        if k % 2 != 0:
            arr = np.pad(arr, ((0, 1), (0, 0)), mode="constant", constant_values=0)
            k += 1

        even_rows = arr[0::2, :]
        odd_rows = arr[1::2, :]
        packed = even_rows | (odd_rows << 4)
        return packed

    @staticmethod
    def pack_2bit_weights(int2_weights: NDArray[np.integer]) -> NDArray[np.uint8]:
        """Packs four 2-bit weights into one uint8 byte.

        Assumes input shape is (K, N) where K is divisible by 4.
        """
        arr = np.asarray(int2_weights, dtype=np.uint8) & 0x03
        k, _ = arr.shape
        pad_k = (4 - (k % 4)) % 4
        if pad_k > 0:
            arr = np.pad(arr, ((0, pad_k), (0, 0)), mode="constant", constant_values=0)

        r0 = arr[0::4, :]
        r1 = arr[1::4, :] << 2
        r2 = arr[2::4, :] << 4
        r3 = arr[3::4, :] << 6
        packed = r0 | r1 | r2 | r3
        return packed

    @classmethod
    def execute_fused_gemm_4bit(
        cls,
        activations: NDArray[np.floating],
        packed_weights: NDArray[np.uint8],
        scales: NDArray[np.floating],
        zero_points: NDArray[np.floating] | None = None,
        original_k: int | None = None,
    ) -> NDArray[np.float32]:
        """Executes fused dequantization matrix multiplication for 4-bit weights.

        Args:
            activations: shape (M, K)
            packed_weights: shape (K // 2, N)
            scales: per-column scale, shape (N,) or (1, N)
            zero_points: per-column zero-point, shape (N,) or (1, N)
            original_k: original unpadded K dimension length

        Returns:
            Output activations: shape (M, N)
        """
        act = np.asarray(activations, dtype=np.float32)
        _, k = act.shape
        packed = np.asarray(packed_weights, dtype=np.uint8)
        packed_k, n = packed.shape

        unpacked_k = packed_k * 2
        actual_k = original_k if original_k is not None else k

        # On-the-fly register unpack simulation
        low_nibbles = packed & 0x0F
        high_nibbles = (packed >> 4) & 0x0F

        unpacked_weights = np.empty((unpacked_k, n), dtype=np.float32)
        unpacked_weights[0::2, :] = low_nibbles
        unpacked_weights[1::2, :] = high_nibbles

        # Trim padding if necessary
        weights_float = unpacked_weights[:actual_k, :]

        scale_arr = np.asarray(scales, dtype=np.float32).reshape(1, n)
        zp_arr = (
            np.asarray(zero_points, dtype=np.float32).reshape(1, n)
            if zero_points is not None
            else np.zeros((1, n), dtype=np.float32)
        )

        dequant_weights = (weights_float - zp_arr) * scale_arr

        # Fused GEMM dot product: Output = Activation @ DequantWeights
        return np.matmul(act[:, :actual_k], dequant_weights)

    @classmethod
    def execute_fused_gemm_2bit(
        cls,
        activations: NDArray[np.floating],
        packed_weights: NDArray[np.uint8],
        scales: NDArray[np.floating],
        original_k: int | None = None,
    ) -> NDArray[np.float32]:
        """Executes fused dequantization matrix multiplication for 2-bit weights."""
        act = np.asarray(activations, dtype=np.float32)
        _, k = act.shape
        packed = np.asarray(packed_weights, dtype=np.uint8)
        packed_k, n = packed.shape

        unpacked_k = packed_k * 4
        actual_k = original_k if original_k is not None else k

        r0 = packed & 0x03
        r1 = (packed >> 2) & 0x03
        r2 = (packed >> 4) & 0x03
        r3 = (packed >> 6) & 0x03

        unpacked = np.empty((unpacked_k, n), dtype=np.float32)
        unpacked[0::4, :] = r0
        unpacked[1::4, :] = r1
        unpacked[2::4, :] = r2
        unpacked[3::4, :] = r3

        # Map {0, 1, 2, 3} to ternary / 2-bit values {-1, 0, 1}
        weights_float = (unpacked[:actual_k, :] - 1.0)
        scale_arr = np.asarray(scales, dtype=np.float32).reshape(1, n)
        dequant_weights = weights_float * scale_arr

        return np.matmul(act[:, :actual_k], dequant_weights)
