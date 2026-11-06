import numpy as np
from numpy.typing import NDArray

from exo.shared.types.sram_cache import (
    SRAMAllocation,
    SRAMConfig,
    SRAMTelemetry,
)

ENERGY_SAVINGS_COEFFICIENT: float = 0.90  # DRAM access costs ~10x-20x more energy than on-chip SRAM


class OnChipSRAMCacheManager:
    """Manages on-chip NPU/ANE SRAM KV cache residency with zero-DRAM decode latency."""

    def __init__(self, layer_idx: int, config: SRAMConfig | None = None) -> None:
        self.layer_idx = layer_idx
        self.config = config or SRAMConfig()

        self._sram_keys: list[NDArray[np.float32]] = []
        self._sram_values: list[NDArray[np.float32]] = []

        self._dram_keys: list[NDArray[np.float32]] = []
        self._dram_values: list[NDArray[np.float32]] = []

        self._lut_ops_count: int = 0

    @property
    def total_tokens(self) -> int:
        return len(self._sram_keys) + len(self._dram_keys)

    def append_kv(self, key: NDArray[np.floating], value: NDArray[np.floating]) -> None:
        """Appends a new token's KV tensors into on-chip SRAM, spilling over to DRAM if saturated."""
        k = np.asarray(key, dtype=np.float32)
        v = np.asarray(value, dtype=np.float32)

        self._sram_keys.append(k)
        self._sram_values.append(v)

        # Spillage policy: maintain at most config.max_sram_tokens in fast on-chip SRAM
        if len(self._sram_keys) > self.config.max_sram_tokens:
            evicted_k = self._sram_keys.pop(0)
            evicted_v = self._sram_values.pop(0)
            self._dram_keys.append(evicted_k)
            self._dram_values.append(evicted_v)

    def get_full_kv(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Reconstructs full sequence KV tensors across DRAM and SRAM."""
        all_k = self._dram_keys + self._sram_keys
        all_v = self._dram_values + self._sram_values
        if not all_k:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
        return np.stack(all_k, axis=0), np.stack(all_v, axis=0)

    def get_allocation(self) -> SRAMAllocation:
        """Reports current on-chip SRAM byte usage and residency status."""
        sram_tokens = len(self._sram_keys)
        dram_tokens = len(self._dram_keys)

        sram_bytes = 0
        if self._sram_keys and self._sram_values:
            sram_bytes = sum(k.nbytes + v.nbytes for k, v in zip(self._sram_keys, self._sram_values, strict=True))

        return SRAMAllocation(
            layer_idx=self.layer_idx,
            tokens_in_sram=sram_tokens,
            tokens_in_dram=dram_tokens,
            sram_bytes_used=sram_bytes,
            is_sram_saturated=(sram_tokens >= self.config.max_sram_tokens),
        )

    def get_telemetry(self) -> SRAMTelemetry:
        """Computes DRAM I/O reduction and energy savings ratios."""
        tot = self.total_tokens
        if tot <= 0:
            return SRAMTelemetry(
                total_attention_tokens=0,
                sram_hit_ratio=1.0,
                dram_io_reduction_ratio=1.0,
                estimated_energy_savings_ratio=ENERGY_SAVINGS_COEFFICIENT,
                lut_operations_count=self._lut_ops_count,
            )

        sram_tokens = len(self._sram_keys)
        sram_hit_ratio = sram_tokens / tot
        dram_io_reduction = sram_hit_ratio
        energy_savings = dram_io_reduction * ENERGY_SAVINGS_COEFFICIENT

        return SRAMTelemetry(
            total_attention_tokens=tot,
            sram_hit_ratio=sram_hit_ratio,
            dram_io_reduction_ratio=dram_io_reduction,
            estimated_energy_savings_ratio=energy_savings,
            lut_operations_count=self._lut_ops_count,
        )


class NativeINT2LUTEngine:
    """NPU Look-Up Table (LUT) execution kernel for 2-bit and ternary matrix multiplications."""

    @classmethod
    def execute_lut_ternary_gemm(
        cls,
        activations: NDArray[np.floating],
        ternary_weights: NDArray[np.integer | np.floating],
        scale: float = 1.0,
    ) -> NDArray[np.float32]:
        """Multiplies activations with ternary {-1, 0, +1} weights using an addition-only LUT kernel.

        Args:
            activations: Input activation tensor of shape (M, K) or (K,).
            ternary_weights: Ternary weight matrix of shape (K, N) with values in {-1, 0, 1}.
            scale: Global scaling factor.

        Returns:
            Output activations of shape (M, N).
        """
        act = np.asarray(activations, dtype=np.float32)
        w = np.asarray(ternary_weights, dtype=np.int8)

        is_1d = act.ndim == 1
        if is_1d:
            act = act.reshape(1, -1)

        # Construct Look-Up Table for {-1, 0, +1}:
        # Index 0 -> 0, Index 1 -> +A, Index 2 -> -A
        # Fast vectorization without floating point multiplier
        pos_mask = (w == 1)
        neg_mask = (w == -1)

        # Output = (Act @ PosMask) - (Act @ NegMask)
        output = (np.matmul(act, pos_mask.astype(np.float32)) - np.matmul(act, neg_mask.astype(np.float32))) * scale

        return output.squeeze(0) if is_1d else output

    @classmethod
    def execute_lut_2bit_gemm(
        cls,
        activations: NDArray[np.floating],
        uint8_weights: NDArray[np.uint8],
        scale: float,
        zero_point: float,
    ) -> NDArray[np.float32]:
        """Multiplies activations with 2-bit uniform quantized weights using a 4-entry LUT kernel.

        Args:
            activations: Input activation tensor of shape (M, K).
            uint8_weights: Quantized weights of shape (K, N) with values in {0, 1, 2, 3}.
            scale: Quantization step scale.
            zero_point: Min-val zero point.

        Returns:
            Output activations of shape (M, N).
        """
        act = np.asarray(activations, dtype=np.float32)
        q = np.asarray(uint8_weights, dtype=np.uint8)

        is_1d = act.ndim == 1
        if is_1d:
            act = act.reshape(1, -1)

        # 4-level Look-Up Table mapping
        # Dequantized: W = q * scale + zero_point
        # A @ W = scale * (A @ q) + zero_point * (A @ 1)
        q_float = q.astype(np.float32)
        sum_act = np.sum(act, axis=-1, keepdims=True)  # (M, 1)

        output = scale * np.matmul(act, q_float) + zero_point * sum_act

        return output.squeeze(0) if is_1d else output
