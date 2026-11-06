from exo.utils.pydantic_ext import FrozenModel

BYTES_PER_FP16: int = 2
BITS_PER_UINT64: int = 64
DEFAULT_ENERGY_REDUCTION_RATIO: float = 0.95  # BNN XNOR logic consumes ~95% less energy than FP16 ALU


class BNNPackedTensor(FrozenModel):
    """Packed 1.0-bit binary weight tensor (64 binary parameters per uint64)."""

    tensor_name: str
    shape: tuple[int, int]
    total_elements: int
    packed_uint64_bytes: bytes
    scale: float = 1.0
    effective_bits_per_param: float = 1.0

    @property
    def total_bytes(self) -> int:
        return len(self.packed_uint64_bytes)

    @property
    def compression_ratio_vs_fp16(self) -> float:
        uncompressed = self.total_elements * BYTES_PER_FP16
        if self.total_bytes <= 0:
            return 0.0
        return uncompressed / self.total_bytes


class BNNTelemetry(FrozenModel):
    """Telemetry report capturing BNN 1.0-bit compression and XNOR execution energy savings."""

    tensor_name: str
    total_parameters: int
    uncompressed_fp16_bytes: int
    bnn_bytes: int
    compression_ratio_vs_fp16: float
    popcount_operations_count: int
    estimated_energy_reduction_ratio: float = DEFAULT_ENERGY_REDUCTION_RATIO
