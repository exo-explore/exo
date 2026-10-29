import numpy as np
from numpy.typing import NDArray

from exo.routing.activation_quantizer import DynamicInt4ActivationQuantizer
from exo.routing.temporal_delta import (
    TemporalDeltaCompressor,
    TemporalDeltaDecompressor,
)
from exo.shared.types.activation_compression import CompressedActivationPayload

HIGH_BANDWIDTH_THRESHOLD_MBPS = 1000.0


class ActivationCompressionManager:
    """Manages adaptive activation compression based on network conditions and pipeline stage."""

    def __init__(self, network_bandwidth_mbps: float = 100.0) -> None:
        self._bandwidth_mbps = network_bandwidth_mbps
        self._compressor = TemporalDeltaCompressor()
        self._decompressor = TemporalDeltaDecompressor()
        self._total_original_bytes: int = 0
        self._total_compressed_bytes: int = 0

    @property
    def should_compress(self) -> bool:
        """Determines if compression is beneficial for current network bandwidth."""
        return self._bandwidth_mbps < HIGH_BANDWIDTH_THRESHOLD_MBPS

    @property
    def total_bytes_saved(self) -> int:
        return max(0, self._total_original_bytes - self._total_compressed_bytes)

    @property
    def overall_saving_percentage(self) -> float:
        if self._total_original_bytes == 0:
            return 0.0
        return (self.total_bytes_saved / self._total_original_bytes) * 100.0

    def reset_sequence(self) -> None:
        """Resets sequence-level delta compression buffers."""
        self._compressor.reset()
        self._decompressor.reset()

    def process_outbound_activations(
        self,
        hidden_states: NDArray[np.floating],
        sequence_id: str = "seq_default",
        step: int = 0,
    ) -> CompressedActivationPayload:
        """Compresses outbound activations before wire transmission."""
        array = np.asarray(hidden_states, dtype=np.float32)
        if array.ndim == 1:
            array = array[np.newaxis, :]
        num_tokens = array.shape[0]

        if num_tokens == 1:
            # Single-token autoregressive decoding step: apply temporal delta compression
            payload = self._compressor.compress_step(array, sequence_id=sequence_id, step=step)
        else:
            # Multi-token prompt prefill step: apply dynamic INT4 quantization
            payload = DynamicInt4ActivationQuantizer.quantize(
                array, sequence_id=sequence_id, step=step
            )

        self._total_original_bytes += payload.original_fp16_size_in_bytes
        self._total_compressed_bytes += payload.compressed_payload_size_in_bytes
        return payload

    def process_inbound_activations(
        self,
        payload: CompressedActivationPayload,
    ) -> NDArray[np.float32]:
        """Decompresses inbound activations received from network socket."""
        if payload.num_tokens == 1:
            # Single-token decoding step: maintain decompressor baseline & delta accumulation
            return self._decompressor.decompress_step(payload)
        return DynamicInt4ActivationQuantizer.dequantize(payload)
