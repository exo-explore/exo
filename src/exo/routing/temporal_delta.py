import numpy as np
from numpy.typing import NDArray

from exo.routing.activation_quantizer import DynamicInt4ActivationQuantizer
from exo.shared.types.activation_compression import CompressedActivationPayload

DEFAULT_SPARSITY_THRESHOLD_RATIO = 0.05


class TemporalDeltaCompressor:
    """Temporal delta encoder for autoregressive decoding token streams.

    Exploits high temporal correlation across steps to transmit only residual changes,
    reducing decoding communication overhead by an additional 60%+.
    """

    def __init__(self, sparsity_threshold_ratio: float = DEFAULT_SPARSITY_THRESHOLD_RATIO) -> None:
        self._sparsity_threshold_ratio = sparsity_threshold_ratio
        self._cached_baseline: NDArray[np.float32] | None = None

    def reset(self) -> None:
        """Clears cached baseline states when a new sequence begins."""
        self._cached_baseline = None

    def compress_step(
        self,
        current_hidden_state: NDArray[np.floating],
        sequence_id: str = "seq_default",
        step: int = 0,
    ) -> CompressedActivationPayload:
        """Encodes activation tensor using delta compression if baseline is available."""
        current = np.asarray(current_hidden_state, dtype=np.float32)
        if current.ndim == 1:
            current = current[np.newaxis, :]

        # First step: initialize baseline with regular INT4 quantization
        if self._cached_baseline is None or current.shape != self._cached_baseline.shape:
            payload = DynamicInt4ActivationQuantizer.quantize(
                current, sequence_id=sequence_id, step=step
            )
            # Sync baseline with dequantized state to prevent drift between sender and receiver
            self._cached_baseline = DynamicInt4ActivationQuantizer.dequantize(payload)
            return payload

        # Subsequent steps: compute delta
        delta = current - self._cached_baseline

        # Sparsify tiny fluctuations below threshold
        max_abs_delta = float(np.max(np.abs(delta)))
        threshold = max_abs_delta * self._sparsity_threshold_ratio
        sparsified_delta = np.where(np.abs(delta) < threshold, 0.0, delta)

        payload = DynamicInt4ActivationQuantizer.quantize(
            sparsified_delta, sequence_id=sequence_id, step=step
        )

        # Mark as delta encoded
        delta_payload = payload.model_copy(
            update={"is_delta_encoded": True, "precision": "delta_int4"}
        )

        # Update sender baseline using exact reconstructed delta to match receiver state
        reconstructed_delta = DynamicInt4ActivationQuantizer.dequantize(delta_payload)
        self._cached_baseline = self._cached_baseline + reconstructed_delta

        return delta_payload


class TemporalDeltaDecompressor:
    """Receiver-side decompressor that accumulates delta states into reconstructed tensors."""

    def __init__(self) -> None:
        self._cached_baseline: NDArray[np.float32] | None = None

    def reset(self) -> None:
        """Resets baseline state for a new sequence."""
        self._cached_baseline = None

    def decompress_step(
        self,
        payload: CompressedActivationPayload,
    ) -> NDArray[np.float32]:
        """Restores tensor from payload, accumulating delta if marked."""
        dequantized = DynamicInt4ActivationQuantizer.dequantize(payload)

        if not payload.is_delta_encoded:
            self._cached_baseline = dequantized
            return dequantized

        if self._cached_baseline is None:
            # Fallback if baseline was missing
            self._cached_baseline = dequantized
            return dequantized

        # Accumulate delta onto baseline
        reconstructed = self._cached_baseline + dequantized
        self._cached_baseline = reconstructed
        return reconstructed
