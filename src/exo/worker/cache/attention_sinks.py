import numpy as np
from numpy.typing import NDArray

DEFAULT_SINK_TOKEN_COUNT = 4


class AttentionSinkManager:
    """Manages attention sinks to prevent Softmax probability collapse in long-sequence streaming."""

    def __init__(self, num_sink_tokens: int = DEFAULT_SINK_TOKEN_COUNT) -> None:
        self._num_sink_tokens = max(1, num_sink_tokens)

    @property
    def num_sink_tokens(self) -> int:
        return self._num_sink_tokens

    def is_sink_index(self, index: int) -> bool:
        """Determines if a given token index belongs to the protected attention sink range."""
        return 0 <= index < self._num_sink_tokens

    def generate_aligned_position_ids(
        self,
        active_indices: NDArray[np.int32],
        current_seq_length: int,
    ) -> NDArray[np.int32]:
        """Aligns position IDs for active tokens in the compressed KV cache.

        Attention sinks retain their original early position IDs (0, 1, ..., S-1).
        Subsequent retained tokens (heavy hitters and recent window) map cleanly to avoid RoPE distortion.
        """
        if len(active_indices) == 0:
            return np.empty(0, dtype=np.int32)

        aligned = active_indices.copy()
        # Ensure sink tokens retain exact 0..S-1 positions
        for i in range(min(self._num_sink_tokens, len(aligned))):
            aligned[i] = i

        return aligned
