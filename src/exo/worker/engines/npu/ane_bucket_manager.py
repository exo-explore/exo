from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from exo.shared.types.ane_compute import ANEShapeBucket

DEFAULT_ANE_CHUNK_SIZES = (64, 128, 256, 512, 1024, 2048)


class ANEBucketManager:
    """Manages static shape bucketing for Apple Neural Engine to eliminate runtime graph recompilation."""

    def __init__(
        self,
        chunk_sizes: Sequence[int] = DEFAULT_ANE_CHUNK_SIZES,
        hidden_dim: int = 4096,
    ) -> None:
        self._hidden_dim = hidden_dim
        self._buckets = sorted(
            [
                ANEShapeBucket(
                    bucket_id=f"ane-bucket-c{cs}-h{hidden_dim}",
                    batch_size=1,
                    chunk_size=cs,
                    hidden_dim=hidden_dim,
                )
                for cs in chunk_sizes
            ],
            key=lambda b: b.chunk_size,
        )

    @property
    def available_buckets(self) -> Sequence[ANEShapeBucket]:
        return tuple(self._buckets)

    def select_optimal_bucket(self, token_count: int) -> ANEShapeBucket:
        """Finds the tightest static shape bucket that fits the requested token count."""
        if token_count <= 0:
            return self._buckets[0]

        for b in self._buckets:
            if b.chunk_size >= token_count:
                return b

        # If token count exceeds maximum single bucket, return largest bucket
        return self._buckets[-1]

    def pad_activations(
        self,
        activations: NDArray[np.floating],
        bucket: ANEShapeBucket,
    ) -> tuple[NDArray[np.float32], int]:
        """Pads activation matrix to align exactly with static bucket geometry."""
        arr = np.asarray(activations, dtype=np.float32)
        seq_len = arr.shape[0] if arr.ndim >= 1 else 1

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
            seq_len = 1

        target_len = bucket.chunk_size
        if seq_len < target_len:
            pad_rows = target_len - seq_len
            padded = np.pad(arr, ((0, pad_rows), (0, 0)), mode="constant", constant_values=0.0)
            return padded, seq_len

        return arr[:target_len, :], seq_len

    def unpad_activations(
        self,
        output_tensor: NDArray[np.floating],
        original_seq_len: int,
    ) -> NDArray[np.float32]:
        """Extracts valid sequence tokens from the static bucket output, trimming padding."""
        arr = np.asarray(output_tensor, dtype=np.float32)
        return arr[:original_seq_len]
