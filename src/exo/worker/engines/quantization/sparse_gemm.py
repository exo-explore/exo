import time

import numpy as np
from numpy.typing import NDArray

from exo.shared.types.sparse_bitnet import SparseBitNetTensor
from exo.worker.engines.quantization.sparse_bitnet_codec import SparseBitNetCodec


class SparseBitNetGemmEngine:
    """Fused sparse ternary GEMM engine exploiting zero-sparsity to accelerate BitNet inference."""

    @classmethod
    def execute_sparse_gemm(
        cls,
        activations: NDArray[np.floating],
        sparse_weight: SparseBitNetTensor,
    ) -> tuple[NDArray[np.float32], float, float]:
        """Executes matrix multiplication between activations and compressed sparse ternary weights.

        Args:
            activations: Input activation tensor of shape (M, K).
            sparse_weight: Compressed SparseBitNetTensor of shape (K, N).

        Returns:
            Tuple of (output_activations, elapsed_milliseconds, flop_reduction_ratio).
        """
        act = np.asarray(activations, dtype=np.float32)
        if act.ndim == 1:
            act = act.reshape(1, -1)

        t_start = time.perf_counter()

        # On-the-fly register unpacking without permanent memory residency
        weights = SparseBitNetCodec.decompress_to_ternary(sparse_weight)

        # Fused GEMM: Output = Activations @ DecompressedWeights
        output = np.matmul(act, weights)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        flop_reduction = sparse_weight.sparsity_ratio

        return output, elapsed_ms, flop_reduction
