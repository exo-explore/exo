import math

import numpy as np
from numpy.typing import NDArray


class SubgroupReductionOptimizer:
    """Optimizes GEMM tiling and register-level reduction for Vulkan subgroups.

    Bypasses high-latency shared memory by utilizing hardware SIMD warp shuffle instructions.
    """

    @staticmethod
    def determine_optimal_tile_size(
        subgroup_size: int,
        k_dim: int,
        n_dim: int,
    ) -> tuple[int, int]:
        """Determines the optimal 2D workgroup tile size (M, N) matching EU subgroup geometry."""
        # Align tile dimension M with the hardware subgroup width (typically 16 on Intel, 32 on AMD)
        tile_m = max(8, min(subgroup_size, 32))
        # Tile N chosen to maximize register reuse without spilling
        tile_n = 16 if tile_m >= 16 else 32
        return (tile_m, tile_n)

    @staticmethod
    def simulate_subgroup_shuffle_reduction(
        vector: NDArray[np.float32],
        subgroup_size: int = 16,
    ) -> float:
        """Simulates parallel tree reduction using subgroupShuffleXor instructions.

        Runs in log2(subgroup_size) steps completely within execution registers.
        """
        arr = np.asarray(vector, dtype=np.float32)
        total_elements = len(arr)
        if total_elements == 0:
            return 0.0

        # Pad to multiple of subgroup_size
        pad_len = (subgroup_size - (total_elements % subgroup_size)) % subgroup_size
        if pad_len > 0:
            arr = np.pad(arr, (0, pad_len), mode="constant", constant_values=0.0)

        # Reshape into chunks of subgroup_size and perform simulated butterfly shuffle
        chunks = arr.reshape(-1, subgroup_size)
        step_sums = np.sum(chunks, axis=0)

        # Butterfly reduction simulation: log2(subgroup_size) steps
        steps = int(math.log2(subgroup_size)) if subgroup_size > 0 else 0
        working = step_sums.copy()
        for s in range(steps):
            offset = 1 << s
            # In hardware: working += subgroupShuffleXor(working, offset)
            rolled = np.roll(working, -offset)
            working += rolled

        # Result is accumulated into lane 0
        return float(working[0])
