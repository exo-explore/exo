import logging
import os
import platform
import time

logger = logging.getLogger(__name__)

from exo.shared.types.cpu_jit import (
    CacheTilingPlan,
    CPUCacheGeometry,
    CPUCapabilityInfo,
    JITExecutionReport,
)
from exo.worker.engines.cpu_jit.cpu_prober import (
    CacheTopologyProber,
    CPUIDInstructionProber,
)
from exo.worker.engines.cpu_jit.vector_jit import VectorizedJITDispatcher


class CacheTilingPlanner:
    """Calculates optimal GEMM matrix tile dimensions (M_tile, N_tile, K_tile)

    tailored to fit within L1 and L2 cache sizes to avoid cache thrashing on legacy CPUs.
    """

    @classmethod
    def plan(
        cls,
        m: int,
        k: int,
        n: int,
        cache_geom: CPUCacheGeometry | None = None,
        vector_width: int = 4,
    ) -> CacheTilingPlan:
        """Determines optimal tile dimensions satisfying L1D and L2 cache capacity limits."""
        geom = cache_geom or CacheTopologyProber.get_cache_geometry()
        l1_budget = int(geom.l1d_cache_bytes * 0.60)  # Reserve 60% of L1 for B_tile
        l2_budget = int(geom.l2_cache_bytes * 0.75)  # Reserve 75% of L2 for A_tile + C_tile

        # Candidate tile sizes aligned with vector width
        # B_tile is (tile_k * tile_n * 4 bytes) -> must fit in l1_budget
        # A_tile is (tile_m * tile_k * 4 bytes) -> must fit in l2_budget
        step = max(vector_width, 4)

        # Baseline search
        target_k = min(k, 64)
        target_n = min(n, 64)
        target_m = min(m, 32)

        # Refine target_k and target_n so (target_k * target_n * 4) <= l1_budget
        while (target_k * target_n * 4) > l1_budget and target_k > step:
            target_k = max(step, target_k // 2)

        while (target_k * target_n * 4) > l1_budget and target_n > step:
            target_n = max(step, target_n // 2)

        # Refine target_m so (target_m * target_k * 4) <= l2_budget
        while (target_m * target_k * 4) > l2_budget and target_m > 1:
            target_m = max(1, target_m // 2)

        tile_m = min(m, max(1, target_m))
        tile_k = min(k, max(1, target_k))
        tile_n = min(n, max(1, target_n))

        l1_working = tile_k * tile_n * 4
        l2_working = (tile_m * tile_k + tile_m * tile_n) * 4

        # Hit rate estimation based on working set ratios
        hit_ratio_l1 = min(0.95, max(0.60, 1.0 - (l1_working / (geom.l1d_cache_bytes + 1e-5))))
        hit_ratio_l2 = min(0.98, max(0.70, 1.0 - (l2_working / (geom.l2_cache_bytes + 1e-5))))
        estimated_hit = (hit_ratio_l1 * 0.7) + (hit_ratio_l2 * 0.3)

        return CacheTilingPlan(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            l1_working_set_bytes=l1_working,
            l2_working_set_bytes=l2_working,
            estimated_cache_hit_rate=round(estimated_hit, 4),
            vector_width_elements=vector_width,
        )


class CoreAffinityManager:
    """Safely manages CPU core pinning and affinity without throwing unhandled exceptions."""

    @classmethod
    def pin_to_cores(cls, core_ids: list[int]) -> bool:
        """Pins current process to specified logical CPU cores."""
        if not core_ids:
            return False

        system = platform.system().lower()
        try:
            if system == "linux" and hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, set(core_ids))
                return True
            elif system == "windows":
                # Using win32 process mask if ctypes is available
                import ctypes

                mask = 0
                for cid in core_ids:
                    mask |= 1 << cid
                handle = ctypes.windll.kernel32.GetCurrentProcess()
                success = ctypes.windll.kernel32.SetProcessAffinityMask(
                    handle, ctypes.c_size_t(mask)
                )
                return bool(success)
        except (OSError, ValueError, AttributeError) as exc:
            logger.debug("Failed to set process core affinity: %s", exc)
        return False


class CacheTiledGEMMEngine:
    """Executes cache-aware partitioned GEMM operations (C = A * B) on edge CPUs."""

    def __init__(
        self,
        capability: CPUCapabilityInfo | None = None,
        cache_geom: CPUCacheGeometry | None = None,
    ):
        self.capability = capability or CPUIDInstructionProber.probe()
        self.cache_geom = cache_geom or CacheTopologyProber.get_cache_geometry()
        self.dispatcher = VectorizedJITDispatcher(
            active_level=self.capability.active_level
        )

    def execute_gemm(
        self,
        a_matrix: list[float],
        b_matrix: list[float],
        m: int,
        k: int,
        n: int,
    ) -> tuple[list[float], JITExecutionReport]:
        """Performs full cache-partitioned matrix multiplication: C = A * B.

        Dimensions:
            a_matrix: (m x k) flattened row-major list
            b_matrix: (k x n) flattened row-major list
        """
        assert len(a_matrix) == m * k, f"A matrix size {len(a_matrix)} != {m * k}"
        assert len(b_matrix) == k * n, f"B matrix size {len(b_matrix)} != {k * n}"

        vector_width = 8 if self.capability.active_level in ("AVX512", "AVX2") else 4
        tiling = CacheTilingPlanner.plan(
            m, k, n, cache_geom=self.cache_geom, vector_width=vector_width
        )
        kernel = self.dispatcher.get_micro_kernel()

        start_time = time.perf_counter()
        c_matrix = [0.0] * (m * n)

        tile_m = tiling.tile_m
        tile_k = tiling.tile_k
        tile_n = tiling.tile_n

        # 3-level Cache Tiled Outer Loops
        for i_base in range(0, m, tile_m):
            cur_m = min(tile_m, m - i_base)
            for j_base in range(0, n, tile_n):
                cur_n = min(tile_n, n - j_base)
                for p_base in range(0, k, tile_k):
                    cur_k = min(tile_k, k - p_base)

                    # Extract A sub-tile of shape (cur_m x cur_k)
                    a_sub = [0.0] * (cur_m * cur_k)
                    for r in range(cur_m):
                        src_start = (i_base + r) * k + p_base
                        dst_start = r * cur_k
                        a_sub[dst_start : dst_start + cur_k] = a_matrix[
                            src_start : src_start + cur_k
                        ]

                    # Extract B sub-tile of shape (cur_k x cur_n)
                    b_sub = [0.0] * (cur_k * cur_n)
                    for r in range(cur_k):
                        src_start = (p_base + r) * n + j_base
                        dst_start = r * cur_n
                        b_sub[dst_start : dst_start + cur_n] = b_matrix[
                            src_start : src_start + cur_n
                        ]

                    # Execute micro-kernel
                    c_sub, _ = kernel(a_sub, b_sub, cur_m, cur_k, cur_n)

                    # Accumulate into C output matrix
                    for r in range(cur_m):
                        dst_row = (i_base + r) * n + j_base
                        src_row = r * cur_n
                        for c in range(cur_n):
                            c_matrix[dst_row + c] += c_sub[src_row + c]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        flops = 2.0 * m * k * n
        gflops = (flops / (elapsed_ms * 1e6)) if elapsed_ms > 0 else 0.0

        report = JITExecutionReport(
            kernel_name=f"TiledGEMM_{self.capability.active_level}",
            instruction_level=self.capability.active_level,
            tiling_plan=tiling,
            m=m,
            k=k,
            n=n,
            elapsed_ms=round(elapsed_ms, 3),
            gflops=round(gflops, 3),
            is_fallback=(self.capability.active_level == "SCALAR"),
        )

        return c_matrix, report
