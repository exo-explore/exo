import time
from collections.abc import Callable

from exo.shared.types.cpu_jit import (
    InstructionSetLevel,
)


class VectorizedJITDispatcher:
    """Dispatches vectorized micro-kernels tailored to detected CPU capabilities.

    Prevents Illegal Instruction crashes on legacy CPUs (e.g. Core 2 Duo, 1st-4th gen Core)
    by dispatching SSE4 unrolled or pure scalar kernels instead of AVX2/AVX-512.
    """

    def __init__(self, active_level: InstructionSetLevel = "SCALAR"):
        self.active_level = active_level

    def get_micro_kernel(
        self,
    ) -> Callable[
        [list[float], list[float], int, int, int],
        tuple[list[float], float],
    ]:
        """Returns the specialized execution kernel for the active instruction set level."""
        if self.active_level in ("AVX512", "AVX2"):
            return self._kernel_avx2_unrolled
        elif self.active_level in ("AVX", "SSE4_2", "SSE4_1"):
            return self._kernel_sse4_unrolled
        elif self.active_level == "ARM_NEON":
            return self._kernel_arm_neon_unrolled
        else:
            return self._kernel_scalar_safe

    @classmethod
    def _kernel_scalar_safe(
        cls,
        a_tile: list[float],
        b_tile: list[float],
        m: int,
        k: int,
        n: int,
    ) -> tuple[list[float], float]:
        """Pure scalar GEMM tile: C = A * B.

        Safe on 100% of CPU architectures without specialized vector flags.
        """
        start = time.perf_counter()
        c_tile = [0.0] * (m * n)

        for i in range(m):
            i_k = i * k
            i_n = i * n
            for p in range(k):
                a_val = a_tile[i_k + p]
                if a_val == 0.0:
                    continue
                p_n = p * n
                for j in range(n):
                    c_tile[i_n + j] += a_val * b_tile[p_n + j]

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return c_tile, elapsed_ms

    @classmethod
    def _kernel_sse4_unrolled(
        cls,
        a_tile: list[float],
        b_tile: list[float],
        m: int,
        k: int,
        n: int,
    ) -> tuple[list[float], float]:
        """128-bit vector unrolled tile (4-wide vector step emulation).

        Ideal for legacy Intel/AMD CPUs lacking AVX2, preventing Illegal Instruction crashes.
        """
        start = time.perf_counter()
        c_tile = [0.0] * (m * n)

        for i in range(m):
            i_k = i * k
            i_n = i * n
            for p in range(k):
                a_val = a_tile[i_k + p]
                if a_val == 0.0:
                    continue
                p_n = p * n

                # Unroll inner loop by 4 elements (128-bit SSE vector width)
                j = 0
                while j + 3 < n:
                    c_tile[i_n + j] += a_val * b_tile[p_n + j]
                    c_tile[i_n + j + 1] += a_val * b_tile[p_n + j + 1]
                    c_tile[i_n + j + 2] += a_val * b_tile[p_n + j + 2]
                    c_tile[i_n + j + 3] += a_val * b_tile[p_n + j + 3]
                    j += 4

                # Remainder loop
                while j < n:
                    c_tile[i_n + j] += a_val * b_tile[p_n + j]
                    j += 1

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return c_tile, elapsed_ms

    @classmethod
    def _kernel_avx2_unrolled(
        cls,
        a_tile: list[float],
        b_tile: list[float],
        m: int,
        k: int,
        n: int,
    ) -> tuple[list[float], float]:
        """256-bit vector unrolled tile (8-wide vector step emulation)."""
        start = time.perf_counter()
        c_tile = [0.0] * (m * n)

        for i in range(m):
            i_k = i * k
            i_n = i * n
            for p in range(k):
                a_val = a_tile[i_k + p]
                if a_val == 0.0:
                    continue
                p_n = p * n

                # Unroll inner loop by 8 elements (256-bit AVX2 vector width)
                j = 0
                while j + 7 < n:
                    c_tile[i_n + j] += a_val * b_tile[p_n + j]
                    c_tile[i_n + j + 1] += a_val * b_tile[p_n + j + 1]
                    c_tile[i_n + j + 2] += a_val * b_tile[p_n + j + 2]
                    c_tile[i_n + j + 3] += a_val * b_tile[p_n + j + 3]
                    c_tile[i_n + j + 4] += a_val * b_tile[p_n + j + 4]
                    c_tile[i_n + j + 5] += a_val * b_tile[p_n + j + 5]
                    c_tile[i_n + j + 6] += a_val * b_tile[p_n + j + 6]
                    c_tile[i_n + j + 7] += a_val * b_tile[p_n + j + 7]
                    j += 8

                # Remainder loop
                while j < n:
                    c_tile[i_n + j] += a_val * b_tile[p_n + j]
                    j += 1

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return c_tile, elapsed_ms

    @classmethod
    def _kernel_arm_neon_unrolled(
        cls,
        a_tile: list[float],
        b_tile: list[float],
        m: int,
        k: int,
        n: int,
    ) -> tuple[list[float], float]:
        """ARM NEON 128-bit vector tile emulation."""
        return cls._kernel_sse4_unrolled(a_tile, b_tile, m, k, n)

    @classmethod
    def unpack_ternary_weights(
        cls,
        packed_bits: bytes,
        num_weights: int,
        scale: float = 1.0,
    ) -> list[float]:
        """Decompresses packed 1.58-bit ternary {-1, 0, +1} weights on CPU.

        Each byte stores 4 ternary values (2 bits each: 00 -> 0, 01 -> +1, 10 -> -1).
        """
        unpacked: list[float] = [0.0] * num_weights
        idx = 0
        lut = [0.0, scale, -scale, 0.0]

        for byte_val in packed_bits:
            for shift in (0, 2, 4, 6):
                if idx >= num_weights:
                    break
                code = (byte_val >> shift) & 0x03
                unpacked[idx] = lut[code]
                idx += 1
            if idx >= num_weights:
                break

        return unpacked
