#!/usr/bin/env python3
"""Kernel picker — single source of truth for selecting optimal matmul
kernel per (N, K, M, dtype).

Used by both the drafter patch (speculative/bf16_lpb_patch.py) and the
target patch (patches/qwen3_5/lpb_patch.py). Selections are derived from
mlx_bench matmul/RESULTS.md winners rounded up to the nearest benchmark
M column (M ∈ {1, 2, 4, 8, 12, 16, 32, 64}).

Returns a uniform-signature callable:
- bf16: fn(x, w, M, N, K)
- int8: fn(x, w, scales, biases, M, N, K, group_size)
"""

from ..kernels.bf16.gemv_loop_over_b import custom_bf16_qmv_loop_over_b as _bf16_lpb
from ..kernels.bf16.gemv_loop_over_b_twice import custom_bf16_qmv_loop_over_b_twice as _bf16_lpb_twice
from ..kernels.bf16.gemm_bm8 import custom_bf16_gemm as _bf16_bm8  # noqa: F401
from ..kernels.bf16.gemm_splitk_steel import custom_bf16_gemm_splitk_steel as _bf16_sk_steel

from ..kernels.quantized.qmv_loop_over_b import custom_qmv_loop_over_b as _int8_lpb
from ..kernels.quantized.qmm_bm8 import custom_qmm_t_bm8 as _int8_bm8  # noqa: F401
from ..kernels.quantized.qmm_bm16 import custom_qmm_t_bm16 as _int8_bm16  # noqa: F401
from ..kernels.quantized.qmm_splitk import custom_qmm_splitk as _int8_qsk


def _round_up_to_bench_col(M):
    for col in (1, 2, 4, 8, 12, 16, 32, 64):
        if M <= col:
            return col
    return 64


def pick_bf16_kernel(N, K, M):
    """Return (name, fn) for the best bf16 kernel at this (N, K, M).

    fn signature: fn(x, w, M, N, K) → y (M, N)

    KNOWN BUG: sk_steel8/16 produce silently incorrect output at very large
    N (verified broken at N=248320, K=5120 with 100% rel error). For
    lm_head-sized projections (N > 50000) we force lpb_twice, which is
    also the fastest CORRECT kernel at those dims.
    """
    M_rnd = _round_up_to_bench_col(M)

    # Safety guard for large-N projections (e.g. lm_head).
    if N > 50000:
        if M_rnd <= 8:
            return "lpb", _bf16_lpb
        return "lpb_twice", _bf16_lpb_twice

    if M_rnd <= 8:
        return "lpb", _bf16_lpb

    if M_rnd == 12:
        if max(N, K) <= 4096:
            return "lpb_twice", _bf16_lpb_twice
        fn = lambda x, w, M_, N_, K_: _bf16_sk_steel(x, w, M_, N_, K_, BM=16)
        return "sk_steel16", fn

    if M_rnd == 16:
        fn = lambda x, w, M_, N_, K_: _bf16_sk_steel(x, w, M_, N_, K_, BM=16)
        return "sk_steel16", fn

    # M = 32, 64
    fn = lambda x, w, M_, N_, K_: _bf16_sk_steel(x, w, M_, N_, K_, BM=32)
    return "sk_steel32", fn


def pick_int8_kernel(N, K, M):
    """Return (name, fn) for the best int8 kernel at this (N, K, M).

    fn signature: fn(x, w, scales, biases, M, N, K, group_size) → y (M, N)
    """
    M_rnd = _round_up_to_bench_col(M)

    if M_rnd <= 8:
        return "lpb", _int8_lpb

    if M_rnd == 12:
        fn = lambda x, w, s, b, M_, N_, K_, gs: _int8_qsk(x, w, s, b, M_, N_, K_, BM=16, group_size=gs)
        return "qsk16", fn

    if M_rnd == 16:
        fn = lambda x, w, s, b, M_, N_, K_, gs: _int8_qsk(x, w, s, b, M_, N_, K_, BM=16, group_size=gs)
        return "qsk16", fn

    # M = 32, 64
    fn = lambda x, w, s, b, M_, N_, K_, gs: _int8_qsk(x, w, s, b, M_, N_, K_, BM=32, group_size=gs)
    return "qsk32", fn
