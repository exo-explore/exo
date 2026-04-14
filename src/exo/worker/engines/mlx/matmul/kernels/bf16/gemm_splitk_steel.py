#!/usr/bin/env python3
"""Split-K bf16 GEMM using actual Steel GEMM infrastructure.

Bundles MLX's Steel headers and passes them via mx.fast.metal_kernel's
header parameter. The kernel source is just the template instantiation
of gemm_splitk — identical to stock MLX.

Only difference from stock: no K >= max(M, N) restriction.

Usage:
    from gemm_splitk_steel import custom_bf16_gemm_splitk_steel
    y = custom_bf16_gemm_splitk_steel(x, w, M=16, N=17408, K=5120, BM=16)
"""

import os
import re
import mlx.core as mx


def ceil_div(a, b):
    return (a + b - 1) // b


def next_power_of_2(n):
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def compute_partitions(M, N, K):
    _tm = ceil_div(M, 32)
    _tn = ceil_div(N, 32)
    _tk = K // 16
    tmtn = max(_tm * _tn, 1)
    return min(max(2, next_power_of_2(_tk // tmtn)), 32)


def _load_steel_headers():
    """Read and concatenate Steel headers in dependency order."""
    base = os.path.join(os.path.dirname(__file__), '..', 'steel_include')

    files_in_order = [
        'steel/defines.h',
        'steel/utils/integral_constant.h',
        'steel/utils/type_traits.h',
        'steel/utils.h',
        'steel/gemm/transforms.h',
        'steel/gemm/params.h',
        'steel/gemm/loader.h',
        'steel/gemm/mma.h',
        'steel/gemm/gemm.h',
        'steel/gemm/kernels/steel_gemm_splitk.h',
    ]

    parts = []
    for f in files_in_order:
        path = os.path.join(base, f)
        with open(path) as fh:
            content = fh.read()
        # Strip #pragma once and #include directives (deps are manually ordered)
        content = re.sub(r'#pragma once', '', content)
        content = re.sub(r'#include\s+"[^"]*"', '', content)
        content = re.sub(r'#include\s+<metal_simdgroup_matrix>', '', content)
        content = re.sub(r'#include\s+<metal_simdgroup>', '', content)
        content = re.sub(r'#include\s+<metal_stdlib>', '', content)
        parts.append(f'// ── {f} ──\n' + content)

    return '\n'.join(parts)


_steel_header = None


def _get_steel_header():
    global _steel_header
    if _steel_header is None:
        _steel_header = _load_steel_headers()
    return _steel_header


def _gen_splitk_instantiation(M_val, N_val, K_val, BM, split_k_partitions, BN=32, BK=16):
    """Generate the kernel source — just template instantiation + dispatch logic.
    The actual GEMM code comes from the Steel headers in the header parameter."""

    WM = 1 if BM <= 8 else 2
    WN = 2

    gemm_k_iterations = (K_val // BK) // split_k_partitions
    split_k_partition_size = gemm_k_iterations * BK

    mn_aligned_str = "true" if M_val % BM == 0 and N_val % BN == 0 else "false"
    k_aligned_str = "true" if K_val % BK == 0 else "false"

    # The kernel body is a direct copy of steel_gemm_splitk.h's gemm_splitk function
    # with template parameters expanded
    return f"""
    using namespace mlx::steel;

    using T = bfloat16_t;
    using U = float;

    const int BM_val = {BM};
    const int BN_val = {BN};
    const int BK_val = {BK};

    using gemm_kernel = GEMMKernel<T, U, {BM}, {BN}, {BK}, {WM}, {WN}, false, true, {mn_aligned_str}, {k_aligned_str}>;
    using loader_a_t = typename gemm_kernel::loader_a_t;
    using loader_b_t = typename gemm_kernel::loader_b_t;
    using mma_t = typename gemm_kernel::mma_t;

    threadgroup T As[gemm_kernel::tgp_mem_size_a];
    threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

    uint simd_lane_id = thread_index_in_simdgroup;
    uint simd_group_id = simdgroup_index_in_threadgroup;
    uint3 tid = threadgroup_position_in_grid;

    const int tiles_n = {ceil_div(N_val, BN)};
    const int tiles_m = {ceil_div(M_val, BM)};
    const int tid_x = tid.x;
    const int tid_y = tid.y;
    const int tid_z = tid.z;

    if (tiles_n <= tid_x || tiles_m <= tid_y) return;

    const int M = {M_val};
    const int N = {N_val};
    const int K = {K_val};
    const int lda = K;  // A is (M, K) row-major
    const int ldb = K;  // B is (N, K) row-major, transposed

    const int split_k_partitions = {split_k_partitions};
    const int split_k_partition_size = {split_k_partition_size};
    const int split_k_partition_stride = M * N;
    const int gemm_k_iterations_aligned = {gemm_k_iterations};

    const int c_row = tid_y * {BM};
    const int c_col = tid_x * {BN};
    const int k_start = split_k_partition_size * tid_z;

    // Pointer setup (matching steel_gemm_splitk.h)
    // transpose_a=false: A += k_start + c_row * lda
    // transpose_b=true:  B += k_start + c_col * ldb
    const device T* A = (const device T*)x + (long)k_start + (long)c_row * lda;
    const device T* B = (const device T*)w + (long)k_start + (long)c_col * ldb;
    device U* C = (device U*)y + (long)split_k_partition_stride * tid_z + (long)c_row * N + c_col;

    // Prepare loaders and MMA
    thread loader_a_t loader_a(A, lda, As, simd_group_id, simd_lane_id);
    thread loader_b_t loader_b(B, ldb, Bs, simd_group_id, simd_lane_id);
    thread mma_t mma_op(simd_group_id, simd_lane_id);

    int gemm_k_iters = gemm_k_iterations_aligned;

    short tgp_bm = min((short){BM}, (short)(M - c_row));
    short tgp_bn = min((short){BN}, (short)(N - c_col));
    short leftover_bk = K % {BK};

    // Main GEMM loop
    if (tgp_bm == {BM} && tgp_bn == {BN}) {{
        gemm_kernel::gemm_loop(
            As, Bs, gemm_k_iters, loader_a, loader_b, mma_op,
            tgp_bm, tgp_bn, leftover_bk,
            LoopAlignment<true, true, true>{{}});
    }} else if (tgp_bn == {BN}) {{
        gemm_kernel::gemm_loop(
            As, Bs, gemm_k_iters, loader_a, loader_b, mma_op,
            tgp_bm, tgp_bn, leftover_bk,
            LoopAlignment<false, true, true>{{}});
    }} else if (tgp_bm == {BM}) {{
        gemm_kernel::gemm_loop(
            As, Bs, gemm_k_iters, loader_a, loader_b, mma_op,
            tgp_bm, tgp_bn, leftover_bk,
            LoopAlignment<true, false, true>{{}});
    }} else {{
        gemm_kernel::gemm_loop(
            As, Bs, gemm_k_iters, loader_a, loader_b, mma_op,
            tgp_bm, tgp_bn, leftover_bk,
            LoopAlignment<false, false, true>{{}});
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Last partition handles leftover K
    if ((tid_z + 1) == split_k_partitions) {{
        int gemm_k_iter_remaining = (K - (k_start + split_k_partition_size)) / {BK};
        if (gemm_k_iter_remaining > 0)
            gemm_kernel::gemm_loop(
                As, Bs, gemm_k_iter_remaining, loader_a, loader_b, mma_op,
                tgp_bm, tgp_bn, leftover_bk,
                LoopAlignment<false, false, {k_aligned_str}>{{}});
    }}

    // Store results
    if (tgp_bm == {BM} && tgp_bn == {BN}) {{
        mma_op.store_result(C, N);
    }} else {{
        mma_op.store_result_safe(C, N, short2(tgp_bn, tgp_bm));
    }}
"""


def _gen_accum_source(split_k_partitions):
    return f"""
    uint gid_x = thread_position_in_grid.x;
    uint gid_y = thread_position_in_grid.y;
    uint N_val = threads_per_grid.x;

    int offset = gid_y * N_val + gid_x;
    int stride = threads_per_grid.x * threads_per_grid.y;

    float out = 0.0f;
    for (int p = 0; p < {split_k_partitions}; p++) {{
        out += ((const device float*)c_split)[offset + p * stride];
    }}

    y[gid_y * N_val + gid_x] = static_cast<bfloat16_t>(out);
"""


_gemm_cache = {}
_accum_cache = {}


def custom_bf16_gemm_splitk_steel(x, w, M, N, K, BM=16, BN=32, BK=16):
    WM = 1 if BM <= 8 else 2
    WN = 2
    P = compute_partitions(M, N, K)

    gemm_key = (M, N, K, BM, BN, BK, P)
    if gemm_key not in _gemm_cache:
        _gemm_cache[gemm_key] = mx.fast.metal_kernel(
            name=f"splitk_steel_bm{BM}_bn{BN}_bk{BK}_M{M}_N{N}_K{K}_P{P}",
            input_names=["x", "w"],
            output_names=["y"],
            header=_get_steel_header(),
            source=_gen_splitk_instantiation(M, N, K, BM, P, BN=BN, BK=BK),
        )
    gemm_kern = _gemm_cache[gemm_key]

    sgs = WM * WN
    n_tg_n = ceil_div(N, BN)
    n_tg_m = ceil_div(M, BM)

    c_split = gemm_kern(
        inputs=[x, w],
        output_shapes=[(P * M * N,)],
        output_dtypes=[mx.float32],
        grid=(32 * n_tg_n, sgs * n_tg_m, P),
        threadgroup=(32, sgs, 1),
    )[0]

    accum_key = P
    if accum_key not in _accum_cache:
        _accum_cache[accum_key] = mx.fast.metal_kernel(
            name=f"splitk_accum_P{P}",
            input_names=["c_split"],
            output_names=["y"],
            source=_gen_accum_source(P),
        )
    accum_kern = _accum_cache[accum_key]

    y = accum_kern(
        inputs=[c_split],
        output_shapes=[(M * N,)],
        output_dtypes=[mx.bfloat16],
        grid=(N, M, 1),
        threadgroup=(min(N, 256), 1, 1),
    )[0]

    return y.reshape(M, N)
