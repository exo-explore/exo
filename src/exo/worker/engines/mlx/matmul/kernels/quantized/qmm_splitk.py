#!/usr/bin/env python3
"""Split-K int8 quantized GEMM (gs=64, bits=8).

Int8 analogue of matmul/kernels/bf16/gemm_splitk.py: partitions K across
P threadgroups per output tile, each writing a partial fp32 result to a
scratch buffer; a second dispatch sums the partials and casts to bf16.

Key modification from the bf16 version: the W loader dequantizes 8-bit
uint8 weights on the fly using per-group bf16 scales/biases, following
the same pattern as matmul/kernels/quantized/qmm_bm16.py.

BK is chosen equal to GROUP_SIZE (64) so each K-iteration advances the
scales/biases pointers by exactly one group, eliminating the need for an
inner group_step counter and keeping partition starts aligned to group
boundaries.

Usage:
    from qmm_splitk import custom_qmm_splitk
    y = custom_qmm_splitk(x, w, scales, biases, M=16, N=8192, K=5120, BM=16)
"""

import mlx.core as mx


def ceil_div(a, b):
    return (a + b - 1) // b


def next_power_of_2(n):
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def compute_partitions(M, N, K):
    """Same formula as gemm_splitk_steel.compute_partitions (bf16)."""
    _tm = ceil_div(M, 32)
    _tn = ceil_div(N, 32)
    _tk = K // 16
    tmtn = max(_tm * _tn, 1)
    return min(max(2, next_power_of_2(_tk // tmtn)), 32)


def _gen_splitk_qmm_source(M_val, N_val, K_val, BM, split_k_partitions, group_size=64):
    """Int8 split-K GEMM source. BK is fixed to group_size so scales/biases
    advance by exactly one group per K-iteration (no group_step counter).
    """
    gs = group_size
    BN = 32
    BK = gs  # = 64
    if BM <= 8:
        WM, WN = 1, 2
    else:
        WM, WN = 2, 2
    tgp_size = WM * WN * 32  # 64 or 128
    BK_PAD = BK + 8  # bank-conflict padding (matches Steel's tgp_padding for bf16)

    gemm_k_iterations = (K_val // BK) // split_k_partitions
    split_k_partition_size = gemm_k_iterations * BK

    # Loader params
    # X (bf16, BROWS=BM, BCOLS=BK): each thread reads x_n_reads values
    x_total = BM * BK
    x_n_reads = max(1, x_total // tgp_size)
    x_TCOLS = BK // x_n_reads if x_n_reads > 0 else BK

    # W (uint8, BROWS=BN, BCOLS=BK): each thread reads w_n_reads bytes
    w_n_reads = (BN * BK) // tgp_size

    TM = BM // (8 * WM)
    TN = BN // (8 * WN)

    # Unrolled accumulator declarations: TM*TN simdgroup_matrix<float,8,8> per SG
    c_decl_lines = []
    for ti in range(TM):
        for tj in range(TN):
            c_decl_lines.append(
                f"    simdgroup_matrix<float, 8, 8> C{ti}{tj} = simdgroup_matrix<float, 8, 8>(0);"
            )
    c_decl = "\n".join(c_decl_lines)

    # Inner MMA step (inside the BK/8 loop over kk)
    mma_lines = []
    for ti in range(TM):
        mma_lines.append(
            f"            simdgroup_matrix<float, 8, 8> A{ti};\n"
            f"            simdgroup_load(A{ti}, &Xs[(tm + {ti} * {WM} * 8) * BK_PAD + kk], BK_PAD);"
        )
    for tj in range(TN):
        mma_lines.append(
            f"            simdgroup_matrix<float, 8, 8> B{tj};\n"
            f"            simdgroup_load(B{tj}, &Ws[(tn + {tj} * {WN} * 8) * BK_PAD + kk], BK_PAD, ulong2(0, 0), true);"
        )
    for ti in range(TM):
        for tj in range(TN):
            mma_lines.append(
                f"            simdgroup_multiply_accumulate(C{ti}{tj}, A{ti}, B{tj}, C{ti}{tj});"
            )
    mma_load = "\n".join(mma_lines)

    # Final accumulator store into Ws (then copied to device y as partial result)
    store_lines = []
    for ti in range(TM):
        for tj in range(TN):
            store_lines.append(
                f"    simdgroup_store(C{ti}{tj}, &Ws[(tm + {ti} * {WM} * 8) * BN + tn + {tj} * {WN} * 8], BN);"
            )
    c_store = "\n".join(store_lines)

    # Unrolled vectorized X load (per thread)
    x_vec = "\n".join(
        f"                x_dst[{i}] = float(x_src[{i}]);" for i in range(x_n_reads)
    )
    # Unrolled W dequantize (per thread)
    w_vec = "\n".join(
        f"                w_dst[{i}] = scale * float(w_src[{i}]) + bias;" for i in range(w_n_reads)
    )

    return f"""
    const int BM = {BM};
    const int BN = {BN};
    const int BK = {BK};
    const int BK_PAD = {BK_PAD};
    const int K = {K_val};
    const int N = {N_val};
    const int M = {M_val};
    const int GROUP_SIZE = {gs};
    const int K_groups = {K_val // gs};
    const int PARTITION_SIZE = {split_k_partition_size};
    const int GEMM_K_ITERS = {gemm_k_iterations};
    const int SPLIT_K_PARTS = {split_k_partitions};
    const int PART_STRIDE = M * N;

    uint3 tid = threadgroup_position_in_grid;
    uint sgid = simdgroup_index_in_threadgroup;
    uint slid = thread_index_in_simdgroup;
    int thread_idx = sgid * 32 + slid;
    int partition_idx = tid.z;

    int y_row = tid.y * BM;
    int y_col = tid.x * BN;

    if (y_row >= M || y_col >= N) return;

    threadgroup float Xs[{BM} * {BK_PAD}];
    threadgroup float Ws[{BN} * {BK_PAD}];

    int k_start = PARTITION_SIZE * partition_idx;
    int k_group_start = k_start / GROUP_SIZE;

    // Pointer setup. Scales/biases are stored as bf16 in real quantized
    // models (the source Linear weight was bf16 before quantization), and the
    // weight itself is a packed uint32 buffer — we reinterpret it as uint8*.
    const device bfloat16_t* x_base = (const device bfloat16_t*)x + y_row * K + k_start;
    const device uint8_t*    w_base = (const device uint8_t*)w    + y_col * K + k_start;
    const device bfloat16_t* s_base = (const device bfloat16_t*)scales + y_col * K_groups + k_group_start;
    const device bfloat16_t* b_base = (const device bfloat16_t*)biases + y_col * K_groups + k_group_start;

    // BlockLoader X: (BROWS=BM, BCOLS=BK)
    int x_bi = thread_idx / {x_TCOLS};
    int x_bj = {x_n_reads} * (thread_idx % {x_TCOLS});
    const device bfloat16_t* x_src = x_base + x_bi * K + x_bj;
    threadgroup float* x_dst = Xs + x_bi * BK_PAD + x_bj;

    // QuantizedBlockLoader W: (BROWS=BN, BCOLS=BK, bits=8, pack_factor=1)
    int w_bi = ({w_n_reads} * thread_idx) / {BK};
    int w_bj = ({w_n_reads} * thread_idx) % {BK};
    const device uint8_t*    w_src    = w_base + w_bi * K + w_bj;
    const device bfloat16_t* w_scales = s_base + w_bi * K_groups;
    const device bfloat16_t* w_biases = b_base + w_bi * K_groups;
    threadgroup float* w_dst = Ws + w_bi * BK_PAD + w_bj;

    // MMA setup
    short sg_row = sgid / {WN};
    short sg_col = sgid % {WN};
    short tm = 8 * sg_row;
    short tn = 8 * sg_col;
    {c_decl}

    int k_iters = GEMM_K_ITERS;
    if (partition_idx == SPLIT_K_PARTS - 1) {{
        k_iters = (K - k_start) / BK;
    }}

    // K-loop: each iter advances by BK == GROUP_SIZE, so scales/biases
    // advance by exactly one group per iteration.
    for (int ki = 0; ki < k_iters; ki++) {{

        // Load X tile (bf16 -> float)
        if (x_bi < BM) {{
{x_vec}
        }}

        // Load + dequantize W tile (uint8 -> float via scale * v + bias)
        if (w_bi < BN) {{
            float scale = float(*w_scales);
            float bias  = float(*w_biases);
{w_vec}
        }}

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        // MMA: BK/8 simdgroup 8x8 accumulations
        for (short kk = 0; kk < BK; kk += 8) {{
{mma_load}
        }}

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        x_src += BK;
        w_src += BK;
        w_scales += 1;
        w_biases += 1;
    }}

    // Write partial fp32 result to y[partition_idx, y_row:y_row+BM, y_col:y_col+BN]
    {c_store}
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    for (int i = thread_idx; i < BM * BN; i += {tgp_size}) {{
        int r = i / BN;
        int c = i % BN;
        if (y_row + r < M && y_col + c < N) {{
            y[partition_idx * PART_STRIDE + (y_row + r) * N + y_col + c] = Ws[r * BN + c];
        }}
    }}
"""


def _gen_accum_source(split_k_partitions):
    """Identical to gemm_splitk._gen_accum_source — sum partitions, cast to bf16."""
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


def custom_qmm_splitk(x, w, scales, biases, M, N, K, BM=16, group_size=64):
    """Split-K int8 quantized GEMM.

    Args:
        x: (M, K) bfloat16 input
        w: (N, K/4) uint32 packed int8 weights (pack_factor=1 for bits=8)
        scales: (N, K/gs) bfloat16
        biases: (N, K/gs) bfloat16
        M, N, K: matrix dimensions
        BM: 8, 16, or 32
        group_size: quantization group size (must be 64 for now — BK is fixed to gs)
    Returns:
        y: (M, N) bfloat16
    """
    assert group_size == 64, "qmm_splitk currently fixes BK=GROUP_SIZE=64"
    BN = 32
    BK = group_size
    if BM <= 8:
        WM, WN = 1, 2
    else:
        WM, WN = 2, 2
    P = compute_partitions(M, N, K)

    gemm_key = (M, N, K, BM, P, group_size)
    if gemm_key not in _gemm_cache:
        _gemm_cache[gemm_key] = mx.fast.metal_kernel(
            name=f"qmm_splitk_bm{BM}_M{M}_N{N}_K{K}_P{P}",
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            source=_gen_splitk_qmm_source(M, N, K, BM, P, group_size),
        )
    gemm_kern = _gemm_cache[gemm_key]

    sgs = WM * WN
    n_tg_n = ceil_div(N, BN)
    n_tg_m = ceil_div(M, BM)

    c_split = gemm_kern(
        inputs=[x, w, scales, biases],
        output_shapes=[(P * M * N,)],
        output_dtypes=[mx.float32],
        grid=(32 * n_tg_n, sgs * n_tg_m, P),
        threadgroup=(32, sgs, 1),
    )[0]

    accum_key = P
    if accum_key not in _accum_cache:
        _accum_cache[accum_key] = mx.fast.metal_kernel(
            name=f"qmm_splitk_accum_P{P}",
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
