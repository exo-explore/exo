#!/usr/bin/env python3
"""Custom bf16 GEMM with BM=8. Same pattern as custom_qmm_bm8 but no dequantization."""

import mlx.core as mx


def ceil_div(a, b):
    return (a + b - 1) // b


def _gen_bf16_gemm_source(M_val, N_val, K_val, BM):
    BN = 32
    BK = 32
    if BM <= 8:
        WM, WN = 1, 2
    else:
        WM, WN = 2, 2
    tgp_size = WM * WN * 32
    BK_PAD = BK + 8

    # BlockLoader for W: (BROWS=BN=32, BCOLS=BK=32, tgp_size)
    w_n_reads = (BN * BK) // tgp_size

    # BlockLoader for X: (BROWS=BM, BCOLS=BK=32, tgp_size)
    x_total = BM * BK
    x_n_reads = max(1, x_total // tgp_size)
    x_TCOLS = BK // x_n_reads if x_n_reads > 0 else BK
    x_TROWS = tgp_size // x_TCOLS if x_TCOLS > 0 else tgp_size

    # MMA fragment layout
    TM = BM // (8 * WM)  # fragments per SG in M dimension
    TN = BN // (8 * WN)  # fragments per SG in N dimension

    # Generate MMA accumulators, load, and store for TM×TN fragments
    # C[tm_i][tn_j] for tm_i in 0..TM-1, tn_j in 0..TN-1
    c_decl_lines = []
    for ti in range(TM):
        for tj in range(TN):
            c_decl_lines.append(f"    simdgroup_matrix<float, 8, 8> C{ti}{tj} = simdgroup_matrix<float, 8, 8>(0);")
    c_decl = "\n".join(c_decl_lines)

    # MMA inner loop: load A for each TM row, B for each TN col, multiply
    mma_lines = []
    for ti in range(TM):
        mma_lines.append(f"""            simdgroup_matrix<float, 8, 8> A{ti};
            simdgroup_load(A{ti}, &Xs[(tm + {ti} * {WM} * 8) * BK_PAD + kk], BK_PAD);""")
    for tj in range(TN):
        mma_lines.append(f"""            simdgroup_matrix<float, 8, 8> B{tj};
            simdgroup_load(B{tj}, &Ws[(tn + {tj} * {WN} * 8) * BK_PAD + kk], BK_PAD, ulong2(0, 0), true);""")
    for ti in range(TM):
        for tj in range(TN):
            mma_lines.append(f"            simdgroup_multiply_accumulate(C{ti}{tj}, A{ti}, B{tj}, C{ti}{tj});")
    mma_load = "\n".join(mma_lines)

    # Store: each fragment at its (tm + ti*8, tn + tj*WN*8) position
    store_lines = []
    for ti in range(TM):
        for tj in range(TN):
            store_lines.append(f"    simdgroup_store(C{ti}{tj}, &Ws[(tm + {ti} * {WM} * 8) * BN + tn + {tj} * {WN} * 8], BN);")
    c_store = "\n".join(store_lines)

    return f"""
    const int BM = {BM};
    const int BN = {BN};
    const int BK = {BK};
    const int BK_PAD = {BK_PAD};
    const int K = {K_val};
    const int N = {N_val};
    const int M = {M_val};

    uint3 tid = threadgroup_position_in_grid;
    uint sgid = simdgroup_index_in_threadgroup;
    uint slid = thread_index_in_simdgroup;
    int thread_idx = sgid * 32 + slid;

    int y_row = tid.y * BM;
    int y_col = tid.x * BN;

    threadgroup float Xs[{BM} * {BK_PAD}];
    threadgroup float Ws[{BN} * {BK_PAD}];

    // Pointer setup
    const device bfloat16_t* x_base = (const device bfloat16_t*)x + y_row * K;
    const device bfloat16_t* w_base = (const device bfloat16_t*)w + y_col * K;

    // BlockLoader for X
    int x_bi = thread_idx / {x_TCOLS};
    int x_bj = {x_n_reads} * (thread_idx % {x_TCOLS});
    const device bfloat16_t* x_src = x_base + x_bi * K + x_bj;
    threadgroup float* x_dst = Xs + x_bi * BK_PAD + x_bj;

    // BlockLoader for W (bf16, no dequant)
    int w_bi = {w_n_reads} * thread_idx / {BK};
    int w_bj = ({w_n_reads} * thread_idx) % {BK};
    const device bfloat16_t* w_src = w_base + w_bi * K + w_bj;
    threadgroup float* w_dst = Ws + w_bi * BK_PAD + w_bj;

    // MMA setup
    short sg_row = sgid / {WN};
    short sg_col = sgid % {WN};
    short tm = 8 * sg_row;
    short tn = 8 * sg_col;
    {c_decl}

    // K-loop
    for (int k = 0; k < K; k += BK) {{

        // Load X tile (bf16 → float)
        if (x_bi < BM) {{
            for (int i = 0; i < {x_n_reads}; i++) {{
                x_dst[i] = float(x_src[i]);
            }}
        }}

        // Load W tile (bf16 → float, no dequant)
        if (w_bi < BN) {{
            for (int i = 0; i < {w_n_reads}; i++) {{
                w_dst[i] = float(w_src[i]);
            }}
        }}

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        // MMA inner loop
        for (short kk = 0; kk < BK; kk += 8) {{
{mma_load}
        }}

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        x_src += BK;
        w_src += BK;
    }}

    // Store results via TG memory → device
    {c_store}
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    for (int i = thread_idx; i < BM * BN; i += {tgp_size}) {{
        int r = i / BN;
        int c = i % BN;
        if (y_row + r < M && y_col + c < N) {{
            y[(y_row + r) * N + y_col + c] = static_cast<bfloat16_t>(Ws[r * BN + c]);
        }}
    }}
"""


_cache = {}


def custom_bf16_gemm(x, w, M, N, K, BM=8):
    key = (M, N, K, BM)
    if key not in _cache:
        WM = 1 if BM <= 8 else 2
        WN = 2
        sgs = WM * WN
        _cache[key] = mx.fast.metal_kernel(
            name=f"custom_bf16_gemm_bm{BM}_M{M}_N{N}_K{K}",
            input_names=["x", "w"],
            output_names=["y"],
            source=_gen_bf16_gemm_source(M, N, K, BM),
        )
    kern = _cache[key]

    WM = 1 if BM <= 8 else 2
    WN = 2
    sgs = WM * WN
    n_tg_n = ceil_div(N, 32)
    n_tg_m = ceil_div(M, BM)

    result = kern(
        inputs=[x, w],
        output_shapes=[(M * N,)],
        output_dtypes=[mx.bfloat16],
        grid=(32 * n_tg_n, sgs * n_tg_m, 1),
        threadgroup=(32, sgs, 1),
    )
    return result[0].reshape(M, N)
