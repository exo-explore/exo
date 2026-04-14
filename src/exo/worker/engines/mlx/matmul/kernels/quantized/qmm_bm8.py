#!/usr/bin/env python3
"""Custom quantized GEMM kernel with BM=8 (vs MLX's GEMV at BS=8).

At BS=8, MLX uses affine_qmv_fast (GEMV) which loads weights 8 times
via grid x. Our GEMM loads weights once via TG memory tiling.

BM=8, BN=32, BK=32, WM=1, WN=2 → 2 SGs = 64 threads.
TM = BM/(8*WM) = 8/8 = 1, TN = BN/(8*WN) = 32/16 = 2.
Each SG: 1×2 output fragments (8×16).

Uses simdgroup_matrix<float, 8, 8> for hardware MMA.
Inlines QuantizedBlockLoader and BlockLoader for 8-bit gs=64.

Usage:
    from custom_qmm_bm8 import custom_qmm_t_bm8
    y = custom_qmm_t_bm8(x, w, scales, biases, M=8, N=8192, K=2048)
"""

import mlx.core as mx


def ceil_div(a, b):
    return (a + b - 1) // b


def _gen_custom_qmm_source(M_val, N_val, K_val, group_size=64):
    gs = group_size
    BM = 8
    BN = 32
    BK = 32
    WM = 1
    WN = 2
    SIMD_SIZE = 32
    tgp_size = WM * WN * SIMD_SIZE  # 64

    BK_PAD = BK + 16 // 2  # +8 for bfloat16 (16/sizeof(bf16)=8)

    # QuantizedBlockLoader params for (BROWS=BN=32, BCOLS=BK=32, tgp_size=64, bits=8)
    # pack_factor=1 for 8-bit, BCOLS_PACKED=32
    # n_reads = (32*32)/64 = 16
    w_n_reads = (BN * BK) // tgp_size  # 16

    # BlockLoader params for input X (BROWS=BM=8, BCOLS=BK=32, tgp_size=64)
    # n_reads = (8*32)/64 = 4
    # TCOLS = 32/4 = 8, TROWS = 64/8 = 8
    x_n_reads = (BM * BK) // tgp_size  # 4
    x_TCOLS = BK // x_n_reads  # 8
    x_TROWS = tgp_size // x_TCOLS  # 8
    x_n_rows = ceil_div(BM, x_TROWS)  # 1

    K_groups = K_val // gs

    return f"""
    // ═══ Constants ═══
    const int BM = {BM};
    const int BN = {BN};
    const int BK = {BK};
    const int BK_PAD = {BK_PAD};
    const int K = {K_val};
    const int N = {N_val};
    const int M = {M_val};
    const int K_groups = {K_groups};
    const int GROUP_SIZE = {gs};

    uint3 tid = threadgroup_position_in_grid;
    uint lid = thread_index_in_threadgroup;
    uint sgid = simdgroup_index_in_threadgroup;
    uint slid = thread_index_in_simdgroup;
    int thread_idx = sgid * 32 + slid;

    int y_row = tid.y * BM;  // M-dimension tile start
    int y_col = tid.x * BN;  // N-dimension tile start

    // ═══ TG memory ═══
    threadgroup float Xs[{BM} * {BK_PAD}];
    threadgroup float Ws[{BN} * {BK_PAD}];

    // ═══ Pointer setup (follows qmm_t_impl lines 1137-1148) ═══
    const device bfloat16_t* x_base = (const device bfloat16_t*)x + y_row * K;
    const device uint8_t* w_base = (const device uint8_t*)w + y_col * K;
    const device bfloat16_t* s_base = (const device bfloat16_t*)scales + y_col * K_groups;
    const device bfloat16_t* b_base = (const device bfloat16_t*)biases + y_col * K_groups;

    // ═══ BlockLoader for X: (BROWS=8, BCOLS=32, tgp_size=64) ═══
    // n_reads=4, TCOLS=8, TROWS=8, n_rows=1
    int x_bi = thread_idx / {x_TCOLS};           // row in tile (0..15)
    int x_bj = {x_n_reads} * (thread_idx % {x_TCOLS});  // col start in tile
    const device bfloat16_t* x_src = x_base + x_bi * K + x_bj;
    threadgroup float* x_dst = Xs + x_bi * BK_PAD + x_bj;

    // ═══ QuantizedBlockLoader for W: (BROWS=32, BCOLS=32, tgp_size=64, bits=8) ═══
    // n_reads=16, pack_factor=1
    int w_bi = {w_n_reads} * thread_idx / {BK};  // row in tile (0..31)
    int w_bj = ({w_n_reads} * thread_idx) % {BK}; // col start in tile
    const device uint8_t* w_src = w_base + w_bi * K + w_bj;
    const device bfloat16_t* w_scales = s_base + w_bi * K_groups;
    const device bfloat16_t* w_biases = b_base + w_bi * K_groups;
    threadgroup float* w_dst = Ws + w_bi * BK_PAD + w_bj;
    short w_group_step_cnt = 0;
    const int w_group_steps = GROUP_SIZE / BK;  // 64/32 = 2

    // ═══ MMA setup (follows BlockMMA constructor, lines 488-505) ═══
    // WM=1, WN=2: sgid layout: sg_row=0 always, sg_col=sgid (0 or 1)
    // TM=1, TN=2: each SG has 1×2 output fragments
    short sg_row = 0;
    short sg_col = sgid;             // 0 or 1
    short tm = 0;                    // M offset (always 0, only 8 rows = 1 fragment)
    short tn = 8 * sg_col;          // N offset: 0 or 8

    // Accumulators: TM=1 × TN=2 = 2 fragments per SG
    simdgroup_matrix<float, 8, 8> C00 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> C01 = simdgroup_matrix<float, 8, 8>(0);

    // Offsets into TG memory for this SG's MMA reads
    short As_offset_m = tm;  // row offset in Xs
    short Bs_offset_n0 = tn;          // first N-fragment
    short Bs_offset_n1 = tn + {WN} * 8; // second N-fragment (offset by WN*8=16)

    // ═══ K-loop ═══
    for (int k = 0; k < K; k += BK) {{

        // ── Load X tile: BM×BK (bf16 → float) ──
        // BlockLoader pattern: each thread loads n_reads={x_n_reads} elements
        if (x_bi < BM) {{
            for (int i = 0; i < {x_n_reads}; i++) {{
                x_dst[i] = float(x_src[i]);
            }}
        }}

        // ── Load + dequantize W tile: BN×BK (uint8 → float) ──
        // QuantizedBlockLoader pattern: each thread loads n_reads={w_n_reads} elements
        if (w_bi < BN) {{
            float scale = float(*w_scales);
            float bias = float(*w_biases);
            for (int i = 0; i < {w_n_reads}; i++) {{
                w_dst[i] = scale * float(w_src[i]) + bias;
            }}
        }}

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        // ── MMA inner loop: BK/8 = 4 iterations ──
        for (short kk = 0; kk < BK; kk += 8) {{
            simdgroup_matrix<float, 8, 8> A_frag;
            simdgroup_matrix<float, 8, 8> B_frag0, B_frag1;

            // Load A fragment: 8×8 from Xs at (As_offset_m, kk)
            simdgroup_load(A_frag, &Xs[As_offset_m * BK_PAD + kk], BK_PAD);

            // Load B fragments: transposed from Ws (stored as BN × BK_PAD)
            // B_frag0: output cols [Bs_offset_n0..+8], K rows [kk..+8]
            // B_frag1: output cols [Bs_offset_n1..+8], K rows [kk..+8]
            simdgroup_load(B_frag0, &Ws[Bs_offset_n0 * BK_PAD + kk], BK_PAD, ulong2(0, 0), true);
            simdgroup_load(B_frag1, &Ws[Bs_offset_n1 * BK_PAD + kk], BK_PAD, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(C00, A_frag, B_frag0, C00);
            simdgroup_multiply_accumulate(C01, A_frag, B_frag1, C01);
        }}

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        // ── Advance loaders ──
        x_src += BK;
        w_src += BK;
        w_group_step_cnt++;
        if (w_group_step_cnt == w_group_steps) {{
            w_group_step_cnt = 0;
            w_scales++;
            w_biases++;
        }}
    }}

    // ═══ Store results to device memory ═══
    // Each SG stores its 1×2 fragments (two 8×8 blocks)
    // Output layout: y[m * N + n] for m in [y_row..y_row+BM), n in [y_col..y_col+BN)
    device bfloat16_t* y_ptr = (device bfloat16_t*)y + y_row * N + y_col;

    // Store C00 at (tm, tn) and C01 at (tm, tn + WN*8)
    simdgroup_store(C00, &Ws[tm * BN + tn], BN);
    simdgroup_store(C01, &Ws[tm * BN + tn + {WN} * 8], BN);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Convert float → bf16 and write to device
    for (int i = thread_idx; i < BM * BN; i += {tgp_size}) {{
        int r = i / BN;
        int c = i % BN;
        if (y_row + r < M && y_col + c < N) {{
            y_ptr[r * N + c] = static_cast<bfloat16_t>(Ws[r * BN + c]);
        }}
    }}
"""


_custom_qmm_cache = {}


def custom_qmm_t_bm8(x, w, scales, biases, M, N, K, group_size=64):
    """Custom quantized GEMM with BM=8.

    Args:
        x: (M, K) bfloat16 input
        w: (N, K/4) uint32 packed 8-bit weights
        scales: (N, K/gs) bfloat16
        biases: (N, K/gs) bfloat16
        M, N, K: matrix dimensions
    Returns:
        y: (M, N) bfloat16
    """
    key = (M, N, K, group_size)
    if key not in _custom_qmm_cache:
        _custom_qmm_cache[key] = mx.fast.metal_kernel(
            name=f"custom_qmm_t_bm8_M{M}_N{N}_K{K}",
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            source=_gen_custom_qmm_source(M, N, K, group_size),
        )
    kern = _custom_qmm_cache[key]

    BN = 32
    BM = 8
    n_tg_n = ceil_div(N, BN)
    n_tg_m = ceil_div(M, BM)

    result = kern(
        inputs=[x, w, scales, biases],
        output_shapes=[(M * N,)],
        output_dtypes=[mx.bfloat16],
        grid=(32 * n_tg_n, 2 * n_tg_m, 1),
        threadgroup=(32, 2, 1),
    )

    return result[0].reshape(M, N)
