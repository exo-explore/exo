#!/usr/bin/env python3
"""Isolated loop-over-B GEMV kernel for quantized matmul.

Extracts the loop-over-B pattern from batched_fused_gdn_projections_8bit
but without any epilogues — pure Y = X @ dequant(W)^T output.

For comparing our GEMV approach against MLX's affine_qmv_fast on
an isolated QuantizedLinear operation (e.g., in_proj_qkv: N=8192, K=2048).

TG: (32, 2, 1) = 64 threads = 2 SGs.
Each SG: 4 output rows.
B loop inside row loop for low register pressure (R = 4B + 5).

Usage:
    from custom_qmv_loop_over_b import custom_qmv_loop_over_b
    y = custom_qmv_loop_over_b(x, w, scales, biases, M=8, N=8192, K=2048)
"""

import mlx.core as mx


def ceil_div(a, b):
    return (a + b - 1) // b


def _gen_custom_qmv_source(M_val, N_val, K_val, group_size=64):
    gs = group_size
    sc_stride = 256 // gs
    slid_div = gs // 8
    K_groups = K_val // gs
    B = M_val  # batch size = M

    return f"""
    const int RESULTS_PER_SG = 4;
    const int VALUES_PER_THREAD = 8;
    const int BLOCK_SIZE = 256;
    const int K = {K_val};
    const int N = {N_val};
    const int M = {M_val};
    const int K_groups = {K_groups};
    const int SC_STRIDE = {sc_stride};
    const int SLID_DIV = {slid_div};

    uint3 tgid = threadgroup_position_in_grid;
    uint sgid = simdgroup_index_in_threadgroup;
    uint slid = thread_index_in_simdgroup;
    int tg = tgid.y;

    int out_row = tg * 8 + sgid * RESULTS_PER_SG;
    if (out_row >= N) return;

    // Weight pointers
    const device uint8_t* ws = (const device uint8_t*)w + (long)out_row * K + slid * VALUES_PER_THREAD;
    const device bfloat16_t* sc = (const device bfloat16_t*)scales + (long)out_row * K_groups + slid / SLID_DIV;
    const device bfloat16_t* bi = (const device bfloat16_t*)biases + (long)out_row * K_groups + slid / SLID_DIV;

    // Result accumulators: 4 rows × B batches
    float result[{4 * B}];
    for (int i = 0; i < {4 * B}; i++) result[i] = 0;

    int x_base = slid * VALUES_PER_THREAD;

    // K-loop: loop over B inside row loop
    for (int k_off = 0; k_off < K; k_off += BLOCK_SIZE) {{

        for (int row = 0; row < RESULTS_PER_SG; row++) {{
            const device uint8_t* wl = ws + row * K;
            float s_val = float(sc[row * K_groups]);
            float b_val = float(bi[row * K_groups]);

            for (int b = 0; b < {B}; b++) {{
                float accum = 0, xsum = 0;
                for (int i = 0; i < VALUES_PER_THREAD; i++) {{
                    float xi = float(((const device bfloat16_t*)x)[b * K + x_base + i]);
                    accum += xi * float(wl[i]);
                    xsum += xi;
                }}
                result[b * 4 + row] += s_val * accum + xsum * b_val;
            }}
        }}

        ws += BLOCK_SIZE; sc += SC_STRIDE; bi += SC_STRIDE; x_base += BLOCK_SIZE;
    }}

    // Reduction
    for (int i = 0; i < {4 * B}; i++) result[i] = simd_sum(result[i]);

    // Write output (bf16)
    if (slid < 4u) {{
        for (int b = 0; b < {B}; b++) {{
            int r = out_row + (int)slid;
            if (r < N) {{
                y[b * N + r] = static_cast<bfloat16_t>(result[b * 4 + slid]);
            }}
        }}
    }}
"""


_custom_qmv_cache = {}


def custom_qmv_loop_over_b(x, w, scales, biases, M, N, K, group_size=64):
    """Loop-over-B GEMV for quantized matmul.

    Args:
        x: (M, K) bfloat16 input
        w: (N, K/4) uint32 packed 8-bit weights
        scales: (N, K/gs) bfloat16
        biases: (N, K/gs) bfloat16
        M, N, K: dimensions
    Returns:
        y: (M, N) bfloat16
    """
    key = (M, N, K, group_size)
    if key not in _custom_qmv_cache:
        _custom_qmv_cache[key] = mx.fast.metal_kernel(
            name=f"custom_qmv_loop_b_M{M}_N{N}_K{K}",
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            source=_gen_custom_qmv_source(M, N, K, group_size),
        )
    kern = _custom_qmv_cache[key]

    n_tg = ceil_div(N, 8)

    result = kern(
        inputs=[x, w, scales, biases],
        output_shapes=[(M * N,)],
        output_dtypes=[mx.bfloat16],
        grid=(32, n_tg * 2, 1),
        threadgroup=(32, 2, 1),
    )

    return result[0].reshape(M, N)
