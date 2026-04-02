#!/usr/bin/env python3
"""Loop-over-B GEMV for bf16 (non-quantized) matmul.

Same pattern as custom_qmv_loop_over_b.py but for bf16 weights:
no dequantization, no scales/biases — just bf16 weight reads.

Y = X @ W^T where W is (N, K) bf16.

TG: (32, 2, 1) = 64 threads = 2 SGs.
Each SG: 4 output rows. B loop inside row loop.

Usage:
    from custom_bf16_qmv_loop_over_b import custom_bf16_qmv_loop_over_b
    y = custom_bf16_qmv_loop_over_b(x, w, M=4, N=8192, K=2048)
"""

import mlx.core as mx


def ceil_div(a, b):
    return (a + b - 1) // b


def _gen_bf16_qmv_source(M_val, N_val, K_val):
    B = M_val
    # bf16 weights: each element is 2 bytes, read as bfloat16_t
    # No quantization groups — just w[row * K + col]
    # Each thread handles 8 K-elements per block (VALUES_PER_THREAD=8)
    # Block size = 32 threads × 8 = 256 elements per K-iteration

    return f"""
    const int RESULTS_PER_SG = 4;
    const int VALUES_PER_THREAD = 8;
    const int BLOCK_SIZE = 256;
    const int K = {K_val};
    const int N = {N_val};

    uint3 tgid = threadgroup_position_in_grid;
    uint sgid = simdgroup_index_in_threadgroup;
    uint slid = thread_index_in_simdgroup;
    int tg = tgid.y;

    int out_row = tg * 8 + sgid * RESULTS_PER_SG;
    if (out_row >= N) return;

    // Weight pointer: w is (N, K) bf16
    const device bfloat16_t* ws = (const device bfloat16_t*)w + (long)out_row * K + slid * VALUES_PER_THREAD;

    // Result accumulators: 4 rows × B batches
    float result[{4 * B}];
    for (int i = 0; i < {4 * B}; i++) result[i] = 0;

    int x_base = slid * VALUES_PER_THREAD;

    // K-loop: loop over B inside row loop
    for (int k_off = 0; k_off < K; k_off += BLOCK_SIZE) {{

        for (int row = 0; row < RESULTS_PER_SG; row++) {{
            const device bfloat16_t* wl = ws + row * K;

            for (int b = 0; b < {B}; b++) {{
                float accum = 0;
                for (int i = 0; i < VALUES_PER_THREAD; i++) {{
                    float xi = float(((const device bfloat16_t*)x)[b * K + x_base + i]);
                    accum += xi * float(wl[i]);
                }}
                result[b * 4 + row] += accum;
            }}
        }}

        ws += BLOCK_SIZE;
        x_base += BLOCK_SIZE;
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


_cache = {}


def custom_bf16_qmv_loop_over_b(x, w, M, N, K):
    key = (M, N, K)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"custom_bf16_qmv_loop_b_M{M}_N{N}_K{K}",
            input_names=["x", "w"],
            output_names=["y"],
            source=_gen_bf16_qmv_source(M, N, K),
        )
    kern = _cache[key]

    n_tg = ceil_div(N, 8)
    result = kern(
        inputs=[x, w],
        output_shapes=[(M * N,)],
        output_dtypes=[mx.bfloat16],
        grid=(32, n_tg * 2, 1),
        threadgroup=(32, 2, 1),
    )
    return result[0].reshape(M, N)
