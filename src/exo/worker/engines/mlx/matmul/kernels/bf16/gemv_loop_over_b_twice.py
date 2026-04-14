#!/usr/bin/env python3
"""Two-pass LPB inside a single kernel for large M.

Same threadgroup processes batches 0..M/2-1 first, then M/2..M-1.
First pass results are written to threadgroup memory to free registers.
Second pass reuses the same registers. Both results written to device
memory at the end.

Register pressure: R = 4*(M/2) + 5 per pass (not 2x).
At M=16: R = 37 per pass instead of R = 69 single-pass.

Usage:
    from gemv_loop_over_b_twice import custom_bf16_qmv_loop_over_b_twice
    y = custom_bf16_qmv_loop_over_b_twice(x, w, M=16, N=17408, K=5120)
"""

import mlx.core as mx


def ceil_div(a, b):
    return (a + b - 1) // b


def _gen_bf16_qmv_twice_source(M_val, N_val, K_val):
    # Pass 0 takes ceil(M/2) rows, pass 1 takes floor(M/2) rows.
    # When M is odd, half_a > half_b by 1.
    half_a = (M_val + 1) // 2
    half_b = M_val - half_a
    # Threadgroup memory sized for the larger pass (half_a)
    tg_size = 2 * 4 * half_a

    return f"""
    const int RESULTS_PER_SG = 4;
    const int VALUES_PER_THREAD = 8;
    const int BLOCK_SIZE = 256;
    const int K = {K_val};
    const int N = {N_val};
    const int HALF_A = {half_a};
    const int HALF_B = {half_b};

    uint3 tgid = threadgroup_position_in_grid;
    uint sgid = simdgroup_index_in_threadgroup;
    uint slid = thread_index_in_simdgroup;
    int tg = tgid.y;

    int out_row = tg * 8 + sgid * RESULTS_PER_SG;
    if (out_row >= N) return;

    const device bfloat16_t* w_base = (const device bfloat16_t*)w + (long)out_row * K + slid * VALUES_PER_THREAD;

    // Threadgroup memory to store pass 0 results (frees registers for pass 1)
    threadgroup float tg_results[{tg_size}];
    int tg_offset = sgid * {4 * half_a};  // each SG gets its own slice

    // ══════ Pass 0: batches 0..HALF_A-1 ══════
    {{
        float result[{4 * half_a}];
        for (int i = 0; i < {4 * half_a}; i++) result[i] = 0;

        const device bfloat16_t* ws = w_base;
        int x_base = slid * VALUES_PER_THREAD;

        for (int k_off = 0; k_off < K; k_off += BLOCK_SIZE) {{
            for (int row = 0; row < RESULTS_PER_SG; row++) {{
                const device bfloat16_t* wl = ws + row * K;
                for (int b = 0; b < {half_a}; b++) {{
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

        // Reduce and store to threadgroup memory
        for (int i = 0; i < {4 * half_a}; i++) {{
            result[i] = simd_sum(result[i]);
        }}
        if (slid == 0u) {{
            for (int i = 0; i < {4 * half_a}; i++) {{
                tg_results[tg_offset + i] = result[i];
            }}
        }}
    }}
    // result[] goes out of scope here — registers freed
""" + (f"""
    // ══════ Pass 1: batches HALF_A..M-1 (HALF_B rows) ══════
    {{
        float result[{4 * half_b}];
        for (int i = 0; i < {4 * half_b}; i++) result[i] = 0;

        const device bfloat16_t* ws = w_base;
        int x_base = slid * VALUES_PER_THREAD;

        for (int k_off = 0; k_off < K; k_off += BLOCK_SIZE) {{
            for (int row = 0; row < RESULTS_PER_SG; row++) {{
                const device bfloat16_t* wl = ws + row * K;
                for (int b = 0; b < {half_b}; b++) {{
                    float accum = 0;
                    for (int i = 0; i < VALUES_PER_THREAD; i++) {{
                        float xi = float(((const device bfloat16_t*)x)[(HALF_A + b) * K + x_base + i]);
                        accum += xi * float(wl[i]);
                    }}
                    result[b * 4 + row] += accum;
                }}
            }}
            ws += BLOCK_SIZE;
            x_base += BLOCK_SIZE;
        }}

        // Reduce
        for (int i = 0; i < {4 * half_b}; i++) {{
            result[i] = simd_sum(result[i]);
        }}

        // Write pass 1 results to device memory
        if (slid < 4u) {{
            for (int b = 0; b < {half_b}; b++) {{
                int r = out_row + (int)slid;
                if (r < N) {{
                    y[(HALF_A + b) * N + r] = static_cast<bfloat16_t>(result[b * 4 + slid]);
                }}
            }}
        }}
    }}
""" if half_b > 0 else "") + f"""

    // Write pass 0 results from threadgroup memory to device memory
    if (slid < 4u) {{
        for (int b = 0; b < {half_a}; b++) {{
            int r = out_row + (int)slid;
            if (r < N) {{
                y[b * N + r] = static_cast<bfloat16_t>(tg_results[tg_offset + b * 4 + slid]);
            }}
        }}
    }}
"""


_cache = {}


def custom_bf16_qmv_loop_over_b_twice(x, w, M, N, K):
    key = (M, N, K)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"custom_bf16_qmv_loop_b_2x_M{M}_N{N}_K{K}",
            input_names=["x", "w"],
            output_names=["y"],
            source=_gen_bf16_qmv_twice_source(M, N, K),
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
