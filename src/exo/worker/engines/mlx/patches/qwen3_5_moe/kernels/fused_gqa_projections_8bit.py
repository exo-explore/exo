"""Fused GQA projections for Qwen3.5-35B-A3B (Dispatch 1).

Single dispatch fuses 4 quantized 8-bit GEMVs with region-specific epilogues:
  - q_proj queries (4096×2048): GEMV → raw bf16 write
  - q_proj gate    (4096×2048): GEMV → sigmoid → f32 write
  - k_proj         (512×2048):  GEMV → raw bf16 write
  - v_proj         (512×2048):  GEMV → raw bf16 write

All 4 projection weight matrices are pre-merged into one contiguous buffer
(W_merged, S_merged, B_merged) for better memory locality.
Merging is done offline at patch time by _patch_gqa_proj_weights().

Gate sigmoid is computed in f32 directly from the GEMV accumulator,
avoiding bf16 round-trip and eliminating a separate sigmoid dispatch.

TG-level multiplexing: tgid.y routes to different epilogues.
Each TG: 64 threads = 2 SGs of 32, produces 8 output rows (4 per SG).
Standard 8-bit affine GEMV: result = scale * Σ(x[i]*w[i]) + bias * Σ(x[i])

Grid: (32, total_tg * 2, B), TG: (32, 2, 1)
"""

import mlx.core as mx


def ceil_div(a, b):
    return (a + b - 1) // b


def _gen_fused_gqa_projections_source(group_size=64):
    """Generate Metal source for fused GQA projections with merged weights.

    8-bit dequantization (group_size=64):
        result = scale * Σ(x[i]*w[i]) + bias * Σ(x[i])

    Single merged weight buffer indexed by absolute out_row.
    TG routing via tgid.y determines region (epilogue):
        [0, N_Q_TG):                    Queries GEMV → raw bf16
        [N_Q_TG, +N_GATE_TG):          Gate GEMV → sigmoid → f32
        [+N_GATE_TG, +N_K_TG):         Keys GEMV → raw bf16
        [+N_K_TG, +N_V_TG):            Values GEMV → raw bf16
    """
    gs = int(group_size)
    sc_stride = 256 // gs   # groups consumed per K-block = 4
    slid_div = gs // 8      # threads per group = 8

    return f"""
    const int RESULTS_PER_SG = 4;
    const int VALUES_PER_THREAD = 8;
    const int BLOCK_SIZE = 256;   // 32 * 8
    const int GROUP_SIZE = {gs};
    const int SC_STRIDE = {sc_stride};
    const int SLID_DIV = {slid_div};

    int K = K_val;
    int K_groups = K / GROUP_SIZE;

    // Dimension boundaries
    int N_Q = N_Q_val;
    int N_GATE = N_GATE_val;
    int N_K = N_K_val;
    int N_TOTAL = N_TOTAL_val;

    // TG boundaries
    int N_Q_TG = N_Q_TG_val;
    int N_GATE_TG = N_GATE_TG_val;
    int N_K_TG = N_K_TG_val;

    uint3 tgid = threadgroup_position_in_grid;
    uint sgid = simdgroup_index_in_threadgroup;  // 0 or 1
    uint slid = thread_index_in_simdgroup;       // 0..31
    int b_idx = tgid.z;

    int tg = tgid.y;

    // ─── Determine region and absolute out_row in merged matrix ───
    int out_row;
    int region;  // 0=Q, 1=Gate, 2=K, 3=V

    if (tg < N_Q_TG) {{
        region = 0;
        out_row = tg * 8 + sgid * RESULTS_PER_SG;
    }} else if (tg < N_Q_TG + N_GATE_TG) {{
        region = 1;
        out_row = N_Q + (tg - N_Q_TG) * 8 + sgid * RESULTS_PER_SG;
    }} else if (tg < N_Q_TG + N_GATE_TG + N_K_TG) {{
        region = 2;
        out_row = N_Q + N_GATE + (tg - N_Q_TG - N_GATE_TG) * 8 + sgid * RESULTS_PER_SG;
    }} else {{
        region = 3;
        out_row = N_Q + N_GATE + N_K + (tg - N_Q_TG - N_GATE_TG - N_K_TG) * 8 + sgid * RESULTS_PER_SG;
    }}

    if (out_row >= N_TOTAL) return;

    // ─── Single pointer into merged weight buffer ───
    const device uint8_t* ws = (const device uint8_t*)W_merged + (long)out_row * K + slid * VALUES_PER_THREAD;
    const device bfloat16_t* sc = (const device bfloat16_t*)S_merged + (long)out_row * K_groups + slid / SLID_DIV;
    const device bfloat16_t* bi = (const device bfloat16_t*)B_merged + (long)out_row * K_groups + slid / SLID_DIV;

    // ─── 8-bit GEMV K-loop (unified for all regions) ───
    float result[4] = {{0, 0, 0, 0}};
    int x_base = b_idx * K + slid * VALUES_PER_THREAD;

    for (int k_off = 0; k_off < K; k_off += BLOCK_SIZE) {{
        float x_thread[8];
        float xsum = 0;
        for (int i = 0; i < 8; i++) {{
            float xi = float(x[x_base + i]);
            x_thread[i] = xi;
            xsum += xi;
        }}

        for (int row = 0; row < RESULTS_PER_SG; row++) {{
            const device uint8_t* w = ws + row * K;
            float s_val = float(sc[row * K_groups]);
            float b_val = float(bi[row * K_groups]);
            float accum = 0;
            for (int i = 0; i < 8; i++) {{
                accum += x_thread[i] * float(w[i]);
            }}
            result[row] += s_val * accum + xsum * b_val;
        }}

        ws += BLOCK_SIZE;
        sc += SC_STRIDE;
        bi += SC_STRIDE;
        x_base += BLOCK_SIZE;
    }}

    // ─── Reduction ───
    for (int row = 0; row < RESULTS_PER_SG; row++) {{
        result[row] = simd_sum(result[row]);
    }}

    // ─── Region-specific epilogues ───
    // After simd_sum, all 32 threads have result[0..3].
    // Threads 0-3 each handle one output row.

    if (region == 0) {{
        // ═══ Queries: raw bf16 write ═══
        int q_row = out_row + (int)slid;
        if (slid < (uint)RESULTS_PER_SG && q_row < N_Q) {{
            q_out[b_idx * N_Q + q_row] = static_cast<bfloat16_t>(result[slid]);
        }}

    }} else if (region == 1) {{
        // ═══ Gate: sigmoid → f32 ═══
        int g_row = out_row - N_Q + (int)slid;
        if (slid < (uint)RESULTS_PER_SG && g_row < N_GATE) {{
            float val = result[slid];
            float sig = 1.0f / (1.0f + metal::exp(-val));
            gate_out[b_idx * N_GATE + g_row] = sig;
        }}

    }} else if (region == 2) {{
        // ═══ Keys: raw bf16 write ═══
        int k_row = out_row - N_Q - N_GATE + (int)slid;
        if (slid < (uint)RESULTS_PER_SG && k_row < N_K) {{
            k_out[b_idx * N_K + k_row] = static_cast<bfloat16_t>(result[slid]);
        }}

    }} else {{
        // ═══ Values: raw bf16 write ═══
        int v_row = out_row - N_Q - N_GATE - N_K + (int)slid;
        int N_V = N_TOTAL - N_Q - N_GATE - N_K;
        if (slid < (uint)RESULTS_PER_SG && v_row < N_V) {{
            v_out[b_idx * N_V + v_row] = static_cast<bfloat16_t>(result[slid]);
        }}
    }}
"""


_fused_gqa_proj_kernel = None


def _get_fused_gqa_proj_kernel():
    """Get or compile the fused GQA projections kernel."""
    global _fused_gqa_proj_kernel
    if _fused_gqa_proj_kernel is None:
        _fused_gqa_proj_kernel = mx.fast.metal_kernel(
            name="fused_gqa_projections_8bit_merged",
            input_names=[
                "x",
                "W_merged", "S_merged", "B_merged",
                "K_val",
                "N_Q_val", "N_GATE_val", "N_K_val", "N_TOTAL_val",
                "N_Q_TG_val", "N_GATE_TG_val", "N_K_TG_val",
            ],
            output_names=["q_out", "gate_out", "k_out", "v_out"],
            source=_gen_fused_gqa_projections_source(),
        )
    return _fused_gqa_proj_kernel


def fused_gqa_projections(
    x,
    W_merged, S_merged, B_merged,
    proj_dims,
    batch_size=1,
    scalars=None, total_tg=None,
):
    """Fused GQA projections: 4 GEMVs with region-specific epilogues.

    Uses pre-merged contiguous weight buffers for all 4 projections.
    Gate epilogue computes sigmoid in f32 directly from GEMV accumulator.

    Args:
        x: [B, 1, K] bf16 — post-RMSNorm hidden state
        W_merged: [N_TOTAL, K/4] uint32 — merged quantized weights
        S_merged: [N_TOTAL, K/gs] bf16 — merged scales
        B_merged: [N_TOTAL, K/gs] bf16 — merged biases
        proj_dims: (N_Q, N_GATE, N_K, N_V) — per-projection output dims
        batch_size: int
        scalars: dict of pre-cached mx.array scalars (optional, avoids per-call creation)
        total_tg: pre-computed total TG count (optional)

    Returns:
        queries:      [B, 1, N_Q] bf16
        gate_sigmoid: [B, 1, N_GATE] f32 — sigmoid(gate), ready for post-SDPA multiply
        keys:         [B, 1, N_K] bf16
        values:       [B, 1, N_V] bf16
    """
    B = batch_size
    kern = _get_fused_gqa_proj_kernel()

    N_Q, N_GATE, N_K, N_V = proj_dims
    K = x.shape[-1]

    # Flatten x to [B, K]
    x_flat = x.reshape(B, K)

    if scalars is not None:
        s = scalars
        inputs = [x_flat, W_merged, S_merged, B_merged,
                  s['K'], s['N_Q'], s['N_GATE'], s['N_K'], s['N_TOTAL'],
                  s['N_Q_TG'], s['N_GATE_TG'], s['N_K_TG']]
    else:
        N_TOTAL = N_Q + N_GATE + N_K + N_V
        total_tg = ceil_div(N_Q, 8) + ceil_div(N_GATE, 8) + ceil_div(N_K, 8) + ceil_div(N_V, 8)
        inputs = [x_flat, W_merged, S_merged, B_merged,
                  mx.array(K, dtype=mx.int32),
                  mx.array(N_Q, dtype=mx.int32),
                  mx.array(N_GATE, dtype=mx.int32),
                  mx.array(N_K, dtype=mx.int32),
                  mx.array(N_TOTAL, dtype=mx.int32),
                  mx.array(ceil_div(N_Q, 8), dtype=mx.int32),
                  mx.array(ceil_div(N_GATE, 8), dtype=mx.int32),
                  mx.array(ceil_div(N_K, 8), dtype=mx.int32)]

    if total_tg is None:
        total_tg = ceil_div(N_Q, 8) + ceil_div(N_GATE, 8) + ceil_div(N_K, 8) + ceil_div(N_V, 8)

    results = kern(
        inputs=inputs,
        output_shapes=[
            (B * N_Q,),        # q_out
            (B * N_GATE,),     # gate_out (f32)
            (B * N_K,),        # k_out
            (B * N_V,),        # v_out
        ],
        output_dtypes=[mx.bfloat16, mx.float32, mx.bfloat16, mx.bfloat16],
        grid=(32, total_tg * 2, B),
        threadgroup=(32, 2, 1),
    )

    queries = results[0].reshape(B, 1, N_Q)
    gate_sigmoid = results[1].reshape(B, 1, N_GATE)
    keys = results[2].reshape(B, 1, N_K)
    values = results[3].reshape(B, 1, N_V)

    return queries, gate_sigmoid, keys, values
