"""Fused GDN projections for Qwen3.5-35B-A3B (Dispatch 2).

Single dispatch fuses 4 quantized 8-bit GEMVs + depthwise conv1d + activations:
  - in_proj_qkv (8192×2048): GEMV → conv1d(4-tap) → SiLU → write bf16 + cache update
  - in_proj_z   (4096×2048): GEMV → SiLU → write f32
  - in_proj_b   (32×2048):   GEMV → sigmoid → write f32 (beta for GDN kernel)
  - in_proj_a   (32×2048):   GEMV → g=exp(-exp(A_log)*softplus(a+dt_bias)) → write f32

All 4 projection weight matrices are pre-merged into one contiguous buffer
(W_merged, S_merged, B_merged) for better memory locality and cache behavior.
Merging is done offline at patch time by _patch_gdn_proj_weights().

B/A epilogues compute g and beta in-kernel, eliminating ~8 micro-dispatches
that gated_delta_update would otherwise generate (sigmoid, exp, log, etc.).
The caller can pass g/beta directly to gated_delta_kernel.

TG-level multiplexing: tgid.y routes to different epilogues.
Each TG: 64 threads = 2 SGs of 32, produces 8 output rows (4 per SG).
Standard 8-bit affine GEMV: result = scale * Σ(x[i]*w[i]) + bias * Σ(x[i])

Grid: (32, total_tg * 2, B), TG: (32, 2, 1)
"""

import mlx.core as mx


def ceil_div(a, b):
    return (a + b - 1) // b


def _gen_fused_gdn_projections_source(group_size=64):
    """Generate Metal source for fused GDN projections with merged weights.

    8-bit dequantization (group_size=64):
        result = scale * Σ(x[i]*w[i]) + bias * Σ(x[i])

    Single merged weight buffer indexed by absolute out_row.
    TG routing via tgid.y determines region (epilogue):
        [0, N_QKV_TG):              QKV GEMV + conv1d + SiLU + cache
        [N_QKV_TG, +N_Z_TG):       Z GEMV + SiLU
        [+N_Z_TG, +N_B_TG):        B GEMV → sigmoid → beta (f32)
        [+N_B_TG, +N_A_TG):        A GEMV → g=exp(-exp(A_log)*softplus(a+dt_bias)) (f32)
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
    int N_QKV = N_QKV_val;
    int N_Z = N_Z_val;
    int N_B = N_B_val;
    int N_TOTAL = N_TOTAL_val;

    // TG boundaries
    int N_QKV_TG = N_QKV_TG_val;
    int N_Z_TG = N_Z_TG_val;
    int N_B_TG = N_B_TG_val;

    uint3 tgid = threadgroup_position_in_grid;
    uint sgid = simdgroup_index_in_threadgroup;  // 0 or 1
    uint slid = thread_index_in_simdgroup;       // 0..31
    int b_idx = tgid.z;

    int tg = tgid.y;

    // ─── Determine region and absolute out_row in merged matrix ───
    int out_row;
    int region;  // 0=QKV, 1=Z, 2=B, 3=A

    if (tg < N_QKV_TG) {{
        region = 0;
        out_row = tg * 8 + sgid * RESULTS_PER_SG;
    }} else if (tg < N_QKV_TG + N_Z_TG) {{
        region = 1;
        out_row = N_QKV + (tg - N_QKV_TG) * 8 + sgid * RESULTS_PER_SG;
    }} else if (tg < N_QKV_TG + N_Z_TG + N_B_TG) {{
        region = 2;
        out_row = N_QKV + N_Z + (tg - N_QKV_TG - N_Z_TG) * 8 + sgid * RESULTS_PER_SG;
    }} else {{
        region = 3;
        out_row = N_QKV + N_Z + N_B + (tg - N_QKV_TG - N_Z_TG - N_B_TG) * 8 + sgid * RESULTS_PER_SG;
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
        // ═══ QKV: conv1d(4-tap) + SiLU + cache update ═══
        int c = out_row + (int)slid;  // channel index (= absolute row for QKV)
        if (slid < (uint)RESULTS_PER_SG && c < N_QKV) {{
            float qkv_val = result[slid];

            int conv_dim = N_QKV;
            long cs_base = (long)b_idx * 3 * conv_dim;
            float s0 = float(conv_state[cs_base + 0 * conv_dim + c]);
            float s1 = float(conv_state[cs_base + 1 * conv_dim + c]);
            float s2 = float(conv_state[cs_base + 2 * conv_dim + c]);

            float conv_out = float(conv_w[c * 4 + 0]) * s0
                           + float(conv_w[c * 4 + 1]) * s1
                           + float(conv_w[c * 4 + 2]) * s2
                           + float(conv_w[c * 4 + 3]) * qkv_val;

            float silu_out = conv_out / (1.0f + metal::exp(-conv_out));

            conv_state_out[cs_base + 0 * conv_dim + c] = static_cast<bfloat16_t>(s1);
            conv_state_out[cs_base + 1 * conv_dim + c] = static_cast<bfloat16_t>(s2);
            conv_state_out[cs_base + 2 * conv_dim + c] = static_cast<bfloat16_t>(qkv_val);

            qkv_out[b_idx * conv_dim + c] = static_cast<bfloat16_t>(silu_out);
        }}

    }} else if (region == 1) {{
        // ═══ Z: SiLU → write f32 ═══
        int z_row = out_row - N_QKV + (int)slid;
        if (slid < (uint)RESULTS_PER_SG && z_row < N_Z) {{
            float val = result[slid];
            float silu_val = val / (1.0f + metal::exp(-val));
            z_silu_out[b_idx * N_Z + z_row] = silu_val;
        }}

    }} else if (region == 2) {{
        // ═══ B: sigmoid(result) → beta (f32) ═══
        int b_row = out_row - N_QKV - N_Z + (int)slid;
        if (slid < (uint)RESULTS_PER_SG && b_row < N_B) {{
            float val = result[slid];
            float beta = 1.0f / (1.0f + metal::exp(-val));
            b_out[b_idx * N_B + b_row] = beta;
        }}

    }} else {{
        // ═══ A: g = exp(-exp(A_log) * softplus(a + dt_bias)) → f32 ═══
        int a_row = out_row - N_QKV - N_Z - N_B + (int)slid;
        int N_A = N_TOTAL - N_QKV - N_Z - N_B;
        if (slid < (uint)RESULTS_PER_SG && a_row < N_A) {{
            float a_val = result[slid];
            float dt = float(dt_bias_arr[a_row]);
            float x_g = a_val + dt;
            // softplus(x) = log(1 + exp(x)), with x>20 shortcut for numerical stability
            float sp = (x_g > 20.0f) ? x_g : metal::log(1.0f + metal::exp(x_g));
            float g_val = metal::exp(-metal::exp(float(A_log_arr[a_row])) * sp);
            a_out[b_idx * N_A + a_row] = g_val;
        }}
    }}
"""


_fused_gdn_proj_kernel = None


def _get_fused_gdn_proj_kernel():
    """Get or compile the fused GDN projections kernel."""
    global _fused_gdn_proj_kernel
    if _fused_gdn_proj_kernel is None:
        _fused_gdn_proj_kernel = mx.fast.metal_kernel(
            name="fused_gdn_projections_8bit_merged",
            input_names=[
                "x",
                "W_merged", "S_merged", "B_merged",
                "conv_state", "conv_w",
                "A_log_arr", "dt_bias_arr",
                "K_val",
                "N_QKV_val", "N_Z_val", "N_B_val", "N_TOTAL_val",
                "N_QKV_TG_val", "N_Z_TG_val", "N_B_TG_val",
            ],
            output_names=["qkv_out", "z_silu_out", "b_out", "a_out", "conv_state_out"],
            source=_gen_fused_gdn_projections_source(),
        )
    return _fused_gdn_proj_kernel


def fused_gdn_projections(
    x,
    W_merged, S_merged, B_merged,
    proj_dims,
    conv_state, conv_weights,
    A_log, dt_bias,
    batch_size=1,
):
    """Fused GDN projections: 4 GEMVs + conv1d + activations + g/beta.

    Uses pre-merged contiguous weight buffers for all 4 projections.
    B epilogue computes beta = sigmoid(b) in f32.
    A epilogue computes g = exp(-exp(A_log) * softplus(a + dt_bias)) in f32.
    Caller passes g/beta directly to gated_delta_kernel (no micro-dispatches).

    Args:
        x: [B, 1, K] bf16 — post-RMSNorm hidden state
        W_merged: [N_TOTAL, K/4] uint32 — merged quantized weights
        S_merged: [N_TOTAL, K/gs] bf16 — merged scales
        B_merged: [N_TOTAL, K/gs] bf16 — merged biases
        proj_dims: (N_QKV, N_Z, N_B, N_A) — per-projection output dims
        conv_state:  [B, 3, conv_dim] bf16 — previous 3 timesteps
        conv_weights: [conv_dim, 4, 1] or [conv_dim, 4] bf16 — depthwise conv filters
        A_log: [Hv] f32 — GDN decay log-parameter
        dt_bias: [Hv] f32 — GDN time constant bias
        batch_size: int

    Returns:
        qkv_conv_silu: [B, 1, N_QKV] bf16 — post-conv, post-SiLU
        z_silu:        [B, 1, N_Z] f32  — post-SiLU
        beta:          [B, 1, N_B] f32  — sigmoid(b), ready for GDN kernel
        g:             [B, 1, N_A] f32  — gating, ready for GDN kernel
        conv_state_out: [B, 3, N_QKV] bf16
    """
    B = batch_size
    kern = _get_fused_gdn_proj_kernel()

    N_QKV, N_Z, N_B, N_A = proj_dims
    N_TOTAL = N_QKV + N_Z + N_B + N_A
    K = x.shape[-1]

    # TG counts (8 rows per TG)
    N_QKV_TG = ceil_div(N_QKV, 8)
    N_Z_TG = ceil_div(N_Z, 8)
    N_B_TG = ceil_div(N_B, 8)
    N_A_TG = ceil_div(N_A, 8)
    total_tg = N_QKV_TG + N_Z_TG + N_B_TG + N_A_TG

    # Flatten conv weights to [conv_dim, 4] if needed
    conv_w_flat = conv_weights.reshape(-1, 4) if conv_weights.ndim == 3 else conv_weights

    # Flatten x to [B, K]
    x_flat = x.reshape(B, K)

    results = kern(
        inputs=[
            x_flat,
            W_merged, S_merged, B_merged,
            conv_state, conv_w_flat,
            A_log, dt_bias,
            mx.array(K, dtype=mx.int32),
            mx.array(N_QKV, dtype=mx.int32),
            mx.array(N_Z, dtype=mx.int32),
            mx.array(N_B, dtype=mx.int32),
            mx.array(N_TOTAL, dtype=mx.int32),
            mx.array(N_QKV_TG, dtype=mx.int32),
            mx.array(N_Z_TG, dtype=mx.int32),
            mx.array(N_B_TG, dtype=mx.int32),
        ],
        output_shapes=[
            (B * N_QKV,),              # qkv_out
            (B * N_Z,),                # z_silu_out
            (B * N_B,),                # beta_out (f32)
            (B * N_A,),                # g_out (f32)
            (B * 3 * N_QKV,),          # conv_state_out
        ],
        output_dtypes=[mx.bfloat16, mx.float32, mx.float32, mx.float32, mx.bfloat16],
        grid=(32, total_tg * 2, B),
        threadgroup=(32, 2, 1),
    )

    qkv_out = results[0].reshape(B, 1, N_QKV)
    z_silu = results[1].reshape(B, 1, N_Z)
    beta = results[2].reshape(B, 1, N_B)
    g = results[3].reshape(B, 1, N_A)
    conv_state_out = results[4].reshape(B, 3, N_QKV)

    return qkv_out, z_silu, beta, g, conv_state_out
