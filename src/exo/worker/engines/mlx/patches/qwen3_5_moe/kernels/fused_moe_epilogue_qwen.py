"""Fused MoE epilogue for Qwen3.5: weighted sum + shared expert gate + residual.

Computes:
  Y[j] = bf16( Σ_a(scores[a] * D_routed[a,j]) + gate_shared * D_shared[j] + H[j] )

Two modes:
  fuse_sigmoid=False: gate_shared is pre-computed sigmoid output (scalar f32)
  fuse_sigmoid=True:  gate_raw is raw dot product; sigmoid computed internally
"""
import mlx.core as mx


def _gen_epilogue_qwen_source(K, n_active, fuse_sigmoid=False):
    """Metal source for fused MoE epilogue with shared expert gate."""
    if fuse_sigmoid:
        gate_line = """
    // Compute sigmoid from raw gate value
    float gate = 1.0f / (1.0f + metal::exp(-shared_gate_val));"""
    else:
        gate_line = """
    float gate = shared_gate_val;"""

    return f"""
    const int K_const = {K};
    const int n_active_const = {n_active};

    uint tid = thread_position_in_grid.x;
    if (tid >= K_const) return;

    // Weighted sum of routed expert outputs
    float acc = 0.0f;
    for (int a = 0; a < n_active_const; a++) {{
        acc += scores[a] * D_routed[a * K_const + tid];
    }}
{gate_line}

    // Shared expert: multiply by gate, add to accumulator
    float shared_val = float(D_shared[tid]) * gate;

    // Add residual and write
    Y[tid] = static_cast<bfloat16_t>(acc + shared_val + float(H[tid]));
"""


_epilogue_qwen_kernels = {}


def _get_epilogue_qwen_kernel(K, n_active, fuse_sigmoid=False):
    key = (K, n_active, fuse_sigmoid)
    if key not in _epilogue_qwen_kernels:
        sig_tag = "_fsig" if fuse_sigmoid else ""
        _epilogue_qwen_kernels[key] = mx.fast.metal_kernel(
            name=f"fused_moe_epilogue_qwen_K{K}_n{n_active}{sig_tag}",
            input_names=["D_routed", "D_shared", "scores", "H",
                         "shared_gate_val"],
            output_names=["Y"],
            source=_gen_epilogue_qwen_source(K, n_active, fuse_sigmoid),
        )
    return _epilogue_qwen_kernels[key]


def fused_moe_epilogue_qwen(d_routed, d_shared, scores, h,
                             shared_gate, k_val, fuse_sigmoid=False):
    """Fused MoE epilogue with shared expert gate.

    Args:
        d_routed: routed expert outputs (n_active, K) float32
        d_shared: shared expert output (K,) float32
        scores: normalized routing scores (n_active,) float32
        h: residual hidden state (K,) bfloat16
        shared_gate: sigmoid output (fuse_sigmoid=False) or raw dot product
                     (fuse_sigmoid=True), scalar float32
        k_val: hidden dimension
        fuse_sigmoid: if True, compute sigmoid(shared_gate) internally

    Returns:
        Y: (K,) bfloat16 — final layer output
    """
    K = int(k_val)
    n_active = scores.shape[0]
    kern = _get_epilogue_qwen_kernel(K, n_active, fuse_sigmoid)

    # shared_gate is a scalar — pass as 0-d array
    if shared_gate.ndim > 0:
        shared_gate = shared_gate.reshape(())

    tg_size = min(K, 1024)
    n_tg = (K + tg_size - 1) // tg_size

    Y = kern(
        inputs=[d_routed, d_shared, scores, h, shared_gate],
        output_shapes=[(K,)],
        output_dtypes=[mx.bfloat16],
        grid=(n_tg * tg_size, 1, 1),
        threadgroup=(tg_size, 1, 1),
    )
    return Y[0]
