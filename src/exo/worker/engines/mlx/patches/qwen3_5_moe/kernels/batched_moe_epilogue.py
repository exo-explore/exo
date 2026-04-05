"""Batched MoE epilogue for Qwen3.5: weighted sum + shared expert gate + residual.

Computes per batch element:
  Y[b, j] = bf16( Σ_a(scores[b,a] * D_routed[b*n_active+a, j])
                 + sigmoid(gate_raw[b]) * D_shared[b, j]
                 + H[b, j] )

Grid z = B (one set of threads per batch element).
All constants baked into Metal source.
"""
import mlx.core as mx

from ..common import COMPUTE_DTYPE, METAL_HALF_TYPE


def _gen_batched_epilogue_source(K, n_active, B):
    return f"""
    const int K_const = {K};
    const int n_active_const = {n_active};
    const int B_const = {B};

    uint tid = thread_position_in_grid.x;
    uint batch_id = thread_position_in_grid.z;

    if (tid >= K_const || batch_id >= B_const) return;

    // Weighted sum of routed expert outputs for this batch element
    float acc = 0.0f;
    int routed_base = (int)batch_id * n_active_const * K_const;
    int score_base = (int)batch_id * n_active_const;
    for (int a = 0; a < n_active_const; a++) {{
        acc += scores[score_base + a] * D_routed[routed_base + a * K_const + tid];
    }}

    // Shared expert: sigmoid(gate_raw) * D_shared
    float gate_raw_val = gate_raw[(int)batch_id];
    float gate = 1.0f / (1.0f + metal::exp(-gate_raw_val));
    float shared_val = D_shared[(int)batch_id * K_const + tid] * gate;

    // Add residual and write
    Y[(int)batch_id * K_const + tid] = static_cast<bfloat16_t>(
        acc + shared_val + float(H[(int)batch_id * K_const + tid])
    );
"""


_batched_epilogue_cache = {}


def _get_batched_epilogue_kernel(K, n_active, B):
    key = (K, n_active, B)
    if key not in _batched_epilogue_cache:
        _batched_epilogue_cache[key] = mx.fast.metal_kernel(
            name=f"batched_epilogue_K{K}_na{n_active}_B{B}",
            input_names=["D_routed", "D_shared", "scores", "H", "gate_raw"],
            output_names=["Y"],
            source=_gen_batched_epilogue_source(K, n_active, B).replace("bfloat16_t", METAL_HALF_TYPE),
        )
    return _batched_epilogue_cache[key]


def batched_moe_epilogue(d_routed, d_shared, scores, h, gate_raw,
                          k_val, batch_size, n_active):
    """Batched MoE epilogue with fused sigmoid.

    Args:
        d_routed: (B * n_active, K) f32
        d_shared: (B, K) f32
        scores: (B * n_active,) f32
        h: (B, K) bf16 — residual
        gate_raw: (B,) f32 — raw shared expert gate (pre-sigmoid)
        k_val: hidden dimension
        batch_size: B
        n_active: experts per token

    Returns:
        Y: (B, K) bf16
    """
    K = int(k_val)
    B = batch_size
    kern = _get_batched_epilogue_kernel(K, n_active, B)

    tg_size = min(K, 1024)
    n_tg = (K + tg_size - 1) // tg_size

    Y = kern(
        inputs=[d_routed, d_shared.reshape(B * K), scores, h.reshape(B * K), gate_raw],
        output_shapes=[(B * K,)],
        output_dtypes=[COMPUTE_DTYPE],
        grid=(n_tg * tg_size, 1, B),
        threadgroup=(tg_size, 1, 1),
    )

    return Y[0].reshape(B, K)
