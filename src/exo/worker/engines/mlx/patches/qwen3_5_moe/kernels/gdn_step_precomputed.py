"""GDN recurrence with pre-computed g and beta (Dispatch 4).

Modified version of gated_delta_step from mlx-lm-fork/mlx_lm/models/gated_delta.py.
Instead of computing g = exp(-exp(A_log) * softplus(a + dt_bias)) and beta = sigmoid(b)
inside the kernel, accepts them as pre-computed f32 inputs from Dispatch 2.

Non-vectorized only (Qwen3.5-35B-A3B uses scalar gating per head).

Grid: (32, Dv, B*Hv) = (32, 128, B*32), TG: (32, 4, 1)
"""

from typing import Optional, Tuple

import mlx.core as mx


def _make_gdn_precomputed_kernel(has_mask=False):
    """Build the GDN kernel with pre-computed g and beta."""
    if not mx.metal.is_available():
        return None

    mask_source = "mask[b_idx * T + t]" if has_mask else "true"

    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        // g: [B, T, Hv] f32 — pre-computed decay gate
        auto g_ = g + b_idx * T * Hv;
        // beta: [B, T, Hv] f32 — pre-computed sigmoid(b)
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          if ({mask_source}) {{
            // Pre-computed g and beta (no softplus/exp/sigmoid needed)
            float g_val = g_[hv_idx];
            float beta_val = beta_[hv_idx];

            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_val;
              kv_mem += state[i] * k_[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_val;

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }}
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {{
              y[dv_idx] = static_cast<InT>(out);
            }}
          }}
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }}
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }}
    """

    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")

    suffix = "_precomputed"
    if has_mask:
        suffix += "_mask"

    return mx.fast.metal_kernel(
        name=f"gated_delta_step{suffix}",
        input_names=inputs,
        output_names=["y", "state_out"],
        source=source,
    )


_gdn_precomputed_kernel = None
_gdn_precomputed_kernel_masked = None


def _get_gdn_precomputed_kernel(has_mask=False):
    """Get or compile the pre-computed GDN kernel."""
    global _gdn_precomputed_kernel, _gdn_precomputed_kernel_masked
    if has_mask:
        if _gdn_precomputed_kernel_masked is None:
            _gdn_precomputed_kernel_masked = _make_gdn_precomputed_kernel(has_mask=True)
        return _gdn_precomputed_kernel_masked
    else:
        if _gdn_precomputed_kernel is None:
            _gdn_precomputed_kernel = _make_gdn_precomputed_kernel(has_mask=False)
        return _gdn_precomputed_kernel


def gated_delta_update_precomputed(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """GDN recurrence with pre-computed g and beta.

    Args:
        q: [B, T, Hk, Dk] bf16 — normalized q from Dispatch 3
        k: [B, T, Hk, Dk] bf16 — normalized k from Dispatch 3
        v: [B, T, Hv, Dv] bf16 — v from Dispatch 2 (qkv_conv_silu[:, :, 4096:])
        g: [B, T, Hv] f32 — pre-computed decay gate from Dispatch 2
        beta: [B, T, Hv] f32 — pre-computed sigmoid(b) from Dispatch 2
        state: [B, Hv, Dv, Dk] bf16 — recurrent state from cache
        mask: [B, T] optional

    Returns:
        y: [B, T, Hv, Dv] bf16
        new_state: [B, Hv, Dv, Dk] bf16
    """
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    input_type = q.dtype

    kernel = _get_gdn_precomputed_kernel(has_mask=mask is not None)
    inputs = [q, k, v, g, beta, state, T]
    if mask is not None:
        inputs.append(mask)

    return kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[input_type, input_type],
    )
