#!/usr/bin/env python3
"""Speculative variant of the GatedDeltaNet kernel.

Identical to the original gated_delta_kernel but also outputs per-step
recurrent states for rollback during speculative decoding.

The original kernel only writes the FINAL state. This variant writes
the state at EVERY timestep to an extra output buffer `all_states`.
"""

from typing import Optional, Tuple
import mlx.core as mx


def _make_speculative_gated_delta_kernel(has_mask=False, vectorized=False):
    mask_source = "mask[b_idx * T + t]" if has_mask else "true"

    if vectorized:
        g_comment = "// g: [B, T, Hv, Dk]"
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_access = "g_[s_idx]"
        g_advance = "g_ += Hv * Dk;"
    else:
        g_comment = "// g: [B, T, Hv]"
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_access = "g_[hv_idx]"
        g_advance = "g_ += Hv;"

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

        // all_states: [B, T, Hv, Dv, Dk] — per-step state output
        auto a_state = all_states + (b_idx * T * Hv * Dv + hv_idx * Dv + dv_idx) * Dk;
        auto a_stride = Hv * Dv * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        {g_comment}
        {g_setup}
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          if ({mask_source}) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * {g_access};
              kv_mem += state[i] * k_[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

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

          // Save per-step state for speculative rollback
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            a_state[s_idx] = static_cast<InT>(state[i]);
          }}
          a_state += a_stride;

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          {g_advance}
          beta_ += Hv;
        }}
        // Write final state (same as original kernel)
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }}
    """
    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")

    suffix = "_spec"
    if vectorized:
        suffix += "_vec"
    if has_mask:
        suffix += "_mask"

    return mx.fast.metal_kernel(
        name=f"gated_delta_step{suffix}",
        input_names=inputs,
        output_names=["y", "state_out", "all_states"],
        source=source,
    )


# Pre-build kernel variants
_spec_kernel = _make_speculative_gated_delta_kernel(has_mask=False, vectorized=False)
_spec_kernel_masked = _make_speculative_gated_delta_kernel(has_mask=True, vectorized=False)
_spec_kernel_vec = _make_speculative_gated_delta_kernel(has_mask=False, vectorized=True)
_spec_kernel_vec_masked = _make_speculative_gated_delta_kernel(has_mask=True, vectorized=True)


def speculative_gated_delta_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Like gated_delta_kernel but also returns per-step states.

    Returns:
        y: [B, T, Hv, Dv] — output (same as original)
        state_out: [B, Hv, Dv, Dk] — final state (same as original)
        all_states: [B, T, Hv, Dv, Dk] — state after each timestep
    """
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    input_type = q.dtype

    if g.ndim == 4:
        kernel = _spec_kernel_vec
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _spec_kernel_vec_masked
            inputs.append(mask)
    else:
        kernel = _spec_kernel
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _spec_kernel_masked
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
        output_shapes=[(B, T, Hv, Dv), state.shape, (B, T, Hv, Dv, Dk)],
        output_dtypes=[input_type, input_type, input_type],
    )
