"""Batched oproj MoE with 4 custom Metal kernel dispatches.

Fuses o_proj + RMSNorm + gate GEMV + softmax + topk + SwiGLU + down_proj + epilogue
into 4 dispatches with register-level weight sharing for the shared expert.
Falls back to vanilla MoE when called without _residual (from vanilla decoder path).
"""

import mlx.core as mx
from mlx.nn.layers.activations import silu as nn_silu

from .common import COMPUTE_DTYPE
from .kernels.batched_merged_down_proj_8bit import batched_merged_down_proj_8bit
from .kernels.batched_moe_epilogue import batched_moe_epilogue
from .kernels.batched_oproj_gate_gemv_8bit import batched_oproj_gate_gemv
from .kernels.batched_softmax_topk_swiglu_8bit import batched_softmax_topk_swiglu_8bit


def _fused_switch_glu(switch_mlp, x, indices):
    """SwitchGLU with fused gate+up gather_qmm (2 dispatches instead of 3).

    Uses pre-concatenated gate+up weights (_fused_w_gu) for a single gather_qmm,
    then splits the output and applies SwiGLU activation before down_proj.
    """
    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

    x = mx.expand_dims(x, (-2, -3))
    do_sort = indices.size >= 64
    idx = indices
    inv_order = None
    if do_sort:
        x, idx, inv_order = _gather_sort(x, indices)

    n_inter = switch_mlp._fused_n_inter

    # Single gather_qmm for gate+up (N=2*n_inter) instead of two (N=n_inter each)
    gu = mx.gather_qmm(
        x,
        switch_mlp._fused_w_gu,
        switch_mlp._fused_s_gu,
        switch_mlp._fused_b_gu,
        rhs_indices=idx,
        transpose=True,
        group_size=switch_mlp._fused_group_size,
        bits=switch_mlp.gate_proj.bits,
        mode=switch_mlp.gate_proj.mode,
        sorted_indices=do_sort,
    )

    # Split into gate and up, apply SwiGLU
    x_gate = gu[..., :n_inter]
    x_up = gu[..., n_inter:]
    x = nn_silu(x_gate) * x_up

    # down_proj (unchanged — still a single gather_qmm)
    x = switch_mlp.down_proj(x, idx, sorted_indices=do_sort)

    if do_sort:
        x = _scatter_unsort(x, inv_order, indices.shape)
    return x.squeeze(-2)


def _fused_shared_expert(moe, x):
    """Shared expert with fused gate+up quantized_matmul (1 dispatch instead of 2+1).

    Uses pre-concatenated gate+up weights (_shared_w_gu) and appended
    shared_expert_gate weight for a single matmul.
    """
    # Fused gate+up for shared expert
    gu = mx.quantized_matmul(
        x,
        moe._shared_w_gu,
        moe._shared_s_gu,
        moe._shared_b_gu,
        transpose=True,
        group_size=moe._shared_gs,
        bits=8,
    )
    shared_inter = moe._shared_inter
    x_gate = gu[..., :shared_inter]
    x_up = gu[..., shared_inter:]
    intermediate = nn_silu(x_gate) * x_up

    # down_proj (unchanged)
    shared_y = mx.quantized_matmul(
        intermediate,
        moe._shared_down_w,
        moe._shared_down_s,
        moe._shared_down_b,
        transpose=True,
        group_size=moe._shared_gs,
        bits=8,
    )

    # shared_expert_gate
    gate = mx.sigmoid(moe.shared_expert_gate(x))
    return gate * shared_y


def _batched_oproj_moe_call(self, attn_out_3d, _residual=None):
    """Batched MoE with full oproj fusion (4 custom dispatches).

    Receives raw attention output (pre-o_proj) and residual for B tokens.
    All 4 dispatches use register-level weight sharing for shared weights.

    When _residual is None, called from vanilla decoder — do vanilla MoE.
    """
    if _residual is None:
        # Prefill / large-batch MoE path — fuse gate+up into single gather_qmm
        x = attn_out_3d
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        # Fused gate+up: one gather_qmm instead of two
        if hasattr(self.switch_mlp, '_fused_w_gu'):
            y = _fused_switch_glu(self.switch_mlp, x, inds)
        else:
            y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        # Fused shared expert: one matmul instead of two for gate+up
        if hasattr(self, '_shared_w_gu'):
            shared_y = _fused_shared_expert(self, x)
        else:
            shared_y = self.shared_expert(x)
            shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        return y + shared_y

    B_dim = attn_out_3d.shape[0]

    K = self._oproj_M
    K_attn = self._oproj_K_attn
    n_active = self.top_k
    E = self._oproj_n_experts

    attn_out = attn_out_3d.reshape(B_dim, K_attn).astype(COMPUTE_DTYPE)
    residual = _residual.reshape(B_dim, K).astype(COMPUTE_DTYPE)

    # ── Dispatch 1: batched o_proj + gate GEMVs ──
    h_scaled, h_out, x2_partials, gate_part_a, gate_part_b = \
        batched_oproj_gate_gemv(
            self._oproj_w, self._oproj_s, self._oproj_b,
            attn_out, residual, self._oproj_rms_weight,
            self._oproj_M1, self._oproj_W_fused,
            M=K, K_attn=K_attn, batch_size=B_dim,
            n_experts=E, gate_bm=self._oproj_gate_bm,
            K_hidden=self._oproj_K_hidden,
        )

    n_oproj_tg = (K + 31) // 32
    N_INTER = self.switch_mlp._fused_n_inter
    SHARED_INTER = self._shared_inter

    # ── Dispatch 2: batched softmax + topk + SwiGLU ──
    y_routed, y_shared, out_inds, norm_scores, gate_raw = \
        batched_softmax_topk_swiglu_8bit(
            self.switch_mlp._fused_w_gu, self.switch_mlp._fused_s_gu,
            self.switch_mlp._fused_b_gu,
            self._shared_w_gu, self._shared_s_gu, self._shared_b_gu,
            self._seg_w, self._seg_s, self._seg_b,
            h_scaled, gate_part_a, gate_part_b, x2_partials,
            n_inter=N_INTER, k_hidden=K, batch_size=B_dim,
            n_active=n_active, n_oproj_tg=n_oproj_tg,
            n_experts=E, shared_inter=SHARED_INTER,
        )

    # ── Dispatch 3: batched merged down_proj ──
    d_routed, d_shared = batched_merged_down_proj_8bit(
        self._down_w, self._down_s, self._down_b,
        self._shared_down_w, self._shared_down_s, self._shared_down_b,
        y_routed, y_shared.reshape(B_dim * SHARED_INTER), out_inds,
        k_out=K, n_in=self._down_N, batch_size=B_dim,
        n_active=n_active, shared_n_in=SHARED_INTER,
    )

    # ── Dispatch 4: batched epilogue ──
    Y = batched_moe_epilogue(
        d_routed, d_shared, norm_scores,
        h_out, gate_raw,
        k_val=K, batch_size=B_dim, n_active=n_active,
    )

    return Y.reshape(B_dim, 1, K).astype(attn_out_3d.dtype)
