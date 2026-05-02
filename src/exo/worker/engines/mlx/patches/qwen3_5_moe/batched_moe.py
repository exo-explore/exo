"""Batched oproj MoE with 4 custom Metal kernel dispatches.

Fuses o_proj + RMSNorm + gate GEMV + softmax + topk + SwiGLU + down_proj + epilogue
into 4 dispatches with register-level weight sharing for the shared expert.
Falls back to vanilla MoE when called without _residual (from vanilla decoder path).
"""

import mlx.core as mx

from .kernels.batched_merged_down_proj_8bit import batched_merged_down_proj_8bit
from .kernels.batched_oproj_gate_gemv_8bit import batched_oproj_gate_gemv
from .kernels.batched_softmax_topk_swiglu_8bit import batched_softmax_topk_swiglu_8bit
from .kernels.batched_moe_epilogue import batched_moe_epilogue
from .kernels.shared_swiglu_with_routing_8bit import shared_swiglu_with_routing_8bit
from .kernels.batched_merged_swiglu_8bit import batched_merged_routed_swiglu_with_seg_8bit


def _batched_oproj_moe_call(self, attn_out_3d, _residual=None):
    """Batched MoE with full oproj fusion (4 custom dispatches).

    Receives raw attention output (pre-o_proj) and residual for B tokens.
    All 4 dispatches use register-level weight sharing for shared weights.

    When _residual is None, called from vanilla decoder — do vanilla MoE.
    """
    if _residual is None:
        # Vanilla MoE path (called from vanilla decoder for B>8 or S>1)
        x = attn_out_3d
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        return y + shared_y

    B_dim = attn_out_3d.shape[0]

    K = self._oproj_M
    K_attn = self._oproj_K_attn
    n_active = self.top_k
    E = self._oproj_n_experts

    attn_out = attn_out_3d.reshape(B_dim, K_attn).astype(mx.bfloat16)
    residual = _residual.reshape(B_dim, K).astype(mx.bfloat16)

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


def _batched_swiglu_down_moe_call_with_epilogue(self, x, _residual=None):
    """MoE call for the fused_attn_batched_moe mode.

    Receives post-attn_layernorm hidden (with residual already inside attn path
    normalization). Five dispatches:
      D1: vanilla self.gate(x) — router scores (E)
      D2: shared_swiglu_with_routing_8bit (heterogeneous-grid):
          - Z=0 (multi-TG): shared expert SwiGLU with register-sharing across B
          - Z=1 (single-TG): softmax → topk argpartition → take_along_axis → normalize
      D3: batched_merged_routed_swiglu_with_seg_8bit — routed-only SwiGLU + seg phase
          (shared_expert_gate GEMV) emitting gate_raw[B]
      D4: batched_merged_down_proj_8bit — merged down for routed + shared
      D5: batched_moe_epilogue — weighted sum + sigmoid(gate_raw)*d_shared + residual

    Returns the FINAL layer output with _residual already added inside D5.
    Designed to be installed via _fused_decoder_call:
        h = x + attn(norm(x))
        out = mlp(norm(h), _residual=h)
    """
    B_dim = x.shape[0]
    S = x.shape[1]
    K = x.shape[2]

    # D1: vanilla gate matmul (router scores)
    gate_scores = self.gate(x)  # (B, S, E)
    k = self.top_k

    if S == 1 and 1 <= B_dim <= 8:
        n_active = k
        switch_mlp = self.switch_mlp
        shared_expert = self.shared_expert
        N_INTER = switch_mlp.gate_proj["weight"].shape[-2]
        SHARED_INTER = shared_expert.gate_proj.weight.shape[0]
        E = gate_scores.shape[-1]

        x_flat = x.reshape(B_dim, K)

        # D2: shared SwiGLU || (softmax+topk+take+normalize) in one heterogeneous dispatch
        y_shared, inds_2d, norm_scores = shared_swiglu_with_routing_8bit(
            self._shared_w_gu, self._shared_s_gu, self._shared_b_gu,
            gate_scores.reshape(B_dim, E).astype(mx.float32),
            x_flat,
            k_hidden=K, batch_size=B_dim, n_active=n_active, n_experts=E,
            shared_inter=SHARED_INTER, norm_topk_prob=self.norm_topk_prob,
        )
        inds_flat = inds_2d.reshape(B_dim * n_active)

        # D3: routed-only SwiGLU + seg (shared phase no-op since done in D2)
        y_routed, gate_raw = batched_merged_routed_swiglu_with_seg_8bit(
            switch_mlp._fused_w_gu, switch_mlp._fused_s_gu, switch_mlp._fused_b_gu,
            self._shared_w_gu, self._shared_s_gu, self._shared_b_gu,
            x_flat, inds_flat,
            self._seg_w, self._seg_s, self._seg_b,
            n_inter=N_INTER, k_hidden=K, batch_size=B_dim,
            n_active=n_active, shared_inter=SHARED_INTER,
        )

        # D4: merged down_proj (routed + shared)
        K_OUT = self._down_K
        N_IN = self._down_N
        d_routed, d_shared = batched_merged_down_proj_8bit(
            self._down_w, self._down_s, self._down_b,
            self._shared_down_w, self._shared_down_s, self._shared_down_b,
            y_routed, y_shared.reshape(B_dim * SHARED_INTER), inds_flat,
            k_out=K_OUT, n_in=N_IN, batch_size=B_dim,
            n_active=n_active, shared_n_in=SHARED_INTER,
        )

        # Residual baked into the epilogue. _fused_decoder_call always passes one;
        # zeros guard handles the no-residual call shape (shouldn't happen here).
        #
        # Under TP, ShardedMoE wraps this call and does mx.distributed.all_sum
        # on our return value. Each rank produces a partial down_proj output
        # plus the SAME replicated residual h; without scaling, all_sum would
        # produce full_routed + full_shared + N*h. Divide H by N so it sums
        # back to exactly h. ShardedMoE stashes sharding_group on us before
        # calling; defaults to N=1 (single-mini, no scaling) when unset.
        N = 1
        sg = getattr(self, "sharding_group", None)
        if sg is not None:
            N = sg.size()
        if _residual is None:
            H = mx.zeros((B_dim, K_OUT), dtype=x.dtype)
        else:
            H = _residual.reshape(B_dim, K_OUT)
            if N > 1:
                H = H / N

        # D5: fused epilogue (weighted sum + sigmoid(gate_raw)*shared + residual)
        Y = batched_moe_epilogue(
            d_routed.reshape(B_dim, n_active, K_OUT),
            d_shared.reshape(B_dim, K_OUT),
            norm_scores.reshape(B_dim, n_active),
            H,
            gate_raw,
            k_val=K_OUT, batch_size=B_dim, n_active=n_active,
        )
        return Y.reshape(B_dim, S, K_OUT).astype(x.dtype)

    # Vanilla fallback (S>1 or B>8) — compute routing inline; D2 path not taken.
    gates = mx.softmax(gate_scores, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    y = self.switch_mlp(x, inds)
    y = (y * scores[..., None]).sum(axis=-2)
    shared_y = self.shared_expert(x)
    shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
    out = y + shared_y
    if _residual is not None:
        out = out + _residual
    return out
