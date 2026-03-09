"""MoE __call__ variants for Qwen3.5.

Two modes:
  _fused_moe_call: gate→SwiGLU→down_proj→epilogue (~15 dispatches, fused to ~4+MLX)
  _oproj_moe_call: o_proj+gate→SwiGLU(w/softmax prologue)→down_proj→epilogue (4 dispatches)

Kernels used:
  merged_routed_shared_swiglu_8bit.py    — 8-bit gate+up+SwiGLU (_fused_moe_call)
  merged_down_proj_8bit.py               — 8-bit down_proj (both modes)
  fused_moe_epilogue_qwen.py             — weighted sum + sigmoid gate + residual (both modes)
  custom_oproj_gate_gemv_8bit.py         — 8-bit o_proj + bf16 gate GEMVs (_oproj_moe_call)
  oproj_softmax_topk_swiglu_8bit.py      — SwiGLU with softmax prologue (_oproj_moe_call)

Adapted from mlx_bench/model_patches/qwen/moe.py.
"""

import mlx.core as mx

from .kernels.merged_routed_shared_swiglu_8bit import fused_merged_gate_up_swiglu_8bit
from .kernels.merged_down_proj_8bit import fused_merged_down_proj_8bit
from .kernels.fused_moe_epilogue_qwen import fused_moe_epilogue_qwen
from .kernels.custom_oproj_gate_gemv_8bit import fused_custom_oproj_8bit
from .kernels.oproj_softmax_topk_swiglu_8bit import fused_oproj_softmax_topk_swiglu_8bit


def _vanilla_moe_call(self, x):
    """Original Qwen3NextSparseMoeBlock.__call__ (for prefill fallback)."""
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


def _fused_moe_call(self, x, _residual=None):
    """Qwen3.5 MoE with fused kernels (4 custom dispatches).

    Falls back to vanilla for prefill (seq_len > 1).

    Args:
        x: (B, S, K) bf16 — post-layernorm hidden state
        _residual: (B, S, K) bf16 — pre-layernorm hidden state for residual add.
                   If None, epilogue skips residual (returns MoE output only).
    """
    # Fused kernels are decode-only (seq_len=1). Fall back for prefill.
    if x.shape[-2] > 1:
        # Try vanilla switch_mlp path if weights still exist
        has_switch_weights = hasattr(self.switch_mlp, 'gate_proj') and \
            hasattr(self.switch_mlp.gate_proj, 'weight')
        if has_switch_weights:
            out = _vanilla_moe_call(self, x)
        else:
            # Weights were freed — process tokens one by one through fused path
            outs = []
            for t in range(x.shape[-2]):
                xt = x[:, t:t+1, :]
                res_t = _residual[:, t:t+1, :] if _residual is not None else None
                outs.append(_fused_moe_call(self, xt, _residual=res_t))
            return mx.concatenate(outs, axis=1)
        if _residual is not None:
            out = out + _residual
        return out

    # ── Gate routing (vanilla MLX ops) ──
    gates = self.gate(x)
    gates = mx.softmax(gates, axis=-1, precise=True)

    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)

    x_flat = x.reshape(-1).astype(mx.bfloat16)  # (K,)
    inds_flat = inds.reshape(-1).astype(mx.uint32)
    scores_flat = scores.reshape(-1).astype(mx.float32)

    # ── Dispatch 1: Merged gate+up+SwiGLU (8-bit routed + 8-bit shared) ──
    y_routed, y_shared = fused_merged_gate_up_swiglu_8bit(
        self.switch_mlp._fused_w_gu,
        self.switch_mlp._fused_s_gu,
        self.switch_mlp._fused_b_gu,
        self._shared_w_gu,
        self._shared_s_gu,
        self._shared_b_gu,
        x_flat,
        inds_flat,
        self.switch_mlp._fused_n_inter,
        self.switch_mlp._fused_k_hidden,
        group_size=self.switch_mlp._fused_group_size,
        shared_inter=self._shared_inter,
    )

    # ── Dispatch 2: Merged down_proj (8-bit routed + 8-bit shared) ──
    y_down_routed, y_down_shared = fused_merged_down_proj_8bit(
        self._down_w,
        self._down_s,
        self._down_b,
        self._shared_down_w,
        self._shared_down_s,
        self._shared_down_b,
        y_routed,
        y_shared,
        inds_flat,
        self._down_K,
        self._down_N,
        group_size=self._down_gs,
        shared_n_in=self._shared_inter,
    )

    # ── Dispatch 3: Shared expert gate (small GEMV) ──
    shared_gate_out = self.shared_expert_gate(x.reshape(1, 1, -1))
    shared_gate_val = mx.sigmoid(shared_gate_out.reshape(()))

    # ── Dispatch 4: Fused epilogue ──
    if _residual is not None:
        h_flat = _residual.reshape(-1).astype(mx.bfloat16)
    else:
        h_flat = mx.zeros((self.switch_mlp._fused_k_hidden,), dtype=mx.bfloat16)

    y = fused_moe_epilogue_qwen(
        y_down_routed,    # (n_active, K) f32
        y_down_shared,    # (K,) f32
        scores_flat,      # (n_active,) f32
        h_flat,           # (K,) bf16
        shared_gate_val,  # scalar f32
        self._down_K,     # K
    )
    return y.reshape(1, 1, -1)


def _oproj_moe_call(self, attn_out_3d, _residual=None):
    """Qwen3.5 MoE with fused o_proj + gate GEMVs (4 custom dispatches).

    Receives raw attention output (pre-o_proj) and residual.
    Fuses o_proj, RMSNorm, gate softmax, top-k, score norm, shared_expert_gate,
    SwiGLU, down_proj, and epilogue into 4 dispatches.

    Dispatch 1: 8-bit o_proj + bf16 M1/W_fused GEMVs → h_scaled, h_out, x2_partials,
                gate_part_a, gate_part_b
    Dispatch 2: SwiGLU with softmax prologue (inv_rms + softmax + top-k + score norm
                + shared_expert_gate in TG(0,0,0)) → y_routed, y_shared, inds, scores,
                gate_raw
    Dispatch 3: Merged down_proj (unchanged)
    Dispatch 4: Fused epilogue with sigmoid(gate_raw)

    Args:
        attn_out_3d: (1, 1, K_attn) bf16 — raw attention output (pre-o_proj)
        _residual: (1, 1, K) bf16 — input residual (before attention)
    """
    # Prefill fallback (S > 1): restore o_proj + vanilla MoE
    if attn_out_3d.shape[-2] > 1:
        from .decoder import _parent_layer_map
        parent = _parent_layer_map[id(self)]
        if parent.is_linear:
            oproj_mod = parent.linear_attn.out_proj
        else:
            oproj_mod = parent.self_attn.o_proj
        projected = oproj_mod(attn_out_3d)
        h = _residual + projected
        out = _vanilla_moe_call(self, parent.post_attention_layernorm(h))
        return out + h

    attn_out = attn_out_3d.reshape(-1).astype(mx.bfloat16)  # (K_attn,)
    residual = _residual.reshape(-1).astype(mx.bfloat16)     # (K,)

    K = self._oproj_M
    K_attn = self._oproj_K_attn

    # ── Dispatch 1: fused o_proj + M1 GEMV + W_fused GEMV ──
    h_scaled, h_out, x2_partials, gate_part_a, gate_part_b = \
        fused_custom_oproj_8bit(
            self._oproj_w, self._oproj_s, self._oproj_b,
            attn_out, residual, self._oproj_rms_weight,
            self._oproj_M1, self._oproj_W_fused,
            M=K, K_attn=K_attn, K_hidden=self._oproj_K_hidden,
            n_experts=self._oproj_n_experts,
            gate_bm=self._oproj_gate_bm,
        )

    # ── Dispatch 2: SwiGLU with softmax prologue ──
    y_routed, y_shared, inds, scores, gate_raw = \
        fused_oproj_softmax_topk_swiglu_8bit(
            self.switch_mlp._fused_w_gu,
            self.switch_mlp._fused_s_gu,
            self.switch_mlp._fused_b_gu,
            self._shared_w_gu,
            self._shared_s_gu,
            self._shared_b_gu,
            h_scaled,
            gate_part_a,
            gate_part_b,
            x2_partials,
            self._seg_w,
            self._seg_s,
            self._seg_b,
            n_inter=self.switch_mlp._fused_n_inter,
            k_hidden=K,
            n_experts=self._oproj_n_experts,
            top_k=self.top_k,
            n_oproj_tg=self._oproj_n_tg,
            group_size=self.switch_mlp._fused_group_size,
            shared_inter=self._shared_inter,
            norm_topk=self.norm_topk_prob,
        )

    # ── Dispatch 3: Merged down_proj (8-bit routed + 8-bit shared) ──
    y_down_routed, y_down_shared = fused_merged_down_proj_8bit(
        self._down_w,
        self._down_s,
        self._down_b,
        self._shared_down_w,
        self._shared_down_s,
        self._shared_down_b,
        y_routed,
        y_shared,
        inds,
        self._down_K,
        self._down_N,
        group_size=self._down_gs,
        shared_n_in=self._shared_inter,
    )

    # ── Dispatch 4: Fused epilogue with sigmoid(gate_raw) ──
    y = fused_moe_epilogue_qwen(
        y_down_routed,    # (n_active, K) f32
        y_down_shared,    # (K,) f32
        scores,           # (n_active,) f32 — norm_topk_prob from prologue
        h_out,            # (K,) bf16 — post-o_proj hidden for residual add
        gate_raw,         # scalar f32 — raw dot product, sigmoid fused
        K,
        fuse_sigmoid=True,
    )
    return y.reshape(1, 1, -1)
