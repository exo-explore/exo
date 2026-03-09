"""Weight preparation and patch orchestration for Qwen3.5 oproj fusion.

Adapted from mlx_bench/model_patches/qwen/common.py.
"""

import mlx.core as mx
import mlx.nn as nn
from loguru import logger
from mlx_lm.models.qwen3_5 import DecoderLayer
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock


def ceil_div(a, b):
    return (a + b - 1) // b


def _patch_swiglu_weights(moe):
    """Stack gate+up weights for fused 8-bit SwiGLU kernel."""
    gate_proj = moe.switch_mlp.gate_proj
    up_proj = moe.switch_mlp.up_proj

    moe.switch_mlp._fused_w_gu = mx.concatenate(
        [gate_proj.weight, up_proj.weight], axis=1)
    moe.switch_mlp._fused_s_gu = mx.concatenate(
        [gate_proj.scales, up_proj.scales], axis=1)
    moe.switch_mlp._fused_b_gu = mx.concatenate(
        [gate_proj.biases, up_proj.biases], axis=1)
    moe.switch_mlp._fused_n_inter = gate_proj.output_dims
    moe.switch_mlp._fused_k_hidden = gate_proj.input_dims
    moe.switch_mlp._fused_group_size = gate_proj.group_size

    mx.eval(moe.switch_mlp._fused_w_gu,
            moe.switch_mlp._fused_s_gu,
            moe.switch_mlp._fused_b_gu)


def _patch_shared_expert(moe):
    """Prepare shared expert quantized weights for fused 8-bit path."""
    shared = moe.shared_expert
    gp = shared.gate_proj
    up = shared.up_proj
    dp = shared.down_proj

    moe._shared_w_gu = mx.concatenate([gp.weight, up.weight], axis=0)
    moe._shared_s_gu = mx.concatenate([gp.scales, up.scales], axis=0)
    moe._shared_b_gu = mx.concatenate([gp.biases, up.biases], axis=0)

    moe._shared_down_w = dp.weight
    moe._shared_down_s = dp.scales
    moe._shared_down_b = dp.biases

    moe._shared_inter = gp.weight.shape[0]
    moe._shared_gs = gp.group_size

    mx.eval(moe._shared_w_gu, moe._shared_s_gu, moe._shared_b_gu,
            moe._shared_down_w, moe._shared_down_s, moe._shared_down_b)


def _patch_down_proj(moe):
    """Extract down_proj weights for merged 8-bit kernel dispatch."""
    dp = moe.switch_mlp.down_proj
    moe._down_w = dp.weight
    moe._down_s = dp.scales
    moe._down_b = dp.biases
    moe._down_K = dp.output_dims
    moe._down_N = dp.input_dims
    moe._down_gs = dp.group_size
    mx.eval(moe._down_w, moe._down_s, moe._down_b)


def _patch_oproj_gate_rms(layer, gate_bm=8):
    """Precompute M1/W_fused for fused o_proj + gate GEMV (oproj 4-dispatch mode).

    Gate decomposition:
      gate_score[e] = W_gate[e,:] @ rms_norm(h)
      where h = residual + W_oproj @ attn_out
      rms_norm(h) = h * w_rms * inv_rms

    Expanding:
      gate_score[e] = (W_fused @ residual + M1 @ attn_out) * inv_rms

    Precomputed offline (per layer, stored on moe block):
      W_fused = dequant(W_gate) · diag(w_rms)    — (E, K) bf16
      M1 = W_fused @ dequant(W_oproj)            — (E, K_attn) bf16
    """
    moe = layer.mlp

    if layer.is_linear:
        oproj = layer.linear_attn.out_proj
    else:
        oproj = layer.self_attn.o_proj

    # Dequantize gate and o_proj (temporary, for M1 computation)
    gate = moe.gate
    W_gate_f32 = mx.dequantize(
        gate.weight, gate.scales, gate.biases,
        group_size=gate.group_size, bits=gate.bits,
    ).astype(mx.float32)

    W_oproj_f32 = mx.dequantize(
        oproj.weight, oproj.scales, oproj.biases,
        group_size=oproj.group_size, bits=oproj.bits,
    ).astype(mx.float32)
    mx.eval(W_gate_f32, W_oproj_f32)

    rms_weight = layer.post_attention_layernorm.weight.astype(mx.bfloat16)

    w_rms_f32 = rms_weight.astype(mx.float32)
    W_fused = (W_gate_f32 * w_rms_f32).astype(mx.bfloat16)
    mx.eval(W_fused)
    del W_gate_f32

    M1 = (W_fused.astype(mx.float32) @ W_oproj_f32).astype(mx.bfloat16)
    mx.eval(M1)
    del W_oproj_f32

    moe._oproj_M1 = M1
    moe._oproj_W_fused = W_fused
    moe._oproj_rms_weight = rms_weight

    moe._oproj_w = oproj.weight
    moe._oproj_s = oproj.scales
    moe._oproj_b = oproj.biases
    moe._oproj_K_attn = oproj.weight.shape[1] * 4

    seg = moe.shared_expert_gate
    moe._seg_w = seg.weight
    moe._seg_s = seg.scales
    moe._seg_b = seg.biases

    M = oproj.weight.shape[0]
    K_hidden = W_fused.shape[1]
    n_experts = W_fused.shape[0]
    moe._oproj_M = M
    moe._oproj_K_hidden = K_hidden
    moe._oproj_n_experts = n_experts
    moe._oproj_n_tg = ceil_div(M, 32)
    moe._oproj_gate_bm = gate_bm

    mx.eval(moe._oproj_rms_weight)


def apply_oproj_fused_patches(layers, gate_bm=8, free_originals=False):
    """Apply oproj-mode fused MoE patches (4-dispatch) to all layers.

    1. Prepare SwiGLU weights (_patch_swiglu_weights)
    2. Prepare shared expert 8-bit weights (_patch_shared_expert)
    3. Prepare down_proj weights (_patch_down_proj)
    4. Precompute M1/W_fused + store o_proj/gate weights (_patch_oproj_gate_rms)
    5. Replace __call__ methods for decoder + MoE + attention
    """
    from .moe import _oproj_moe_call
    from .decoder import (
        _oproj_decoder_call,
        _pre_oproj_attention_call,
        _pre_oproj_qwen35_linear_attn_call,
    )
    from mlx_lm.models.qwen3_next import Qwen3NextAttention
    from mlx_lm.models.qwen3_5 import GatedDeltaNet

    n_patched = 0
    for li, layer in enumerate(layers):
        moe = layer.mlp
        if isinstance(moe, Qwen3NextSparseMoeBlock):
            _patch_swiglu_weights(moe)
            _patch_shared_expert(moe)
            _patch_down_proj(moe)
            _patch_oproj_gate_rms(layer, gate_bm=gate_bm)

            if free_originals:
                for attr in ('weight', 'scales', 'biases'):
                    for proj in (moe.switch_mlp.gate_proj,
                                 moe.switch_mlp.up_proj):
                        try:
                            delattr(proj, attr)
                        except AttributeError:
                            pass

            n_patched += 1
            if (li + 1) % 10 == 0 or li == 0:
                logger.info(f"  Patched layer {li+1}/{len(layers)} (oproj mode)")

    Qwen3NextAttention.__call__ = _pre_oproj_attention_call
    GatedDeltaNet.__call__ = _pre_oproj_qwen35_linear_attn_call
    Qwen3NextSparseMoeBlock.__call__ = _oproj_moe_call
    DecoderLayer.__call__ = _oproj_decoder_call
    logger.info(f"  Patched {n_patched} MoE blocks (oproj mode, 4 dispatches)")
