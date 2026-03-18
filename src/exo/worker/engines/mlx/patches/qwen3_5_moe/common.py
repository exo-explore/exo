"""Common weight preparation functions for Qwen3.5 fused kernel patches.

Functions:
  ceil_div               — integer ceiling division
  _patch_swiglu_weights  — stack gate+up weights for fused SwiGLU kernel
  _patch_down_proj       — extract down_proj weights for merged kernel dispatch
  _patch_shared_expert   — prepare shared expert weights (8-bit)
  dequantize_shared_expert — convert shared expert from 8-bit to bf16
  _patch_oproj_gate_rms  — precompute M1/W_fused for fused o_proj + gate GEMV
  _patch_gdn_proj_weights — merge GDN projection weights for fused GEMV
  _patch_gqa_proj_weights — merge GQA q/k/v weights with q_proj permutation
  make_qwen_random_cache — create pre-filled cache for testing
  build_model            — build Qwen3.5 MoE layers with 8-bit quantization
"""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5 import (
    DecoderLayer,
    TextModelArgs,
)
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock


def ceil_div(a, b):
    return (a + b - 1) // b


def _patch_swiglu_weights(moe):
    """Stack gate+up weights for fused 8-bit SwiGLU kernel.

    Creates concatenated (E, 2*N_INTER, K/4) weight, (E, 2*N_INTER, K/gs) scales/biases
    from the separate gate_proj and up_proj QuantizedSwitchLinear layers.
    """
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
    """Prepare shared expert quantized weights for fused 8-bit path.

    Stacks shared gate+up quantized weights (weight, scales, biases).
    Stores down_proj quantized weights separately.
    Shared expert stays in 8-bit — same as vanilla MLX dispatch.
    """
    shared = moe.shared_expert
    gp = shared.gate_proj
    up = shared.up_proj
    dp = shared.down_proj

    # Gate+up stacked: (2*SHARED_INTER, K/4) uint32, (2*SHARED_INTER, K/gs) bf16
    moe._shared_w_gu = mx.concatenate([gp.weight, up.weight], axis=0)
    moe._shared_s_gu = mx.concatenate([gp.scales, up.scales], axis=0)
    moe._shared_b_gu = mx.concatenate([gp.biases, up.biases], axis=0)

    # Down_proj: (K, SHARED_INTER/4) uint32, (K, SHARED_INTER/gs) bf16
    moe._shared_down_w = dp.weight
    moe._shared_down_s = dp.scales
    moe._shared_down_b = dp.biases

    # QuantizedLinear: weight is (out_features, in_features/pack_factor) uint32
    # For 8-bit: pack_factor = 4, so in_features = weight.shape[1] * 4
    moe._shared_inter = gp.weight.shape[0]  # SHARED_INTER (= out_features)
    moe._shared_gs = gp.group_size           # gs (64)

    mx.eval(moe._shared_w_gu, moe._shared_s_gu, moe._shared_b_gu,
            moe._shared_down_w, moe._shared_down_s, moe._shared_down_b)


def _patch_down_proj(moe):
    """Extract down_proj weights for merged 8-bit kernel dispatch."""
    dp = moe.switch_mlp.down_proj
    moe._down_w = dp.weight       # (E, K_OUT, N_IN/4) uint32
    moe._down_s = dp.scales       # (E, K_OUT, N_IN/gs) bf16
    moe._down_b = dp.biases       # (E, K_OUT, N_IN/gs) bf16
    moe._down_K = dp.output_dims  # K = 4096
    moe._down_N = dp.input_dims   # N = 1024
    moe._down_gs = dp.group_size  # gs = 64
    mx.eval(moe._down_w, moe._down_s, moe._down_b)


def dequantize_shared_expert(moe):
    """Convert shared expert from 8-bit QuantizedLinear to bf16 weight wrappers.

    The fused kernels expect bf16 shared expert weights. The real model (and our
    random model) has shared expert quantized to 8-bit. This dequantizes in-place.
    """
    shared = moe.shared_expert
    for proj_name in ["gate_proj", "up_proj", "down_proj"]:
        proj = getattr(shared, proj_name)
        if hasattr(proj, 'scales') and hasattr(proj, 'biases'):
            w_bf16 = mx.dequantize(
                proj.weight, proj.scales, proj.biases,
                group_size=proj.group_size, bits=proj.bits,
            ).astype(mx.bfloat16)
            mx.eval(w_bf16)
            setattr(shared, proj_name, SimpleNamespace(weight=w_bf16))


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

    Also stores o_proj quantized weights and shared_expert_gate weights
    on the moe block for use by Dispatch 1 and Dispatch 2.

    Args:
        layer: DecoderLayer instance
        gate_bm: SGs per gate TG in Dispatch 1 (1,2,4,8)
    """
    moe = layer.mlp

    # ── Get attention output projection (works for both attention types) ──
    if layer.is_linear:
        oproj = layer.linear_attn.out_proj
    else:
        oproj = layer.self_attn.o_proj

    # ── Dequantize gate and o_proj (temporary, for M1 computation) ──
    # Eval incrementally to limit peak memory (E=512: dequant temps are ~140 MB)
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

    # ── RMSNorm weight ──
    rms_weight = layer.post_attention_layernorm.weight.astype(mx.bfloat16)

    # ── W_fused = dequant(W_gate) · diag(w_rms) ──
    w_rms_f32 = rms_weight.astype(mx.float32)
    W_fused = (W_gate_f32 * w_rms_f32).astype(mx.bfloat16)
    mx.eval(W_fused)
    del W_gate_f32  # free ~8 MB (E=512) or ~1 MB (E=64)

    # ── M1 = W_fused @ W_oproj — precomputed in f32, stored bf16 ──
    M1 = (W_fused.astype(mx.float32) @ W_oproj_f32).astype(mx.bfloat16)
    mx.eval(M1)
    del W_oproj_f32  # free ~128 MB

    # Store on moe block
    moe._oproj_M1 = M1              # (E, K_attn) bf16
    moe._oproj_W_fused = W_fused    # (E, K) bf16
    moe._oproj_rms_weight = rms_weight  # (K,) bf16

    # ── O_proj quantized weights (for 8-bit GEMV in Dispatch 1) ──
    moe._oproj_w = oproj.weight      # (K, K_attn/4) uint32
    moe._oproj_s = oproj.scales      # (K, K_attn/gs) bf16
    moe._oproj_b = oproj.biases      # (K, K_attn/gs) bf16
    moe._oproj_K_attn = oproj.weight.shape[1] * 4  # 8192 (8-bit: pack_factor=4)

    # ── Shared expert gate weights (for TG(0,0,0) fusion in Dispatch 2) ──
    seg = moe.shared_expert_gate
    moe._seg_w = seg.weight    # (1, K/4) uint32
    moe._seg_s = seg.scales    # (1, K/gs) bf16
    moe._seg_b = seg.biases    # (1, K/gs) bf16

    # ── Dimensions ──
    M = oproj.weight.shape[0]                    # 4096 (hidden_size)
    K_hidden = W_fused.shape[1]                   # 4096 (same as M for Qwen)
    n_experts = W_fused.shape[0]                  # E
    moe._oproj_M = M
    moe._oproj_K_hidden = K_hidden
    moe._oproj_n_experts = n_experts
    moe._oproj_n_tg = ceil_div(M, 32)            # 128 for M=4096
    moe._oproj_gate_bm = gate_bm

    mx.eval(moe._oproj_rms_weight)



def _patch_gdn_proj_weights(attn):
    """Merge all 4 GDN projection weights into contiguous buffers.

    Concatenates in_proj_qkv/z/b/a weights, scales, biases into single
    contiguous arrays for better memory locality in the fused GEMV kernel.
    Stored on the GatedDeltaNet module as _merged_proj_*.
    """
    W_merged = mx.concatenate([
        attn.in_proj_qkv.weight,
        attn.in_proj_z.weight,
        attn.in_proj_b.weight,
        attn.in_proj_a.weight,
    ], axis=0)
    S_merged = mx.concatenate([
        attn.in_proj_qkv.scales,
        attn.in_proj_z.scales,
        attn.in_proj_b.scales,
        attn.in_proj_a.scales,
    ], axis=0)
    B_merged = mx.concatenate([
        attn.in_proj_qkv.biases,
        attn.in_proj_z.biases,
        attn.in_proj_b.biases,
        attn.in_proj_a.biases,
    ], axis=0)
    attn._merged_proj_w = W_merged
    attn._merged_proj_s = S_merged
    attn._merged_proj_b = B_merged
    attn._merged_proj_dims = (
        attn.in_proj_qkv.weight.shape[0],  # N_QKV = 8192
        attn.in_proj_z.weight.shape[0],     # N_Z = 4096
        attn.in_proj_b.weight.shape[0],     # N_B = 32
        attn.in_proj_a.weight.shape[0],     # N_A = 32
    )
    mx.eval(W_merged, S_merged, B_merged)


def _patch_gqa_proj_weights(attn):
    """Merge GQA q_proj, k_proj, v_proj weights into contiguous buffers.

    q_proj outputs (H_q * 2 * D) = interleaved [queries, gate] per head.
    We permute rows so queries (H_q * D) come first, then gate (H_q * D),
    then k_proj, then v_proj. This gives clean contiguous regions for
    the fused GEMV kernel's TG routing.

    Permutation for q_proj:
        Original row layout: [head0_q[0:D], head0_gate[0:D], head1_q[0:D], ...]
        After permutation:   [head0_q, head1_q, ..., head0_gate, head1_gate, ...]

    Stored on Qwen3NextAttention as _merged_proj_*.
    """
    q = attn.q_proj
    k = attn.k_proj
    v = attn.v_proj

    H_q = attn.num_attention_heads
    D = attn.head_dim

    # Permute q_proj weights: separate queries and gate rows
    # q_proj.weight shape: (H_q * 2 * D, K / pack_factor) for 8-bit
    # Reshape to (H_q, 2*D, ...), split into queries[:, :D, :] and gate[:, D:, :]
    W_q = q.weight.reshape(H_q, 2 * D, -1)
    S_q = q.scales.reshape(H_q, 2 * D, -1)
    B_q = q.biases.reshape(H_q, 2 * D, -1)

    W_queries = W_q[:, :D, :].reshape(H_q * D, -1)
    W_gate = W_q[:, D:, :].reshape(H_q * D, -1)
    S_queries = S_q[:, :D, :].reshape(H_q * D, -1)
    S_gate = S_q[:, D:, :].reshape(H_q * D, -1)
    B_queries = B_q[:, :D, :].reshape(H_q * D, -1)
    B_gate = B_q[:, D:, :].reshape(H_q * D, -1)

    # Merge: [queries, gate, keys, values]
    W_merged = mx.contiguous(mx.concatenate([W_queries, W_gate, k.weight, v.weight], axis=0))
    S_merged = mx.contiguous(mx.concatenate([S_queries, S_gate, k.scales, v.scales], axis=0))
    B_merged = mx.contiguous(mx.concatenate([B_queries, B_gate, k.biases, v.biases], axis=0))

    attn._merged_proj_w = W_merged
    attn._merged_proj_s = S_merged
    attn._merged_proj_b = B_merged
    attn._merged_proj_dims = (
        H_q * D,           # N_Q = 4096 (queries)
        H_q * D,           # N_GATE = 4096 (gate)
        k.weight.shape[0], # N_K = 512
        v.weight.shape[0], # N_V = 512
    )
    mx.eval(W_merged, S_merged, B_merged)

    # Pre-cache constant scalar arrays for kernel dispatch (avoid per-call creation)
    N_Q, N_GATE, N_K, N_V = attn._merged_proj_dims
    N_TOTAL = N_Q + N_GATE + N_K + N_V
    K = q.weight.shape[1] * 4  # 8-bit: pack_factor=4
    attn._kernel_scalars = {
        # Dispatch 1: fused_gqa_projections
        'K': mx.array(K, dtype=mx.int32),
        'N_Q': mx.array(N_Q, dtype=mx.int32),
        'N_GATE': mx.array(N_GATE, dtype=mx.int32),
        'N_K': mx.array(N_K, dtype=mx.int32),
        'N_TOTAL': mx.array(N_TOTAL, dtype=mx.int32),
        'N_Q_TG': mx.array(ceil_div(N_Q, 8), dtype=mx.int32),
        'N_GATE_TG': mx.array(ceil_div(N_GATE, 8), dtype=mx.int32),
        'N_K_TG': mx.array(ceil_div(N_K, 8), dtype=mx.int32),
        # Dispatch 4-5: custom SDPA
        'scale': mx.array(attn.head_dim ** -0.5, dtype=mx.float32),
        'H_Q': mx.array(attn.num_attention_heads, dtype=mx.int32),
        'H_KV': mx.array(attn.num_key_value_heads, dtype=mx.int32),
        'N_blocks': mx.array(128, dtype=mx.int32),
    }
    mx.eval(*attn._kernel_scalars.values())

    # Precompute grid/TG dims for Dispatch 1
    N_V_TG = ceil_div(N_V, 8)
    attn._d1_total_tg = ceil_div(N_Q, 8) + ceil_div(N_GATE, 8) + ceil_div(N_K, 8) + N_V_TG

    # Precompute RoPE inv_freq for fused norm+rope kernel (Dispatch 2)
    # inv_freq[d] = theta^(-d / half_dims) for d in {0, ..., half_dims-1}
    rope_dims = attn.rope.dims           # 64 (partial_rotary_factor * head_dim)
    half_dims = rope_dims // 2            # 32
    theta = attn.rope.base                # 10000000
    d_indices = mx.arange(half_dims, dtype=mx.float32)
    attn._rope_inv_freq = theta ** (-d_indices / half_dims)
    mx.eval(attn._rope_inv_freq)



def make_qwen_random_cache(layer, config, prefill_len):
    """Create a pre-filled cache for a single Qwen3.5 decoder layer.

    GatedDeltaNet layers get ArraysCache(size=2) with fixed-size state:
      cache[0] = conv state: (B, conv_kernel_size-1, conv_dim) bf16
      cache[1] = SSM state:  (B, num_v_heads, head_k_dim, head_v_dim) bf16

    GQA layers get KVCache with prefill_len tokens:
      keys:   (B, n_kv_heads, alloc_len, head_dim) bf16
      values: (B, n_kv_heads, alloc_len, head_dim) bf16
    """
    if layer.is_linear:
        from mlx_lm.models.cache import ArraysCache
        cache = ArraysCache(size=2)
        attn = layer.linear_attn
        cache[0] = mx.random.normal(
            (1, attn.conv_kernel_size - 1, attn.conv_dim)
        ).astype(mx.bfloat16)
        cache[1] = mx.random.normal(
            (1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim)
        ).astype(mx.bfloat16)
        return cache
    else:
        from mlx_lm.models.cache import KVCache
        cache = KVCache()
        n_steps = (prefill_len + KVCache.step - 1) // KVCache.step
        alloc_len = n_steps * KVCache.step
        n_kv = config.num_key_value_heads
        hd = config.head_dim
        cache.keys = mx.random.normal((1, n_kv, alloc_len, hd)).astype(mx.bfloat16)
        cache.values = mx.random.normal((1, n_kv, alloc_len, hd)).astype(mx.bfloat16)
        cache.offset = prefill_len
        return cache


def build_model(n_experts=16, n_layers=1, top_k=4,
                hidden_size=4096, moe_intermediate_size=1024,
                shared_expert_intermediate_size=2048, tp=1,
                n_attn_heads=32, n_kv_heads=2,
                lin_v_heads=64, lin_k_heads=16,
                head_dim=256):
    """Build Qwen3.5 MoE decoder layers with 8-bit gs=64 quantization.

    Matches real mlx-community Qwen3.5 quantization:
      Everything 8-bit gs=64 (gate, experts, shared expert, shared_expert_gate, attention/SSM).
      RMSNorm weights: bf16.

    Default dimensions are for Qwen3.5-397B-A17B. For 35B-A3B, pass:
      n_attn_heads=16, lin_v_heads=32, hidden_size=2048

    Uses qwen3_5.DecoderLayer (same class as real model) with hybrid attention:
      3/4 GatedDeltaNet (SSM-like), 1/4 full attention (full_attention_interval=4).

    TP=2 halves all column-parallel dimensions.

    Returns:
        layers: list of DecoderLayer instances
        config: TextModelArgs
        GROUP_SIZE: int (64)
    """
    GROUP_SIZE = 64
    BITS = 8

    # Apply TP sharding
    moe_inter = moe_intermediate_size // tp
    shared_inter = shared_expert_intermediate_size // tp
    n_attn_heads_tp = n_attn_heads // tp
    n_kv_heads_tp = max(1, n_kv_heads // tp)
    lin_v_heads_tp = lin_v_heads // tp
    lin_k_heads_tp = lin_k_heads // tp

    config = TextModelArgs(
        model_type="qwen3_5_moe",
        hidden_size=hidden_size,
        num_hidden_layers=n_layers,
        intermediate_size=moe_inter,
        num_attention_heads=n_attn_heads_tp,
        num_key_value_heads=n_kv_heads_tp,
        linear_num_value_heads=lin_v_heads_tp,
        linear_num_key_heads=lin_k_heads_tp,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_experts=n_experts,
        num_experts_per_tok=top_k,
        decoder_sparse_step=1,
        shared_expert_intermediate_size=shared_inter,
        moe_intermediate_size=moe_inter,
        norm_topk_prob=True,
        rms_norm_eps=1e-6,
        vocab_size=248320,
        head_dim=head_dim,
        full_attention_interval=4,
        max_position_embeddings=262144,
        rope_theta=10000000,
        partial_rotary_factor=0.25,
        rope_parameters={
            "type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
    )

    tp_str = f" (TP={tp})" if tp > 1 else ""
    print(f"  Config: {n_layers} layer(s), {n_experts} experts, top_k={top_k}, "
          f"hidden={hidden_size}, inter={moe_inter}, shared={shared_inter}{tp_str}")
    print(f"  Quant: {BITS}-bit gs={GROUP_SIZE} (all weights)")

    layers = [DecoderLayer(config, idx) for idx in range(n_layers)]

    for li, layer in enumerate(layers):
        # Cast all attention/SSM params to bf16 before quantizing, matching
        # real safetensors model where all non-quantized params are bf16.
        # nn.quantize only touches nn.Linear; other params (conv1d, dt_bias,
        # norm, A_log) must be cast manually.
        attn_mod = layer.linear_attn if layer.is_linear else layer.self_attn
        for name, mod in attn_mod.named_modules():
            if isinstance(mod, nn.Linear):
                mod.weight = mod.weight.astype(mx.bfloat16)
            elif isinstance(mod, nn.Conv1d):
                mod.weight = mod.weight.astype(mx.bfloat16)
        # Cast leaf parameters (dt_bias, norm.weight, q/k_norm) to bf16
        # A_log stays f32 (matches real model)
        if layer.is_linear:
            gdn = layer.linear_attn
            gdn.dt_bias = gdn.dt_bias.astype(mx.bfloat16)
            gdn.norm.weight = gdn.norm.weight.astype(mx.bfloat16)
        else:
            gqa = layer.self_attn
            gqa.q_norm.weight = gqa.q_norm.weight.astype(mx.bfloat16)
            gqa.k_norm.weight = gqa.k_norm.weight.astype(mx.bfloat16)
        nn.quantize(attn_mod, bits=BITS, group_size=GROUP_SIZE)
        mx.eval(attn_mod.parameters())

        # RMSNorm to bf16 (norms are never quantized)
        layer.input_layernorm.weight = layer.input_layernorm.weight.astype(mx.bfloat16)
        layer.post_attention_layernorm.weight = layer.post_attention_layernorm.weight.astype(mx.bfloat16)
        mx.eval(layer.input_layernorm.weight, layer.post_attention_layernorm.weight)

        # MoE block: quantize everything to 8-bit gs=64
        moe = layer.mlp
        if isinstance(moe, Qwen3NextSparseMoeBlock):
            # Gate: random init (zeros get optimized away), then quantize.
            # nn.quantize on a leaf nn.Linear is a no-op (walks children, finds none).
            # Use QuantizedLinear.from_linear directly.
            moe.gate.weight = (
                mx.random.normal(moe.gate.weight.shape) * 0.01
            ).astype(mx.float32)
            moe.gate = nn.QuantizedLinear.from_linear(
                moe.gate, group_size=GROUP_SIZE, bits=BITS)
            mx.eval(moe.gate.parameters())

            # Routed experts: quantize per-projection to limit peak memory
            nn.quantize(moe.switch_mlp, bits=BITS, group_size=GROUP_SIZE)
            mx.eval(moe.switch_mlp.gate_proj.parameters())
            mx.eval(moe.switch_mlp.up_proj.parameters())
            mx.eval(moe.switch_mlp.down_proj.parameters())

            # Shared expert: quantize to 8-bit (matching real model)
            nn.quantize(moe.shared_expert, bits=BITS, group_size=GROUP_SIZE)
            mx.eval(moe.shared_expert.parameters())

            # shared_expert_gate: quantize to 8-bit gs=64 (leaf nn.Linear fix)
            moe.shared_expert_gate = nn.QuantizedLinear.from_linear(
                moe.shared_expert_gate, group_size=GROUP_SIZE, bits=BITS)
            mx.eval(moe.shared_expert_gate.parameters())

        if (li + 1) % 10 == 0 or li == 0:
            print(f"  Layer {li+1}/{n_layers} ready")

    return layers, config, GROUP_SIZE
