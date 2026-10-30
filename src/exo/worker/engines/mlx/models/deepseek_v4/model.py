# type: ignore
# DeepSeek V4 (Flash / Pro) model definition for MLX, vendored into exo.
#
# Source:   https://github.com/rltakashige/mlx-lm
# Commit:   6a3df6cd6b00a347ee40f12d97a182aaf86ea599 (2026-04-26), the revision exo pinned
# Path:     mlx_lm/models/deepseek_v4.py
# Licence:  MIT (mlx-lm), Copyright (c) 2023 Apple Inc. and contributors
# Vendored: 2026-10-30, when exo moved from that fork to upstream ml-explore/mlx-lm,
#           which has no deepseek_v4 model.
#
# Local modifications:
#   - this header;
#   - the relative imports of mlx-lm modules (".base", ".cache", ".switch_layers")
#     are absolute ("mlx_lm.models.base", ...);
#   - the AutoConfig registration of a stand-in "deepseek_v4" config runs only when
#     transformers has no deepseek_v4 config (transformers >= 5.8 has one; replacing it
#     breaks loading the tokenizer);
#   - fixes that make it match DeepSeek's official reference implementation
#     (inference/model.py in deepseek-ai/DeepSeek-V4-Flash-0731 at 7872f01b), each
#     marked in place:
#       - Indexer: RoPE on the query runs along the sequence axis (it ran along the
#         heads), and in multi-token steps the top-k sees only compressed rows that
#         are complete at each query's position (it could pick future rows);
#       - the reference's FP8 rounding of the non-RoPE KV dims and FP4 rounding (after
#         a Hadamard rotation) of the indexer's query and compressed keys, which the
#         model was trained with (_fp8_kv_sim, _fp4_indexer_sim).
# Everything else below is the file as it was at that commit. exo loads it for
# config model_type "deepseek_v4" through exo.worker.engines.mlx.models.get_model_classes.
# Delete this copy once upstream mlx-lm ships a deepseek_v4 that exo can use.

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs, scaled_dot_product_attention
from mlx_lm.models.cache import BatchRotatingKVCache, RotatingKVCache
from mlx_lm.models.switch_layers import SwitchGLU

# State-key constants used by the Compressor / Indexer when reading and
# writing into a DeepseekV4Cache. The cache keeps one independent set of
# per-key state (buffer_kv, buffer_gate, prev_kv, prev_gate, pool and the
# corresponding per-row length lists) for each of these.
_K_COMP = "compressor"
_K_IDX = "indexer"


# Register a minimal HF config so AutoConfig / AutoTokenizer recognize
# ``deepseek_v4`` until huggingface/transformers#45616 merges. Once that PR
# ships, this registration becomes a no-op (``exist_ok=True``).
try:
    from transformers import AutoConfig, PretrainedConfig

    class _DeepseekV4HFConfig(PretrainedConfig):
        model_type = "deepseek_v4"

        def __init__(self, rope_scaling=None, **kwargs):
            self.rope_scaling = rope_scaling
            super().__init__(**kwargs)

    # exo: register only when transformers has no deepseek_v4 config of its own. Newer
    # transformers ship one, and replacing it breaks AutoConfig / AutoTokenizer.
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if "deepseek_v4" not in CONFIG_MAPPING:
        AutoConfig.register("deepseek_v4", _DeepseekV4HFConfig, exist_ok=True)
except ImportError:
    pass


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1

    # MLA-style attention
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    attention_bias: bool = False
    sliding_window: int = 128
    compress_ratios: List[int] = field(default_factory=list)

    # Compressor / Indexer
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    compress_rope_theta: float = 160000.0

    # MoE
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    num_hash_layers: int = 3
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0

    # Hyper-Connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # MTP (dropped in sanitize)
    num_nextn_predict_layers: int = 1

    # RoPE / YaRN
    max_position_embeddings: int = 1048576
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    rms_norm_eps: float = 1e-6


# --------------------------------------------------------------------------- #
# RoPE (traditional pair-wise, YaRN-aware, supports inverse rotation)         #
# --------------------------------------------------------------------------- #


class DeepseekV4RoPE(nn.Module):
    """Traditional (consecutive-pair) RoPE with optional YaRN frequency scaling
    and an ``inverse=True`` path that applies the conjugate rotation. V4 rotates
    K (which is also V since K==V for the shared head) and un-rotates the
    attention output on the same dims.

    Only the last ``dims`` of the input are rotated; preceding dims pass through.
    """

    def __init__(self, dims: int, base: float, scaling_config: Optional[Dict] = None):
        super().__init__()
        self.dims = dims
        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        rope_type = None
        if scaling_config is not None:
            rope_type = scaling_config.get("type") or scaling_config.get("rope_type")

        if rope_type in ("yarn", "deepseek_yarn"):
            factor = scaling_config["factor"]
            orig = scaling_config["original_max_position_embeddings"]
            beta_fast = scaling_config.get("beta_fast", 32)
            beta_slow = scaling_config.get("beta_slow", 1)

            def correction_dim(num_rotations):
                return (
                    dims
                    * math.log(orig / (num_rotations * 2 * math.pi))
                    / (2 * math.log(base))
                )

            low = max(math.floor(correction_dim(beta_fast)), 0)
            high = min(math.ceil(correction_dim(beta_slow)), dims - 1)
            if low == high:
                high += 0.001

            ramp = (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low)
            smooth = 1 - mx.clip(ramp, 0, 1)
            inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth
        elif rope_type not in (None, "default", "linear"):
            raise ValueError(f"Unsupported DeepSeek-V4 RoPE type {rope_type!r}")

        # Tuple-wrap keeps inv_freq out of the module's parameter tree.
        # mx.fast.rope takes the period (1/inv_freq), not the frequency.
        self._inv_freq = (inv_freq,)
        self._freqs = (1.0 / inv_freq,)

    @property
    def inv_freq(self) -> mx.array:
        return self._inv_freq[0]

    @property
    def freqs(self) -> mx.array:
        return self._freqs[0]

    def __call__(self, x: mx.array, offset=0, inverse: bool = False) -> mx.array:
        """``offset`` may be a Python int (all sequences at the same position)
        or an ``mx.array`` of shape ``[B]`` for per-batch-item positions.
        """
        scale = -1.0 if inverse else 1.0
        # mx.fast.rope is a single fused kernel; rotates only the last
        # ``self.dims`` dims and passes through the rest.
        return mx.fast.rope(
            x,
            self.dims,
            traditional=True,
            base=None,
            scale=scale,
            offset=offset,
            freqs=self.freqs,
        )


# --------------------------------------------------------------------------- #
# Sinkhorn-based mHC (Manifold-constrained Hyper-Connections)                 #
# --------------------------------------------------------------------------- #


@mx.compile
def _hc_split_sinkhorn_ops(
    mixes: mx.array,
    hc_scale: mx.array,
    hc_base: mx.array,
    hc_mult: int,
    iters: int,
    eps: float,
):
    """Compiled-op fallback used when the Metal kernel is unavailable (CPU, or
    odd hc_mult). ``mixes`` has shape ``[..., (2+hc)*hc]``; returns
    ``pre [..., hc]``, ``post [..., hc]``, ``comb [..., hc, hc]``.
    """
    mixes = mixes.astype(mx.float32)
    hc_scale = hc_scale.astype(mx.float32)
    hc_base = hc_base.astype(mx.float32)
    s0, s1, s2 = hc_scale[0], hc_scale[1], hc_scale[2]

    pre = mx.sigmoid(mixes[..., :hc_mult] * s0 + hc_base[:hc_mult]) + eps
    post = 2 * mx.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * s1 + hc_base[hc_mult : 2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult :].reshape(
        *mixes.shape[:-1], hc_mult, hc_mult
    ) * s2 + hc_base[2 * hc_mult :].reshape(hc_mult, hc_mult)
    comb = mx.softmax(comb, axis=-1, precise=True) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(max(iters - 1, 0)):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def _make_hc_split_sinkhorn_kernel():
    # Single Metal kernel doing sigmoid+softmax+N Sinkhorn iters in registers.
    # Unrolls on HC and ITERS as template params, so each layer's HC call is
    # one dispatch instead of ~40 from the compiled-op path.
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint idx = thread_position_in_grid.x;
        constexpr int MIX = (2 + HC) * HC;
        float epsv = static_cast<float>(eps[0]);

        auto mix = mixes + idx * MIX;
        auto pre_out = pre + idx * HC;
        auto post_out = post + idx * HC;
        auto comb_out = comb + idx * HC * HC;

        float pre_scale = static_cast<float>(scale[0]);
        float post_scale = static_cast<float>(scale[1]);
        float comb_scale = static_cast<float>(scale[2]);

        for (int i = 0; i < HC; ++i) {
            float z = static_cast<float>(mix[i]) * pre_scale
                + static_cast<float>(base[i]);
            pre_out[i] = 1.0f / (1.0f + metal::fast::exp(-z)) + epsv;
        }
        for (int i = 0; i < HC; ++i) {
            int off = HC + i;
            float z = static_cast<float>(mix[off]) * post_scale
                + static_cast<float>(base[off]);
            post_out[i] = 2.0f / (1.0f + metal::fast::exp(-z));
        }

        float c[HC * HC];
        for (int i = 0; i < HC; ++i) {
            float row_max = -INFINITY;
            for (int j = 0; j < HC; ++j) {
                int cidx = i * HC + j;
                int off = 2 * HC + cidx;
                float v = static_cast<float>(mix[off]) * comb_scale
                    + static_cast<float>(base[off]);
                c[cidx] = v;
                row_max = metal::max(row_max, v);
            }
            float row_sum = 0.0f;
            for (int j = 0; j < HC; ++j) {
                int cidx = i * HC + j;
                float v = metal::fast::exp(c[cidx] - row_max);
                c[cidx] = v;
                row_sum += v;
            }
            float inv_sum = 1.0f / row_sum;
            for (int j = 0; j < HC; ++j) {
                int cidx = i * HC + j;
                c[cidx] = c[cidx] * inv_sum + epsv;
            }
        }
        for (int j = 0; j < HC; ++j) {
            float col_sum = 0.0f;
            for (int i = 0; i < HC; ++i) {
                col_sum += c[i * HC + j];
            }
            float inv_denom = 1.0f / (col_sum + epsv);
            for (int i = 0; i < HC; ++i) {
                c[i * HC + j] *= inv_denom;
            }
        }
        for (int iter = 1; iter < ITERS; ++iter) {
            for (int i = 0; i < HC; ++i) {
                float row_sum = 0.0f;
                for (int j = 0; j < HC; ++j) {
                    row_sum += c[i * HC + j];
                }
                float inv_denom = 1.0f / (row_sum + epsv);
                for (int j = 0; j < HC; ++j) {
                    c[i * HC + j] *= inv_denom;
                }
            }
            for (int j = 0; j < HC; ++j) {
                float col_sum = 0.0f;
                for (int i = 0; i < HC; ++i) {
                    col_sum += c[i * HC + j];
                }
                float inv_denom = 1.0f / (col_sum + epsv);
                for (int i = 0; i < HC; ++i) {
                    c[i * HC + j] *= inv_denom;
                }
            }
        }
        for (int i = 0; i < HC * HC; ++i) {
            comb_out[i] = c[i];
        }
    """
    return mx.fast.metal_kernel(
        name="deepseek_v4_hc_split_sinkhorn",
        input_names=["mixes", "scale", "base", "eps"],
        output_names=["pre", "post", "comb"],
        source=source,
    )


_hc_split_sinkhorn_kernel = _make_hc_split_sinkhorn_kernel()


def hc_split_sinkhorn(
    mixes: mx.array,
    hc_scale: mx.array,
    hc_base: mx.array,
    hc_mult: int,
    iters: int,
    eps,
):
    """Dispatch to the Metal kernel if available, else the compiled-op fallback.

    ``mixes`` has shape ``[..., (2+hc)*hc]``; returns ``pre [..., hc]``,
    ``post [..., hc]``, ``comb [..., hc, hc]``.
    """
    if _hc_split_sinkhorn_kernel is None:
        eps_val = eps.item() if isinstance(eps, mx.array) else float(eps)
        return _hc_split_sinkhorn_ops(
            mixes, hc_scale, hc_base, hc_mult, iters, eps_val
        )
    eps_arr = eps if isinstance(eps, mx.array) else mx.array([eps], dtype=mx.float32)
    return _hc_split_sinkhorn_kernel(
        inputs=[mixes, hc_scale, hc_base, eps_arr],
        template=[("HC", hc_mult), ("ITERS", iters)],
        grid=(mixes.size // ((2 + hc_mult) * hc_mult), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (*mixes.shape[:-1], hc_mult),
            (*mixes.shape[:-1], hc_mult),
            (*mixes.shape[:-1], hc_mult, hc_mult),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )


@mx.compile
def _hc_expand_ops(
    f_out: mx.array,  # [B, S, D]       input dtype (bf16)
    residual: mx.array,  # [B, S, hc, D]  input dtype
    post: mx.array,  # [B, S, hc]         fp32
    comb: mx.array,  # [B, S, hc, hc]     fp32
):
    """y[b,s,j,d] = post[j] * f_out[d] + sum_i(comb[i,j] * residual[i,d])."""
    y = post[..., None] * f_out[:, :, None, :]
    y = y + mx.matmul(comb.swapaxes(-1, -2), residual.astype(mx.float32))
    return y.astype(f_out.dtype)


def _make_hc_sinkhorn_collapse_kernel():
    """Single-dispatch fused sinkhorn + collapse (from PR 1192).

    Combines what was two dispatches (sinkhorn → emit pre/post/comb, then
    collapse pre·x → bf16) into one Metal kernel:

      Phase 1 — simdgroup 0 runs the branchless sinkhorn on HC=4 lanes,
                writing post (fp32, HC) and comb (fp32, HC*HC) to outputs and
                pre to threadgroup memory.
      Phase 2 — all 256 threads do the bfloat16 collapse against pre,
                streaming bfloat4 reads/writes for D-divisible-by-4.

    HC is hardcoded to 4 via float4 vectorization (matches DSV4 hc_mult=4 for
    both Flash and Pro). For other HC values the kernel returns None and the
    caller falls back to the separate sinkhorn-then-collapse path.
    """
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    source = """
        uint tid  = thread_position_in_threadgroup.x;
        uint row  = threadgroup_position_in_grid.x;
        uint lane = tid % 32;
        uint sg   = tid / 32;

        constexpr int MIX      = (2 + HC) * HC;
        constexpr int BASE_OFF = 2 * HC;

        const device float* mix      = (const device float*)mixes + row * MIX;
        device float*       post_out = (device float*)post + row * HC;
        device float*       comb_out = (device float*)comb + row * HC * HC;

        threadgroup float pre_shared[HC];

        if (sg == 0) {
            const float pre_scale  = scale[0];
            const float post_scale = scale[1];
            const float comb_scale = scale[2];
            const float epsv       = eps[0];

            const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
            const uint  llane  = metal::min(lane, (uint)(HC - 1));

            float pre_z  = mix[llane]      * pre_scale  + base[llane];
            float post_z = mix[HC + llane] * post_scale + base[HC + llane];
            float pre_v  = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + epsv;
            float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));

            if (lane < (uint)HC) {
                pre_shared[lane] = pre_v;
                post_out[lane]   = post_v;
            }

            float4 v = (*(const device float4*)(mix  + BASE_OFF + llane * HC)
                            * comb_scale
                      + *(const device float4*)(base + BASE_OFF + llane * HC))
                     * active;

            float row_max = metal::max(metal::max(v.x, v.y),
                                       metal::max(v.z, v.w));
            float4 e = metal::fast::exp(v - row_max) * active;
            float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + epsv))
                     + epsv * active;

            float4 col_inv = 1.0f / (float4(
                simd_sum(r.x), simd_sum(r.y),
                simd_sum(r.z), simd_sum(r.w)
            ) + epsv);
            r *= col_inv;

            for (int iter = 1; iter < ITERS; ++iter) {
                r *= (1.0f / (r.x + r.y + r.z + r.w + epsv)) * active;
                col_inv = 1.0f / (float4(
                    simd_sum(r.x), simd_sum(r.y),
                    simd_sum(r.z), simd_sum(r.w)
                ) + epsv);
                r *= col_inv;
            }

            if (lane < (uint)HC) {
                *(device float4*)(comb_out + lane * HC) = r;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float p0 = pre_shared[0];
        const float p1 = pre_shared[1];
        const float p2 = pre_shared[2];
        const float p3 = pre_shared[3];

        const device bfloat16_t* x_row  = (const device bfloat16_t*)x_in
                                         + row * (HC * D);
        device bfloat16_t*       out_row = (device bfloat16_t*)collapsed
                                         + row * D;

        using bf4 = vec<bfloat16_t, 4>;
        const device bf4* x_row0 = (const device bf4*)(x_row + 0*D);
        const device bf4* x_row1 = (const device bf4*)(x_row + 1*D);
        const device bf4* x_row2 = (const device bf4*)(x_row + 2*D);
        const device bf4* x_row3 = (const device bf4*)(x_row + 3*D);
        device bf4*       out4   = (device bf4*)out_row;

        constexpr uint D4 = (uint)D / 4;

        for (uint d4 = tid; d4 < D4; d4 += 256) {
            float4 x0 = float4(x_row0[d4]);
            float4 x1 = float4(x_row1[d4]);
            float4 x2 = float4(x_row2[d4]);
            float4 x3 = float4(x_row3[d4]);

            float4 result = fma(float4(p0), x0,
                            fma(float4(p1), x1,
                            fma(float4(p2), x2, float4(p3) * x3)));

            out4[d4] = bf4(result);
        }
    """
    return mx.fast.metal_kernel(
        name="deepseek_v4_hc_sinkhorn_collapse",
        input_names=["mixes", "scale", "base", "eps", "x_in"],
        output_names=["post", "comb", "collapsed"],
        source=source,
    )


_hc_sinkhorn_collapse_kernel = _make_hc_sinkhorn_collapse_kernel()


def hc_sinkhorn_collapse(mixes, scale, base, x, hc_mult, sinkhorn_iters, eps):
    """Fused sinkhorn + collapse. Returns (collapsed_bf16, post_fp32, comb_fp32).

    Falls back to the separate sinkhorn + matmul-collapse path when:
      - The Metal kernel is unavailable (CPU / non-Metal device).
      - hc_mult != 4 (kernel uses float4 vectorization).
      - x dtype isn't bfloat16 (kernel hardcodes bfloat16_t loads/stores).
      - x's last dim D % 4 != 0 (kernel uses bfloat4 streaming).
    """
    B, S, hc, D = x.shape
    use_kernel = (
        _hc_sinkhorn_collapse_kernel is not None
        and hc == 4
        and hc_mult == 4
        and x.dtype == mx.bfloat16
        and D % 4 == 0
    )
    if not use_kernel:
        if not isinstance(eps, mx.array):
            eps_arr = mx.array([eps], dtype=mx.float32)
        else:
            eps_arr = eps
        pre, post, comb = hc_split_sinkhorn(
            mixes, scale, base, hc_mult, sinkhorn_iters, eps_arr
        )
        collapsed = (pre[:, :, None, :] @ x.astype(mx.float32)).squeeze(2).astype(x.dtype)
        return collapsed, post, comb

    eps_arr = eps if isinstance(eps, mx.array) else mx.array([eps], dtype=mx.float32)
    n_rows = mixes.size // ((2 + hc_mult) * hc_mult)
    post, comb, collapsed = _hc_sinkhorn_collapse_kernel(
        inputs=[mixes, scale, base, eps_arr, x],
        template=[("HC", hc_mult), ("ITERS", sinkhorn_iters), ("D", D)],
        grid=(n_rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (*mixes.shape[:-1], hc_mult),
            (*mixes.shape[:-1], hc_mult, hc_mult),
            (*x.shape[:-2], D),
        ],
        output_dtypes=[mx.float32, mx.float32, x.dtype],
    )
    return collapsed, post, comb


class HyperConnection(nn.Module):
    """Per-block mHC: projects an ``[..., hc, D]`` state to ``pre``/``post``/``comb``.

    Parameter layout matches the reference checkpoint naming:
        fn    : ``[mix_hc, hc*D]``   (fp32)
        base  : ``[mix_hc]``         (fp32)
        scale : ``[3]``              (fp32)
    where ``mix_hc = (2 + hc_mult) * hc_mult``.
    """

    def __init__(
        self,
        dim: int,
        hc_mult: int,
        norm_eps: float,
        sinkhorn_iters: int,
        hc_eps: float,
    ):
        super().__init__()
        self.dim = dim
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.hc_eps = hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * dim
        self.fn = mx.zeros((mix_hc, hc_dim), dtype=mx.float32)
        self.base = mx.zeros((mix_hc,), dtype=mx.float32)
        self.scale = mx.zeros((3,), dtype=mx.float32)
        # Cache so the kernel launch doesn't allocate a new mx.array each call.
        self._eps_arr = mx.array([hc_eps], dtype=mx.float32)

    def hc_pre(self, x: mx.array):
        # x: [B, S, hc, D]  ->  (y [B, S, D], post [B, S, hc], comb [B, S, hc, hc])
        B, S, hc, D = x.shape
        # mx.fast.rms_norm is Apple's optimized Metal kernel for the RMS+rsqrt
        # path; replacing it with a hand-rolled compile-fuse regresses prefill
        # because the manual rsqrt(mean(x*x)) chain doesn't dispatch to the
        # same fused kernel as fast.rms_norm.
        xf = mx.fast.rms_norm(
            x.reshape(B, S, hc * D).astype(mx.float32), None, self.norm_eps
        )
        mixes = xf @ self.fn.T
        # Fused sinkhorn + collapse — single Metal dispatch when HC=4 and x is
        # bf16; falls back to separate sinkhorn-then-matmul-collapse otherwise.
        return hc_sinkhorn_collapse(
            mixes, self.scale, self.base, x,
            self.hc_mult, self.sinkhorn_iters, self._eps_arr,
        )

    def hc_post(
        self,
        f_out: mx.array,
        residual: mx.array,
        post: mx.array,
        comb: mx.array,
    ):
        # f_out    [B, S, D]          (input dtype)
        # residual [B, S, hc, D]      (input dtype)
        # post     [B, S, hc]         (fp32, from sinkhorn)
        # comb     [B, S, hc, hc]     (fp32, from sinkhorn)
        return _hc_expand_ops(f_out, residual, post, comb)


class HyperHead(nn.Module):
    """Final head mHC: reduces ``[B, S, hc, D]`` -> ``[B, S, D]`` via sigmoid-
    weighted sum (no Sinkhorn normalization)."""

    def __init__(self, dim: int, hc_mult: int, norm_eps: float, hc_eps: float):
        super().__init__()
        self.dim = dim
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.fn = mx.zeros((hc_mult, hc_mult * dim), dtype=mx.float32)
        self.base = mx.zeros((hc_mult,), dtype=mx.float32)
        self.scale = mx.zeros((1,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        B, S, hc, D = x.shape
        # Use mx.fast.rms_norm rather than a manual rsqrt(mean(x*x)) compile
        # fuse — the optimized Metal kernel is meaningfully faster on prefill.
        xf = mx.fast.rms_norm(
            x.reshape(B, S, hc * D).astype(mx.float32), None, self.norm_eps
        )
        mixes = xf @ self.fn.T
        pre = mx.sigmoid(mixes * self.scale[0] + self.base) + self.hc_eps
        return (pre[:, :, None, :] @ x.astype(mx.float32)).squeeze(2).astype(x.dtype)


# --------------------------------------------------------------------------- #
# DeepseekV4Cache                                                             #
# --------------------------------------------------------------------------- #


def _as_lengths_list(lengths, batch_size: int, default: Optional[int] = None):
    """Normalize ``lengths`` (list, mx.array, or None) into a Python list of
    ints of length ``batch_size``. ``None`` falls back to ``default`` for
    every row, or returns ``None`` if no default is given."""
    if lengths is None:
        if default is None:
            return None
        return [default] * batch_size
    if isinstance(lengths, mx.array):
        lengths = lengths.tolist()
    return [int(x) for x in lengths]


def _filter_lengths(lengths, batch_indices):
    if lengths is None:
        return None
    if isinstance(lengths, mx.array):
        lengths = lengths.tolist()
    if isinstance(batch_indices, mx.array):
        batch_indices = batch_indices.tolist()
    if len(lengths) == 1 and any(idx != 0 for idx in batch_indices):
        lengths = lengths * (max(batch_indices) + 1)
    return [int(lengths[idx]) for idx in batch_indices]


class _CompressorBranch:
    """Per-state-key (compressor or indexer) state held by DeepseekV4Cache.

    Fields:
      buffer_kv, buffer_gate: [B, k, coff*d] carry of unprocessed tokens.
      prev_kv,  prev_gate  : [B, ratio, coff*d] previous full window (overlap only).
      pool                 : [B, n_emitted, head_dim] growing pool of compressed rows.
      buffer_lengths       : per-row int list — number of valid tokens per row in
                             ``buffer_kv``. None means uniform (= buffer_kv.shape[1]).
      pool_lengths         : per-row int list — number of valid pool rows per row.
                             None means uniform (= pool.shape[1]).
    """

    __slots__ = (
        "buffer_kv",
        "buffer_gate",
        "prev_kv",
        "prev_gate",
        "pool",
        "buffer_lengths",
        "pool_lengths",
        "buffer_count",
        "_new_pool_lengths",
    )

    def __init__(self):
        self.buffer_kv = None
        self.buffer_gate = None
        self.prev_kv = None
        self.prev_gate = None
        self.pool = None
        self.buffer_lengths = None
        self.pool_lengths = None
        # In-place fast path: buffer_kv has shape [B, ratio, coff*d] and
        # buffer_count tracks the number of valid leading positions.
        self.buffer_count = 0
        self._new_pool_lengths = None


class DeepseekV4Cache:
    """Cache for a single compressed-attention layer.

    Wraps a sliding-window KV cache (RotatingKVCache or BatchRotatingKVCache
    after ``prepare(left_padding|right_padding)``) and tracks the Compressor /
    Indexer state with per-row length tracking so that variable-length batches
    (right-padded prefill, multi-request decode after extend) stay correct.
    """

    def __init__(self, sliding_window: int):
        self.local = RotatingKVCache(max_size=sliding_window)
        self._branches: Dict[str, _CompressorBranch] = {
            _K_COMP: _CompressorBranch(),
            _K_IDX: _CompressorBranch(),
        }
        # Per-step right-padding info captured by ``prepare`` and consumed
        # by the very next call to ``accumulate_windows``.
        self._pending_lengths: Optional[List[int]] = None

    # ------------------------------------------------------------------ #
    # Pass-through API expected by V4Attention and the generator runtime  #
    # ------------------------------------------------------------------ #

    @property
    def offset(self):
        return self.local.offset

    @property
    def keys(self):
        return self.local.keys

    @keys.setter
    def keys(self, value):
        self.local.keys = value

    @property
    def values(self):
        return self.local.values

    @values.setter
    def values(self, value):
        self.local.values = value

    def update_and_fetch(self, keys, values):
        return self.local.update_and_fetch(keys, values)

    def is_trimmable(self):
        # The compressor pool isn't trimmable in lock-step with the local
        # cache: rotating it would require recomputing pool rows. Disable.
        return False

    def trim(self, n):
        return 0

    def empty(self):
        return self.local.empty()

    def size(self):
        return self.local.size()

    @property
    def nbytes(self):
        total = self.local.nbytes
        for branch in self._branches.values():
            for v in (
                branch.buffer_kv,
                branch.buffer_gate,
                branch.prev_kv,
                branch.prev_gate,
                branch.pool,
            ):
                if v is not None and hasattr(v, "nbytes"):
                    total += v.nbytes
        return total

    @property
    def state(self):
        local_state = None if self.local.empty() else self.local.state
        return (local_state, self._branch_tuple(_K_COMP), self._branch_tuple(_K_IDX))

    @state.setter
    def state(self, value):
        local_state, comp, idx = value
        if local_state is None:
            self.local.keys = None
            self.local.values = None
        else:
            self.local.state = local_state
        self._set_branch_tuple(_K_COMP, comp)
        self._set_branch_tuple(_K_IDX, idx)

    @property
    def meta_state(self):
        return self.local.meta_state

    @meta_state.setter
    def meta_state(self, value):
        self.local.meta_state = value

    def _branch_tuple(self, key):
        b = self._branches[key]
        return (
            b.buffer_kv,
            b.buffer_gate,
            b.prev_kv,
            b.prev_gate,
            b.pool,
            b.buffer_lengths,
            b.pool_lengths,
            b.buffer_count,
        )

    def _set_branch_tuple(self, key, value):
        b = _CompressorBranch()
        if value is not None:
            if len(value) == 7:
                # Backwards compat: old serialized tuples without buffer_count.
                (
                    b.buffer_kv,
                    b.buffer_gate,
                    b.prev_kv,
                    b.prev_gate,
                    b.pool,
                    b.buffer_lengths,
                    b.pool_lengths,
                ) = value
            else:
                (
                    b.buffer_kv,
                    b.buffer_gate,
                    b.prev_kv,
                    b.prev_gate,
                    b.pool,
                    b.buffer_lengths,
                    b.pool_lengths,
                    b.buffer_count,
                ) = value
        self._branches[key] = b

    # ------------------------------------------------------------------ #
    # Variable-length prepare / finalize                                 #
    # ------------------------------------------------------------------ #

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        # Right-padding tokens at the end of each row's prompt must NOT
        # contribute to the compressor pool. Stash the per-row prompt
        # lengths so the next ``accumulate_windows`` call can clip.
        if right_padding is not None and max(right_padding) > 0:
            self._pending_lengths = [int(x) for x in lengths]
        else:
            self._pending_lengths = None

        # Left-padding (cache-extend with shorter prompts joining a longer
        # batch) is handled by swapping the local cache to BatchRotatingKVCache
        # since RotatingKVCache only supports a uniform offset.
        if left_padding is not None or (
            right_padding is not None and max(right_padding) > 0
        ):
            target_batch = (
                len(left_padding) if left_padding is not None
                else (len(lengths) if lengths is not None else None)
            )
            if not isinstance(self.local, BatchRotatingKVCache):
                if (
                    target_batch is not None
                    and self._cache_batch_size(self.local) != target_batch
                    and self.local.keys is None
                ):
                    # Empty cache being prepared for a B-row batch: just allocate
                    # a fresh BatchRotatingKVCache of that batch size. Avoids
                    # _batch_rotating_from's "extract 1 row" path which would
                    # produce a 1-row cache that doesn't fit the prepare call.
                    self.local = BatchRotatingKVCache(
                        self.local.max_size, [0] * target_batch
                    )
                else:
                    self.local = self._batch_rotating_from(self.local)
            self.local.prepare(
                left_padding=left_padding,
                lengths=lengths,
                right_padding=right_padding,
            )

    def finalize(self):
        if hasattr(self.local, "finalize"):
            self.local.finalize()
        self._pending_lengths = None

    # ------------------------------------------------------------------ #
    # Filter / extend / extract / merge for batched generation           #
    # ------------------------------------------------------------------ #

    def filter(self, batch_indices):
        if hasattr(self.local, "filter"):
            self.local.filter(batch_indices)
        elif self.local.keys is not None:
            self.local.keys = self.local.keys[batch_indices]
            self.local.values = self.local.values[batch_indices]
        for branch in self._branches.values():
            for name in ("buffer_kv", "buffer_gate", "prev_kv", "prev_gate", "pool"):
                v = getattr(branch, name)
                if v is not None:
                    setattr(branch, name, v[batch_indices])
            branch.buffer_lengths = _filter_lengths(branch.buffer_lengths, batch_indices)
            branch.pool_lengths = _filter_lengths(branch.pool_lengths, batch_indices)

    def extend(self, other):
        a_batch = self._cache_batch_size(self.local)
        b_batch = self._cache_batch_size(other.local)
        if hasattr(self.local, "extend"):
            other_local = other.local
            if not isinstance(other_local, BatchRotatingKVCache):
                other_local = self._batch_rotating_from(other_local)
            self.local.extend(other_local)
        elif (
            not isinstance(other.local, BatchRotatingKVCache)
            and self.local.offset == other.local.offset
            and self.local._idx == other.local._idx
        ):
            if self.local.keys is not None or other.local.keys is not None:
                self.local.keys = self._concat_optional(self.local.keys, other.local.keys)
                self.local.values = self._concat_optional(self.local.values, other.local.values)
        else:
            self.local = self._batch_rotating_from(self.local)
            other_local = (
                other.local
                if isinstance(other.local, BatchRotatingKVCache)
                else self._batch_rotating_from(other.local)
            )
            self.local.extend(other_local)

        for key, b_self in self._branches.items():
            b_other = other._branches[key]
            for name in ("buffer_kv", "buffer_gate", "prev_kv", "prev_gate", "pool"):
                merged = self._concat_batch_state(
                    getattr(b_self, name),
                    getattr(b_other, name),
                    a_batch,
                    b_batch,
                )
                setattr(b_self, name, merged)
            b_self.buffer_lengths = self._concat_lengths(
                b_self.buffer_lengths,
                b_other.buffer_lengths,
                b_self.buffer_kv,
                b_other.buffer_kv,
                a_batch,
                b_batch,
            )
            b_self.pool_lengths = self._concat_lengths(
                b_self.pool_lengths,
                b_other.pool_lengths,
                b_self.pool,
                b_other.pool,
                a_batch,
                b_batch,
            )

    def extract(self, idx):
        cache = DeepseekV4Cache(self.local.max_size)
        cache.local = (
            self.local.extract(idx)
            if hasattr(self.local, "extract")
            else self._extract_local(self.local, idx)
        )
        for key, src in self._branches.items():
            dst = cache._branches[key]
            for name in ("buffer_kv", "buffer_gate", "prev_kv", "prev_gate", "pool"):
                v = getattr(src, name)
                setattr(dst, name, None if v is None else mx.contiguous(v[idx : idx + 1]))
            for name in ("buffer_lengths", "pool_lengths"):
                lengths = getattr(src, name)
                if lengths is None:
                    setattr(dst, name, None)
                else:
                    if isinstance(lengths, mx.array):
                        lengths = lengths.tolist()
                    setattr(dst, name, [int(lengths[idx])])
        return cache

    @classmethod
    def merge(cls, caches: List["DeepseekV4Cache"]):
        if not caches:
            raise ValueError("Cannot merge empty cache list")
        if not all(c.local.max_size == caches[0].local.max_size for c in caches):
            raise ValueError("DeepseekV4Cache merge requires the same sliding window")
        out = cls(caches[0].local.max_size)
        out.local = cls._merge_local([c.local for c in caches])
        for key in (_K_COMP, _K_IDX):
            dst = out._branches[key]
            for name in ("buffer_kv", "buffer_gate", "prev_kv", "prev_gate", "pool"):
                tensors = [getattr(c._branches[key], name) for c in caches]
                setattr(dst, name, cls._merge_batch_state(tensors))
            dst.buffer_lengths = cls._merge_lengths(
                [c._branches[key].buffer_lengths for c in caches],
                [c._branches[key].buffer_kv for c in caches],
            )
            dst.pool_lengths = cls._merge_lengths(
                [c._branches[key].pool_lengths for c in caches],
                [c._branches[key].pool for c in caches],
            )
        return out

    # ------------------------------------------------------------------ #
    # The two methods Compressor calls during forward                    #
    # ------------------------------------------------------------------ #

    def accumulate_windows(self, kv, gate, key, ratio, start_pos):
        """Append (kv, gate) for this forward step to the carry buffer for
        ``key`` (compressor / indexer) and return the slice that's ready
        to be folded into ratio-sized windows.

        Returns (ready_kv, ready_gate, pool_base) where:
          - ready_kv/ready_gate have shape [B, n_full_windows*ratio, coff*d].
            n_full_windows = max_usable // ratio.
          - pool_base is the raw position of ``ready_*[:, 0]`` in the input
            sequence. Either an int (uniform offset) or an mx.array of shape
            [B] (per-row).

        Side-effects: updates branch.buffer_kv, branch.buffer_gate, and
        records a pending ``_new_pool_lengths`` list to be consumed by
        ``update_pool``.
        """
        branch = self._branches[key]
        buf_kv = branch.buffer_kv
        buf_gate = branch.buffer_gate
        B, L = kv.shape[:2]
        chunk_lengths = self._pending_lengths

        # Hot decode path: uniform offsets, S=1, fixed-size buffer of shape
        # [B, ratio, coff*d] with branch.buffer_count valid leading positions.
        # In-place slot writes (no concat per step) — matches the old slot-
        # based decode code's allocation profile.
        if (
            branch.buffer_lengths is None
            and chunk_lengths is None
            and L == 1
        ):
            coff_d = kv.shape[-1]
            count = branch.buffer_count
            if buf_kv is None or buf_kv.shape[1] != ratio:
                # First decode step OR transitioning from variable-length carry
                # left over from prefill. Pad / allocate to fixed [B, ratio, _].
                prev_n = 0 if buf_kv is None else buf_kv.shape[1]
                fixed_kv = mx.zeros((B, ratio, coff_d), dtype=kv.dtype)
                fixed_gate = mx.zeros((B, ratio, coff_d), dtype=gate.dtype)
                if prev_n:
                    fixed_kv[:, :prev_n] = buf_kv
                    fixed_gate[:, :prev_n] = buf_gate
                buf_kv = fixed_kv
                buf_gate = fixed_gate
                count = prev_n
            buf_kv[:, count, :] = kv[:, 0, :]
            buf_gate[:, count, :] = gate[:, 0, :]
            new_count = count + 1
            branch._new_pool_lengths = None
            if new_count < ratio:
                branch.buffer_kv = buf_kv
                branch.buffer_gate = buf_gate
                branch.buffer_count = new_count
                empty_kv = mx.zeros((B, 0, coff_d), dtype=kv.dtype)
                empty_gate = mx.zeros((B, 0, coff_d), dtype=gate.dtype)
                if isinstance(start_pos, mx.array):
                    pool_base = mx.maximum(start_pos, 0) + 1 - new_count
                else:
                    pool_base = max(0, start_pos) + 1 - new_count
                return empty_kv, empty_gate, pool_base
            # Emit: full buffer becomes ready_kv. Allocate a fresh empty
            # buffer for the next round.
            ready_kv = buf_kv
            ready_gate = buf_gate
            branch.buffer_kv = mx.zeros((B, ratio, coff_d), dtype=kv.dtype)
            branch.buffer_gate = mx.zeros((B, ratio, coff_d), dtype=gate.dtype)
            branch.buffer_count = 0
            if isinstance(start_pos, mx.array):
                pool_base = mx.maximum(start_pos, 0) + 1 - ratio
            else:
                pool_base = max(0, start_pos) + 1 - ratio
            return ready_kv, ready_gate, pool_base

        # Multi-token (prefill / chunked-decode) uniform path: concat carry +
        # new tokens, slice off full windows. Variable-sized buffer carry.
        if branch.buffer_lengths is None and chunk_lengths is None:
            # If we were in the fixed-size in-place state, fold buffer_count
            # back into a variable-sized [:count] slice before the concat.
            if buf_kv is not None and buf_kv.shape[1] == ratio and branch.buffer_count < ratio:
                count = branch.buffer_count
                buf_kv = buf_kv[:, :count] if count else None
                buf_gate = buf_gate[:, :count] if count else None
            if buf_kv is not None and buf_kv.shape[1]:
                kv = mx.concatenate([buf_kv, kv], axis=1)
                gate = mx.concatenate([buf_gate, gate], axis=1)
            usable = (kv.shape[1] // ratio) * ratio
            branch.buffer_kv = kv[:, usable:] if usable < kv.shape[1] else None
            branch.buffer_gate = gate[:, usable:] if usable < gate.shape[1] else None
            branch.buffer_count = 0 if usable >= kv.shape[1] else kv.shape[1] - usable
            branch._new_pool_lengths = None
            buf_len = 0 if buf_kv is None else buf_kv.shape[1]
            if isinstance(start_pos, mx.array):
                pool_base = mx.maximum(start_pos, 0) - buf_len
            else:
                pool_base = max(0, start_pos) - buf_len
            return kv[:, :usable], gate[:, :usable], pool_base

        # Slow path: per-row variable lengths. Build a max-length buffer,
        # copy each row's valid tokens in, slice off ready windows per-row.
        buf_lengths = _as_lengths_list(
            branch.buffer_lengths, B, 0 if buf_kv is None else buf_kv.shape[1]
        )
        chunk_lengths = _as_lengths_list(chunk_lengths, B, L)
        total_lengths = [
            bl + min(cl, L) for bl, cl in zip(buf_lengths, chunk_lengths)
        ]
        usable_lengths = [(t // ratio) * ratio for t in total_lengths]
        new_buf_lengths = [t - u for t, u in zip(total_lengths, usable_lengths)]
        max_total = max(total_lengths, default=0)
        max_usable = max(usable_lengths, default=0)
        max_buf = max(new_buf_lengths, default=0)

        combined_kv = mx.zeros((B, max_total, kv.shape[-1]), dtype=kv.dtype)
        combined_gate = mx.zeros((B, max_total, gate.shape[-1]), dtype=gate.dtype)
        for i, (bl, cl, tl) in enumerate(
            zip(buf_lengths, chunk_lengths, total_lengths)
        ):
            parts_kv, parts_gate = [], []
            if bl:
                parts_kv.append(buf_kv[i : i + 1, :bl])
                parts_gate.append(buf_gate[i : i + 1, :bl])
            if cl:
                parts_kv.append(kv[i : i + 1, : min(cl, L)])
                parts_gate.append(gate[i : i + 1, : min(cl, L)])
            if parts_kv:
                row_kv = parts_kv[0] if len(parts_kv) == 1 else mx.concatenate(
                    parts_kv, axis=1
                )
                row_gate = parts_gate[0] if len(parts_gate) == 1 else mx.concatenate(
                    parts_gate, axis=1
                )
                combined_kv[i : i + 1, :tl] = row_kv
                combined_gate[i : i + 1, :tl] = row_gate

        ready_kv = combined_kv[:, :max_usable]
        ready_gate = combined_gate[:, :max_usable]
        new_buf_kv = mx.zeros((B, max_buf, kv.shape[-1]), dtype=kv.dtype)
        new_buf_gate = mx.zeros((B, max_buf, gate.shape[-1]), dtype=gate.dtype)
        for i, (u, bl) in enumerate(zip(usable_lengths, new_buf_lengths)):
            if bl:
                new_buf_kv[i : i + 1, :bl] = combined_kv[i : i + 1, u : u + bl]
                new_buf_gate[i : i + 1, :bl] = combined_gate[i : i + 1, u : u + bl]
        branch.buffer_kv = new_buf_kv if max_buf > 0 else None
        branch.buffer_gate = new_buf_gate if max_buf > 0 else None
        branch.buffer_lengths = new_buf_lengths if max_buf > 0 else None
        branch._new_pool_lengths = [u // ratio for u in usable_lengths]

        prev_buf_lengths = mx.array(buf_lengths, dtype=mx.float32)
        if isinstance(start_pos, mx.array):
            base = mx.maximum(start_pos, 0).astype(mx.float32)
        else:
            base = mx.full((B,), max(0, start_pos), dtype=mx.float32)
        return ready_kv, ready_gate, base - prev_buf_lengths

    def update_pool(self, new_pooled, key):
        """Append ``new_pooled`` to the pool for ``key``. ``new_pooled`` has
        shape [B, n_new, head_dim]. Per-row pool lengths are updated based
        on the ``_new_pool_lengths`` set by the matching ``accumulate_windows``
        call (or fall through to uniform if there was none).
        """
        branch = self._branches[key]
        new_lengths = branch._new_pool_lengths
        branch._new_pool_lengths = None

        pool = branch.pool
        if new_lengths is not None:
            B = new_pooled.shape[0]
            cur_lengths = _as_lengths_list(
                branch.pool_lengths, B, 0 if pool is None else pool.shape[1]
            )
            total_lengths = [c + n for c, n in zip(cur_lengths, new_lengths)]
            max_total = max(total_lengths, default=0)
            merged = mx.zeros(
                (B, max_total, new_pooled.shape[-1]), dtype=new_pooled.dtype
            )
            for i, (c, n) in enumerate(zip(cur_lengths, new_lengths)):
                if pool is not None and c:
                    merged[i : i + 1, :c] = pool[i : i + 1, :c]
                if n:
                    merged[i : i + 1, c : c + n] = new_pooled[i : i + 1, :n]
            branch.pool = merged
            branch.pool_lengths = total_lengths
            return merged

        if new_pooled.shape[1] > 0:
            pool = (
                new_pooled
                if pool is None
                else mx.concatenate([pool, new_pooled], axis=1)
            )
            branch.pool = pool
            branch.pool_lengths = None
        if pool is None:
            pool = mx.zeros(
                (new_pooled.shape[0], 0, new_pooled.shape[-1]), dtype=new_pooled.dtype
            )
        return pool

    def pooled_lengths(self, key):
        return self._branches[key].pool_lengths

    def get_branch(self, key) -> _CompressorBranch:
        return self._branches[key]

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cache_batch_size(local):
        offset = getattr(local, "offset", 0)
        if isinstance(offset, mx.array) and offset.ndim:
            return offset.shape[0]
        if local.keys is not None:
            return local.keys.shape[0]
        return 1

    @staticmethod
    def _extract_local(local, idx):
        cache = RotatingKVCache(local.max_size, keep=getattr(local, "keep", 0))
        if local.keys is not None:
            keys = local._temporal_order(local.keys)
            values = local._temporal_order(local.values)
            cache.keys = mx.contiguous(keys[idx : idx + 1])
            cache.values = mx.contiguous(values[idx : idx + 1])
            cache._idx = cache.keys.shape[2]
        cache.offset = local.offset
        return cache

    @classmethod
    def _batch_rotating_from(cls, local):
        batch = cls._cache_batch_size(local)
        return BatchRotatingKVCache.merge(
            [cls._extract_local(local, i) for i in range(batch)]
        )

    @staticmethod
    def _concat_optional(a, b):
        if a is None:
            return b
        if b is None:
            return a
        return mx.concatenate([a, b], axis=0)

    @classmethod
    def _merge_local(cls, locals_):
        offsets = [l.offset for l in locals_]
        sizes = [l.size() for l in locals_]
        uniform = (
            all(not isinstance(o, mx.array) for o in offsets)
            and all(o == offsets[0] for o in offsets)
            and all(s == sizes[0] for s in sizes)
        )
        if not uniform:
            if hasattr(locals_[0], "merge"):
                return locals_[0].merge(locals_)
            return BatchRotatingKVCache.merge(locals_)
        out = RotatingKVCache(locals_[0].max_size, keep=getattr(locals_[0], "keep", 0))
        out.offset = offsets[0]
        if sizes[0] == 0:
            return out
        out.keys = mx.concatenate(
            [l._temporal_order(l.keys) for l in locals_], axis=0
        )
        out.values = mx.concatenate(
            [l._temporal_order(l.values) for l in locals_], axis=0
        )
        out._idx = out.keys.shape[2]
        return out

    @staticmethod
    def _concat_batch_state(a, b, a_batch, b_batch):
        if a is None and b is None:
            return None
        if a is None:
            a = mx.zeros((a_batch,) + b.shape[1:], dtype=b.dtype)
        if b is None:
            b = mx.zeros((b_batch,) + a.shape[1:], dtype=a.dtype)
        if a.shape[2:] != b.shape[2:]:
            raise ValueError("DeepseekV4Cache extend: state tail shape mismatch")
        if a.shape[1] != b.shape[1]:
            seq_len = max(a.shape[1], b.shape[1])
            if a.shape[1] != seq_len:
                pad = mx.zeros((a.shape[0], seq_len, *a.shape[2:]), dtype=a.dtype)
                pad[:, : a.shape[1]] = a
                a = pad
            if b.shape[1] != seq_len:
                pad = mx.zeros((b.shape[0], seq_len, *b.shape[2:]), dtype=b.dtype)
                pad[:, : b.shape[1]] = b
                b = pad
        return mx.concatenate([a, b], axis=0)

    @staticmethod
    def _full_lengths(lengths, value, batch_size):
        if lengths is not None:
            if isinstance(lengths, mx.array):
                lengths = lengths.tolist()
            return [int(x) for x in lengths]
        length = 0 if value is None else value.shape[1]
        return [length] * batch_size

    @classmethod
    def _concat_lengths(cls, a_lengths, b_lengths, a_value, b_value, a_batch, b_batch):
        a = cls._full_lengths(a_lengths, a_value, a_batch)
        b = cls._full_lengths(b_lengths, b_value, b_batch)
        merged = a + b
        m = max(merged, default=0)
        return None if all(x == m for x in merged) else merged

    @classmethod
    def _merge_lengths(cls, lengths, values):
        per_batch = [
            cls._full_lengths(l, v, 1)[0] for l, v in zip(lengths, values)
        ]
        m = max(per_batch, default=0)
        return None if all(x == m for x in per_batch) else per_batch

    @staticmethod
    def _merge_batch_state(values):
        present = [v for v in values if v is not None]
        if not present:
            return None
        if not all(v.shape[2:] == present[0].shape[2:] for v in present):
            raise ValueError("DeepseekV4Cache merge: state tail shape mismatch")
        seq_len = max(v.shape[1] for v in present)
        shape = present[0].shape
        dtype = present[0].dtype
        merged = []
        for v in values:
            if v is None:
                merged.append(mx.zeros((1, seq_len, *shape[2:]), dtype=dtype))
            else:
                if v.shape[1] != seq_len:
                    pad = mx.zeros((v.shape[0], seq_len, *v.shape[2:]), dtype=v.dtype)
                    pad[:, : v.shape[1]] = v
                    v = pad
                merged.append(v)
        return mx.concatenate(merged, axis=0)


# --------------------------------------------------------------------------- #
# Compressor                                                                  #
# --------------------------------------------------------------------------- #


def _make_overlap_emit_kernel():
    """Single-dispatch overlap emit. Reads state_kv / state_score in the slot
    layout [B, 2*ratio, 2*head_dim] (prev window in [0..ratio), current window
    in [ratio..2*ratio)) and runs softmax-weighted-sum entirely in fp32
    registers, casting to the input dtype only at the end. Used by the decode
    emit path (S=1) so its output is bit-identical to the pre-refactor code,
    which used this same kernel.
    """
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    src = """
        uint idx = thread_position_in_grid.x;
        uint total = B * D;
        if (idx >= total) return;
        uint b = idx / D;
        uint d = idx % D;

        auto skv = state_kv + b * (2 * RATIO) * (2 * D);
        auto ssc = state_score + b * (2 * RATIO) * (2 * D);

        float scores[2 * RATIO];
        float kvs[2 * RATIO];

        for (int i = 0; i < RATIO; ++i) {
            scores[i] = static_cast<float>(ssc[i * (2 * D) + d]);
            kvs[i] = static_cast<float>(skv[i * (2 * D) + d]);
        }
        for (int i = 0; i < RATIO; ++i) {
            scores[RATIO + i] = static_cast<float>(ssc[(RATIO + i) * (2 * D) + D + d]);
            kvs[RATIO + i] = static_cast<float>(skv[(RATIO + i) * (2 * D) + D + d]);
        }

        float m = -INFINITY;
        for (int i = 0; i < 2 * RATIO; ++i) m = metal::max(m, scores[i]);
        float s = 0.0f;
        for (int i = 0; i < 2 * RATIO; ++i) {
            scores[i] = metal::fast::exp(scores[i] - m);
            s += scores[i];
        }
        float inv_s = 1.0f / s;
        float acc = 0.0f;
        for (int i = 0; i < 2 * RATIO; ++i) {
            acc += scores[i] * inv_s * kvs[i];
        }
        y[idx] = static_cast<OUT_T>(acc);
    """
    return mx.fast.metal_kernel(
        name="dsv4_compressor_overlap_emit",
        input_names=["state_kv", "state_score"],
        output_names=["y"],
        source=src,
    )


_overlap_emit_kernel = _make_overlap_emit_kernel()


def _overlap_emit(state_kv: mx.array, state_score: mx.array, ratio: int) -> mx.array:
    """Single overlap-emit pool row. Falls back to a pure-MLX equivalent if the
    Metal kernel isn't available — the fallback's accumulation order differs by
    a single bf16 ULP, so the kernel path is required for byte-identical
    parity with the pre-refactor decode code."""
    if _overlap_emit_kernel is None:
        B, _, coff_d = state_kv.shape
        d = coff_d // 2
        first = state_kv[:, :ratio, :d]
        second = state_kv[:, ratio:, d:]
        merged_kv = mx.concatenate([first, second], axis=1)
        first_s = state_score[:, :ratio, :d]
        second_s = state_score[:, ratio:, d:]
        merged_score = mx.concatenate([first_s, second_s], axis=1)
        weights = mx.softmax(
            merged_score.astype(mx.float32), axis=1, precise=True
        ).astype(merged_kv.dtype)
        return (merged_kv * weights).sum(axis=1)

    B = state_kv.shape[0]
    D = state_kv.shape[-1] // 2
    total = B * D
    tg = 256
    grid = ((total + tg - 1) // tg) * tg
    return _overlap_emit_kernel(
        inputs=[state_kv, state_score],
        template=[("B", B), ("RATIO", ratio), ("D", D), ("OUT_T", state_kv.dtype)],
        grid=(grid, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B, D)],
        output_dtypes=[state_kv.dtype],
    )[0]


@mx.compile
def _compressor_wkv_gate_split_quant(x, w, s, group_size, bits, split):
    """Fused mxfp4 wkv_gate matmul + slice into kv/score halves."""
    kv_gate = mx.quantized_matmul(
        x, w, scales=s, transpose=True,
        group_size=group_size, bits=bits, mode="mxfp4",
    )
    return kv_gate[..., :split], kv_gate[..., split:]


@mx.compile
def _compressor_split(kv_gate, split):
    """Slice the wkv_gate output into kv/score halves (compile fuse with the
    callsite's downstream ops)."""
    return kv_gate[..., :split], kv_gate[..., split:]


@mx.compile
def _compressor_norm_strided_rope(c, norm_w, norm_eps, offset, rd, scale, freqs):
    """Fused: RMSNorm + strided rope on the last rd dims."""
    c = mx.fast.rms_norm(c, norm_w, norm_eps)
    rotated = mx.fast.rope(
        c[..., -rd:], rd, traditional=True, base=None,
        scale=scale, offset=offset, freqs=freqs,
    )
    return mx.concatenate([c[..., :-rd], rotated], axis=-1)


class Compressor(nn.Module):
    """Learned gated pooling over ``ratio`` consecutive tokens.

    Prefill: chunk the input into windows of ``ratio`` tokens, softmax-gate
    across each window, sum to one compressed row per window; any tail shorter
    than ``ratio`` lives in the cache's ``comp_*_state`` buffer until enough
    tokens accumulate.

    Decode (S==1): append the new token's kv/score into the accumulator; if we
    just filled a window, emit one compressed row and rotate the buffer.
    """

    def __init__(
        self,
        dim: int,
        compress_ratio: int,
        head_dim: int,
        rope_head_dim: int,
        rms_norm_eps: float,
        rope: "DeepseekV4RoPE",
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        coff = 2 if self.overlap else 1
        self._coff = coff
        # Fused wkv + wgate — same input x, concat output dims. Sanitize
        # concats the two weight tensors along axis=0 at load time.
        self._wkv_gate_split = coff * head_dim
        self.wkv_gate = nn.Linear(dim, 2 * coff * head_dim, bias=False)
        self.ape = mx.zeros((compress_ratio, coff * head_dim), dtype=mx.float32)
        self.norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)
        self.rope = rope  # shared with attention, applies compress_rope_theta + YaRN

    def _overlap_transform_kv(self, kv: mx.array) -> mx.array:
        # kv: [B, n_windows, ratio, coff*d]; output [B, n_windows, 2*ratio, d]
        B, S, R, _ = kv.shape
        d = self.head_dim
        out = mx.zeros((B, S, 2 * R, d), dtype=kv.dtype)
        # second half: "current" window's last-d dims
        out[:, :, R:, :] = kv[:, :, :, d:]
        # first half: previous window's first-d dims
        out[:, 1:, :R, :] = kv[:, :-1, :, :d]
        return out

    def _overlap_transform_score(self, score: mx.array) -> mx.array:
        B, S, R, _ = score.shape
        d = self.head_dim
        out = mx.full((B, S, 2 * R, d), float("-inf"), dtype=score.dtype)
        out[:, :, R:, :] = score[:, :, :, d:]
        out[:, 1:, :R, :] = score[:, :-1, :, :d]
        return out

    def _apply_compressor_rope(
        self, compressed_kv: mx.array, first_pos: int
    ) -> mx.array:
        """Apply RoPE on the last ``rope_head_dim`` dims at strided positions
        ``first_pos, first_pos + ratio, first_pos + 2*ratio, ...``.

        ``first_pos`` must be divisible by ``compress_ratio`` (caller guarantees).
        """
        rd = self.rope_head_dim
        # mx.fast.rope generates positions as ``scale * (offset + arange(T))``.
        # We want ``first_pos + arange(T) * ratio`` = ``ratio * (first_pos/ratio + arange(T))``.
        rotated = mx.fast.rope(
            compressed_kv[..., -rd:],
            rd,
            traditional=True,
            base=None,
            scale=float(self.compress_ratio),
            offset=first_pos // self.compress_ratio,
            freqs=self.rope.freqs,
        )
        return mx.concatenate([compressed_kv[..., :-rd], rotated], axis=-1)

    def _apply_compressor_rope_per_row(
        self, compressed_kv: mx.array, pool_base
    ) -> mx.array:
        """Per-row RoPE for the variable-length case where ``pool_base`` is
        an mx.array of shape [B] (one starting raw-token position per row).
        Each pool row k for batch row b has position ``pool_base[b] + k*ratio``.
        """
        rd = self.rope_head_dim
        ratio = self.compress_ratio
        B, T = compressed_kv.shape[0], compressed_kv.shape[1]
        # Positions [B, T] = pool_base[:, None] + ratio * arange(T)
        positions = (
            pool_base[:, None] + ratio * mx.arange(T, dtype=pool_base.dtype)
        ).astype(mx.float32)
        # Apply 2D rope manually: cos/sin from positions, rotate last rd dims.
        inv = 1.0 / self.rope.freqs[: rd // 2]
        freqs = positions[..., None] * inv  # [B, T, rd/2]
        cos = mx.cos(freqs).astype(compressed_kv.dtype)
        sin = mx.sin(freqs).astype(compressed_kv.dtype)
        pe = compressed_kv[..., -rd:]
        pe = pe.reshape(B, T, rd // 2, 2)
        x0, x1 = pe[..., 0], pe[..., 1]
        out = mx.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)
        pe_out = out.reshape(B, T, rd)
        return mx.concatenate([compressed_kv[..., :-rd], pe_out], axis=-1)

    def __call__(
        self,
        x: mx.array,
        cache: "DeepseekV4Cache",
        offset,
        key: str = _K_COMP,
    ) -> mx.array:
        """Compute zero or more compressed pool rows for this forward step
        and append them to ``cache``'s pool for ``key``.

        Returns the (possibly empty) per-row pool tensor. Empty when no row
        in the batch has accumulated enough tokens to emit. Per-row pool
        lengths are tracked inside ``cache``; querying via
        ``cache.pooled_lengths(key)`` returns ``None`` for the uniform case
        or a length list for the variable-length case.
        """
        B, S, _ = x.shape
        ratio = self.compress_ratio
        d = self.head_dim
        coff_d = self._coff * d

        # Single fused matmul + slice into kv/score halves. Quantized mxfp4
        # path collapses the two ops into one dispatch; everything else
        # (bf16, affine-quantized, etc.) goes through the layer's normal
        # __call__ and the slice fuses into the downstream graph.
        wg = self.wkv_gate
        if isinstance(wg, nn.QuantizedLinear) and wg.mode == "mxfp4":
            kv, score = _compressor_wkv_gate_split_quant(
                x, wg.weight, wg.scales, wg.group_size, wg.bits, self._wkv_gate_split,
            )
        else:
            kv, score = _compressor_split(wg(x), self._wkv_gate_split)

        # Append to the carry buffer; pull out the slice that's ready to be
        # folded into ratio-sized windows. ``pool_base`` is the raw-token
        # position of the first ready token (int for uniform offsets, [B]
        # mx.array per-row for variable-length).
        ready_kv, ready_score, pool_base = cache.accumulate_windows(
            kv, score, key, ratio, offset
        )

        # Always run the emit path (with shape-stable empty tensors) so the
        # graph stays consistent. For overlap layers we additionally need the
        # previous full window's data; that lives in branch.prev_kv / prev_gate
        # and is updated as we emit.
        branch = cache.get_branch(key)
        max_usable = ready_kv.shape[1]
        if max_usable == 0:
            # No new rows this step. Skip the update_pool call entirely
            # (saves an mx.zeros allocation + a no-op concat per non-emit
            # step in every compress!=0 layer — significant on Flash where
            # ~90% of layers are compress!=0).
            pool = branch.pool
            if pool is None:
                pool = mx.zeros((B, 0, d), dtype=x.dtype)
            return pool

        W = max_usable // ratio
        kv_win = ready_kv.reshape(B, W, ratio, coff_d)
        score_win = ready_score.reshape(B, W, ratio, coff_d) + self.ape.astype(
            ready_score.dtype
        )

        is_decode_emit = self.overlap and S < ratio and W == 1
        if is_decode_emit:
            buf_score_with_ape = score_win[:, 0, :, :]
            cur_kv = kv_win[:, 0, :, :]
            prev_kv = branch.prev_kv
            prev_gate = branch.prev_gate
            if prev_kv is None:
                prev_kv = mx.zeros_like(cur_kv)
                prev_gate = mx.full(cur_kv.shape, float("-inf"), dtype=cur_kv.dtype)
            state_kv = mx.concatenate([prev_kv, cur_kv], axis=1)
            state_score = mx.concatenate([prev_gate, buf_score_with_ape], axis=1)
            new_pooled = _overlap_emit(state_kv, state_score, ratio)[:, None, :]
            branch.prev_kv = cur_kv
            branch.prev_gate = buf_score_with_ape
        elif self.overlap:
            prev_kv = branch.prev_kv
            prev_score = branch.prev_gate
            kv_trans = self._overlap_transform_kv_runtime(kv_win, prev_kv)
            score_trans = self._overlap_transform_score_runtime(score_win, prev_score)
            weights = mx.softmax(
                score_trans.astype(mx.float32), axis=2, precise=True
            ).astype(kv_trans.dtype)
            new_pooled = (kv_trans * weights).sum(axis=2)
            branch.prev_kv = kv_win[:, -1, :, :]
            branch.prev_gate = score_win[:, -1, :, :]
        else:
            weights = mx.softmax(
                score_win.astype(mx.float32), axis=2, precise=True
            ).astype(kv_win.dtype)
            new_pooled = (kv_win * weights).sum(axis=2)[..., :d]

        if isinstance(pool_base, mx.array) and pool_base.ndim:
            # Per-row positions — can't use mx.fast.rope (it only takes a
            # scalar offset). Fall through to manual norm + per-row rope.
            new_pooled = self.norm(new_pooled)
            new_pooled = self._apply_compressor_rope_per_row(new_pooled, pool_base)
        else:
            # Hot uniform path — fused RMSNorm + strided rope.
            new_pooled = _compressor_norm_strided_rope(
                new_pooled,
                self.norm.weight,
                self.norm.eps,
                int(pool_base) // ratio,
                self.rope_head_dim,
                float(ratio),
                self.rope.freqs,
            )
        # exo fix: the reference's FP4 (indexer) / FP8 (attention) rounding of new rows
        if key == _K_IDX:
            new_pooled = _fp4_indexer_sim(new_pooled)
        else:
            new_pooled = _fp8_kv_sim(new_pooled, self.rope_head_dim)
        return cache.update_pool(new_pooled, key)

    def _overlap_transform_kv_runtime(
        self, kv_win: mx.array, prev_kv: Optional[mx.array]
    ) -> mx.array:
        """Build [B, W, 2*ratio, d] from current windows + previous-window carry.

        kv_win  : [B, W, ratio, 2*d] — current windows; first ``d`` channels are
                  the "previous-half" (used for the next window), last ``d`` are
                  the "current-half" (used for this window's output).
        prev_kv : [B, ratio, 2*d] — last window from a previous call, or None.

        Output [b, w, 2*ratio, d]:
          - rows [ratio..2*ratio): kv_win[b, w, :, d:]   (always)
          - rows [0..ratio):       kv_win[b, w-1, :, :d] for w >= 1
                                   prev_kv[b, :, :d]      for w == 0 (if prev_kv)
                                   zeros                  otherwise
        """
        B, W, R, _ = kv_win.shape
        d = self.head_dim
        out = mx.zeros((B, W, 2 * R, d), dtype=kv_win.dtype)
        out[:, :, R:, :] = kv_win[:, :, :, d:]
        if W >= 2:
            out[:, 1:, :R, :] = kv_win[:, :-1, :, :d]
        if prev_kv is not None:
            out[:, 0, :R, :] = prev_kv[:, :, :d]
        return out

    def _overlap_transform_score_runtime(
        self, score_win: mx.array, prev_score: Optional[mx.array]
    ) -> mx.array:
        B, W, R, _ = score_win.shape
        d = self.head_dim
        out = mx.full((B, W, 2 * R, d), float("-inf"), dtype=score_win.dtype)
        out[:, :, R:, :] = score_win[:, :, :, d:]
        if W >= 2:
            out[:, 1:, :R, :] = score_win[:, :-1, :, :d]
        if prev_score is not None:
            out[:, 0, :R, :] = prev_score[:, :, :d]
        return out


# --------------------------------------------------------------------------- #
# exo fix: reference QAT simulation (FP8 KV, FP4 indexer)                     #
# --------------------------------------------------------------------------- #
# The reference rounds the non-RoPE KV dims to FP8 (e4m3, one power-of-two
# scale per 64 values) and the indexer q / compressed k to FP4 (e2m1, one
# power-of-two scale per 32 values, after a Hadamard rotation). The model was
# trained with these roundings, so reproduce them (values stay bf16).


def _pow2_ceil(x: mx.array) -> mx.array:
    """Exact 2**ceil(log2(x)) for positive float32 x (exponent bits)."""
    bits = x.astype(mx.float32).view(mx.int32)
    exponent = mx.bitwise_and(mx.right_shift(bits, 23), 0xFF)
    mantissa = mx.bitwise_and(bits, (1 << 23) - 1)
    e = exponent - 127 + (mantissa != 0).astype(mx.int32)
    return mx.left_shift(e + 127, 23).view(mx.float32)


def _round_e2m1(v: mx.array) -> mx.array:
    """Round to the nearest FP4 (e2m1) value, ties to even; |v| <= 6."""
    a = mx.abs(v)
    q = mx.where(a <= 0.25, 0.0, mx.where(a < 0.75, 0.5, mx.where(
        a <= 1.25, 1.0, mx.where(a < 1.75, 1.5, mx.where(
            a <= 2.5, 2.0, mx.where(a < 3.5, 3.0, mx.where(a <= 5.0, 4.0, 6.0)))))))
    return mx.where(v < 0, -q, q)


def _block_round(x: mx.array, block: int, fp4: bool) -> mx.array:
    qmax, amin = (6.0, 6.0 * 2.0**-126) if fp4 else (448.0, 1e-4)
    shape, dtype = x.shape, x.dtype
    xf = x.astype(mx.float32).reshape(*shape[:-1], shape[-1] // block, block)
    amax = mx.maximum(mx.max(mx.abs(xf), axis=-1, keepdims=True), amin)
    s = _pow2_ceil(amax * (1.0 / qmax))
    y = mx.clip(xf / s, -qmax, qmax)
    y = _round_e2m1(y) if fp4 else mx.from_fp8(mx.to_fp8(y), dtype=mx.float32)
    return (y * s).reshape(shape).astype(dtype)


def _fp8_kv_sim(kv: mx.array, rd: int) -> mx.array:
    return mx.concatenate([_block_round(kv[..., :-rd], 64, False), kv[..., -rd:]], axis=-1)


_HADAMARD: Dict[int, mx.array] = {}


def _fp4_indexer_sim(x: mx.array) -> mx.array:
    n = x.shape[-1]
    if n not in _HADAMARD:
        h = mx.array([[1.0]])
        while h.shape[0] < n:
            h = mx.concatenate([mx.concatenate([h, h], 1), mx.concatenate([h, -h], 1)], 0)
        _HADAMARD[n] = h * (n**-0.5)
    rotated = (x.astype(mx.float32) @ _HADAMARD[n].T).astype(x.dtype)
    return _block_round(rotated, 32, True)


# --------------------------------------------------------------------------- #
# Indexer                                                                     #
# --------------------------------------------------------------------------- #


class Indexer(nn.Module):
    """Scores per-query visibility over the main compressed KV buffer and
    returns the top-k compressed-row indices per query. V4Attention turns
    those indices into a boolean mask on the compressed portion of SDPA's KV
    so each query only attends to its top-k far-context slots.
    """

    def __init__(
        self,
        args: ModelArgs,
        compress_ratio: int,
        rope: "DeepseekV4RoPE",
    ):
        super().__init__()
        self.dim = args.hidden_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        self.compress_ratio = compress_ratio
        self.softmax_scale = self.head_dim ** -0.5
        self.rope = rope
        self.wq_b = nn.Linear(
            args.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.weights_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)
        self.compressor = Compressor(
            dim=args.hidden_size,
            compress_ratio=compress_ratio,
            head_dim=self.head_dim,
            rope_head_dim=args.qk_rope_head_dim,
            rms_norm_eps=args.rms_norm_eps,
            rope=rope,
        )

    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        cache: "DeepseekV4Cache",
        offset,
    ) -> Optional[mx.array]:
        """Update the Indexer's own compressed buffer and return
        ``top_idxs [B, S, k]`` (``k = min(index_topk, pooled_len)``) into that
        buffer. Returns ``None`` if no compressed rows exist yet. Indices are
        always in ``[0, pooled_len)``; no ``-1`` padding. Per-row pool
        lengths (variable-length batch case) are masked by zeroing scores
        for indices >= length[row] before topk so each row's selection
        stays inside its valid pool.
        """
        B, S, _ = x.shape
        rd = self.rope_head_dim

        idx_kv = self.compressor(x, cache, offset, key=_K_IDX)
        if idx_kv is None or idx_kv.shape[1] == 0:
            return None

        # exo fix: RoPE positions run along the sequence axis. mx.fast.rope takes them
        # from axis -2, so rotate q as [B, n_heads, S, head_dim].
        q = self.wq_b(qr).reshape(B, S, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        q = mx.concatenate(
            [q[..., :-rd], self.rope(q[..., -rd:], offset=offset)], axis=-1
        )
        # exo fix: Hadamard rotation and FP4 rounding of the query, like the reference
        q = _fp4_indexer_sim(q.transpose(0, 2, 1, 3))
        per_head_weights = self.weights_proj(x) * (
            self.softmax_scale * (self.n_heads ** -0.5)
        )
        score = mx.einsum("bshd,btd->bsht", q.astype(idx_kv.dtype), idx_kv)
        score = mx.maximum(score, 0)  # ReLU
        # Reduce over n_heads as a matmul: [B,S,1,n_heads] @ [B,S,n_heads,T] → [B,S,T]
        score = (per_head_weights[:, :, None, :] @ score).squeeze(2)

        # Per-row pool length masking — zero scores for invalid pool slots so
        # topk doesn't pick padding zeros from short rows.
        pool_lengths = cache.pooled_lengths(_K_IDX)
        if pool_lengths is not None:
            lengths_a = mx.array(pool_lengths, dtype=mx.int32)
            valid = (
                mx.arange(idx_kv.shape[1], dtype=mx.int32)[None, None, :]
                < lengths_a[:, None, None]
            )
            score = mx.where(valid, score, mx.array(-1e30, dtype=score.dtype))

        # exo fix: in multi-token steps (prefill), rows that complete later in the step are
        # not visible to earlier queries. Exclude them before top-k, like the
        # reference, so each query keeps index_topk rows it can attend to.
        if S > 1:
            visible = _compressed_visibility(
                B, S, offset, idx_kv.shape[1], self.compress_ratio
            )[:, 0]
            score = mx.where(visible, score, mx.array(-math.inf, dtype=score.dtype))

        k = min(self.index_topk, idx_kv.shape[1])
        return mx.argpartition(-score, kth=k - 1, axis=-1)[..., :k].astype(
            mx.int32
        )


# --------------------------------------------------------------------------- #
# Attention                                                                   #
# --------------------------------------------------------------------------- #


def _build_window_mask(
    B: int,
    S: int,
    offset,
    window: int,
    window_len: int,
) -> mx.array:
    """Sliding-window causal mask of shape ``[B, 1, S, window_len]``."""
    if isinstance(offset, mx.array):
        off = offset.astype(mx.int32).reshape(-1)
        q_pos = off[:, None] + mx.arange(S, dtype=mx.int32)
        end = off + S
        cache_k = mx.arange(window_len, dtype=mx.int32)
        raw_pos_at_k = end[:, None] - window_len + cache_k[None, :]
        win_visible = (
            (raw_pos_at_k[:, None, :] <= q_pos[:, :, None])
            & (raw_pos_at_k[:, None, :] > q_pos[:, :, None] - window)
        )
    else:
        q_pos = mx.broadcast_to(
            offset + mx.arange(S, dtype=mx.int32)[None, :], (B, S)
        )
        cache_k = mx.arange(window_len, dtype=mx.int32)
        raw_pos_at_k = (offset + S) - window_len + cache_k
        win_visible = (
            (raw_pos_at_k[None, None, :] <= q_pos[:, :, None])
            & (raw_pos_at_k[None, None, :] > q_pos[:, :, None] - window)
        )
    return win_visible[:, None, :, :]


def _compressed_visibility(
    B: int,
    S: int,
    offset,
    compressed_len: int,
    ratio: int,
) -> mx.array:
    """Bool mask ``[B, 1, S, compressed_len]``: row ``k`` visible to query at
    raw pos ``p`` iff ``(k+1)*ratio <= p+1``."""
    if isinstance(offset, mx.array):
        off = offset.astype(mx.int32).reshape(-1)
        q_pos = off[:, None] + mx.arange(S, dtype=mx.int32)
    else:
        q_pos = mx.broadcast_to(
            offset + mx.arange(S, dtype=mx.int32)[None, :], (B, S)
        )
    k = mx.arange(compressed_len, dtype=mx.int32)
    comp_visible = (k + 1)[None, None, :] * ratio <= (q_pos + 1)[:, :, None]
    return comp_visible[:, None, :, :]


# Attention-side compile fuses (per-layer cost reduction). These collapse the
# sequence  matmul → reshape/transpose/slice → norm → (rope)  into single
# compiled kernels so each prefill attention call shaves several dispatches.
# Quantized variants take the QuantizedLinear's (weight, scales, group_size,
# bits) tuple and call mx.quantized_matmul directly — letting the same
# compile graph cover both the matmul and the post-ops.


@mx.compile
def _attn_wqkv_quant_split_norm(x, w, s, group_size, bits, q_w, kv_w, q_lora, eps):
    """Fused mxfp4 wqkv_a matmul + slice + 2 RMSNorms (q half + kv half)."""
    qkv_a = mx.quantized_matmul(
        x, w, scales=s, transpose=True,
        group_size=group_size, bits=bits, mode="mxfp4",
    )
    qr = mx.fast.rms_norm(qkv_a[..., :q_lora], q_w, eps)
    kv = mx.fast.rms_norm(qkv_a[..., q_lora:], kv_w, eps)
    return qr, kv


@mx.compile
def _attn_qkv_split_norm(qkv_a, q_w, kv_w, q_lora, eps):
    """Non-quant variant: slice + 2 RMSNorms."""
    qr = mx.fast.rms_norm(qkv_a[..., :q_lora], q_w, eps)
    kv = mx.fast.rms_norm(qkv_a[..., q_lora:], kv_w, eps)
    return qr, kv


@mx.compile
def _attn_q_proj_quant_norm(qr, w, s, group_size, bits, n_heads, head_dim, eps):
    """Fused mxfp4 wq_b matmul + reshape/transpose to [B, n_heads, S, head_dim]
    + per-head RMSNorm (no learned weight)."""
    q = mx.quantized_matmul(
        qr, w, scales=s, transpose=True,
        group_size=group_size, bits=bits, mode="mxfp4",
    )
    B, S = q.shape[0], q.shape[1]
    q = q.reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)
    return mx.fast.rms_norm(q, None, eps)


@mx.compile
def _attn_q_proj_norm(q_flat, n_heads, head_dim, eps):
    """Non-quant variant: reshape/transpose + per-head RMSNorm."""
    B, S = q_flat.shape[0], q_flat.shape[1]
    q = q_flat.reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)
    return mx.fast.rms_norm(q, None, eps)


@mx.compile
def _attn_qkv_partial_rope(q, kv, offset, rd, freqs):
    """Fused partial-RoPE on the trailing rd dims of q [B,H,S,D] and kv [B,S,D].
    Slice + rope + concat × 2 → 2 dispatches collapse to 2 compiled kernels."""
    q_pe = mx.fast.rope(
        q[..., -rd:], rd, traditional=True, base=None,
        scale=1.0, offset=offset, freqs=freqs,
    )
    q = mx.concatenate([q[..., :-rd], q_pe], axis=-1)
    kv_pe = mx.fast.rope(
        kv[..., -rd:], rd, traditional=True, base=None,
        scale=1.0, offset=offset, freqs=freqs,
    )
    kv = mx.concatenate([kv[..., :-rd], kv_pe], axis=-1)
    return q, kv


@mx.compile
def _attn_inv_rope_flatten(o, offset, rd, freqs, flat_dim):
    """Inverse-RoPE on trailing rd dims + transpose-and-reshape to
    ``[B, S, n_heads*head_dim]`` for the wo_a input."""
    pe = mx.fast.rope(
        o[..., -rd:], rd, traditional=True, base=None,
        scale=-1.0, offset=offset, freqs=freqs,
    )
    o = mx.concatenate([o[..., :-rd], pe], axis=-1)
    o = o.transpose(0, 2, 1, 3)
    return o.reshape(o.shape[0], o.shape[1], flat_dim)


@mx.compile
def _attn_wo_chain_quant(
    o, woa_w, woa_s, wob_w, wob_s, n_groups, o_lora_rank,
    woa_group_size, woa_bits, wob_group_size, wob_bits,
):
    """Fused: grouped wo_a mxfp4 matmul + transpose-reshape + wo_b mxfp4 matmul.
    Replaces the separate _grouped_output_projection + self.wo_b(...) calls
    with a single compile graph (one fewer dispatch per layer per call).
    """
    B, S, F = o.shape
    group_feat = F // n_groups
    o_g = o.reshape(B, S, n_groups, group_feat).transpose(2, 0, 1, 3)
    weight = woa_w.reshape(n_groups, o_lora_rank, -1)[:, None]
    scales = woa_s.reshape(n_groups, o_lora_rank, -1)[:, None]
    y = mx.quantized_matmul(
        o_g, weight, scales=scales, transpose=True,
        group_size=woa_group_size, bits=woa_bits, mode="mxfp4",
    )
    y = y.transpose(1, 2, 0, 3).reshape(B, S, n_groups * o_lora_rank)
    return mx.quantized_matmul(
        y, wob_w, scales=wob_s, transpose=True,
        group_size=wob_group_size, bits=wob_bits, mode="mxfp4",
    )


class V4Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.nope_head_dim = args.head_dim - args.qk_rope_head_dim
        self.n_groups = args.o_groups
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.window = args.sliding_window
        self.eps = args.rms_norm_eps
        self.scale = self.head_dim ** -0.5

        ratios = args.compress_ratios or []
        self.compress_ratio = ratios[layer_id] if layer_id < len(ratios) else 0

        # Fused wq_a + wkv: single matmul on the hidden state, split the output
        # into the q_a and kv halves. Saves one dispatch + one x read per layer.
        # Sanitize concats the two checkpoint tensors along axis=0 at load time.
        self.wqkv_a = nn.Linear(
            self.dim, self.q_lora_rank + self.head_dim, bias=False
        )
        self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=self.eps)

        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        group_feat = (self.n_heads * self.head_dim) // self.n_groups
        self.wo_a = nn.Linear(group_feat, self.n_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(
            self.n_groups * self.o_lora_rank, self.dim, bias=args.attention_bias
        )

        if self.compress_ratio:
            base = args.compress_rope_theta
            scaling = args.rope_scaling
        else:
            base = args.rope_theta
            scaling = None
        self.rope = DeepseekV4RoPE(self.rope_head_dim, base, scaling)

        if self.compress_ratio:
            self.compressor = Compressor(
                dim=self.dim,
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                rms_norm_eps=self.eps,
                rope=self.rope,
            )
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio, self.rope)

        # Cache for attn_sink cast (lazy; populated on first forward).
        self._sink_cache_dtype = None
        self._sink_cache = None

    def _sink_for(self, dtype) -> mx.array:
        if self._sink_cache is None or self._sink_cache_dtype is not dtype:
            self._sink_cache = self.attn_sink.astype(dtype)
            self._sink_cache_dtype = dtype
        return self._sink_cache

    def _grouped_output_projection(self, out: mx.array) -> mx.array:
        """Grouped low-rank projection via wo_a. Handles both dense and quantized.

        wo_a has shape ``[n_groups * o_lora_rank, group_feat]`` (rows = output
        features). For each group, rows ``g * o_lora_rank : (g+1) * o_lora_rank``
        and the same input columns are used.
        """
        B, S = out.shape[:2]
        group_feat = (self.n_heads * self.head_dim) // self.n_groups
        out = out.reshape(B, S, self.n_groups, group_feat)

        if isinstance(self.wo_a, nn.QuantizedLinear):
            # Batched over groups: one quantized_matmul for all groups instead
            # of a per-group Python loop (O(n_groups) → 1 kernel dispatch).
            out_g = out.transpose(2, 0, 1, 3)  # [G, B, S, group_feat]
            weight = self.wo_a.weight.reshape(self.n_groups, self.o_lora_rank, -1)[:, None]
            scales = self.wo_a.scales.reshape(self.n_groups, self.o_lora_rank, -1)[:, None]
            biases = (
                None
                if self.wo_a.biases is None
                else self.wo_a.biases.reshape(self.n_groups, self.o_lora_rank, -1)[:, None]
            )
            y = mx.quantized_matmul(
                out_g,
                weight,
                scales=scales,
                biases=biases,
                transpose=True,
                group_size=self.wo_a.group_size,
                bits=self.wo_a.bits,
                mode=self.wo_a.mode,
            )
            return y.transpose(1, 2, 0, 3).reshape(B, S, self.n_groups * self.o_lora_rank)

        wa = self.wo_a.weight.reshape(self.n_groups, self.o_lora_rank, group_feat)
        y = mx.einsum("bsgd,grd->bsgr", out, wa)
        return y.reshape(B, S, self.n_groups * self.o_lora_rank)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = x.shape
        rd = self.rope_head_dim

        # Fused: wqkv_a matmul + slice + 2 RMSNorms (q half + kv half).
        wqkv = self.wqkv_a
        if isinstance(wqkv, nn.QuantizedLinear) and wqkv.mode == "mxfp4":
            qr, kv = _attn_wqkv_quant_split_norm(
                x, wqkv.weight, wqkv.scales, wqkv.group_size, wqkv.bits,
                self.q_norm.weight, self.kv_norm.weight,
                self.q_lora_rank, self.eps,
            )
        else:
            qkv_a = wqkv(x)
            qr, kv = _attn_qkv_split_norm(
                qkv_a, self.q_norm.weight, self.kv_norm.weight,
                self.q_lora_rank, self.eps,
            )

        # Fused: wq_b matmul + reshape/transpose + per-head RMSNorm.
        wqb = self.wq_b
        if isinstance(wqb, nn.QuantizedLinear) and wqb.mode == "mxfp4":
            q = _attn_q_proj_quant_norm(
                qr, wqb.weight, wqb.scales, wqb.group_size, wqb.bits,
                self.n_heads, self.head_dim, self.eps,
            )
        else:
            q = _attn_q_proj_norm(wqb(qr), self.n_heads, self.head_dim, self.eps)

        # All layers carry a DeepseekV4Cache (per make_cache); the Compressor
        # / Indexer branches are only used for compress_ratio != 0 layers.
        # The local sliding-window cache may be RotatingKVCache (uniform-batch
        # path) or BatchRotatingKVCache (after prepare with right_padding) —
        # both expose the same update_and_fetch / offset interface.
        v4_cache: Optional[DeepseekV4Cache] = (
            cache if isinstance(cache, DeepseekV4Cache) else None
        )
        win_cache = v4_cache.local if v4_cache is not None else cache
        offset = win_cache.offset if win_cache is not None else 0
        # BatchRotatingKVCache stores offset as an mx.array and mutates it
        # in place via ``self.offset += S`` during update_and_fetch. Snapshot
        # here so our local ``offset`` reflects the pre-update position for
        # every downstream use (RoPE, Compressor, Indexer, topk helpers).
        if isinstance(offset, mx.array):
            offset = offset + 0

        # Fused: partial-RoPE on q + kv in one compiled call.
        q, kv = _attn_qkv_partial_rope(q, kv, offset, rd, self.rope.freqs)
        # exo fix: FP8 rounding of the non-RoPE KV dims, like the reference
        kv = _fp8_kv_sim(kv, rd)

        if self.compress_ratio:
            if v4_cache is None:
                v4_cache = DeepseekV4Cache(self.window)
                win_cache = v4_cache.local
            k4 = kv[:, None, :, :]
            win_keys, _ = win_cache.update_and_fetch(k4, k4)
            window_kv = win_keys.squeeze(1)
            self.compressor(x, v4_cache, offset, key=_K_COMP)
            indexer_topk = (
                self.indexer(x, qr, v4_cache, offset)
                if self.compress_ratio == 4
                else None
            )
            compressed = v4_cache.get_branch(_K_COMP).pool
            compressed_len = 0 if compressed is None else compressed.shape[1]
        else:
            k = kv[:, None, :, :]
            if win_cache is not None:
                k_ret, _ = win_cache.update_and_fetch(k, k)
                window_kv = k_ret.squeeze(1)
            else:
                window_kv = kv
            compressed = None
            compressed_len = 0
            indexer_topk = None

        window_len = window_kv.shape[1]
        # Decode (S=1) fast path: gather the Indexer's top-k compressed rows
        # directly into SDPA's key tensor so the kernel only attends to those
        # K rows + the window. For prefill (S>1) fall back to attending to the
        # full compressed buffer with a scatter-based mask — the gather-flat
        # approach would over-include across query positions, changing output.
        use_gather = (
            S == 1 and compressed_len > 0 and indexer_topk is not None
        )
        if use_gather:
            d = compressed.shape[-1]
            # take_along_axis on a broadcast of the pool is ~15% faster at b=1
            # than reshape+flat-gather (specialized fused kernel vs fancy-index).
            expanded = mx.broadcast_to(
                compressed[:, None, None, :, :], (B, 1, S, compressed_len, d)
            )
            idx = mx.broadcast_to(
                indexer_topk[:, None, :, :, None],
                (B, 1, S, indexer_topk.shape[-1], d),
            )
            gathered = mx.take_along_axis(expanded, idx, axis=3).reshape(B, -1, d)
            kv_all = mx.concatenate([window_kv, gathered], axis=1)
        elif compressed_len > 0:
            kv_all = mx.concatenate([window_kv, compressed], axis=1)
        else:
            kv_all = window_kv

        # Decode (S=1): every token in the window cache is valid past context
        # and we're not doing causal masking (only one query), so skip the
        # window mask entirely. For the gather path the Indexer already
        # returns only valid indices (no -1 padding), so no compressed mask
        # either.
        if S == 1:
            # Pool rows are emitted only after a full window of raw tokens
            # has been processed, so at any decode step every pool row is in
            # the past — no compressed-visibility mask needed.
            mask = None
        else:
            win_mask = _build_window_mask(B, S, offset, self.window, window_len)
            if compressed_len > 0:
                comp_mask = _compressed_visibility(
                    B, S, offset, compressed_len, self.compress_ratio
                )
                if indexer_topk is not None:
                    k_range = mx.arange(compressed_len, dtype=mx.int32)
                    selected = (
                        indexer_topk[..., None] == k_range[None, None, None, :]
                    ).any(axis=-2)[:, None, :, :]
                    comp_mask = comp_mask & selected
                mask = mx.concatenate([win_mask, comp_mask], axis=-1)
            else:
                mask = win_mask

        kv_all_4d = kv_all[:, None, :, :]
        o = scaled_dot_product_attention(
            q, kv_all_4d, kv_all_4d,
            cache=None, scale=self.scale, mask=mask,
            sinks=self._sink_for(q.dtype),
        )

        # Fused: inverse-RoPE on the trailing rd dims + transpose-and-reshape
        # to [B, S, n_heads*head_dim] for wo_a.
        o = _attn_inv_rope_flatten(
            o, offset, rd, self.rope.freqs, self.n_heads * self.head_dim
        )
        # Both wo_a and wo_b mxfp4 → single compile graph for the chain.
        if (
            isinstance(self.wo_a, nn.QuantizedLinear) and self.wo_a.mode == "mxfp4"
            and isinstance(self.wo_b, nn.QuantizedLinear) and self.wo_b.mode == "mxfp4"
        ):
            return _attn_wo_chain_quant(
                o,
                self.wo_a.weight, self.wo_a.scales,
                self.wo_b.weight, self.wo_b.scales,
                self.n_groups, self.o_lora_rank,
                self.wo_a.group_size, self.wo_a.bits,
                self.wo_b.group_size, self.wo_b.bits,
            )
        o = self._grouped_output_projection(o)
        return self.wo_b(o)


# --------------------------------------------------------------------------- #
# MoE                                                                          #
# --------------------------------------------------------------------------- #


def _make_moe_gate_kernel():
    """Fused MoE gate post-matmul kernel. Takes pre-matmul ``scores [B,S,N_ROUTED]``
    (bf16) and ``bias [N_ROUTED]`` (fp32) and returns ``(inds [B,S,TOP_K] int32,
    weights [B,S,TOP_K] bf16)``. Does sqrtsoftplus + bias-add + top-k partial
    sort + gather + renormalize in one dispatch.

    Scoped to the ``sqrtsoftplus`` + no-hash case (the common DSV4 path); other
    score funcs fall back to the multi-op Python path.
    """
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    src = """
        uint tid = thread_position_in_grid.x;
        uint total = B * S;
        if (tid >= total) return;

        auto s_ptr = scores + tid * N_ROUTED;
        auto i_ptr = inds + tid * TOP_K;
        auto w_ptr = weights + tid * TOP_K;
        float rscale = route_scale[0];

        float activated[N_ROUTED];
        float biased[N_ROUTED];

        for (int i = 0; i < N_ROUTED; ++i) {
            float v = static_cast<float>(s_ptr[i]);
            float sp = (v > 20.0f) ? v : metal::fast::log(1.0f + metal::fast::exp(v));
            activated[i] = metal::sqrt(sp);
            biased[i] = activated[i] + static_cast<float>(bias[i]);
        }

        float topk_vals[TOP_K];
        int topk_idx[TOP_K];
        for (int k = 0; k < TOP_K; ++k) {
            topk_vals[k] = -INFINITY;
            topk_idx[k] = 0;
        }
        for (int i = 0; i < N_ROUTED; ++i) {
            float v = biased[i];
            int min_pos = 0;
            float min_val = topk_vals[0];
            for (int k = 1; k < TOP_K; ++k) {
                if (topk_vals[k] < min_val) {
                    min_val = topk_vals[k];
                    min_pos = k;
                }
            }
            if (v > min_val) {
                topk_vals[min_pos] = v;
                topk_idx[min_pos] = i;
            }
        }

        float w[TOP_K];
        float sum = 0.0f;
        for (int k = 0; k < TOP_K; ++k) {
            w[k] = activated[topk_idx[k]];
            sum += w[k];
        }
        float scale_factor = rscale / (sum + 1e-20f);

        for (int k = 0; k < TOP_K; ++k) {
            w_ptr[k] = static_cast<OUT_T>(w[k] * scale_factor);
            i_ptr[k] = topk_idx[k];
        }
    """
    return mx.fast.metal_kernel(
        name="dsv4_moe_gate_posmm",
        input_names=["scores", "bias", "route_scale"],
        output_names=["inds", "weights"],
        source=src,
    )


_moe_gate_kernel = _make_moe_gate_kernel()


def _score_func(scores: mx.array, func: str) -> mx.array:
    if func == "softmax":
        return mx.softmax(scores, axis=-1, precise=True)
    if func == "sigmoid":
        return mx.sigmoid(scores)
    # sqrtsoftplus = sqrt(softplus(x)) = sqrt(logaddexp(x, 0))
    return mx.sqrt(mx.logaddexp(scores, 0))


@mx.compile
def _limited_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    if limit and limit > 0:
        gate = mx.minimum(gate, limit)
        up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


class _DSV4SwiGLU(nn.Module):
    """SwiGLU with optional clipping of ``gate`` / ``up`` to ``limit``, wrapped
    in ``mx.compile`` so the silu+clip+min+mul stack runs as a single fused
    kernel. Used for routed experts (limit = args.swiglu_limit = 10.0) and
    shared experts (limit = 0 — no clip, still benefits from fusion)."""

    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return _limited_swiglu(gate, x, self.limit)


class MoEGate(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_routed = args.n_routed_experts
        self.top_k = args.num_experts_per_tok
        self.hash = layer_id < args.num_hash_layers
        self.score_func = args.scoring_func
        self.route_scale = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob

        self.weight = mx.zeros((self.n_routed, args.hidden_size))
        # Pre-allocated route_scale array for the fused kernel (avoids per-call
        # mx.array allocation on the hot path).
        self._route_scale_arr = mx.array([args.routed_scaling_factor], dtype=mx.float32)
        if self.hash:
            self.tid2eid = mx.zeros(
                (args.vocab_size, self.top_k), dtype=mx.int32
            )
        else:
            self.e_score_correction_bias = mx.zeros(
                (self.n_routed,), dtype=mx.float32
            )

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None):
        if (
            _moe_gate_kernel is not None
            and not self.hash
            and self.score_func == "sqrtsoftplus"
            and self.norm_topk_prob
        ):
            scores_bf = x @ self.weight.T
            B, S, _ = x.shape
            total = B * S
            tg = 32
            grid = ((total + tg - 1) // tg) * tg
            rscale = self._route_scale_arr
            inds, weights = _moe_gate_kernel(
                inputs=[scores_bf, self.e_score_correction_bias, rscale],
                template=[
                    ("B", B),
                    ("S", S),
                    ("N_ROUTED", self.n_routed),
                    ("TOP_K", self.top_k),
                    ("OUT_T", x.dtype),
                ],
                grid=(grid, 1, 1),
                threadgroup=(tg, 1, 1),
                output_shapes=[(B, S, self.top_k), (B, S, self.top_k)],
                output_dtypes=[mx.int32, x.dtype],
            )
            return inds, weights

        # Fallback: general path (hash-routed layers, or non-sqrtsoftplus configs).
        scores = x.astype(mx.float32) @ self.weight.T.astype(mx.float32)
        scores = _score_func(scores, self.score_func)
        orig = scores
        if not self.hash:
            scores = scores + self.e_score_correction_bias
            inds = mx.stop_gradient(
                mx.argpartition(-scores, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
            )
        else:
            ids = input_ids.reshape(-1)
            inds = self.tid2eid[ids]
            inds = inds.reshape(*x.shape[:-1], self.top_k)

        weights = mx.take_along_axis(orig, inds, axis=-1)
        if self.score_func != "softmax" and self.norm_topk_prob:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        weights = (weights * self.route_scale).astype(x.dtype)
        return inds, weights


class DeepseekV4MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            _limited_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit)
        )


class DeepseekV4MoE(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        # Routed experts ship as FP4 (E2M1) with E8M0 per-32 scales — a 1:1
        # match for MLX's mxfp4. Build a dense SwitchGLU and immediately
        # quantize its three projections in-place; no bf16 intermediate.
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=_DSV4SwiGLU(args.swiglu_limit),
        )
        for name in ("gate_proj", "up_proj", "down_proj"):
            sub = getattr(self.switch_mlp, name)
            setattr(
                self.switch_mlp,
                name,
                sub.to_quantized(group_size=32, bits=4, mode="mxfp4"),
            )
        self.gate = MoEGate(args, layer_id)
        if args.n_shared_experts:
            self.shared_experts = DeepseekV4MLP(
                args.hidden_size,
                args.moe_intermediate_size * args.n_shared_experts,
                swiglu_limit=0.0,
            )

    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array:
        inds, weights = self.gate(x, input_ids)
        # Use upstream SwitchGLU.__call__ — its sort_threshold=64 path gates
        # ``sorted_indices=True`` on gather_qmm during prefill (indices.size
        # = S * top_k >> 64), which dispatches to a much faster Metal kernel
        # than the unsorted path. A custom @mx.compile fuse here bypasses
        # that and tanks prefill (Optimizations 7→8 regression).
        y = self.switch_mlp(x, inds)
        # Combine as matmul [B,S,1,top_k] @ [B,S,top_k,hidden] → [B,S,hidden].
        y = (weights[:, :, None, :] @ y).squeeze(2).astype(y.dtype)
        if hasattr(self, "shared_experts"):
            y = y + self.shared_experts(x)
        return y


# --------------------------------------------------------------------------- #
# Block                                                                        #
# --------------------------------------------------------------------------- #


class DeepseekV4Block(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attn = V4Attention(args, layer_id)
        self.hc_attn = HyperConnection(
            args.hidden_size,
            args.hc_mult,
            args.rms_norm_eps,
            args.hc_sinkhorn_iters,
            args.hc_eps,
        )
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn = DeepseekV4MoE(args, layer_id)
        self.hc_ffn = HyperConnection(
            args.hidden_size,
            args.hc_mult,
            args.rms_norm_eps,
            args.hc_sinkhorn_iters,
            args.hc_eps,
        )

    def __call__(
        self,
        h: mx.array,
        cache: Optional[Any],
        input_ids: mx.array,
    ) -> mx.array:
        # h: [B, S, hc, D]
        residual = h
        y, post, comb = self.hc_attn.hc_pre(h)
        y = self.attn_norm(y)
        y = self.attn(y, cache=cache)
        h = self.hc_attn.hc_post(y, residual, post, comb)

        residual = h
        y, post, comb = self.hc_ffn.hc_pre(h)
        y = self.ffn_norm(y)
        y = self.ffn(y, input_ids)
        h = self.hc_ffn.hc_post(y, residual, post, comb)
        return h


# --------------------------------------------------------------------------- #
# Top-level model                                                              #
# --------------------------------------------------------------------------- #


class DeepseekV4Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DeepseekV4Block(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hc_head = HyperHead(
            args.hidden_size, args.hc_mult, args.rms_norm_eps, args.hc_eps
        )

    def __call__(self, inputs: mx.array, cache: Optional[List[Any]] = None) -> mx.array:
        B, S = inputs.shape
        h = self.embed_tokens(inputs)  # [B, S, D]
        h = mx.broadcast_to(
            h[:, :, None, :],
            (B, S, self.args.hc_mult, h.shape[-1]),
        )
        h = mx.contiguous(h)

        if cache is None:
            cache = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            h = layer(h, cache[i], inputs)

        h = self.hc_head(h)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = DeepseekV4Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self, inputs: mx.array, cache: Optional[List[Any]] = None
    ) -> mx.array:
        h = self.model(inputs, cache)
        return self.lm_head(h)

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        def pred(k: str) -> bool:
            # Keep mHC parameters, attention sinks, and gate biases in fp32.
            keep_fp32 = (
                ".hc_attn." in k
                or ".hc_ffn." in k
                or ".hc_head." in k
                or "e_score_correction_bias" in k
                or "attn_sink" in k
            )
            return not keep_fp32
        return pred

    def make_cache(self):
        return [
            DeepseekV4Cache(self.args.sliding_window) for _ in self.layers
        ]

    # --------------------------------------------------------------------- #
    # Weight loading                                                        #
    # --------------------------------------------------------------------- #

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Follow the DeepSeek-V3.2 sanitize structure (MTP drop -> FP8 dequant
        -> expert stacking) with three DSV4-specific additions:

            1. ``F8_E8M0`` scale tensors arrive as ``uint8`` (see utils.py
               loader shim) and are decoded as ``2^(b-127)`` before use.
            2. Routed experts in Flash are FP4-packed ``uint8``; identify via
               ``scale.shape[-1] * 16 == weight.shape[-1]`` and dequant with a
               nibble table.
            3. DSV4 uses a different checkpoint key scheme (``embed`` /
               ``head`` / ``attn.*`` / ``ffn.*`` / ``hc_{attn,ffn}_{fn,base,scale}``
               / ``gate.bias``); remap into mlx-lm ``model.*`` space.
        """
        n_layers = self.args.num_hidden_layers

        # 1) Drop MTP blocks and layers past num_hidden_layers
        filtered = {}
        for k, v in weights.items():
            if k.startswith("mtp."):
                continue
            parts = k.split(".")
            if len(parts) >= 2 and parts[0] == "layers":
                try:
                    idx = int(parts[1])
                except ValueError:
                    filtered[k] = v
                    continue
                if idx >= n_layers:
                    continue
            filtered[k] = v
        weights = filtered

        def _scale_to_float(scale: mx.array) -> mx.array:
            if scale.dtype == mx.uint8:
                return mx.exp((scale.astype(mx.float32) - 127.0) * math.log(2.0))
            return scale.astype(mx.float32)

        def dequant_fp8_block(weight: mx.array, scale: mx.array) -> mx.array:
            # DSV3.2-style block dequant, plus F8_E8M0 scale decode.
            bs = 128
            w = mx.from_fp8(weight, dtype=mx.bfloat16)
            s = _scale_to_float(scale)
            m, n = w.shape
            pad_b = (-m) % bs
            pad_s = (-n) % bs
            w = mx.pad(w, ((0, pad_b), (0, pad_s)))
            w = w.reshape((m + pad_b) // bs, bs, (n + pad_s) // bs, bs)
            w = (w * s[:, None, :, None]).reshape(m + pad_b, n + pad_s)
            return w[:m, :n].astype(mx.bfloat16)

        # 2) Handle each `.scale` sibling entry:
        #    - Routed-expert FP4 (uint8 [out, in//2]) + E8M0 (uint8 [out, in//32])
        #      → reinterpret as MLX mxfp4 (uint32 [out, in//8] + uint8 scale).
        #      The bit layout matches: element 2i in low nibble, 2i+1 in high;
        #      MLX's mxfp4 packs 8 fp4 values per uint32 in little-endian, so a
        #      run of 4 uint8 bytes reads as the right uint32. No allocation.
        #    - FP8 (uint8 [M, N]) + E8M0 128x128 block scale → bf16 dequant.
        dequanted = {}
        for k, v in weights.items():
            if not k.endswith(".scale"):
                if k not in dequanted:
                    dequanted[k] = v
                continue
            wk = k[: -len(".scale")] + ".weight"
            weight = weights.get(wk)
            if weight is None:
                dequanted[k] = v
                continue
            is_routed_expert = (
                ".ffn.experts." in wk
                and "shared_experts" not in wk
                and weight.dtype in (mx.int8, mx.uint8)
                and v.shape[-1] * 16 == weight.shape[-1]
            )
            if is_routed_expert:
                packed = weight.astype(mx.uint8)
                dequanted[wk] = packed.view(mx.uint32).reshape(
                    packed.shape[0], packed.shape[-1] // 4
                )
                dequanted[k] = v.astype(mx.uint8)
            elif weight.dtype in (mx.uint8,):
                dequanted[wk] = dequant_fp8_block(weight, v)
            else:
                dequanted[k] = v
                dequanted[wk] = weight
        weights = dequanted

        # 3) Remap keys to mlx-lm layout
        top_remap = {
            "embed.weight":   "model.embed_tokens.weight",
            "norm.weight":    "model.norm.weight",
            "head.weight":    "lm_head.weight",
            "hc_head_fn":     "model.hc_head.fn",
            "hc_head_base":   "model.hc_head.base",
            "hc_head_scale":  "model.hc_head.scale",
        }
        for src, dst in top_remap.items():
            if src in weights:
                weights[dst] = weights.pop(src)

        remapped = {}
        w_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for k, v in weights.items():
            nk = k
            if nk.startswith("layers."):
                nk = "model." + nk
            nk = nk.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
            for sub in ("attn", "ffn"):
                for p in ("fn", "base", "scale"):
                    nk = nk.replace(f".hc_{sub}_{p}", f".hc_{sub}.{p}")
            for wo, wn in w_remap.items():
                nk = nk.replace(f".shared_experts.{wo}.", f".shared_experts.{wn}.")
            remapped[nk] = v
        weights = remapped

        # 3b) Fuse wq_a + wkv → wqkv_a, and Compressor wkv + wgate → wkv_gate.
        # The model only declares the fused Linear, so the two checkpoint
        # tensors must be concatenated along the output dim at load time.
        def _fuse_pair(keys, out_key):
            for sfx in ("weight", "scales", "biases"):
                parts = [f"{k}.{sfx}" for k in keys]
                if all(p in weights for p in parts):
                    weights[f"{out_key}.{sfx}"] = mx.concatenate(
                        [weights.pop(p) for p in parts], axis=0
                    )

        for l in range(n_layers):
            attn = f"model.layers.{l}.attn"
            _fuse_pair([f"{attn}.wq_a", f"{attn}.wkv"], f"{attn}.wqkv_a")
            for parent in (f"{attn}.compressor", f"{attn}.indexer.compressor"):
                _fuse_pair([f"{parent}.wkv", f"{parent}.wgate"], f"{parent}.wkv_gate")

        # 4) Stack per-expert {weight, scales} into the SwitchGLU layout.
        for l in range(n_layers):
            prefix = f"model.layers.{l}.ffn.experts"
            for src, dst in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                key0 = f"{prefix}.0.{src}.weight"
                if key0 in weights:
                    stack = [
                        weights.pop(f"{prefix}.{e}.{src}.weight")
                        for e in range(self.args.n_routed_experts)
                    ]
                    weights[f"model.layers.{l}.ffn.switch_mlp.{dst}.weight"] = (
                        mx.stack(stack)
                    )
                skey0 = f"{prefix}.0.{src}.scale"
                if skey0 in weights:
                    sstack = [
                        weights.pop(f"{prefix}.{e}.{src}.scale")
                        for e in range(self.args.n_routed_experts)
                    ]
                    weights[
                        f"model.layers.{l}.ffn.switch_mlp.{dst}.scales"
                    ] = mx.stack(sstack)

        return weights
