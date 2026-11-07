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
#         model was trained with (_fp8_kv_sim, _fp4_indexer_sim);
#   - a work-around for an MLX bug: scaled_dot_product_attention with sinks returns
#     wrong values for head_dim 256/512 once batch * heads * queries * keys passes
#     2**31, so V4Attention splits the queries of large prefill steps (_SDPA_MAX_SCORES);
#   - fixes for batching rows of different lengths (merged / extended caches), each
#     marked in place: one-token steps mask the padded window slots and compressed rows
#     (_decode_padding_mask), multi-token steps the padded window slots; the overlapping
#     compressor keeps each row's own last window (_carry_last_window); merge / extend /
#     extract keep every row's carry and pool length, also for the fixed decode buffer
#     (_buffer_row_lengths, _set_row_lengths), and a missing overlap gate is -inf.
#   - performance changes that keep those numerics, each marked "exo perf" in place: the
#     FP8 projections held as mxfp8 (Fp8BlockLinear), prefill attention over the keys
#     each query can see (_prefill_attention), decode attention kernels, the rounding,
#     hyper-connection and indexer-scoring kernels, and whole-head RoPE.
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

    # exo perf: the official checkpoints' FP8 format (see Fp8BlockLinear)
    quantization_config: Optional[Dict] = None

    @property
    def fp8_block_weights(self) -> bool:
        """True for DeepSeek's own FP8 checkpoints: e4m3 weights with power-of-two
        (ue8m0) scales over 128x128 blocks. Their dense projections are kept in that
        format (as MLX mxfp8) instead of being expanded to bf16."""
        qc = self.quantization_config or {}
        return (
            qc.get("quant_method") == "fp8"
            and qc.get("fmt", "e4m3") == "e4m3"
            and qc.get("scale_fmt") == "ue8m0"
            and list(qc.get("weight_block_size") or ()) == [128, 128]
        )


# --------------------------------------------------------------------------- #
# exo perf: FP8 dense projections kept in their checkpoint format             #
# --------------------------------------------------------------------------- #
# DeepSeek's checkpoints store the dense projections as FP8 (e4m3) with one
# power-of-two (ue8m0) scale per 128x128 block. Repeating each block scale over
# its 32-wide groups makes that exactly an MLX mxfp8 weight (e4m3 values, one
# ue8m0 scale per 32): the dequantized values are bit-identical to the bf16
# expansion. Decode reads half the bytes; prefill dequantizes to bf16 first,
# which is faster than the quantized matmul there and gives exactly the bf16
# GEMM's result.

# At or above this many input rows, dequantize and run the bf16 matmul.
_FP8_DEQUANT_MIN_ROWS = 512


def _fp8_rows(x: mx.array) -> int:
    return x.size // x.shape[-1]


class Fp8BlockLinear(nn.QuantizedLinear):
    def __init__(self, input_dims: int, output_dims: int, bias: bool = False):
        nn.Module.__init__(self)
        self.group_size, self.bits, self.mode = 32, 8, "mxfp8"
        self.weight = mx.zeros((output_dims, input_dims // 4), dtype=mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // 32), dtype=mx.uint8)
        self.biases = None
        if bias:
            self.bias = mx.zeros((output_dims,))
        self.freeze()

    def dense_weight(self, dtype) -> mx.array:
        return mx.dequantize(
            self["weight"], self["scales"], group_size=32, bits=8, mode="mxfp8",
            dtype=dtype,
        )

    def __call__(self, x: mx.array) -> mx.array:
        if _fp8_rows(x) < _FP8_DEQUANT_MIN_ROWS:
            return super().__call__(x)
        y = x @ self.dense_weight(x.dtype).T
        if "bias" in self:
            y = y + self["bias"]
        return y


def _dense_linear(args: "ModelArgs", input_dims: int, output_dims: int, bias: bool = False):
    """A projection that the checkpoint stores in FP8: Fp8BlockLinear for DeepSeek's own
    FP8 checkpoints (when the dims fit mxfp8 groups), nn.Linear otherwise."""
    if args.fp8_block_weights and input_dims % 32 == 0:
        return Fp8BlockLinear(input_dims, output_dims, bias=bias)
    return nn.Linear(input_dims, output_dims, bias=bias)


def _ue8m0_bytes(name: str, scale: mx.array) -> mx.array:
    """A checkpoint's ue8m0 block scales as their exponent bytes. MLX loads F8_E8M0 as
    uint8 (the bytes themselves). Float scales must be exact powers of two; a float32
    power of two's exponent field is its ue8m0 byte (both biased by 127)."""
    if scale.dtype == mx.uint8:
        return scale
    bits = scale.astype(mx.float32).view(mx.int32)
    exponent = mx.bitwise_and(mx.right_shift(bits, 23), 0xFF)
    exact = mx.all((mx.bitwise_and(bits, (1 << 23) - 1) == 0) & (exponent > 0) & (bits > 0))
    if not exact.item():
        raise ValueError(f"{name}: an FP8 block scale is not a power of two (scale_fmt ue8m0)")
    return exponent.astype(mx.uint8)


def _fp8_block_to_mxfp8(weight: mx.array, scale: mx.array):
    """FP8 weight [M, N] (uint8 e4m3 bytes) and ue8m0 block scales [ceil(M/128),
    ceil(N/128)] (uint8) -> mxfp8 (packed uint32 [M, N/4], scales uint8 [M, N/32])."""
    m, n = weight.shape
    packed = weight.astype(mx.uint8).view(mx.uint32).reshape(m, n // 4)
    scales = mx.repeat(mx.repeat(scale.astype(mx.uint8), 128, axis=0)[:m], 4, axis=1)
    return packed, scales[:, : n // 32]


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

    def head_freqs(self, head_dim: int) -> mx.array:
        """exo perf: periods for rotating a whole ``head_dim`` vector whose last ``dims``
        are the RoPE dims: the leading pairs get an infinite period (angle 0: cos 1, sin 0),
        so they pass through unchanged. One rope kernel instead of slice + rope +
        concatenate; the output is bit-identical."""
        cache = self.__dict__.setdefault("_head_freqs", {})
        if head_dim not in cache:
            lead = mx.full(((head_dim - self.dims) // 2,), math.inf, dtype=mx.float32)
            cache[head_dim] = mx.concatenate([lead, self.freqs])
            mx.eval(cache[head_dim])
        return cache[head_dim]

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


# exo perf: two kernels for the hyper-connections around every attention and MoE block.
#   - hc mix (decode-sized calls): mixes = rms_norm(x, fp32) @ fn.T as one kernel, one
#     threadgroup per (row, mix): the sum of squares and the dot product in one pass,
#     then dot * rsqrt(mean + eps). It replaces a cast, an RMSNorm and a matmul; the
#     float32 result differs only by summation order.
#   - hc post: y[j] = post[j] * f + sum_i comb[i, j] * residual[i] in float32, one
#     kernel instead of a cast, a matmul and an elementwise kernel (and no float32
#     copy of the residual in prefill).
_HC_MIX_MAX_ROWS = 32


def _make_hc_mix_kernel():
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    source = """
        uint tid = thread_position_in_threadgroup.x;
        uint tg = threadgroup_position_in_grid.x;
        uint lane = tid % 32;
        uint sg = tid / 32;
        uint j = tg % MIX;
        uint row = tg / MIX;
        using bf4 = vec<bfloat16_t, 4>;
        const device bf4* xr = (const device bf4*)(x + row * N);
        const device float4* w = (const device float4*)(fn + j * N);
        float ss = 0.0f;
        float dot = 0.0f;
        for (uint i = tid; i < N / 4; i += 256) {
            float4 v = float4(xr[i]);
            ss += metal::dot(v, v);
            dot += metal::dot(v, w[i]);
        }
        ss = simd_sum(ss);
        dot = simd_sum(dot);
        threadgroup float s_ss[8];
        threadgroup float s_dot[8];
        if (lane == 0) {
            s_ss[sg] = ss;
            s_dot[sg] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float t_ss = 0.0f;
            float t_dot = 0.0f;
            for (uint k = 0; k < 8; ++k) {
                t_ss += s_ss[k];
                t_dot += s_dot[k];
            }
            mixes[row * MIX + j] = t_dot * metal::rsqrt(t_ss / float(N) + eps[0]);
        }
    """
    return mx.fast.metal_kernel(
        name="dsv4_hc_mix",
        input_names=["x", "fn", "eps"],
        output_names=["mixes"],
        source=source,
    )


def _make_hc_post_kernel():
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    source = """
        uint gid = thread_position_in_grid.x;
        constexpr uint D4 = D / 4;
        uint d4 = gid % D4;
        uint rj = gid / D4;
        uint j = rj % 4;
        uint row = rj / 4;
        using bf4 = vec<bfloat16_t, 4>;
        const device bf4* f4 = (const device bf4*)f_out;
        const device bf4* r4 = (const device bf4*)residual;
        device bf4* o4 = (device bf4*)out;
        const device float* c = comb + row * 16;
        float4 y = post[row * 4 + j] * float4(f4[row * D4 + d4]);
        float4 acc = c[0 * 4 + j] * float4(r4[(row * 4 + 0) * D4 + d4]);
        acc += c[1 * 4 + j] * float4(r4[(row * 4 + 1) * D4 + d4]);
        acc += c[2 * 4 + j] * float4(r4[(row * 4 + 2) * D4 + d4]);
        acc += c[3 * 4 + j] * float4(r4[(row * 4 + 3) * D4 + d4]);
        o4[(row * 4 + j) * D4 + d4] = bf4(y + acc);
    """
    return mx.fast.metal_kernel(
        name="dsv4_hc_post",
        input_names=["f_out", "residual", "post", "comb"],
        output_names=["out"],
        source=source,
    )


_hc_mix_kernel = _make_hc_mix_kernel()
_hc_post_kernel = _make_hc_post_kernel()


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
        self._rms_eps_arr = mx.array([norm_eps], dtype=mx.float32)

    def hc_pre(self, x: mx.array):
        # x: [B, S, hc, D]  ->  (y [B, S, D], post [B, S, hc], comb [B, S, hc, hc])
        B, S, hc, D = x.shape
        if (
            _hc_mix_kernel is not None
            and B * S <= _HC_MIX_MAX_ROWS
            and x.dtype == mx.bfloat16
            and (hc * D) % 1024 == 0
        ):
            mix = self.fn.shape[0]
            mixes = _hc_mix_kernel(
                inputs=[x, self.fn, self._rms_eps_arr],
                template=[("N", hc * D), ("MIX", mix)],
                grid=(B * S * mix * 256, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(B, S, mix)],
                output_dtypes=[mx.float32],
            )[0]
        else:
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
        if (
            _hc_post_kernel is not None
            and residual.shape[-2] == 4
            and f_out.dtype == mx.bfloat16
            and residual.dtype == mx.bfloat16
            and f_out.shape[-1] % 4 == 0
        ):
            B, S, hc, D = residual.shape
            return _hc_post_kernel(
                inputs=[f_out, residual, post, comb],
                template=[("D", D)],
                grid=(B * S * hc * (D // 4), 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[residual.shape],
                output_dtypes=[f_out.dtype],
            )[0]
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


def _missing_state_fill(name: str) -> float:
    """exo fix: the value that stands for a row's missing state when caches are batched.
    A row that has not emitted a window yet has no overlap carry; its gate must be -inf
    (no weight), like the compressor's own default, not 0."""
    return float("-inf") if name == "prev_gate" else 0.0


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


def _buffer_row_lengths(branch: _CompressorBranch, batch: int) -> List[int]:
    """Valid carry tokens per row of ``branch.buffer_kv``. exo fix: a uniform branch holds
    ``buffer_count`` of them, also when decode keeps the carry in a fixed ``[B, ratio]``
    buffer (its length is then not the count)."""
    if branch.buffer_lengths is not None:
        return _as_lengths_list(branch.buffer_lengths, batch)
    if branch.buffer_kv is None:
        return [0] * batch
    return [branch.buffer_count] * batch


def _pool_row_lengths(branch: _CompressorBranch, batch: int) -> List[int]:
    if branch.pool_lengths is not None:
        return _as_lengths_list(branch.pool_lengths, batch)
    return [0 if branch.pool is None else branch.pool.shape[1]] * batch


def _set_row_lengths(
    branch: _CompressorBranch, buffer_lengths: List[int], pool_lengths: List[int]
) -> None:
    """Record per-row lengths after batching states; when every row has the same length the
    branch goes back to the uniform state (buffers cut to that length, ``buffer_count`` set)."""
    n = max(buffer_lengths, default=0)
    if all(b == n for b in buffer_lengths):
        branch.buffer_lengths = None
        branch.buffer_count = n
        if n == 0 or branch.buffer_kv is None or branch.buffer_gate is None:
            branch.buffer_kv = None
            branch.buffer_gate = None
        else:
            branch.buffer_kv = branch.buffer_kv[:, :n]
            branch.buffer_gate = branch.buffer_gate[:, :n]
    else:
        branch.buffer_lengths = list(buffer_lengths)
    n = max(pool_lengths, default=0)
    if all(p == n for p in pool_lengths):
        branch.pool_lengths = None
        if branch.pool is not None:
            branch.pool = branch.pool[:, :n] if n else None
    else:
        branch.pool_lengths = list(pool_lengths)


def _buffer_length(local) -> int:
    return 0 if local.keys is None else local.keys.shape[2]


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
            # exo fix: the same position can sit in buffers of different lengths (a one-token
            # step grows the buffer ahead of the position); those take the batch cache below
            and _buffer_length(self.local) == _buffer_length(other.local)
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
            buffer_lengths = _buffer_row_lengths(b_self, a_batch) + _buffer_row_lengths(
                b_other, b_batch
            )
            pool_lengths = _pool_row_lengths(b_self, a_batch) + _pool_row_lengths(
                b_other, b_batch
            )
            for name in ("buffer_kv", "buffer_gate", "prev_kv", "prev_gate", "pool"):
                merged = self._concat_batch_state(
                    getattr(b_self, name),
                    getattr(b_other, name),
                    a_batch,
                    b_batch,
                    _missing_state_fill(name),
                )
                setattr(b_self, name, merged)
            _set_row_lengths(b_self, buffer_lengths, pool_lengths)

    def extract(self, idx):
        cache = DeepseekV4Cache(self.local.max_size)
        cache.local = (
            self.local.extract(idx)
            if hasattr(self.local, "extract")
            else self._extract_local(self.local, idx)
        )
        batch = self._cache_batch_size(self.local)
        for key, src in self._branches.items():
            dst = cache._branches[key]
            for name in ("buffer_kv", "buffer_gate", "prev_kv", "prev_gate", "pool"):
                v = getattr(src, name)
                setattr(dst, name, None if v is None else mx.contiguous(v[idx : idx + 1]))
            _set_row_lengths(
                dst,
                [_buffer_row_lengths(src, batch)[idx]],
                [_pool_row_lengths(src, batch)[idx]],
            )
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
                setattr(
                    dst, name, cls._merge_batch_state(tensors, _missing_state_fill(name))
                )
            _set_row_lengths(
                dst,
                [
                    n
                    for c in caches
                    for n in _buffer_row_lengths(
                        c._branches[key], cls._cache_batch_size(c.local)
                    )
                ],
                [
                    n
                    for c in caches
                    for n in _pool_row_lengths(
                        c._branches[key], cls._cache_batch_size(c.local)
                    )
                ],
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
    def _concat_batch_state(a, b, a_batch, b_batch, fill: float = 0.0):
        if a is None and b is None:
            return None
        if a is None:
            a = mx.full((a_batch,) + b.shape[1:], fill, dtype=b.dtype)
        if b is None:
            b = mx.full((b_batch,) + a.shape[1:], fill, dtype=a.dtype)
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
    def _merge_batch_state(values, fill: float = 0.0):
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
                merged.append(mx.full((1, seq_len, *shape[2:]), fill, dtype=dtype))
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
    """Fused: RMSNorm + strided rope on the last rd dims (``freqs`` covers the whole
    row, see DeepseekV4RoPE.head_freqs)."""
    c = mx.fast.rms_norm(c, norm_w, norm_eps)
    return mx.fast.rope(
        c, c.shape[-1], traditional=True, base=None,
        scale=scale, offset=offset, freqs=freqs,
    )


def _carry_last_window(
    branch: "_CompressorBranch", kv_win: mx.array, score_win: mx.array
) -> None:
    """Keep each row's last emitted window ``[B, ratio, coff*d]`` for the next window's
    overlap. exo fix: in a batch whose rows have different lengths
    (``branch._new_pool_lengths`` set by ``accumulate_windows``), a row emits fewer windows
    than ``kv_win`` holds, or none: its carry is its own last window, or stays as it was."""
    counts = branch._new_pool_lengths
    if counts is None:
        branch.prev_kv = kv_win[:, -1, :, :]
        branch.prev_gate = score_win[:, -1, :, :]
        return
    last = mx.array([max(n - 1, 0) for n in counts], dtype=mx.int32)[:, None, None, None]
    emitted = mx.array([n > 0 for n in counts])[:, None, None]
    new_kv = mx.take_along_axis(kv_win, last, axis=1)[:, 0]
    new_gate = mx.take_along_axis(score_win, last, axis=1)[:, 0]
    old_kv = branch.prev_kv
    old_gate = branch.prev_gate
    if old_kv is None or old_gate is None:
        old_kv = mx.zeros_like(new_kv)
        old_gate = mx.full(new_gate.shape, float("-inf"), dtype=new_gate.dtype)
    branch.prev_kv = mx.where(emitted, new_kv, old_kv)
    branch.prev_gate = mx.where(emitted, new_gate, old_gate)


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
            _carry_last_window(branch, kv_win, score_win)
        elif self.overlap:
            prev_kv = branch.prev_kv
            prev_score = branch.prev_gate
            kv_trans = self._overlap_transform_kv_runtime(kv_win, prev_kv)
            score_trans = self._overlap_transform_score_runtime(score_win, prev_score)
            weights = mx.softmax(
                score_trans.astype(mx.float32), axis=2, precise=True
            ).astype(kv_trans.dtype)
            new_pooled = (kv_trans * weights).sum(axis=2)
            _carry_last_window(branch, kv_win, score_win)
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
                self.rope.head_freqs(d),
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


@mx.compile
def _fp8_kv_sim_ops(kv: mx.array, rd: int) -> mx.array:
    return mx.concatenate([_block_round(kv[..., :-rd], 64, False), kv[..., -rd:]], axis=-1)


# exo perf: _block_round as one Metal kernel (the ops above are ~10 kernels even when
# compiled, and decode runs them on a few hundred values per layer). One simdgroup per
# block; the same float32 steps as _block_round, so the output is bit-identical (the
# tests compare the two, and e4m3 rounding against mx.to_fp8 on every bf16 value).
def _make_block_round_kernel():
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    source = """
        uint gid = thread_position_in_grid.x;
        uint lane = gid % 32;
        uint blk = gid / 32;
        constexpr uint NB = (N + BLOCK - 1) / BLOCK;
        constexpr uint EPL = BLOCK / 32;
        // _block_round's constants: qmax, and amin (1e-4, or 6 * 2**-126 for FP4)
        const float QMAX = FP4 ? 6.0f : 448.0f;
        const float AMIN = FP4 ? as_type<float>(0x01C00000) : 1e-4f;
        uint row = blk / NB;
        uint col0 = (blk % NB) * BLOCK;
        uint base = row * N + col0;
        if (col0 >= NR) {
            for (uint i = 0; i < EPL; ++i) {
                uint c = col0 + lane + 32 * i;
                if (c < N) {
                    out[row * N + c] = static_cast<OUT_T>(inp[row * N + c]);
                }
            }
            return;
        }
        float v[EPL];
        float amax = 0.0f;
        for (uint i = 0; i < EPL; ++i) {
            float x = static_cast<float>(inp[base + lane + 32 * i]);
            if (VIA_BF16) {
                x = static_cast<float>(static_cast<bfloat16_t>(x));
            }
            v[i] = x;
            amax = metal::max(amax, metal::abs(x));
        }
        amax = simd_max(amax);
        amax = metal::max(amax, AMIN);
        float t = amax * (1.0f / QMAX);
        int tb = as_type<int>(t);
        int e = ((tb >> 23) & 0xFF) - 127 + (((tb & 0x7FFFFF) != 0) ? 1 : 0);
        float s = as_type<float>((e + 127) << 23);
        float inv_s = as_type<float>((127 - e) << 23);
        for (uint i = 0; i < EPL; ++i) {
            float y = metal::clamp(v[i] * inv_s, -QMAX, QMAX);
            float a = metal::abs(y);
            float q;
            if (FP4) {
                q = a <= 0.25f ? 0.0f : a < 0.75f ? 0.5f : a <= 1.25f ? 1.0f
                  : a < 1.75f ? 1.5f : a <= 2.5f ? 2.0f : a < 3.5f ? 3.0f
                  : a <= 5.0f ? 4.0f : 6.0f;
                y = y < 0.0f ? -q : q;
            } else {
                int ea = metal::max(((as_type<int>(a) >> 23) & 0xFF) - 127, -6);
                float ulp = as_type<float>((ea - 3 + 127) << 23);
                float inv_ulp = as_type<float>((127 - (ea - 3)) << 23);
                q = metal::min(metal::rint(a * inv_ulp) * ulp, 448.0f);
                y = metal::copysign(q, y);
            }
            out[base + lane + 32 * i] = static_cast<OUT_T>(y * s);
        }
    """
    return mx.fast.metal_kernel(
        name="dsv4_block_round",
        input_names=["inp"],
        output_names=["out"],
        source=source,
    )


_block_round_kernel = _make_block_round_kernel()


def _block_round_fused(
    x: mx.array, block: int, fp4: bool, rounded: int, out_dtype, via_bf16: bool = False
) -> mx.array:
    """_block_round over the first `rounded` values of each row of x (the rest copied),
    as one kernel. x may be float32 when via_bf16 (it is rounded to bf16 first)."""
    n = x.shape[-1]
    rows = x.size // n
    nb = (n + block - 1) // block
    return _block_round_kernel(
        inputs=[x],
        template=[
            ("N", n), ("NR", rounded), ("BLOCK", block), ("FP4", fp4),
            ("VIA_BF16", via_bf16), ("OUT_T", out_dtype),
        ],
        grid=(rows * nb * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[out_dtype],
    )[0]


def _fp8_kv_sim(kv: mx.array, rd: int) -> mx.array:
    n = kv.shape[-1]
    if _block_round_kernel is None or (n - rd) % 64 or rd % 64:
        return _fp8_kv_sim_ops(kv, rd)
    return _block_round_fused(kv, 64, False, n - rd, kv.dtype)


_HADAMARD: Dict[int, mx.array] = {}


def _hadamard(n: int) -> mx.array:
    if n not in _HADAMARD:
        h = mx.array([[1.0]])
        while h.shape[0] < n:
            h = mx.concatenate([mx.concatenate([h, h], 1), mx.concatenate([h, -h], 1)], 0)
        _HADAMARD[n] = h * (n**-0.5)
        mx.eval(_HADAMARD[n])
    return _HADAMARD[n]


@mx.compile
def _fp4_rotate_round(x: mx.array, hadamard: mx.array) -> mx.array:
    rotated = (x.astype(mx.float32) @ hadamard.T).astype(x.dtype)
    return _block_round(rotated, 32, True)


def _fp4_indexer_sim(x: mx.array) -> mx.array:
    n = x.shape[-1]
    if _block_round_kernel is None or n % 32 or x.dtype != mx.bfloat16:
        return _fp4_rotate_round(x, _hadamard(n))
    rotated = x.astype(mx.float32) @ _hadamard(n).T
    return _block_round_fused(rotated, 32, True, n, x.dtype, via_bf16=True)


# --------------------------------------------------------------------------- #
# Indexer                                                                     #
# --------------------------------------------------------------------------- #


# exo perf: the indexer's decode scoring (q . k per head, bf16, ReLU, weighted sum over
# heads) as one kernel instead of a matmul, a ReLU and a second matmul. One simdgroup
# per (row, compressed row); each lane scores n_heads/32 heads in float32, rounds each
# head's score to bf16 like the matmul did, and the weighted sum is accumulated in
# float32 and rounded to bf16 once. Only the summation order differs from the matmuls.
def _make_indexer_score_kernel():
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    source = """
        // the row count comes from the pool's shape, not a template parameter: it grows
        // every few decode steps and each new value would compile a new kernel
        uint B = k_shape[0];
        uint T = k_shape[1];
        uint gid = thread_position_in_grid.x;
        uint lane = gid % 32;
        uint key = gid / 32;
        uint b = key / T;
        if (b >= B) return;
        uint t = key % T;
        constexpr uint HPL = H / 32;
        constexpr uint D4 = D / 4;
        using bf4 = vec<bfloat16_t, 4>;
        const device bf4* k4 = (const device bf4*)(k + (b * T + t) * D);
        float acc = 0.0f;
        for (uint j = 0; j < HPL; ++j) {
            uint h = lane + 32 * j;
            const device bf4* q4 = (const device bf4*)(q + (b * H + h) * D);
            float dot = 0.0f;
            for (uint d = 0; d < D4; ++d) {
                dot += metal::dot(float4(q4[d]), float4(k4[d]));
            }
            float s = static_cast<float>(static_cast<bfloat16_t>(dot));
            s = metal::max(s, 0.0f);
            acc += static_cast<float>(w[b * H + h]) * s;
        }
        acc = simd_sum(acc);
        if (lane == 0) {
            out[b * T + t] = static_cast<bfloat16_t>(acc);
        }
    """
    return mx.fast.metal_kernel(
        name="dsv4_indexer_score",
        input_names=["q", "k", "w"],
        output_names=["out"],
        source=source,
    )


_indexer_score_kernel = _make_indexer_score_kernel()


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
        self.wq_b = _dense_linear(args, args.q_lora_rank, self.n_heads * self.head_dim)
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
        q = mx.fast.rope(
            q, self.head_dim, traditional=True, base=None, scale=1.0,
            offset=offset, freqs=self.rope.head_freqs(self.head_dim),
        )
        # exo fix: Hadamard rotation and FP4 rounding of the query, like the reference
        q = _fp4_indexer_sim(q.transpose(0, 2, 1, 3))
        # exo perf: the weights carry a minus sign, so the head reduction yields -score
        # (bit for bit: IEEE negation commutes with every rounding step) and the top-k
        # needs no separate negation kernel before argpartition.
        per_head_weights = self.weights_proj(x) * (
            -(self.softmax_scale * (self.n_heads ** -0.5))
        )
        T = idx_kv.shape[1]
        k = min(self.index_topk, T)
        q = q.astype(idx_kv.dtype)
        pool_lengths = cache.pooled_lengths(_K_IDX)
        # exo perf: score long prefill steps a block of queries at a time, so the
        # [queries, heads, rows] intermediate stays bounded (it is 2 GB per 4096-token
        # chunk at 16k tokens of context and grows with the context).
        q_block = max(1, _INDEXER_MAX_SCORES // max(1, B * self.n_heads * T))
        if S <= q_block:
            return self._top_rows(q, per_head_weights, idx_kv, pool_lengths, offset, k)
        return mx.concatenate(
            [
                self._top_rows(
                    q[:, s0 : s0 + q_block], per_head_weights[:, s0 : s0 + q_block],
                    idx_kv, pool_lengths, offset + s0, k, several_queries=True,
                )
                for s0 in range(0, S, q_block)
            ],
            axis=1,
        )

    def _top_rows(
        self, q, per_head_weights, idx_kv, pool_lengths, offset, k, several_queries=False
    ):
        B, S, H, Dh = q.shape
        T = idx_kv.shape[1]
        if (
            S == 1
            and _indexer_score_kernel is not None
            and H % 32 == 0
            and Dh % 4 == 0
            and q.dtype == mx.bfloat16
            and idx_kv.dtype == mx.bfloat16
            and per_head_weights.dtype == mx.bfloat16
        ):
            neg_score = _indexer_score_kernel(
                inputs=[q, idx_kv, per_head_weights],
                template=[("H", H), ("D", Dh)],
                grid=(B * T * 32, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(B, 1, T)],
                output_dtypes=[mx.bfloat16],
            )[0]
        else:
            # exo perf: the einsum as one matmul with heads folded into the rows: the same
            # GEMM (bit-identical scores) but a contiguous result, which halves the ReLU's time
            score = (q.reshape(B, S * H, Dh) @ idx_kv.swapaxes(-1, -2)).reshape(B, S, H, T)
            score = mx.maximum(score, 0)  # ReLU
            # Reduce over n_heads as a matmul: [B,S,1,n_heads] @ [B,S,n_heads,T] → [B,S,T]
            # (negated weights: this is -score, see Indexer.__call__)
            neg_score = (per_head_weights[:, :, None, :] @ score).squeeze(2)

        # Per-row pool length masking — zero scores for invalid pool slots so
        # topk doesn't pick padding zeros from short rows.
        if pool_lengths is not None:
            lengths_a = mx.array(pool_lengths, dtype=mx.int32)
            valid = (
                mx.arange(idx_kv.shape[1], dtype=mx.int32)[None, None, :]
                < lengths_a[:, None, None]
            )
            neg_score = mx.where(valid, neg_score, mx.array(1e30, dtype=neg_score.dtype))

        # exo fix: in multi-token steps (prefill), rows that complete later in the step are
        # not visible to earlier queries. Exclude them before top-k, like the
        # reference, so each query keeps index_topk rows it can attend to.
        if S > 1 or several_queries:
            visible = _compressed_visibility(
                B, S, offset, idx_kv.shape[1], self.compress_ratio
            )[:, 0]
            neg_score = mx.where(
                visible, neg_score, mx.array(math.inf, dtype=neg_score.dtype)
            )

        # exo perf: argpartition's uint32 indices are used as they are
        return mx.argpartition(neg_score, kth=k - 1, axis=-1)[..., :k]


# --------------------------------------------------------------------------- #
# Attention                                                                   #
# --------------------------------------------------------------------------- #

# exo fix: the most attention scores (batch * heads * queries * keys) one sinks SDPA
# call computes; larger prefill steps are split by queries (see V4Attention.__call__).
_SDPA_MAX_SCORES = 2**30


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
        # exo fix: a row shorter than the others is left-padded; its padding slots get
        # negative positions and hold no token.
        win_visible = (
            (raw_pos_at_k[:, None, :] <= q_pos[:, :, None])
            & (raw_pos_at_k[:, None, :] > q_pos[:, :, None] - window)
            & (raw_pos_at_k[:, None, :] >= 0)
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


def _decode_padding_mask(
    B: int,
    window_valid: Optional[mx.array],
    window_len: int,
    pool_lengths,
    compressed_len: int,
    topk: Optional[mx.array],
) -> Optional[mx.array]:
    """exo fix: for a one-token step of a batch whose rows have different lengths, the
    ``[B, 1, 1, keys]`` mask of the slots that hold that row's tokens: the window cache's
    valid slots (``window_valid``, from BatchRotatingKVCache.make_mask) and the compressed
    rows below the row's pool length (all of them, or the gathered ``topk`` ones). None when
    every row is full length."""
    if window_valid is None and pool_lengths is None:
        return None
    if window_valid is None:
        window_valid = mx.ones((B, 1, 1, window_len), dtype=mx.bool_)
    else:
        window_valid = mx.broadcast_to(window_valid, (B, 1, 1, window_len))
    if compressed_len == 0:
        return window_valid
    if pool_lengths is None:
        n = compressed_len if topk is None else topk.shape[-1]
        pool_valid = mx.ones((B, 1, 1, n), dtype=mx.bool_)
    else:
        lengths = mx.array(_as_lengths_list(pool_lengths, B), dtype=mx.int32)
        lengths = lengths[:, None, None, None]
        rows = (
            mx.arange(compressed_len, dtype=mx.int32)[None, None, None, :]
            if topk is None
            else topk.astype(mx.int32)[:, None, :, :]
        )
        pool_valid = rows < lengths
    return mx.concatenate([window_valid, pool_valid], axis=-1)


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


# --------------------------------------------------------------------------- #
# exo perf: prefill attention over the keys each query can see                 #
# --------------------------------------------------------------------------- #
# MLX has no fused attention kernel for head_dim 512, so sinks attention runs as
# its unfused fallback: one bf16 score matrix over every query x every key,
# masked, softmaxed and multiplied back. In a prefill chunk that is all S window
# keys and all compressed rows for every query, while each query can only see
# its last `window` raw tokens and, in indexer layers, `index_topk` compressed
# rows. The functions below do the fallback's math (scale q in bf16, bf16
# scores, finfo.min masking, the sink as an extra logit, precise softmax, bf16
# probabilities times the keys) on just those keys: masked keys get probability
# exactly 0 in the fallback, so only the GEMMs' accumulation order changes.

_PREFILL_Q_BLOCK = 128  # queries per block
_INDEXER_MAX_SCORES = 2**30  # indexer scores (queries x heads x rows) per block
_PREFILL_PER_QUERY_KEYS = True  # False: the masked full attention (tests compare the two)


def _attend(q_rows, keys, valid, sinks_col, neg):
    """q_rows [..., Q, D] (scaled), keys [..., K, D], valid bool broadcastable to
    [..., Q, K], sinks_col broadcastable to [..., Q, 1] -> [..., Q, D]."""
    scores = q_rows @ keys.swapaxes(-1, -2)
    scores = mx.where(valid, scores, neg)
    sink = mx.broadcast_to(sinks_col, (*scores.shape[:-1], 1))
    scores = mx.concatenate([sink, scores], axis=-1)
    probs = mx.softmax(scores, axis=-1, precise=True)[..., 1:]
    return probs @ keys


# exo perf: decode attention (one query per row) as two kernels instead of a gather, two
# concatenates, a scale, two matmuls and a softmax. The first scores blocks of 64 keys
# for 8 heads per threadgroup, reading the window rows and the selected (or all)
# compressed rows where they are, and keeps a float32 running max, sum and weighted
# value sum per block; the second merges the blocks with the sink logit. The query is
# scaled in bf16 and each score rounded to bf16 as MLX's fallback does; the softmax
# weights stay float32 (the fallback rounds them to bf16 before the second matmul).
_DECODE_KEY_BLOCK = 32
_DECODE_ATTENTION_KERNELS = True  # False: the gather + MLX attention (tests compare)


def _make_decode_attention_kernels():
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    partial_source = """
        // grid: (B * NB * H/8) threadgroups of 256 threads; one simdgroup per head
        uint tid = thread_position_in_threadgroup.x;
        uint tg = threadgroup_position_in_grid.x;
        uint lane = tid % 32;
        uint sg = tid / 32;
        uint B = q_shape[0];
        uint H = q_shape[1];
        uint W = window_shape[1];
        uint T = pool_shape[1];
        uint KC = !HAS_POOL ? 0 : (USE_IDX ? idx_shape[2] : T);
        uint K = W + KC;
        uint NB = (K + KB - 1) / KB;
        uint HG = H / 8;
        uint hg = tg % HG;
        uint blk = (tg / HG) % NB;
        uint b = tg / (HG * NB);
        uint h = hg * 8 + sg;
        constexpr uint EPL = D / 32;
        float qs[EPL];
        float sc = static_cast<float>(scale[0]);
        for (uint i = 0; i < EPL; ++i) {
            float v = static_cast<float>(q[(b * H + h) * D + lane * EPL + i]);
            qs[i] = static_cast<float>(static_cast<bfloat16_t>(v * sc));
        }
        float m = -INFINITY;
        float l = 0.0f;
        float o[EPL];
        for (uint i = 0; i < EPL; ++i) o[i] = 0.0f;
        uint j0 = blk * KB;
        uint j1 = metal::min(j0 + KB, K);
        for (uint j = j0; j < j1; ++j) {
            const device bfloat16_t* row;
            if (j < W) {
                if (MASKED && !window_valid[b * W + j]) continue;  // padding of a short row
                row = window + (b * W + j) * D;
            } else {
                uint c = USE_IDX ? idx[b * KC + (j - W)] : (j - W);
                if (MASKED && c >= static_cast<uint>(pool_len[b])) continue;
                row = pool + (b * T + c) * D;
            }
            float kv[EPL];
            float dot = 0.0f;
            for (uint i = 0; i < EPL; ++i) {
                kv[i] = static_cast<float>(row[lane * EPL + i]);
                dot += qs[i] * kv[i];
            }
            dot = simd_sum(dot);
            float s = static_cast<float>(static_cast<bfloat16_t>(dot));
            float m_new = metal::max(m, s);
            float corr = metal::exp(m - m_new);
            float e = metal::exp(s - m_new);
            l = l * corr + e;
            for (uint i = 0; i < EPL; ++i) o[i] = o[i] * corr + e * kv[i];
            m = m_new;
        }
        uint slot = (b * H + h) * NB + blk;
        if (lane == 0) {
            part_m[slot] = m;
            part_l[slot] = l;
        }
        for (uint i = 0; i < EPL; ++i) part_o[slot * D + lane * EPL + i] = o[i];
    """
    merge_source = """
        // grid: B * H threadgroups of 128 threads
        uint tid = thread_position_in_threadgroup.x;
        uint bh = threadgroup_position_in_grid.x;
        uint H = sinks_shape[0];
        uint NB = part_m_shape[1];
        uint h = bh % H;
        float sink = static_cast<float>(sinks[h]);
        float m = sink;
        for (uint k = 0; k < NB; ++k) m = metal::max(m, part_m[bh * NB + k]);
        threadgroup float w[MAXB];
        for (uint k = tid; k < NB; k += 128) w[k] = metal::exp(part_m[bh * NB + k] - m);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float l = metal::exp(sink - m);
        for (uint k = 0; k < NB; ++k) l += part_l[bh * NB + k] * w[k];
        float inv = 1.0f / l;
        for (uint d = tid; d < D; d += 128) {
            float acc = 0.0f;
            for (uint k = 0; k < NB; ++k) acc += part_o[(bh * NB + k) * D + d] * w[k];
            out[bh * D + d] = static_cast<bfloat16_t>(acc * inv);
        }
    """
    partial = mx.fast.metal_kernel(
        name="dsv4_decode_attention_partial",
        input_names=["q", "window", "pool", "idx", "scale", "window_valid", "pool_len"],
        output_names=["part_m", "part_l", "part_o"],
        source=partial_source,
    )
    merge = mx.fast.metal_kernel(
        name="dsv4_decode_attention_merge",
        input_names=["part_m", "part_l", "part_o", "sinks"],
        output_names=["out"],
        source=merge_source,
    )
    return partial, merge


_decode_attention_kernels = _make_decode_attention_kernels()
_DECODE_MAX_BLOCKS = 512  # merge kernel's threadgroup buffer: up to 16384 keys
_DECODE_NO_PADDING = (mx.array([True]), mx.array([0], dtype=mx.int32))


def _decode_attention(q, window_kv, pool, idx, scale, sinks, window_valid=None, pool_lengths=None):
    """q [B, H, 1, D] (RoPE applied); window_kv [B, W, D]; pool [B, T, D] or None; idx
    [B, 1, k] (rows of pool to use) or None (all of them). In a batch of rows of different
    lengths, window_valid ([B, 1, 1, W] bool, None: all) and pool_lengths (per row, None:
    all T) give each row's own keys; the others are skipped. Returns [B, H, 1, D], or None
    when the kernels don't apply."""
    if (
        _decode_attention_kernels is None
        or not _DECODE_ATTENTION_KERNELS
        or q.dtype != mx.bfloat16
        or window_kv.dtype != mx.bfloat16
        or q.shape[1] % 8
        or q.shape[-1] % 32
    ):
        return None
    B, H, _, D = q.shape
    W = window_kv.shape[1]
    has_pool = pool is not None and pool.shape[1] > 0
    if not has_pool:
        pool, idx = window_kv, None  # unused (HAS_POOL false)
    kc = 0 if not has_pool else (idx.shape[-1] if idx is not None else pool.shape[1])
    K = W + kc
    nb = (K + _DECODE_KEY_BLOCK - 1) // _DECODE_KEY_BLOCK
    if nb > _DECODE_MAX_BLOCKS:
        return None
    use_idx = idx is not None
    partial, merge = _decode_attention_kernels
    scale_arr = mx.array([scale], dtype=mx.bfloat16)
    idx_in = idx if use_idx else mx.zeros((B, 1, 1), dtype=mx.uint32)
    # exo fix (batched rows of different lengths): skip each row's padding slots
    masked = window_valid is not None or (has_pool and pool_lengths is not None)
    if not masked:
        valid_in, len_in = _DECODE_NO_PADDING  # unused
    else:
        valid_in = (
            mx.ones((B, W), dtype=mx.bool_)
            if window_valid is None
            else mx.broadcast_to(window_valid, (B, 1, 1, W)).reshape(B, W)
        )
        lengths = _as_lengths_list(pool_lengths, B, pool.shape[1] if has_pool else 0)
        len_in = mx.array(lengths * B if len(lengths) == 1 else lengths, dtype=mx.int32)
    part_m, part_l, part_o = partial(
        inputs=[q.reshape(B, H, D), window_kv, pool, idx_in, scale_arr, valid_in, len_in],
        template=[
            ("D", D), ("KB", _DECODE_KEY_BLOCK), ("USE_IDX", use_idx), ("HAS_POOL", has_pool),
            ("MASKED", masked),
        ],
        grid=(B * nb * (H // 8) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B * H, nb), (B * H, nb), (B * H, nb, D)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    out = merge(
        inputs=[part_m, part_l, part_o, sinks],
        template=[("D", D), ("MAXB", _DECODE_MAX_BLOCKS)],
        grid=(B * H * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(B * H, D)],
        output_dtypes=[mx.bfloat16],
    )[0]
    return out.reshape(B, H, 1, D)


def _prefill_attention(
    q: mx.array,
    window_kv: mx.array,
    compressed: Optional[mx.array],
    topk: Optional[mx.array],
    offset: int,
    window: int,
    ratio: int,
    scale: float,
    sinks: mx.array,
) -> mx.array:
    """Sinks attention for a multi-token step with a uniform (int) offset.

    q [B, H, S, D] with RoPE; window_kv [B, Wl, D], rows in position order ending at
    raw position offset + S - 1; compressed [B, T, D] or None; topk [B, S, k] (indexer
    layers) or None. Returns [B, H, S, D], what one masked attention over
    concat(window_kv, compressed) gives (window rows within `window` of the query,
    compressed rows complete at the query, and only the top-k ones when the indexer
    selects)."""
    B, H, S, D = q.shape
    dtype = q.dtype
    q = mx.array(scale, dtype) * q
    neg = mx.array(mx.finfo(dtype).min, dtype)
    Wl = window_kv.shape[1]
    win_start = offset + S - Wl
    T = 0 if compressed is None else compressed.shape[1]
    sinks = sinks.astype(dtype)
    gather = topk is not None and T > topk.shape[-1]
    outs = []
    if gather:
        # Each query attends to its own `window` raw tokens and its own top-k rows:
        # gather them per query (like the reference's sparse attention).
        kvcat = mx.concatenate([window_kv, compressed], axis=1).reshape(B * (Wl + T), D)
        base = (mx.arange(B, dtype=mx.int32) * (Wl + T))[:, None, None]
        sinks_col = sinks[None, None, :, None]
        for s0 in range(0, S, _PREFILL_Q_BLOCK):
            s1 = min(S, s0 + _PREFILL_Q_BLOCK)
            qpos = mx.arange(offset + s0, offset + s1, dtype=mx.int32)
            widx = (qpos - win_start)[:, None] - (window - 1) + mx.arange(window, dtype=mx.int32)[None, :]
            wvalid = widx >= 0
            sel = topk[:, s0:s1].astype(mx.int32)
            cvalid = (sel + 1) * ratio <= (qpos[None, :, None] + 1)
            idx = mx.concatenate(
                [mx.broadcast_to(mx.maximum(widx, 0)[None], (B, s1 - s0, window)), sel + Wl],
                axis=-1,
            )
            valid = mx.concatenate(
                [mx.broadcast_to(wvalid[None], (B, s1 - s0, window)), cvalid], axis=-1
            )
            keys = mx.take(kvcat, idx + base, axis=0)  # [B, Q, window + k, D]
            q_rows = q[:, :, s0:s1].transpose(0, 2, 1, 3)  # [B, Q, H, D]
            o = _attend(q_rows, keys, valid[:, :, None, :], sinks_col, neg)
            outs.append(o.transpose(0, 2, 1, 3))
    else:
        # Every query of a block shares one key set: the window rows the block can see
        # plus all compressed rows (masked per query).
        sink_logit = sinks[None, :, None, None]
        for s0 in range(0, S, _PREFILL_Q_BLOCK):
            s1 = min(S, s0 + _PREFILL_Q_BLOCK)
            nq = s1 - s0
            ka = max(0, offset + s0 - (window - 1) - win_start)
            kb = offset + s1 - win_start
            qpos = mx.arange(offset + s0, offset + s1, dtype=mx.int32)[:, None]
            kpos = mx.arange(win_start + ka, win_start + kb, dtype=mx.int32)[None, :]
            valid = (kpos <= qpos) & (kpos > qpos - window)
            keys = window_kv[:, ka:kb]
            if T:
                cvis = mx.arange(1, T + 1, dtype=mx.int32)[None, :] * ratio <= qpos + 1
                valid = mx.concatenate([valid, cvis], axis=-1)
                keys = mx.concatenate([keys, compressed], axis=1)
            # MQA: all heads share the keys, so fold heads into the GEMM's rows.
            q_rows = q[:, :, s0:s1].reshape(B, H * nq, D)
            sc = (q_rows @ keys.swapaxes(-1, -2)).reshape(B, H, nq, keys.shape[1])
            sc = mx.where(valid[None, None], sc, neg)
            sink = mx.broadcast_to(sink_logit, (B, H, nq, 1))
            sc = mx.concatenate([sink, sc], axis=-1)
            probs = mx.softmax(sc, axis=-1, precise=True)[..., 1:]
            o = (probs.reshape(B, H * nq, keys.shape[1]) @ keys).reshape(B, H, nq, D)
            outs.append(o)
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


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
    """Partial-RoPE on the trailing rd dims of q [B,H,S,D] and kv [B,S,D]; ``freqs``
    covers the whole head (see DeepseekV4RoPE.head_freqs), so each is one rope kernel."""
    d = q.shape[-1]
    q = mx.fast.rope(q, d, traditional=True, base=None, scale=1.0, offset=offset, freqs=freqs)
    kv = mx.fast.rope(kv, d, traditional=True, base=None, scale=1.0, offset=offset, freqs=freqs)
    return q, kv


@mx.compile
def _attn_inv_rope_flatten(o, offset, rd, freqs, flat_dim):
    """Inverse-RoPE on trailing rd dims + transpose-and-reshape to
    ``[B, S, n_heads*head_dim]`` for the wo_a input."""
    o = mx.fast.rope(
        o, o.shape[-1], traditional=True, base=None,
        scale=-1.0, offset=offset, freqs=freqs,
    )
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
        self.wqkv_a = _dense_linear(args, self.dim, self.q_lora_rank + self.head_dim)
        self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = _dense_linear(args, self.q_lora_rank, self.n_heads * self.head_dim)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=self.eps)

        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        group_feat = (self.n_heads * self.head_dim) // self.n_groups
        self.wo_a = _dense_linear(args, group_feat, self.n_groups * self.o_lora_rank)
        self.wo_b = _dense_linear(
            args, self.n_groups * self.o_lora_rank, self.dim, bias=args.attention_bias
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

        if isinstance(self.wo_a, Fp8BlockLinear) and _fp8_rows(out) >= _FP8_DEQUANT_MIN_ROWS * self.n_groups:
            # exo perf: prefill runs the bf16 path on the dequantized (identical) weight
            wa = self.wo_a.dense_weight(out.dtype).reshape(
                self.n_groups, self.o_lora_rank, group_feat
            )
            y = mx.einsum("bsgd,grd->bsgr", out, wa)
            return y.reshape(B, S, self.n_groups * self.o_lora_rank)

        if isinstance(self.wo_a, nn.QuantizedLinear):
            # Batched over groups: one quantized_matmul for all groups instead
            # of a per-group Python loop (O(n_groups) → 1 kernel dispatch).
            # exo perf: the batch's rows are the matmul's rows ([G, B*S, group_feat]), so each
            # group's weight is read once per step, not once per row (it was a broadcast batch
            # dimension); one row gives the same result bit for bit
            out_g = out.reshape(B * S, self.n_groups, group_feat).transpose(1, 0, 2)
            weight = self.wo_a.weight.reshape(self.n_groups, self.o_lora_rank, -1)
            scales = self.wo_a.scales.reshape(self.n_groups, self.o_lora_rank, -1)
            biases = (
                None
                if self.wo_a.biases is None
                else self.wo_a.biases.reshape(self.n_groups, self.o_lora_rank, -1)
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
            return y.transpose(1, 0, 2).reshape(B, S, self.n_groups * self.o_lora_rank)

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

        # exo fix: rows of a batch with different lengths (merged caches) are left-padded
        # in the window cache and right-padded in the compressor pools. A one-token step
        # must not attend to those slots; the window cache knows which are valid before
        # this step's update.
        window_valid = (
            win_cache.make_mask(1)
            if S == 1 and isinstance(win_cache, BatchRotatingKVCache)
            else None
        )

        # Fused: partial-RoPE on q + kv in one compiled call.
        q, kv = _attn_qkv_partial_rope(q, kv, offset, rd, self.rope.head_freqs(self.head_dim))
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

        sinks = self._sink_for(q.dtype)
        uniform = (
            _PREFILL_PER_QUERY_KEYS
            and S > 1
            and not isinstance(offset, mx.array)
            and (v4_cache is None or v4_cache.pooled_lengths(_K_COMP) is None)
            and (indexer_topk is None or v4_cache.pooled_lengths(_K_IDX) is None)
        )
        if uniform:
            # exo perf: attend only to the keys each query can see (see _prefill_attention)
            o = _prefill_attention(
                q, window_kv, compressed if compressed_len else None, indexer_topk,
                offset, self.window, self.compress_ratio, self.scale, sinks,
            )
        else:
            window_len = window_kv.shape[1]
            # Decode (S=1) fast path: gather the Indexer's top-k compressed rows
            # directly into SDPA's key tensor so the kernel only attends to those
            # K rows + the window. For prefill (S>1) fall back to attending to the
            # full compressed buffer with a scatter-based mask — the gather-flat
            # approach would over-include across query positions, changing output.
            use_gather = (
                S == 1 and compressed_len > 0 and indexer_topk is not None
            )
            pool_lengths = (
                v4_cache.pooled_lengths(_K_COMP)
                if S == 1 and v4_cache is not None and compressed_len > 0
                else None
            )
            o = None
            if S == 1:
                o = _decode_attention(
                    q, window_kv, compressed if compressed_len else None,
                    indexer_topk if use_gather else None, self.scale, sinks,
                    window_valid, pool_lengths,
                )
            if o is None:
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

                # Decode (S=1): one query, so no causal mask; every pool row is in the
                # past (rows are emitted only after a full window of raw tokens). Only
                # when the batch's rows have different lengths are some slots padding,
                # which the mask excludes.
                if S == 1:
                    mask = _decode_padding_mask(
                        B,
                        window_valid,
                        window_len,
                        pool_lengths,
                        compressed_len,
                        indexer_topk if use_gather else None,
                    )
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
                # exo fix: mx.fast.scaled_dot_product_attention with sinks returns wrong
                # values for head_dim 256/512 once n_heads * L_q * L_kv passes 2**31 (seen
                # in MLX 0.32.3 and 0.32.4): a 4096-token prefill chunk crosses it in the
                # CSA layers after ~16k tokens of context. Split the queries (counting the
                # batch too, to stay clear of the limit when prompts are batched).
                max_q = max(
                    1, _SDPA_MAX_SCORES // (B * q.shape[1] * kv_all_4d.shape[2])
                )
                if S <= max_q:
                    o = scaled_dot_product_attention(
                        q, kv_all_4d, kv_all_4d,
                        cache=None, scale=self.scale, mask=mask, sinks=sinks,
                    )
                else:
                    o = mx.concatenate(
                        [
                            scaled_dot_product_attention(
                                q[:, :, s0 : s0 + max_q], kv_all_4d, kv_all_4d,
                                cache=None, scale=self.scale,
                                mask=None if mask is None else mask[:, :, s0 : s0 + max_q],
                                sinks=sinks,
                            )
                            for s0 in range(0, S, max_q)
                        ],
                        axis=2,
                    )

        # Fused: inverse-RoPE on the trailing rd dims + transpose-and-reshape
        # to [B, S, n_heads*head_dim] for wo_a.
        o = _attn_inv_rope_flatten(
            o, offset, rd, self.rope.head_freqs(self.head_dim), self.n_heads * self.head_dim
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
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: float = 0.0,
        args: Optional["ModelArgs"] = None,
    ):
        super().__init__()
        if args is None:
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        else:
            self.gate_proj = _dense_linear(args, hidden_size, intermediate_size)
            self.up_proj = _dense_linear(args, hidden_size, intermediate_size)
            self.down_proj = _dense_linear(args, intermediate_size, hidden_size)
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
                args=args,
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
        fp8_native = self.args.fp8_block_weights

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
            elif weight.dtype in (mx.uint8,) and fp8_native and weight.shape[-1] % 32 == 0:
                # exo perf: keep the FP8 weight as mxfp8 (same values, see Fp8BlockLinear)
                packed, scales = _fp8_block_to_mxfp8(weight, _ue8m0_bytes(k, v))
                dequanted[wk] = packed
                dequanted[k[: -len(".scale")] + ".scales"] = scales
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
