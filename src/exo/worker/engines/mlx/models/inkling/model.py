"""Thinking Machines Inkling (config ``model_type`` ``inkling_mm_model``): the text model,
for MLX.

Provenance
- Written for exo from Hugging Face transformers' Inkling implementation:
  https://github.com/huggingface/transformers, file
  ``src/transformers/models/inkling/modeling_inkling.py`` (generated from
  ``modular_inkling.py``), at commit ``17558143a725d0ad66e86e0c07fcfce657422fd5``
  (2026-10-29), and the checkpoint layout in ``conversion_mapping.py`` at the same commit.
  Licence: Apache-2.0, Copyright 2026 the HuggingFace Team.
- Written on 2026-10-30. No code is copied verbatim: the PyTorch modules are rewritten for
  MLX, reusing mlx-lm's ``SwitchLinear`` gathers and ``swiglu`` (the ``SwitchGLU``
  arithmetic, with fused gate/up weights; routed and shared experts), ``nn.RMSNorm``
  (``mx.fast.rms_norm`` rounds the normalised activations to the input dtype before
  applying the weight, as ``InklingRMSNorm`` does), its caches and
  ``scaled_dot_product_attention``.
- Metal kernels (``mx.fast.metal_kernel``, written for exo on 2026-10-30) replace the short
  convolutions, the relative-position bias, the router and the expert combine. Each
  reproduces the result of the MLX ops it replaces bit for bit (the unit tests check it):
  the same float32 operations in the same order, and for the router MLX's own sigmoid,
  stable sort, log-add-exp and softmax reductions (from MLX's Metal kernels,
  ml-explore/mlx, MIT licence).
- Local changes against the reference:
  - text only: the vision and audio towers and the multi-token-prediction layers are not
    loaded;
  - the attention bias is added to the scores inside MLX's fused attention, as an additive
    mask, instead of in an eager softmax; long prefills are split into query blocks so
    that the dense bias stays small;
  - the key positions come from mlx-lm's cache layout (temporal, or a ring for
    sliding-window layers) instead of ``arange``;
  - the router's top-k choice and weights are computed in float32 from the logits;
  - routed experts are summed in float32 and rounded once, and the routed and shared
    outputs are added in float32 before that rounding;
  - the LM head is cut to ``unpadded_vocab_size`` rows at load instead of slicing the
    logits;
  - the cache is a flat list (every layer's convolution states, then every layer's keys
    and values) rather than transformers' per-layer cache;
  - the query, key, value and relative-bias projections are one weight, and each expert
    stack's gate and up projections are one weight (Model.sanitize stacks them).

What the model is (Inkling-Small: 42 layers, hidden 4096):
- no rotary embedding: position comes from a learned, hidden-state-conditioned relative
  position bias (``r_proj`` mixes ``d_rel`` bias profiles over the backward distance,
  ``rel_proj``), and from depthwise causal short convolutions (kernel 4) on the keys, the
  values, the attention output and the MLP output, each with a residual, in float32;
- per-head RMSNorm on queries and keys, attention scale ``1 / head_dim``;
- sliding-window layers (window 512, bias over the window) and global layers (bias over
  the last ``rel_extent`` = 1024 positions, log-length query scaling from
  ``log_scaling_n_floor`` on);
- dense SwiGLU MLPs on the first ``dense_mlp_idx`` layers, then a sigmoid-gated MoE whose
  softmax over the chosen experts also includes the shared experts (``shared_expert_sink``);
- an RMSNorm after the token embedding, and the final hidden state divided by
  ``logits_mup_width_multiplier`` before the LM head.
"""

import functools
import itertools
import math
import re
from dataclasses import dataclass, field
from typing import Final, Literal, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import (
    swiglu,  # pyright: ignore[reportUnknownVariableType]
)
from mlx_lm.models.base import (
    create_attention_mask,  # pyright: ignore[reportUnknownVariableType]
    create_ssm_mask,  # pyright: ignore[reportUnknownVariableType]
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache, RotatingKVCache
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear

__all__ = ["CacheEntry", "Model", "ModelArgs", "TextArgs"]

# Elements of the dense bias materialised per attention call (per batch row). Longer
# prefills are split into query blocks to stay under it.
_BIAS_ELEMENTS_PER_CALL: Final = 1 << 26
# Global layers: queries per call, and the larger budget for their dense bias.
_GLOBAL_QUERY_BLOCK: Final = 512
_GLOBAL_BIAS_ELEMENTS_PER_CALL: Final = 1 << 28
# MLX's fused attention takes its vector kernel for calls of at most this many queries.
_VECTOR_MAX_QUERIES: Final = 8
# The four short convolutions of a layer, by their slot in the layer's ArraysCache.
_K_CONV: Final = 0
_V_CONV: Final = 1
_ATTN_CONV: Final = 2
_MLP_CONV: Final = 3

LayerType = Literal["hybrid", "hybrid_sliding"]
# The model's cache is one flat list: every layer's short-convolution states (an
# ArraysCache of 4), then every layer's keys and values (rotating for sliding-window
# layers). Keeping the two apart, rather than a CacheList per layer, lets exo's prefix cache
# trim the global layers' keys and values instead of copying them into every prefill
# snapshot (it snapshots only caches it cannot trim), and keeps a KV cache last, where
# exo's distributed patches look for one.
KVEntry = KVCache | RotatingKVCache
CacheEntry = ArraysCache | KVEntry


def _get(config: dict[str, object], key: str, default: object = None) -> object:
    return config.get(key, default)


def _int(config: dict[str, object], key: str, default: int | None = None) -> int:
    value = _get(config, key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Inkling config: {key} must be an integer, got {value!r}")
    return value


def _float(config: dict[str, object], key: str, default: float | None = None) -> float:
    value = _get(config, key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Inkling config: {key} must be a number, got {value!r}")
    return float(value)


def _optional_int(config: dict[str, object], key: str) -> int | None:
    value = _get(config, key)
    return None if value is None else _int(config, key)


@dataclass(frozen=True)
class TextArgs:
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    unpadded_vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    swa_num_attention_heads: int
    swa_num_key_value_heads: int
    swa_head_dim: int
    sliding_window_size: int
    d_rel: int
    rel_extent: int
    log_scaling_n_floor: int | None
    log_scaling_alpha: float
    rms_norm_eps: float
    conv_kernel_size: int
    layer_types: tuple[LayerType, ...]
    # Whether each layer has a dense MLP (otherwise the MoE).
    dense_layers: tuple[bool, ...]
    dense_intermediate_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    route_scale: float
    logits_mup_width_multiplier: float

    @classmethod
    def from_text_config(cls, config: dict[str, object]) -> "TextArgs":
        """Reads a checkpoint's ``text_config`` (the field names the official checkpoints
        use; transformers' ``InklingTextConfig`` maps them the same way)."""
        # Settings the reference implementation hard-codes: refuse checkpoints that ask
        # for anything else rather than run them wrongly.
        for key, expected in (
            ("hidden_act", "silu"),
            ("gate_activation", "sigmoid"),
            ("use_gate_bias", True),
            ("norm_after_topk", True),
            ("shared_expert_sink", True),
            ("use_sconv", True),
            ("use_embed_norm", True),
            ("use_global_scale", True),
            ("q_bias", False),
            ("o_bias", False),
            ("final_logit_softcapping", None),
        ):
            if key in config and config[key] != expected:
                raise ValueError(
                    f"Inkling config: {key}={config[key]!r} is not supported"
                )
        layers = _int(config, "num_hidden_layers")
        layer_types = _layer_types(config, layers)
        dense_layers = _dense_layers(config, layers)
        vocab_size = _int(config, "vocab_size")
        unpadded = _optional_int(config, "unpadded_vocab_size")
        # Checkpoints name the routed experts' width intermediate_size when they also
        # have dense_intermediate_size (transformers' InklingTextConfig.__post_init__).
        if "dense_intermediate_size" in config:
            dense_width = _int(config, "dense_intermediate_size")
            expert_width = _int(config, "intermediate_size")
        else:
            dense_width = _int(config, "intermediate_size")
            expert_width = _int(config, "moe_intermediate_size")
        return cls(
            hidden_size=_int(config, "hidden_size"),
            num_hidden_layers=layers,
            vocab_size=vocab_size,
            unpadded_vocab_size=vocab_size if unpadded is None else unpadded,
            num_attention_heads=_int(config, "num_attention_heads"),
            num_key_value_heads=_int(config, "num_key_value_heads"),
            head_dim=_int(config, "head_dim"),
            swa_num_attention_heads=_int(config, "swa_num_attention_heads"),
            swa_num_key_value_heads=_int(config, "swa_num_key_value_heads"),
            swa_head_dim=_int(config, "swa_head_dim"),
            sliding_window_size=_int(config, "sliding_window_size"),
            d_rel=_int(config, "d_rel"),
            rel_extent=_int(config, "rel_extent"),
            log_scaling_n_floor=_optional_int(config, "log_scaling_n_floor"),
            log_scaling_alpha=_float(config, "log_scaling_alpha", 0.1),
            rms_norm_eps=_float(config, "rms_norm_eps", 1e-6),
            conv_kernel_size=_int(
                config, "sconv_kernel_size", _int(config, "conv_kernel_size", 4)
            ),
            layer_types=layer_types,
            dense_layers=dense_layers,
            dense_intermediate_size=dense_width,
            moe_intermediate_size=expert_width,
            n_routed_experts=_int(config, "n_routed_experts"),
            num_experts_per_tok=_int(config, "num_experts_per_tok"),
            n_shared_experts=_int(config, "n_shared_experts"),
            route_scale=_float(config, "route_scale"),
            logits_mup_width_multiplier=_float(config, "logits_mup_width_multiplier"),
        )


def _layer_types(config: dict[str, object], layers: int) -> tuple[LayerType, ...]:
    """Explicit ``layer_types``, else from ``local_layer_ids``, else transformers'
    default (every layer whose index is not 5 mod 6 is local)."""
    explicit = _get(config, "layer_types")
    if isinstance(explicit, list):
        kinds = [str(kind) for kind in cast(list[object], explicit)]
        if len(kinds) != layers or not set(kinds) <= {"hybrid", "hybrid_sliding"}:
            raise ValueError(f"Inkling config: unsupported layer_types {kinds}")
        return tuple(
            "hybrid_sliding" if kind == "hybrid_sliding" else "hybrid" for kind in kinds
        )
    local_layers = _get(config, "local_layer_ids")
    if local_layers is None:
        local = {i for i in range(layers) if (i + 1) % 6}
    elif isinstance(local_layers, list):
        local = {_int({"id": i}, "id") for i in cast(list[object], local_layers)}
    else:
        raise ValueError("Inkling config: local_layer_ids must be a list")
    return tuple("hybrid_sliding" if i in local else "hybrid" for i in range(layers))


def _dense_layers(config: dict[str, object], layers: int) -> tuple[bool, ...]:
    """Explicit ``mlp_layer_types``, else the first ``dense_mlp_idx`` layers are dense."""
    explicit = _get(config, "mlp_layer_types")
    if isinstance(explicit, list):
        kinds = [str(kind) for kind in cast(list[object], explicit)]
        if len(kinds) != layers or not set(kinds) <= {"dense", "sparse"}:
            raise ValueError(f"Inkling config: unsupported mlp_layer_types {kinds}")
        return tuple(kind == "dense" for kind in kinds)
    dense = _int(config, "dense_mlp_idx", 0)
    return tuple(i < dense for i in range(layers))


@dataclass
class ModelArgs:
    model_type: str
    text_config: dict[str, object] = field(default_factory=dict[str, object])

    @property
    def text(self) -> TextArgs:
        return TextArgs.from_text_config(self.text_config)

    @classmethod
    def from_dict(cls, params: dict[str, object]) -> "ModelArgs":
        text_config = params.get("text_config")
        if not isinstance(text_config, dict):
            raise ValueError("Inkling config: text_config is missing")
        return cls(
            model_type=str(params["model_type"]),
            text_config=cast(dict[str, object], text_config),
        )


def _metal_kernels_available() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


# Up to this many tokens per call (decoding, alone or in a batch), the small elementwise
# steps run as compiled graphs: one cached graph per shape, so they are not compiled for
# every prefill length.
_COMPILED_MAX_TOKENS: Final = 64


def _decoding(x: mx.array) -> bool:
    return x.size // x.shape[-1] <= _COMPILED_MAX_TOKENS


# One thread per output element (a channel at a position), plus one per channel for each
# of the kernel_size - 1 state rows. ``window`` is the state followed by the inputs; the
# products are added one at a time in float32, in the order of the op-by-op path below,
# so both give the same result.
_SHORT_CONV_SOURCE: Final = """
    const int K1 = K - 1;
    const int length = x_shape[1];
    const int channels = x_shape[2];
    const int c = int(thread_position_in_grid.x);
    const int row = int(thread_position_in_grid.y);
    const int b = int(thread_position_in_grid.z);
    if (c >= channels) {
        return;
    }
    const size_t x_base = size_t(b) * size_t(length) * size_t(channels) + size_t(c);
    const size_t s_base = size_t(b) * size_t(K1) * size_t(channels) + size_t(c);
    if (row < length) {
        float acc = 0.0f;
        for (int j = 0; j < K; ++j) {
            const int i = row + j;
            float v = 0.0f;
            if (i < K1) {
                if (HAS_STATE) {
                    v = static_cast<float>(state[s_base + size_t(i) * size_t(channels)]);
                }
            } else {
                v = static_cast<float>(x[x_base + size_t(i - K1) * size_t(channels)]);
            }
            const float product = v * static_cast<float>(weight[c * K + j]);
            if (j == 0) {
                acc = product;
            } else {
                acc = acc + product;
            }
        }
        const size_t at = x_base + size_t(row) * size_t(channels);
        const float total = acc + static_cast<float>(x[at]);
        if (HAS_SKIP) {
            // the decoder layer's residual stream, added in the output dtype as an op would
            const OutT convolved = static_cast<OutT>(total);
            out[at] = static_cast<OutT>(static_cast<float>(skip[at]) + static_cast<float>(convolved));
        } else {
            out[at] = static_cast<OutT>(total);
        }
    } else {
        // new_state[row - length] = window[row]
        float v = 0.0f;
        if (row < K1) {
            if (HAS_STATE) {
                v = static_cast<float>(state[s_base + size_t(row) * size_t(channels)]);
            }
        } else {
            v = static_cast<float>(x[x_base + size_t(row - K1) * size_t(channels)]);
        }
        new_state[s_base + size_t(row - length) * size_t(channels)] = v;
    }
"""


@functools.cache
def _short_conv_kernel() -> "mx.fast.MetalKernel":
    return mx.fast.metal_kernel(
        name="inkling_short_conv",
        input_names=["x", "state", "weight", "skip"],
        output_names=["out", "new_state"],
        source=_SHORT_CONV_SOURCE,
    )


class ShortConvolution(nn.Module):
    """Depthwise causal convolution over the sequence with a residual, in float32
    (``InklingShortConvolution``: transformers keeps these modules in float32). Its last
    ``kernel_size - 1`` inputs are the state carried between calls."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        # MLX's conv1d layout (channels, kernel, 1); the official checkpoints store
        # (channels, 1, kernel), which Model.sanitize transposes.
        self.weight = mx.zeros((channels, kernel_size, 1))

    def __call__(
        self,
        x: mx.array,
        cache: ArraysCache | None,
        slot: int,
        mask: mx.array | None,
        skip: mx.array | None = None,
    ) -> mx.array:
        """The convolution of ``x`` plus ``x`` (float32, rounded to ``x``'s dtype), plus
        ``skip`` if given (the decoder layer's residual stream, added after that rounding)."""
        if (
            mask is None
            and (cache is None or cache.lengths is None)
            and _metal_kernels_available()
        ):
            return self._fused(x, cache, slot, skip)
        out = self._by_ops(x, cache, slot, mask)
        return out if skip is None else skip + out

    def _fused(
        self,
        x: mx.array,
        cache: ArraysCache | None,
        slot: int,
        skip: mx.array | None = None,
    ) -> mx.array:
        """One kernel: the convolution, the residual, the skip and the next state."""
        batch, length, channels = x.shape
        keep = self.kernel_size - 1
        state = cast(mx.array | None, cache[slot]) if cache is not None else None
        if skip is not None and skip.dtype != x.dtype:
            return skip + self._fused(x, cache, slot)
        out, new_state = _short_conv_kernel()(
            inputs=[
                x,
                x if state is None else state,
                self.weight,
                x if skip is None else skip,
            ],
            template=[
                ("K", self.kernel_size),
                ("HAS_STATE", state is not None),
                ("HAS_SKIP", skip is not None),
                ("OutT", x.dtype),
            ],
            grid=(channels, length + keep, batch),
            threadgroup=(min(channels, 256), 1, 1),
            output_shapes=[x.shape, (batch, keep, channels)],
            output_dtypes=[x.dtype, mx.float32],
        )
        if cache is not None:
            cache[slot] = new_state
        return out

    def _by_ops(
        self,
        x: mx.array,
        cache: ArraysCache | None,
        slot: int,
        mask: mx.array | None,
    ) -> mx.array:
        batch, length, channels = x.shape
        keep = self.kernel_size - 1
        x32 = x.astype(mx.float32)
        # Padded positions of a batch must not leak into the state (the reference zeroes
        # them the same way); the residual keeps the unmasked input.
        inputs = x32 if mask is None else mx.where(mask[..., None], x32, 0.0)
        state = cast(mx.array | None, cache[slot]) if cache is not None else None
        if state is None:
            state = mx.zeros((batch, keep, channels), dtype=mx.float32)
        window = mx.concatenate([state, inputs], axis=1)
        weight = self.weight.astype(mx.float32)[:, :, 0]
        out = window[:, :length] * weight[:, 0]
        for tap in range(1, self.kernel_size):
            out = out + window[:, tap : tap + length] * weight[:, tap]
        if cache is not None:
            if cache.lengths is not None:
                # Right-padded batch prompt: keep the inputs before each row's padding.
                ends = mx.clip(cache.lengths, 0, length)
                positions = (ends[:, None] + mx.arange(keep))[..., None]
                cache[slot] = mx.take_along_axis(window, positions, axis=1)
            else:
                cache[slot] = mx.contiguous(window[:, -keep:, :])
        return (out + x32).astype(x.dtype)


def _newest_key(cache: object | None, num_keys: int) -> int:
    """Index, among the keys the cache returned after its update, of the newest key.

    mlx-lm's caches return keys in temporal order (``KVCache``, ``BatchKVCache``, and the
    rotating caches after a multi-token update), except that a rotating cache that has
    wrapped returns its ring buffer after a one-token update, whose newest slot is
    ``_idx - 1``.
    """
    if cache is not None and isinstance(cache, RotatingKVCache):
        return cache._idx - 1
    if cache is not None and hasattr(cache, "rotated"):  # BatchRotatingKVCache
        return cast(int, getattr(cache, "_idx")) - 1  # noqa: B009
    return num_keys - 1


def key_distances(cache: object | None, num_queries: int, num_keys: int) -> mx.array:
    """Backward distance from each of the ``num_queries`` new tokens to each key the cache
    returned after its update, as ``(num_queries, num_keys)`` int32.

    The newest key is the last query's own (see ``_newest_key``). Rows of a batch are
    aligned on their newest token, so one distance matrix serves every row; padding is
    masked.
    """
    newest = _newest_key(cache, num_keys)
    age = (newest - mx.arange(num_keys)) % num_keys
    lag = num_queries - 1 - mx.arange(num_queries)
    return (age[None, :] - lag[:, None]).astype(mx.int32)


def derives_causal_mask(cache: object | None) -> bool:
    """Whether attention over this cache can work its mask out from key distances alone
    (one sequence: causal, and within the window on sliding-window layers). A batch cache
    also masks each row's padding, so its mask is built and passed in."""
    return _metal_kernels_available() and (
        cache is None or isinstance(cache, (KVCache, RotatingKVCache))
    )


# One thread per bias element of a block of queries and keys. The same arithmetic as
# ``key_distances`` and ``_RelativeBias._by_ops``: distance = age of the key - lag of the
# query; the profile at that distance inside [0, E), zero outside; and the mask (passed in,
# or derived from the distance when there is none: causal, within WINDOW if set) fills the
# rest with the dtype's lowest value.
_RELATIVE_BIAS_SOURCE: Final = """
    const int heads = params[0];
    const int q0 = params[1];
    const int k0 = params[2];
    const int newest = params[3];
    const int num_keys = params[4];
    const int num_queries = params[5];
    const int block_queries = params[6];
    const int block_keys = params[7];
    const int mask_batches = params[8];
    const int kk = int(thread_position_in_grid.x);
    const int qq = int(thread_position_in_grid.y);
    const int bh = int(thread_position_in_grid.z);
    if (kk >= block_keys) {
        return;
    }
    const int b = bh / heads;
    const int h = bh - b * heads;
    const int q = q0 + qq;
    const int k = k0 + kk;
    int age = (newest - k) % num_keys;
    if (age < 0) {
        age += num_keys;
    }
    const int d = age - (num_queries - 1 - q);
    T value = static_cast<T>(0.0f);
    if (d >= 0 && d < E) {
        value = profiles[((size_t(b) * size_t(num_queries) + size_t(q)) * size_t(heads)
                          + size_t(h)) * size_t(E) + size_t(d)];
    }
    bool keep;
    if (HAS_MASK) {
        const int mb = mask_batches == 1 ? 0 : b;
        keep = bool(mask[(size_t(mb) * size_t(num_queries) + size_t(q)) * size_t(num_keys)
                         + size_t(k)]);
    } else {
        keep = d >= 0 && (WINDOW == 0 || d < WINDOW);
    }
    out[(size_t(bh) * size_t(block_queries) + size_t(qq)) * size_t(block_keys) + size_t(kk)] =
        keep ? value : static_cast<T>(fill[0]);
"""


@functools.cache
def _lowest(dtype: mx.Dtype) -> mx.array:
    """The dtype's lowest value (the masked fill), as a float32 kernel input."""
    return mx.array([mx.finfo(dtype).min], dtype=mx.float32)


@functools.cache
def _relative_bias_kernel() -> "mx.fast.MetalKernel":
    return mx.fast.metal_kernel(
        name="inkling_relative_bias",
        input_names=["profiles", "mask", "params", "fill"],
        output_names=["out"],
        source=_RELATIVE_BIAS_SOURCE,
    )


def _last_position_bound(cache: object | None, num_queries: int) -> int | None:
    """An upper bound on the absolute position of the last new token, without reading
    arrays from the GPU (None when there is none). A batch cache's rows are left-padded,
    so each row's offset is at most the cache's own index."""
    offset = 0 if cache is None else getattr(cache, "offset", 0)
    if isinstance(offset, int):
        return offset + num_queries - 1
    if isinstance(cache, BatchKVCache):
        return cache._idx + num_queries - 1
    return None


def _query_positions(cache: object | None, num_queries: int) -> mx.array:
    """Absolute positions of the new tokens, ``(1 or batch, num_queries)``."""
    offset = 0 if cache is None else cast(int | mx.array, getattr(cache, "offset", 0))
    steps = mx.arange(num_queries)
    if isinstance(offset, mx.array):
        return offset[:, None] + steps[None, :]
    return (steps + offset)[None, :]


# Up to this many tokens per call, a matmul with a weight is a matrix-vector product in
# MLX (one token, or a few in MLX's wide variant), whose rows don't depend on the
# weight's height.
_FUSED_PROJECTION_MAX_TOKENS: Final = 15


def _rows_matmul(linear: nn.Module, x: mx.array, start: int, stop: int) -> mx.array:
    """``x`` times rows ``start:stop`` of a linear layer's weight (transposed), plain or
    quantized, as that layer would compute it with only those rows."""
    weight = cast(mx.array, linear["weight"])[start:stop]
    if isinstance(linear, nn.QuantizedLinear):
        biases = cast(mx.array | None, linear.get("biases"))
        return mx.quantized_matmul(
            x,
            weight,
            cast(mx.array, linear["scales"])[start:stop],
            None if biases is None else biases[start:stop],
            transpose=True,
            group_size=cast(int, linear.group_size),
            bits=cast(int, linear.bits),
            mode=cast(str, linear.mode),
        )
    return x @ weight.T


class Attention(nn.Module):
    def __init__(self, args: TextArgs, layer_index: int):
        super().__init__()
        self.is_sliding: bool = args.layer_types[layer_index] == "hybrid_sliding"
        if self.is_sliding:
            heads = args.swa_num_attention_heads
            kv_heads = args.swa_num_key_value_heads
            head_dim = args.swa_head_dim
        else:
            heads = args.num_attention_heads
            kv_heads = args.num_key_value_heads
            head_dim = args.head_dim
        self.num_heads = heads
        self.num_kv_heads = kv_heads
        self.head_dim = head_dim
        self.d_rel = args.d_rel
        self.window: int | None = args.sliding_window_size if self.is_sliding else None
        self.rel_extent: int = (
            args.sliding_window_size if self.is_sliding else args.rel_extent
        )
        # Queries and keys are RMS-normalised per head, hence 1/d rather than 1/sqrt(d).
        self.scale = 1.0 / head_dim
        self.log_scaling_n_floor: int | None = (
            None if self.is_sliding else args.log_scaling_n_floor
        )
        self.log_scaling_alpha = args.log_scaling_alpha

        hidden = args.hidden_size
        # The query, key, value and relative-bias mixing projections (the reference's
        # q_proj, k_proj, v_proj and r_proj), fused into one matmul: their output rows in
        # that order (see ``projection_widths``).
        self.qkvr_proj = nn.Linear(hidden, sum(self.projection_widths()), bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.k_sconv = ShortConvolution(kv_heads * head_dim, args.conv_kernel_size)
        self.v_sconv = ShortConvolution(kv_heads * head_dim, args.conv_kernel_size)
        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        # Bias profiles over the backward distance (InklingRelativeLogits.proj).
        self.rel_proj = mx.zeros((args.d_rel, self.rel_extent))

    def projection_widths(self) -> tuple[int, int, int, int]:
        """Output widths of the query, key, value and relative-bias projections in
        ``qkvr_proj`` (for this rank's heads under tensor parallelism)."""
        return (
            self.num_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
            self.num_heads * self.d_rel,
        )

    def _project(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Queries, keys, values and relative-bias mixing weights.

        Every projection keeps the result MLX gives it alone. A few tokens (decoding)
        take one matrix-vector product over the rows of consecutive projections: MLX
        computes each output row the same way whatever the matrix's height, except that
        it switches kernels for outputs narrower than a sixteenth of the input (e.g. the
        relative-bias rows of one rank under tensor parallelism), which then run alone.
        More tokens take one matmul per projection over its rows of the fused weight:
        MLX picks a matmul's tiling from its shape."""
        bounds: list[int] = [0]
        for width in self.projection_widths():
            bounds.append(bounds[-1] + width)
        spans = list(itertools.pairwise(bounds))
        parts: list[mx.array] = []
        if x.size // x.shape[-1] > _FUSED_PROJECTION_MAX_TOKENS:
            parts = [_rows_matmul(self.qkvr_proj, x, a, b) for a, b in spans]
        else:
            run: list[tuple[int, int]] = []
            for a, b in [*spans, (-1, -1)]:
                if a >= 0 and (b - a) * 16 > x.shape[-1]:
                    run.append((a, b))
                    continue
                if run:
                    rows = _rows_matmul(self.qkvr_proj, x, run[0][0], run[-1][1])
                    parts += [rows[..., c - run[0][0] : d - run[0][0]] for c, d in run]
                    run = []
                if a >= 0:
                    parts.append(_rows_matmul(self.qkvr_proj, x, a, b))
        queries, keys, values, relative = parts
        return queries, keys, values, relative

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None,
        conv_mask: mx.array | None,
        kv_cache: KVEntry | None,
        conv_cache: ArraysCache | None,
    ) -> mx.array:
        batch, length, _ = x.shape

        queries, keys, values, relative = self._project(x)
        keys = self.k_sconv(keys, conv_cache, _K_CONV, conv_mask)
        values = self.v_sconv(values, conv_cache, _V_CONV, conv_mask)
        relative = relative.reshape(batch, length, self.num_heads, self.d_rel)

        queries = self.q_norm(
            queries.reshape(batch, length, self.num_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(batch, length, self.num_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch, length, self.num_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        tau: mx.array | None = None
        if self.log_scaling_n_floor is not None:
            # Only positions past the floor scale (below it the factor is exactly 1);
            # skip the work when no query can reach it.
            last = _last_position_bound(kv_cache, length)
            if last is None or last >= self.log_scaling_n_floor:
                positions = _query_positions(kv_cache, length)
                effective = (positions + 1).astype(mx.float32)
                tau = 1.0 + self.log_scaling_alpha * mx.log(
                    mx.maximum(effective / self.log_scaling_n_floor, 1.0)
                )
                queries = (queries.astype(mx.float32) * tau[:, None, :, None]).astype(
                    queries.dtype
                )

        if kv_cache is not None:
            keys, values = cast(
                tuple[mx.array, mx.array],
                kv_cache.update_and_fetch(keys, values),  # pyright: ignore[reportUnknownMemberType]
            )

        # (batch, length, heads, rel_extent): each query's bias at each backward distance.
        profiles = relative @ self.rel_proj.astype(relative.dtype)
        if tau is not None:
            profiles = (profiles.astype(mx.float32) * tau[:, :, None, None]).astype(
                profiles.dtype
            )

        output = self._attend(queries, keys, values, profiles, mask, kv_cache)
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)

    def _attend(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        profiles: mx.array,
        mask: mx.array | None,
        kv_cache: KVEntry | None,
    ) -> mx.array:
        length = queries.shape[2]
        num_keys = keys.shape[2]
        bias = _RelativeBias(self, profiles, mask, kv_cache, length, num_keys)
        outputs: list[mx.array] = []
        for start, stop, key_start, key_stop in self._spans(length, num_keys):
            outputs.append(
                self._attend_block(
                    queries[:, :, start:stop],
                    keys[:, :, key_start:key_stop],
                    values[:, :, key_start:key_stop],
                    bias.block(start, stop, key_start, key_stop),
                    kv_cache,
                )
            )
        return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=2)

    def _spans(self, length: int, num_keys: int) -> list[tuple[int, int, int, int]]:
        """The attention calls of ``length`` new queries over ``num_keys`` keys, as
        ``(first query, end, first key, end)``.

        The bias of a call is materialised, so a long prefill splits its queries into
        blocks under ``_BIAS_ELEMENTS_PER_CALL``; its keys are then in temporal order (a
        multi-token update), each block stops at its last query's key, and a
        sliding-window block starts at the first key its window reaches.

        Global layers at long contexts: that budget leaves calls of a few dozen queries,
        which keep the GPU half idle. Their blocks of more than ``_VECTOR_MAX_QUERIES``
        queries are regrouped into calls of up to ``_GLOBAL_QUERY_BLOCK`` queries within
        ``_GLOBAL_BIAS_ELEMENTS_PER_CALL``. MLX's fused attention gives each query the
        same result in any such call (every call starts at the first key and later keys
        are masked); a trailing block of at most ``_VECTOR_MAX_QUERIES`` queries stays
        alone, because MLX attends that few queries with another kernel."""
        block = max(1, _BIAS_ELEMENTS_PER_CALL // (self.num_heads * num_keys))
        first = num_keys - length
        if length <= block:
            return [(0, length, 0, num_keys)]
        if self.window is not None:
            block = max(1, int(math.sqrt(_BIAS_ELEMENTS_PER_CALL / self.num_heads)))
        spans: list[tuple[int, int, int, int]] = []
        for start in range(0, length, block):
            stop = min(start + block, length)
            key_start = 0
            if self.window is not None:
                key_start = max(0, first + start - (self.window - 1))
            spans.append((start, stop, key_start, first + stop))
        if self.window is not None or block <= _VECTOR_MAX_QUERIES:
            return spans
        limit = max(
            block,
            min(
                _GLOBAL_QUERY_BLOCK,
                _GLOBAL_BIAS_ELEMENTS_PER_CALL // (self.num_heads * num_keys),
            ),
        )
        tail = (
            [spans.pop()] if spans[-1][1] - spans[-1][0] <= _VECTOR_MAX_QUERIES else []
        )
        regrouped = spans[-1][1]
        count = -(-regrouped // limit)
        bounds = [regrouped * i // count for i in range(count + 1)]
        return [(a, b, 0, first + b) for a, b in itertools.pairwise(bounds)] + tail

    def _attend_block(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        bias: mx.array,
        kv_cache: KVEntry | None,
    ) -> mx.array:
        return scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=kv_cache,
            scale=self.scale,
            mask=bias.astype(queries.dtype),
        )


class _RelativeBias:
    """The additive attention mask of one attention call, block by block: each query's
    relative-position bias at each key's backward distance (zero outside the bias extent),
    and the lowest value where the mask hides the key.

    With Metal, one kernel writes each block. ``mask`` is None when the caller left the
    mask to the distances (one sequence: causal, and within the window on sliding-window
    layers; see ``derives_causal_mask``). Otherwise the bias is built op by op, from
    ``key_distances``."""

    def __init__(
        self,
        attention: "Attention",
        profiles: mx.array,
        mask: mx.array | None,
        kv_cache: KVEntry | None,
        num_queries: int,
        num_keys: int,
    ):
        self.attention = attention
        self.profiles = profiles  # (batch, queries, heads, rel_extent)
        self.mask = mask
        self.kv_cache = kv_cache
        self.num_queries = num_queries
        self.num_keys = num_keys
        self.newest = _newest_key(kv_cache, num_keys)
        self.fused = _metal_kernels_available()
        self._distances: mx.array | None = None
        self._mask3: mx.array | None = None
        if self.fused and mask is not None:
            # (mask batches, queries, keys), from (queries, keys) or (batch, 1, ...)
            self._mask3 = mask.reshape(-1, num_queries, num_keys)

    def block(self, q0: int, q1: int, k0: int, k1: int) -> mx.array:
        if self.fused:
            return self._by_kernel(q0, q1, k0, k1)
        return self._by_ops(q0, q1, k0, k1)

    def _by_kernel(self, q0: int, q1: int, k0: int, k1: int) -> mx.array:
        batch, _, heads, extent = self.profiles.shape
        dtype = self.profiles.dtype
        mask = self._mask3
        params = mx.array(
            [
                heads,
                q0,
                k0,
                self.newest,
                self.num_keys,
                self.num_queries,
                q1 - q0,
                k1 - k0,
                1 if mask is None else mask.shape[0],
            ],
            dtype=mx.int32,
        )
        window = self.attention.window
        (bias,) = _relative_bias_kernel()(
            inputs=[
                self.profiles,
                params if mask is None else mask,
                params,
                _lowest(dtype),
            ],
            template=[
                ("T", dtype),
                ("E", extent),
                ("WINDOW", 0 if window is None else window),
                ("HAS_MASK", mask is not None),
            ],
            grid=(k1 - k0, q1 - q0, batch * heads),
            threadgroup=(min(k1 - k0, 256), 1, 1),
            output_shapes=[(batch, heads, q1 - q0, k1 - k0)],
            output_dtypes=[dtype],
        )
        return bias

    def _by_ops(self, q0: int, q1: int, k0: int, k1: int) -> mx.array:
        if self._distances is None:
            self._distances = key_distances(
                self.kv_cache, self.num_queries, self.num_keys
            )
        distances = self._distances[q0:q1, k0:k1]
        extent = self.attention.rel_extent
        profiles = self.profiles[:, q0:q1].transpose(0, 2, 1, 3)
        in_band = (distances >= 0) & (distances < extent)
        index = mx.clip(distances, 0, extent - 1)
        bias = mx.take_along_axis(profiles, index[None, None], axis=-1)
        bias = mx.where(in_band, bias, mx.zeros_like(bias))
        if self.mask is not None:
            mask = self.mask[..., q0:q1, k0:k1]
            bias = mx.where(mask, bias, mx.finfo(bias.dtype).min)
        return bias


class MLP(nn.Module):
    """The dense SwiGLU MLP of the first layers, scaled by a learned scalar."""

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.global_scale = mx.ones((1,))

    def __call__(self, x: mx.array) -> mx.array:
        out = self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
        return out * self.global_scale.astype(out.dtype)


class Router(nn.Module):
    """``InklingTopkRouter``: sigmoid scores plus a correction bias choose the routed
    experts; the chosen experts and the shared experts then share one softmax over their
    log-sigmoid logits, scaled by ``route_scale`` and a learned scalar."""

    def __init__(self, args: TextArgs):
        super().__init__()
        self.num_routed = args.n_routed_experts
        self.top_k = args.num_experts_per_tok
        self.route_scale = args.route_scale
        self.weight = mx.zeros(
            (args.n_routed_experts + args.n_shared_experts, args.hidden_size)
        )
        self.e_score_correction_bias = mx.zeros((args.n_routed_experts,))
        self.global_scale = mx.ones((1,))

    def route(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """The chosen routed experts, their weights, and the shared experts' weights."""
        # The kernel's softmax holds four values per SIMD lane, as MLX's does.
        chosen = self.top_k + self.weight.shape[0] - self.num_routed
        if _metal_kernels_available() and chosen <= 4 * 32:
            return _route_by_kernel(
                x @ self.weight.T,
                self.e_score_correction_bias,
                self.global_scale,
                self.num_routed,
                self.top_k,
                self.route_scale,
            )
        route = _route_compiled if _decoding(x) else _route
        return route(
            x,
            self.weight,
            self.e_score_correction_bias,
            self.global_scale,
            self.num_routed,
            self.top_k,
            self.route_scale,
        )


def _route(
    x: mx.array,
    weight: mx.array,
    correction: mx.array,
    global_scale: mx.array,
    num_routed: int,
    top_k: int,
    route_scale: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """``Router.route``'s arithmetic (compiled below for decoding)."""
    logits = (x @ weight.T).astype(mx.float32)
    routed = logits[..., :num_routed]
    shared = logits[..., num_routed:]
    choice = mx.sigmoid(routed) + correction.astype(mx.float32)
    indices = mx.argpartition(-choice, kth=top_k - 1, axis=-1)[..., :top_k]
    chosen = mx.concatenate(
        [mx.take_along_axis(routed, indices, axis=-1), shared], axis=-1
    )
    weights = mx.softmax(nn.log_sigmoid(chosen), axis=-1, precise=True)
    weights = weights * (route_scale * global_scale.astype(mx.float32))
    return indices, weights[..., :top_k], weights[..., top_k:]


_route_compiled = mx.compile(_route)


# One SIMD group per token. The same arithmetic as ``_route`` as MLX computes it: the
# sigmoid of its unary ops, the stable sort its argpartition runs (the chosen experts come
# out by decreasing score, ties by index), the log-add-exp of its binary ops, and the
# lane layout and reductions of its softmax kernel (4 values per lane, a SIMD sum), so the
# results are the same bits.
_ROUTE_SOURCE: Final = """
    constexpr int PER_LANE = (ROUTED + 31) / 32;
    constexpr int CHOSEN = TOP_K + SHARED;
    const uint lane = thread_position_in_grid.x;
    const uint position = thread_position_in_grid.y;
    const device T* row = logits + size_t(position) * (ROUTED + SHARED);

    float choice[PER_LANE];
    for (int j = 0; j < PER_LANE; ++j) {
        const int e = int(lane) + 32 * j;
        if (e < ROUTED) {
            const float x = static_cast<float>(row[e]);
            const float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
            const float score = (x < 0) ? y : 1 - y;
            choice[j] = score + static_cast<float>(correction[e]);
        } else {
            choice[j] = -INFINITY;
        }
    }
    int chosen[TOP_K];
    for (int r = 0; r < TOP_K; ++r) {
        float best = -INFINITY;
        int best_index = 0x7fffffff;
        for (int j = 0; j < PER_LANE; ++j) {
            const int e = int(lane) + 32 * j;
            if (e < ROUTED && (choice[j] > best || (choice[j] == best && e < best_index))) {
                best = choice[j];
                best_index = e;
            }
        }
        const float top = simd_max(best);
        const int index = simd_min(best == top ? best_index : 0x7fffffff);
        chosen[r] = index;
        if (int(lane) == index % 32) {
            choice[index / 32] = -INFINITY;
        }
    }

    // log-sigmoid of the chosen experts' logits, then the shared experts'
    float values[4];
    for (int i = 0; i < 4; ++i) {
        const int k = int(lane) * 4 + i;
        float v = -INFINITY;
        if (k < CHOSEN) {
            const float logit = static_cast<float>(k < TOP_K ? row[chosen[k]] : row[ROUTED + k - TOP_K]);
            const float a = -logit;
            const float b = 0.0f;
            float sum;
            if (metal::isnan(a)) {
                sum = metal::numeric_limits<float>::quiet_NaN();
            } else {
                const float maxval = metal::max(a, b);
                const float minval = metal::min(a, b);
                sum = (minval == -INFINITY || maxval == INFINITY)
                    ? maxval
                    : (maxval + log1p(metal::exp(minval - maxval)));
            }
            v = -sum;
        }
        values[i] = v;
    }
    // softmax (precise), as MLX's single-row softmax kernel
    float maxval = -metal::numeric_limits<float>::max();
    for (int i = 0; i < 4; ++i) {
        maxval = (maxval < values[i]) ? values[i] : maxval;
    }
    maxval = simd_max(maxval);
    float normalizer = 0;
    for (int i = 0; i < 4; ++i) {
        const float e = fast::exp(values[i] - maxval);
        values[i] = e;
        normalizer += e;
    }
    normalizer = simd_sum(normalizer);
    normalizer = 1 / normalizer;
    const float scale = static_cast<float>(route_scale[0]) * static_cast<float>(global_scale[0]);
    for (int i = 0; i < 4; ++i) {
        const int k = int(lane) * 4 + i;
        if (k < TOP_K) {
            weights[size_t(position) * TOP_K + k] = (values[i] * normalizer) * scale;
        } else if (k < CHOSEN) {
            shared[size_t(position) * SHARED + (k - TOP_K)] = (values[i] * normalizer) * scale;
        }
    }
    if (lane < TOP_K) {
        indices[size_t(position) * TOP_K + lane] = uint32_t(chosen[lane]);
    }
"""


@functools.cache
def _route_kernel() -> "mx.fast.MetalKernel":
    return mx.fast.metal_kernel(
        name="inkling_route",
        input_names=["logits", "correction", "global_scale", "route_scale"],
        output_names=["indices", "weights", "shared"],
        source=_ROUTE_SOURCE,
    )


def _route_by_kernel(
    logits: mx.array,
    correction: mx.array,
    global_scale: mx.array,
    num_routed: int,
    top_k: int,
    route_scale: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """``_route`` from the router's logits, in one kernel."""
    leading = logits.shape[:-1]
    shared = logits.shape[-1] - num_routed
    tokens = logits.size // logits.shape[-1]
    indices, weights, shared_weights = _route_kernel()(
        inputs=[
            logits,
            correction,
            global_scale.astype(mx.float32),
            mx.array([route_scale], dtype=mx.float32),
        ],
        template=[
            ("T", logits.dtype),
            ("ROUTED", num_routed),
            ("SHARED", shared),
            ("TOP_K", top_k),
        ],
        grid=(32, tokens, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(*leading, top_k), (*leading, top_k), (*leading, shared)],
        output_dtypes=[mx.uint32, mx.float32, mx.float32],
    )
    return indices, weights, shared_weights


# Below this many rows (tokens x shared experts), every token gathers the shared experts
# the way a routed token gathers its experts; from it on, the rows are laid out expert by
# expert so that each expert's weights are read once.
_SORTED_SHARED_ROWS: Final = 16


@functools.lru_cache(maxsize=16)
def _every_expert(shape: tuple[int, ...]) -> mx.array:
    """Indices 0..count-1 of the shared experts for every token, ``shape[:-1]`` tokens."""
    indices = mx.broadcast_to(mx.arange(shape[-1], dtype=mx.uint32), shape)
    mx.eval(indices)
    return indices


def _gate_activations(gate: mx.array, up: mx.array, gammas: mx.array) -> mx.array:
    return nn.silu(gate) * up * gammas.astype(up.dtype)


_gate_activations_compiled = mx.compile(_gate_activations)


def _halves(gate_up: mx.array) -> tuple[mx.array, mx.array]:
    """The gate and up projections' outputs from a fused gate/up projection."""
    width = gate_up.shape[-1] // 2
    return gate_up[..., :width], gate_up[..., width:]


class RoutedExperts(nn.Module):
    """mlx-lm's ``SwitchGLU`` with the gate and up projections fused into one gathered
    matmul (``gate_up_proj``: the gate's output rows, then the up projection's)."""

    def __init__(self, hidden: int, intermediate: int, count: int):
        super().__init__()
        self.gate_up_proj = SwitchLinear(hidden, 2 * intermediate, count, bias=False)
        self.down_proj = SwitchLinear(intermediate, hidden, count, bias=False)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        # Many tokens: sort them by expert, so each expert's weights are read in order.
        do_sort = indices.size >= 64
        idx = indices
        inverse: mx.array | None = None
        if do_sort:
            per_row = indices.shape[-1]
            flat = indices.flatten()
            order = mx.argsort(flat)
            inverse = mx.argsort(order)
            x = x.flatten(0, -3)[order // per_row]
            idx = flat[order]
        gate, up = _halves(
            cast(mx.array, self.gate_up_proj(x, idx, sorted_indices=do_sort))
        )
        activated = cast(mx.array, swiglu(gate, up))
        x = cast(mx.array, self.down_proj(activated, idx, sorted_indices=do_sort))
        if inverse is not None:
            x = mx.unflatten(x[inverse], 0, indices.shape)
        return x.squeeze(-2)


class SharedExperts(nn.Module):
    """The always-on experts, stored like routed ones (one weight tensor per projection,
    stacked over experts; gate and up fused as in ``RoutedExperts``). As in the
    reference, each expert's gate weight multiplies its activations before the down
    projection, and the experts are summed in float32."""

    def __init__(self, hidden: int, intermediate: int, count: int):
        super().__init__()
        self.count = count
        self.gate_up_proj = SwitchLinear(hidden, 2 * intermediate, count, bias=False)
        self.down_proj = SwitchLinear(intermediate, hidden, count, bias=False)

    def few(self, x: mx.array) -> bool:
        """Whether ``x`` has few enough tokens for ``each_expert``. Quantized experts only:
        MLX gathers unquantized weights for unsorted indices with another kernel than for
        sorted ones, whose sums can differ in the last bit; the sorted layout keeps the
        model PR's results."""
        quantized = isinstance(self.gate_up_proj, QuantizedSwitchLinear)
        return quantized and (x.size // x.shape[-1]) * self.count < _SORTED_SHARED_ROWS

    def each_expert(self, x: mx.array, gammas: mx.array) -> mx.array:
        """A few tokens (decoding): each shared expert's output, ``(..., experts,
        hidden)``, gathered the way a routed token gathers its experts."""
        indices = _every_expert((*x.shape[:-1], self.count))
        rows = mx.expand_dims(x, (-2, -3))
        gate, up = _halves(cast(mx.array, self.gate_up_proj(rows, indices)))
        activated = _gate_activations_compiled(gate, up, gammas[..., None, None])
        return cast(mx.array, self.down_proj(activated, indices)).squeeze(-2)

    def __call__(self, x: mx.array, gammas: mx.array) -> mx.array:
        hidden = x.shape[-1]
        tokens = x.size // hidden
        if self.few(x):
            return self.each_expert(x, gammas).astype(mx.float32).sum(axis=-2)
        rows = x.reshape(-1, 1, hidden)
        # Every token goes to every shared expert: lay the rows out expert by expert, so the
        # gathered matmuls see sorted indices and read each expert's weights once rather
        # than once per token.
        rows = mx.concatenate([rows] * self.count, axis=0)
        indices = mx.repeat(mx.arange(self.count, dtype=mx.uint32), tokens)
        gate, up = _halves(
            cast(mx.array, self.gate_up_proj(rows, indices, sorted_indices=True))
        )
        weights = gammas.reshape(tokens, self.count).T.reshape(-1, 1, 1)
        activated = _gate_activations(gate, up, weights)
        down = cast(mx.array, self.down_proj(activated, indices, sorted_indices=True))
        out = down.reshape(self.count, tokens, hidden).astype(mx.float32).sum(axis=0)
        return out.reshape(*x.shape[:-1], hidden)


def _combine_experts(routed: mx.array, weights: mx.array, shared: mx.array) -> mx.array:
    """The routed experts' outputs weighted and summed in float32, plus the shared
    experts' (float32) sum, rounded once."""
    total = (routed.astype(mx.float32) * weights[..., None]).sum(axis=-2) + shared
    return total.astype(routed.dtype)


_combine_experts_compiled = mx.compile(_combine_experts)


# One thread per output element: the routed experts' outputs times their weights, summed
# one after another in float32, plus the shared experts' outputs summed the same way, then
# rounded once: the order in which MLX's column reduction adds a handful of rows, so the
# result is the same bits as ``_combine_experts`` on the shared experts' float32 sum.
_COMBINE_SOURCE: Final = """
    const uint h = thread_position_in_grid.x;
    const uint position = thread_position_in_grid.y;
    const int hidden = routed_shape[routed_ndim - 1];
    if (int(h) >= hidden) {
        return;
    }
    const size_t base = size_t(position) * TOP_K;
    float total = 0.0f;
    for (int k = 0; k < TOP_K; ++k) {
        const float product = static_cast<float>(routed[(base + k) * hidden + h]) * weights[base + k];
        total = k == 0 ? product : product + total;
    }
    float common = 0.0f;
    if (SUMMED) {
        common = static_cast<float>(shared[size_t(position) * hidden + h]);
    } else {
        for (int k = 0; k < SHARED; ++k) {
            const float value = static_cast<float>(shared[(size_t(position) * SHARED + k) * hidden + h]);
            common = k == 0 ? value : value + common;
        }
    }
    out[size_t(position) * hidden + h] = static_cast<T>(total + common);
"""


@functools.cache
def _combine_kernel() -> "mx.fast.MetalKernel":
    return mx.fast.metal_kernel(
        name="inkling_combine_experts",
        input_names=["routed", "weights", "shared"],
        output_names=["out"],
        source=_COMBINE_SOURCE,
    )


def _combine_by_kernel(
    routed: mx.array, weights: mx.array, shared: mx.array, summed: bool = False
) -> mx.array:
    """``_combine_experts`` in one kernel, from each shared expert's output (``(...,
    experts, hidden)``) or, if ``summed``, their float32 sum as ``_combine_experts``
    takes it."""
    *leading, top_k, hidden = routed.shape
    tokens = routed.size // (top_k * hidden)
    (out,) = _combine_kernel()(
        inputs=[routed, weights, shared],
        template=[
            ("T", routed.dtype),
            ("TOP_K", top_k),
            ("SHARED", 1 if summed else shared.shape[-2]),
            ("SUMMED", summed),
        ],
        grid=(hidden, tokens, 1),
        threadgroup=(min(hidden, 256), 1, 1),
        output_shapes=[(*leading, hidden)],
        output_dtypes=[routed.dtype],
    )
    return out


class MoE(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.gate = Router(args)
        self.experts = RoutedExperts(
            args.hidden_size, args.moe_intermediate_size, args.n_routed_experts
        )
        self.shared_experts = SharedExperts(
            args.hidden_size, args.moe_intermediate_size, args.n_shared_experts
        )

    def __call__(self, x: mx.array) -> mx.array:
        indices, weights, shared_weights = self.gate.route(x)
        routed = self.experts(x, indices)
        if self.shared_experts.few(x) and _metal_kernels_available():
            shared = self.shared_experts.each_expert(x, shared_weights)
            return _combine_by_kernel(routed, weights, shared)
        shared = self.shared_experts(x, shared_weights)
        if _metal_kernels_available():
            return _combine_by_kernel(routed, weights, shared, summed=True)
        combine = _combine_experts_compiled if _decoding(x) else _combine_experts
        return combine(routed, weights, shared)


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_index: int):
        super().__init__()
        self.self_attn = Attention(args, layer_index)
        self.is_sliding: bool = self.self_attn.is_sliding
        self.mlp: MLP | MoE = (
            MLP(args.hidden_size, args.dense_intermediate_size)
            if args.dense_layers[layer_index]
            else MoE(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.attn_sconv = ShortConvolution(args.hidden_size, args.conv_kernel_size)
        self.mlp_sconv = ShortConvolution(args.hidden_size, args.conv_kernel_size)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None,
        conv_mask: mx.array | None,
        cache: KVEntry | None = None,
        conv_cache: ArraysCache | None = None,
    ) -> mx.array:
        attended = self.self_attn(
            self.input_layernorm(x), mask, conv_mask, cache, conv_cache
        )
        h = self.attn_sconv(attended, conv_cache, _ATTN_CONV, conv_mask, skip=x)
        mixed = self.mlp(self.post_attention_layernorm(h))
        out = self.mlp_sconv(mixed, conv_cache, _MLP_CONV, conv_mask, skip=h)
        if conv_cache is not None:
            conv_cache.advance(x.shape[1])
        return out


class InklingTextModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.embed_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[CacheEntry] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        h = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_norm(self.embed_tokens(inputs))
        )
        count = len(self.layers)
        convs: list[ArraysCache | None] = [None] * count
        kvs: list[KVEntry | None] = [None] * count
        if cache is not None:
            convs = [cast(ArraysCache, entry) for entry in cache[:count]]
            kvs = [cast(KVEntry, entry) for entry in cache[count:]]
        # The masks come from the first sliding and first global layer of *these* layers,
        # which in a pipeline shard are not the whole model's.
        sliding = [layer.is_sliding for layer in self.layers]
        sliding_mask: mx.array | None = None
        global_mask: mx.array | None = None
        # One sequence's attention works its causal and window masks out from the key
        # distances, in the bias kernel; a batch's masks (with each row's padding) are
        # built here.
        if True in sliding and not derives_causal_mask(kvs[sliding.index(True)]):
            sliding_mask = cast(
                mx.array | None,
                create_attention_mask(
                    h,
                    kvs[sliding.index(True)],
                    window_size=self.args.sliding_window_size,
                    return_array=True,
                ),
            )
        if False in sliding and not derives_causal_mask(kvs[sliding.index(False)]):
            global_mask = cast(
                mx.array | None,
                create_attention_mask(h, kvs[sliding.index(False)], return_array=True),
            )
        conv_mask = cast(mx.array | None, create_ssm_mask(h, convs[0]))
        for layer, kv_cache, conv_cache, is_sliding in zip(
            self.layers, kvs, convs, sliding, strict=True
        ):
            h = layer(
                h,
                sliding_mask if is_sliding else global_mask,
                conv_mask,
                cache=kv_cache,
                conv_cache=conv_cache,
            )
        return self.norm(h)


_RENAMES: Final[tuple[tuple[str, str], ...]] = (
    (r"^embed\.", "model.embed_tokens."),
    (r"^embed_norm\.", "model.embed_norm."),
    (r"^norm\.", "model.norm."),
    (r"^unembed\.", "lm_head."),
    (r"^layers\.", "model.layers."),
    (r"\.attn\.wq_du\.", ".self_attn.q_proj."),
    (r"\.attn\.wk_dv\.", ".self_attn.k_proj."),
    (r"\.attn\.wv_dv\.", ".self_attn.v_proj."),
    (r"\.attn\.wr_du\.", ".self_attn.r_proj."),
    (r"\.attn\.wo_ud\.", ".self_attn.o_proj."),
    (r"\.attn\.rel_logits_proj\.proj$", ".self_attn.rel_proj"),
    (r"\.attn\.", ".self_attn."),
    (r"\.attn_norm\.", ".input_layernorm."),
    (r"\.mlp_norm\.", ".post_attention_layernorm."),
    (r"\.mlp\.w2_md\.", ".mlp.down_proj."),
    (r"\.mlp\.gate\.bias$", ".mlp.gate.e_score_correction_bias"),
    (r"\.experts\.w2_weight$", ".experts.down_proj.weight"),
    (r"\.shared_experts\.shared_w2_weight$", ".shared_experts.down_proj.weight"),
)
# Fused gate and up projections, interleaved row by row (gate, up, gate, up, ...) along
# the axis before the input dimension, and where their halves go ("{}" is gate_proj or
# up_proj).
_FUSED_GATE_UP: Final[tuple[tuple[str, str], ...]] = (
    (r"\.mlp\.w13_dn\.weight$", ".mlp.{}.weight"),
    (r"\.experts\.w13_weight$", ".experts.{}.weight"),
    (r"\.shared_experts\.shared_w13_weight$", ".shared_experts.{}.weight"),
)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        text = args.text
        self.text_args = text
        self.model = InklingTextModel(text)
        self.lm_head = nn.Linear(text.hidden_size, text.unpadded_vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[CacheEntry] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        return self.lm_head(out / self.text_args.logits_mup_width_multiplier)

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.model.layers

    def make_cache(self) -> list[CacheEntry]:
        """Every layer's short-convolution states, then every layer's keys and values."""
        window = self.text_args.sliding_window_size
        convs: list[CacheEntry] = [ArraysCache(4) for _ in self.layers]
        kvs: list[CacheEntry] = [
            RotatingKVCache(max_size=window) if layer.is_sliding else KVCache()
            for layer in self.layers
        ]
        return convs + kvs

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Accepts the official checkpoint layout (``model.llm.*``: fused, interleaved
        gate/up projections, convolutions as (channels, 1, kernel)) and MLX conversions
        that keep its names but split the gate/up projections (and may quantize them).
        The vision and audio towers and the multi-token-prediction layers are dropped."""
        out: dict[str, mx.array] = {}
        unpadded = self.text_args.unpadded_vocab_size
        for name, value in weights.items():
            if not name.startswith("model.llm."):
                if name.startswith(("model.", "lm_head.")) and not name.startswith(
                    ("model.mtp.", "model.visual.", "model.audio.")
                ):
                    out[name] = value  # already in this model's layout
                continue
            key = name[len("model.llm.") :]
            fused = next(
                (
                    re.sub(pattern, target, key)
                    for pattern, target in _FUSED_GATE_UP
                    if re.search(pattern, key)
                ),
                None,
            )
            if fused is not None:
                template = _rename(fused)
                gate, up = value[..., 0::2, :], value[..., 1::2, :]
                if ".mlp.experts." in template or ".shared_experts." in template:
                    out[template.format("gate_up_proj")] = mx.concatenate(
                        [gate, up], axis=-2
                    )
                else:
                    out[template.format("gate_proj")] = gate
                    out[template.format("up_proj")] = up
                continue
            key = _rename(key)
            if (
                key.endswith("sconv.weight")
                and value.shape[1] == 1
                and value.shape[2] > 1
            ):
                value = value.swapaxes(1, 2)
            if key.startswith("lm_head.") and value.shape[0] > unpadded:
                value = value[:unpadded]
            out[key] = value
        out = _fuse_input_projections(out)
        out = _fuse_expert_gate_up(out)
        # exo loads checkpoints non-strictly: a layout this model can't read would run on
        # its initial weights without an error. The embeddings and the LM head are in
        # every shard's files.
        for required in ("model.embed_tokens.weight", "lm_head.weight"):
            if required not in out:
                raise ValueError(
                    "Unrecognised Inkling checkpoint layout: expected the official "
                    f"tensor names (model.llm.*), found none for {required}"
                )
        return out


_INPUT_PROJECTIONS: Final = ("q_proj", "k_proj", "v_proj", "r_proj")


def _fuse_input_projections(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Each attention layer's q/k/v/r projections (and their quantization scales and
    biases, if any) stacked along the output rows into ``qkvr_proj``."""
    prefixes = {
        name.removesuffix(".q_proj.weight")
        for name in weights
        if name.endswith(".self_attn.q_proj.weight")
    }
    for prefix in prefixes:
        for part in ("weight", "scales", "biases"):
            names = [
                f"{prefix}.{projection}.{part}" for projection in _INPUT_PROJECTIONS
            ]
            present = [name in weights for name in names]
            if not any(present):
                continue
            if not all(present):
                missing = [
                    n for n, here in zip(names, present, strict=True) if not here
                ]
                raise ValueError(f"Inkling checkpoint: {missing} missing")
            weights[f"{prefix}.qkvr_proj.{part}"] = mx.concatenate(
                [weights.pop(name) for name in names], axis=0
            )
    return weights


def _fuse_expert_gate_up(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Routed and shared experts stored with separate gate and up projections (MLX
    conversions), stacked along the output rows into ``gate_up_proj``: gate, then up."""
    gates = [
        n
        for n in weights
        if re.search(r"experts\.gate_proj\.(weight|scales|biases)$", n)
    ]
    for gate_name in gates:
        up_name = gate_name.replace(".gate_proj.", ".up_proj.")
        if up_name not in weights:
            raise ValueError(f"Inkling checkpoint: {up_name} missing")
        weights[gate_name.replace(".gate_proj.", ".gate_up_proj.")] = mx.concatenate(
            [weights.pop(gate_name), weights.pop(up_name)], axis=-2
        )
    return weights


def _rename(key: str) -> str:
    for pattern, replacement in _RENAMES:
        key = re.sub(pattern, replacement, key)
    return key
