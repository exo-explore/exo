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
  MLX, reusing mlx-lm's ``SwitchGLU``/``SwitchLinear`` (routed and shared experts),
  ``nn.RMSNorm`` (``mx.fast.rms_norm`` rounds the normalised activations to the input
  dtype before applying the weight, as ``InklingRMSNorm`` does), its caches and
  ``scaled_dot_product_attention``.
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
    and values) rather than transformers' per-layer cache.

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

import math
import re
from dataclasses import dataclass, field
from typing import Final, Literal, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import (
    create_attention_mask,  # pyright: ignore[reportUnknownVariableType]
    create_ssm_mask,  # pyright: ignore[reportUnknownVariableType]
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache
from mlx_lm.models.switch_layers import SwitchGLU, SwitchLinear

__all__ = ["CacheEntry", "Model", "ModelArgs", "TextArgs"]

# Elements of the dense bias materialised per attention call (per batch row). Longer
# prefills are split into query blocks to stay under it.
_BIAS_ELEMENTS_PER_CALL: Final = 1 << 26
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


def key_distances(cache: object | None, num_queries: int, num_keys: int) -> mx.array:
    """Backward distance from each of the ``num_queries`` new tokens to each key the cache
    returned after its update, as ``(num_queries, num_keys)`` int32.

    The newest key is the last query's own. mlx-lm's caches return keys in temporal order
    (``KVCache``, ``BatchKVCache``, and the rotating caches after a multi-token update),
    except that a rotating cache that has wrapped returns its ring buffer after a
    one-token update, whose newest slot is ``_idx - 1``. Rows of a batch are aligned on
    their newest token, so one distance matrix serves every row; padding is masked.
    """
    newest = num_keys - 1
    if cache is not None and isinstance(cache, RotatingKVCache):
        newest = cache._idx - 1
    elif cache is not None and hasattr(cache, "rotated"):  # BatchRotatingKVCache
        newest = cast(int, getattr(cache, "_idx")) - 1  # noqa: B009
    age = (newest - mx.arange(num_keys)) % num_keys
    lag = num_queries - 1 - mx.arange(num_queries)
    return (age[None, :] - lag[:, None]).astype(mx.int32)


def _query_positions(cache: object | None, num_queries: int) -> mx.array:
    """Absolute positions of the new tokens, ``(1 or batch, num_queries)``."""
    offset = 0 if cache is None else cast(int | mx.array, getattr(cache, "offset", 0))
    steps = mx.arange(num_queries)
    if isinstance(offset, mx.array):
        return offset[:, None] + steps[None, :]
    return (steps + offset)[None, :]


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
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.r_proj = nn.Linear(hidden, heads * args.d_rel, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.k_sconv = ShortConvolution(kv_heads * head_dim, args.conv_kernel_size)
        self.v_sconv = ShortConvolution(kv_heads * head_dim, args.conv_kernel_size)
        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        # Bias profiles over the backward distance (InklingRelativeLogits.proj).
        self.rel_proj = mx.zeros((args.d_rel, self.rel_extent))

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None,
        conv_mask: mx.array | None,
        kv_cache: KVEntry | None,
        conv_cache: ArraysCache | None,
    ) -> mx.array:
        batch, length, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_sconv(self.k_proj(x), conv_cache, _K_CONV, conv_mask)
        values = self.v_sconv(self.v_proj(x), conv_cache, _V_CONV, conv_mask)
        relative = self.r_proj(x).reshape(batch, length, self.num_heads, self.d_rel)

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
            positions = _query_positions(kv_cache, length)
            # Only positions past the floor scale; skip the work below it.
            if isinstance(getattr(kv_cache, "offset", 0), mx.array) or (
                int(getattr(kv_cache, "offset", 0)) + length > self.log_scaling_n_floor
            ):
                effective = (positions + 1).astype(mx.float32)
                tau = 1.0 + self.log_scaling_alpha * mx.log(
                    mx.maximum(effective / self.log_scaling_n_floor, 1.0)
                )
                tau = tau[:, None, :, None]
                queries = (queries.astype(mx.float32) * tau).astype(queries.dtype)

        if kv_cache is not None:
            keys, values = cast(
                tuple[mx.array, mx.array],
                kv_cache.update_and_fetch(keys, values),  # pyright: ignore[reportUnknownMemberType]
            )
        num_keys = keys.shape[2]
        distances = key_distances(kv_cache, length, num_keys)

        # (batch, heads, length, rel_extent): each query's bias at each backward distance.
        profiles = (relative @ self.rel_proj.astype(relative.dtype)).transpose(
            0, 2, 1, 3
        )
        if tau is not None:
            profiles = (profiles.astype(mx.float32) * tau).astype(profiles.dtype)

        output = self._attend(
            queries, keys, values, profiles, distances, mask, kv_cache
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)

    def _attend(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        profiles: mx.array,
        distances: mx.array,
        mask: mx.array | None,
        kv_cache: KVCache | RotatingKVCache | None,
    ) -> mx.array:
        length = queries.shape[2]
        num_keys = keys.shape[2]
        block = max(1, _BIAS_ELEMENTS_PER_CALL // (self.num_heads * num_keys))
        if length <= block:
            return self._attend_block(
                queries, keys, values, profiles, distances, mask, kv_cache
            )
        # A long prefill: split the queries. Keys are in temporal order here (a multi-token
        # update), so a sliding-window block only needs the keys its window reaches.
        outputs: list[mx.array] = []
        first_key_of_query = num_keys - length
        if self.window is not None:
            block = max(1, int(math.sqrt(_BIAS_ELEMENTS_PER_CALL / self.num_heads)))
        for start in range(0, length, block):
            stop = min(start + block, length)
            key_stop = first_key_of_query + stop
            key_start = 0
            if self.window is not None:
                key_start = max(0, first_key_of_query + start - (self.window - 1))
            block_mask = (
                None if mask is None else mask[..., start:stop, key_start:key_stop]
            )
            outputs.append(
                self._attend_block(
                    queries[:, :, start:stop],
                    keys[:, :, key_start:key_stop],
                    values[:, :, key_start:key_stop],
                    profiles[:, :, start:stop],
                    distances[start:stop, key_start:key_stop],
                    block_mask,
                    kv_cache,
                )
            )
        return mx.concatenate(outputs, axis=2)

    def _attend_block(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        profiles: mx.array,
        distances: mx.array,
        mask: mx.array | None,
        kv_cache: KVCache | RotatingKVCache | None,
    ) -> mx.array:
        in_band = (distances >= 0) & (distances < self.rel_extent)
        index = mx.clip(distances, 0, self.rel_extent - 1)
        bias = mx.take_along_axis(profiles, index[None, None], axis=-1)
        bias = mx.where(in_band, bias, mx.zeros_like(bias))
        if mask is not None:
            bias = mx.where(mask, bias, mx.finfo(bias.dtype).min)
        return scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=kv_cache,
            scale=self.scale,
            mask=bias.astype(queries.dtype),
        )


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
        logits = (x @ self.weight.T).astype(mx.float32)
        routed = logits[..., : self.num_routed]
        shared = logits[..., self.num_routed :]
        choice = mx.sigmoid(routed) + self.e_score_correction_bias.astype(mx.float32)
        indices = mx.stop_gradient(
            mx.argpartition(-choice, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
        )
        chosen = mx.concatenate(
            [mx.take_along_axis(routed, indices, axis=-1), shared], axis=-1
        )
        weights = mx.softmax(nn.log_sigmoid(chosen), axis=-1, precise=True)
        weights = weights * (self.route_scale * self.global_scale.astype(mx.float32))
        return indices, weights[..., : self.top_k], weights[..., self.top_k :]


class SharedExperts(nn.Module):
    """The always-on experts, stored like routed ones (one weight tensor per projection,
    stacked over experts). As in the reference, each expert's gate weight multiplies its
    activations before the down projection, and the experts are summed in float32."""

    def __init__(self, hidden: int, intermediate: int, count: int):
        super().__init__()
        self.count = count
        self.gate_proj = SwitchLinear(hidden, intermediate, count, bias=False)
        self.up_proj = SwitchLinear(hidden, intermediate, count, bias=False)
        self.down_proj = SwitchLinear(intermediate, hidden, count, bias=False)

    def __call__(self, x: mx.array, gammas: mx.array) -> mx.array:
        hidden = x.shape[-1]
        rows = x.reshape(-1, 1, hidden)
        tokens = rows.shape[0]
        # Every token goes to every shared expert: lay the rows out expert by expert, so the
        # gathered matmuls see sorted indices and read each expert's weights once rather
        # than once per token.
        rows = mx.concatenate([rows] * self.count, axis=0)
        indices = mx.repeat(mx.arange(self.count, dtype=mx.uint32), tokens)
        up = cast(mx.array, self.up_proj(rows, indices, sorted_indices=True))
        gate = cast(mx.array, self.gate_proj(rows, indices, sorted_indices=True))
        weights = gammas.reshape(tokens, self.count).T.reshape(-1, 1, 1)
        activated = nn.silu(gate) * up * weights.astype(up.dtype)
        down = cast(mx.array, self.down_proj(activated, indices, sorted_indices=True))
        out = down.reshape(self.count, tokens, hidden).astype(mx.float32).sum(axis=0)
        return out.reshape(*x.shape[:-1], hidden)


class MoE(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.gate = Router(args)
        self.experts = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.n_routed_experts
        )
        self.shared_experts = SharedExperts(
            args.hidden_size, args.moe_intermediate_size, args.n_shared_experts
        )

    def __call__(self, x: mx.array) -> mx.array:
        indices, weights, shared_weights = self.gate.route(x)
        routed = self.experts(x, indices)
        routed = (routed.astype(mx.float32) * weights[..., None]).sum(axis=-2)
        shared = self.shared_experts(x, shared_weights)
        return (routed + shared).astype(x.dtype)


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
        h = x + self.attn_sconv(attended, conv_cache, _ATTN_CONV, conv_mask)
        mixed = self.mlp(self.post_attention_layernorm(h))
        out = h + self.mlp_sconv(mixed, conv_cache, _MLP_CONV, conv_mask)
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
        if True in sliding:
            sliding_mask = cast(
                mx.array | None,
                create_attention_mask(
                    h,
                    kvs[sliding.index(True)],
                    window_size=self.args.sliding_window_size,
                    return_array=True,
                ),
            )
        if False in sliding:
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
                out[template.format("gate_proj")] = value[..., 0::2, :]
                out[template.format("up_proj")] = value[..., 1::2, :]
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


def _rename(key: str) -> str:
    for pattern, replacement in _RENAMES:
        key = re.sub(pattern, replacement, key)
    return key
