"""Native MTP (Multi-Token Prediction) sidecar model class for Qwen3.6/Qwen3.5.

This module provides a self-contained ``Model`` class that composes over
stock ``mlx_lm.models.qwen3_5`` and ``mlx_lm.models.qwen3_next`` with
first-class support for Qwen3.6's MTP head. It is intended to graduate
upstream as a new mlx-lm model class (in the same shape as the
``gemma4_assistant`` mlx-lm PR #1276), but lives here in exo's vendor
directory first.

The design corrects seven concrete bugs in the prior upstream attempt
(``ml-explore/mlx-lm`` PR #1226 by chuaaron) that caused the original
work to be self-abandoned:

1. **Strict weight loading.** Loader keeps ``strict=True`` for all
   weights -- main and MTP -- so missing MTP shards raise instead of
   silently initialising to random.
2. **MTP sidecar file.** Qwen3.6 ships MTP weights in a separate
   ``mtp.safetensors`` file (not embedded in the main shards). The
   loader probes that location explicitly.
3. **Per-weight norm-shift gate.** Stock ``Qwen3_5TextModel.sanitize``
   shifts RMSNorm weights by +1 unconditionally when it sees MTP keys.
   This double-shifts already-shifted norms. Our sanitize gates the
   shift on ``value.mean() < 0.5`` per weight, matching MTPLX's
   ``_finalize_mtp_weights``.
4. **Post-norm hidden variant.** MTP was trained to ingest POST-norm
   hidden states (i.e. ``self.model.norm(hidden)``), not the pre-norm
   variant PR #1226 fed in. The ``pre_fc_norm_hidden`` RMSNorm inside
   the MTP module assumes post-norm input.
5. **MTP-specific quantization policies.** Cyankiwi prequantized
   shards quantize only the attention/MLP linears inside MTP; ``fc``
   and the norms stay in BF16. We detect the policy from the key set
   and apply it. "all" prequantized shards also quantize ``fc``.
6. **MTP KV cache priming.** ``make_mtp_cache`` returns the empty
   cache shape; ``mtp_update_cache`` walks prefill tokens through the
   MTP layer with logits suppressed, populating K/V so the draft loop
   inherits prefill context.
7. **Tests check numerical correctness.** The companion test module
   includes both synthetic structural tests (no real download) and an
   opt-in parity test against the real MTPLX artifact that asserts
   top-1 agreement >= 60% against the main lm_head -- this catches
   silent random-init regressions that shape-only tests would miss.

The MTPLX runtime injection in ``mtplx/mtp_patch.py`` is the reference
behaviour we mirror; that file is Apache 2.0 and we re-implement here
in idiomatic mlx-lm class form rather than monkey-patching at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear
from mlx.utils import tree_map
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.qwen3_5 import DecoderLayer as StockDecoderLayer
from mlx_lm.models.qwen3_5 import TextModelArgs as StockTextModelArgs
from mlx_lm.models.qwen3_next import Qwen3NextMLP as MLP  # noqa: N814

# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportPrivateUsage=false, reportIncompatibleMethodOverride=false, reportArgumentType=false, reportOptionalMemberAccess=false
# This module composes over untyped mlx-lm and mlx.nn modules; their
# attribute surface is dynamic and the type-checker can't see through
# nn.Module subclassing in the way mlx-lm uses it.


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MTPWeightsNotFound(RuntimeError):  # noqa: N818 - public API name
    """Raised when ``mtp_num_hidden_layers > 0`` but no MTP weights are loadable.

    Carries the candidate filenames probed so the operator can diagnose
    whether the sidecar wasn't downloaded, mounted, or named as expected.
    """

    def __init__(self, message: str, *, candidates: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.candidates = candidates


# ---------------------------------------------------------------------------
# Model args
# ---------------------------------------------------------------------------


_RMSNORM_SUFFIXES: tuple[str, ...] = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "pre_fc_norm_hidden.weight",
    "pre_fc_norm_embedding.weight",
    "norm.weight",
)


@dataclass
class TextModelArgs(StockTextModelArgs):
    """Stock ``Qwen3_5`` text args extended with MTP fields.

    Defaults match the cyankiwi-prequantized MTPLX contract:
      - ``mtp_num_hidden_layers``: 0 (no MTP unless config declares one)
      - ``mtp_hidden_variant``: 'post_norm' (THE critical PR #1226 fix)
      - ``mtp_concat_order``: 'embedding_hidden' (embedding first)
    """

    mtp_num_hidden_layers: int = 0
    mtp_hidden_variant: str = "post_norm"
    mtp_concat_order: str = "embedding_hidden"


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: Dict[str, Any]

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelArgs":
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


# ---------------------------------------------------------------------------
# MTP module
# ---------------------------------------------------------------------------


class MTPModule(nn.Module):
    """The MTP head: ``[pre_fc_norm_hidden, pre_fc_norm_embedding, fc, layers, norm]``.

    ``layers`` is a list of stock ``qwen3_5.DecoderLayer`` instances
    constructed with ``layer_idx = full_attention_interval - 1`` so each
    layer takes the full-attention (not linear-attention) branch
    (``is_linear = False``). This matches MTPLX, which uses the
    full-attention layer for MTP regardless of the main-model layer
    cadence.
    """

    def __init__(self, args: TextModelArgs, n_layers: int) -> None:
        super().__init__()
        fa_layer_idx = args.full_attention_interval - 1
        self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        self.layers = [
            StockDecoderLayer(args, layer_idx=fa_layer_idx) for _ in range(n_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)


# ---------------------------------------------------------------------------
# Inner text model with MTP attached
# ---------------------------------------------------------------------------


class Qwen3_5MTPInner(nn.Module):  # noqa: N801 - mirrors mlx-lm's Qwen3_5TextModel naming
    """Mirrors stock ``Qwen3_5TextModel`` plus an attached ``self.mtp``.

    ``__call__`` accepts ``return_hidden``, ``emit_logits`` and
    ``logits_keep`` kwargs for cooperative use with the MTP draft loop.
    """

    def __init__(self, args: TextModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            StockDecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1
        self.mtp: Optional[MTPModule] = None
        if args.mtp_num_hidden_layers > 0:
            self.mtp = MTPModule(args, args.mtp_num_hidden_layers)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
        return_hidden: bool = False,
    ) -> Union[mx.array, tuple[mx.array, mx.array]]:
        hidden_states = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(inputs)
        )
        if cache is None:
            cache = [None] * len(self.layers)
        fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
        post = self.norm(hidden_states)
        if return_hidden:
            return post, hidden_states
        return post


# ---------------------------------------------------------------------------
# TextModel wrapper (lm_head + cache builders + sanitize)
# ---------------------------------------------------------------------------


class TextModel(nn.Module):
    """Wraps ``Qwen3_5MTPInner`` + ``lm_head``; mirrors stock ``TextModel`` API."""

    def __init__(self, args: TextModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3_5MTPInner(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        # Storage for an optional sidecar weight loader supplied by the
        # outer loader (qwen3_5_mtp_loader). Sanitize uses it to pick up
        # the separate mtp.safetensors file when no MTP keys appear in
        # the main shards. Type: Callable[[], dict[str, mx.array]] | None.
        self._mtp_sidecar_loader: Optional[Callable[[], Dict[str, mx.array]]] = None

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
        return_hidden: bool = False,
    ) -> Union[mx.array, tuple[mx.array, mx.array]]:
        result = self.model(
            inputs,
            cache,
            input_embeddings=input_embeddings,
            return_hidden=return_hidden,
        )
        if return_hidden:
            assert isinstance(result, tuple)
            post, _pre = result
        else:
            assert isinstance(result, mx.array)
            post = result
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(post)
        else:
            logits = self.lm_head(post)
        if return_hidden:
            return logits, post
        return logits

    @property
    def layers(self) -> List[StockDecoderLayer]:
        return self.model.layers

    def make_cache(self) -> List[Any]:
        return [
            ArraysCache(size=2) if layer.is_linear else KVCache()
            for layer in self.layers
        ]

    def make_mtp_cache(self) -> List[Any]:
        if self.model.mtp is None:
            return []
        return [KVCache() for _ in self.model.mtp.layers]

    # -------------------- sanitize --------------------

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Sanitize stock+MTP weights.

        The incoming weight dict is keyed as it appears in safetensors:
        for Qwen3.6-27B that's the ``language_model.model.*`` /
        ``language_model.lm_head.*`` form. (The outer
        :meth:`Model.sanitize` already normalized any other-prefix
        variants into this shape.)

        Behavior:
          1. The MTP submodule's parameters live at
             ``language_model.model.mtp.*``. Any incoming key in the
             ``language_model.mtp.*`` or bare ``mtp.*`` (or
             ``language_model.model.mtp.*``) namespaces is normalized
             into the canonical model-tree key.
          2. If ``mtp_num_hidden_layers > 0`` and no MTP keys are
             present, the sidecar loader (set by the outer loader) is
             invoked. If that still produces no MTP keys,
             :class:`MTPWeightsNotFound` is raised.
          3. Per-weight norm-shift gate: ``v + 1.0`` only when ``v.ndim
             == 1`` and ``v.mean() < 0.5`` and the suffix matches a
             RMSNorm weight. This prevents double-shifting weights that
             are already in the post-shift form.
          4. conv1d axis fix (matches stock).
          5. ``lm_head.weight`` is dropped if ``tie_word_embeddings``.
        """
        # ----- 1. namespace normalization -----
        normalized: Dict[str, mx.array] = {}
        canonical_mtp_prefix = "language_model.model.mtp."
        for key, value in weights.items():
            if key.startswith("language_model.model.mtp."):
                normalized[key] = value
            elif key.startswith("language_model.mtp."):
                normalized[canonical_mtp_prefix + key[len("language_model.mtp.") :]] = (
                    value
                )
            elif key.startswith("mtp."):
                normalized[canonical_mtp_prefix + key[len("mtp.") :]] = value
            else:
                normalized[key] = value

        embedded_mtp_keys = [
            k for k in normalized if k.startswith(canonical_mtp_prefix)
        ]
        wants_mtp = self.args.mtp_num_hidden_layers > 0

        # ----- 2. sidecar fallback -----
        if wants_mtp and not embedded_mtp_keys:
            loader = self._mtp_sidecar_loader
            if loader is not None:
                extra = loader()
                if extra:
                    for k, v in extra.items():
                        # Sidecar keys come in bare ``mtp.*`` form.
                        if k.startswith("mtp."):
                            normalized[canonical_mtp_prefix + k[len("mtp.") :]] = v
                        elif k.startswith(canonical_mtp_prefix):
                            normalized[k] = v
                        else:
                            # Unexpected; let load_weights complain.
                            normalized[k] = v
                    embedded_mtp_keys = [
                        k for k in normalized if k.startswith(canonical_mtp_prefix)
                    ]

        if wants_mtp and not embedded_mtp_keys:
            raise MTPWeightsNotFound(
                "MTP declared in config (mtp_num_hidden_layers="
                f"{self.args.mtp_num_hidden_layers}) but no MTP weights were "
                "found in the main shards and no sidecar loader produced any. "
                "Candidate sidecar files (relative to model dir): "
                "mtp.safetensors, mtp/weights.safetensors, model-mtp.safetensors, "
                "or config.mlx_lm_extra_tensors.mtp_file",
                candidates=(
                    "mtp.safetensors",
                    "mtp/weights.safetensors",
                    "model-mtp.safetensors",
                ),
            )

        # ----- 3-5. tie/conv/norm fixes -----
        if self.args.tie_word_embeddings:
            normalized.pop("language_model.lm_head.weight", None)
            normalized.pop("lm_head.weight", None)

        for k, v in list(normalized.items()):
            if "conv1d.weight" in k and v.shape[-1] != 1:
                normalized[k] = v.moveaxis(2, 1)
            if v.ndim == 1 and any(k.endswith(sfx) for sfx in _RMSNORM_SUFFIXES):
                mean = float(v.astype(mx.float32).mean().item())
                if mean < 0.5:
                    normalized[k] = v + 1.0

        return normalized

    # -------------------- quant predicates --------------------

    @property
    def quant_predicate(self) -> Optional[Callable[[str, nn.Module], Any]]:
        """Main-model quant predicate (delegates to MoE-style if used).

        For Qwen3.6-27B (no MoE), this returns None and per-layer quant
        comes from the ``quantization`` dict in config.json.
        """
        if self.args.num_experts <= 0:
            return None

        def predicate(path: str, _: nn.Module) -> Any:
            if path.endswith(("mlp.gate", "shared_expert_gate")):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self) -> Callable[[str], bool]:
        def predicate(path: str) -> bool:
            return not path.endswith("A_log")

        return predicate


# ---------------------------------------------------------------------------
# Outer Model
# ---------------------------------------------------------------------------


def _classify_mtp_key_set(keys: tuple[str, ...]) -> str:
    """Return one of ``'unquantized'``, ``'cyankiwi'``, ``'all'``.

    The classification is exactly mirrored from MTPLX constants.py:
      - 'cyankiwi': attention + mlp quantized; ``fc`` + norms BF16
      - 'all': everything quantized including ``fc``
      - 'unquantized': bf16 throughout (only ``.weight`` keys, no
        ``.scales`` / ``.biases``)
    """
    has_fc_scales = any(k == "mtp.fc.scales" for k in keys)
    has_attn_scales = any(k.endswith(".self_attn.q_proj.scales") for k in keys)
    if has_fc_scales:
        return "all"
    if has_attn_scales:
        return "cyankiwi"
    return "unquantized"


def _quantize_mtp_module(
    mtp: MTPModule,
    *,
    policy: str,
    bits: int,
    group_size: int,
    mode: str = "affine",
    quant_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Apply MTP-specific quantization according to ``policy``.

    - ``'all'``: quantize every quantizable module in the MTP subtree.
    - ``'cyankiwi'``: quantize attention/MLP linears only; ``fc``,
      ``pre_fc_norm_*``, ``norm`` stay unquantized.
    - ``'unquantized'``: no-op.
    """
    if policy == "unquantized":
        return
    if policy == "all":
        nn.quantize(mtp, group_size=group_size, bits=bits, mode=mode)
        return
    if policy != "cyankiwi":
        raise ValueError(f"Unsupported MTP quantization policy: {policy!r}")

    def predicate(path: str, module: nn.Module) -> Any:
        if path == "fc" or path.startswith("pre_fc_norm") or path == "norm":
            return False
        if path.endswith("mlp.gate"):
            return False
        if path.startswith("layers.") and hasattr(module, "to_quantized"):
            override = (quant_overrides or {}).get(f"language_model.mtp.{path}")
            if override is not None:
                return {
                    "group_size": int(override.get("group_size", group_size)),
                    "bits": int(override.get("bits", bits)),
                    "mode": str(override.get("mode", mode)),
                }
            return {"group_size": group_size, "bits": bits, "mode": mode}
        return False

    nn.quantize(mtp, class_predicate=predicate)


class Model(nn.Module):
    """Outer model class. Exposes ``language_model: TextModel`` + ``mtp_forward``."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(TextModelArgs.from_dict(args.text_config))

    # -------------------- forward --------------------

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
        return_hidden: bool = False,
    ) -> Union[mx.array, tuple[mx.array, mx.array]]:
        return self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            return_hidden=return_hidden,
        )

    # -------------------- MTP forward --------------------

    def _mtp_core(
        self,
        hidden_states: mx.array,
        next_token_ids: mx.array,
        *,
        mtp_cache: Optional[List[Any]],
        concat_order: Optional[str],
        mtp_hidden_variant: str,
        emit_logits: bool,
    ) -> tuple[Optional[mx.array], mx.array]:
        text_model = self.language_model
        inner = text_model.model
        mtp = inner.mtp
        if mtp is None:
            raise RuntimeError("Model was constructed without an MTP head")

        input_embeds = inner.embed_tokens(next_token_ids)
        e = mtp.pre_fc_norm_embedding(input_embeds)
        h = mtp.pre_fc_norm_hidden(hidden_states)
        order = concat_order or text_model.args.mtp_concat_order
        parts = [e, h] if order == "embedding_hidden" else [h, e]
        x = mtp.fc(mx.concatenate(parts, axis=-1))
        layer_cache = mtp_cache[0] if mtp_cache else None
        mask = create_attention_mask(x, layer_cache)
        x = mtp.layers[0](x, mask=mask, cache=layer_cache)
        post_norm = mtp.norm(x)
        if mtp_hidden_variant == "post_norm":
            hidden = post_norm
        elif mtp_hidden_variant == "pre_norm":
            hidden = x
        else:
            raise ValueError(
                f"Unsupported mtp_hidden_variant={mtp_hidden_variant!r}; "
                "use 'post_norm' or 'pre_norm'"
            )
        if not emit_logits:
            return None, hidden
        if text_model.args.tie_word_embeddings:
            logits = inner.embed_tokens.as_linear(post_norm)
        else:
            logits = text_model.lm_head(post_norm)
        return logits, hidden

    def mtp_forward(
        self,
        hidden_states: mx.array,
        next_token_ids: mx.array,
        *,
        mtp_cache: Optional[List[Any]] = None,
        concat_order: Optional[str] = None,
        mtp_hidden_variant: str = "post_norm",
        emit_logits: bool = True,
        return_hidden: bool = False,
    ) -> Union[mx.array, tuple[mx.array, mx.array]]:
        logits, hidden = self._mtp_core(
            hidden_states,
            next_token_ids,
            mtp_cache=mtp_cache,
            concat_order=concat_order,
            mtp_hidden_variant=mtp_hidden_variant,
            emit_logits=emit_logits,
        )
        if not emit_logits:
            # caller asked for cache-update only
            return hidden
        assert logits is not None
        if return_hidden:
            return logits, hidden
        return logits

    def mtp_update_cache(
        self,
        hidden_states: mx.array,
        next_token_ids: mx.array,
        *,
        mtp_cache: Optional[List[Any]] = None,
        concat_order: Optional[str] = None,
    ) -> mx.array:
        """Populate MTP KV cache without emitting logits.

        Used during prefill to seed the MTP cache with K/V derived from
        the prompt history, so subsequent draft calls see prefill
        context. Returns the post-norm hidden the MTP head produced (for
        chained ``mtp_update_cache`` calls if you want them).
        """
        _, hidden = self._mtp_core(
            hidden_states,
            next_token_ids,
            mtp_cache=mtp_cache,
            concat_order=concat_order,
            mtp_hidden_variant="post_norm",
            emit_logits=False,
        )
        return hidden

    # -------------------- cache builders --------------------

    def make_cache(self) -> List[Any]:
        return self.language_model.make_cache()

    def make_mtp_cache(self) -> List[Any]:
        return self.language_model.make_mtp_cache()

    @property
    def layers(self) -> List[StockDecoderLayer]:
        return self.language_model.model.layers

    # -------------------- sanitize --------------------

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Outer sanitize: normalize prefixes, then delegate to TextModel.

        Stock ``qwen3_5.Model.sanitize`` returns weights with the
        ``language_model.`` prefix intact so the outer ``load_weights``
        can route them to ``self.language_model.*``. We keep that
        convention. MTP keys are normalized to ``language_model.mtp.*``
        before delegation so :meth:`TextModel.sanitize` -- which keys off
        bare ``mtp.*`` -- can detect and process them.
        """
        sanitized: Dict[str, mx.array] = {}
        for key, value in weights.items():
            if key.startswith(("vision_tower", "model.visual")):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif key.startswith("language_model."):
                pass
            else:
                key = "language_model." + key
            sanitized[key] = value
        # Hand off to TextModel.sanitize, which sees keys WITH the
        # "language_model." prefix and returns them WITH the prefix
        # preserved (except for MTP keys, which it normalizes).
        return self.language_model.sanitize(sanitized)

    # -------------------- quantize hook --------------------

    @property
    def quant_predicate(self) -> Optional[Callable[[str, nn.Module], Any]]:
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self) -> Callable[[str], bool]:
        return self.language_model.cast_predicate

    # -------------------- shard (DELIBERATELY replicates MTP) --------------------

    def shard(self, group: Optional[Any] = None) -> None:
        """Stock-style shard for main layers; MTP is REPLICATED, not sharded.

        Replicating the MTP module across all nodes keeps the draft loop
        independent of inter-node communication latency, which is the
        whole point of speculative decoding.
        """
        group = group or mx.distributed.init()
        N = group.size()  # noqa: N806 - stock qwen3_5.Model.shard uses N
        rank = group.rank()

        def conv_sharding(
            key_dim: int,
        ) -> Callable[[str, mx.array], tuple[int, list[int]]]:
            return lambda p, w: (0, [key_dim, 2 * key_dim])

        def repeat_kv_layer_inplace(layer: nn.Module, h: int) -> None:
            if h >= N:
                return

            def _repeat(p: mx.array) -> mx.array:
                s = p.shape
                p = p.reshape(h, s[0] // h, *s[1:])
                p = mx.repeat(p, N // h, axis=0)
                p = p.reshape(-1, *s[1:])
                return p

            layer.update(tree_map(_repeat, layer.parameters()))

        for layer in self.layers:
            if layer.is_linear:
                kd = layer.linear_attn.key_dim
                layer.linear_attn.sharding_group = group
                shard_inplace(layer.linear_attn.conv1d, conv_sharding(kd), group=group)
                layer.linear_attn.conv1d.groups //= N
                shard_inplace(
                    layer.linear_attn.in_proj_qkv,
                    "all-to-sharded",
                    segments=[kd, 2 * kd],
                    group=group,
                )
                shard_inplace(
                    layer.linear_attn.in_proj_z, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.linear_attn.in_proj_b, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.linear_attn.in_proj_a, "all-to-sharded", group=group
                )
                layer.linear_attn.dt_bias = mx.contiguous(
                    mx.split(layer.linear_attn.dt_bias, N)[rank]
                )
                layer.linear_attn.A_log = mx.contiguous(
                    mx.split(layer.linear_attn.A_log, N)[rank]
                )
                shard_inplace(layer.linear_attn.out_proj, "sharded-to-all", group=group)
                layer.linear_attn.num_k_heads //= N
                layer.linear_attn.num_v_heads //= N
                layer.linear_attn.key_dim //= N
                layer.linear_attn.value_dim //= N
                layer.linear_attn.conv_dim //= N
            else:
                layer.self_attn.o_proj = shard_linear(
                    layer.self_attn.o_proj, "sharded-to-all", group=group
                )
                layer.self_attn.q_proj = shard_linear(
                    layer.self_attn.q_proj, "all-to-sharded", group=group
                )
                repeat_kv_layer_inplace(
                    layer.self_attn.k_proj, layer.self_attn.num_key_value_heads
                )
                repeat_kv_layer_inplace(
                    layer.self_attn.v_proj, layer.self_attn.num_key_value_heads
                )
                layer.self_attn.k_proj = shard_linear(
                    layer.self_attn.k_proj, "all-to-sharded", group=group
                )
                layer.self_attn.v_proj = shard_linear(
                    layer.self_attn.v_proj, "all-to-sharded", group=group
                )
                layer.self_attn.num_attention_heads //= N
                layer.self_attn.num_key_value_heads = max(
                    1, layer.self_attn.num_key_value_heads // N
                )

            if isinstance(layer.mlp, MLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )

        # MTP is intentionally NOT sharded -- it's replicated on every
        # node so the draft loop is fully local. See class docstring.


# ---------------------------------------------------------------------------
# Public helpers (used by the loader and tests)
# ---------------------------------------------------------------------------


__all__ = [
    "MTPWeightsNotFound",
    "TextModelArgs",
    "ModelArgs",
    "MTPModule",
    "Qwen3_5MTPInner",
    "TextModel",
    "Model",
    "_classify_mtp_key_set",
    "_quantize_mtp_module",
    "_RMSNORM_SUFFIXES",
]
