"""DFlash target-side hooks for Qwen 3.5, vendored from mlx-vlm.

mlx-vlm's DFlash drafter (``mlx_vlm.speculative.drafters.qwen3_dflash``)
calls two methods on the Qwen 3.5 target language model that don't exist
in mlx-lm's :mod:`mlx_lm.models.qwen3_5`:

1. ``forward_with_capture`` -- a forward pass that returns logits **plus**
   per-layer captured hidden states **plus** per-SSM-layer ``gdn_state``
   11-tuples (``q, k, v, a, b, A_log, dt_bias, state, mask, conv_input,
   conv_kernel_size``) that DFlash's round loop walks. Qwen 3.5 has a
   hybrid attention + gated-delta architecture, so the capture surface
   is broader than Gemma 4's KV-only capture: every linear-attention
   layer must export the inputs needed to replay its SSM update.

2. ``rollback_speculative_cache`` -- per-layer KV trim **and** per-row
   SSM-state rewind (via batched ``gated_delta_update`` with a replay
   mask) used after partial-acceptance rounds. Unlike the Gemma 4 hook
   (which discards ``gdn_states``), the Qwen 3.5 hook has to actually
   rewind the gated-delta state because Qwen 3.5 layers alternate
   attention / SSM and the SSM cache is path-dependent.

Why we vendor as functions, not class patches
---------------------------------------------
Identical reasoning to :mod:`gemma4_mtp_hooks`: Python special-method
lookup bypasses instance ``__call__`` attributes, so a true
``__call__`` replacement on ``GatedDeltaNet`` would have to mutate the
*class* and persist for every other instance the runner ever loads.
Function-level vendoring keeps the surface contained.

Why "vendor" and not "import from mlx-vlm"
------------------------------------------
The hook lives on mlx-vlm's ``Qwen3_5Model`` class (the multimodal
sibling). It cannot be reused directly because mlx-vlm's
``Qwen3_5Model`` and mlx-lm's ``Qwen3_5TextModel`` are distinct classes
with different attribute spellings (``inputs_embeds`` vs
``input_embeddings``, the multimodal forward returning
``LanguageModelOutput`` vs an ``mx.array``, an extra ``position_ids``
threading through self-attn for multimodal RoPE, etc.). So we
re-implement against mlx-lm's attribute names with behaviour that
mirrors mlx-vlm's at the layer-loop level.

Vendor source: mlx-vlm 0.5.0
``mlx_vlm/models/qwen3_5/language.py``:

- ``LanguageModel.rollback_speculative_cache`` body (~137 lines)
- ``Qwen3_5Model.__call__`` body (the inner forward with
  ``capture_layer_ids`` / ``hidden_sink`` / ``gdn_sink`` plumbing)
- ``Qwen3_5GatedDeltaNet.__call__`` body (the gdn_sink append between
  q/k normalisation and ``gated_delta_update``)

Status
------
Vendor functions are dispatched end-to-end:
:data:`exo.worker.engines.mlx.generator.coupled_drafter.DISPATCHABLE_COUPLED_DRAFTER_KINDS`
includes ``"dflash"`` and
:class:`~exo.worker.engines.mlx.generator.coupled_drafter.Qwen3_5DFlashTargetAdapter`
delegates ``forward_with_capture`` /
``rollback_speculative_cache`` to the functions in this module.

Numerical validation against a real hybrid Qwen 3.5 target
(gated-delta + attention) is the next operational follow-up, gated
on a hybrid-checkpoint download landing on the bench machine. The
unit test surface
(:mod:`exo.worker.tests.unittests.test_mlx.test_qwen3_5_dflash_hooks`)
exercises the hooks against synthetic models on every run, so
loader / adapter drift surfaces immediately without requiring the
production checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final, cast, final

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_5 as _mlx_lm_qwen3_5
from mlx_lm.models.qwen3_5 import (
    DecoderLayer as Qwen3_5DecoderLayer,
)
from mlx_lm.models.qwen3_5 import (
    GatedDeltaNet as Qwen3_5GatedDeltaNet,
)
from mlx_lm.models.qwen3_5 import (
    Model as Qwen3_5Model,
)
from mlx_lm.models.qwen3_5 import (
    Qwen3_5TextModel,
)
from mlx_lm.models.qwen3_5 import (
    TextModel as Qwen3_5LanguageModel,
)

# Module-level helpers from mlx-lm's qwen3_5 that the captured forward
# and rollback both need. mlx-lm's stub doesn't re-export them through
# the typed surface, so we resolve through the module reference and
# cast to a typed callable. Same pattern as
# :mod:`gemma4_mtp_hooks._logit_softcap`.
_CreateMaskFn = Callable[..., "mx.array | None"]
_GatedDeltaUpdateFn = Callable[..., tuple[mx.array, mx.array]]
_SumGradientsFn = Callable[[Any], Callable[[mx.array], mx.array]]

_create_attention_mask: _CreateMaskFn = cast(
    _CreateMaskFn,
    _mlx_lm_qwen3_5.create_attention_mask,  # pyright: ignore[reportAttributeAccessIssue]
)
_create_ssm_mask: _CreateMaskFn = cast(
    _CreateMaskFn,
    _mlx_lm_qwen3_5.create_ssm_mask,  # pyright: ignore[reportAttributeAccessIssue]
)
_gated_delta_update: _GatedDeltaUpdateFn = cast(
    _GatedDeltaUpdateFn,
    _mlx_lm_qwen3_5.gated_delta_update,  # pyright: ignore[reportAttributeAccessIssue]
)
_sum_gradients: _SumGradientsFn = cast(
    _SumGradientsFn,
    _mlx_lm_qwen3_5.sum_gradients,  # pyright: ignore[reportAttributeAccessIssue]
)


# Attribute name that marks a target instance as "DFlash hooks attached".
# Symmetric with :data:`gemma4_mtp_hooks._MTP_HOOKS_ATTACHED_ATTR`. Kept
# on its own constant (rather than a single shared "coupled hooks
# attached" flag) so a target wired for one kind cannot be silently
# mistaken for a target wired for the other.
_DFLASH_HOOKS_ATTACHED_ATTR: Final[str] = "_exo_dflash_hooks_attached"


# Tuple shape the captured forward emits per gated-delta layer and
# :func:`qwen3_5_rollback_speculative_cache` consumes. Mirrors mlx-vlm's
# ``gdn_sink`` entries: (q, k, v, a, b, A_log, dt_bias, state, mask,
# conv_input, conv_kernel_size). The two ``Optional`` slots correspond
# to the optional initial SSM state and the optional padding mask --
# both flow straight through to the rollback's batched
# ``gated_delta_update`` call without being unwrapped.
GdnState = tuple[
    mx.array,  # q -- post-norm queries used by the SSM update
    mx.array,  # k -- post-norm keys
    mx.array,  # v -- values
    mx.array,  # a -- in_proj_a output (gating projection)
    mx.array,  # b -- in_proj_b output (gating projection)
    mx.array,  # A_log
    mx.array,  # dt_bias
    "mx.array | None",  # init_state (carried in from the cache)
    "mx.array | None",  # mask (padding mask used at forward time)
    mx.array,  # conv_input (pre-convolution buffer)
    int,  # conv_kernel_size (K)
]


class DFlashHooksNotImplementedError(RuntimeError):
    """Raised when the DFlash hook surface cannot run for this target.

    Caught by :data:`utils_mlx._COUPLED_HOOK_ATTACH_FALLBACK_EXCEPTIONS`
    so the runner degrades to standard drafting instead of crashing.
    Distinct exception type so logs are unambiguous (``TypeError`` ->
    wrong target architecture; this class -> right architecture, but
    something else prevents the hooks from running -- today reserved
    for build-environment mismatches).
    """


@final
@dataclass(frozen=True, kw_only=True)
class Qwen3DFlashForwardOutput:
    """Captured output of a DFlash-flavoured Qwen 3.5 forward pass.

    Mirrors :class:`gemma4_mtp_hooks.Gemma4MTPForwardOutput` but adapts
    the capture surface to Qwen 3.5's hybrid attention + SSM
    architecture. Where Gemma 4 captures per-layer-type *shared KV*
    slots, Qwen 3.5 captures per-layer ``gdn_states`` 11-tuples that
    encode everything the rollback needs to replay each gated-delta
    update.

    - ``logits``: ``[B, T, vocab]`` post-LM-head logits, computed via
      ``lm_head`` (or tied embeddings when
      ``args.tie_word_embeddings``).
    - ``hidden_states``: list of ``[B, T, hidden]`` post-decoder-layer
      hiddens (one per ``capture_layer_ids`` entry, in layer order).
      Empty when the caller did not request hidden capture.
    - ``gdn_states``: list of :data:`GdnState` tuples in
      decoder-layer order, ONE per gated-delta (linear) layer
      encountered. Empty when ``capture_gdn_states`` is ``False``.

    Frozen-dataclass discipline matches the rest of exo's typed surface
    even though MLX arrays are themselves mutable. The default factories
    let call sites that don't care about a capture leave the kwarg out
    rather than allocate sentinel lists.
    """

    logits: mx.array
    hidden_states: list[mx.array] = field(default_factory=list)
    gdn_states: list[GdnState] = field(default_factory=list)


def resolve_qwen3_5_text_model(target_model: object) -> Qwen3_5TextModel | None:
    """Return the inner :class:`Qwen3_5TextModel` or ``None``.

    Qwen 3.5 in mlx-lm is wrapped by an outer ``Model`` whose
    ``language_model`` is a ``TextModel`` whose ``model`` is the inner
    ``Qwen3_5TextModel`` (the layer walker we need). This helper unwraps
    either shape so call sites stay shape-agnostic regardless of which
    layer of the wrapper exo's loader hands us.
    """
    if isinstance(target_model, Qwen3_5TextModel):
        return target_model
    language_model: object = getattr(target_model, "language_model", None)
    if language_model is not None:
        text_model: object = getattr(language_model, "model", None)
        if isinstance(text_model, Qwen3_5TextModel):
            return text_model
        if isinstance(language_model, Qwen3_5TextModel):
            return language_model
    text_model = getattr(target_model, "model", None)
    if isinstance(text_model, Qwen3_5TextModel):
        return text_model
    return None


def _resolve_lm_head_owner(target_model: object) -> Qwen3_5LanguageModel | None:
    """Return the wrapper module that owns ``lm_head`` / ``args``.

    Used by :func:`qwen3_5_dflash_forward` to decide between
    ``lm_head(h)`` and ``embed_tokens.as_linear(h)`` based on the
    target's ``tie_word_embeddings`` config. The wrapper is either
    ``target_model.language_model`` (mlx-lm ``Model`` shape) or
    ``target_model`` itself (raw ``TextModel`` shape).
    """
    language_model = getattr(target_model, "language_model", None)
    if isinstance(language_model, Qwen3_5LanguageModel):
        return language_model
    if isinstance(target_model, Qwen3_5LanguageModel):
        return target_model
    return None


def attach_dflash_hooks(target_model: object) -> None:
    """Mark a Qwen 3.5 target as DFlash-hooks-attached.

    Sets a sentinel attribute on both the outer wrapper and the inner
    text model so :func:`has_dflash_hooks` can answer either-or without
    a ``resolve_qwen3_5_text_model`` round-trip on every dispatch. The
    sentinel is the ONLY mutation we apply to the loaded model -- the
    captured forward and rollback live as package-level functions that
    take the target as their first argument, exactly mirroring
    :func:`gemma4_mtp_hooks.attach_mtp_hooks`.

    Raises:
        TypeError: ``target_model`` is not (and does not wrap) a
            :class:`Qwen3_5TextModel`. The loader's caller catches this
            via :data:`utils_mlx._COUPLED_HOOK_ATTACH_FALLBACK_EXCEPTIONS`
            and degrades to standard drafting.
    """
    inner = resolve_qwen3_5_text_model(target_model)
    if inner is None:
        raise TypeError(
            "attach_dflash_hooks expected a Qwen 3.5 target "
            "(``Qwen3_5TextModel`` or wrapper); "
            f"got {type(target_model).__name__!r}."
        )
    setattr(target_model, _DFLASH_HOOKS_ATTACHED_ATTR, True)
    if inner is not target_model:
        setattr(inner, _DFLASH_HOOKS_ATTACHED_ATTR, True)


def has_dflash_hooks(target_model: object) -> bool:
    """True iff :func:`attach_dflash_hooks` has run on this target.

    Reads the sentinel set by :func:`attach_dflash_hooks` on either the
    wrapper or the inner text model. Kept separate from
    :func:`gemma4_mtp_hooks.has_mtp_hooks` so the dispatch path can
    answer "this target is wired for DFlash" without conflating it with
    "this target is wired for MTP" -- the two coupled-drafter kinds are
    mutually exclusive on a given runner.
    """
    if bool(getattr(target_model, _DFLASH_HOOKS_ATTACHED_ATTR, False)):
        return True
    inner = resolve_qwen3_5_text_model(target_model)
    if inner is None or inner is target_model:
        return False
    return bool(getattr(inner, _DFLASH_HOOKS_ATTACHED_ATTR, False))


def _gated_delta_net_forward_with_capture(
    layer: Qwen3_5GatedDeltaNet,
    inputs: mx.array,
    mask: mx.array | None,
    cache: object,
    gdn_sink: list[GdnState] | None,
) -> mx.array:
    """Vendor of ``GatedDeltaNet.__call__`` with a ``gdn_sink`` injection.

    Body mirrors :meth:`mlx_lm.models.qwen3_5.GatedDeltaNet.__call__`
    line-for-line. The ONLY behavioural change from the upstream
    forward is the ``gdn_sink.append((...))`` step inserted between the
    q/k normalisation and the call to ``gated_delta_update`` -- exactly
    where :class:`Qwen3_5GatedDeltaNet` in mlx-vlm puts the same append.
    The captured 11-tuple is the canonical :data:`GdnState` shape
    consumed by :func:`qwen3_5_rollback_speculative_cache`.

    The ``cache`` argument is typed ``object`` (rather than the runtime
    ``ArraysCache`` class) because the ``[0]`` / ``[1]`` subscripting
    used here is not surfaced through mlx-lm's typed stub. We narrow
    once at the function entry through ``cast(Any, cache)`` and keep
    every subsequent cache access on the narrowed local; the
    layer-level ``sharding_group``/``training`` attributes share the
    same loose stub surface and are bracketed with their own ``cast``
    boundaries.
    """
    cache_: Any = cast(Any, cache)  # pyright: ignore[reportAny]
    sharding_group: Any = cast(Any, layer.sharding_group)  # pyright: ignore[reportAny]
    inputs_: mx.array = (
        _sum_gradients(sharding_group)(inputs) if sharding_group is not None else inputs
    )

    batch, seq, _ = inputs_.shape

    qkv: mx.array = layer.in_proj_qkv(inputs_)
    z: mx.array = layer.in_proj_z(inputs_).reshape(
        batch, seq, layer.num_v_heads, layer.head_v_dim
    )
    b: mx.array = layer.in_proj_b(inputs_)
    a: mx.array = layer.in_proj_a(inputs_)

    if cache_ is not None and cache_[0] is not None:
        conv_state: mx.array = cast(mx.array, cache_[0])
    else:
        conv_state = mx.zeros(
            (batch, layer.conv_kernel_size - 1, layer.conv_dim),
            dtype=inputs_.dtype,
        )

    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    if cache_ is not None:
        n_keep = layer.conv_kernel_size - 1
        cache_lengths: mx.array | None = cast(
            "mx.array | None",
            getattr(cache_, "lengths", None),  # pyright: ignore[reportAny]
        )
        if cache_lengths is not None:
            ends = mx.clip(cache_lengths, 0, seq)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            cache_[0] = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            cache_[0] = mx.contiguous(conv_input[:, -n_keep:, :])
    conv_out: mx.array = nn.silu(layer.conv1d(conv_input))

    q_split, k_split, v_split = mx.split(
        conv_out, [layer.key_dim, 2 * layer.key_dim], -1
    )
    q = q_split.reshape(batch, seq, layer.num_k_heads, layer.head_k_dim)
    k = k_split.reshape(batch, seq, layer.num_k_heads, layer.head_k_dim)
    v = v_split.reshape(batch, seq, layer.num_v_heads, layer.head_v_dim)

    state: mx.array | None = cast(
        "mx.array | None", cache_[1] if cache_ is not None else None
    )
    # CRITICAL: keep ``inv_scale`` a plain Python float (NOT
    # ``mx.array(...)``), mirroring mlx-lm's upstream
    # ``GatedDeltaNet.__call__`` exactly. ``mx.array(scalar)`` is a
    # float32 0-D array, which promotes the subsequent
    # ``inv_scale * q`` to float32 even when ``q`` is bf16, and the
    # promoted dtype cascades into the next full-attention layer's
    # SDPA call. On Apple Silicon the float32 SDPA kernel for
    # ``head_dim=256, bq=32`` (Qwen 3.5's hybrid layout with a
    # 16-token block-diffusion drafter verify) exceeds Metal's 32 KB
    # threadgroup memory and raises ``Unable to load kernel
    # steel_attention_float32_bq32_bk16_bd256_...``. A plain scalar
    # preserves the operand's dtype under MLX's promotion rules and
    # keeps the bf16 attention kernel reachable.
    head_k_dim: int = k.shape[-1]
    inv_scale = cast(float, head_k_dim**-0.5)
    q = inv_scale * q * mx.rsqrt((q * q).sum(axis=-1, keepdims=True) + 1e-6)
    k = k * mx.rsqrt((k * k).sum(axis=-1, keepdims=True) + 1e-6)

    # Inject the SSM-state capture exactly where mlx-vlm's
    # Qwen3_5GatedDeltaNet inserts ``gdn_sink.append((...))``.
    if gdn_sink is not None:
        gdn_sink.append(
            (
                q,
                k,
                v,
                a,
                b,
                layer.A_log,
                layer.dt_bias,
                state,
                mask,
                conv_input,
                int(layer.conv_kernel_size),
            )
        )

    training = cast(bool, layer.training)
    out, state_out = _gated_delta_update(
        q,
        k,
        v,
        a,
        b,
        layer.A_log,
        layer.dt_bias,
        state,
        mask,
        use_kernel=not training,
    )

    if cache_ is not None:
        cache_[1] = state_out
        cache_advance = cast(
            "Callable[[int], None] | None",
            getattr(cache_, "advance", None),  # pyright: ignore[reportAny]
        )
        if cache_advance is not None:
            cache_advance(seq)

    out = layer.norm(out, z)
    out = layer.out_proj(out.reshape(batch, seq, -1))

    if sharding_group is not None:
        out = mx.distributed.all_sum(out, group=sharding_group)  # pyright: ignore[reportAny]

    return out


def _decoder_layer_forward_with_capture(
    layer: Qwen3_5DecoderLayer,
    x: mx.array,
    mask_attention: mx.array | None,
    mask_ssm: mx.array | None,
    cache: object,
    gdn_sink: list[GdnState] | None,
) -> mx.array:
    """Vendor of ``DecoderLayer.__call__`` routing the SSM branch through capture.

    Mirrors :meth:`mlx_lm.models.qwen3_5.DecoderLayer.__call__` but
    swaps the linear-attn call for
    :func:`_gated_delta_net_forward_with_capture` so the gdn_sink hook
    fires at every gated-delta layer. The attention branch is
    unmodified -- DFlash captures attention KVs through the cache
    itself, not through a sink.
    """
    # mlx-lm's ``DecoderLayer`` declares ``linear_attn``/``self_attn``
    # as a class-level discriminated union (only one is bound per
    # instance, decided in ``__init__`` from ``layer_idx %
    # full_attention_interval``). The typed surface of mlx-lm doesn't
    # surface that discriminator, so we resolve through ``cast`` after
    # branching on ``is_linear`` -- runtime guarantees the right
    # attribute exists, basedpyright sees the narrow type at the call
    # site.
    input_layernorm = cast("Callable[[mx.array], mx.array]", layer.input_layernorm)
    pre_attention = input_layernorm(x)
    if bool(layer.is_linear):
        residual = _gated_delta_net_forward_with_capture(
            layer.linear_attn,
            pre_attention,
            mask_ssm,
            cache,
            gdn_sink,
        )
    else:
        self_attn = cast(
            "Callable[[mx.array, mx.array | None, Any], mx.array]",
            layer.self_attn,
        )
        residual = self_attn(pre_attention, mask_attention, cache)
    h = x + residual
    mlp = cast("Callable[[mx.array], mx.array]", layer.mlp)
    post_norm = cast("Callable[[mx.array], mx.array]", layer.post_attention_layernorm)
    return h + mlp(post_norm(h))


def qwen3_5_dflash_forward(
    target: object,
    inputs: mx.array,
    *,
    cache: list[Any] | None = None,
    capture_layer_ids: list[int] | None = None,
    capture_gdn_states: bool | None = None,
    input_embeddings: mx.array | None = None,
) -> Qwen3DFlashForwardOutput:
    """Captured forward over a Qwen 3.5 target.

    Vendor of mlx-vlm's ``Qwen3_5Model.__call__`` plus the surrounding
    ``LanguageModel`` LM head. Returns logits + per-layer hidden
    captures + per-SSM-layer ``GdnState`` tuples.

    Capture flag semantics mirror mlx-vlm's ``LanguageModel.__call__``
    in ``mlx_vlm.models.qwen3_5.language``: when ``capture_layer_ids``
    is non-empty mlx-vlm allocates BOTH ``hidden_sink`` and ``gdn_sink``
    unconditionally, because the round-loop driver
    (:func:`mlx_vlm.generate._dflash_rounds`) reads
    ``verify_out.gdn_states`` immediately after every verify forward
    to drive ``rollback_speculative_cache``. We mirror that contract:
    if ``capture_gdn_states`` is left at its default ``None``, gdn
    capture is enabled whenever ``capture_layer_ids`` is non-empty.
    Tests that want to check the flag independently can still pass
    ``capture_gdn_states=False`` explicitly to suppress the sink
    allocation.

    Raises:
        TypeError: ``target`` is not (and does not wrap) a Qwen 3.5
            text model. Loader-fallback territory.
    """
    inner = resolve_qwen3_5_text_model(target)
    if inner is None:
        raise TypeError(
            "qwen3_5_dflash_forward expected a Qwen 3.5 target "
            "(``Qwen3_5TextModel`` or wrapper); "
            f"got {type(target).__name__!r}."
        )

    h: mx.array = (
        inner.embed_tokens(inputs) if input_embeddings is None else input_embeddings
    )

    layers = list(inner.layers)
    layer_caches: list[Any] = [None] * len(layers) if cache is None else list(cache)
    if len(layer_caches) != len(layers):
        raise ValueError(
            f"qwen3_5_dflash_forward: cache length ({len(layer_caches)}) does "
            f"not match layer count ({len(layers)})."
        )

    fa_idx = int(inner.fa_idx)
    ssm_idx = int(inner.ssm_idx)
    fa_mask = _create_attention_mask(h, layer_caches[fa_idx])
    ssm_mask = _create_ssm_mask(h, layer_caches[ssm_idx])

    capture_set: set[int] = (
        set(capture_layer_ids) if capture_layer_ids is not None else set()
    )
    hidden_states_out: list[mx.array] = []
    # Mirror mlx-vlm's ``LanguageModel.__call__``: when
    # ``capture_layer_ids`` is provided it allocates both sinks because
    # the round-loop driver reads ``verify_out.gdn_states`` after every
    # verify forward. ``capture_gdn_states=None`` (the default) follows
    # that contract; explicit ``False`` overrides it for tests that want
    # to verify the suppression path.
    gdn_capture_enabled: bool = (
        capture_gdn_states if capture_gdn_states is not None else capture_set != set()
    )
    gdn_sink: list[GdnState] | None = [] if gdn_capture_enabled else None

    for layer_index, layer in enumerate(layers):
        layer_cache: object = cast(object, layer_caches[layer_index])
        h = _decoder_layer_forward_with_capture(
            layer, h, fa_mask, ssm_mask, layer_cache, gdn_sink
        )
        if layer_index in capture_set:
            hidden_states_out.append(h)

    h = inner.norm(h)

    # LM head dispatch mirrors mlx-lm's TextModel.__call__: tied
    # embeddings round-trip through ``embed_tokens.as_linear``,
    # otherwise we lean on the wrapper's ``lm_head``. When exo's loader
    # handed us the inner ``Qwen3_5TextModel`` directly (no wrapper),
    # tied embeddings is the only safe path -- the inner model doesn't
    # own the LM head.
    lm_head_owner = _resolve_lm_head_owner(target)
    if lm_head_owner is not None:
        tie_word_embeddings = bool(
            getattr(lm_head_owner.args, "tie_word_embeddings", False)
        )
        if tie_word_embeddings:
            logits: mx.array = inner.embed_tokens.as_linear(h)
        else:
            logits = lm_head_owner.lm_head(h)
    else:
        # No wrapper -> tied embeddings is the only valid LM-head path.
        # If the model was trained with an explicit ``lm_head`` we
        # cannot route through here without it; the standard forward
        # path's contract states the wrapper owns the head.
        logits = inner.embed_tokens.as_linear(h)

    return Qwen3DFlashForwardOutput(
        logits=logits,
        hidden_states=hidden_states_out,
        gdn_states=gdn_sink if gdn_sink is not None else [],
    )


def qwen3_5_rollback_speculative_cache(
    target: object,
    *,
    caches: list[Any],
    gdn_states: list[GdnState],
    accepted: int | mx.array,
    block_size: int,
) -> int:
    """Per-layer KV trim + per-row SSM rewind after a partial-acceptance round.

    Direct port of mlx-vlm's
    :meth:`mlx_vlm.models.qwen3_5.language.LanguageModel.rollback_speculative_cache`.
    The flow is:

    1. **Separate caches.** Trimmable caches (KV) get ``trim()``-ed in
       place and, in batched mode, have their post-acceptance KV rows
       zeroed so the next forward sees a clean continuation.
    2. **Replay SSM caches.** All non-trimmable caches are batched into
       a single ``gated_delta_update`` call (one kernel launch instead
       of one per layer) using a replay mask that covers ``accepted+1``
       tokens per row. The resulting state is scattered back into each
       cache's slot 1, and slot 0 (conv_input) is rewound to the same
       prefix.

    Returns ``max(accepted)`` so the caller knows how many tokens to
    commit downstream. ``target`` is unused -- accepted for API parity
    with :func:`qwen3_5_dflash_forward`.
    """
    del target  # unused, see docstring

    accepted_array: mx.array = (
        mx.array([accepted]) if isinstance(accepted, int) else accepted
    )

    max_accepted = int(accepted_array.max().item())
    n_keep = max_accepted + 1
    trim = block_size - n_keep
    is_batch = accepted_array.size > 1
    valid_ends = accepted_array + 1

    # First pass: trim KV caches in place; collect SSM caches for the
    # batched gated_delta_update below.
    #
    # mlx-lm's cache classes expose ``is_trimmable``, ``trim``, ``_idx``,
    # ``keys``, and ``values`` at runtime but their ``.pyi`` stubs don't
    # surface those attributes. Same duck-typed contract
    # :func:`gemma4_rollback_speculative_cache` relies on; we silence
    # the per-attribute ``reportAny`` noise inside the loop block
    # rather than masking the whole function so any genuinely-untyped
    # surface elsewhere stays loud.
    ssm_caches: list[Any] = []
    for cache_entry in cast("list[Any | None]", caches):
        if cache_entry is None:
            continue
        if cache_entry.is_trimmable():  # pyright: ignore[reportAny]
            if trim > 0:
                cache_entry.trim(trim)  # pyright: ignore[reportAny]
            if (
                is_batch
                and hasattr(cache_entry, "_idx")  # pyright: ignore[reportAny]
                and cache_entry.keys is not None  # pyright: ignore[reportAny]
                and max_accepted > 0
            ):
                kv_len = int(cast(int, cache_entry._idx))
                ve = cast("list[int]", valid_ends.tolist())
                verify_start = kv_len - n_keep
                for batch_index in range(int(accepted_array.shape[0])):
                    row_start = verify_start + int(ve[batch_index])
                    if row_start < kv_len:
                        cache_entry.keys[batch_index, :, row_start:kv_len, :] = 0  # pyright: ignore[reportAny]
                        cache_entry.values[batch_index, :, row_start:kv_len, :] = 0  # pyright: ignore[reportAny]
        else:
            ssm_caches.append(cache_entry)

    if not ssm_caches:
        return max_accepted

    # Second pass: batch every gated-delta replay into one kernel
    # launch via ``gated_delta_update``. The flatten-then-scatter dance
    # is what mlx-vlm 0.5.0 does to amortize ~30 layer launches into 1.
    n_ssm = len(ssm_caches)
    replay_mask: mx.array | None = None
    if is_batch:
        replay_mask = mx.arange(n_keep)[None, :] <= accepted_array[:, None]

    q_list: list[mx.array] = []
    k_list: list[mx.array] = []
    v_list: list[mx.array] = []
    a_list: list[mx.array] = []
    b_list: list[mx.array] = []
    a_log_list: list[mx.array] = []
    dt_bias_list: list[mx.array] = []
    state_list: list[mx.array] = []
    layer_batch_sizes: list[int] = []
    conv_data: list[tuple[mx.array, int]] = []
    for ssm_layer_index in range(n_ssm):
        (
            q_capture,
            k_capture,
            v_capture,
            a_capture,
            b_capture,
            a_log,
            dt_bias,
            init_state,
            captured_mask,
            conv_input,
            conv_kernel_size,
        ) = gdn_states[ssm_layer_index]
        q_capture = q_capture[:, :n_keep]
        k_capture = k_capture[:, :n_keep]
        v_capture = v_capture[:, :n_keep]
        a_capture = a_capture[:, :n_keep]
        b_capture = b_capture[:, :n_keep]
        batch_rows = int(q_capture.shape[0])
        q_list.append(q_capture)
        k_list.append(k_capture)
        v_list.append(v_capture)
        a_list.append(a_capture)
        b_list.append(b_capture)
        a_log_list.append(
            mx.broadcast_to(a_log[None, None, :], (batch_rows, 1, a_log.shape[0]))
        )
        dt_bias_list.append(
            mx.broadcast_to(dt_bias[None, None, :], (batch_rows, 1, dt_bias.shape[0]))
        )
        if init_state is None:
            init_state = mx.zeros(
                (
                    batch_rows,
                    v_capture.shape[-2],
                    v_capture.shape[-1],
                    k_capture.shape[-1],
                ),
                dtype=mx.float32,
            )
        state_list.append(init_state)
        layer_batch_sizes.append(batch_rows)
        conv_data.append((conv_input, conv_kernel_size))
        if not is_batch and replay_mask is None and captured_mask is not None:
            replay_mask = captured_mask[:, :n_keep]

    q_batched = mx.concatenate(q_list, axis=0)
    k_batched = mx.concatenate(k_list, axis=0)
    v_batched = mx.concatenate(v_list, axis=0)
    a_batched = mx.concatenate(a_list, axis=0)
    b_batched = mx.concatenate(b_list, axis=0)
    a_log_batched = mx.concatenate(a_log_list, axis=0)
    dt_bias_batched = mx.concatenate(dt_bias_list, axis=0)
    state_batched = mx.concatenate(state_list, axis=0)

    # Replay-mask alignment guard: when a batch SSM rollback flattens
    # multiple layers' rows, the mask has to repeat to match. mlx-vlm
    # raises ValueError on misalignment; we keep that contract.
    if replay_mask is not None and replay_mask.shape[0] != q_batched.shape[0]:
        if q_batched.shape[0] % replay_mask.shape[0] != 0:
            raise ValueError(
                "qwen3_5_rollback_speculative_cache: replay mask batch "
                "does not align with flattened SSM rollback rows "
                f"(mask rows={replay_mask.shape[0]}, "
                f"total rows={q_batched.shape[0]})."
            )
        repeats = q_batched.shape[0] // replay_mask.shape[0]
        replay_mask = mx.concatenate([replay_mask] * repeats, axis=0)

    _, states_out = _gated_delta_update(
        q_batched,
        k_batched,
        v_batched,
        a_batched,
        b_batched,
        a_log_batched,
        dt_bias_batched,
        state_batched,
        replay_mask,
        use_kernel=True,
    )

    # Scatter results back to individual caches and rewind conv_input
    # to the accepted-prefix slice. The two branches mirror mlx-vlm:
    # batch dispatch uses per-row accepted offsets; single-batch uses
    # the scalar offset.
    single_accept_offset: int | None = (
        None if is_batch else int(accepted_array[0].item())
    )
    state_offset = 0
    for ssm_layer_index in range(len(ssm_caches)):
        ssm_cache: Any = ssm_caches[ssm_layer_index]  # pyright: ignore[reportAny]
        batch_rows = layer_batch_sizes[ssm_layer_index]
        ssm_cache[1] = states_out[state_offset : state_offset + batch_rows]
        state_offset += batch_rows
        conv_input, conv_kernel_size = conv_data[ssm_layer_index]
        if is_batch:
            acc_list = cast("list[int]", accepted_array.tolist())
            slices: list[mx.array] = [
                conv_input[
                    batch_index : batch_index + 1,
                    int(acc_list[batch_index]) + 1 : int(acc_list[batch_index])
                    + conv_kernel_size,
                ]
                for batch_index in range(int(accepted_array.shape[0]))
            ]
            ssm_cache[0] = mx.concatenate(slices, axis=0)
        else:
            assert single_accept_offset is not None  # narrowed above
            ssm_cache[0] = conv_input[
                :,
                single_accept_offset + 1 : single_accept_offset + conv_kernel_size,
            ]
    return max_accepted


# Re-export ``Qwen3_5Model`` so the ``__all__`` consumers (and the
# loader's runtime ``isinstance`` checks) see the same class object as
# this module's internal references. This is purely a convenience -- the
# canonical home is :mod:`mlx_lm.models.qwen3_5`.
_ = Qwen3_5Model

__all__ = [
    "DFlashHooksNotImplementedError",
    "GdnState",
    "Qwen3DFlashForwardOutput",
    "attach_dflash_hooks",
    "has_dflash_hooks",
    "qwen3_5_dflash_forward",
    "qwen3_5_rollback_speculative_cache",
    "resolve_qwen3_5_text_model",
]
