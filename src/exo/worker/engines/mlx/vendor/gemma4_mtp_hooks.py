"""MTP target-side hooks for Gemma 4, vendored from mlx-vlm.

mlx-vlm's MTP round loop (``mlx_vlm.generate._mtp_rounds``) calls two
methods on the target language model that don't exist in our pinned
``rltakashige/mlx-lm`` fork's :mod:`mlx_lm.models.gemma4_text`:

1. ``forward_with_capture(inputs, cache, return_hidden, return_shared_kv)``
   -- a forward pass that returns logits **plus** the last decoder
   layer's pre-norm hidden state and a ``{layer_type: (K, V)}``
   snapshot of each layer-type's shared-KV slot (the per-layer-type
   KV cache that Gemma 4's coupled drafter consumes).

2. ``rollback_speculative_cache(caches, gdn_states, accepted, block_size)``
   -- per-layer KV trim + per-row tail-zero used after a partial-
   acceptance round to restore the cache to the accepted-prefix
   state. Gemma 4 has no SSM/GDN cache; ``gdn_states`` is accepted
   for API parity with ``qwen3_5``'s hook (which DFlash uses).

We deliberately vendor these as **package-level functions** taking the
target model as the first argument, instead of monkey-patching
``__call__`` on the loaded instance. Two reasons:

- Python special-method lookup bypasses instance ``__call__``
  attributes, so a true ``__call__`` replacement would have to mutate
  the *class* -- and that mutation would persist for every other
  instance the runner ever loads. A function-level seam keeps the
  surface contained.
- The mlx-lm forward already does everything we need (per-layer KV
  iteration, per-layer-type ``previous_kvs`` indirection,
  ``embed_scale`` multiplication, masks). The hook just adds two
  capture buffers around the existing layer loop. Vendoring lets us
  share code with mlx-lm's own decode path -- normal forward continues
  to use ``Model.__call__`` unchanged.

Why "vendor" and not "import from mlx-vlm"
------------------------------------------
The hook lives on mlx-vlm's ``Gemma4TextModel`` class (the multimodal
sibling of ours). It cannot be reused directly because:

- mlx-vlm's ``Gemma4TextModel`` and ours are different classes (different
  parents, different attribute spellings -- ``inputs_embeds`` vs
  ``input_embeddings``, ``get_per_layer_inputs`` vs
  ``_get_per_layer_inputs``, etc.).
- mlx-vlm wraps the LM in a multimodal ``Model`` whose forward returns
  ``LanguageModelOutput`` -- structurally incompatible with our
  ``mx.array``-returning forward and would force every call site in
  exo's generator to unwrap.

So we re-implement the two hook functions against mlx-lm's attribute
names, with behaviour that mirrors mlx-vlm's at the layer-loop level.
The vendor source is ``mlx_vlm.models.gemma4.language``: the
``Gemma4TextModel.__call__`` body (lines 463-555 in mlx-vlm 0.5.0) and
the ``LanguageModel.rollback_speculative_cache`` body (lines 608-646).

Vendor-source hash: mlx-vlm 0.5.0 (mlx_vlm/models/gemma4/language.py)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, cast, final

import mlx.core as mx
from mlx_lm.models import gemma4_text as _mlx_lm_gemma4_text
from mlx_lm.models.gemma4_text import Gemma4TextModel
from mlx_lm.models.gemma4_text import Model as Gemma4Model

# mlx-lm's gemma4_text exposes ``logit_softcap`` at module scope but the
# stub in ``.mlx_typings`` doesn't re-export it (the only call site lives
# inside ``Model.__call__``, which never escapes the module). Resolve
# the binding through the imported module so the runtime lookup is
# unambiguous and the typed surface stays clean -- the cast pins the
# signature we vendor against.
_LogitSoftcapFn = Callable[[float, mx.array], mx.array]
_logit_softcap: _LogitSoftcapFn = cast(
    _LogitSoftcapFn,
    _mlx_lm_gemma4_text.logit_softcap,  # pyright: ignore[reportAttributeAccessIssue]
)


# Attribute name we use to mark a target instance as "MTP hooks attached".
# Set by :func:`attach_mtp_hooks`; reading lets ``mlx_generate`` dispatch
# verify the target is hook-capable without re-running ``isinstance``
# against a type-import that pulls mlx-lm's gemma4_text into every cold
# code path.
_MTP_HOOKS_ATTACHED_ATTR: Final[str] = "_exo_mtp_hooks_attached"


@final
@dataclass(frozen=True, kw_only=True)
class Gemma4MTPForwardOutput:
    """Captured output of an MTP-flavoured Gemma 4 forward pass.

    Mirrors the three fields that mlx-vlm's ``LanguageModelOutput``
    exposes and that ``_mtp_rounds`` reads:

    - ``logits``: ``[B, T, vocab]`` logits (post-softcap, post-lm-head).
    - ``hidden_states``: list of ``[B, T, hidden]`` pre-norm hidden
      tensors. Empty when the caller did not request hidden capture;
      otherwise contains the last decoder layer's output (or the
      layers named in ``capture_layer_ids`` when supplied).
    - ``shared_kv_states``: ``{layer_type: (K, V)}`` snapshot of the
      per-layer-type shared-KV slot at the END of the forward. Empty
      when not requested.

    The ``frozen=True`` discipline matches the rest of exo's typed
    surface even though MLX arrays are themselves mutable -- the
    intent is "don't reassign these fields" rather than "no array
    can ever change".
    """

    logits: mx.array
    hidden_states: list[mx.array]
    shared_kv_states: dict[str, tuple[mx.array, mx.array]]


def resolve_gemma4_text_model(target_model: object) -> Gemma4Model | None:
    """Return the inner ``mlx_lm.models.gemma4_text.Model`` or ``None``.

    mlx-lm exposes Gemma 4 in two shapes that both reach this
    function via :func:`utils_mlx.load_mlx_items`:

    - text-only checkpoints (e.g. ``gemma-4-26b-a4b-it-bf16``) load
      as ``mlx_lm.models.gemma4_text.Model`` -- the inner LM IS the
      ``target_model`` itself.
    - multimodal checkpoints (e.g. ``gemma-4-26b-a4b-it-4bit`` with
      vision) load as ``mlx_lm.models.gemma4.Model``, which wraps the
      same gemma4_text Model under ``.language_model``. Vision /
      multi-modal projector slots are stripped at load time, so the
      wrapper exists purely to keep the model-id → class map flat.

    The MTP hook surface (``forward_with_capture``,
    ``rollback_speculative_cache``) operates on the gemma4_text
    Model's attributes (``model.embed_tokens``, ``lm_head``,
    ``tie_word_embeddings``, ``final_logit_softcapping``). Walking
    one level lets the dispatch path stay correct for both shapes
    without importing the multimodal wrapper class (which would
    pull mlx-lm's vision deps into every cold path).
    """
    if isinstance(target_model, Gemma4Model):
        return target_model
    inner = getattr(target_model, "language_model", None)
    if isinstance(inner, Gemma4Model):
        return inner
    return None


def attach_mtp_hooks(target_model: object) -> None:
    """Mark a loaded Gemma 4 model as MTP-hooks-attached.

    Idempotent. We don't actually monkey-patch any methods on the
    instance -- the hooks are package-level functions that take the
    target as their first argument -- but we set a sentinel attribute
    so generator dispatch can verify the target is hook-capable.

    The runtime check pairs the coupled-drafter kind declared on the
    card with the actual class we got from mlx-lm's auto-loader.
    Phase 2a's loader can degrade silently when mlx-vlm reports a
    kind we don't dispatch; this gate catches the dual failure mode
    where the card declares ``coupled_drafter`` but the target was
    loaded as something other than a Gemma 4 ``Model`` (e.g. operator
    pointed the card at a non-Gemma checkpoint).

    The sentinel is set on BOTH the outer ``target_model`` (whatever
    mlx-lm handed us) AND the inner ``gemma4_text.Model``. The outer
    write keeps cheap ``has_mtp_hooks(model)`` checks at the dispatch
    site working without forcing every call site to re-walk the
    wrapper; the inner write means the adapter (which always operates
    on the gemma4_text instance) sees the sentinel via the same
    attribute lookup.

    Raises:
        TypeError: when ``target_model`` is not a Gemma 4 target,
            either directly or via a ``language_model`` slot. Caught
            one level up in :mod:`exo.worker.engines.mlx.utils_mlx` to
            log + degrade to standard drafting; never propagates to
            the generator dispatch.
    """
    inner = resolve_gemma4_text_model(target_model)
    if inner is None:
        raise TypeError(
            f"attach_mtp_hooks expected mlx_lm.models.gemma4_text.Model "
            "(directly or via a multimodal wrapper exposing "
            f"``.language_model``); got {type(target_model).__name__!r}. "
            "The card's coupled_drafter must be paired with a Gemma 4 target."
        )
    setattr(target_model, _MTP_HOOKS_ATTACHED_ATTR, True)
    if inner is not target_model:
        setattr(inner, _MTP_HOOKS_ATTACHED_ATTR, True)


def has_mtp_hooks(target_model: object) -> bool:
    """True iff :func:`attach_mtp_hooks` has run on this target.

    Walks the multimodal wrapper -- ``attach_mtp_hooks`` marks both
    the outer wrapper and the inner gemma4_text.Model, but defensive
    callers (e.g. tests that build a wrapper around an already-marked
    inner) get the right answer either way.
    """
    if bool(getattr(target_model, _MTP_HOOKS_ATTACHED_ATTR, False)):
        return True
    inner = resolve_gemma4_text_model(target_model)
    if inner is None or inner is target_model:
        return False
    return bool(getattr(inner, _MTP_HOOKS_ATTACHED_ATTR, False))


def _gemma4_text_forward_with_capture(
    text_model: Gemma4TextModel,
    inputs: mx.array,
    *,
    cache: list[Any] | None,
    hidden_sink: list[mx.array],
    shared_kv_sink: dict[str, tuple[mx.array, mx.array]],
    capture_layer_ids: list[int] | None,
    input_embeddings: mx.array | None,
    per_layer_inputs: mx.array | None,
) -> mx.array:
    """Run a Gemma 4 text-model forward and capture MTP intermediates.

    Mirrors mlx-vlm's ``Gemma4TextModel.__call__`` against mlx-lm's
    attribute spelling. Returns the post-norm hidden ``h`` (same shape
    and semantics as ``Gemma4TextModel.__call__`` returns today), plus
    populates ``hidden_sink`` and ``shared_kv_sink`` in place.

    The capture happens in the existing layer loop -- no extra forward
    pass -- so the cost over a normal forward is one append per layer
    (hidden_sink) and a few dict writes (shared_kv_sink).

    ``capture_layer_ids``: when provided, capture the post-layer
    hidden BEFORE pre-norm at exactly those layer indices. When
    omitted (the MTP common case), capture only the LAST layer's
    output -- matches HF's ``_can_record_outputs={"hidden_states":
    Gemma4TextDecoderLayer}`` behaviour and the slot the assistant
    drafter's ``pre_projection`` was trained against.
    """
    if input_embeddings is None:
        h = text_model.embed_tokens(inputs)
    else:
        h = input_embeddings
    h = h * text_model.embed_scale

    per_layer_inputs_list: list[mx.array | None]
    if text_model.hidden_size_per_layer_input:
        if per_layer_inputs is None:
            per_layer_inputs = text_model._get_per_layer_inputs(
                inputs, input_embeddings
            )
        per_layer_inputs = text_model._project_per_layer_inputs(h, per_layer_inputs)
        per_layer_inputs_list = [
            per_layer_inputs[:, :, i, :] for i, _ in enumerate(text_model.layers)
        ]
    else:
        per_layer_inputs_list = [None] * len(text_model.layers)

    layer_caches: list[Any | None]
    if cache is None:
        layer_caches = [None] * len(text_model.layers)
    else:
        layer_caches = list(cache) + [None] * (len(text_model.layers) - len(cache))

    # ``_make_masks`` returns ``list[Any]`` per the stub; the items are
    # ``mx.array | None`` at runtime (one per layer, may be None for
    # full-attention layers when prefill is unmasked). We cast at the
    # boundary so the layer-call type-checks cleanly.
    masks = cast("list[mx.array | None]", text_model._make_masks(h, layer_caches))

    capture_set: set[int] = (
        set(capture_layer_ids) if capture_layer_ids is not None else set()
    )

    # Per-layer ``(shared_kv, offset)`` tuple. ``DecoderLayer.__call__``
    # returns ``(h, (K, V), offset)`` so the kvs slot is always
    # ``tuple[mx.array, mx.array]`` after a layer runs; ``None``
    # entries hold the unrun-yet placeholder used as the prev-kv
    # source for the first layer of each layer-type.
    intermediates: list[tuple[tuple[mx.array, mx.array] | None, mx.array | None]]
    intermediates = [(None, None)] * len(text_model.layers)
    for idx, (layer, layer_cache, mask, prev_idx, per_layer_input) in enumerate(
        zip(
            text_model.layers,
            layer_caches,
            masks,
            text_model.previous_kvs,
            per_layer_inputs_list,
            strict=True,
        )
    ):
        prev_kv_pair, prev_offset = intermediates[prev_idx]
        h, kvs, offset = layer(
            h,
            mask,
            layer_cache,
            per_layer_input=per_layer_input,
            shared_kv=prev_kv_pair,
            offset=prev_offset,
        )
        intermediates[idx] = (kvs, offset)
        if capture_set and idx in capture_set:
            hidden_sink.append(h)

    # When the caller didn't ask for specific layer ids, fall back to
    # the HF / drafter-trained convention: capture the LAST decoder
    # layer's output BEFORE the final norm. The drafter's
    # ``pre_projection`` head was trained against this pre-norm hidden
    # so we MUST emit it from this slot, not the post-norm slot the
    # standard forward path returns.
    if not capture_set:
        hidden_sink.append(h)

    for idx, layer in enumerate(text_model.layers):
        kvs, _offset = intermediates[idx]
        if kvs is not None:
            shared_kv_sink[layer.layer_type] = kvs

    return text_model.norm(h)


def gemma4_mtp_forward(
    target_model: Gemma4Model,
    inputs: mx.array,
    *,
    cache: list[Any] | None = None,
    return_hidden: bool = True,
    return_shared_kv: bool = True,
    capture_layer_ids: list[int] | None = None,
    input_embeddings: mx.array | None = None,
    per_layer_inputs: mx.array | None = None,
) -> Gemma4MTPForwardOutput:
    """Forward pass with MTP-flavoured intermediate capture.

    Equivalent to ``target_model(inputs, cache=cache, ...)`` for the
    purpose of computing ``logits``, but additionally returns the
    pre-norm last-layer hidden state and per-layer-type shared-KV
    snapshot when requested. When BOTH ``return_hidden=False`` and
    ``return_shared_kv=False`` the call still works -- the sinks are
    populated but ignored -- but the standard ``__call__`` is
    cheaper, so callers should only enter this path when they
    actually need the captures.

    Use case: the MTP round loop's verify step
    (:mod:`exo.worker.engines.mlx.generator.coupled_drafter`).

    Non-coupled traffic continues to use the unwrapped
    ``Model.__call__`` and pays no overhead.
    """
    hidden_sink: list[mx.array] = []
    shared_kv_sink: dict[str, tuple[mx.array, mx.array]] = {}

    out = _gemma4_text_forward_with_capture(
        target_model.model,
        inputs,
        cache=cache,
        hidden_sink=hidden_sink,
        shared_kv_sink=shared_kv_sink,
        capture_layer_ids=capture_layer_ids,
        input_embeddings=input_embeddings,
        per_layer_inputs=per_layer_inputs,
    )
    if target_model.tie_word_embeddings:
        logits = target_model.model.embed_tokens.as_linear(out)
    else:
        logits = target_model.lm_head(out)
    softcap = target_model.final_logit_softcapping
    # The ``.pyi`` stub types ``final_logit_softcapping`` as ``float``
    # (default 30.0), but a sanitized config can pass ``None`` to
    # disable softcapping. Keep the runtime guard; basedpyright sees
    # the comparison as always-true given the stub and we silence it.
    if softcap is not None:  # pyright: ignore[reportUnnecessaryComparison]
        logits = _logit_softcap(softcap, logits)

    return Gemma4MTPForwardOutput(
        logits=logits,
        hidden_states=hidden_sink if return_hidden else [],
        shared_kv_states=shared_kv_sink if return_shared_kv else {},
    )


def gemma4_rollback_speculative_cache(
    target_model: Gemma4Model,
    caches: list[Any],
    gdn_states: object,
    accepted: int | mx.array,
    block_size: int,
) -> int:
    """Rewind target KV caches after a speculative-decoding round.

    Vendored verbatim (modulo type annotations) from mlx-vlm
    ``LanguageModel.rollback_speculative_cache``. Gemma 4 has only
    ``KVCache`` / ``RotatingKVCache`` (no SSM/GDN), so this is a
    simple ``cache.trim(...)`` plus a per-row tail-zero on partial
    acceptance. ``gdn_states`` is accepted (and ignored) for API
    parity with ``qwen3_5``'s hook -- DFlash will route through the
    same call site and pass actual GDN state.

    Returns ``max(accepted)`` (the longest accepted-prefix length
    across the batch, or ``accepted`` itself when the batch dimension
    is 1) so the caller can advance its emit loop without re-reducing
    the array.

    The ``target_model`` argument is unused at runtime -- the function
    operates purely on the cache list -- but is required for API
    parity with mlx-vlm's instance method and to keep dispatch
    self-documenting at the call site (the rollback is a target-side
    operation, not a free-standing utility).
    """
    del target_model, gdn_states
    accepted_arr = mx.array([accepted]) if isinstance(accepted, int) else accepted

    max_a = int(accepted_arr.max().item())
    n = max_a + 1
    trim = block_size - n
    is_batch = accepted_arr.size > 1
    valid_ends = accepted_arr + 1

    # mlx-lm's cache classes (``KVCache``, ``RotatingKVCache``) expose
    # ``trim`` / ``_idx`` / ``keys`` / ``values`` but their stubs
    # don't surface those attributes; the cache list also accepts
    # ``None`` placeholders for KV-shared layer slots. ``hasattr`` is
    # how we distinguish the two without importing every concrete
    # cache class -- DFlash will reuse this same loop and may pass
    # cache types we haven't seen yet, so the duck-typed check is the
    # right contract.
    # mlx-lm's cache classes (``KVCache``, ``RotatingKVCache``) expose
    # ``trim`` / ``_idx`` / ``keys`` / ``values`` at runtime but their
    # ``.pyi`` stubs don't surface those attributes -- the cache list
    # is heterogeneous (some entries are ``None`` placeholders for
    # KV-shared layer slots) and DFlash will reuse this loop with
    # additional cache types we haven't seen yet. ``hasattr`` is the
    # right contract; we silence the per-attribute ``reportAny``
    # noise inside the loop block rather than masking the whole
    # function so any genuinely-untyped surface elsewhere stays loud.
    for raw_cache in cast("list[Any | None]", caches):
        if raw_cache is None:
            continue
        if trim > 0 and hasattr(raw_cache, "trim"):  # pyright: ignore[reportAny]
            raw_cache.trim(trim)  # pyright: ignore[reportAny]
        if (
            is_batch
            and hasattr(raw_cache, "_idx")  # pyright: ignore[reportAny]
            and raw_cache.keys is not None  # pyright: ignore[reportAny]
            and max_a > 0
        ):
            kv_len = int(cast(int, raw_cache._idx))
            ve = cast("list[int]", valid_ends.tolist())
            verify_start = kv_len - n
            for bi in range(accepted_arr.shape[0]):
                start = verify_start + int(ve[bi])
                if start < kv_len:
                    raw_cache.keys[bi, :, start:kv_len, :] = 0  # pyright: ignore[reportAny]
                    raw_cache.values[bi, :, start:kv_len, :] = 0  # pyright: ignore[reportAny]
    return max_a


__all__ = [
    "Gemma4MTPForwardOutput",
    "attach_mtp_hooks",
    "gemma4_mtp_forward",
    "gemma4_rollback_speculative_cache",
    "has_mtp_hooks",
    "resolve_gemma4_text_model",
]
