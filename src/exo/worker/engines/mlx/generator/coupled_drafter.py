"""Single-node coupled (mtp/dflash) speculative-decoding dispatch.

mlx-vlm ships a ready-to-use round loop -- :func:`mlx_vlm.generate._mtp_rounds`
-- but it expects the target language model to expose two methods that
exo's pinned ``mlx-lm`` fork doesn't natively provide:

- ``rollback_speculative_cache``
- ``__call__(..., return_hidden=True, return_shared_kv=True)`` returning a
  ``LanguageModelOutput``-shaped object.

We satisfy that contract WITHOUT mutating mlx-lm's classes (which
would persist for every other instance the runner ever loads) by
wrapping the loaded target in :class:`Gemma4MTPTargetAdapter`.
The adapter forwards forward-passes and rollbacks through the
package-level functions in
:mod:`exo.worker.engines.mlx.vendor.gemma4_mtp_hooks`, which were
vendored from mlx-vlm's gemma4 language model.

This module provides:

- :class:`Gemma4MTPTargetAdapter` -- the wrapper that satisfies
  ``_mtp_rounds`` -- and bind-time access to the underlying
  ``embed_tokens`` slot the drafter walks during ``bind``.
- :func:`run_coupled_round_loop` -- a thin generator that drives the
  mlx-vlm round loop given a prefilled target cache and the captured
  prefill intermediates. The caller owns prefill + emission +
  cancellation; this function is a pure round-loop driver, kept narrow
  so it can be swapped for a vendored loop later without disturbing
  ``mlx_generate``'s control flow.
- :class:`CoupledModelDrafter` -- a :class:`Drafter`-protocol shim
  that hides the prefill-capture and round-loop driver from
  ``mlx_generate``. The drafter's ``stream()`` advances the prompt
  cache by a single captured forward, samples the first bonus with
  the request's logits-processors applied, then drives
  :func:`run_coupled_round_loop` and yields ``mlx_lm.GenerationResponse``-
  shaped tokens identically to :class:`ModelDrafter`.

DFlash dispatch (Qwen 3.5): mirrored from the MTP wiring above.
:class:`Qwen3_5DFlashTargetAdapter` is the sibling adapter against the
vendored DFlash hooks (:mod:`exo.worker.engines.mlx.vendor.qwen3_5_dflash_hooks`),
and :func:`run_coupled_round_loop` dispatches between
:func:`mlx_vlm.generate._mtp_rounds` and
:func:`mlx_vlm.generate._dflash_rounds` based on the adapter type.
:class:`CoupledModelDrafter` selects the prefill capture flags
(MTP: ``return_hidden`` + ``return_shared_kv``;
DFlash: ``capture_layer_ids=draft.config.target_layer_ids``) per kind.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast, final

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import GenerationResponse
from mlx_lm.models.gemma4_text import Model as Gemma4Model
from mlx_lm.models.qwen3_5 import Model as Qwen3_5Model
from mlx_lm.models.qwen3_5 import Qwen3_5TextModel
from mlx_lm.models.qwen3_5 import TextModel as Qwen3_5LanguageModel
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.vendor.gemma4_mtp_hooks import (
    Gemma4MTPForwardOutput,
    gemma4_mtp_forward,
    gemma4_rollback_speculative_cache,
    has_mtp_hooks,
    resolve_gemma4_text_model,
)
from exo.worker.engines.mlx.vendor.qwen3_5_dflash_hooks import (
    GdnState,
    Qwen3DFlashForwardOutput,
    has_dflash_hooks,
    qwen3_5_dflash_forward,
    qwen3_5_rollback_speculative_cache,
    resolve_qwen3_5_text_model,
)

if TYPE_CHECKING:
    from exo.worker.engines.mlx.generator.drafter import DraftMode


# Coupled-drafter architecture identifier surfaced on
# :class:`exo.api.types.api.GenerationStats` via ``drafter_kind``. Mirrors
# :data:`exo.worker.engines.mlx.utils_mlx.CoupledDrafterKind` but is duplicated
# here to avoid the import cycle (utils_mlx imports from this package).
CoupledDrafterKind = Literal["mtp", "dflash"]


# The set of coupled-drafter kinds the generator dispatch can actually
# drive end-to-end. Both ``"mtp"`` (Gemma 4 + assistant drafter) and
# ``"dflash"`` (Qwen 3.5 + DFlash drafter) are wired through
# :class:`CoupledModelDrafter` against the architecture-specific
# adapter (:class:`Gemma4MTPTargetAdapter` /
# :class:`Qwen3_5DFlashTargetAdapter`) and round-loop driver
# (:func:`mlx_vlm.generate._mtp_rounds` /
# :func:`mlx_vlm.generate._dflash_rounds`).
#
# Builder-side gates that decide "is a coupled drafter usable for this
# request" consult this set rather than treating
# ``coupled_drafter is not None`` as proof of dispatchability -- a
# future kind would otherwise be forced through :class:`SequentialGenerator`
# (losing batch throughput) before the dispatch is wired.
DISPATCHABLE_COUPLED_DRAFTER_KINDS: frozenset[CoupledDrafterKind] = frozenset(
    {"mtp", "dflash"}
)


def is_coupled_drafter_dispatchable(kind: CoupledDrafterKind) -> bool:
    """Return ``True`` iff the generator dispatch can drive ``kind``.

    Mirrors the dispatch's own kind check in :func:`mlx_generate`.
    Used by :class:`exo.worker.engines.mlx.builder.MlxModelBuilder` to
    gate "drafter loaded" predicates on whether the drafter is
    runnable, not just whether it's loaded.
    """
    return kind in DISPATCHABLE_COUPLED_DRAFTER_KINDS


# mlx-vlm's ``_mtp_rounds`` and ``_dflash_rounds`` are private
# module-level helpers without typed stubs; resolve them lazily through
# ``importlib`` so the type-check narrowing happens at the import
# boundary instead of every call site. The eager-import path would
# force every coupled-drafter call site to ride a multi-line
# ``pyright: ignore`` block.
def _resolve_mtp_rounds_fn() -> Callable[..., Generator[tuple[int, None], None, None]]:
    import importlib

    module: Any = importlib.import_module("mlx_vlm.generate")
    return cast(
        "Callable[..., Generator[tuple[int, None], None, None]]",
        module._mtp_rounds,
    )


def _resolve_dflash_rounds_fn() -> Callable[
    ..., Generator[tuple[int, None], None, None]
]:
    import importlib

    module: Any = importlib.import_module("mlx_vlm.generate")
    return cast(
        "Callable[..., Generator[tuple[int, None], None, None]]",
        module._dflash_rounds,
    )


@final
class Gemma4MTPTargetAdapter:
    """Adapter that exposes the ``_mtp_rounds`` target contract.

    mlx-vlm's ``_mtp_rounds`` does three things with the target it
    receives:

    1. ``lm = model.language_model if hasattr(model, "language_model") else model``
       and then walks ``lm.embed_tokens``, ``lm.embed_scale``, etc.
       via the drafter's ``bind`` step.
    2. ``lm.rollback_speculative_cache(cache, gdn_states, accepted, bs)``.
    3. ``lm(verify_input, cache=..., return_hidden=True, return_shared_kv=True)``
       returning an object with ``.logits``, ``.hidden_states``, and
       ``.shared_kv_states``.

    For (1), the underlying ``mlx_lm.models.gemma4_text.Model``
    already satisfies the structure: ``Model.model.embed_tokens`` is
    populated and the drafter's ``bind`` walks ``model.embed_tokens``
    OR ``model.model.embed_tokens``. The adapter exposes ``model``
    as a passthrough so the drafter binds to the SAME embed_tokens
    instance it would have bound to without the adapter -- no weight
    duplication, no bind-time divergence.

    For (2) and (3), the adapter wires ``rollback_speculative_cache``
    and ``__call__`` through to the vendored hook functions.

    The adapter is a plain class (NOT an ``nn.Module``) -- it holds no
    parameters of its own and the ``__call__`` return type
    (:class:`Gemma4MTPForwardOutput`) is incompatible with
    ``Module.__call__``'s ``mx.array`` return. The wrapped target
    keeps its own parameters and continues to be eligible for
    ``mx.eval`` / cache-resizing as before.
    """

    def __init__(self, target_model: object) -> None:
        # Accept either ``mlx_lm.models.gemma4_text.Model`` directly or
        # the multimodal ``mlx_lm.models.gemma4.Model`` wrapper that
        # exposes the LM under ``.language_model``. Vision-capable
        # checkpoints (e.g. ``gemma-4-26b-a4b-it-4bit``) load as the
        # latter; the adapter's hooks operate on the inner LM either
        # way, so we resolve once at construction time.
        inner = resolve_gemma4_text_model(target_model)
        if inner is None:
            raise TypeError(
                "Gemma4MTPTargetAdapter expected a Gemma 4 target "
                "(``mlx_lm.models.gemma4_text.Model`` directly, or a "
                "multimodal wrapper exposing it via ``.language_model``); "
                f"got {type(target_model).__name__!r}."
            )
        if not has_mtp_hooks(inner):
            # The hook attach is gated by ``utils_mlx.load_mlx_items`` --
            # if we got here without the attach call, the loader's
            # post-load wiring drifted from this dispatch.
            raise RuntimeError(
                "Gemma4MTPTargetAdapter requires a target with attached "
                "MTP hooks; call attach_mtp_hooks(target) at load time. "
                "This is a runtime guard against loader/dispatch drift."
            )
        self._target: Gemma4Model = inner

    @property
    def target(self) -> Gemma4Model:
        """The underlying mlx-lm gemma4 model (escape hatch for tests)."""
        return self._target

    @property
    def model(self) -> nn.Module:
        """``Model.model`` passthrough used by the drafter's ``bind``."""
        return self._target.model

    def __call__(
        self,
        inputs: mx.array,
        *,
        cache: list[Any] | None = None,
        return_hidden: bool = False,
        return_shared_kv: bool = False,
    ) -> Gemma4MTPForwardOutput:
        """Forward pass returning the MTP-flavoured capture tuple.

        ``_mtp_rounds`` always passes ``return_hidden=True`` and
        ``return_shared_kv=True``, so the hot path is the captured-
        forward case. We accept the off variants for API parity.

        Note that the return type is :class:`Gemma4MTPForwardOutput`,
        NOT raw logits. Calling sites that want raw logits (e.g. the
        prefill path before entering the round loop) should call this
        the same way and read ``.logits`` -- the structural shape lets
        ``_mtp_rounds`` read ``.hidden_states[-1]`` and
        ``.shared_kv_states`` directly without an unwrap step.
        """
        return gemma4_mtp_forward(
            self._target,
            inputs,
            cache=cache,
            return_hidden=return_hidden,
            return_shared_kv=return_shared_kv,
        )

    def rollback_speculative_cache(
        self,
        caches: list[Any],
        gdn_states: object,
        accepted: int | mx.array,
        block_size: int,
    ) -> int:
        """Trim target KV caches after partial-acceptance.

        Delegated to :func:`gemma4_rollback_speculative_cache`; see
        that function's docstring for ``gdn_states`` semantics
        (accepted-and-ignored for Gemma 4, used by DFlash on Qwen3).
        """
        return gemma4_rollback_speculative_cache(
            self._target,
            caches=caches,
            gdn_states=gdn_states,
            accepted=accepted,
            block_size=block_size,
        )


@final
class Qwen3_5DFlashTargetAdapter:  # noqa: N801 -- mirrors mlx-lm's "Qwen3_5" naming (version pinned with underscore separator); renaming here would diverge from upstream type naming and obscure the binding.
    """Adapter that exposes the ``_dflash_rounds`` target contract.

    mlx-vlm's :func:`mlx_vlm.generate._dflash_rounds` does three things
    with the target it receives:

    1. ``lm = model.language_model if hasattr(model, "language_model") else model``
       and then walks ``lm.embed_tokens`` etc. via the drafter's
       ``reset(model)`` step.
    2. ``lm.rollback_speculative_cache(prompt_cache, verify_out.gdn_states, accepted, bs)``.
    3. ``lm(verify_input, cache=prompt_cache, capture_layer_ids=target_layer_ids)``
       returning an object with ``.logits``, ``.hidden_states`` (a
       ``list[mx.array]`` indexed by ``capture_layer_ids``), and
       ``.gdn_states`` (a list of ``GdnState`` 11-tuples populated by
       every gated-delta layer touched in the forward).

    Mirrors :class:`Gemma4MTPTargetAdapter` exactly except for the
    capture flag set: where MTP wants ``return_hidden=True,
    return_shared_kv=True``, DFlash wants ``capture_layer_ids=[...]``.
    The underlying contract -- a plain class (NOT an ``nn.Module``)
    holding no parameters of its own, exposing the inner LM through
    ``model`` for drafter binding -- is identical.

    The adapter is Qwen 3.5-specific because that's the only target
    type the vendored DFlash hooks support today.
    """

    def __init__(self, target_model: object) -> None:
        # Accept either ``mlx_lm.models.qwen3_5.Model`` directly (the
        # text-only LM with lm_head) or a multimodal wrapper exposing
        # it via ``.language_model``. The vendored hooks operate on
        # the inner LM either way; we resolve once at construction.
        inner = resolve_qwen3_5_text_model(target_model)
        if inner is None:
            raise TypeError(
                "Qwen3_5DFlashTargetAdapter expected a Qwen 3.5 target "
                "(``mlx_lm.models.qwen3_5.Model`` directly, or a "
                "multimodal wrapper exposing it via ``.language_model``); "
                f"got {type(target_model).__name__!r}."
            )
        if not has_dflash_hooks(inner):
            # Same loader/dispatch drift guard as the MTP adapter:
            # the hook attach is gated by ``utils_mlx.load_mlx_items``,
            # so reaching this code path without the marker means the
            # post-load wiring drifted from this dispatch.
            raise RuntimeError(
                "Qwen3_5DFlashTargetAdapter requires a target with attached "
                "DFlash hooks; call attach_dflash_hooks(target) at load time. "
                "This is a runtime guard against loader/dispatch drift."
            )
        self._target: Qwen3_5TextModel = inner
        # Preserve the original wrapper so :func:`qwen3_5_dflash_forward`
        # can locate ``lm_head`` / ``args.tie_word_embeddings`` via
        # ``_resolve_lm_head_owner``. The inner ``Qwen3_5TextModel`` is
        # the layer walker; ``lm_head`` lives on the enclosing
        # ``TextModel`` / ``Model`` wrapper. Passing the inner alone to
        # the forward would silently force the tied-embeddings code
        # path on untied-head checkpoints (``tie_word_embeddings=False``
        # is common for Qwen 3.5/3.6), corrupting verifier logits and
        # therefore accept / reject decisions in coupled decoding.
        # ``rollback_speculative_cache`` ignores its ``target`` argument
        # but accepts the wrapper for API parity with the forward.
        # The union covers the three shapes the loader can hand us:
        # the inner walker (no wrapper present → tied-only path), the
        # mid wrapper ``TextModel`` (owns ``lm_head`` + ``args``), or
        # the multimodal ``Model`` wrapper (exposes the LM via
        # ``.language_model``). All three are accepted by
        # :func:`qwen3_5_dflash_forward` and
        # :func:`_resolve_lm_head_owner`.
        self._lm_head_owner: Qwen3_5Model | Qwen3_5LanguageModel | Qwen3_5TextModel = (
            cast(
                "Qwen3_5Model | Qwen3_5LanguageModel | Qwen3_5TextModel",
                target_model,
            )
        )

    @property
    def target(self) -> Qwen3_5TextModel:
        """The underlying mlx-lm Qwen 3.5 text model (escape hatch for tests)."""
        return self._target

    @property
    def model(self) -> nn.Module:
        """Inner-model passthrough used by the drafter's ``reset`` step.

        mlx-vlm's DFlash drafter walks ``model.embed_tokens`` (et al.)
        during ``reset``; exposing the underlying text model directly
        keeps the drafter binding to the SAME parameter instances it
        would bind to without the adapter -- no weight duplication.
        Mirrors :meth:`Gemma4MTPTargetAdapter.model` (which exposes
        ``self._target.model``); for Qwen 3.5 the resolved text model
        IS the layer walker the drafter needs, so we surface it as-is.
        """
        return self._target

    def __call__(
        self,
        inputs: mx.array,
        *,
        cache: list[Any] | None = None,
        capture_layer_ids: list[int] | None = None,
    ) -> Qwen3DFlashForwardOutput:
        """Forward pass returning the DFlash-flavoured capture tuple.

        ``_dflash_rounds`` always passes ``capture_layer_ids=target_layer_ids``
        (a non-empty list configured on the drafter) in its verify
        forward. The adapter mirrors mlx-vlm's
        ``LanguageModel.__call__`` semantics: when ``capture_layer_ids``
        is non-empty BOTH ``hidden_states`` and ``gdn_states`` are
        captured automatically (the round loop reads
        ``verify_out.gdn_states`` immediately after every verify to
        drive ``rollback_speculative_cache``).

        The structural shape of the return -- ``.logits``,
        ``.hidden_states``, ``.gdn_states`` -- matches mlx-vlm's
        ``LanguageModelOutput`` so ``_dflash_rounds`` can read those
        attributes directly without an unwrap step.

        ``self._lm_head_owner`` (the wrapper, not the inner walker)
        is forwarded so :func:`qwen3_5_dflash_forward` can route
        through the real ``lm_head`` on untied-head checkpoints. See
        :meth:`__init__` for the rationale.
        """
        return qwen3_5_dflash_forward(
            self._lm_head_owner,
            inputs,
            cache=cache,
            capture_layer_ids=capture_layer_ids,
        )

    def rollback_speculative_cache(
        self,
        caches: list[Any],
        gdn_states: list[GdnState],
        accepted: int | mx.array,
        block_size: int,
    ) -> int:
        """Trim target KV caches and rewind SSM state after partial-acceptance.

        Unlike Gemma 4 MTP (where rollback only trims KV), the Qwen 3.5
        DFlash rollback ALSO replays SSM (gated-delta) state because
        the recurrence pollutes those caches with rejected drafts.
        Delegated to :func:`qwen3_5_rollback_speculative_cache`.

        Passes ``self._lm_head_owner`` (the wrapper) for API parity
        with :meth:`__call__` -- the rollback function ignores its
        ``target`` argument but accepts whatever the forward passes.
        """
        return qwen3_5_rollback_speculative_cache(
            self._lm_head_owner,
            caches=caches,
            gdn_states=gdn_states,
            accepted=accepted,
            block_size=block_size,
        )


def run_coupled_round_loop(
    *,
    adapter: Gemma4MTPTargetAdapter | Qwen3_5DFlashTargetAdapter,
    drafter: nn.Module,
    prompt_cache: list[Any],
    prefill_output: Gemma4MTPForwardOutput | Qwen3DFlashForwardOutput,
    first_bonus: int,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: int | None,
    token_dtype: mx.Dtype = mx.int32,
) -> Generator[int, None, None]:
    """Drive mlx-vlm's MTP / DFlash round loop and yield decoded token ids.

    The caller (``CoupledModelDrafter.stream``) is responsible for:

    - building the right ``adapter`` (Gemma 4 MTP or Qwen 3.5 DFlash);
    - prefilling ``prompt_cache`` via the adapter's ``__call__`` to
      obtain ``prefill_output`` carrying the hidden capture the round
      loop needs as round-1 input;
    - sampling the first bonus token from ``prefill_output.logits[:, -1:, :]``
      and emitting it as the first decode token (this function does
      NOT yield the first bonus -- it picks up from round 1);
    - threading the yielded tokens through the existing emission path
      (cancellation checks, stop-token detection, ``GenerationResponse``
      construction, usage accounting).

    Why split the loop driver from the surrounding I/O contract: the
    round loop's correctness (accept-walk, rollback, KV / SSM
    sequencing) is independent of how exo emits tokens. Keeping the
    driver narrow means tests can mock target + drafter and exercise
    the loop without instantiating the full ``GenerationResponse``
    pipeline.

    Why the adapter, not the bare target: mlx-vlm's round loops walk
    ``adapter.rollback_speculative_cache`` and forward the adapter
    through to its ``__call__`` for verifies. The adapter holds the
    architecture-specific contract; this function is just the
    architecture dispatch.

    Implementation note: we delegate the actual round logic to
    mlx-vlm's :func:`mlx_vlm.generate._mtp_rounds` /
    :func:`mlx_vlm.generate._dflash_rounds` rather than re-implement
    them. mlx-vlm owns the canonical accept-walk + rollback semantics,
    and re-implementing them would create a silent-divergence risk
    every time mlx-vlm tightens the loop.
    """
    if not prefill_output.hidden_states:
        # Should be unreachable: callers MUST request hidden capture
        # in the prefill forward (otherwise the drafter has nothing
        # to consume on round 1). Surface as a clear error rather
        # than letting the round loop index into an empty list.
        raise RuntimeError(
            "run_coupled_round_loop requires the prefill_output to "
            "carry a captured hidden state. Configure the adapter's "
            "prefill call to request hidden capture before entering "
            "the round loop."
        )

    if isinstance(adapter, Gemma4MTPTargetAdapter):
        # MTP-flavoured prefill: ``Gemma4MTPForwardOutput`` exposes
        # ``hidden_states`` (list per layer; round loop wants the
        # last) and ``shared_kv_states`` (per-layer-type shared KV
        # the assistant drafter consumes every round).
        if not isinstance(prefill_output, Gemma4MTPForwardOutput):
            raise TypeError(
                "Gemma4MTPTargetAdapter requires a Gemma4MTPForwardOutput "
                f"prefill; got {type(prefill_output).__name__!r}."
            )
        last_hidden = prefill_output.hidden_states[-1]
        shared_kv = prefill_output.shared_kv_states
        mtp_rounds = _resolve_mtp_rounds_fn()
        for token, _unused in mtp_rounds(
            adapter,
            drafter,
            prompt_cache,
            last_hidden,
            shared_kv,
            first_bonus=first_bonus,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_block_size=draft_block_size,
            token_dtype=token_dtype,
        ):
            yield token
        return

    # DFlash-flavoured prefill: ``Qwen3DFlashForwardOutput.hidden_states``
    # is a list captured at ``capture_layer_ids`` (one entry per id).
    # mlx-vlm's :func:`_dflash_rounds` consumes a SINGLE ``hidden``
    # tensor formed by feature-axis concatenation of those captures,
    # mirroring what its own ``generate_step`` does at prefill exit
    # (``hidden = mx.concatenate(out.hidden_states, axis=-1)``).
    if not isinstance(prefill_output, Qwen3DFlashForwardOutput):
        raise TypeError(
            "Qwen3_5DFlashTargetAdapter requires a Qwen3DFlashForwardOutput "
            f"prefill; got {type(prefill_output).__name__!r}."
        )
    hidden = mx.concatenate(prefill_output.hidden_states, axis=-1)
    dflash_rounds = _resolve_dflash_rounds_fn()
    for token, _unused in dflash_rounds(
        adapter,
        drafter,
        prompt_cache,
        hidden,
        first_bonus=first_bonus,
        max_tokens=max_tokens,
        sampler=sampler,
        draft_block_size=draft_block_size,
        token_dtype=token_dtype,
    ):
        yield token


def _make_processor_aware_sampler(
    *,
    sampler: Callable[[mx.array], mx.array],
    logits_processors: Sequence[Callable[[mx.array, mx.array], mx.array]],
    running_tokens: list[int],
) -> Callable[[mx.array], mx.array]:
    """Wrap ``sampler`` to apply ``logits_processors`` on every call.

    Codex P1 (PR #25 round-(N+3), coupled_drafter.py:566): mlx-vlm's
    ``_mtp_rounds`` accepts a single-argument ``sampler(logits) ->
    token`` callable and applies it both during drafter proposal
    generation and target verification. Per-request
    ``logits_processors`` (repetition / presence / frequency penalties
    plus the bench EOS-ban processor) take ``(prev_tokens, logits)``,
    so without an adapter they don't reach the round loop and coupled
    requests diverge from non-coupled decoding semantics from token 2
    onwards.

    The wrapper closes over a mutable ``running_tokens`` buffer that
    the caller updates after each emitted token. Each invocation
    snapshots the buffer into an ``mx.array``, runs every processor
    against it, then samples. Because the buffer reflects only
    COMMITTED emissions (not speculative drafts the verifier may
    reject), repetition / presence penalties react to actual output
    history -- which is the correct semantic. Drafts that haven't
    been accepted yet don't pollute the penalty's view.

    Empty processor list is a fast path: we return ``sampler``
    unchanged so the no-processor case pays no overhead.
    """
    if not logits_processors:
        return sampler

    processors = list(logits_processors)

    def _wrapped(logits: mx.array) -> mx.array:
        prev_tokens_array = mx.array(running_tokens, dtype=mx.uint32)
        adjusted = logits
        for proc in processors:
            adjusted = proc(prev_tokens_array, adjusted)
        return sampler(adjusted)

    return _wrapped


def _coerce_int_list(values: list[Any]) -> list[int]:
    """Narrow a ``list[Any]`` to ``list[int]`` via per-element ``int(...)``.

    The DFlash drafter exposes ``config.target_layer_ids`` as an
    untyped attribute on an ``nn.Module``, so :func:`getattr` round-
    trips to ``object`` and the contained list to ``list[Any]``.
    Containing the ``Any`` propagation in this single helper keeps
    the comprehension out of the call site (where the formatter
    splits it across lines and breaks the ``# pyright: ignore``
    placement).
    """
    return [int(item) for item in values]  # pyright: ignore[reportAny]


def _select_first_bonus(
    *,
    last_logits: mx.array,
    prev_tokens: mx.array,
    sampler: Callable[[mx.array], mx.array],
    logits_processors: Sequence[Callable[[mx.array, mx.array], mx.array]],
) -> tuple[int, mx.array]:
    """Sample the first bonus token after the captured prefill.

    Mirrors what :func:`mlx_lm.generate.stream_generate` does at the
    decode-loop entry: apply the request's logits processors against
    the running prev-token sequence, normalise to logprobs, and sample.

    The result feeds :func:`run_coupled_round_loop` as ``first_bonus``.
    Round-loop tokens skip logits-processor application -- mlx-vlm's
    ``_mtp_rounds`` runs ``sampler(verify_out.logits)`` directly with
    no processor hook -- which is fine for the temperature-0 / argmax
    case (the dominant production sampler) and matches what mlx-vlm's
    own MTP path does. Stochastic + repetition-penalty parity with the
    non-coupled path is a Phase 2d concern.

    Returns ``(token_id, logprobs)`` where ``logprobs`` is the
    vocab-sized log-probability array for the sampled position; the
    caller forwards it to ``mlx_lm.GenerationResponse`` so OpenAI-style
    ``logprobs`` requests still see real numbers on the first emitted
    token.

    Shape contract: ``mlx_lm.sample_utils`` logits processors (used by
    ``repetition_penalty``, ``presence_penalty``, ``frequency_penalty``,
    and ``logit_bias``) all index logits as ``[:, tokens]``, i.e. they
    require a 2D ``(batch=1, vocab)`` array; squeezing to 1D before
    running them raises ``ValueError: Too many indices for array with
    1 dimensions``. We therefore normalise the prefill logits to
    ``(1, vocab)`` for the processor pipeline (mirroring what
    ``mlx_lm.generate.generate_step`` does), then squeeze back to
    ``(vocab,)`` at return so ``GenerationResponse.logprobs`` keeps
    the vocab-vector shape the rest of the coupled-drafter loop
    assumes (see ``zero_logprobs`` reset in ``stream``).
    """
    raw = last_logits
    while raw.ndim > 2:
        raw = raw.squeeze(0)
    if raw.ndim == 1:
        raw = raw[None, :]
    for proc in logits_processors:
        raw = proc(prev_tokens, raw)
    logprobs = raw - mx.logsumexp(raw, axis=-1, keepdims=True)
    sampled = sampler(logprobs)
    mx.eval(sampled)
    return int(sampled.item()), logprobs.squeeze(0)


@final
class CoupledModelDrafter:
    """Drafter-protocol shim around :func:`run_coupled_round_loop`.

    Single-node coupled drafters (mtp/dflash) cannot ride the standard
    :class:`ModelDrafter` path because mlx-lm's ``stream_generate``
    speculative loop assumes an *external* drafter that maintains its
    own KV cache and consumes only token ids. Coupled drafters
    architecturally share state with the target (the assistant drafter
    walks the target's last-layer hidden + per-layer-type shared KV
    every round), which the upstream loop has no hook for.

    The shim's ``stream()`` therefore takes ownership of the inner
    decode loop:

    1. Run :func:`gemma4_mtp_forward` on the prefill-tail (the last two
       prompt tokens, identical to what :class:`ModelDrafter` receives)
       with hidden + shared-kv capture. This advances ``prompt_cache``
       to the post-prompt offset and yields the captures
       ``_mtp_rounds`` needs as round-1 input.
    2. Apply the request's logits processors to the last-position
       logits, sample the first bonus, and yield it as a
       ``GenerationResponse`` (so the caller sees a uniform stream
       shape across drafter modes).
    3. Drive :func:`run_coupled_round_loop` -- which delegates to
       :func:`mlx_vlm.generate._mtp_rounds` -- and yield each emitted
       token wrapped in ``GenerationResponse``.

    The drafter's ``mode`` is reported as ``"model"`` so existing
    telemetry (acceptance fraction, drafter-id stamping) flows
    unchanged. The architecture (``"mtp"`` / ``"dflash"``) is surfaced
    via the separate ``drafter_kind`` field on
    :class:`exo.api.types.api.GenerationStats`.

    Round-loop tokens carry zero-valued logprobs because the upstream
    round loop yields ``(token, None)`` and rerunning a forward to
    recover logprobs would defeat the speedup; clients that need
    logprobs on every position should opt out of coupled drafting via
    ``draft_mode="none"`` or ``use_drafter=False``. The first bonus
    carries real logprobs since we computed them ourselves.
    """

    def __init__(
        self,
        *,
        target_adapter: Gemma4MTPTargetAdapter | Qwen3_5DFlashTargetAdapter,
        drafter: nn.Module,
        kind: CoupledDrafterKind,
        num_draft_tokens: int,
        draft_block_size: int | None = None,
    ) -> None:
        if num_draft_tokens < 1:
            raise ValueError(f"num_draft_tokens must be >= 1, got {num_draft_tokens}")
        # Cross-validate kind vs adapter type so a misrouted dispatch
        # surfaces a clear error here instead of an opaque ``AttributeError``
        # deep inside the round-loop driver.
        if kind == "mtp" and not isinstance(target_adapter, Gemma4MTPTargetAdapter):
            raise TypeError(
                f"CoupledModelDrafter(kind='mtp') requires a "
                f"Gemma4MTPTargetAdapter; got {type(target_adapter).__name__!r}."
            )
        if kind == "dflash" and not isinstance(
            target_adapter, Qwen3_5DFlashTargetAdapter
        ):
            raise TypeError(
                f"CoupledModelDrafter(kind='dflash') requires a "
                f"Qwen3_5DFlashTargetAdapter; got {type(target_adapter).__name__!r}."
            )
        self._target_adapter: Gemma4MTPTargetAdapter | Qwen3_5DFlashTargetAdapter = (
            target_adapter
        )
        self._drafter: nn.Module = drafter
        self._kind: CoupledDrafterKind = kind
        self._num_draft_tokens: int = num_draft_tokens
        self._draft_block_size: int | None = draft_block_size

    @property
    def mode(self) -> DraftMode:
        # Coupled drafters present as ``"model"`` mode for telemetry
        # so the existing drafter-id stamping in :mod:`mlx_generate`
        # flows unchanged. ``GenerationStats.drafter_kind`` carries the
        # architecture (``"mtp"`` / ``"dflash"``) so dashboards can
        # disambiguate without re-shaping the ``DraftMode`` literal.
        return "model"

    @property
    def kind(self) -> CoupledDrafterKind:
        """Coupled-drafter architecture this instance dispatches to."""
        return self._kind

    @property
    def num_draft_tokens(self) -> int:
        """K -- per-round draft budget. Mirrors :class:`ModelDrafter`."""
        return self._num_draft_tokens

    def metrics(self) -> dict[str, int]:
        """Per-stream counters surfaced on :class:`GenerationStats`.

        mlx-vlm's ``_mtp_rounds`` appends one entry to
        ``drafter.accept_lens`` per round, so ``len(accept_lens)`` is
        the round count and ``accept_lens[i]`` is the number of
        proposed drafts the verifier accepted in round ``i``. Each
        round emits ``accept_lens[i] + 1`` tokens total: the accepted
        drafts plus one verifier bonus. Each round proposes
        ``block_size - 1`` drafts (mlx-vlm's round loop reduces the
        block when it would overrun ``max_tokens``, but we don't have
        per-round sizing so we use the configured block as an upper
        bound on proposals).

        Codex P2 (PR #25 round-(N+2), coupled_drafter.py:569):
        ``accepted_draft_tokens`` MUST be surfaced authoritatively
        here rather than inferred from ``GenerationResponse.from_draft``
        downstream. The round loop emits both accepted drafts AND the
        verifier bonus per round; without per-token provenance we
        cannot tell which a given emission is, so the
        ``GenerationResponse.from_draft`` flag is set to ``False`` on
        every coupled emission and ``GenerationStats.accepted_draft_tokens``
        is sourced from this metric instead. Pre-fix the code marked
        every round-loop emit as ``from_draft=True``, which produced
        ``accepted_draft_tokens > proposed_draft_tokens`` on
        high-acceptance runs (full-acceptance round of K drafts emits
        K+1 tokens, all flagged accepted, while proposed counts only
        K). The corrected accounting keeps acceptance ratios bounded
        in [0, 1].
        """
        accept_lens_obj: object = getattr(self._drafter, "accept_lens", [])
        accept_lens: list[int] = (
            list(cast("list[int]", accept_lens_obj))
            if isinstance(accept_lens_obj, list)
            else []
        )
        rounds = len(accept_lens)
        block_size = self._resolve_block_size()
        proposed = rounds * max(0, block_size - 1)
        accepted = sum(accept_lens)
        return {
            "spec_decode_rounds": rounds,
            "proposed_draft_tokens": proposed,
            "accepted_draft_tokens": accepted,
        }

    def _prefill(
        self,
        prompt_batch: mx.array,
        prompt_cache: list[Any],
    ) -> Gemma4MTPForwardOutput | Qwen3DFlashForwardOutput:
        """Run the architecture-specific prefill capture.

        MTP and DFlash differ in what the round loop reads off the
        prefill output:

        - MTP (Gemma 4) reads ``prefill_output.hidden_states[-1]`` and
          ``prefill_output.shared_kv_states``; both are populated by
          calling ``adapter(..., return_hidden=True, return_shared_kv=True)``.
        - DFlash (Qwen 3.5) reads ``prefill_output.hidden_states`` (the
          full per-layer capture, concatenated feature-axis-wise) and
          relies on every gated-delta layer pushing its
          :class:`GdnState` to ``prefill_output.gdn_states`` so the
          first ``rollback_speculative_cache`` after partial-acceptance
          can rewind the SSM state. Both sinks are populated when the
          prefill call passes ``capture_layer_ids=draft_model.config.target_layer_ids``.

        The drafter's ``target_layer_ids`` is the canonical source for
        the DFlash capture set; mlx-vlm's :func:`_dflash_rounds` reads
        the same field on every verify forward, so mirroring it here
        keeps prefill and round-loop captures aligned.
        """
        if isinstance(self._target_adapter, Gemma4MTPTargetAdapter):
            return self._target_adapter(
                prompt_batch,
                cache=prompt_cache,
                return_hidden=True,
                return_shared_kv=True,
            )
        # mlx-vlm's DFlash drafter is an untyped ``nn.Module`` subclass
        # whose ``config.target_layer_ids`` is the canonical source for
        # the round-loop's per-verify capture set. ``getattr``
        # round-trips through ``object`` so we narrow with isinstance
        # before consuming. The two ``pyright: ignore`` comments
        # contain the ``Any`` propagation from the upstream untyped
        # ``config`` slot to a single line each.
        config_obj: object = getattr(self._drafter, "config", None)
        target_layer_ids_obj: object = getattr(config_obj, "target_layer_ids", None)
        if not isinstance(target_layer_ids_obj, list):
            raise RuntimeError(
                "DFlash drafter is missing config.target_layer_ids; "
                "the round-loop driver requires this list to size the "
                "hidden-state capture and the prefill helper mirrors "
                "that contract."
            )
        # ``list[Any]`` cast → ``int(...)`` per-element. The per-element
        # ``Any`` is unavoidable here (the upstream drafter config is
        # an untyped ``nn.Module`` slot) so we contain it to one line
        # via a helper rather than a multi-line comprehension that the
        # formatter would split into a per-element pyright ignore.
        target_layer_ids_any: list[Any] = cast("list[Any]", target_layer_ids_obj)
        target_layer_ids: list[int] = _coerce_int_list(target_layer_ids_any)
        return self._target_adapter(
            prompt_batch,
            cache=prompt_cache,
            capture_layer_ids=target_layer_ids,
        )

    def _resolve_block_size(self) -> int:
        """Block size used by :func:`_mtp_rounds` for proposal sizing.

        Honours an explicit ``draft_block_size`` override (Phase 2c
        leaves this at ``None``; a future tuning knob would surface it
        through env var or task params), otherwise falls back to the
        drafter's ``config.block_size`` -- which is the upstream
        default and what ``_mtp_rounds`` itself reads when its
        ``draft_block_size`` argument is ``None``.
        """
        if self._draft_block_size is not None:
            return self._draft_block_size
        config: object = getattr(self._drafter, "config", None)
        block: object = getattr(config, "block_size", None)
        if isinstance(block, int) and block > 0:
            return block
        # Defensive fallback: a misconfigured drafter without a block_size
        # would otherwise produce an opaque ``TypeError`` deep inside
        # ``_mtp_rounds``. ``num_draft_tokens + 1`` mirrors the standard
        # drafter's per-round budget and keeps the loop functional.
        return self._num_draft_tokens + 1

    def stream(
        self,
        *,
        model: Model,
        tokenizer: TokenizerWrapper,
        prompt: mx.array,
        context_tokens: Sequence[int],
        prompt_cache: KVCacheType,
        max_tokens: int,
        sampler: Callable[[mx.array], mx.array],
        logits_processors: Sequence[Callable[[mx.array, mx.array], mx.array]],
        prefill_step_size: int = 1,
    ) -> Generator[GenerationResponse, None, None]:
        """Drive the coupled round loop and yield mlx_lm-shaped responses.

        ``prompt`` is the prefill-tail (typically size 2 in production:
        ``mlx_generate`` aligns ``prompt_cache`` to ``full_prompt[:-2]``
        and hands us ``decode_prompt = full_prompt[-2:]``). We process
        the entire tail through :func:`gemma4_mtp_forward` -- one
        captured forward advances the cache to the post-prompt offset
        and gives us the round-1 inputs in a single pass.

        ``model`` and ``prefill_step_size`` are accepted for protocol
        parity but unused: the adapter holds the actual target reference
        and the round loop is autoregressive (single-token verifies),
        so the prefill chunking knob doesn't apply.
        """
        del model, prefill_step_size
        prompt_batch = prompt[None] if prompt.ndim == 1 else prompt
        prompt_tail_size = int(prompt.size)

        # Codex P2 (PR #25 round-(N+0), coupled_drafter.py:484): pre-fix
        # the prompt-TPS timer was started AFTER prefill had already
        # completed, so ``prompt_time`` was effectively zero and
        # ``prompt_tps`` was massively inflated for coupled-drafter
        # requests. ``GenerationStats`` then surfaced bogus telemetry,
        # especially when the upstream ``prefill_tps`` source was
        # unavailable and fell back to ``out.prompt_tps``. We now bracket
        # the prefill call (``mx.eval`` materializes the lazy compute so
        # the wall-clock window covers actual GPU work, mirroring the
        # standard ``ModelDrafter`` flow which pays the prefill cost
        # before its first iteration emit).
        prefill_tic = time.perf_counter()
        prefill_output: Gemma4MTPForwardOutput | Qwen3DFlashForwardOutput = (
            self._prefill(prompt_batch, list(prompt_cache))
        )
        mx.eval(prefill_output.logits)
        prompt_time = max(time.perf_counter() - prefill_tic, 0.0)
        prompt_tps = prompt_tail_size / prompt_time if prompt_time > 0 else 0.0

        # Codex P1 (PR #25 round-(N+3), coupled_drafter.py:566): the
        # round loop must keep request-level logits processors
        # (repetition / presence / frequency penalties, custom token
        # bans, etc.) active for every emitted token, not just the
        # first bonus. Pre-fix only ``_select_first_bonus`` ran the
        # processors and ``run_coupled_round_loop`` received the bare
        # ``sampler``, so requests that set logits_processors got
        # different output distributions from token 2 onwards
        # depending on whether the request landed on the coupled or
        # standard path. ``running_tokens`` is a mutable closure-state
        # buffer that grows as the round loop yields; the wrapped
        # sampler reads it before each ``sampler(logits)`` call inside
        # ``_mtp_rounds`` (drafter proposal AND target verify), so
        # every call sees the latest emitted-token history. The
        # snapshot is intentionally stale w.r.t. drafts that haven't
        # been accepted yet -- you want repetition penalty to react to
        # COMMITTED tokens, not speculative drafts that might be
        # rejected.
        running_tokens: list[int] = (
            list(context_tokens)
            if context_tokens
            else [int(t) for t in cast(list[int], prompt.tolist())]
        )
        processors_list: list[Callable[[mx.array, mx.array], mx.array]] = list(
            logits_processors
        )

        first_bonus, first_logprobs = _select_first_bonus(
            last_logits=prefill_output.logits[:, -1:, :],
            prev_tokens=mx.array(running_tokens, dtype=mx.uint32),
            sampler=sampler,
            logits_processors=processors_list,
        )

        running_tokens.append(first_bonus)
        wrapped_sampler = _make_processor_aware_sampler(
            sampler=sampler,
            logits_processors=processors_list,
            running_tokens=running_tokens,
        )

        detokenizer = tokenizer.detokenizer
        detokenizer.reset()  # type: ignore[reportUnknownMemberType]
        eos_ids = _eos_ids_from_tokenizer(tokenizer)

        # Mark the start of generation timing.
        tic = time.perf_counter()

        emitted = 0
        last_token = first_bonus
        last_logprobs = first_logprobs
        finish_reason: str | None = None
        zero_logprobs = mx.zeros(
            (int(prefill_output.logits.shape[-1]),),
            dtype=mx.float32,
        )

        # Yield the first bonus -- caller treats this identically to a
        # standard drafter's first emitted token. ``finish_reason`` is
        # ``None`` here even when the bonus IS an EOS, so the early-stop
        # check below runs once before we emit the closing chunk.
        emitted += 1
        is_eos = first_bonus in eos_ids
        if is_eos:
            finish_reason = "stop"
        elif emitted >= max_tokens:
            finish_reason = "length"
        detokenizer.add_token(first_bonus)  # type: ignore[reportUnknownMemberType]
        elapsed = time.perf_counter() - tic
        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=first_bonus,
            logprobs=first_logprobs,
            from_draft=False,
            prompt_tokens=prompt_tail_size,
            prompt_tps=prompt_tps,
            generation_tokens=emitted,
            generation_tps=emitted / elapsed if elapsed > 0 else 0.0,
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason=None,
        )

        if finish_reason is None:
            for token in run_coupled_round_loop(
                adapter=self._target_adapter,
                drafter=self._drafter,
                prompt_cache=cast("list[Any]", list(prompt_cache)),
                prefill_output=prefill_output,
                first_bonus=first_bonus,
                max_tokens=max_tokens,
                sampler=wrapped_sampler,
                draft_block_size=self._draft_block_size,
            ):
                running_tokens.append(token)
                emitted += 1
                last_token = token
                last_logprobs = zero_logprobs
                if token in eos_ids:
                    finish_reason = "stop"
                    break
                detokenizer.add_token(token)  # type: ignore[reportUnknownMemberType]
                if emitted >= max_tokens:
                    finish_reason = "length"
                    break
                elapsed = time.perf_counter() - tic
                yield GenerationResponse(
                    text=detokenizer.last_segment,
                    token=token,
                    logprobs=zero_logprobs,
                    # Codex P2 (PR #25 round-(N+2), coupled_drafter.py:569):
                    # each ``_mtp_rounds`` round emits both the accepted
                    # draft tokens AND one verifier bonus, but we only
                    # see a flat token stream out of the round loop
                    # without per-token provenance. Pre-fix every
                    # round-loop emission was flagged ``from_draft=True``,
                    # which let ``from_draft_count`` exceed
                    # ``proposed_draft_tokens`` on high-acceptance runs
                    # (full-acceptance round of K drafts produces K+1
                    # emits, all marked accepted, while proposed counts
                    # only K) and corrupted acceptance-rate dashboards.
                    # We now set ``from_draft=False`` for every coupled
                    # emission and surface the authoritative acceptance
                    # count via :meth:`metrics`'s
                    # ``accepted_draft_tokens`` (sum of
                    # ``drafter.accept_lens``). ``mlx_generate`` prefers
                    # the metric over the per-emit flag.
                    from_draft=False,
                    prompt_tokens=prompt_tail_size,
                    prompt_tps=prompt_tps,
                    generation_tokens=emitted,
                    generation_tps=emitted / elapsed if elapsed > 0 else 0.0,
                    peak_memory=mx.get_peak_memory() / 1e9,
                    finish_reason=None,
                )

        detokenizer.finalize()  # type: ignore[reportUnknownMemberType]
        elapsed = time.perf_counter() - tic
        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=last_token,
            logprobs=last_logprobs,
            # Codex P2 (PR #25 round-(N+2), coupled_drafter.py:569):
            # the closing chunk also avoids claiming draft attribution.
            # Pre-fix this used ``emitted > 1`` as a heuristic ("any
            # round-loop activity counted as drafted"), which double-
            # counted alongside per-emit flags. ``GenerationStats``
            # now sources the acceptance count from :meth:`metrics`
            # exclusively.
            from_draft=False,
            prompt_tokens=prompt_tail_size,
            prompt_tps=prompt_tps,
            generation_tokens=emitted,
            generation_tps=emitted / elapsed if elapsed > 0 else 0.0,
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason=finish_reason
            or ("stop" if last_token in eos_ids else "length"),
        )


def _eos_ids_from_tokenizer(tokenizer: TokenizerWrapper) -> list[int]:
    """Tokenizer-agnostic EOS lookup mirroring :mod:`drafter`'s helper.

    Duplicated here (instead of imported) to keep ``coupled_drafter`` free of
    a back-import on ``drafter``; the function body is two lines and the
    duplication is cheaper than the cycle.
    """
    eos_obj: object = getattr(tokenizer, "eos_token_ids", None)
    if eos_obj is None:
        return []
    if isinstance(eos_obj, list):
        # Cache as ``list[Any]`` then coerce each element through ``int(...)``
        # individually; the runtime type is ``list[int]`` but the upstream
        # tokenizer surface is untyped, so per-element narrowing keeps
        # basedpyright's ``reportAny`` quiet without a wholesale
        # ``# pyright: ignore``.
        items = cast("list[Any]", eos_obj)
        return [int(item) for item in items]  # pyright: ignore[reportAny]
    return []


__all__ = [
    "DISPATCHABLE_COUPLED_DRAFTER_KINDS",
    "CoupledDrafterKind",
    "CoupledModelDrafter",
    "Gemma4MTPTargetAdapter",
    "Qwen3_5DFlashTargetAdapter",
    "is_coupled_drafter_dispatchable",
    "run_coupled_round_loop",
]
