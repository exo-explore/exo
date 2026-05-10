"""Adapter + dispatch tests for the DFlash coupled-drafter path.

These tests pin the contract of
:class:`~exo.worker.engines.mlx.generator.coupled_drafter.Qwen3_5DFlashTargetAdapter`
and the kind-aware branches of
:func:`~exo.worker.engines.mlx.generator.coupled_drafter.run_coupled_round_loop`.
The numerical correctness of the underlying vendor hooks (forward
parity, gdn-state capture, KV trim + SSM rewind) is covered by
:file:`test_qwen3_5_dflash_hooks.py`; here we validate that the
adapter wraps those hooks correctly and that the round-loop driver
routes the right way based on the adapter type.

Why a separate file from :file:`test_coupled_drafter_round_loop.py`:
the synthetic-target setup for Qwen 3.5 (gated-delta caches, attention
caches, mixed layer types) is materially different from the Gemma 4
setup, and pytest collection time stays predictable when each
synthetic-target file owns its own fixture set.

Why we don't drive the real DFlash drafter end-to-end here: the
upstream :func:`mlx_vlm.generate._dflash_rounds` reads the drafter's
``config.target_layer_ids`` (which sizes the prefill capture) and
expects a real DFlash drafter ``nn.Module`` with ``draft_block`` /
``reset`` / ``accept_lens``. That drafter's weight init alone takes
seconds and the round loop's correctness against tiny synthetic
weights is already covered indirectly by the Gemma 4 round-loop
tests (which exercise the SAME ``_*_rounds`` driver shape). The
DFlash-specific surface that needs explicit coverage here is the
adapter's ``__call__`` shape and the round-loop driver's kind
branching.
"""

from __future__ import annotations

from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.gemma4_text import Model as Gemma4Model
from mlx_lm.models.gemma4_text import ModelArgs as Gemma4ModelArgs
from mlx_lm.models.qwen3_5 import (
    TextModel as Qwen3_5LanguageModel,
)
from mlx_lm.models.qwen3_5 import (
    TextModelArgs,
)

from exo.worker.engines.mlx.generator.coupled_drafter import (
    DISPATCHABLE_COUPLED_DRAFTER_KINDS,
    CoupledModelDrafter,
    Gemma4MTPTargetAdapter,
    Qwen3_5DFlashTargetAdapter,
    is_coupled_drafter_dispatchable,
    run_coupled_round_loop,
)
from exo.worker.engines.mlx.vendor.gemma4_mtp_hooks import (
    attach_mtp_hooks,
    gemma4_mtp_forward,
)
from exo.worker.engines.mlx.vendor.qwen3_5_dflash_hooks import (
    attach_dflash_hooks,
    qwen3_5_dflash_forward,
)


def _build_tiny_qwen3_5_with_hooks() -> Qwen3_5LanguageModel:
    """Mirror of :func:`test_qwen3_5_dflash_hooks._build_tiny_qwen3_5`.

    Same minimum-viable head dim/count combination required to keep
    the gated-delta Metal kernel inside a valid specialisation.
    """
    args = TextModelArgs(
        model_type="qwen3_5_text",
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=4,
        intermediate_size=256,
        vocab_size=128,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        head_dim=32,
        full_attention_interval=2,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_value_head_dim=64,
        num_experts=0,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        attention_bias=False,
        num_experts_per_tok=0,
        decoder_sparse_step=1,
        shared_expert_intermediate_size=0,
        moe_intermediate_size=0,
        norm_topk_prob=True,
        partial_rotary_factor=0.25,
        rope_scaling=None,
        rope_parameters={},
    )
    model = Qwen3_5LanguageModel(args)
    model.eval()
    attach_dflash_hooks(model)
    return model


def _build_tiny_gemma4_with_hooks() -> Gemma4Model:
    """Tiny Gemma 4 used for the cross-kind dispatch guard tests."""
    args = Gemma4ModelArgs(
        model_type="gemma4_text",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=2,
        head_dim=32,
        global_head_dim=32,
        num_key_value_heads=1,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        vocab_size=100,
        vocab_size_per_layer_input=100,
        sliding_window=32,
        sliding_window_pattern=2,
        max_position_embeddings=256,
        layer_types=["sliding_attention", "full_attention"],
        tie_word_embeddings=True,
        final_logit_softcapping=30.0,
    )
    model = Gemma4Model(args)
    model.eval()
    attach_mtp_hooks(model)
    return model


def test_dispatch_includes_dflash() -> None:
    """The frozenset must list both supported kinds.

    Builder-side gates consult this set to decide whether a coupled
    drafter is "usable for this request"; if the set drifted out of
    sync with the dispatch wiring in ``mlx_generate``, a dflash-only
    setup would either lose batch throughput (forced into
    :class:`SequentialGenerator` while the dispatch silently ran
    plain decoding) or burn the dispatch path on a kind it can't
    drive. Pin both possibilities here.
    """
    assert "mtp" in DISPATCHABLE_COUPLED_DRAFTER_KINDS
    assert "dflash" in DISPATCHABLE_COUPLED_DRAFTER_KINDS
    assert is_coupled_drafter_dispatchable("mtp")
    assert is_coupled_drafter_dispatchable("dflash")


def test_dflash_adapter_requires_attached_hooks() -> None:
    """Constructing the adapter without ``attach_dflash_hooks`` must fail.

    The adapter is the only entry point through which the dispatch
    can reach :func:`_dflash_rounds`; if it accepted unhooked
    targets, a card declaring ``coupled_drafter.kind='dflash'``
    against a non-Qwen-3.5 model would only surface the mismatch on
    the first verify forward (a much more confusing failure than a
    guard-rail at construction).
    """
    args = TextModelArgs(
        model_type="qwen3_5_text",
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=4,
        intermediate_size=256,
        vocab_size=128,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        head_dim=32,
        full_attention_interval=2,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_value_head_dim=64,
        num_experts=0,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        attention_bias=False,
        num_experts_per_tok=0,
        decoder_sparse_step=1,
        shared_expert_intermediate_size=0,
        moe_intermediate_size=0,
        norm_topk_prob=True,
        partial_rotary_factor=0.25,
        rope_scaling=None,
        rope_parameters={},
    )
    target_without_hooks = Qwen3_5LanguageModel(args)
    target_without_hooks.eval()

    with pytest.raises(RuntimeError, match="attach_dflash_hooks"):
        Qwen3_5DFlashTargetAdapter(target_without_hooks)


def test_dflash_adapter_rejects_non_qwen_target() -> None:
    """A non-Qwen 3.5 target must surface a focused ``TypeError``.

    Mirrors :class:`Gemma4MTPTargetAdapter`'s symmetric rejection
    test: the loader's hook-attach gate is the upstream defence, but
    the adapter holds the dispatch's sole reference to the target
    type, so the construction-time check is what stops a misrouted
    card from quietly running with the wrong vendor hooks.
    """
    gemma_target = _build_tiny_gemma4_with_hooks()
    with pytest.raises(TypeError, match="Qwen 3.5"):
        Qwen3_5DFlashTargetAdapter(gemma_target)


def test_dflash_adapter_call_returns_dflash_forward_output() -> None:
    """The adapter's ``__call__`` returns the captured-forward triple.

    :func:`_dflash_rounds` reads ``out.logits``, ``out.hidden_states``
    (a per-capture-id list), and ``out.gdn_states`` (a per-linear-
    layer list of ``GdnState`` 11-tuples). All three must be
    populated when ``capture_layer_ids`` is non-empty; the adapter
    sets ``capture_gdn_states`` automatically (mirroring mlx-vlm's
    own ``LanguageModel.__call__``).
    """
    target = _build_tiny_qwen3_5_with_hooks()
    adapter = Qwen3_5DFlashTargetAdapter(target)

    inputs = mx.array([[1, 2, 3]])
    cache = cast("list[Any]", target.make_cache())

    out = adapter(inputs, cache=cache, capture_layer_ids=[0, 2])

    assert out.logits.shape == (1, 3, 128)
    assert len(out.hidden_states) == 2, (
        "capture_layer_ids=[0, 2] should produce 2 hidden snapshots"
    )
    # full_attention_interval=2 → gated-delta layers at indices 0 and 2.
    # Both linear layers fire on every forward, so 2 GdnState tuples
    # are expected when capture is automatically enabled.
    assert len(out.gdn_states) == 2, (
        "automatic gdn capture should populate one entry per linear layer"
    )


def test_dflash_adapter_preserves_lm_head_owner_on_untied_target() -> None:
    """Adapter must thread the wrapper through to ``qwen3_5_dflash_forward``.

    The forward routes through ``lm_head(h)`` vs
    ``embed_tokens.as_linear(h)`` based on the wrapper's
    ``args.tie_word_embeddings`` (via
    :func:`_resolve_lm_head_owner`). The wrapper-resolution step needs
    the *wrapper* in hand -- the inner ``Qwen3_5TextModel`` doesn't
    own ``lm_head`` or ``args``. Pre-fix the adapter stored only the
    inner and silently forced the tied-embeddings path on untied-head
    checkpoints (``tie_word_embeddings=False`` is common for Qwen 3.5
    / 3.6), corrupting verifier logits and therefore accept / reject
    decisions in coupled decoding.

    Asserts ``adapter(inputs) is byte-equivalent to`` the wrapper-routed
    forward and **distinct** from the inner-routed forward whenever
    ``lm_head`` and ``embed_tokens`` carry different weights.
    """
    target = _build_tiny_qwen3_5_with_hooks()
    assert target.args.tie_word_embeddings is False, (
        "fixture must be untied-head to exercise the lm_head path"
    )
    # Force ``lm_head.weight`` to a distinguishable value so the two
    # code paths (``lm_head(h)`` vs ``embed_tokens.as_linear(h)``)
    # produce visibly different logits. Without this the test would
    # pass trivially on init noise convergence.
    target.lm_head.weight = mx.ones_like(target.lm_head.weight)

    adapter = Qwen3_5DFlashTargetAdapter(target)
    cache_adapter = cast("list[Any]", target.make_cache())
    cache_wrapper = cast("list[Any]", target.make_cache())
    cache_inner = cast("list[Any]", target.make_cache())
    prompt = mx.array([[1, 2, 3]])

    # Adapter route (post-fix: routes through wrapper).
    adapter_out = adapter(prompt, cache=cache_adapter, capture_layer_ids=[0])
    # Direct wrapper route -- the post-fix adapter must match this.
    wrapper_out = qwen3_5_dflash_forward(
        target, prompt, cache=cache_wrapper, capture_layer_ids=[0]
    )
    # Direct inner route -- pre-fix adapter degraded to this path.
    inner_out = qwen3_5_dflash_forward(
        target.model, prompt, cache=cache_inner, capture_layer_ids=[0]
    )

    assert mx.allclose(adapter_out.logits, wrapper_out.logits, atol=1e-5).item(), (
        "adapter forward must route through the wrapper-aware path so "
        "untied lm_head logits match the wrapper-routed forward"
    )
    assert not mx.allclose(adapter_out.logits, inner_out.logits, atol=1e-5).item(), (
        "adapter forward must NOT degrade to embed_tokens.as_linear; "
        "if it does, untied-head Qwen targets are scored with the wrong "
        "LM head and accept / reject decisions diverge from upstream"
    )


def test_dflash_adapter_rollback_passes_through() -> None:
    """``rollback_speculative_cache`` returns ``max(accepted)`` per the contract.

    Mirrors :class:`Gemma4MTPTargetAdapter`'s rollback test, but now
    we also run a real verify forward first so the gated-delta
    caches have populated SSM state -- the rewind path is non-trivial
    and zero-state caches would be a degenerate cover.
    """
    target = _build_tiny_qwen3_5_with_hooks()
    adapter = Qwen3_5DFlashTargetAdapter(target)
    cache = cast("list[Any]", target.make_cache())

    # Prime the caches with a forward so they have rewindable state.
    _ = adapter(mx.array([[1, 2, 3, 4]]), cache=cache, capture_layer_ids=[0, 2])

    # Run a verify-shaped forward (block of 3 candidate tokens) to
    # produce ``gdn_states`` we can hand to the rollback. Mirrors what
    # ``_dflash_rounds`` does on every round.
    verify_out = adapter(mx.array([[5, 6, 7]]), cache=cache, capture_layer_ids=[0, 2])

    # Accepting 1 of 2 drafts (block_size=3 → drafted_count=2; we
    # accept index 0 → ``accepted=1``). The rollback must NOT raise
    # and must echo the accepted count.
    accepted_count = adapter.rollback_speculative_cache(
        caches=cache,
        gdn_states=verify_out.gdn_states,
        accepted=1,
        block_size=3,
    )
    assert accepted_count == 1


def test_dflash_adapter_model_property_exposes_text_model() -> None:
    """The drafter's ``reset`` walks ``adapter.model.embed_tokens``.

    For Qwen 3.5 the inner ``Qwen3_5TextModel`` IS the layer walker
    the drafter needs, so ``adapter.model`` resolves to the text
    model itself (not a ``.model`` sub-attribute as in the Gemma 4
    case). Either way the binding goes to the SAME embed_tokens
    parameters the wrapper owns -- no weight duplication.
    """
    target = _build_tiny_qwen3_5_with_hooks()
    adapter = Qwen3_5DFlashTargetAdapter(target)

    assert hasattr(adapter, "model")
    assert hasattr(adapter.model, "embed_tokens")
    # ``target.model`` is the underlying ``Qwen3_5TextModel``; the
    # adapter exposes the same instance.
    assert adapter.model is target.model


def test_dflash_round_loop_rejects_missing_hidden_capture() -> None:
    """The DFlash branch must surface the same clear-error guard as MTP.

    Pre-fix, ``_dflash_rounds`` would index into an empty
    ``hidden_states`` list and raise an opaque ``IndexError`` deep in
    the round loop. The driver's boundary check catches this so the
    operator gets a focused error pointing at the prefill call.
    """
    target = _build_tiny_qwen3_5_with_hooks()
    adapter = Qwen3_5DFlashTargetAdapter(target)
    cache = cast("list[Any]", target.make_cache())

    prompt = mx.array([[1, 2]])
    # Calling the underlying hook with ``capture_layer_ids=None`` is
    # the only way to produce a prefill output with empty
    # ``hidden_states`` -- the adapter itself always passes a
    # non-empty list, which is why this test goes around the adapter
    # to construct the degenerate input.
    prefill = qwen3_5_dflash_forward(
        target, prompt, cache=cache, capture_layer_ids=None, capture_gdn_states=False
    )

    with pytest.raises(RuntimeError, match="captured hidden state"):
        list(
            run_coupled_round_loop(
                adapter=adapter,
                drafter=nn.Module(),  # never reached
                prompt_cache=cache,
                prefill_output=prefill,
                first_bonus=0,
                max_tokens=2,
                sampler=lambda logits: mx.argmax(logits, axis=-1).astype(mx.int32),
                draft_block_size=None,
            )
        )


def test_round_loop_rejects_mtp_prefill_with_dflash_adapter() -> None:
    """Routing a ``Gemma4MTPForwardOutput`` into a DFlash adapter must fail.

    The two adapters expect their own prefill output type (MTP →
    ``Gemma4MTPForwardOutput``; DFlash → ``Qwen3DFlashForwardOutput``).
    A type mismatch here is unreachable from production code paths
    (the adapter's ``__call__`` produces the right type by
    construction) but the type-narrowed ``isinstance`` check inside
    :func:`run_coupled_round_loop` is what makes the dispatch's
    static guarantees survive a future refactor that adds a third
    adapter, so we pin it explicitly.
    """
    gemma_target = _build_tiny_gemma4_with_hooks()
    qwen_target = _build_tiny_qwen3_5_with_hooks()
    gemma_cache = cast("list[Any]", gemma_target.make_cache())
    mtp_prefill = gemma4_mtp_forward(
        gemma_target,
        mx.array([[1, 2]]),
        cache=gemma_cache,
        return_hidden=True,
        return_shared_kv=True,
    )
    dflash_adapter = Qwen3_5DFlashTargetAdapter(qwen_target)
    qwen_cache = cast("list[Any]", qwen_target.make_cache())

    with pytest.raises(TypeError, match="Qwen3DFlashForwardOutput"):
        list(
            run_coupled_round_loop(
                adapter=dflash_adapter,
                drafter=nn.Module(),  # never reached
                prompt_cache=qwen_cache,
                prefill_output=mtp_prefill,
                first_bonus=0,
                max_tokens=2,
                sampler=lambda logits: mx.argmax(logits, axis=-1).astype(mx.int32),
                draft_block_size=None,
            )
        )


def test_coupled_model_drafter_kind_must_match_adapter_type() -> None:
    """The drafter's ``__init__`` cross-validates kind vs adapter type.

    A future refactor that derived ``kind`` from a different source
    than the adapter could route MTP through DFlash branches (or
    vice versa) without a clear failure. The construction-time
    cross-check lights up the divergence at exactly the boundary
    where it can still be caught.
    """
    qwen_target = _build_tiny_qwen3_5_with_hooks()
    dflash_adapter = Qwen3_5DFlashTargetAdapter(qwen_target)

    # Dummy drafter; the cross-validation runs before any drafter
    # access, so a bare ``nn.Module`` is enough to reach the assertion.
    bare_drafter = nn.Module()

    with pytest.raises(TypeError, match="Qwen3_5DFlashTargetAdapter"):
        _ = CoupledModelDrafter(
            target_adapter=dflash_adapter,
            drafter=bare_drafter,
            kind="mtp",  # MISMATCH: dflash adapter + mtp kind
            num_draft_tokens=4,
        )

    gemma_target = _build_tiny_gemma4_with_hooks()
    mtp_adapter = Gemma4MTPTargetAdapter(gemma_target)
    with pytest.raises(TypeError, match="Gemma4MTPTargetAdapter"):
        _ = CoupledModelDrafter(
            target_adapter=mtp_adapter,
            drafter=bare_drafter,
            kind="dflash",  # MISMATCH: mtp adapter + dflash kind
            num_draft_tokens=4,
        )
