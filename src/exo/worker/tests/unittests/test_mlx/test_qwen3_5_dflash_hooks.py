"""Tests for the Qwen 3.5 DFlash target-side hooks.

We build a tiny ``Qwen3_5TextModel`` directly from ``TextModelArgs`` (no
checkpoint download) and exercise the surface vendored from mlx-vlm:

- :func:`attach_dflash_hooks` / :func:`has_dflash_hooks` -- gating used
  by the loader to confirm a target is hook-capable.
- :func:`qwen3_5_dflash_forward` -- runs the layer loop, captures
  per-layer hiddens, and emits one :data:`GdnState` 11-tuple per
  gated-delta layer.
- :func:`qwen3_5_rollback_speculative_cache` -- trims KV caches AND
  rewinds the per-row SSM state via ``gated_delta_update`` after a
  partial-acceptance round.

The "tiny" Qwen 3.5 (4 layers, hidden_size=64, vocab_size=128) is small
enough to exercise both the gated-delta-net and full-attention layer
paths within a CPU-only test budget. ``full_attention_interval=2``
gives the layer-types pattern ``[linear, attn, linear, attn]`` so the
mixed-cache rollback is genuinely covered.

Status note: the dispatch frozenset (``DISPATCHABLE_COUPLED_DRAFTER_KINDS``)
now includes ``"dflash"`` -- these tests prove the vendored hooks
compile, attach, and produce shape-correct outputs against a
synthetic Qwen 3.5 target, so production drift between
:mod:`exo.worker.engines.mlx.vendor.qwen3_5_dflash_hooks`,
:class:`~exo.worker.engines.mlx.generator.coupled_drafter.Qwen3_5DFlashTargetAdapter`,
and the loader's ``attach_dflash_hooks`` gate surfaces immediately
without needing a real hybrid Qwen 3.5 checkpoint on the test box.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import mlx.core as mx
import pytest
from mlx_lm.models.qwen3_5 import (
    Qwen3_5TextModel,
    TextModelArgs,
)
from mlx_lm.models.qwen3_5 import (
    TextModel as Qwen3_5LanguageModel,
)

from exo.worker.engines.mlx.vendor.qwen3_5_dflash_hooks import (
    Qwen3DFlashForwardOutput,
    attach_dflash_hooks,
    has_dflash_hooks,
    qwen3_5_dflash_forward,
    qwen3_5_rollback_speculative_cache,
    resolve_qwen3_5_text_model,
)


def _build_tiny_qwen3_5(*, num_layers: int = 4) -> Qwen3_5LanguageModel:
    """Construct a small Qwen 3.5 language model in-memory.

    ``full_attention_interval=2`` interleaves linear-attention (gated
    delta) and full-attention layers, so the per-layer-type cache
    handling is exercised. ``num_experts=0`` keeps the dense MLP path
    -- the MoE path is not on the DFlash hook surface and would just
    add weight init time without any hook coverage.

    Returns the outer ``TextModel`` (the wrapper that owns ``lm_head``)
    so the captured forward's tied-vs-untied LM-head dispatch is
    exercised end-to-end.
    """
    # Head dims/counts must be large enough to keep the gated-delta
    # Metal kernel's tile-per-thread sizing positive. The 32/64 head
    # dims and 4/4 head counts here are the smallest combination that
    # lands inside a valid kernel specialisation on Apple Silicon (the
    # 16/2/2 combination triggers a zero-length-array compile error).
    #
    # ``TextModelArgs`` has runtime defaults for the remaining fields,
    # but mlx-lm's typed stub doesn't surface them so we list every
    # field explicitly to keep basedpyright happy.
    args = TextModelArgs(
        model_type="qwen3_5_text",
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=num_layers,
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
    return model


def _fresh_cache(model: Qwen3_5LanguageModel) -> list[Any]:
    """Build a one-cache-per-layer list using mlx-lm's defaults.

    ``TextModel.make_cache`` returns the right per-layer cache types
    for Qwen 3.5 (linear layers get ``ArraysCache`` for conv-state +
    SSM-state; full-attention layers get ``KVCache``). The hooked
    forward expects this exact list shape.
    """
    return cast("list[Any]", model.make_cache())


def _kv_offsets(caches: list[Any]) -> list[int]:
    """Snapshot the post-trim ``offset`` field on every KV cache.

    Linear (gated-delta) caches don't expose ``offset`` at all, so we
    skip them. Routing through this helper keeps the test bodies free
    of the per-attribute ``cast(Any, ...)`` ceremony basedpyright
    otherwise demands -- the cache classes' offset attribute is typed
    ``int`` at runtime but ``Any`` in mlx-lm's stub.
    """
    offsets: list[int] = []
    for index in range(len(caches)):
        cache_: Any = caches[index]  # pyright: ignore[reportAny]
        if hasattr(cache_, "offset"):  # pyright: ignore[reportAny]
            offsets.append(int(cast(int, cache_.offset)))
    return offsets


def _ssm_caches(caches: list[Any]) -> list[Any]:
    """Filter the cache list to only the gated-delta (non-trimmable) entries.

    The trimmable / non-trimmable split is what
    :func:`qwen3_5_rollback_speculative_cache` itself uses internally;
    the test asserts against the same partition.
    """
    ssm_caches: list[Any] = []
    for index in range(len(caches)):
        cache_: Any = caches[index]  # pyright: ignore[reportAny]
        # Reach through ``getattr`` so the typed surface stays
        # ``Callable[[], bool]`` instead of leaking ``Any`` from the
        # mlx-lm cache stub.
        is_trimmable_method = cast(
            "Callable[[], bool]",
            getattr(cache_, "is_trimmable"),  # noqa: B009 # pyright: ignore[reportAny]
        )
        if not is_trimmable_method():
            ssm_caches.append(cache_)
    return ssm_caches


def test_attach_dflash_hooks_marks_target() -> None:
    model = _build_tiny_qwen3_5()
    assert not has_dflash_hooks(model)

    attach_dflash_hooks(model)

    assert has_dflash_hooks(model)


def test_attach_dflash_hooks_idempotent() -> None:
    model = _build_tiny_qwen3_5()
    attach_dflash_hooks(model)
    attach_dflash_hooks(model)

    assert has_dflash_hooks(model)


def test_attach_dflash_hooks_rejects_non_qwen3_5() -> None:
    """The dispatch gate refuses targets that aren't Qwen 3.5."""

    class NotQwen:
        pass

    target = NotQwen()

    with pytest.raises(TypeError, match="Qwen 3.5 target"):
        attach_dflash_hooks(target)

    assert not has_dflash_hooks(target)


def test_has_dflash_hooks_default_false() -> None:
    model = _build_tiny_qwen3_5()
    assert not has_dflash_hooks(model)
    assert not has_dflash_hooks(object())


def test_attach_dflash_hooks_walks_inner_text_model() -> None:
    """Attach against the wrapper marks the inner ``Qwen3_5TextModel``.

    mlx-lm's ``TextModel`` wraps the layer-walking
    ``Qwen3_5TextModel``; the dispatch site can hold either handle, so
    the sentinel must end up on both. Symmetric with the Gemma 4
    wrapper test.
    """
    wrapper = _build_tiny_qwen3_5()
    inner = wrapper.model
    assert isinstance(inner, Qwen3_5TextModel)
    assert not has_dflash_hooks(wrapper)
    assert not has_dflash_hooks(inner)

    attach_dflash_hooks(wrapper)

    assert has_dflash_hooks(wrapper)
    assert has_dflash_hooks(inner)


def test_resolve_qwen3_5_text_model_unwraps_wrapper() -> None:
    """``resolve_qwen3_5_text_model`` returns the layer walker for both
    the wrapper and the inner model."""
    wrapper = _build_tiny_qwen3_5()
    inner = wrapper.model

    assert resolve_qwen3_5_text_model(wrapper) is inner
    assert resolve_qwen3_5_text_model(inner) is inner
    assert resolve_qwen3_5_text_model(object()) is None


def test_dflash_forward_logits_match_unhooked_call() -> None:
    """The hook must NOT change the logits the target produces.

    Same contract as the Gemma 4 MTP test: the verify forward must be
    numerically equivalent to the standard forward, otherwise drafted-
    token acceptance would be silently miscalibrated.
    """
    model = _build_tiny_qwen3_5()
    inputs = mx.array([[1, 2, 3, 4, 5]])

    cache_unhooked = _fresh_cache(model)
    unhooked_logits = model(inputs, cache=cache_unhooked)

    cache_hooked = _fresh_cache(model)
    out = qwen3_5_dflash_forward(model, inputs, cache=cache_hooked)

    assert isinstance(out, Qwen3DFlashForwardOutput)
    assert mx.allclose(out.logits, unhooked_logits, atol=1e-4).item() is True


def test_dflash_forward_captures_requested_hidden_states() -> None:
    """``capture_layer_ids`` populates one hidden state per requested layer."""
    model = _build_tiny_qwen3_5(num_layers=4)
    inputs = mx.array([[1, 2, 3]])

    out = qwen3_5_dflash_forward(
        model,
        inputs,
        cache=_fresh_cache(model),
        capture_layer_ids=[1, 3],
        capture_gdn_states=False,
    )

    assert len(out.hidden_states) == 2
    for hidden in out.hidden_states:
        assert hidden.shape == (1, 3, 128)
    assert out.gdn_states == []


def test_dflash_forward_captures_gdn_states_per_linear_layer() -> None:
    """``capture_gdn_states=True`` emits one 11-tuple per gated-delta layer.

    ``full_attention_interval=2`` produces the layer-type pattern
    ``[linear, attn, linear, attn]``, so a 4-layer model has 2 linear
    layers and we expect 2 gdn_state tuples.
    """
    model = _build_tiny_qwen3_5(num_layers=4)
    inputs = mx.array([[1, 2, 3, 4]])

    out = qwen3_5_dflash_forward(
        model,
        inputs,
        cache=_fresh_cache(model),
        capture_layer_ids=None,
        capture_gdn_states=True,
    )

    layer_is_linear = [bool(layer.is_linear) for layer in model.layers]
    expected_gdn_count = sum(layer_is_linear)
    assert len(out.gdn_states) == expected_gdn_count

    for gdn_state in out.gdn_states:
        assert len(gdn_state) == 11
        # First 5 elements are (q, k, v, a, b) -- per-token tensors with
        # T==seq dim. Tensor shapes vary by layer config (q/k vs v
        # heads), but they all share a leading (B, T) prefix.
        for tensor in gdn_state[:5]:
            assert tensor.shape[0] == 1, "batch dim"
            assert tensor.shape[1] == 4, "sequence dim"
        # ``state``/``mask`` are optional; ``conv_input``+``K`` are the
        # rollback's rewind handle.
        conv_input = gdn_state[9]
        conv_kernel_size = gdn_state[10]
        assert conv_input.shape[0] == 1
        assert conv_kernel_size == 4


def test_dflash_forward_returns_empty_sinks_when_disabled() -> None:
    """All flags off still produces a valid output (empty sinks)."""
    model = _build_tiny_qwen3_5()
    inputs = mx.array([[1, 2]])

    out = qwen3_5_dflash_forward(
        model,
        inputs,
        cache=_fresh_cache(model),
        capture_layer_ids=None,
        capture_gdn_states=False,
    )

    assert out.hidden_states == []
    assert out.gdn_states == []
    assert out.logits.shape == (1, 2, 128)


def test_rollback_speculative_cache_trims_kv_and_rewinds_ssm() -> None:
    """Rollback trims attention KV caches AND rewinds gated-delta state.

    Run a 4-token speculative block, then rollback claiming
    ``accepted=1`` (so we keep 2 of 4 tokens and trim 2). The KV
    caches must report ``offset==2`` post-trim; the SSM caches must
    have their conv-state slot rewound to a length-3 window
    (``conv_kernel_size - 1``).
    """
    model = _build_tiny_qwen3_5(num_layers=4)
    block_size = 4
    inputs = mx.array([[1, 2, 3, 4]])

    caches = _fresh_cache(model)
    out = qwen3_5_dflash_forward(model, inputs, cache=caches, capture_gdn_states=True)
    assert len(out.gdn_states) == 2  # two linear layers in this config

    # Pre-rollback: KV caches advanced by block_size; SSM caches hold
    # the post-block conv-input and state.
    pre_rollback_offsets = _kv_offsets(caches)
    assert pre_rollback_offsets, "expected at least one KV cache to inspect"
    assert all(off == block_size for off in pre_rollback_offsets)

    result = qwen3_5_rollback_speculative_cache(
        model,
        caches=caches,
        gdn_states=out.gdn_states,
        accepted=1,
        block_size=block_size,
    )
    assert result == 1  # max(accepted)

    # Post-rollback: KV caches trimmed back to (accepted+1)==2 tokens.
    post_rollback_offsets = _kv_offsets(caches)
    assert all(off == 2 for off in post_rollback_offsets)

    # SSM caches: slot 0 (conv_input) rewound to ``conv_kernel_size-1``
    # tokens; slot 1 (state) replaced by the replayed gated-delta state.
    ssm_caches = _ssm_caches(caches)
    for index in range(len(ssm_caches)):
        ssm_cache_: Any = ssm_caches[index]  # pyright: ignore[reportAny]
        conv_state = cast(mx.array, ssm_cache_[0])
        ssm_state = cast("mx.array | None", ssm_cache_[1])
        assert int(conv_state.shape[1]) == 3
        assert ssm_state is not None


def test_rollback_speculative_cache_zero_acceptance() -> None:
    """``accepted=0`` is the canonical "verify rejected token 0" case.

    Block of size 4, accepted=0 means we keep 1 token and trim 3.
    Both the KV trim path (``trim>0``) and the SSM rewind (with the
    smallest valid replay length) are exercised.
    """
    model = _build_tiny_qwen3_5()
    block_size = 4
    inputs = mx.array([[5, 6, 7, 8]])

    caches = _fresh_cache(model)
    out = qwen3_5_dflash_forward(model, inputs, cache=caches, capture_gdn_states=True)

    result = qwen3_5_rollback_speculative_cache(
        model,
        caches=caches,
        gdn_states=out.gdn_states,
        accepted=0,
        block_size=block_size,
    )
    assert result == 0

    # accepted=0 -> keep 1 of 4 -> offset trimmed to 1.
    assert all(off == 1 for off in _kv_offsets(caches))
