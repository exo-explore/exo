"""Tests for the Gemma 4 MTP target-side hooks.

We build a tiny ``Gemma4Model`` directly from ``ModelArgs`` (no
checkpoint download) and exercise the three hooks vendored from
mlx-vlm:

- :func:`attach_mtp_hooks` / :func:`has_mtp_hooks` -- gating used by
  the generator dispatch to confirm a target is hook-capable.
- :func:`gemma4_mtp_forward` -- captures pre-norm last-layer hidden +
  per-layer-type shared-KV snapshot WITHOUT changing the logits.
- :func:`gemma4_rollback_speculative_cache` -- trims target KV
  caches after a partial-acceptance speculative round.

The "tiny" Gemma 4 (2 layers, hidden_size=64, vocab_size=100) is
small enough to exercise both the sliding-attention and
full-attention layer paths plus the shared-KV ``previous_kvs``
indirection within a CPU-only test budget. We do NOT exercise the
per-layer-input branch (``hidden_size_per_layer_input>0``) here --
that path is exercised by the round-loop integration test which
runs against a card-flavoured config.
"""

from __future__ import annotations

from typing import Any, cast

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache
from mlx_lm.models.gemma4_text import Model as Gemma4Model
from mlx_lm.models.gemma4_text import ModelArgs

from exo.worker.engines.mlx.vendor.gemma4_mtp_hooks import (
    Gemma4MTPForwardOutput,
    attach_mtp_hooks,
    gemma4_mtp_forward,
    gemma4_rollback_speculative_cache,
    has_mtp_hooks,
)


def _build_tiny_gemma4(*, num_layers: int = 2) -> Gemma4Model:
    """Construct a small Gemma 4 model in-memory.

    Both ``sliding_attention`` and ``full_attention`` layer types are
    represented (when ``num_layers >= 2``) so the per-layer-type
    shared-KV capture path is exercised. ``hidden_size_per_layer_input``
    is 0 so the per-layer-input projection branch is skipped --
    that branch is tested separately by the round-loop integration
    test against a real card config.
    """
    layer_types = (
        ["sliding_attention", "full_attention"]
        if num_layers >= 2
        else ["sliding_attention"]
    ) * (num_layers // 2 + 1)
    args = ModelArgs(
        model_type="gemma4_text",
        hidden_size=64,
        num_hidden_layers=num_layers,
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
        layer_types=layer_types[:num_layers],
        tie_word_embeddings=True,
        final_logit_softcapping=30.0,
    )
    model = Gemma4Model(args)
    model.eval()
    return model


def _fresh_cache(model: Gemma4Model) -> list[Any]:
    """Build a one-cache-per-layer list using mlx-lm's defaults.

    ``Model.make_cache`` returns the right per-layer cache types for
    Gemma 4 (sliding layers get ``RotatingKVCache``, full-attention
    layers get ``KVCache``). The hooked forward expects this exact
    list shape -- reusing the helper keeps the test honest about
    the integration surface.
    """
    return cast("list[Any]", model.make_cache())


def test_attach_mtp_hooks_marks_target() -> None:
    model = _build_tiny_gemma4()
    assert not has_mtp_hooks(model)

    attach_mtp_hooks(model)

    assert has_mtp_hooks(model)


def test_attach_mtp_hooks_idempotent() -> None:
    model = _build_tiny_gemma4()
    attach_mtp_hooks(model)
    attach_mtp_hooks(model)

    assert has_mtp_hooks(model)


def test_attach_mtp_hooks_rejects_non_gemma4() -> None:
    """The dispatch gate refuses targets that aren't Gemma 4."""

    class NotGemma4:
        pass

    target = NotGemma4()

    with pytest.raises(TypeError, match="gemma4_text.Model"):
        attach_mtp_hooks(target)

    assert not has_mtp_hooks(target)


def test_has_mtp_hooks_default_false() -> None:
    model = _build_tiny_gemma4()
    assert not has_mtp_hooks(model)
    assert not has_mtp_hooks(object())


def test_attach_mtp_hooks_walks_multimodal_wrapper() -> None:
    """Vision-capable Gemma 4 loads as a wrapper exposing the LM via
    ``.language_model``. The attach gate must walk the wrapper so the
    multimodal target is treated identically to the text-only one --
    otherwise vision-capable cards (e.g. ``gemma-4-26b-a4b-it-4bit``)
    would always degrade to the standard drafter despite declaring a
    ``coupled_drafter``.
    """

    text_model = _build_tiny_gemma4()

    class _MultimodalWrapper:
        """Mimics ``mlx_lm.models.gemma4.Model``'s relevant surface."""

        def __init__(self, lm: Gemma4Model) -> None:
            self.language_model: Gemma4Model = lm

    wrapper = _MultimodalWrapper(text_model)

    assert not has_mtp_hooks(wrapper)
    assert not has_mtp_hooks(text_model)

    attach_mtp_hooks(wrapper)

    # Both the wrapper and the inner LM see the sentinel; the dispatch
    # site reads it on whichever instance it has a handle to.
    assert has_mtp_hooks(wrapper)
    assert has_mtp_hooks(text_model)


def test_gemma4_mtp_forward_logits_match_unhooked_call() -> None:
    """The hook must NOT change the logits the target produces.

    The MTP round loop's "verify" forward replaces the standard
    ``Model.__call__`` only for the verify slot; if the hooked path
    diverged numerically, drafted-token acceptance would be
    silently miscalibrated.
    """
    model = _build_tiny_gemma4()
    inputs = mx.array([[1, 2, 3, 4, 5]])

    cache_unhooked = _fresh_cache(model)
    unhooked_logits = model(inputs, cache=cache_unhooked)

    cache_hooked = _fresh_cache(model)
    out = gemma4_mtp_forward(model, inputs, cache=cache_hooked)

    assert isinstance(out, Gemma4MTPForwardOutput)
    assert mx.allclose(out.logits, unhooked_logits, atol=1e-5).item() is True


def test_gemma4_mtp_forward_captures_hidden_state() -> None:
    """``return_hidden=True`` populates the last decoder layer's pre-norm output."""
    model = _build_tiny_gemma4()
    inputs = mx.array([[1, 2, 3]])

    out = gemma4_mtp_forward(
        model,
        inputs,
        cache=_fresh_cache(model),
        return_hidden=True,
        return_shared_kv=False,
    )

    assert len(out.hidden_states) == 1
    last_hidden = out.hidden_states[0]
    assert last_hidden.shape == (1, 3, 64)
    assert out.shared_kv_states == {}


def test_gemma4_mtp_forward_captures_shared_kv_per_layer_type() -> None:
    """``return_shared_kv=True`` populates one (K, V) per layer type."""
    model = _build_tiny_gemma4(num_layers=2)
    inputs = mx.array([[7, 8, 9, 10]])

    out = gemma4_mtp_forward(
        model,
        inputs,
        cache=_fresh_cache(model),
        return_hidden=False,
        return_shared_kv=True,
    )

    assert out.hidden_states == []
    assert set(out.shared_kv_states.keys()) == {"sliding_attention", "full_attention"}
    for layer_type, (keys, values) in out.shared_kv_states.items():
        assert keys.shape[0] == 1, f"{layer_type} keys batch dim"
        assert keys.shape == values.shape, f"{layer_type} K/V shape parity"


def test_gemma4_mtp_forward_returns_empty_sinks_when_disabled() -> None:
    """Both flags off still produces a valid output (empty sinks)."""
    model = _build_tiny_gemma4()
    inputs = mx.array([[1, 2]])

    out = gemma4_mtp_forward(
        model,
        inputs,
        cache=_fresh_cache(model),
        return_hidden=False,
        return_shared_kv=False,
    )

    assert out.hidden_states == []
    assert out.shared_kv_states == {}
    assert out.logits.shape == (1, 2, 100)


def test_rollback_speculative_cache_trims_block_tail() -> None:
    """``trim`` removes ``block_size - (max(accepted)+1)`` tokens per cache.

    Set up a cache that's been advanced ``block_size`` tokens (a full
    speculative block); after rollback for ``accepted=2`` (3 of 4
    tokens accepted) the cache should retain only those 3 tokens.
    """
    model = _build_tiny_gemma4()
    block_size = 4

    full_cache = KVCache()
    initial_keys = mx.zeros((1, 1, block_size, 32))
    initial_values = mx.zeros((1, 1, block_size, 32))
    full_cache.update_and_fetch(initial_keys, initial_values)

    assert full_cache.offset == block_size

    accepted_count = gemma4_rollback_speculative_cache(
        model,
        caches=[full_cache, None],
        gdn_states=None,
        accepted=2,
        block_size=block_size,
    )

    assert accepted_count == 2
    assert full_cache.offset == 3, "trim should retain accepted+1 tokens"


def test_rollback_speculative_cache_full_acceptance_no_trim() -> None:
    """When all drafted tokens are accepted, the cache is unchanged."""
    model = _build_tiny_gemma4()
    block_size = 4

    cache = KVCache()
    keys = mx.zeros((1, 1, block_size, 32))
    values = mx.zeros((1, 1, block_size, 32))
    cache.update_and_fetch(keys, values)

    accepted_count = gemma4_rollback_speculative_cache(
        model,
        caches=[cache],
        gdn_states=None,
        accepted=block_size - 1,
        block_size=block_size,
    )

    assert accepted_count == block_size - 1
    assert cache.offset == block_size


def test_rollback_speculative_cache_skips_none_slots() -> None:
    """Shared-KV layers carry ``None`` cache slots and must be skipped."""
    model = _build_tiny_gemma4()
    block_size = 3

    cache = KVCache()
    keys = mx.zeros((1, 1, block_size, 32))
    values = mx.zeros((1, 1, block_size, 32))
    cache.update_and_fetch(keys, values)

    accepted_count = gemma4_rollback_speculative_cache(
        model,
        caches=[None, cache, None],
        gdn_states=None,
        accepted=1,
        block_size=block_size,
    )

    assert accepted_count == 1
    assert cache.offset == 2


def test_rollback_speculative_cache_accepts_mx_array_accepted() -> None:
    """``accepted`` may be an ``mx.array`` (batched) -- single-row case."""
    model = _build_tiny_gemma4()
    block_size = 4

    cache = KVCache()
    keys = mx.zeros((1, 1, block_size, 32))
    values = mx.zeros((1, 1, block_size, 32))
    cache.update_and_fetch(keys, values)

    accepted_count = gemma4_rollback_speculative_cache(
        model,
        caches=[cache],
        gdn_states=None,
        accepted=mx.array([2]),
        block_size=block_size,
    )

    assert accepted_count == 2
    assert cache.offset == 3
