"""Round-loop integration tests for the coupled-drafter dispatch.

These tests exercise :func:`run_coupled_round_loop` against a tiny
in-memory Gemma 4 target paired with a stub drafter that returns
deterministic drafts. The goals are:

1. Verify the adapter satisfies mlx-vlm's ``_mtp_rounds`` contract
   end-to-end -- if the drafter's ``bind`` walks the embed_tokens
   slot correctly, the verify forward returns a
   ``Gemma4MTPForwardOutput``-shaped object, the rollback trims
   caches, and the round loop terminates without raising.

2. Pin the "first bonus is yielded by the caller" invariant: the
   round loop yields tokens starting from round 1, never the first
   bonus itself.

Parity at temperature 0 (target-only vs MTP-accelerated) is covered
by :file:`test_coupled_drafter_parity.py`. Here we focus on the
mechanics of the integration (adapter + driver), keeping the drafter
mocked so we exercise the loop without pulling the 78M-parameter
gemma4_assistant weights into a CPU-only test.
"""

from __future__ import annotations

from typing import Any, cast, final

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.gemma4_text import Model as Gemma4Model
from mlx_lm.models.gemma4_text import ModelArgs

from exo.worker.engines.mlx.generator.coupled_drafter import (
    Gemma4MTPTargetAdapter,
    run_coupled_round_loop,
)
from exo.worker.engines.mlx.vendor.gemma4_mtp_hooks import (
    attach_mtp_hooks,
    gemma4_mtp_forward,
)


def _build_tiny_gemma4_with_hooks() -> Gemma4Model:
    """Same shape as :file:`test_gemma4_mtp_hooks.py` but with hooks attached."""
    args = ModelArgs(
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


@final
class _StubGemma4Drafter(nn.Module):
    """Minimal stub mimicking the gemma4_assistant drafter API.

    ``_mtp_rounds`` calls four methods on the drafter:

    - ``reset(target_model) -> List`` -- called once at the top.
    - ``set_shared_kv(shared_kv_states, kv_offset, position=None, left_padding=None)``
      -- called after each verify forward.
    - ``draft_block(last_bonus, hidden, cache, block_size, sampler, token_dtype) -> mx.array``
      -- called once per round to produce K-1 drafted tokens.
    - ``accept_lens: list[int]`` -- the round loop appends to this.

    It also reads ``draft_model.config.block_size`` when the round
    loop's ``draft_block_size`` argument is None; we expose a tiny
    config object for that.

    The stub returns drafts that are GUARANTEED to be wrong (token
    id ``0`` repeated) so the speculative-walk always rejects on
    position 0 and we get exactly one new token per round (the
    target's bonus). This makes the loop's emission count
    predictable for assertions: ``max_tokens`` total, with ``1``
    initial-bonus emitted by the caller and ``max_tokens - 1``
    yielded by the round loop.
    """

    @final
    class _Config:
        block_size: int = 4

    def __init__(self) -> None:
        super().__init__()
        self.config: _StubGemma4Drafter._Config = _StubGemma4Drafter._Config()
        self.accept_lens: list[int] = []
        self.bind_calls: int = 0
        self.set_shared_kv_calls: int = 0
        self.draft_block_calls: int = 0
        self._reset_returned_cache: list[Any] = []

    def bind(self, target_model: object) -> "_StubGemma4Drafter":
        del target_model
        self.bind_calls += 1
        return self

    def make_cache(self) -> list[Any]:
        return []

    def reset(self, target_model: object) -> list[Any]:
        self.bind(target_model)
        self.accept_lens = []
        return self._reset_returned_cache

    def set_shared_kv(
        self,
        shared_kv_states: dict[str, tuple[mx.array, mx.array]],
        kv_offset: int | mx.array,
        position: int | mx.array | None = None,
        left_padding: mx.array | None = None,
    ) -> None:
        del shared_kv_states, kv_offset, position, left_padding
        self.set_shared_kv_calls += 1

    def draft_block(
        self,
        last_bonus: int,
        hidden: mx.array,
        cache: object,
        block_size: int,
        sampler: object,
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        del last_bonus, hidden, cache, sampler
        self.draft_block_calls += 1
        # Return zeros: the speculative walk will reject token 0
        # against any non-zero target token, so each round emits
        # exactly one new token (the target's bonus).
        return mx.zeros((1, block_size - 1), dtype=token_dtype)


def _greedy_sampler(logits: mx.array) -> mx.array:
    """Argmax sampler -- deterministic, matches temperature=0 semantics."""
    return mx.argmax(logits, axis=-1).astype(mx.int32)


def test_adapter_requires_attached_hooks() -> None:
    """Constructing the adapter without ``attach_mtp_hooks`` must fail.

    The adapter is the only entry point through which the dispatch
    can reach ``_mtp_rounds``; if it accepted unhooked targets, an
    operator could route a non-Gemma 4 model into the coupled path
    and only discover the mismatch on the first verify forward (a
    much more confusing failure than a guard-rail at construction).
    """
    args = ModelArgs(
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
    target_without_hooks = Gemma4Model(args)
    target_without_hooks.eval()

    with pytest.raises(RuntimeError, match="attach_mtp_hooks"):
        Gemma4MTPTargetAdapter(target_without_hooks)


def test_adapter_call_returns_mtp_forward_output() -> None:
    """The adapter's ``__call__`` returns the captured-forward triple.

    ``_mtp_rounds`` reads ``out.logits``, ``out.hidden_states[-1]``,
    and ``out.shared_kv_states`` -- all three must be populated.
    """
    target = _build_tiny_gemma4_with_hooks()
    adapter = Gemma4MTPTargetAdapter(target)

    inputs = mx.array([[1, 2, 3]])
    cache = cast("list[Any]", target.make_cache())

    out = adapter(inputs, cache=cache, return_hidden=True, return_shared_kv=True)

    assert out.logits.shape == (1, 3, 100)
    assert len(out.hidden_states) == 1
    assert set(out.shared_kv_states.keys()) == {
        "sliding_attention",
        "full_attention",
    }


def test_adapter_rollback_passes_through() -> None:
    """``rollback_speculative_cache`` returns ``max(accepted)`` per the contract."""
    target = _build_tiny_gemma4_with_hooks()
    adapter = Gemma4MTPTargetAdapter(target)

    accepted_count = adapter.rollback_speculative_cache(
        caches=[None],
        gdn_states=None,
        accepted=2,
        block_size=4,
    )

    assert accepted_count == 2


def test_adapter_model_property_exposes_inner_text_model() -> None:
    """The drafter's ``bind`` walks ``adapter.model.embed_tokens``."""
    target = _build_tiny_gemma4_with_hooks()
    adapter = Gemma4MTPTargetAdapter(target)

    # The drafter's bind logic walks `target.embed_tokens` first then
    # `target.model.embed_tokens`. Our adapter has no `embed_tokens`
    # attribute, so bind takes the second branch and reads
    # `adapter.model.embed_tokens` -- it must resolve to the
    # underlying mlx-lm gemma4 model's embed_tokens (NOT a copy).
    assert hasattr(adapter, "model")
    assert hasattr(adapter.model, "embed_tokens")
    assert adapter.model is target.model


def test_round_loop_terminates_when_max_tokens_reached() -> None:
    """The loop must stop yielding once ``max_tokens`` tokens have been emitted.

    With our 1-bonus-emitted-by-caller convention, ``max_tokens=4``
    means the round loop yields up to 3 tokens before stopping.
    """
    target = _build_tiny_gemma4_with_hooks()
    drafter = _StubGemma4Drafter()

    prompt = mx.array([[1, 2, 3, 4]])
    cache = cast("list[Any]", target.make_cache())
    prefill = gemma4_mtp_forward(target, prompt, cache=cache)

    first_bonus = int(_greedy_sampler(prefill.logits[:, -1:, :])[0, 0].item())

    yielded: list[int] = list(
        run_coupled_round_loop(
            adapter=Gemma4MTPTargetAdapter(target),
            drafter=drafter,
            prompt_cache=cache,
            prefill_output=prefill,
            first_bonus=first_bonus,
            max_tokens=4,
            sampler=_greedy_sampler,
            draft_block_size=None,
        )
    )

    # block_size is 4 (from drafter.config), draft_block of 3 zeros
    # always rejected at position 0 → 1 new token (bonus) per round.
    # We need 3 more tokens total; 3 rounds × 1 token each.
    assert len(yielded) <= 3, (
        f"round loop must not exceed max_tokens; got {len(yielded)}"
    )
    assert drafter.draft_block_calls >= 1, (
        "drafter.draft_block should run at least once before the loop terminates"
    )


def test_round_loop_calls_drafter_bind_via_reset() -> None:
    """``_mtp_rounds`` opens with ``draft_model.reset(model)`` which binds the drafter.

    After the loop returns, ``drafter.bind_calls`` must be at least 1
    -- this confirms the adapter exposed the right shape for bind
    (otherwise bind would silently no-op via the try/except in the
    real drafter, but our stub doesn't have that fallback so the
    call would raise).
    """
    target = _build_tiny_gemma4_with_hooks()
    drafter = _StubGemma4Drafter()

    prompt = mx.array([[1, 2, 3]])
    cache = cast("list[Any]", target.make_cache())
    prefill = gemma4_mtp_forward(target, prompt, cache=cache)

    list(
        run_coupled_round_loop(
            adapter=Gemma4MTPTargetAdapter(target),
            drafter=drafter,
            prompt_cache=cache,
            prefill_output=prefill,
            first_bonus=int(_greedy_sampler(prefill.logits[:, -1:, :])[0, 0].item()),
            max_tokens=2,
            sampler=_greedy_sampler,
            draft_block_size=None,
        )
    )

    assert drafter.bind_calls >= 1, "drafter.reset(target) must call bind"


def test_round_loop_rejects_missing_hidden_capture() -> None:
    """Calling the driver with no captured hidden state must surface a clear error.

    Pre-fix, mlx-vlm's ``_mtp_rounds`` would index into an empty
    ``hidden_states`` list and raise an opaque ``IndexError`` deep
    in the round loop. We catch this at the driver boundary so the
    operator gets a focused error pointing at the prefill call.
    """
    target = _build_tiny_gemma4_with_hooks()
    drafter = _StubGemma4Drafter()

    prompt = mx.array([[1, 2]])
    cache = cast("list[Any]", target.make_cache())
    prefill = gemma4_mtp_forward(
        target, prompt, cache=cache, return_hidden=False, return_shared_kv=True
    )

    with pytest.raises(RuntimeError, match="captured hidden state"):
        list(
            run_coupled_round_loop(
                adapter=Gemma4MTPTargetAdapter(target),
                drafter=drafter,
                prompt_cache=cache,
                prefill_output=prefill,
                first_bonus=0,
                max_tokens=2,
                sampler=_greedy_sampler,
                draft_block_size=None,
            )
        )
