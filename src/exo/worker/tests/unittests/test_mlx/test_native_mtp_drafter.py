"""Unit tests for :mod:`exo.worker.engines.mlx.generator.native_mtp_drafter`.

Exercises:

- ``is_native_mtp_dispatchable`` correctly disambiguates the vendored
  MTP-capable :class:`vendor.qwen3_5_mtp.Model` from plain ``mx``
  modules and from ``unittest.mock.MagicMock`` (which auto-creates
  any attribute on access).
- :class:`NativeMTPDrafter` constructor validation (rejects ``k<1``)
  and trivial property surface (``mode``, ``num_draft_tokens``).
- :func:`prime_mtp_cache_from_prompt` returns ``N-1`` positions
  primed for prompts of size ``>=2`` and ``0`` otherwise.
- End-to-end ``stream`` smoke against the tiny synthetic Qwen3.5/6
  model fixture (mirrors the pattern used in
  ``vendor/tests/test_qwen3_5_mtp.py``): runs K=1 / K=2 / K=3 and
  asserts the drafter emits tokens, populates metrics, and respects
  ``max_tokens``.

These tests use the synthetic tiny model so the whole suite runs in
seconds without requiring any real model download. The opt-in parity
test against the real MTPLX artifact lives separately.
"""

# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportAny=false

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Dict, cast
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.generator.native_mtp_drafter import (
    NativeMTPDrafter,
    _eos_ids_from_tokenizer,
    _gdn_state_history_commit_default,
    _moe_verifier_policy,
    _target_post_norm_hidden,
    is_native_mtp_dispatchable,
    prime_mtp_cache_from_prompt,
    prime_mtp_cache_from_prompt_incremental,
    rebuild_prompt_cache_and_prime_mtp_cache_incremental,
    rebuild_prompt_cache_incremental,
)
from exo.worker.engines.mlx.types import Model as ExoModel
from exo.worker.engines.mlx.vendor.qwen3_5_mtp import Model, ModelArgs


def _tiny_text_config(*, with_mtp: bool = True) -> Dict[str, Any]:
    """Mirror the tiny config used by ``vendor/tests/test_qwen3_5_mtp.py``."""
    cfg: Dict[str, Any] = {
        "model_type": "qwen3_5_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "num_key_value_heads": 2,
        "max_position_embeddings": 256,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "head_dim": 16,
        "full_attention_interval": 2,
        "num_experts": 0,
        "rope_parameters": {
            "type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
        },
    }
    if with_mtp:
        cfg["mtp_num_hidden_layers"] = 1
    return cfg


def _tiny_model(*, with_mtp: bool = True) -> ExoModel:
    args = ModelArgs(
        model_type="qwen3_5", text_config=_tiny_text_config(with_mtp=with_mtp)
    )
    model = Model(args)
    mx.eval(model.parameters())
    # The vendored ``Model`` subclass satisfies the exo Model Protocol's
    # ``__call__`` shape modulo the optional ``return_hidden`` kwarg the
    # native-MTP path uses. Cast through ``object`` to defuse the
    # basedpyright ``reportInvalidCast`` for not-sufficiently-overlapping
    # types; runtime stream tests below catch any real Protocol drift.
    return cast(ExoModel, cast(object, model))


class _FakeDetokenizer:
    """Minimal detokenizer that records tokens added between yields.

    Stream consumers walk ``last_segment`` after every ``add_token`` and
    then ``finalize`` is called once before the closing yield. We only
    need an empty ``last_segment`` for the drafter to construct
    :class:`GenerationResponse` -- the test asserts emitted tokens via
    ``GenerationResponse.token`` directly, not via the segment string.
    """

    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.last_segment: str = ""
        self.finalized: bool = False

    def reset(self) -> None:
        self.tokens = []
        self.last_segment = ""
        self.finalized = False

    def add_token(self, token: int) -> None:
        self.tokens.append(int(token))

    def finalize(self) -> None:
        self.finalized = True


class _FakeTokenizer:
    """Fake :class:`mlx_lm.tokenizer_utils.TokenizerWrapper` minimal shim."""

    def __init__(self, eos_token_ids: Iterable[int] | None = None) -> None:
        self.detokenizer: _FakeDetokenizer = _FakeDetokenizer()
        self.eos_token_ids: Iterable[int] = list(eos_token_ids or [])


def _identity_sampler(logits: mx.array) -> mx.array:
    """Greedy sampler used by smoke tests."""
    return mx.argmax(logits, axis=-1)


def _empty_processors() -> list[Callable[[mx.array, mx.array], mx.array]]:
    return []


# --------------------------------------------------------------------------- #
# is_native_mtp_dispatchable
# --------------------------------------------------------------------------- #


class TestIsNativeMtpDispatchable:
    def test_vendored_model_is_dispatchable(self) -> None:
        model = _tiny_model(with_mtp=True)
        assert is_native_mtp_dispatchable(model) is True

    def test_vendored_model_without_mtp_layers_is_still_class_dispatchable(
        self,
    ) -> None:
        """The class-level marker doesn't depend on whether MTP layers exist.

        The dispatch-side gate only checks the model class; the runtime
        check for actual MTP availability is the loader / probe gate.
        A card without MTP would never produce this class via the loader,
        so the gate is conservative in the right direction.
        """
        model = _tiny_model(with_mtp=False)
        assert is_native_mtp_dispatchable(model) is True

    def test_magicmock_is_not_dispatchable(self) -> None:
        """``MagicMock`` auto-creates attributes; the marker check rejects it.

        Pre-fix the dispatcher used ``hasattr(model, "mtp_forward")`` which
        returned ``True`` for any ``MagicMock`` instance because attribute
        access auto-creates a child mock. That falsely engaged the
        NativeMTPDrafter in unrelated routing tests.
        """
        fake = MagicMock()
        assert is_native_mtp_dispatchable(fake) is False

    def test_plain_object_is_not_dispatchable(self) -> None:
        assert is_native_mtp_dispatchable(object()) is False

    def test_none_is_not_dispatchable(self) -> None:
        assert is_native_mtp_dispatchable(None) is False


# --------------------------------------------------------------------------- #
# NativeMTPDrafter constructor / property surface
# --------------------------------------------------------------------------- #


class TestNativeMTPDrafterConstructor:
    def test_rejects_zero_k(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            NativeMTPDrafter(k=0)

    def test_rejects_negative_k(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            NativeMTPDrafter(k=-1)

    def test_mode_is_model(self) -> None:
        drafter = NativeMTPDrafter(k=1)
        assert drafter.mode == "model"

    def test_num_draft_tokens_matches_k(self) -> None:
        for k in (1, 2, 3):
            assert NativeMTPDrafter(k=k).num_draft_tokens == k

    def test_initial_metrics_are_zeroed(self) -> None:
        drafter = NativeMTPDrafter(k=2)
        assert drafter.metrics() == {
            "proposed_draft_tokens": 0,
            "accepted_draft_tokens": 0,
            "spec_decode_rounds": 0,
        }

    def test_metrics_is_a_copy(self) -> None:
        """``metrics()`` returns a fresh dict so callers can't mutate state."""
        drafter = NativeMTPDrafter(k=1)
        snap = drafter.metrics()
        snap["proposed_draft_tokens"] = 99
        assert drafter.metrics()["proposed_draft_tokens"] == 0


def test_eos_ids_from_tokenizer_accepts_set_eos_ids() -> None:
    """Real MLX TokenizerWrapper exposes Qwen EOS IDs as a set."""
    tokenizer = _FakeTokenizer(eos_token_ids={248044, 248046})

    assert sorted(_eos_ids_from_tokenizer(cast(Any, tokenizer))) == [248044, 248046]


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_text"])
def test_gdn_state_history_commit_defaults_on_for_dense_qwen3_5(
    model_type: str,
) -> None:
    assert (
        _gdn_state_history_commit_default(
            model_type=model_type,
            moe_verifier_policy="safe",
        )
        is True
    )


def test_gdn_state_history_commit_defaults_on_for_route_locked_moe_only() -> None:
    assert (
        _gdn_state_history_commit_default(
            model_type="qwen3_5_moe",
            moe_verifier_policy="route_locked",
        )
        is True
    )
    assert (
        _gdn_state_history_commit_default(
            model_type="qwen3_5_moe",
            moe_verifier_policy="safe",
        )
        is False
    )


def test_moe_verifier_policy_defaults_to_route_locked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXO_NATIVE_MTP_MOE_VERIFY", raising=False)

    assert _moe_verifier_policy() == "route_locked"


def test_moe_verifier_policy_can_fall_back_to_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_NATIVE_MTP_MOE_VERIFY", "safe")

    assert _moe_verifier_policy() == "safe"


# --------------------------------------------------------------------------- #
# prime_mtp_cache_from_prompt
# --------------------------------------------------------------------------- #


class TestPrimeMTPCacheFromPrompt:
    def test_returns_zero_for_empty_prompt(self) -> None:
        model = _tiny_model(with_mtp=True)
        mtp_cache = model.make_mtp_cache()
        assert (
            prime_mtp_cache_from_prompt(
                model=model, full_prompt_tokens=[], mtp_cache=mtp_cache
            )
            == 0
        )

    def test_returns_zero_for_single_token_prompt(self) -> None:
        model = _tiny_model(with_mtp=True)
        mtp_cache = model.make_mtp_cache()
        assert (
            prime_mtp_cache_from_prompt(
                model=model, full_prompt_tokens=[5], mtp_cache=mtp_cache
            )
            == 0
        )

    def test_returns_nminus1_for_normal_prompt(self) -> None:
        model = _tiny_model(with_mtp=True)
        mtp_cache = model.make_mtp_cache()
        primed = prime_mtp_cache_from_prompt(
            model=model,
            full_prompt_tokens=[1, 2, 3, 4, 5],
            mtp_cache=mtp_cache,
        )
        assert primed == 4

    def test_primes_advances_mtp_cache_offset(self) -> None:
        """After priming N positions the MTP cache holds N entries."""
        model = _tiny_model(with_mtp=True)
        mtp_cache = model.make_mtp_cache()
        prompt = [1, 2, 3, 4, 5, 6]
        primed = prime_mtp_cache_from_prompt(
            model=model, full_prompt_tokens=prompt, mtp_cache=mtp_cache
        )
        assert primed == len(prompt) - 1
        # The MTP cache is a stock KVCache; its ``offset`` should be the
        # number of positions written.
        offsets = [getattr(c, "offset", -1) for c in mtp_cache]
        assert offsets == [len(prompt) - 1]


class TestHiddenWithoutLogits:
    def test_matches_outer_model_return_hidden_and_cache(self) -> None:
        model = _tiny_model(with_mtp=True)
        token = mx.array([[7]], dtype=mx.int32)
        next_token = mx.array([[8]], dtype=mx.int32)

        outer_cache = list(cast(Any, model).make_cache())
        _outer_logits, outer_hidden = cast(Any, model)(
            token,
            cache=outer_cache,
            return_hidden=True,
        )
        body_cache = list(cast(Any, model).make_cache())
        body_hidden = _target_post_norm_hidden(
            model=model,
            inputs=token,
            cache=cast(Any, body_cache),
        )
        mx.eval(outer_hidden, body_hidden)
        assert float(mx.max(mx.abs(outer_hidden - body_hidden)).item()) == 0.0

        outer_next_logits, _outer_next_hidden = cast(Any, model)(
            next_token,
            cache=outer_cache,
            return_hidden=True,
        )
        body_next_logits, _body_next_hidden = cast(Any, model)(
            next_token,
            cache=body_cache,
            return_hidden=True,
        )
        mx.eval(outer_next_logits, body_next_logits)
        assert float(mx.max(mx.abs(outer_next_logits - body_next_logits)).item()) == 0.0


class TestFusedPromptAndMtpPriming:
    def test_matches_separate_incremental_paths(self) -> None:
        model = _tiny_model(with_mtp=True)
        prompt = [1, 2, 3, 4, 5, 6]

        separate_prompt_cache = list(cast(Any, model).make_cache())
        separate_mtp_cache = list(cast(Any, model).make_mtp_cache())
        separate_rebuilt = rebuild_prompt_cache_incremental(
            model=model,
            full_prompt_tokens=prompt,
            prompt_cache=cast(Any, separate_prompt_cache),
        )
        separate_primed = prime_mtp_cache_from_prompt_incremental(
            model=model,
            full_prompt_tokens=prompt,
            mtp_cache=separate_mtp_cache,
        )

        fused_prompt_cache = list(cast(Any, model).make_cache())
        fused_mtp_cache = list(cast(Any, model).make_mtp_cache())
        fused_rebuilt, fused_primed = (
            rebuild_prompt_cache_and_prime_mtp_cache_incremental(
                model=model,
                full_prompt_tokens=prompt,
                prompt_cache=cast(Any, fused_prompt_cache),
                mtp_cache=fused_mtp_cache,
            )
        )

        assert fused_rebuilt == separate_rebuilt == len(prompt) - 2
        assert fused_primed == separate_primed == len(prompt) - 1
        assert [getattr(c, "offset", -1) for c in fused_prompt_cache] == [
            getattr(c, "offset", -1) for c in separate_prompt_cache
        ]
        assert [getattr(c, "offset", -1) for c in fused_mtp_cache] == [
            getattr(c, "offset", -1) for c in separate_mtp_cache
        ]

        def next_logits_and_draft(
            prompt_cache: list[Any], mtp_cache: list[Any]
        ) -> tuple[mx.array, mx.array]:
            prompt_tail = mx.array([prompt[-2:]], dtype=mx.int32)
            _tail_logits, _tail_hidden = cast(Any, model)(
                prompt_tail[:, :-1], cache=prompt_cache, return_hidden=True
            )
            first_logits, first_hidden = cast(Any, model)(
                prompt_tail[:, -1:], cache=prompt_cache, return_hidden=True
            )
            current_token_arr = mx.argmax(first_logits, axis=-1).astype(mx.int32)
            mtp_logits, _mtp_hidden = cast(Any, model).mtp_forward(
                first_hidden[:, -1:, :],
                current_token_arr,
                mtp_cache=mtp_cache,
                return_hidden=True,
            )
            mx.eval(first_logits, mtp_logits)
            return first_logits, mtp_logits

        separate_first, separate_mtp = next_logits_and_draft(
            separate_prompt_cache, separate_mtp_cache
        )
        fused_first, fused_mtp = next_logits_and_draft(
            fused_prompt_cache, fused_mtp_cache
        )

        assert float(mx.max(mx.abs(separate_first - fused_first)).item()) == 0.0
        assert float(mx.max(mx.abs(separate_mtp - fused_mtp)).item()) == 0.0


# --------------------------------------------------------------------------- #
# End-to-end stream smoke tests
# --------------------------------------------------------------------------- #


def _drive_stream(
    *,
    drafter: NativeMTPDrafter,
    model: ExoModel,
    tokenizer: _FakeTokenizer,
    prompt_full: list[int],
    max_tokens: int,
) -> list[Any]:
    """Run the K-step drafter end-to-end and collect every yielded response.

    Mirrors what ``mlx_generate`` does at the dispatch site: prefills
    the prompt cache aligned to ``full_prompt[:-2]``, then hands the
    last two tokens to ``drafter.stream``.
    """
    cache: list[Any] = list(cast(Any, model).make_cache())
    # exo.prefill equivalent: feed prompt[:-2] (so cache holds N-2 positions).
    if len(prompt_full) >= 3:
        prefill = mx.array(prompt_full[:-2], dtype=mx.int32)[None]
        _ = model(prefill, cache=cache)
        mx.eval([c.state for c in cache if hasattr(c, "state")])
    prompt_tail = mx.array(prompt_full[-2:], dtype=mx.int32)
    responses: list[Any] = []
    for response in drafter.stream(
        model=model,
        tokenizer=cast(Any, tokenizer),
        prompt=prompt_tail,
        context_tokens=prompt_full,
        prompt_cache=cast(Any, cache),
        max_tokens=max_tokens,
        sampler=_identity_sampler,
        logits_processors=_empty_processors(),
    ):
        responses.append(response)
        if response.finish_reason is not None:
            break
    return responses


@pytest.mark.parametrize("k", [1, 2, 3])
def test_stream_emits_tokens_and_populates_metrics(k: int) -> None:
    """K=1/2/3 streams emit tokens, terminate, and stamp metrics."""
    model = _tiny_model(with_mtp=True)
    tokenizer = _FakeTokenizer(eos_token_ids=[])
    drafter = NativeMTPDrafter(k=k)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    max_tokens = 8

    responses = _drive_stream(
        drafter=drafter,
        model=model,
        tokenizer=tokenizer,
        prompt_full=prompt,
        max_tokens=max_tokens,
    )

    # At least one response is yielded (the first emitted token from the
    # initial forward) plus a closing chunk.
    assert len(responses) >= 2
    final = responses[-1]
    assert final.finish_reason in {"stop", "length"}
    assert final.generation_tokens <= max_tokens
    assert final.generation_tokens >= 1

    metrics = drafter.metrics()
    # spec_decode_rounds is 0 when the first emitted token alone covered
    # max_tokens; for max_tokens=8 with random init the loop should run
    # at least one round.
    assert metrics["spec_decode_rounds"] >= 0
    assert metrics["proposed_draft_tokens"] == k * metrics["spec_decode_rounds"]
    assert (
        metrics["accepted_draft_tokens"] >= 0
        and metrics["accepted_draft_tokens"] <= metrics["proposed_draft_tokens"]
    )


def test_stream_raises_on_non_dispatchable_model() -> None:
    """A bare MagicMock target fails the dispatch-side gate inside ``stream``."""
    drafter = NativeMTPDrafter(k=1)
    fake_model = MagicMock()
    fake_model.make_mtp_cache.return_value = [MagicMock()]
    tokenizer = _FakeTokenizer()
    with pytest.raises(RuntimeError, match="is_native_mtp_dispatchable"):
        # ``next`` to actually enter the generator body; the dispatchable
        # check happens before any yield.
        next(
            drafter.stream(
                model=cast(Any, fake_model),
                tokenizer=cast(Any, tokenizer),
                prompt=mx.array([1, 2], dtype=mx.int32),
                context_tokens=[1, 2],
                prompt_cache=cast(Any, []),
                max_tokens=4,
                sampler=_identity_sampler,
                logits_processors=_empty_processors(),
            )
        )


def test_stream_stops_at_immediate_eos() -> None:
    """When the first emitted token IS an EOS, the stream terminates immediately."""
    model = _tiny_model(with_mtp=True)
    # Pin every vocab id as EOS so the first emitted token (whatever it
    # is) terminates the stream. This is the cheapest way to exercise
    # the immediate-EOS branch without needing to control the model's
    # output distribution.
    tokenizer = _FakeTokenizer(eos_token_ids=set(range(256)))
    drafter = NativeMTPDrafter(k=1)
    responses = _drive_stream(
        drafter=drafter,
        model=model,
        tokenizer=tokenizer,
        prompt_full=[1, 2, 3, 4],
        max_tokens=8,
    )
    # The immediate-EOS branch yields a SINGLE response with finish_reason="stop".
    assert len(responses) == 1
    assert responses[0].finish_reason == "stop"
    assert responses[0].generation_tokens == 1
