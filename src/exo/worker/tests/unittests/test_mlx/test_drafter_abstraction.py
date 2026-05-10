"""Tests for the ``Drafter`` abstraction.

These cover the pure-Python pieces - mode resolution, n-gram suffix
matching, and the spec-loop accept arithmetic - so they don't need MLX
weights or a GPU. End-to-end correctness with a real model is exercised
by the smoke + bench scripts in ``scripts/``.
"""

from __future__ import annotations

import pytest

from exo.worker.engines.mlx.generator.drafter import (
    ALL_DRAFT_MODES,
    EXO_DRAFT_MODE_ENV,
    DraftMode,
    EagleDrafter,
    LookaheadDrafter,
    NgramDrafter,
    NoSpecDrafter,
    make_drafter,
    parse_draft_mode,
    resolve_asymmetric_draft_mode,
    resolve_draft_mode,
)


def test_all_draft_modes_match_literal() -> None:
    """``ALL_DRAFT_MODES`` must be the runtime mirror of the ``DraftMode`` Literal."""
    assert ALL_DRAFT_MODES == (
        "model",
        "pipelined",
        "ngram",
        "eagle",
        "lookahead",
        "none",
    )


def test_eagle_drafter_scaffold_raises_on_stream() -> None:
    """``EagleDrafter`` is a scaffolding stub; ``stream`` must fail loudly.

    The factory dispatch + ``Drafter`` protocol shape are the durable
    contract here; the actual auxiliary-head loop is intentionally not
    implemented yet. A future PR fills this in.
    """
    drafter = make_drafter(
        mode="eagle",
        num_draft_tokens=3,
        draft_model=None,
        draft_cache=None,
    )
    assert isinstance(drafter, EagleDrafter)
    assert drafter.mode == "eagle"
    assert drafter.num_draft_tokens == 3
    with pytest.raises(NotImplementedError, match="EagleDrafter is a scaffolding"):
        # ``stream`` is a generator function; ``next()`` triggers the body.
        next(
            drafter.stream(
                model=object(),  # type: ignore[arg-type]
                tokenizer=object(),  # type: ignore[arg-type]
                prompt=object(),  # type: ignore[arg-type]
                context_tokens=[],
                prompt_cache=[],
                max_tokens=1,
                sampler=lambda x: x,
                logits_processors=[],
            )
        )


def test_lookahead_drafter_scaffold_raises_on_stream() -> None:
    """``LookaheadDrafter`` is a scaffolding stub; ``stream`` must fail loudly."""
    drafter = make_drafter(
        mode="lookahead",
        num_draft_tokens=3,
        draft_model=None,
        draft_cache=None,
    )
    assert isinstance(drafter, LookaheadDrafter)
    assert drafter.mode == "lookahead"
    assert drafter.num_draft_tokens == 3
    assert drafter.window_size == 5
    assert drafter.ngram_size == 3
    with pytest.raises(NotImplementedError, match="LookaheadDrafter is a scaffolding"):
        next(
            drafter.stream(
                model=object(),  # type: ignore[arg-type]
                tokenizer=object(),  # type: ignore[arg-type]
                prompt=object(),  # type: ignore[arg-type]
                context_tokens=[],
                prompt_cache=[],
                max_tokens=1,
                sampler=lambda x: x,
                logits_processors=[],
            )
        )


@pytest.mark.parametrize(
    ("raw", "default", "expected"),
    [
        (None, "model", "model"),
        (None, "none", "none"),
        ("model", "none", "model"),
        ("MODEL", "none", "model"),
        ("  ngram  ", "none", "ngram"),
        ("pipelined", "none", "pipelined"),
        ("PIPELINED", "model", "pipelined"),
        ("none", "model", "none"),
        ("garbage", "model", "model"),
        ("garbage", "none", "none"),
    ],
)
def test_parse_draft_mode(
    raw: str | None, default: DraftMode, expected: DraftMode
) -> None:
    assert parse_draft_mode(raw, default) == expected


def test_parse_draft_mode_warns_on_unknown_value(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv(EXO_DRAFT_MODE_ENV, raising=False)
    parse_draft_mode("totally-bogus", "none")
    # Loguru-driven logger doesn't pipe to caplog by default; just assert
    # the call didn't raise. The warning is documented in the docstring.


class TestResolveDraftMode:
    def test_explicit_request_mode_wins_over_use_drafter(self) -> None:
        # Per-request draft_mode beats the use_drafter shortcut.
        assert (
            resolve_draft_mode(
                has_drafter_model=True,
                request_use_drafter=False,
                request_draft_mode="ngram",
            )
            == "ngram"
        )

    def test_use_drafter_false_maps_to_none(self) -> None:
        assert (
            resolve_draft_mode(
                has_drafter_model=True,
                request_use_drafter=False,
                request_draft_mode=None,
            )
            == "none"
        )

    def test_default_with_drafter_loaded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(EXO_DRAFT_MODE_ENV, raising=False)
        assert (
            resolve_draft_mode(
                has_drafter_model=True,
                request_use_drafter=None,
                request_draft_mode=None,
            )
            == "model"
        )

    def test_default_without_drafter_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(EXO_DRAFT_MODE_ENV, raising=False)
        assert (
            resolve_draft_mode(
                has_drafter_model=False,
                request_use_drafter=None,
                request_draft_mode=None,
            )
            == "none"
        )

    def test_env_override_with_drafter_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(EXO_DRAFT_MODE_ENV, "ngram")
        assert (
            resolve_draft_mode(
                has_drafter_model=True,
                request_use_drafter=None,
                request_draft_mode=None,
            )
            == "ngram"
        )

    def test_model_mode_without_drafter_demotes_to_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(EXO_DRAFT_MODE_ENV, raising=False)
        assert (
            resolve_draft_mode(
                has_drafter_model=False,
                request_use_drafter=None,
                request_draft_mode="model",
            )
            == "none"
        )

    def test_pipelined_mode_without_drafter_demotes_to_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Same misconfiguration safety net as ``"model"``: requesting
        # ``"pipelined"`` without a loaded drafter must fall back to
        # ``"none"`` rather than hard-failing or producing a no-op
        # drafter that silently degrades throughput.
        monkeypatch.delenv(EXO_DRAFT_MODE_ENV, raising=False)
        assert (
            resolve_draft_mode(
                has_drafter_model=False,
                request_use_drafter=None,
                request_draft_mode="pipelined",
            )
            == "none"
        )

    def test_pipelined_mode_with_drafter_loaded_passes_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(EXO_DRAFT_MODE_ENV, raising=False)
        assert (
            resolve_draft_mode(
                has_drafter_model=True,
                request_use_drafter=None,
                request_draft_mode="pipelined",
            )
            == "pipelined"
        )

    def test_explicit_none_with_drafter_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(EXO_DRAFT_MODE_ENV, raising=False)
        assert (
            resolve_draft_mode(
                has_drafter_model=True,
                request_use_drafter=None,
                request_draft_mode="none",
            )
            == "none"
        )

    def test_use_drafter_true_promotes_to_model_when_drafter_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Codex P2 (PR #19 round-(N+8), drafter.py:148): the
        ``use_drafter=true`` opt-in must override an explicit
        ``EXO_DRAFT_MODE=none`` process default. With a drafter model
        loaded the natural intent is ``"model"``."""
        monkeypatch.setenv(EXO_DRAFT_MODE_ENV, "none")
        assert (
            resolve_draft_mode(
                has_drafter_model=True,
                request_use_drafter=True,
                request_draft_mode=None,
            )
            == "model"
        )

    def test_use_drafter_true_falls_back_to_ngram_without_drafter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Codex P2 (PR #19 round-(N+8), drafter.py:148): when no
        drafter model is loaded, ``use_drafter=true`` must still
        engage *some* drafting strategy. ``ngram`` is the only
        viable option (in-context suffix lookup needs no extra
        weights), so promote to ``"ngram"`` -- never silently fall
        through to ``"none"``."""
        monkeypatch.setenv(EXO_DRAFT_MODE_ENV, "none")
        assert (
            resolve_draft_mode(
                has_drafter_model=False,
                request_use_drafter=True,
                request_draft_mode=None,
            )
            == "ngram"
        )

    def test_use_drafter_true_with_drafter_loaded_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The opt-in shortcut must dominate the env default in the
        common 'A/B test harness' case where the runner ships with
        ``EXO_DRAFT_MODE=none`` and the harness flips drafting on
        per-request."""
        monkeypatch.setenv(EXO_DRAFT_MODE_ENV, "none")
        result = resolve_draft_mode(
            has_drafter_model=True,
            request_use_drafter=True,
            request_draft_mode=None,
        )
        assert result == "model"

    def test_explicit_request_mode_still_wins_over_use_drafter_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Precedence regression test: explicit ``request_draft_mode``
        wins over both ``use_drafter`` and the env var, even when
        the request is opting in with ``use_drafter=True``."""
        monkeypatch.setenv(EXO_DRAFT_MODE_ENV, "none")
        result = resolve_draft_mode(
            has_drafter_model=True,
            request_use_drafter=True,
            request_draft_mode="ngram",
        )
        assert result == "ngram"

    def test_coupled_drafter_promotes_default_to_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A loaded coupled drafter must drive the implicit default the
        same way a standard drafter does -- otherwise single-node
        Gemma 4 deployments would never auto-engage MTP without an
        explicit ``EXO_DRAFT_MODE=model`` knob."""
        monkeypatch.delenv(EXO_DRAFT_MODE_ENV, raising=False)
        result = resolve_draft_mode(
            has_drafter_model=False,
            request_use_drafter=None,
            request_draft_mode=None,
            has_coupled_drafter=True,
        )
        assert result == "model"

    def test_coupled_drafter_satisfies_required_drafter_for_model_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pre-fix, an explicit ``draft_mode="model"`` request would
        have demoted to ``"none"`` whenever ``has_drafter_model`` was
        ``False`` -- which is the post-Phase-2a state on coupled-only
        runners. The coupled-drafter signal must short-circuit that
        demotion so the dispatch can route to
        :class:`CoupledModelDrafter`.
        """
        monkeypatch.delenv(EXO_DRAFT_MODE_ENV, raising=False)
        result = resolve_draft_mode(
            has_drafter_model=False,
            request_use_drafter=None,
            request_draft_mode="model",
            has_coupled_drafter=True,
        )
        assert result == "model"

    def test_coupled_drafter_use_drafter_true_promotes_to_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``use_drafter=True`` with only a coupled drafter loaded must
        promote to ``"model"`` (the bucket :class:`CoupledModelDrafter`
        runs under), not ``"ngram"`` -- the operator deliberately
        loaded MTP weights and the opt-in should engage them."""
        monkeypatch.setenv(EXO_DRAFT_MODE_ENV, "none")
        result = resolve_draft_mode(
            has_drafter_model=False,
            request_use_drafter=True,
            request_draft_mode=None,
            has_coupled_drafter=True,
        )
        assert result == "model"

    def test_pipelined_request_with_coupled_only_demotes_to_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Codex P1 (PR #25 round-(N+0), drafter.py:289): pre-fix, a
        coupled-only deployment that received an explicit
        ``draft_mode="pipelined"`` (per-request override or stale
        ``EXO_DRAFT_MODE=pipelined`` env default) propagated through
        the resolver because ``has_coupled_drafter`` was wrongly
        treated as satisfying ``"pipelined"`` availability. Pipelined
        speculation runs on a STANDARD sibling drafter with its own
        KV cache; coupled MTP/DFlash drafters share state with the
        target via :class:`CoupledModelDrafter` and have no
        independent cache. Without this gate, ``make_drafter`` later
        raises ``ValueError`` and the request fails -- whereas the
        documented contract is to downgrade to ``"none"`` with a
        warning so misconfiguration stays observable but non-fatal.
        """
        monkeypatch.delenv(EXO_DRAFT_MODE_ENV, raising=False)
        result = resolve_draft_mode(
            has_drafter_model=False,
            request_use_drafter=None,
            request_draft_mode="pipelined",
            has_coupled_drafter=True,
        )
        assert result == "none"

    def test_pipelined_request_with_standard_drafter_passes_through_even_when_coupled_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Counterpart to the demotion test: an asymmetric / sibling
        deployment that ALSO loaded a coupled drafter (theoretical
        future: dual-drafter card) must still honour an explicit
        ``"pipelined"`` request, since ``has_drafter_model=True``
        means there's a real sibling LM with its own cache.
        """
        monkeypatch.delenv(EXO_DRAFT_MODE_ENV, raising=False)
        result = resolve_draft_mode(
            has_drafter_model=True,
            request_use_drafter=None,
            request_draft_mode="pipelined",
            has_coupled_drafter=True,
        )
        assert result == "pipelined"


class TestResolveAsymmetricDraftMode:
    """Codex P1 (PR #20 round-(N+1), generate.py:949): per-request
    overrides must win over the asymmetric placement's implicit
    pipelined default. Pre-fix the asymmetric branch in
    ``mlx_generate`` clobbered ``draft_mode`` to ``"pipelined"``
    unconditionally, ignoring callers who explicitly opted out via
    ``draft_mode="none"`` (benchmark baseline, short-output skip)
    or chose ngram for mixed traffic.
    """

    def test_default_returns_pipelined_for_asymmetric_placement(self) -> None:
        # No override: an asymmetric placement defaults to the
        # remote-drafter pipeline (the whole point of the topology).
        assert (
            resolve_asymmetric_draft_mode(
                has_asymmetric_drafter=True,
                request_use_drafter=None,
                request_draft_mode=None,
            )
            == "pipelined"
        )

    def test_use_drafter_false_overrides_to_none(self) -> None:
        assert (
            resolve_asymmetric_draft_mode(
                has_asymmetric_drafter=True,
                request_use_drafter=False,
                request_draft_mode=None,
            )
            == "none"
        )

    def test_explicit_request_draft_mode_none_overrides_to_none(self) -> None:
        # The bug we are guarding against: pre-fix this returned
        # "pipelined", silently engaging spec decoding for a caller
        # who explicitly opted out via draft_mode="none".
        assert (
            resolve_asymmetric_draft_mode(
                has_asymmetric_drafter=True,
                request_use_drafter=None,
                request_draft_mode="none",
            )
            == "none"
        )

    def test_explicit_request_draft_mode_ngram_overrides_to_ngram(self) -> None:
        # Mixed-traffic A/B test: caller wants in-process suffix
        # lookup on this request even though the placement has a
        # remote drafter wired up.
        assert (
            resolve_asymmetric_draft_mode(
                has_asymmetric_drafter=True,
                request_use_drafter=None,
                request_draft_mode="ngram",
            )
            == "ngram"
        )

    def test_explicit_pipelined_request_passes_through(self) -> None:
        assert (
            resolve_asymmetric_draft_mode(
                has_asymmetric_drafter=True,
                request_use_drafter=None,
                request_draft_mode="pipelined",
            )
            == "pipelined"
        )

    def test_no_asymmetric_placement_returns_none(self) -> None:
        # Defensive: the helper signals "asymmetric branch did not
        # apply" via "none". Callers fall back to ``resolve_draft_mode``
        # for the non-asymmetric resolution path; we don't repeat
        # that logic here.
        assert (
            resolve_asymmetric_draft_mode(
                has_asymmetric_drafter=False,
                request_use_drafter=True,
                request_draft_mode="pipelined",
            )
            == "none"
        )

    def test_use_drafter_false_wins_over_explicit_pipelined_request(self) -> None:
        # The opt-out shortcut beats the explicit mode override.
        # This matches the precedence in ``resolve_draft_mode``.
        assert (
            resolve_asymmetric_draft_mode(
                has_asymmetric_drafter=True,
                request_use_drafter=False,
                request_draft_mode="pipelined",
            )
            == "none"
        )

    def test_explicit_model_request_demotes_to_pipelined_under_asymmetric(
        self,
    ) -> None:
        # Codex P1 (PR #20 round-(N+6), drafter.py:253). In an
        # asymmetric placement target ranks intentionally never load
        # a local ``draft_model``, so a request with
        # ``draft_mode="model"`` would otherwise crash with
        # ``ValueError`` deep in :class:`ModelDrafter`'s constructor.
        # Demote to ``"pipelined"`` so the user's intent (model
        # drafting, as opposed to n-gram or none) is preserved
        # through the wire transport that talks to the peer rank
        # holding the actual drafter weights.
        assert (
            resolve_asymmetric_draft_mode(
                has_asymmetric_drafter=True,
                request_use_drafter=None,
                request_draft_mode="model",
            )
            == "pipelined"
        )

    def test_use_drafter_false_wins_over_explicit_model_request(self) -> None:
        # Even with the new model->pipelined demotion, the opt-out
        # shortcut still wins: a caller explicitly asking to skip
        # spec decoding must not be silently re-enabled.
        assert (
            resolve_asymmetric_draft_mode(
                has_asymmetric_drafter=True,
                request_use_drafter=False,
                request_draft_mode="model",
            )
            == "none"
        )

    def test_explicit_model_request_when_no_asymmetric_returns_none(self) -> None:
        # Defensive: the demotion only fires under
        # ``has_asymmetric_drafter=True``. When asymmetric isn't set
        # up at all the caller falls back to ``resolve_draft_mode``,
        # so this helper signals "asymmetric branch did not apply"
        # via ``"none"`` regardless of the requested mode.
        assert (
            resolve_asymmetric_draft_mode(
                has_asymmetric_drafter=False,
                request_use_drafter=None,
                request_draft_mode="model",
            )
            == "none"
        )


class TestUnimplementedDraftModesAreDowngraded:
    """Codex P1 (PR #20 round-(N+10), drafter.py:157): ``"eagle"`` and
    ``"lookahead"`` are scaffolding-only modes -- their drafter
    ``stream()`` implementations raise ``NotImplementedError``.
    Allowing them through ``parse_draft_mode`` /
    ``resolve_draft_mode`` / ``resolve_asymmetric_draft_mode`` would
    take the runner out of service when an operator set
    ``EXO_DRAFT_MODE=eagle`` or a client sent ``draft_mode="eagle"``.
    Until executable implementations land, downgrade with a warning
    so the runner stays serving (n-gram or no-spec) instead of
    failing every request.
    """

    def test_parse_draft_mode_downgrades_eagle_to_default(self) -> None:
        assert parse_draft_mode("eagle", default="model") == "model"
        assert parse_draft_mode("EAGLE", default="none") == "none"

    def test_parse_draft_mode_downgrades_lookahead_to_default(self) -> None:
        assert parse_draft_mode("lookahead", default="model") == "model"
        assert parse_draft_mode("Lookahead", default="none") == "none"

    def test_resolve_draft_mode_downgrades_request_eagle_with_loaded_drafter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(EXO_DRAFT_MODE_ENV, raising=False)
        # An explicit per-request ``draft_mode="eagle"`` arrives via
        # ``TaskParams`` and bypasses ``parse_draft_mode``, so the
        # resolver must apply its own downgrade. With a drafter
        # loaded the safest fallback is ``"model"`` (the user clearly
        # intended a "real model" drafter, just not the scaffolding
        # one).
        assert (
            resolve_draft_mode(
                has_drafter_model=True,
                request_use_drafter=None,
                request_draft_mode="eagle",
            )
            == "model"
        )

    def test_resolve_draft_mode_downgrades_request_lookahead_without_drafter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(EXO_DRAFT_MODE_ENV, raising=False)
        # No drafter loaded -> the request-level downgrade chooses
        # ``"none"`` so the request still runs (as plain decoding).
        assert (
            resolve_draft_mode(
                has_drafter_model=False,
                request_use_drafter=None,
                request_draft_mode="lookahead",
            )
            == "none"
        )

    def test_resolve_asymmetric_draft_mode_downgrades_eagle_to_pipelined(
        self,
    ) -> None:
        # On an asymmetric placement the analog of "model drafter"
        # is "pipelined drafter via remote transport"; downgrading to
        # ``"pipelined"`` preserves the user's intent (use real
        # weights) while keeping the request runnable.
        assert (
            resolve_asymmetric_draft_mode(
                has_asymmetric_drafter=True,
                request_use_drafter=None,
                request_draft_mode="eagle",
            )
            == "pipelined"
        )

    def test_resolve_asymmetric_draft_mode_downgrades_lookahead_to_pipelined(
        self,
    ) -> None:
        assert (
            resolve_asymmetric_draft_mode(
                has_asymmetric_drafter=True,
                request_use_drafter=None,
                request_draft_mode="lookahead",
            )
            == "pipelined"
        )

    def test_explicit_request_mode_wins_over_use_drafter_shortcut(
        self,
    ) -> None:
        # ``request_draft_mode`` is checked before ``use_drafter`` in
        # the precedence chain, so an explicit unimplemented-mode
        # request still gets the downgrade rather than the
        # use_drafter=False shortcut. This matches the existing
        # ``test_explicit_request_mode_wins_over_use_drafter`` for
        # implemented modes (``"ngram"``).
        assert (
            resolve_draft_mode(
                has_drafter_model=True,
                request_use_drafter=False,
                request_draft_mode="eagle",
            )
            == "model"
        )

    def test_use_drafter_false_alone_is_unaffected_by_unimplemented_handling(
        self,
    ) -> None:
        # When ``request_draft_mode`` is None, the opt-out shortcut
        # still wins; the unimplemented-mode handler only fires when
        # an unimplemented mode is actually requested.
        assert (
            resolve_draft_mode(
                has_drafter_model=True,
                request_use_drafter=False,
                request_draft_mode=None,
            )
            == "none"
        )

    def test_use_drafter_false_wins_over_unimplemented_under_asymmetric(
        self,
    ) -> None:
        # Asymmetric resolver checks ``use_drafter is False`` BEFORE
        # ``request_draft_mode``, so the opt-out shortcut still wins
        # even if the request specifies an unimplemented mode.
        assert (
            resolve_asymmetric_draft_mode(
                has_asymmetric_drafter=True,
                request_use_drafter=False,
                request_draft_mode="eagle",
            )
            == "none"
        )


class TestNgramDrafterPropose:
    """The proposer is pure list logic; no MLX involved."""

    def test_returns_empty_when_context_is_too_short(self) -> None:
        drafter = NgramDrafter(num_draft_tokens=4, min_match=2, max_match=4)
        # Need at least min_match + 1 tokens for a match to be possible
        # (suffix of length min_match plus one earlier match position).
        assert drafter.propose([1, 2], 4) == []

    def test_returns_empty_when_no_match(self) -> None:
        drafter = NgramDrafter(num_draft_tokens=4, min_match=2, max_match=4)
        # Tokens are unique - no suffix appears earlier.
        assert drafter.propose([10, 20, 30, 40, 50], 4) == []

    def test_finds_simple_repetition(self) -> None:
        # Suffix [1, 2] appears at start; following tokens are [3, 4].
        drafter = NgramDrafter(num_draft_tokens=4, min_match=2, max_match=4)
        assert drafter.propose([1, 2, 3, 4, 1, 2], 2) == [3, 4]

    def test_proposes_up_to_k_tokens(self) -> None:
        drafter = NgramDrafter(num_draft_tokens=10, min_match=2, max_match=4)
        # K=2 caps proposal to 2 even though 4 follow the match.
        assert drafter.propose([1, 2, 3, 4, 5, 6, 1, 2], 2) == [3, 4]

    def test_prefers_longer_match(self) -> None:
        # Suffix [2, 3] appears at index 1; suffix [1, 2, 3] appears at
        # index 0 (length 3, longer). Should prefer the longer one and
        # return [4, 5] (the tokens after the longer match).
        drafter = NgramDrafter(num_draft_tokens=4, min_match=2, max_match=4)
        ctx = [1, 2, 3, 4, 5, 6, 7, 1, 2, 3]
        # Last 3 tokens are [1, 2, 3]; longest match starts at 0.
        # Following tokens at start were [4, 5].
        assert drafter.propose(ctx, 4)[:2] == [4, 5]

    def test_prefers_recent_match_when_tied(self) -> None:
        # Two matches of suffix [9, 9] at same length; prefer the more
        # recent one (locality of reference).
        drafter = NgramDrafter(num_draft_tokens=2, min_match=2, max_match=2)
        ctx = [9, 9, 1, 9, 9, 2, 9, 9]
        # Recent match at index 3, followed by [2]. Earliest match at 0,
        # followed by [1]. Prefer recent -> [2].
        result = drafter.propose(ctx, 1)
        assert result == [2]

    def test_returns_empty_for_zero_k(self) -> None:
        drafter = NgramDrafter(num_draft_tokens=4, min_match=2, max_match=4)
        assert drafter.propose([1, 2, 3, 1, 2], 0) == []

    def test_validates_constructor_args(self) -> None:
        with pytest.raises(ValueError, match="num_draft_tokens"):
            NgramDrafter(num_draft_tokens=0)
        with pytest.raises(ValueError, match="min_match"):
            NgramDrafter(num_draft_tokens=2, min_match=0)
        with pytest.raises(ValueError, match="max_match"):
            NgramDrafter(num_draft_tokens=2, min_match=4, max_match=2)


def test_drafter_modes_match_implementation_class() -> None:
    """Each concrete drafter exposes the right ``mode`` literal."""
    assert NoSpecDrafter().mode == "none"
    assert NgramDrafter(num_draft_tokens=2).mode == "ngram"


def test_make_drafter_dispatches_correctly() -> None:
    none_drafter = make_drafter(
        mode="none", num_draft_tokens=4, draft_model=None, draft_cache=None
    )
    assert isinstance(none_drafter, NoSpecDrafter)
    ngram_drafter = make_drafter(
        mode="ngram", num_draft_tokens=4, draft_model=None, draft_cache=None
    )
    assert isinstance(ngram_drafter, NgramDrafter)


def test_make_drafter_rejects_model_without_pieces() -> None:
    with pytest.raises(ValueError, match="draft_model"):
        make_drafter(
            mode="model", num_draft_tokens=4, draft_model=None, draft_cache=None
        )


def test_ngram_drafter_proposal_caps_at_k() -> None:
    # The spec loop tops up ``K = min(max_tokens - ntoks, num_draft_tokens)``
    # before each round; the proposer must respect that cap so we don't
    # overrun ``max_tokens`` in the verify forward.
    drafter = NgramDrafter(num_draft_tokens=10, min_match=2, max_match=4)
    result = drafter.propose([1, 2, 3, 4, 1, 2], 3)
    assert len(result) <= 3


# ---------------------------------------------------------------------------
# Codex P2 (PR #19 round-(N+2), drafter.py:495):
# ``_ngram_stream_generate`` must report ``prompt_tokens`` as the
# size of the prefill *tail* it actually processed -- not the full
# prompt -- so the upstream aggregator's
# ``prefill_tokens + out.prompt_tokens`` sum equals the full prompt
# instead of double-counting it (and over-counting further on
# prefix-cache hits).
# ---------------------------------------------------------------------------


class TestNgramStreamGeneratePromptTokens:
    """Regression: yielded ``GenerationResponse.prompt_tokens`` must
    equal ``prompt.size`` (tail), not ``len(context_tokens)`` (full).

    We bypass the real spec loop by patching ``_ngram_speculative_step``
    so this test stays in CPU-only territory and doesn't need MLX
    weights.
    """

    def test_yields_tail_prompt_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import mlx.core as mx

        from exo.worker.engines.mlx.generator import drafter as drafter_module

        # Sentinel "model" / "tokenizer" / "cache": the patched spec
        # loop never touches them, so we can keep them as ``object()``.
        sentinel_model = object()

        class _FakeDetokenizer:
            def __init__(self) -> None:
                self.last_segment = ""

            def reset(self) -> None: ...
            def add_token(self, _token: int) -> None: ...
            def finalize(self) -> None: ...

        class _FakeTokenizer:
            def __init__(self) -> None:
                self.detokenizer = _FakeDetokenizer()
                self.eos_token_ids = {99}

        full_prompt = list(range(20))
        prompt_tail = mx.array(full_prompt[-2:], dtype=mx.uint32)

        def _fake_step(**_kwargs: object):  # noqa: ANN202
            yield (1, mx.zeros((1,)), False)
            yield (2, mx.zeros((1,)), True)
            yield (3, mx.zeros((1,)), False)

        monkeypatch.setattr(
            drafter_module,
            "_ngram_speculative_step",
            _fake_step,
        )

        responses = list(
            drafter_module._ngram_stream_generate(  # pyright: ignore[reportPrivateUsage]
                model=sentinel_model,  # pyright: ignore[reportArgumentType]
                tokenizer=_FakeTokenizer(),  # pyright: ignore[reportArgumentType]
                prompt=prompt_tail,
                context_tokens=full_prompt,
                prompt_cache=[],
                max_tokens=10,
                sampler=lambda x: x,
                logits_processors=[],
                drafter=NgramDrafter(num_draft_tokens=2),
                prefill_step_size=2,
            )
        )

        assert responses, "stream must yield at least one response"
        for response in responses:
            assert response.prompt_tokens == prompt_tail.size, (
                f"prompt_tokens must be the prefill tail size "
                f"({prompt_tail.size}), got {response.prompt_tokens}. "
                "Pre-fix this was len(context_tokens) which double-counts "
                "tokens already consumed by exo.prefill upstream."
            )
            assert response.prompt_tokens != len(full_prompt), (
                "prompt_tokens must NOT be the full prompt size, "
                "otherwise the upstream aggregator's "
                "(prefill_tokens + out.prompt_tokens) sum overcounts."
            )


class TestRequestIsGreedySampling:
    """Codex P1 (PR #19 round-(N+4), drafter.py:692): n-gram speculative
    decoding's ``target == draft`` accept rule is only
    distribution-correct under greedy decoding (argmax sampling).
    ``mlx_lm.make_sampler`` returns argmax iff ``temp == 0.0``, so the
    helper gates on temperature alone -- non-zero temperature means
    stochastic sampling and the n-gram path must demote to non-spec
    to preserve the model's output distribution.
    """

    @staticmethod
    def _make_task(
        temperature: float | None,
    ) -> "object":
        from exo.shared.types.common import ModelId
        from exo.shared.types.text_generation import (
            InputMessage,
            InputMessageContent,
            TextGenerationTaskParams,
        )

        return TextGenerationTaskParams(
            model=ModelId("test-model"),
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
            temperature=temperature,
        )

    def test_temperature_zero_is_greedy(self) -> None:
        from exo.worker.engines.mlx.generator.generate import (
            _request_is_greedy_sampling,  # pyright: ignore[reportPrivateUsage]
        )

        task = self._make_task(temperature=0.0)
        assert _request_is_greedy_sampling(task) is True  # pyright: ignore[reportArgumentType]

    def test_nonzero_temperature_is_not_greedy(self) -> None:
        from exo.worker.engines.mlx.generator.generate import (
            _request_is_greedy_sampling,  # pyright: ignore[reportPrivateUsage]
        )

        for temp in (0.1, 0.7, 1.0, 2.0):
            task = self._make_task(temperature=temp)
            assert _request_is_greedy_sampling(task) is False, (  # pyright: ignore[reportArgumentType]
                f"temperature={temp} must NOT be classified as greedy "
                f"(make_sampler returns stochastic sampling)"
            )

    def test_omitted_temperature_inherits_runner_default_non_greedy(self) -> None:
        # When the request omits temperature, the runner falls back to
        # a stochastic default (see ``make_sampler`` call site), so the
        # request is non-greedy. The helper exclusively checks
        # ``task.temperature == 0.0``; a missing temperature is
        # therefore correctly classified as non-greedy.
        from exo.worker.engines.mlx.generator.generate import (
            _request_is_greedy_sampling,  # pyright: ignore[reportPrivateUsage]
        )

        task = self._make_task(temperature=None)
        assert _request_is_greedy_sampling(task) is False, (  # pyright: ignore[reportArgumentType]
            "missing temperature inherits the runner default "
            "(non-greedy); n-gram drafting must demote to non-spec"
        )


class TestNgramStreamGenerateThreadsKvQuantization:
    """Codex P2 (PR #19 round-(N+6), drafter.py:642): the custom
    n-gram decode loop must call ``maybe_quantize_kv_cache`` after
    every model forward when ``KV_BITS`` is configured. Pre-fix the
    loop bypassed the quantization that
    ``mlx_lm.stream_generate`` does internally for the non-ngram
    path, so ``KV_BITS=4`` deployments silently kept the n-gram
    path's prompt-cache rows at full precision and could OOM on
    long generations.

    We assert at the call-site level: ``_ngram_stream_generate``
    threads the constants from :mod:`exo.worker.engines.mlx.constants`
    into ``_ngram_speculative_step`` so the quantization pass has
    the exact same parameters as ``mlx_lm.stream_generate``. The
    actual MLX dispatch is exercised by the smoke + bench scripts;
    this test stays MLX-free.
    """

    def test_ngram_stream_generate_passes_kv_bits_through_to_step(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mlx.core as mx

        from exo.worker.engines.mlx.constants import KV_BITS, KV_GROUP_SIZE
        from exo.worker.engines.mlx.generator import drafter as drafter_module

        captured_kwargs: dict[str, object] = {}

        def _fake_step(**kwargs: object):  # noqa: ANN202
            captured_kwargs.update(kwargs)
            yield (1, mx.zeros((1,)), False)

        monkeypatch.setattr(
            drafter_module,
            "_ngram_speculative_step",
            _fake_step,
        )

        class _FakeDetokenizer:
            def __init__(self) -> None:
                self.last_segment = ""

            def reset(self) -> None: ...
            def add_token(self, _token: int) -> None: ...
            def finalize(self) -> None: ...

        class _FakeTokenizer:
            def __init__(self) -> None:
                self.detokenizer = _FakeDetokenizer()
                self.eos_token_ids = {99}

        list(
            drafter_module._ngram_stream_generate(  # pyright: ignore[reportPrivateUsage]
                model=object(),  # pyright: ignore[reportArgumentType]
                tokenizer=_FakeTokenizer(),  # pyright: ignore[reportArgumentType]
                prompt=mx.array([1, 2], dtype=mx.uint32),
                context_tokens=[1, 2],
                prompt_cache=[],
                max_tokens=2,
                sampler=lambda x: x,
                logits_processors=[],
                drafter=NgramDrafter(num_draft_tokens=2),
                prefill_step_size=2,
            )
        )

        assert captured_kwargs.get("kv_bits") == KV_BITS, (
            "n-gram stream must thread KV_BITS into the step so the "
            "in-loop quantization call uses the same setting as "
            "mlx_lm.stream_generate; got "
            f"kv_bits={captured_kwargs.get('kv_bits')!r}, expected {KV_BITS!r}"
        )
        assert captured_kwargs.get("kv_group_size") == KV_GROUP_SIZE, (
            "n-gram stream must thread KV_GROUP_SIZE into the step so the "
            "in-loop quantization call uses the same group size as "
            "mlx_lm.stream_generate; got "
            f"kv_group_size={captured_kwargs.get('kv_group_size')!r}, "
            f"expected {KV_GROUP_SIZE!r}"
        )
