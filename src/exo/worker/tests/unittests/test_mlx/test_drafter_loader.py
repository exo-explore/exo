"""Tests for ``_maybe_load_drafter`` and the surrounding load path.

These tests exercise the policy-only branches of drafter loading so they can
run in CI without GPUs or downloaded model weights:

- Cards with no drafters return ``None``.
- Drafter weights missing from disk falls back to ``None`` (warned, not
  errored).
- ``EXO_DISABLE_DRAFTER`` short-circuits even when weights are present.
- ``EXO_DRAFTER_PREFERENCE`` picks the right drafter from the candidate list
  (fastest = head, highest_acceptance = tail), and on-disk drafters are
  preferred over not-yet-downloaded ones.

The "actually call ``mlx_lm.utils.load_model``" branch is exercised by the
end-to-end smoke harness, not unit tests.
"""

from pathlib import Path
from typing import cast

import pytest

from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.types.memory import Memory
from exo.worker.engines.mlx import utils_mlx
from exo.worker.engines.mlx.types import Model


def _card_with_drafters(drafter_ids: list[ModelId]) -> ModelCard:
    return ModelCard(
        model_id=ModelId("mlx-community/test-target"),
        storage_size=Memory.from_gb(1.0),
        n_layers=12,
        hidden_size=768,
        supports_tensor=True,
        tasks=["TextGeneration"],  # pyright: ignore[reportArgumentType]
        drafter_model_ids=drafter_ids,
    )


def test_maybe_load_drafter_returns_none_when_no_drafters_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, raising=False)
    card = _card_with_drafters([])

    def fail_resolve(*_args: object, **_kwargs: object) -> Path | None:
        raise AssertionError("resolve_existing_model should not be called")

    monkeypatch.setattr(utils_mlx, "resolve_existing_model", fail_resolve)

    assert utils_mlx._maybe_load_drafter(card) is None  # pyright: ignore[reportPrivateUsage]


def test_maybe_load_drafter_returns_none_when_drafter_weights_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, raising=False)
    monkeypatch.delenv(utils_mlx.EXO_DRAFTER_PREFERENCE_ENV, raising=False)
    card = _card_with_drafters([ModelId("mlx-community/missing-drafter")])

    def missing_resolve(_model_id: ModelId) -> Path | None:
        return None

    monkeypatch.setattr(utils_mlx, "resolve_existing_model", missing_resolve)

    def fail_load(*_args: object, **_kwargs: object) -> tuple[Model, dict[str, object]]:
        raise AssertionError("load_model must not run when weights are missing")

    monkeypatch.setattr(utils_mlx, "load_model", fail_load)

    assert utils_mlx._maybe_load_drafter(card) is None  # pyright: ignore[reportPrivateUsage]


def test_maybe_load_drafter_disabled_by_env_skips_filesystem_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, "1")
    card = _card_with_drafters([ModelId("mlx-community/some-drafter")])

    def fail_resolve(*_args: object, **_kwargs: object) -> Path | None:
        raise AssertionError("resolve_existing_model must not run when disabled")

    monkeypatch.setattr(utils_mlx, "resolve_existing_model", fail_resolve)

    assert utils_mlx._maybe_load_drafter(card) is None  # pyright: ignore[reportPrivateUsage]


def test_maybe_load_drafter_swallows_load_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A drafter present on disk that fails to load must not break the target."""
    monkeypatch.delenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, raising=False)
    monkeypatch.delenv(utils_mlx.EXO_DRAFTER_PREFERENCE_ENV, raising=False)
    card = _card_with_drafters([ModelId("mlx-community/broken-drafter")])

    def fixed_resolve(_model_id: ModelId) -> Path | None:
        return tmp_path

    monkeypatch.setattr(utils_mlx, "resolve_existing_model", fixed_resolve)

    def boom_load(*_args: object, **_kwargs: object) -> tuple[Model, dict[str, object]]:
        raise RuntimeError("simulated load failure")

    monkeypatch.setattr(utils_mlx, "load_model", boom_load)

    assert utils_mlx._maybe_load_drafter(card) is None  # pyright: ignore[reportPrivateUsage]


def test_maybe_load_drafter_returns_loaded_model_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, raising=False)
    monkeypatch.delenv(utils_mlx.EXO_DRAFTER_PREFERENCE_ENV, raising=False)
    card = _card_with_drafters([ModelId("mlx-community/fake-drafter")])

    def fixed_resolve(_model_id: ModelId) -> Path | None:
        return tmp_path

    monkeypatch.setattr(utils_mlx, "resolve_existing_model", fixed_resolve)

    sentinel = object()

    def fake_load(
        *_args: object, **_kwargs: object
    ) -> tuple[object, dict[str, object]]:
        return sentinel, {}

    def noop_eval(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(utils_mlx, "load_model", fake_load)
    monkeypatch.setattr(utils_mlx.mx, "eval", noop_eval)

    result = utils_mlx._maybe_load_drafter(card)  # pyright: ignore[reportPrivateUsage]
    assert result is not None
    drafter_id, drafter_model = result
    assert drafter_id == ModelId("mlx-community/fake-drafter")
    assert drafter_model is cast(Model, sentinel)


def test_select_drafter_id_default_is_fastest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When all candidates are on disk and preference is 'fastest' (default),
    return the head of the candidate list (smallest by convention)."""

    def resolve_all_on_disk(_model_id: ModelId) -> Path | None:
        return tmp_path

    monkeypatch.setattr(utils_mlx, "resolve_existing_model", resolve_all_on_disk)
    candidates = [
        ModelId("mlx-community/e2b-drafter"),
        ModelId("mlx-community/e4b-drafter"),
    ]
    chosen = utils_mlx._select_drafter_id(candidates, "fastest")  # pyright: ignore[reportPrivateUsage]
    assert chosen == ModelId("mlx-community/e2b-drafter")


def test_select_drafter_id_highest_acceptance_picks_tail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def resolve_all_on_disk(_model_id: ModelId) -> Path | None:
        return tmp_path

    monkeypatch.setattr(utils_mlx, "resolve_existing_model", resolve_all_on_disk)
    candidates = [
        ModelId("mlx-community/e2b-drafter"),
        ModelId("mlx-community/e4b-drafter"),
    ]
    chosen = utils_mlx._select_drafter_id(candidates, "highest_acceptance")  # pyright: ignore[reportPrivateUsage]
    assert chosen == ModelId("mlx-community/e4b-drafter")


def test_select_drafter_id_prefers_on_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the user prefers e4b but only e2b is on disk, fall back to e2b
    rather than logging a 'weights missing' warning the user didn't cause."""
    e2b = ModelId("mlx-community/e2b-drafter")
    e4b = ModelId("mlx-community/e4b-drafter")

    def resolve_only_e2b(model_id: ModelId) -> Path | None:
        return tmp_path if model_id == e2b else None

    monkeypatch.setattr(utils_mlx, "resolve_existing_model", resolve_only_e2b)
    chosen = utils_mlx._select_drafter_id([e2b, e4b], "highest_acceptance")  # pyright: ignore[reportPrivateUsage]
    assert chosen == e2b


def test_drafter_preference_unknown_value_falls_back_to_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(utils_mlx.EXO_DRAFTER_PREFERENCE_ENV, "totally-bogus")
    assert utils_mlx._drafter_preference() == "auto"  # pyright: ignore[reportPrivateUsage]
