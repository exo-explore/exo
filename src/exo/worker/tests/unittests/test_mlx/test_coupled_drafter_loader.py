"""Tests for ``_try_load_coupled_drafter`` and Phase 2a foundation behavior.

Phase 2a (loader-only) ships the plumbing for MTP/DFlash coupled drafters
without yet routing them through the generator. These tests lock in the
loader contract so that the Phase 2b follow-up (which adds the
``rollback_speculative_cache`` + extended forward kwargs to the mlx-lm
fork's gemma4_text and the round-loop dispatch in ``mlx_generate``) can
swap in the actual MTP path without re-relitigating the policy bits:

- A card with no ``coupled_drafter`` gets ``None`` without touching mlx-vlm.
- A card with ``coupled_drafter`` set but ``EXO_DISABLE_DRAFTER`` honored
  short-circuits before any filesystem or import work.
- Missing weights on disk surface a warning and degrade to ``None`` (so
  the standard external-drafter list can take over).
- Unrecognised drafter kinds reported by mlx-vlm degrade to ``None``
  rather than returning a model the generator can't drive.
- The success path returns a ``CoupledDrafter`` with a ``Literal``-typed
  ``kind`` and the loaded model object.
- Wired-memory budget for cards declaring both ``coupled_drafter`` and
  ``drafter_model_ids`` covers the larger of the two so the runtime
  fallback path is never under-wired.

These tests deliberately do NOT exercise generator-side dispatch -- that
path doesn't exist yet in Phase 2a. Generator dispatch tests land with
Phase 2b and verify the ``bind`` / ``set_shared_kv`` / ``draft_block``
round loop end-to-end.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.memory import Memory
from exo.worker.engines.mlx import utils_mlx


def _card(*, coupled_id: ModelId | None, standard_ids: list[ModelId]) -> ModelCard:
    return ModelCard(
        model_id=ModelId("mlx-community/test-target"),
        storage_size=Memory.from_gb(1.0),
        n_layers=12,
        hidden_size=768,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        drafter_model_ids=standard_ids,
        coupled_drafter=coupled_id,
    )


def _stub_mlx_vlm_drafters(
    monkeypatch: pytest.MonkeyPatch,
    *,
    load_drafter_returns: tuple[object, str] | None = None,
    load_drafter_raises: Exception | None = None,
    known_kinds: tuple[str, ...] = ("mtp", "dflash"),
) -> MagicMock:
    """Install a fake ``mlx_vlm.speculative.drafters`` module.

    The real module imports MLX kernels and would crash on a CPU-only test
    runner; the loader only depends on ``load_drafter`` and
    ``KNOWN_DRAFTER_KINDS`` from it, so we stub those two attributes.
    """

    fake_load = MagicMock(name="load_drafter")
    if load_drafter_raises is not None:
        fake_load.side_effect = load_drafter_raises
    else:
        fake_load.return_value = load_drafter_returns or (
            MagicMock(name="fake_drafter_model"),
            "mtp",
        )

    fake_speculative = types.ModuleType("mlx_vlm.speculative")
    fake_drafters = types.ModuleType("mlx_vlm.speculative.drafters")
    fake_drafters.load_drafter = fake_load  # type: ignore[attr-defined]
    fake_drafters.KNOWN_DRAFTER_KINDS = frozenset(known_kinds)  # type: ignore[attr-defined]
    fake_speculative.drafters = fake_drafters  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlx_vlm.speculative", fake_speculative)
    monkeypatch.setitem(sys.modules, "mlx_vlm.speculative.drafters", fake_drafters)
    return fake_load


def test_no_coupled_drafter_declared_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, raising=False)
    card = _card(coupled_id=None, standard_ids=[])

    def fail_resolve(*_args: object, **_kwargs: object) -> Path | None:
        raise AssertionError(
            "resolve_existing_model must not be called when no "
            "coupled_drafter is declared on the card"
        )

    monkeypatch.setattr(utils_mlx, "resolve_existing_model", fail_resolve)
    assert utils_mlx._try_load_coupled_drafter(card) is None


def test_disabled_by_env_short_circuits_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, "1")
    card = _card(coupled_id=ModelId("mlx-community/coupled"), standard_ids=[])

    def fail_resolve(*_args: object, **_kwargs: object) -> Path | None:
        raise AssertionError(
            "EXO_DISABLE_DRAFTER must be checked before any filesystem "
            "or mlx-vlm import work"
        )

    monkeypatch.setattr(utils_mlx, "resolve_existing_model", fail_resolve)
    assert utils_mlx._try_load_coupled_drafter(card) is None


def _resolve_to(path: Path | None) -> object:
    """Build a ``resolve_existing_model`` stub returning ``path`` for any id.

    Wrapping a plain function (rather than ``lambda``) keeps basedpyright
    happy without sprinkling pyright-ignore comments through every test.
    """

    def _stub(_model_id: ModelId) -> Path | None:
        return path

    return _stub


def test_missing_weights_returns_none_without_calling_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, raising=False)
    card = _card(coupled_id=ModelId("mlx-community/missing"), standard_ids=[])

    monkeypatch.setattr(utils_mlx, "resolve_existing_model", _resolve_to(None))
    fake_load = _stub_mlx_vlm_drafters(monkeypatch)

    assert utils_mlx._try_load_coupled_drafter(card) is None
    assert fake_load.call_count == 0, (
        "load_drafter must not run when the drafter weights are absent"
    )


def test_load_drafter_failure_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A coupled drafter present on disk that fails to load via mlx-vlm must
    degrade to ``None`` so the caller can fall back to the standard
    external drafter list (or to plain decoding)."""
    monkeypatch.delenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, raising=False)
    card = _card(coupled_id=ModelId("mlx-community/broken"), standard_ids=[])

    monkeypatch.setattr(utils_mlx, "resolve_existing_model", _resolve_to(tmp_path))
    _stub_mlx_vlm_drafters(
        monkeypatch, load_drafter_raises=RuntimeError("simulated mlx-vlm failure")
    )

    assert utils_mlx._try_load_coupled_drafter(card) is None


def test_partial_mlxvlm_install_falls_back_without_attribute_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex P2 (PR #23 round-(N+0), utils_mlx.py:809): a partial / drifted
    ``mlx-vlm`` install where ``mlx_vlm.speculative.drafters`` imports
    cleanly but is missing ``load_drafter`` / ``KNOWN_DRAFTER_KINDS``
    must degrade to the standard drafter path -- not raise
    ``AttributeError`` and abort the runner.

    Reproduces the failure mode where ``except ImportError`` alone is
    insufficient: the import itself succeeds, but the symbol resolution
    (or ``cast()`` site that touches the attribute) blows up.
    """
    monkeypatch.delenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, raising=False)
    card = _card(coupled_id=ModelId("mlx-community/coupled"), standard_ids=[])

    fake_speculative = types.ModuleType("mlx_vlm.speculative")
    # Module imports successfully but the drafters submodule is empty
    # -- e.g. an old mlx-vlm release that namespaces ``speculative``
    # without having shipped the drafter API yet, or a future release
    # that renames the symbols. Either way, we must not crash the
    # caller; we must degrade.
    fake_drafters = types.ModuleType("mlx_vlm.speculative.drafters")
    fake_speculative.drafters = fake_drafters  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlx_vlm.speculative", fake_speculative)
    monkeypatch.setitem(sys.modules, "mlx_vlm.speculative.drafters", fake_drafters)
    monkeypatch.setattr(
        utils_mlx, "resolve_existing_model", _resolve_to(Path("/tmp/should-not-matter"))
    )

    assert utils_mlx._try_load_coupled_drafter(card) is None


def test_unknown_kind_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """mlx-vlm may evolve to recognise drafter kinds exo's loader cannot
    drive. We must refuse rather than return a model the generator
    cannot dispatch."""
    monkeypatch.delenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, raising=False)
    card = _card(coupled_id=ModelId("mlx-community/future-kind"), standard_ids=[])

    monkeypatch.setattr(utils_mlx, "resolve_existing_model", _resolve_to(tmp_path))
    _stub_mlx_vlm_drafters(
        monkeypatch,
        load_drafter_returns=(
            MagicMock(name="future_kind_model"),
            "speculative_eagle_v3",
        ),
        known_kinds=("mtp", "dflash", "speculative_eagle_v3"),
    )

    assert utils_mlx._try_load_coupled_drafter(card) is None


def test_success_returns_coupled_drafter_with_literal_kind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, raising=False)
    coupled_id = ModelId("mlx-community/gemma-4-E2B-it-assistant-bf16")
    card = _card(coupled_id=coupled_id, standard_ids=[])

    monkeypatch.setattr(utils_mlx, "resolve_existing_model", _resolve_to(tmp_path))
    sentinel = MagicMock(name="loaded_drafter")
    fake_load = _stub_mlx_vlm_drafters(
        monkeypatch, load_drafter_returns=(sentinel, "mtp")
    )

    result = utils_mlx._try_load_coupled_drafter(card)
    assert result is not None, "successful load must return a CoupledDrafter"
    assert result.model_id == coupled_id
    assert result.kind == "mtp"
    assert result.model is sentinel
    assert fake_load.call_count == 1
    fake_load.assert_called_once_with(str(tmp_path), kind=None)


def test_dflash_kind_is_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``dflash`` is the second supported coupled-drafter kind (Qwen3
    family). Phase 2a's loader must accept it even though Phase 2b
    initially focuses on MTP for Gemma 4."""
    monkeypatch.delenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, raising=False)
    card = _card(
        coupled_id=ModelId("mlx-community/qwen3-dflash-drafter"), standard_ids=[]
    )
    monkeypatch.setattr(utils_mlx, "resolve_existing_model", _resolve_to(tmp_path))
    _stub_mlx_vlm_drafters(
        monkeypatch,
        load_drafter_returns=(MagicMock(name="dflash_model"), "dflash"),
    )

    result = utils_mlx._try_load_coupled_drafter(card)
    assert result is not None
    assert result.kind == "dflash"


def test_wired_budget_uses_max_when_card_has_both_drafter_kinds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A card that declares both a coupled drafter and a standard list
    can fall back to the standard one at runtime if the coupled load
    fails (mlx-vlm missing, weights absent). The wired-memory limit is
    set ONCE before any drafter loads, so it must cover the larger of
    the two on-disk sizes -- otherwise the standard drafter would be
    pageable across requests."""
    monkeypatch.delenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, raising=False)
    monkeypatch.delenv(utils_mlx.EXO_DRAFTER_PREFERENCE_ENV, raising=False)

    coupled_id = ModelId("mlx-community/coupled-tiny")
    standard_id = ModelId("mlx-community/standard-large")

    sizes: dict[ModelId, int] = {coupled_id: 158_000_000, standard_id: 3_000_000_000}

    def fake_size(model_id: ModelId) -> int:
        return sizes.get(model_id, 0)

    monkeypatch.setattr(utils_mlx, "_coupled_drafter_weight_size_bytes", fake_size)
    monkeypatch.setattr(utils_mlx, "_drafter_weight_size_bytes", fake_size)

    captured: dict[str, Memory] = {}

    def capture_limit(size: Memory) -> None:
        captured["size"] = size

    def fake_build_path(_id: ModelId) -> str:
        return "/tmp/fake"

    def fake_load_model(
        *_args: object, **_kwargs: object
    ) -> tuple[object, dict[str, object]]:
        return MagicMock(), {}

    def fake_inner(_m: object) -> object:
        return MagicMock(layers=[])

    def fake_layers(_m: object) -> list[object]:
        return []

    def fake_tokenizer(*_args: object) -> object:
        return MagicMock()

    def returns_none(_card: ModelCard) -> None:
        return None

    monkeypatch.setattr(utils_mlx, "set_wired_limit_for_model", capture_limit)
    monkeypatch.setattr(utils_mlx, "build_model_path", fake_build_path)
    monkeypatch.setattr(utils_mlx, "load_model", fake_load_model)
    monkeypatch.setattr(utils_mlx, "get_inner_model", fake_inner)
    monkeypatch.setattr(utils_mlx, "get_layers", fake_layers)
    monkeypatch.setattr(utils_mlx, "get_tokenizer", fake_tokenizer)
    monkeypatch.setattr(utils_mlx, "_try_load_coupled_drafter", returns_none)
    monkeypatch.setattr(utils_mlx, "_maybe_load_drafter", returns_none)
    import mlx.core as mx_core

    def noop(*_args: object, **_kwargs: object) -> None:
        return None

    def noop_clear() -> None:
        return None

    monkeypatch.setattr(mx_core, "eval", noop)
    monkeypatch.setattr(mx_core, "clear_cache", noop_clear)

    target_card = ModelCard(
        model_id=ModelId("mlx-community/test-target"),
        storage_size=Memory.from_gb(2.0),
        n_layers=12,
        hidden_size=768,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        drafter_model_ids=[standard_id],
        coupled_drafter=coupled_id,
    )

    from exo.shared.types.common import NodeId
    from exo.shared.types.worker.instances import (
        BoundInstance,
        InstanceId,
        MlxRingInstance,
    )
    from exo.shared.types.worker.runners import RunnerId, ShardAssignments
    from exo.shared.types.worker.shards import (
        PipelineShardMetadata,
        ShardMetadata,
    )

    target_node = NodeId()
    target_runner_id = RunnerId()
    shard = PipelineShardMetadata(
        model_card=target_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=12,
        n_layers=12,
    )
    instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=ModelId("mlx-community/test-target"),
            runner_to_shard={target_runner_id: cast(ShardMetadata, shard)},
            node_to_runner={target_node: target_runner_id},
        ),
        hosts_by_node={target_node: []},
        ephemeral_port=60000,
        drafter_placement=None,
    )
    bound_instance = BoundInstance(
        instance=instance,
        bound_runner_id=target_runner_id,
        bound_node_id=target_node,
    )

    list(utils_mlx.load_mlx_items(bound_instance, group=None))

    assert "size" in captured, "set_wired_limit_for_model must be called once"
    target_bytes = target_card.storage_size.in_bytes
    expected_bytes = target_bytes + sizes[standard_id]
    assert captured["size"].in_bytes == expected_bytes, (
        f"wired budget must cover the LARGER of the two drafter sizes "
        f"(target={target_bytes}B + max_drafter={sizes[standard_id]}B), "
        f"got {captured['size'].in_bytes}B. Otherwise a runtime fallback "
        f"from coupled (smaller) to standard (larger) under-wires the "
        f"weights and the OS pages them out between requests."
    )
