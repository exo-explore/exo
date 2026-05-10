"""Tests for ``drafter_id`` propagation in ``load_mlx_items``.

Codex P2 (PR #20 round-(N+10), utils_mlx.py:578): when an asymmetric
``DrafterPlacement`` exists (drafter weights live on a separate
node), ``load_mlx_items`` must surface the drafter model id from
placement so downstream telemetry can attribute requests to the
remote drafter even though no local weights are loaded. Pre-fix the
single-target asymmetric branch (``group is None`` AND
``drafter_placement is not None``) skipped the
``drafter_pair = _maybe_load_drafter(...)`` call and never copied
``drafter_placement.drafter_model_id`` into the returned tuple, so
``GenerationStats.drafter_model_id`` stayed ``None`` for every
single-target asymmetric request and dashboards lost attribution.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Generator
from typing import cast
from unittest.mock import MagicMock

import pytest

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.instances import (
    BoundInstance,
    DrafterPlacement,
    InstanceId,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import (
    RunnerId,
    ShardAssignments,
)
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    ShardMetadata,
)
from exo.worker.engines.mlx import utils_mlx


def _target_card(
    *,
    coupled_drafter: ModelId | None = None,
) -> ModelCard:
    return ModelCard(
        model_id=ModelId("mlx-community/test-target"),
        storage_size=Memory.from_gb(1.0),
        n_layers=12,
        hidden_size=768,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        drafter_model_ids=[ModelId("mlx-community/test-drafter")],
        coupled_drafter=coupled_drafter,
    )


def _make_single_target_bound_instance(
    drafter_placement: DrafterPlacement | None,
    *,
    coupled_drafter: ModelId | None = None,
) -> BoundInstance:
    target_node = NodeId()
    target_runner_id = RunnerId()
    shard = PipelineShardMetadata(
        model_card=_target_card(coupled_drafter=coupled_drafter),
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
        drafter_placement=drafter_placement,
    )
    return BoundInstance(
        instance=instance,
        bound_runner_id=target_runner_id,
        bound_node_id=target_node,
    )


_LoadResult = tuple[object, object, object, object, object, object]


def _consume_generator(
    gen: Generator[object, None, _LoadResult],
) -> _LoadResult:
    """Run a generator until it returns its tuple.

    ``load_mlx_items`` is a generator that yields progress and returns
    the final tuple via ``StopIteration.value``. This helper
    consumes all yields and returns the final value so tests can
    inspect it.
    """
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return cast(_LoadResult, stop.value)


def _patch_loader(
    monkeypatch: pytest.MonkeyPatch, *, drafter_resolves_to_path: bool = False
) -> None:
    """Stub out the heavy MLX call sites used by ``load_mlx_items``.

    Returns a fake (model, _) so we can drive ``load_mlx_items`` without
    a real model checkpoint or filesystem; the test only inspects the
    ``drafter_id`` field of the returned tuple, never the model.
    """

    fake_model = MagicMock(name="fake_target_model")
    fake_inner = MagicMock(name="fake_inner_model")
    fake_inner.layers = []
    fake_tokenizer = MagicMock(name="fake_tokenizer")

    def fake_load_model(
        _path: object, **_kwargs: object
    ) -> tuple[object, dict[str, object]]:
        return fake_model, {}

    def fake_get_inner(_model: object) -> object:
        return fake_inner

    def fake_get_layers(_inner: object) -> list[object]:
        return []

    def fake_get_tokenizer(_path: object, _shard: object) -> object:
        return fake_tokenizer

    def fake_set_wired_limit(_size: object) -> None:
        return None

    def fake_build_model_path(_model_id: object) -> str:
        return "/tmp/fake-model-path"

    def fake_resolve_existing(_model_id: object) -> object:
        # Pre-(N+10) tests verified that None was returned when
        # weights were absent. We return None here so the in-process
        # drafter load path stays inactive; the asymmetric branch
        # bypasses ``_maybe_load_drafter`` entirely.
        return "/tmp/fake-drafter" if drafter_resolves_to_path else None

    def fake_drafter_weight_size(_model_id: object) -> int:
        return 0

    monkeypatch.setattr(utils_mlx, "load_model", fake_load_model)
    monkeypatch.setattr(utils_mlx, "get_inner_model", fake_get_inner)
    monkeypatch.setattr(utils_mlx, "get_layers", fake_get_layers)
    monkeypatch.setattr(utils_mlx, "get_tokenizer", fake_get_tokenizer)
    monkeypatch.setattr(utils_mlx, "set_wired_limit_for_model", fake_set_wired_limit)
    monkeypatch.setattr(utils_mlx, "build_model_path", fake_build_model_path)
    monkeypatch.setattr(utils_mlx, "resolve_existing_model", fake_resolve_existing)
    monkeypatch.setattr(
        utils_mlx, "_drafter_weight_size_bytes", fake_drafter_weight_size
    )

    import mlx.core as mx_core

    def _noop_eval(*_args: object, **_kwargs: object) -> None:
        return None

    def _noop_clear_cache() -> None:
        return None

    monkeypatch.setattr(mx_core, "eval", _noop_eval)
    monkeypatch.setattr(mx_core, "clear_cache", _noop_clear_cache)


class TestSingleTargetAsymmetricDrafterIdPropagation:
    """``load_mlx_items`` must copy the asymmetric drafter id from
    placement when the local rank does not load drafter weights, both
    in the single-target (``group is None``) and multi-target
    (``group is not None``) asymmetric paths.
    """

    def test_single_target_asymmetric_propagates_drafter_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_loader(monkeypatch)
        drafter_placement = DrafterPlacement(
            drafter_node_id=NodeId(),
            drafter_runner_id=RunnerId(),
            drafter_model_id=ModelId("mlx-community/test-drafter"),
            drafter_rank=1,
            drafter_socket_host="169.254.0.10",
            drafter_socket_port=60001,
        )
        bound_instance = _make_single_target_bound_instance(drafter_placement)

        gen = cast(
            Generator[object, None, _LoadResult],
            utils_mlx.load_mlx_items(bound_instance, group=None),
        )
        result = _consume_generator(gen)
        (
            _model,
            _tokenizer,
            _vision,
            drafter_model,
            drafter_id,
            coupled_drafter,
        ) = result

        assert drafter_model is None, (
            "single-target asymmetric must NOT load drafter weights "
            "locally (the drafter rank is on a separate node); pre-"
            "fix this branch was already correct, the regression was "
            "in losing the drafter id."
        )
        assert coupled_drafter is None, (
            "single-target asymmetric placement is incompatible with "
            "coupled (mtp/dflash) drafters: their wire would have to "
            "ship full hidden states / KV cache cross-node. Phase 2 "
            "loader skips coupled-drafter entirely when drafter_placement "
            "is set."
        )
        assert drafter_id == ModelId("mlx-community/test-drafter"), (
            "Codex P2 (PR #20 round-(N+10), utils_mlx.py:578): "
            "single-target asymmetric MUST copy the drafter model id "
            "from placement so GenerationStats.drafter_model_id "
            "surfaces the remote drafter for telemetry; pre-fix it "
            "stayed None and dashboards lost attribution. "
            f"got drafter_id={drafter_id!r}"
        )

    def test_single_target_legacy_no_placement_keeps_local_loader(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression guard: when there is NO drafter placement, the
        single-target path still defers to ``_maybe_load_drafter`` for
        in-process drafting. This must not be perturbed by the
        N+10 fix.
        """
        _patch_loader(monkeypatch)
        called: dict[str, bool] = {"_maybe_load_drafter": False}

        def fake_maybe_load(_card: object) -> object:
            called["_maybe_load_drafter"] = True
            return None

        monkeypatch.setattr(utils_mlx, "_maybe_load_drafter", fake_maybe_load)
        bound_instance = _make_single_target_bound_instance(drafter_placement=None)
        gen = cast(
            Generator[object, None, _LoadResult],
            utils_mlx.load_mlx_items(bound_instance, group=None),
        )
        result = _consume_generator(gen)
        (
            _model,
            _tokenizer,
            _vision,
            drafter_model,
            drafter_id,
            coupled_drafter,
        ) = result

        assert called["_maybe_load_drafter"], (
            "single-target without placement must still try to load "
            "the in-process drafter; the N+10 fix only changes the "
            "asymmetric (placement is set) branch"
        )
        assert drafter_model is None
        assert drafter_id is None
        assert coupled_drafter is None, (
            "card has no coupled_drafter declared, so the new Phase 2 "
            "coupled-drafter path must stay inactive."
        )

    def test_asymmetric_placement_skips_coupled_drafter_even_when_card_declares_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Phase 3 placement gate: a card declaring ``coupled_drafter``
        must NOT activate the coupled path under asymmetric placement.

        Coupled drafters (MTP / DFlash) consume the target's hidden state
        and -- for MTP -- read the target's KV cache directly every
        round. Splitting target and coupled drafter across two nodes
        would force ``mlx_generate`` to ship full hidden tensors and
        per-layer-type KV snapshots over the wire every speculative
        round, which negates the speedup over any practical link
        (Thunderbolt RDMA included). The Phase 2 loader gate is the
        ``if bound_instance.instance.drafter_placement is None`` guard
        in ``load_mlx_items``: when ``DrafterPlacement`` is set the
        coupled load is skipped and the standard external drafter id
        is surfaced from placement instead. The Phase 3 ship of
        Gemma 4 cards (which now declare ``coupled_drafter``) must
        not unintentionally regress this for clusters that opt into
        asymmetric placement via ``drafter_eligible_nodes``.
        """
        _patch_loader(monkeypatch)
        called: dict[str, bool] = {"_try_load_coupled_drafter": False}

        def fake_try_coupled(_card: object) -> object:
            called["_try_load_coupled_drafter"] = True
            return MagicMock(name="should_never_be_seen")

        monkeypatch.setattr(utils_mlx, "_try_load_coupled_drafter", fake_try_coupled)

        drafter_placement = DrafterPlacement(
            drafter_node_id=NodeId(),
            drafter_runner_id=RunnerId(),
            drafter_model_id=ModelId("mlx-community/test-drafter"),
            drafter_rank=1,
            drafter_socket_host="169.254.0.10",
            drafter_socket_port=60001,
        )
        bound_instance = _make_single_target_bound_instance(
            drafter_placement,
            coupled_drafter=ModelId("mlx-community/test-coupled-drafter"),
        )
        gen = cast(
            Generator[object, None, _LoadResult],
            utils_mlx.load_mlx_items(bound_instance, group=None),
        )
        result = _consume_generator(gen)
        (_model, _tokenizer, _vision, _drafter_model, drafter_id, coupled_drafter) = (
            result
        )

        assert not called["_try_load_coupled_drafter"], (
            "asymmetric placement must short-circuit before "
            "_try_load_coupled_drafter; otherwise we'd materialise a "
            "coupled drafter that the spec-decode loop cannot use "
            "across nodes (its wire would have to ship hidden states / "
            "per-layer-type KV every round)."
        )
        assert coupled_drafter is None, (
            "asymmetric placement returns coupled_drafter=None even "
            "when the card declares one; the standard external drafter "
            "(reachable via DrafterPlacement) is the only spec-decode "
            "path under cross-node deployment."
        )
        assert drafter_id == ModelId("mlx-community/test-drafter"), (
            "asymmetric placement still surfaces the standard drafter "
            "id from placement so GenerationStats attributes the "
            "request correctly."
        )
