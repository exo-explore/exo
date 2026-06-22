"""Tests for the native-MTP loader branch in ``utils_mlx.load_mlx_items``.

The native-MTP path was added so cards declaring
:class:`exo.shared.models.model_cards.NativeMTPConfig` get loaded
through :func:`exo.worker.engines.mlx.vendor.qwen3_5_mtp_loader.load_mtp_model`
instead of the stock ``mlx_lm.utils.load_model``.

The tests use ``monkeypatch`` to stub the heavy MLX call sites so we can
drive the loader without a real checkpoint or filesystem. Coverage:

- Card declares ``native_mtp`` AND placement is single-node AND probe
  says "recoverable": ``load_mtp_model`` is called, stock ``load_model``
  is NOT.
- Card has no ``native_mtp``: stock ``load_model`` is called.
- Card declares ``native_mtp`` but the probe returns "stripped" (e.g.
  mlx-community-style stripped quants): we degrade to the stock loader.

We do NOT exercise the real Qwen3.5/3.6 MTP loader here -- the vendor
module has its own parity tests. The goal is to verify routing.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import cast

import pytest

from exo.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
    NativeMTPConfig,
)
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.instances import (
    BoundInstance,
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
from exo.worker.engines.mlx import mtp_probe, utils_mlx


def _target_card(*, native_mtp: NativeMTPConfig | None = None) -> ModelCard:
    return ModelCard(
        model_id=ModelId("Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality"),
        storage_size=Memory.from_gb(1.0),
        n_layers=64,
        hidden_size=5120,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
        native_mtp=native_mtp,
    )


def _make_single_target_bound_instance(card: ModelCard) -> BoundInstance:
    target_node = NodeId()
    target_runner_id = RunnerId()
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=64,
        n_layers=64,
    )
    instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            runner_to_shard={target_runner_id: cast(ShardMetadata, shard)},
            node_to_runner={target_node: target_runner_id},
        ),
        hosts_by_node={target_node: []},
        ephemeral_port=60000,
    )
    return BoundInstance(
        instance=instance,
        bound_runner_id=target_runner_id,
        bound_node_id=target_node,
    )


_LoadResult = tuple[object, object, object]


def _consume_generator(
    gen: Generator[object, None, _LoadResult],
) -> _LoadResult:
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return cast(_LoadResult, stop.value)


def _recoverable_probe() -> mtp_probe.MtpProbeResult:
    return mtp_probe.MtpProbeResult(
        model_declares_mtp=True,
        mtp_tensors_found=True,
        mtp_format=mtp_probe.MtpFormat.MTPLX_SEPARATE_FILE,
        mtp_count=29,
        mtp_path="/tmp/fake/mtp.safetensors",
        mtp_tensor_keys=(),
    )


def _stripped_probe() -> mtp_probe.MtpProbeResult:
    return mtp_probe.MtpProbeResult(
        model_declares_mtp=True,
        mtp_tensors_found=False,
        mtp_format=mtp_probe.MtpFormat.STRIPPED,
        mtp_count=0,
        mtp_path=None,
        mtp_tensor_keys=(),
    )


def _patch_loader_routing(
    monkeypatch: pytest.MonkeyPatch, *, probe_recoverable: bool
) -> dict[str, bool]:
    """Stub the heavy load call sites and return a flag dict recording
    which loader fired."""
    calls = {"stock_load_model": False, "native_load_mtp_model": False}

    fake_model: object = object()
    fake_inner: object = object()
    fake_tokenizer: object = object()

    def fake_stock_load(
        _path: object, **_kwargs: object
    ) -> tuple[object, dict[str, object]]:
        calls["stock_load_model"] = True
        return fake_model, {}

    monkeypatch.setattr(utils_mlx, "load_model", fake_stock_load)

    # The native loader is imported lazily inside ``load_mlx_items``; patch
    # it on the source module so the lazy import sees our stub.
    from exo.worker.engines.mlx.vendor import qwen3_5_mtp_loader as _mtp_loader_mod

    def fake_native_load(
        _path: object, **_kwargs: object
    ) -> tuple[object, dict[str, object]]:
        calls["native_load_mtp_model"] = True
        return fake_model, {}

    monkeypatch.setattr(_mtp_loader_mod, "load_mtp_model", fake_native_load)

    def fake_probe(_path: object) -> mtp_probe.MtpProbeResult:
        return _recoverable_probe() if probe_recoverable else _stripped_probe()

    monkeypatch.setattr(mtp_probe, "probe_mtp_weights", fake_probe)

    def fake_inner_model(_model: object) -> object:
        return fake_inner

    def fake_layers(_inner: object) -> list[object]:
        return []

    def fake_get_tokenizer(_path: object, _shard: object) -> object:
        return fake_tokenizer

    def fake_set_wired_limit(_size: object) -> None:
        return None

    def fake_build_model_path(_model_id: object) -> Path:
        return Path("/tmp/fake-model-path")

    monkeypatch.setattr(utils_mlx, "get_inner_model", fake_inner_model)
    monkeypatch.setattr(utils_mlx, "get_layers", fake_layers)
    monkeypatch.setattr(utils_mlx, "get_tokenizer", fake_get_tokenizer)
    monkeypatch.setattr(utils_mlx, "set_wired_limit_for_model", fake_set_wired_limit)
    monkeypatch.setattr(utils_mlx, "build_model_path", fake_build_model_path)

    import mlx.core as mx_core

    def fake_eval(*_args: object, **_kwargs: object) -> None:
        return None

    def fake_clear_cache() -> None:
        return None

    monkeypatch.setattr(mx_core, "eval", fake_eval)
    monkeypatch.setattr(mx_core, "clear_cache", fake_clear_cache)

    return calls


class TestNativeMTPLoaderRouting:
    """``load_mlx_items`` dispatches the native MTP loader iff the card
    declares ``native_mtp``, the probe says recoverable, and the
    placement is single-node.
    """

    def test_native_path_when_card_declares_and_single_node(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _patch_loader_routing(monkeypatch, probe_recoverable=True)
        card = _target_card(native_mtp=NativeMTPConfig(num_layers=1))
        bound = _make_single_target_bound_instance(card)
        _consume_generator(
            cast(
                Generator[object, None, _LoadResult],
                utils_mlx.load_mlx_items(bound, group=None),
            )
        )
        assert calls["native_load_mtp_model"] is True, (
            "card declares native_mtp and probe says recoverable: the "
            "loader MUST dispatch through load_mtp_model"
        )
        assert calls["stock_load_model"] is False, (
            "native loader handles weight loading end-to-end; stock "
            "load_model must not be called when the native path fires"
        )

    def test_stock_path_when_card_lacks_native_mtp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _patch_loader_routing(monkeypatch, probe_recoverable=True)
        card = _target_card(native_mtp=None)
        bound = _make_single_target_bound_instance(card)
        _consume_generator(
            cast(
                Generator[object, None, _LoadResult],
                utils_mlx.load_mlx_items(bound, group=None),
            )
        )
        assert calls["stock_load_model"] is True
        assert calls["native_load_mtp_model"] is False, (
            "the native MTP loader must never fire for cards that don't opt in"
        )

    def test_stock_fallback_when_probe_says_stripped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Card declares native_mtp but probe returns STRIPPED → stock path
        (silent degrade so the request still completes)."""
        calls = _patch_loader_routing(monkeypatch, probe_recoverable=False)
        card = _target_card(native_mtp=NativeMTPConfig(num_layers=1))
        bound = _make_single_target_bound_instance(card)
        _consume_generator(
            cast(
                Generator[object, None, _LoadResult],
                utils_mlx.load_mlx_items(bound, group=None),
            )
        )
        assert calls["stock_load_model"] is True, (
            "probe says STRIPPED → fall back to stock loader so the "
            "request still completes (silent degrade, not a hard fail)"
        )
        assert calls["native_load_mtp_model"] is False


class TestNativeMtpLoaderEligible:
    """``_native_mtp_loader_eligible`` is the loader-side gate. It is only
    called from the single-node (``group is None``) branch."""

    def test_true_for_declared_card_with_recoverable_probe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_probe(_path: object) -> mtp_probe.MtpProbeResult:
            return _recoverable_probe()

        monkeypatch.setattr(mtp_probe, "probe_mtp_weights", fake_probe)
        card = _target_card(native_mtp=NativeMTPConfig(num_layers=1))
        assert utils_mlx._native_mtp_loader_eligible(card, Path("/tmp/fake")) is True

    def test_false_without_native_mtp_skips_probe(self) -> None:
        """Cards that don't declare native_mtp short-circuit before the
        probe (which would touch disk)."""
        card = _target_card(native_mtp=None)
        assert (
            utils_mlx._native_mtp_loader_eligible(
                card, Path("/nonexistent/never-probed")
            )
            is False
        )

    def test_false_when_probe_says_stripped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_probe(_path: object) -> mtp_probe.MtpProbeResult:
            return _stripped_probe()

        monkeypatch.setattr(mtp_probe, "probe_mtp_weights", fake_probe)
        card = _target_card(native_mtp=NativeMTPConfig(num_layers=1))
        assert utils_mlx._native_mtp_loader_eligible(card, Path("/tmp/fake")) is False
