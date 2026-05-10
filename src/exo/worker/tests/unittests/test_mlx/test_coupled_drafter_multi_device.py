"""Multi-device (tensor-parallel) coupled-drafter loader + dispatch tests.

Locks in the contract that lifts the historical ``group is None`` gate
around coupled-drafter loading and dispatch:

- :func:`utils_mlx._try_load_collocated_drafter` honours
  ``allow_standard_drafter_fallback`` so multi-device callers don't waste
  memory loading a standard drafter ``mlx_generate`` can't dispatch.
- A successful coupled load on the multi-device path still attaches the
  target-side hooks (the capability gate :func:`mlx_generate` reads to
  decide whether to route the request through the coupled path).
- Hook-attachment failures still fall through to the no-drafter outcome
  -- never crash the multi-device load.

End-to-end TP execution requires a real multi-process MLX group and is
exercised by the operator-side ``bench/`` harness on the
``wc-smbp + wc-smbpt`` two-node setup. The dispatch-shape coverage in
:mod:`test_coupled_drafter_dispatch` already validates the generator's
single-process round loop; this module adds the loader + gate seam
that decides whether that round loop *gets to run* on a TP placement.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import pytest

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.instances import DrafterPlacement
from exo.shared.types.worker.runners import RunnerId
from exo.worker.engines.mlx import utils_mlx
from exo.worker.engines.mlx.utils_mlx import CoupledDrafter


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
    load_drafter_returns: tuple[object, str],
    known_kinds: tuple[str, ...] = ("mtp", "dflash"),
) -> MagicMock:
    fake_load = MagicMock(name="load_drafter")
    fake_load.return_value = load_drafter_returns
    fake_speculative = types.ModuleType("mlx_vlm.speculative")
    fake_drafters = types.ModuleType("mlx_vlm.speculative.drafters")
    fake_drafters.load_drafter = fake_load  # type: ignore[attr-defined]
    fake_drafters.KNOWN_DRAFTER_KINDS = frozenset(known_kinds)  # type: ignore[attr-defined]
    fake_speculative.drafters = fake_drafters  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_vlm.speculative", fake_speculative)
    monkeypatch.setitem(sys.modules, "mlx_vlm.speculative.drafters", fake_drafters)
    return fake_load


def _resolve_to(path: Path | None) -> object:
    def _stub(_model_id: ModelId) -> Path | None:
        return path

    return _stub


def test_multi_device_loads_coupled_drafter_when_card_declares_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Coupled drafter loading must run on tensor-parallel placements.

    The legacy ``group is None`` gate (see ``utils_mlx.py`` git history)
    silently skipped coupled-drafter loading for any TP runner, so the
    drafter was downloaded but never wired -- ``GenerationStats`` came
    back with ``drafter_model_id=None`` and the 4-x DFlash speedup went
    unrealised. This test pins the lifted gate by exercising the
    helper directly with ``allow_standard_drafter_fallback=False`` (the
    multi-device caller's flag).
    """
    monkeypatch.delenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, raising=False)
    coupled_id = ModelId("z-lab/Qwen3.5-122B-A10B-DFlash")
    card = _card(coupled_id=coupled_id, standard_ids=[])
    monkeypatch.setattr(utils_mlx, "resolve_existing_model", _resolve_to(tmp_path))
    sentinel_drafter = MagicMock(name="dflash_drafter_model")
    _stub_mlx_vlm_drafters(
        monkeypatch, load_drafter_returns=(sentinel_drafter, "dflash")
    )
    attached: list[tuple[str, object]] = []

    def fake_dispatch_attach(kind: str, model: object) -> None:
        attached.append((kind, model))

    monkeypatch.setattr(
        utils_mlx, "_dispatch_attach_coupled_hooks", fake_dispatch_attach
    )

    fake_model = nn.Module()
    coupled, drafter_id, drafter_model = utils_mlx._try_load_collocated_drafter(
        card, fake_model, allow_standard_drafter_fallback=False
    )

    assert coupled is not None, "multi-device must still load the coupled drafter"
    assert coupled.model_id == coupled_id
    assert coupled.kind == "dflash"
    assert coupled.model is sentinel_drafter
    assert drafter_id is None
    assert drafter_model is None
    assert attached == [("dflash", fake_model)], (
        "the capability-gate hook attachment must run on multi-device "
        "too -- the generator's coupled dispatch reads this sentinel"
    )


def test_multi_device_skips_standard_drafter_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-device runners must NOT load a standard drafter as fallback.

    ``mlx_generate`` declines to dispatch standard / n-gram drafters on
    multi-device placements today (``draft_mode='none'`` when
    ``coupled_drafter_eligible`` is False). Loading the standard
    drafter anyway would waste tens of GB of unified memory on the
    122B-A10B class. The flag is the contract.
    """
    monkeypatch.delenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, raising=False)
    coupled_id = ModelId("z-lab/some-coupled-drafter")
    standard_id = ModelId("mlx-community/some-standard-drafter")
    card = _card(coupled_id=coupled_id, standard_ids=[standard_id])

    monkeypatch.setattr(utils_mlx, "resolve_existing_model", _resolve_to(None))

    def fail_maybe_load(_card: ModelCard) -> object:
        raise AssertionError(
            "_maybe_load_drafter must not run when "
            "allow_standard_drafter_fallback=False -- multi-device "
            "callers can't dispatch a standard drafter, so loading "
            "one wastes memory"
        )

    monkeypatch.setattr(utils_mlx, "_maybe_load_drafter", fail_maybe_load)

    fake_model = nn.Module()
    coupled, drafter_id, drafter_model = utils_mlx._try_load_collocated_drafter(
        card, fake_model, allow_standard_drafter_fallback=False
    )

    assert coupled is None
    assert drafter_id is None
    assert drafter_model is None


def test_single_device_uses_standard_drafter_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Single-device runners keep the historical standard-drafter fallback.

    When the coupled load fails (e.g. weights absent) and the card also
    declares ``drafter_model_ids``, the helper falls through to
    :func:`_maybe_load_drafter` so the request still benefits from
    standard spec decoding. Multi-device opts out via the flag; this
    test pins the single-device default-True branch.
    """
    monkeypatch.delenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, raising=False)
    monkeypatch.delenv(utils_mlx.EXO_DRAFTER_PREFERENCE_ENV, raising=False)
    coupled_id = ModelId("mlx-community/some-coupled")
    standard_id = ModelId("mlx-community/some-standard")
    card = _card(coupled_id=coupled_id, standard_ids=[standard_id])

    monkeypatch.setattr(utils_mlx, "resolve_existing_model", _resolve_to(None))
    sentinel_standard_model = MagicMock(name="standard_drafter_model")

    def fake_maybe_load(_card: ModelCard) -> tuple[ModelId, object] | None:
        return standard_id, sentinel_standard_model

    monkeypatch.setattr(utils_mlx, "_maybe_load_drafter", fake_maybe_load)

    fake_model = nn.Module()
    coupled, drafter_id, drafter_model = utils_mlx._try_load_collocated_drafter(
        card, fake_model, allow_standard_drafter_fallback=True
    )

    assert coupled is None
    assert drafter_id == standard_id
    assert drafter_model is sentinel_standard_model


def test_coupled_hook_attachment_failure_falls_through_to_none_on_multi_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Hook-attachment failure must NOT crash the multi-device load.

    A card might mis-pair a coupled drafter with an incompatible target
    (Gemma 4 MTP card on a Qwen target after a card rewrite, or a
    drafter loaded against a sharded model whose vendor hooks haven't
    learned the wrapper shape yet). The historic behaviour on
    single-device is to log, discard the coupled drafter, and degrade
    to standard drafting. Multi-device degrades to no-drafter (because
    the standard fallback is gated off) -- it must never raise.
    """
    monkeypatch.delenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, raising=False)
    coupled_id = ModelId("mlx-community/coupled-but-wrong-target")
    card = _card(coupled_id=coupled_id, standard_ids=[])
    monkeypatch.setattr(utils_mlx, "resolve_existing_model", _resolve_to(tmp_path))
    _stub_mlx_vlm_drafters(
        monkeypatch, load_drafter_returns=(MagicMock(name="x"), "dflash")
    )

    def boom(_kind: str, _model: object) -> None:
        raise TypeError("wrong target architecture (simulated)")

    monkeypatch.setattr(utils_mlx, "_dispatch_attach_coupled_hooks", boom)

    fake_model = nn.Module()
    coupled, drafter_id, drafter_model = utils_mlx._try_load_collocated_drafter(
        card, fake_model, allow_standard_drafter_fallback=False
    )

    assert coupled is None
    assert drafter_id is None
    assert drafter_model is None


# --------------------------------------------------------------------------- #
# Generator-side gate
# --------------------------------------------------------------------------- #


def test_coupled_drafter_eligible_no_longer_gates_on_group_is_none() -> None:
    """Pin the lifted ``group is None`` gate on ``coupled_drafter_eligible``.

    The dispatch in :func:`mlx_generate` previously refused to mark a
    coupled drafter eligible whenever ``group is not None``, defeating
    the loader's multi-device coupled load. This test inspects the
    source so a future "tidy-up the conditional" pass can't accidentally
    re-add the gate without surfacing in CI. We avoid spinning up a
    real distributed group (CI is single-process) and instead lock the
    surrounding text so the intent is durable.
    """
    import inspect

    from exo.worker.engines.mlx.generator import generate as _generate

    source = inspect.getsource(_generate)
    eligible_lines = [
        line for line in source.splitlines() if "coupled_drafter_eligible: bool" in line
    ]
    assert len(eligible_lines) == 1, (
        "coupled_drafter_eligible must be declared exactly once; "
        f"found {len(eligible_lines)} declarations -- update the test "
        "to match the new surface if this is intentional."
    )
    declaration_block = source.split("coupled_drafter_eligible: bool")[1].split(")", 1)[
        0
    ]
    assert "group is None" not in declaration_block, (
        "coupled_drafter_eligible must not gate on ``group is None`` -- "
        "multi-device (tensor-parallel) placements now drive coupled "
        "drafters per-rank against the post-all-reduce hidden state. "
        "If you're intentionally re-introducing the gate, update this "
        "test along with the bench harness so the regression is loud."
    )


def test_multi_device_draft_mode_routing_keeps_coupled_path_open() -> None:
    """``draft_mode`` must not be hard-forced to ``"none"`` on TP runs.

    The legacy gate

        elif group is not None:
            draft_mode = "none"

    short-circuited every multi-device request to non-spec decoding
    even when a coupled drafter was loaded. The lifted gate narrows on
    ``not coupled_drafter_eligible`` so coupled drafters drive the TP
    path through :func:`resolve_draft_mode` like single-device, while
    standard drafters still degrade to ``"none"`` on multi-device.
    """
    import inspect

    from exo.worker.engines.mlx.generator import generate as _generate

    source = inspect.getsource(_generate)
    # Match the structural pattern, not exact whitespace, so cosmetic
    # reformatting (black, ruff format) doesn't break the assertion.
    assert "group is not None and not coupled_drafter_eligible" in source, (
        "the multi-device draft_mode='none' gate must AND in "
        "``not coupled_drafter_eligible`` so coupled drafters keep "
        "driving the TP path through resolve_draft_mode."
    )


def test_builder_force_sequential_includes_coupled_dispatchable() -> None:
    """``drafting_can_run_here`` must include coupled-drafter dispatchable.

    The builder picks ``SequentialGenerator`` over ``BatchGenerator``
    when ``drafting_can_run_here AND drafter_loaded_will_run`` (and a
    few other clauses). On multi-device, ``is_single_device`` is False
    but a coupled drafter is still dispatchable, so we OR in
    ``coupled_drafter_dispatchable`` -- otherwise the multi-device
    runner loads the coupled drafter, lifts the dispatch gate, and
    then loses the speedup to BatchGenerator's no-spec-decoding code
    path.
    """
    import inspect

    from exo.worker.engines.mlx import builder as _builder

    source = inspect.getsource(_builder)
    assert (
        "drafting_can_run_here = is_single_device or coupled_drafter_dispatchable"
        in source
    ), (
        "drafting_can_run_here must OR ``coupled_drafter_dispatchable`` so "
        "TP runners with a coupled drafter take the SequentialGenerator "
        "path. BatchGenerator has no spec-decoding hook."
    )


# --------------------------------------------------------------------------- #
# Behaviour assertions that don't require an actual MLX distributed group
# --------------------------------------------------------------------------- #


def test_multi_device_wired_bump_includes_coupled_drafter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TP wired-memory limit must reserve the full coupled-drafter size.

    Pre-fix the wired-memory bump was gated on ``group is None``, so a
    multi-device runner that lifted the loader / dispatch gates would
    load a coupled drafter into wired pool sized for the target shard
    alone. Under macOS' wired-memory policy the OS is then free to
    page the drafter out between requests -- exactly when speculative
    decoding's per-round latency is what makes the coupled path
    worthwhile. The helper must bump by the full coupled-drafter on-
    disk size for any TP placement that will load one.

    Distinct from the standard-drafter case below: TP runs pass
    ``allow_standard_drafter_fallback=False`` to the loader, so the
    standard drafter size is intentionally excluded from the bump to
    keep the wired pool minimal on already-memory-tight TP ranks.
    """
    coupled_id = ModelId("z-lab/Qwen3.5-122B-A10B-DFlash")
    standard_id = ModelId("mlx-community/some-standard-drafter")
    card = _card(coupled_id=coupled_id, standard_ids=[standard_id])

    sizes: dict[ModelId, int] = {coupled_id: 3_000_000_000, standard_id: 5_000_000_000}

    def fake_coupled_size(model_id: ModelId) -> int:
        return sizes[model_id]

    def fake_standard_size(model_id: ModelId) -> int:
        return sizes[model_id]

    monkeypatch.setattr(
        utils_mlx, "_coupled_drafter_weight_size_bytes", fake_coupled_size
    )
    monkeypatch.setattr(utils_mlx, "_drafter_weight_size_bytes", fake_standard_size)
    monkeypatch.delenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, raising=False)

    # Multi-device: ``group`` is a sentinel, ``drafter_placement`` is
    # None (symmetric TP). Helper must reserve the COUPLED size and
    # ignore the (larger) standard one.
    fake_group = cast(mx.distributed.Group, MagicMock(name="mlx_distributed_group"))
    bump_tp = utils_mlx._collocated_drafter_wired_bytes(
        target_card=card,
        group=fake_group,
        drafter_placement=None,
    )
    assert bump_tp.in_bytes == sizes[coupled_id], (
        f"TP wired bump must equal the coupled-drafter size "
        f"({sizes[coupled_id]} bytes); got {bump_tp.in_bytes} bytes. "
        "Including the larger standard-drafter size here would over-"
        "wire the TP rank; excluding the coupled size paged the "
        "drafter out under load."
    )

    # Single-device: the legacy max-of-both rule survives because the
    # standard-drafter fallback can still fire if the coupled load fails.
    bump_single = utils_mlx._collocated_drafter_wired_bytes(
        target_card=card,
        group=None,
        drafter_placement=None,
    )
    assert bump_single.in_bytes == sizes[standard_id], (
        f"single-device wired bump must reserve max(coupled, standard) "
        f"({sizes[standard_id]} bytes); got {bump_single.in_bytes} bytes"
    )


def test_wired_bump_skipped_for_asymmetric_drafter_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asymmetric remote drafters live on a different node, so this rank
    must not reserve any wired bytes for them.

    Pre-existing behaviour pinned here so a future refactor that
    centralises the wired-bump logic can't accidentally drop this
    guard. Without it, an asymmetric placement would over-reserve
    wired memory for a drafter whose weights never enter this rank's
    address space, starving the target's KV cache.
    """
    coupled_id = ModelId("z-lab/some-coupled")
    standard_id = ModelId("mlx-community/some-standard")
    card = _card(coupled_id=coupled_id, standard_ids=[standard_id])

    def fake_size_two_gb(_id: ModelId) -> int:
        return 2_000_000_000

    monkeypatch.setattr(
        utils_mlx, "_coupled_drafter_weight_size_bytes", fake_size_two_gb
    )
    monkeypatch.setattr(utils_mlx, "_drafter_weight_size_bytes", fake_size_two_gb)
    monkeypatch.delenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, raising=False)

    asymmetric_placement = DrafterPlacement(
        drafter_node_id=NodeId(),
        drafter_runner_id=RunnerId(),
        drafter_model_id=standard_id,
        drafter_rank=1,
        drafter_socket_host="127.0.0.1",
        drafter_socket_port=60001,
    )
    bump = utils_mlx._collocated_drafter_wired_bytes(
        target_card=card,
        group=None,
        drafter_placement=asymmetric_placement,
    )
    assert bump.in_bytes == 0, (
        f"asymmetric drafter placement must contribute 0 wired bytes; "
        f"got {bump.in_bytes}. The drafter weights live on a different "
        "node and never enter this rank's address space."
    )


def test_wired_bump_skipped_when_drafter_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``EXO_DISABLE_DRAFTER=1`` short-circuits the loader before any
    drafter weights enter memory, so the wired-bump helper must also
    return zero. Otherwise a user disabling drafting via env still
    pays the wired-pool reservation."""
    coupled_id = ModelId("z-lab/some-coupled")
    card = _card(coupled_id=coupled_id, standard_ids=[])

    def fake_size_one_point_five_gb(_id: ModelId) -> int:
        return 1_500_000_000

    monkeypatch.setattr(
        utils_mlx,
        "_coupled_drafter_weight_size_bytes",
        fake_size_one_point_five_gb,
    )
    monkeypatch.setenv(utils_mlx.EXO_DISABLE_DRAFTER_ENV, "1")

    fake_group = cast(mx.distributed.Group, MagicMock(name="group"))
    bump = utils_mlx._collocated_drafter_wired_bytes(
        target_card=card,
        group=fake_group,
        drafter_placement=None,
    )
    assert bump.in_bytes == 0


def test_coupled_drafter_kind_is_literal_friendly() -> None:
    """Sanity: the loaded ``CoupledDrafter`` exposes a kind the generator
    can match.

    Defensive guard against the legacy "load returns the drafter but
    ``kind`` is ``Any``" failure mode -- if the loader ever loses its
    ``Literal[...]`` narrowing, generator dispatch will silently fall
    through the ``coupled_drafter.kind == "mtp"`` branch instead of
    routing to the DFlash adapter.
    """
    drafter = CoupledDrafter(
        model_id=ModelId("z-lab/Qwen3.5-122B-A10B-DFlash"),
        kind="dflash",
        model=cast(object, MagicMock(name="drafter_model")),
    )
    assert drafter.kind == "dflash"
    assert isinstance(drafter.kind, str)
