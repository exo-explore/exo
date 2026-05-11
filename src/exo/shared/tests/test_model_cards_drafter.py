"""Tests for the optional `drafter_model_ids` field on ModelCard.

The field declares a preference-ordered list of speculative-decoding draft
models that runners may load alongside the target. Coverage:
- ModelCard accepts and serialises the field.
- Cards with no drafters declared default to an empty list.
- Gemma 4 large-instruct cards declare both e2b and e4b drafters at matching
  quantisation, in fastest-first order.

Also covers the asymmetric placement opt-in field
``drafter_eligible_nodes``: empty by default (legacy in-process drafter),
populated to designate per-deployment hosts for drafter-only ranks. The
field round-trips through Pydantic serialisation.
"""

from pathlib import Path

import pytest
from anyio import Path as AsyncPath

from exo.shared.models import model_cards
from exo.shared.models.model_cards import ModelCard, ModelId, get_model_cards
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory


@pytest.fixture(autouse=True)
def _isolate_custom_cards(  # pyright: ignore[reportUnusedFunction]
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Insulate these tests from operator-local custom card overrides.

    ``_custom_cards_dir`` resolves to ``$EXO_DATA_HOME/custom_model_cards``,
    which on dev workstations holds operator-edited cards (e.g. trimmed
    drafter lists for memory-constrained clusters). Those overrides are
    layered on top of the shipped TOML, so without isolation the assertions
    below describe whatever the operator last wrote, not the shipped data
    the gate is supposed to protect. Reset the in-memory cache too so the
    next test refreshes from the now-empty custom dir.
    """
    custom_dir = tmp_path / "custom_model_cards"
    custom_dir.mkdir()
    monkeypatch.setattr(model_cards, "_custom_cards_dir", AsyncPath(custom_dir))
    monkeypatch.setattr(model_cards, "_card_cache", {})


@pytest.mark.asyncio
async def test_drafter_model_ids_defaults_to_empty_list() -> None:
    cards = {card.model_id: card for card in await get_model_cards()}
    qwen_id = ModelId("mlx-community/Qwen3-30B-A3B-4bit")
    if qwen_id in cards:
        assert cards[qwen_id].drafter_model_ids == []


def _gemma4_31b_expectations() -> dict[str, list[str]]:
    return {
        "mlx-community/gemma-4-31b-it-4bit": [
            "mlx-community/gemma-4-e2b-it-4bit",
            "mlx-community/gemma-4-e4b-it-4bit",
        ],
        "mlx-community/gemma-4-31b-it-6bit": [
            "mlx-community/gemma-4-e2b-it-6bit",
            "mlx-community/gemma-4-e4b-it-6bit",
        ],
        "mlx-community/gemma-4-31b-it-8bit": [
            "mlx-community/gemma-4-e2b-it-8bit",
            "mlx-community/gemma-4-e4b-it-8bit",
        ],
        "mlx-community/gemma-4-31b-it-bf16": [
            "mlx-community/gemma-4-e2b-it-bf16",
            "mlx-community/gemma-4-e4b-it-bf16",
        ],
    }


def _gemma4_26b_expectations() -> dict[str, list[str]]:
    return {
        "mlx-community/gemma-4-26b-a4b-it-4bit": [
            "mlx-community/gemma-4-e2b-it-4bit",
            "mlx-community/gemma-4-e4b-it-4bit",
        ],
        "mlx-community/gemma-4-26b-a4b-it-6bit": [
            "mlx-community/gemma-4-e2b-it-6bit",
            "mlx-community/gemma-4-e4b-it-6bit",
        ],
        "mlx-community/gemma-4-26b-a4b-it-8bit": [
            "mlx-community/gemma-4-e2b-it-8bit",
            "mlx-community/gemma-4-e4b-it-8bit",
        ],
        "mlx-community/gemma-4-26b-a4b-it-bf16": [
            "mlx-community/gemma-4-e2b-it-bf16",
            "mlx-community/gemma-4-e4b-it-bf16",
        ],
    }


@pytest.mark.asyncio
async def test_gemma4_31b_cards_declare_e2b_then_e4b_drafters() -> None:
    cards = {card.model_id: card for card in await get_model_cards()}
    for target_str, expected_drafters in _gemma4_31b_expectations().items():
        target_id = ModelId(target_str)
        assert target_id in cards, f"{target_id} card missing"
        card = cards[target_id]
        assert card.drafter_model_ids == [ModelId(d) for d in expected_drafters], (
            f"{target_id} drafter mismatch: got {card.drafter_model_ids!r}"
        )


@pytest.mark.asyncio
async def test_gemma4_26b_cards_declare_e2b_then_e4b_drafters() -> None:
    cards = {card.model_id: card for card in await get_model_cards()}
    for target_str, expected_drafters in _gemma4_26b_expectations().items():
        target_id = ModelId(target_str)
        assert target_id in cards, f"{target_id} card missing"
        card = cards[target_id]
        assert card.drafter_model_ids == [ModelId(d) for d in expected_drafters], (
            f"{target_id} drafter mismatch: got {card.drafter_model_ids!r}"
        )


def test_model_card_explicit_drafters_round_trip() -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/test-target"),
        storage_size=Memory.from_gb(1.0),
        n_layers=12,
        hidden_size=768,
        supports_tensor=True,
        tasks=["TextGeneration"],  # pyright: ignore[reportArgumentType]
        drafter_model_ids=[
            ModelId("mlx-community/test-drafter-fast"),
            ModelId("mlx-community/test-drafter-accurate"),
        ],
    )
    assert card.drafter_model_ids == [
        ModelId("mlx-community/test-drafter-fast"),
        ModelId("mlx-community/test-drafter-accurate"),
    ]
    dump = card.model_dump(exclude_none=True)
    assert dump["drafter_model_ids"] == [
        "mlx-community/test-drafter-fast",
        "mlx-community/test-drafter-accurate",
    ]


def test_drafter_eligible_nodes_defaults_to_empty() -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/test-target-2"),
        storage_size=Memory.from_gb(1.0),
        n_layers=12,
        hidden_size=768,
        supports_tensor=True,
        tasks=["TextGeneration"],  # pyright: ignore[reportArgumentType]
    )
    assert card.drafter_eligible_nodes == []
    dump = card.model_dump(exclude_none=True)
    assert dump["drafter_eligible_nodes"] == []


def test_drafter_eligible_nodes_round_trip() -> None:
    eligible = [NodeId(), NodeId()]
    card = ModelCard(
        model_id=ModelId("mlx-community/test-target-3"),
        storage_size=Memory.from_gb(1.0),
        n_layers=12,
        hidden_size=768,
        supports_tensor=True,
        tasks=["TextGeneration"],  # pyright: ignore[reportArgumentType]
        drafter_model_ids=[ModelId("mlx-community/test-drafter")],
        drafter_eligible_nodes=eligible,
    )
    assert card.drafter_eligible_nodes == eligible
    dump = card.model_dump(exclude_none=True)
    assert dump["drafter_eligible_nodes"] == eligible
    rehydrated = ModelCard.model_validate(dump)
    assert rehydrated.drafter_eligible_nodes == eligible


def test_coupled_drafter_defaults_to_none() -> None:
    """Cards that don't declare a coupled drafter retain legacy behaviour.

    Phase-1 invariant: the field is purely additive. Existing cards that omit
    ``coupled_drafter`` must validate and serialise as if the field weren't
    there (``model_dump(exclude_none=True)`` drops the ``None`` so the TOML
    on disk stays untouched for the steady-state of cards that haven't been
    updated).
    """
    card = ModelCard(
        model_id=ModelId("mlx-community/test-target-no-coupled"),
        storage_size=Memory.from_gb(1.0),
        n_layers=12,
        hidden_size=768,
        supports_tensor=True,
        tasks=["TextGeneration"],  # pyright: ignore[reportArgumentType]
    )
    assert card.coupled_drafter is None
    dump = card.model_dump(exclude_none=True)
    assert "coupled_drafter" not in dump


def test_coupled_drafter_round_trip() -> None:
    """``coupled_drafter`` accepts a ModelId and round-trips through dump/validate.

    Drafter-kind resolution happens at *load* time (Phase 2) via
    ``mlx_vlm.speculative.drafters.resolve_drafter_kind`` reading the
    drafter's HF ``config.json``; the card stores only the model id so it
    stays decoupled from the mlx-vlm runtime API surface.
    """
    coupled = ModelId("mlx-community/gemma-4-E2B-it-assistant-bf16")
    card = ModelCard(
        model_id=ModelId("mlx-community/test-target-coupled"),
        storage_size=Memory.from_gb(1.0),
        n_layers=12,
        hidden_size=768,
        supports_tensor=True,
        tasks=["TextGeneration"],  # pyright: ignore[reportArgumentType]
        coupled_drafter=coupled,
    )
    assert card.coupled_drafter == coupled
    dump = card.model_dump(exclude_none=True)
    assert dump["coupled_drafter"] == coupled
    rehydrated = ModelCard.model_validate(dump)
    assert rehydrated.coupled_drafter == coupled


def test_coupled_drafter_composes_with_standard_drafter_list() -> None:
    """A card may declare both a coupled drafter AND a standard sibling list.

    The two fields are not mutually exclusive: placement chooses between them
    based on topology (asymmetric placement → standard list; single-node →
    coupled). The card schema must accept both side-by-side without
    validation error so a single Gemma 4 31B card can serve every deployment
    shape from "one Mac" to "asymmetric pipeline across a Thunderbolt RDMA
    cluster."
    """
    standard_list = [
        ModelId("mlx-community/gemma-4-e2b-it-4bit"),
        ModelId("mlx-community/gemma-4-e4b-it-4bit"),
    ]
    coupled = ModelId("mlx-community/gemma-4-E2B-it-assistant-bf16")
    card = ModelCard(
        model_id=ModelId("mlx-community/test-target-hybrid"),
        storage_size=Memory.from_gb(1.0),
        n_layers=12,
        hidden_size=768,
        supports_tensor=True,
        tasks=["TextGeneration"],  # pyright: ignore[reportArgumentType]
        drafter_model_ids=standard_list,
        coupled_drafter=coupled,
    )
    assert card.drafter_model_ids == standard_list
    assert card.coupled_drafter == coupled
    dump = card.model_dump(exclude_none=True)
    assert dump["drafter_model_ids"] == standard_list
    assert dump["coupled_drafter"] == coupled
    rehydrated = ModelCard.model_validate(dump)
    assert rehydrated.drafter_model_ids == standard_list
    assert rehydrated.coupled_drafter == coupled


_GEMMA4_31B_MTP_DRAFTER = ModelId("mlx-community/gemma-4-31B-it-assistant-bf16")
_GEMMA4_26B_A4B_MTP_DRAFTER = ModelId("mlx-community/gemma-4-26B-A4B-it-assistant-bf16")


@pytest.mark.asyncio
async def test_shipped_gemma4_cards_declare_mtp_coupled_drafter() -> None:
    """All shipped Gemma 4 31B / 26B-A4B cards declare a target-matched MTP drafter.

    Phase-3 contract: every shipped Gemma 4 large-target quant ships a
    ``coupled_drafter`` pointed at the bf16 MTP assistant trained
    against THAT target's hidden size. mlx-community publishes one
    assistant per target family:

    - ``gemma-4-31b-it-*`` →  ``gemma-4-31B-it-assistant-bf16`` (~0.5B)
    - ``gemma-4-26b-a4b-it-*`` →  ``gemma-4-26B-A4B-it-assistant-bf16`` (~0.4B)

    The ``E2B`` / ``E4B`` assistants exist but are sized for the
    ``gemma-4-e2b`` / ``gemma-4-e4b`` targets respectively; pairing them
    with a 26B-A4B or 31B target raises a matmul shape mismatch in
    ``mlx_vlm.speculative.drafters.gemma4_assistant.draft_block`` because
    the drafter's pre-projection head is sized to the trained-against
    target's ``hidden_size``. Pinning the target-matched assistant per
    quant variant locks that pairing in.

    The assistants are published only as bf16: at ~80 MB - 0.5 GB they
    cost no memory pressure, and quant noise on the drafter materially
    hurts acceptance rate (which is what drives the speedup).
    """
    cards = {card.model_id: card for card in await get_model_cards()}

    expected_drafter_per_family: dict[ModelId, set[str]] = {
        _GEMMA4_31B_MTP_DRAFTER: set(_gemma4_31b_expectations()),
        _GEMMA4_26B_A4B_MTP_DRAFTER: set(_gemma4_26b_expectations()),
    }
    for expected_drafter, target_strs in expected_drafter_per_family.items():
        for target_str in target_strs:
            target_id = ModelId(target_str)
            assert target_id in cards, f"{target_id} card missing"
            card = cards[target_id]
            assert card.coupled_drafter == expected_drafter, (
                f"{target_id} coupled_drafter mismatch: got "
                f"{card.coupled_drafter!r}, expected {expected_drafter!r}"
            )


@pytest.mark.asyncio
async def test_shipped_gemma4_cards_keep_standard_drafter_list_alongside_mtp() -> None:
    """Phase-3 cards keep ``drafter_model_ids`` populated next to ``coupled_drafter``.

    The two drafter paths are complementary, not exclusive:
    - On a single-node placement (``drafter_placement is None``) the
      worker tries the coupled MTP drafter first (``utils_mlx.load_mlx_items``).
    - On an asymmetric placement (``drafter_placement is not None``,
      driven by populated ``drafter_eligible_nodes``) the coupled path is
      bypassed and the standard external drafter list is used because
      coupled drafters can't ship hidden states / KV across the wire
      cheaply. Removing ``drafter_model_ids`` would silently disable
      drafting for every cluster that has ``drafter_eligible_nodes``
      populated -- a mode regression we want to prevent at the card
      level.

    This test pins both lists side-by-side so a future "simplification"
    PR doesn't drop the standard drafters under the assumption that
    MTP supersedes them.
    """
    cards = {card.model_id: card for card in await get_model_cards()}

    paired_expectations: list[tuple[dict[str, list[str]], ModelId]] = [
        (_gemma4_31b_expectations(), _GEMMA4_31B_MTP_DRAFTER),
        (_gemma4_26b_expectations(), _GEMMA4_26B_A4B_MTP_DRAFTER),
    ]
    for expectations, expected_drafter in paired_expectations:
        for target_str, expected_drafters in expectations.items():
            target_id = ModelId(target_str)
            assert target_id in cards, f"{target_id} card missing"
            card = cards[target_id]
            assert card.coupled_drafter == expected_drafter
            assert card.drafter_model_ids == [ModelId(d) for d in expected_drafters], (
                f"{target_id} drafter_model_ids mismatch: got "
                f"{card.drafter_model_ids!r}"
            )
