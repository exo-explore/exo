"""Tests for the optional `drafter_model_id` field on ModelCard.

The field declares a speculative-decoding draft model that runners may load
alongside the target. Coverage:
- ModelCard accepts and serialises the field.
- Cards with no drafter declared default to `None`.
- The Gemma 4 large-instruct cards point to the e2b drafter.
"""

import pytest

from exo.shared.models.model_cards import ModelCard, ModelId, get_model_cards
from exo.shared.types.memory import Memory


@pytest.mark.asyncio
async def test_drafter_model_id_defaults_to_none() -> None:
    cards = {card.model_id: card for card in await get_model_cards()}
    qwen_id = ModelId("mlx-community/Qwen3-30B-A3B-4bit")
    if qwen_id in cards:
        assert cards[qwen_id].drafter_model_id is None


@pytest.mark.asyncio
async def test_gemma4_31b_cards_declare_e2b_drafter() -> None:
    cards = {card.model_id: card for card in await get_model_cards()}
    expectations = {
        "mlx-community/gemma-4-31b-it-4bit": "mlx-community/gemma-4-e2b-it-4bit",
        "mlx-community/gemma-4-31b-it-6bit": "mlx-community/gemma-4-e2b-it-6bit",
        "mlx-community/gemma-4-31b-it-8bit": "mlx-community/gemma-4-e2b-it-8bit",
        "mlx-community/gemma-4-31b-it-bf16": "mlx-community/gemma-4-e2b-it-bf16",
    }
    for target_str, expected_drafter_str in expectations.items():
        target_id = ModelId(target_str)
        assert target_id in cards, f"{target_id} card missing"
        card = cards[target_id]
        assert card.drafter_model_id == ModelId(expected_drafter_str), (
            f"{target_id} drafter mismatch: got {card.drafter_model_id!r}"
        )


@pytest.mark.asyncio
async def test_gemma4_26b_cards_declare_e2b_drafter() -> None:
    cards = {card.model_id: card for card in await get_model_cards()}
    expectations = {
        "mlx-community/gemma-4-26b-a4b-it-4bit": "mlx-community/gemma-4-e2b-it-4bit",
        "mlx-community/gemma-4-26b-a4b-it-6bit": "mlx-community/gemma-4-e2b-it-6bit",
        "mlx-community/gemma-4-26b-a4b-it-8bit": "mlx-community/gemma-4-e2b-it-8bit",
        "mlx-community/gemma-4-26b-a4b-it-bf16": "mlx-community/gemma-4-e2b-it-bf16",
    }
    for target_str, expected_drafter_str in expectations.items():
        target_id = ModelId(target_str)
        assert target_id in cards, f"{target_id} card missing"
        card = cards[target_id]
        assert card.drafter_model_id == ModelId(expected_drafter_str), (
            f"{target_id} drafter mismatch: got {card.drafter_model_id!r}"
        )


def test_model_card_explicit_drafter_round_trip() -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/test-target"),
        storage_size=Memory.from_gb(1.0),
        n_layers=12,
        hidden_size=768,
        supports_tensor=True,
        tasks=["TextGeneration"],  # pyright: ignore[reportArgumentType]
        drafter_model_id=ModelId("mlx-community/test-drafter"),
    )
    assert card.drafter_model_id == ModelId("mlx-community/test-drafter")
    dump = card.model_dump(exclude_none=True)
    assert dump["drafter_model_id"] == "mlx-community/test-drafter"
