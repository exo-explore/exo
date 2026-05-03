# pyright: reportPrivateUsage=false

import pytest

import exo.worker.main as worker_main
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.worker.main import Worker


def _model_card(model_id: ModelId) -> ModelCard:
    return ModelCard(
        model_id=model_id,
        n_layers=1,
        storage_size=Memory.from_bytes(1),
        hidden_size=1,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )


def _worker() -> Worker:
    worker = object.__new__(Worker)
    worker.state = State()
    worker._synced_custom_cards = {}
    return worker


@pytest.mark.asyncio
async def test_worker_syncs_custom_cards_from_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved: list[ModelId] = []
    cached: list[ModelId] = []

    async def save_to_custom_dir(card: ModelCard) -> None:
        saved.append(card.model_id)

    def add_to_card_cache(card: ModelCard) -> None:
        cached.append(card.model_id)

    monkeypatch.setattr(ModelCard, "save_to_custom_dir", save_to_custom_dir)
    monkeypatch.setattr(worker_main, "add_to_card_cache", add_to_card_cache)

    card = _model_card(ModelId("custom/model"))
    worker = _worker()
    worker.state = State(custom_model_cards={card.model_id: card})

    await worker._sync_custom_cards_from_state()
    await worker._sync_custom_cards_from_state()

    assert saved == [card.model_id]
    assert cached == [card.model_id]
    assert worker._synced_custom_cards == {card.model_id: card}


@pytest.mark.asyncio
async def test_worker_deletes_custom_cards_missing_from_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deleted: list[ModelId] = []

    async def delete_custom_card(model_id: ModelId) -> bool:
        deleted.append(model_id)
        return True

    monkeypatch.setattr(worker_main, "delete_custom_card", delete_custom_card)

    card = _model_card(ModelId("custom/model"))
    worker = _worker()
    worker._synced_custom_cards = {card.model_id: card}

    await worker._sync_custom_cards_from_state()

    assert deleted == [card.model_id]
    assert worker._synced_custom_cards == {}
