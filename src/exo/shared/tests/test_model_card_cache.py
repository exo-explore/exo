"""The model card cache never hands out a partly read catalog."""

import anyio
import pytest

from exo.shared.models.model_cards import ModelCard, ModelId, card_cache


async def test_list_all_while_the_cache_is_being_read_returns_every_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty: dict[ModelId, ModelCard] = {}
    monkeypatch.setattr(card_cache, "cc", empty)
    every_card = len(await card_cache.list_all())
    assert every_card > 20

    reads = 0
    load_from_path = ModelCard.load_from_path

    async def counting_load(path: anyio.Path) -> ModelCard:
        nonlocal reads
        reads += 1
        return await load_from_path(path)

    monkeypatch.setattr(ModelCard, "load_from_path", staticmethod(counting_load))
    monkeypatch.setattr(card_cache, "cc", {})
    async with anyio.create_task_group() as tg:
        # The first caller starts reading the cards (as the downloader does at startup)
        tg.start_soon(card_cache.list_all)
        with anyio.fail_after(10):
            while reads < 10:
                await anyio.sleep(0)
        # A second caller (the dashboard's /models) asks while it's part way through
        assert len(await card_cache.list_all()) == every_card

    assert len(await card_cache.list_all()) == every_card
