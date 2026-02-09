from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from _pytest.monkeypatch import MonkeyPatch

import exo.download.download_utils as download_utils
import exo.shared.models.model_cards as model_cards
from exo.shared.models.model_cards import (
    ModelCard,
    ModelId,
    fetch_config_data,
    fetch_safetensors_size,
)
from exo.shared.types.memory import Memory


@pytest.mark.asyncio
async def test_fetch_config_data_raises_helpful_error_when_missing(
    monkeypatch: MonkeyPatch,
) -> None:
    async def _raise_not_found(*_args: object, **_kwargs: object) -> Path:
        raise FileNotFoundError("missing")

    monkeypatch.setattr(download_utils, "download_file_with_retry", _raise_not_found)

    with pytest.raises(ValueError, match=r"does not contain config\.json"):
        await fetch_config_data(ModelId("fake/model"))


@pytest.mark.asyncio
async def test_fetch_safetensors_size_falls_back_when_index_missing(
    monkeypatch: MonkeyPatch,
) -> None:
    async def _raise_not_found(*_args: object, **_kwargs: object) -> Path:
        raise FileNotFoundError("missing")

    monkeypatch.setattr(download_utils, "download_file_with_retry", _raise_not_found)

    class _FakeSafetensors:
        total: int = 1234

    class _FakeInfo:
        safetensors: _FakeSafetensors = _FakeSafetensors()

    def _fake_model_info(_m: str) -> _FakeInfo:
        return _FakeInfo()

    monkeypatch.setattr(model_cards, "model_info", _fake_model_info)

    size = await fetch_safetensors_size(ModelId("fake/model"))
    assert size == Memory.from_bytes(1234)


@pytest.mark.asyncio
async def test_image_model_cards_load_even_when_disabled(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(model_cards, "EXO_ENABLE_IMAGE_MODELS", False)
    cache = cast(dict[ModelId, ModelCard], model_cards.__dict__["_card_cache"])
    cache.clear()

    async def _fetch_from_hf(_model_id: ModelId) -> ModelCard:
        raise AssertionError("fetch_from_hf should not be called for image models")

    monkeypatch.setattr(ModelCard, "fetch_from_hf", staticmethod(_fetch_from_hf))

    image_card = await ModelCard.load(ModelId("exolabs/FLUX.1-dev"))
    assert any(task in ("TextToImage", "ImageToImage") for task in image_card.tasks)

    listed = await model_cards.get_model_cards()
    assert all(
        not any(task in ("TextToImage", "ImageToImage") for task in card.tasks)
        for card in listed
    )
