import pytest
from exo.shared.models.model_cards import (
    MODEL_CARDS,
    ModelCard,
    ModelId,
    save_custom_models,
    load_custom_models,
    CUSTOM_MODELS_PATH,
)
from exo.download.impl_shard_downloader import build_full_shard
from pathlib import Path
import json
import os


@pytest.mark.asyncio
async def test_custom_model_registration():
    # Test valid model registration
    model_id = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

    # 1. Register logic (simulated from API)
    model_card = await ModelCard.from_hf(ModelId(model_id))
    short_id = f"custom-{model_id.split('/')[-1]}"
    model_card.pretty_name = model_id.split("/")[-1].replace("-", " ")
    MODEL_CARDS[short_id] = model_card
    save_custom_models()

    assert short_id in MODEL_CARDS
    assert MODEL_CARDS[short_id].pretty_name == "Qwen2.5 0.5B Instruct 4bit"
    assert MODEL_CARDS[short_id].storage_size.in_bytes > 0

    # 2. Verify persistence
    custom_models_path = CUSTOM_MODELS_PATH
    assert custom_models_path.exists()

    with open(custom_models_path, "r") as f:
        data = json.load(f)
        assert short_id in data
        assert data[short_id]["pretty_name"] == "Qwen2.5 0.5B Instruct 4bit"


@pytest.mark.asyncio
async def test_lazy_loading_logic():
    # This would require mocking execution context but we can verify plan logic if we extracted it.
    # For now, let's focus on registration and persistence.
    pass


@pytest.mark.asyncio
async def test_shard_download_registration():
    # Test that ensure_shard registers the model if unknown
    # Need to mock download_shard to avoid actual download
    pass
