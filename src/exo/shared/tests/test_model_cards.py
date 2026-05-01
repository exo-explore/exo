from pathlib import Path

from anyio import Path as AsyncPath
import pytest

from exo.shared.models import model_cards
from exo.shared.models.model_cards import ModelCard, ModelId, get_model_cards


@pytest.mark.asyncio
async def test_gpt_oss_cards_advertise_tools() -> None:
    cards = {card.model_id: card for card in await get_model_cards()}

    for model_id in (
        ModelId("mlx-community/gpt-oss-120b-MXFP4-Q8"),
        ModelId("mlx-community/gpt-oss-20b-MXFP4-Q8"),
    ):
        assert model_id in cards
        assert "tools" in cards[model_id].capabilities


@pytest.mark.asyncio
async def test_custom_cards_override_cached_builtin_cards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builtin_dir = tmp_path / "builtin"
    custom_dir = tmp_path / "custom"
    builtin_dir.mkdir()
    custom_dir.mkdir()

    card = """\
model_id = "test/model"
n_layers = 1
hidden_size = 1
supports_tensor = false
tasks = ["TextGeneration"]
capabilities = {capabilities}

[storage_size]
in_bytes = 1
"""
    (builtin_dir / "test--model.toml").write_text(
        card.format(capabilities='["text"]')
    )
    (custom_dir / "test--model.toml").write_text(
        card.format(capabilities='["text", "tools"]')
    )

    monkeypatch.setattr(model_cards, "_BUILTIN_CARD_DIRS", [AsyncPath(builtin_dir)])
    monkeypatch.setattr(model_cards, "_custom_cards_dir", AsyncPath(custom_dir))
    monkeypatch.setattr(model_cards, "_card_cache", {})

    loaded = {card.model_id: card for card in await get_model_cards()}

    assert loaded[ModelId("test/model")].capabilities == ["text", "tools"]
    assert loaded[ModelId("test/model")].is_custom is True
