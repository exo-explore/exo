import pytest

from exo.api.main import API
from exo.shared.models import model_cards
from exo.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    NativeMTPConfig,
)
from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State


def _native_mtp_card(model_id: str, *, default_k: int, max_k: int) -> ModelCard:
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_mb(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
        native_mtp=NativeMTPConfig(
            num_layers=1,
            default_k=default_k,
            max_k=max_k,
        ),
    )


@pytest.mark.asyncio
async def test_models_response_includes_native_mtp_for_native_mtp_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cards = [
        _native_mtp_card("Jundot/Qwen3.6-27B-oQ8-mtp", default_k=2, max_k=3),
        _native_mtp_card(
            "alvarolizama/Qwen3.6-35B-A3B-oQ8-mtp",
            default_k=1,
            max_k=3,
        ),
    ]

    async def _fake_list_all() -> list[ModelCard]:
        return cards

    api = API.__new__(API)
    api.state = State()
    monkeypatch.setattr(model_cards.card_cache, "list_all", _fake_list_all)

    response = await api.get_models()

    by_id = {item.id: item for item in response.data}
    qwen27_native_mtp = by_id["Jundot/Qwen3.6-27B-oQ8-mtp"].native_mtp
    qwen35_native_mtp = by_id["alvarolizama/Qwen3.6-35B-A3B-oQ8-mtp"].native_mtp
    assert qwen27_native_mtp is not None
    assert qwen27_native_mtp.default_k == 2
    assert qwen27_native_mtp.max_k == 3
    assert qwen35_native_mtp is not None
    assert qwen35_native_mtp.default_k == 1
    assert qwen35_native_mtp.max_k == 3
