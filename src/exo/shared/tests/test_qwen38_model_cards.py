import pytest

from exo.shared.models.model_cards import ModelTask, card_cache
from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId


@pytest.mark.parametrize(
    ("quantization", "storage_bytes"),
    [("8bit", 29500938720), ("bf16", 54713457120)],
)
async def test_qwen38_builtin_registration(
    quantization: str, storage_bytes: int
) -> None:
    cache = type(card_cache)()
    await cache.refresh()
    model_id = ModelId(f"mlx-community/Qwen3.8-27B-{quantization}")
    card = cache.get(model_id)
    assert card is not None
    assert not card.is_custom
    assert card.n_layers == 64
    assert card.hidden_size == 5120
    assert card.num_key_value_heads == 4
    assert card.context_length == 262144
    assert card.supports_tensor
    assert card.tasks == [ModelTask.TextGeneration]
    assert card.backends == [Backend.MlxMetal]
    assert card.quantization == quantization
    assert card.storage_size.in_bytes == storage_bytes
    assert card.reasoning_dialect == "post_last_user"
    assert card.capabilities == ["text", "thinking", "thinking_toggle"]
