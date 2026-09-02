from exo.shared.models.model_cards import ConfigData, ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.memory import Memory


def test_config_detects_verified_ring_architectures() -> None:
    for architecture in ("LlamaForCausalLM", "Qwen3ForCausalLM"):
        config = ConfigData.model_validate(
            {"architectures": [architecture], "num_hidden_layers": 1}
        )

        assert config.supports_ring is True


def test_config_rejects_unverified_ring_architecture() -> None:
    config = ConfigData.model_validate(
        {"architectures": ["Gemma2ForCausalLM"], "num_hidden_layers": 1}
    )

    assert config.supports_ring is False


def test_model_card_defaults_ring_support_to_false() -> None:
    card = ModelCard(
        model_id=ModelId("test/model"),
        storage_size=Memory.from_bytes(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )

    assert card.supports_ring is False
