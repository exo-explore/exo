import asyncio
from pathlib import Path

from anyio import Path as AsyncPath

from exo.shared.models.model_cards import ConfigData, ModelCard


def test_recommended_qwen38_card_is_loadable_and_conservative() -> None:
    card_path = (
        Path(__file__).resolve().parents[4]
        / "resources"
        / "inference_model_cards"
        / "sh0wie--Qwen3.8-Flash-Next-REAP-288-MLX-4bit.toml"
    )

    card = asyncio.run(ModelCard.load_from_path(AsyncPath(card_path)))

    assert card.model_id == "sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit"
    assert card.storage_size.in_bytes == 73_504_903_388
    assert card.n_layers == 48
    assert card.supports_tensor is False
    assert "text" in card.capabilities
    assert "vision" not in card.capabilities


def test_qwen4_exp_config_uses_text_shape_without_claiming_tensor_support() -> None:
    config = ConfigData.model_validate(
        {
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp",
            "image_token_id": 248056,
            "text_config": {
                "model_type": "qwen4_exp_text",
                "hidden_size": 2560,
                "num_hidden_layers": 48,
                "num_key_value_heads": 2,
                "max_position_embeddings": 262144,
            },
            "vision_config": {"model_type": "qwen3_5_vision"},
        },
        context={"model_id": "Qwen/Qwen3.8-Flash-Next"},
    )

    assert config.architectures == ["Qwen4ExpForConditionalGeneration"]
    assert config.hidden_size == 2560
    assert config.layer_count == 48
    assert config.num_key_value_heads == 2
    assert config.max_position_embeddings == 262144
    assert config.supports_tensor is False
    assert config.vision is not None
    assert config.vision.model_type == "qwen4_exp"
    assert config.vision.image_token_id == 248056
    assert config.vision.weights_repo == "Qwen/Qwen3.8-Flash-Next"
