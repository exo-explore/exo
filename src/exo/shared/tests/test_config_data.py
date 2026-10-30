"""What exo reads from a Hugging Face config.json when a model is added by id."""

from typing import Any

from exo.shared.models.model_cards import ConfigData

# Abridged from nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16: transformers saves the
# block list and no layer count.
_ULTRA_BLOCKS = [
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "attention",
] + ["moe", "mamba"] * 50
ULTRA: dict[str, Any] = {
    "architectures": ["NemotronHForCausalLM"],
    "model_type": "nemotron_h",
    "hidden_size": 8192,
    "num_key_value_heads": 2,
    "max_position_embeddings": 262144,
    "layers_block_type": _ULTRA_BLOCKS,
}

# Abridged from nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16: the language model's
# config is under `llm_config`, next to the vision tower's.
SUPER_VL: dict[str, Any] = {
    "architectures": ["NemotronH_Omni_Reasoning_V3"],
    "model_type": "nemotron_h_omni",
    "img_context_token_id": 18,
    "llm_config": {
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "hidden_size": 4096,
        "num_key_value_heads": 2,
        "max_position_embeddings": 1048576,
        "layers_block_type": ["mamba", "moe"] * 40 + ["attention"] * 8,
    },
    "vision_config": {"model_type": "radio"},
}


def _parse(config: dict[str, Any]) -> ConfigData:
    return ConfigData.model_validate(config, context={"model_id": "nvidia/model"})


def test_layer_count_comes_from_the_block_list() -> None:
    data = _parse(ULTRA)
    assert data.layer_count == 108
    assert data.supports_tensor


def test_omni_checkpoints_are_described_by_their_language_model() -> None:
    data = _parse(SUPER_VL)
    assert data.layer_count == 88
    assert data.hidden_size == 4096
    assert data.num_key_value_heads == 2
    assert data.max_position_embeddings == 1048576
    assert data.supports_tensor
    assert data.vision is None


def test_an_explicit_layer_count_wins() -> None:
    data = _parse({**ULTRA, "num_hidden_layers": 4})
    assert data.layer_count == 4
