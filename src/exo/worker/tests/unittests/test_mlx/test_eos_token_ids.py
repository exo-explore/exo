from exo.shared.models.model_cards import ModelId
from exo.worker.engines.mlx.utils_mlx import get_eos_token_ids_for_model


def test_glm_47_uses_glm_47_tokenizer_stop_ids() -> None:
    """GLM-4.7 model cards use the GLM-4 tokenizer vocabulary, not GLM-5 IDs."""
    assert get_eos_token_ids_for_model(ModelId("mlx-community/GLM-4.7-4bit")) == [
        151336,
        151329,
        151338,
    ]


def test_glm_5_uses_glm_5_tokenizer_stop_ids() -> None:
    """GLM-5 keeps its separate tokenizer stop IDs."""
    assert get_eos_token_ids_for_model(ModelId("zai-org/GLM-5-9B-0414")) == [
        154820,
        154827,
        154829,
    ]
