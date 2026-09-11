import pytest
from pydantic import ValidationError

from exo.shared.types.common import ModelId
from exo.utils.pydantic_ext import FrozenModel


@pytest.mark.parametrize(
    "value",
    [
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "Qwen/Qwen3-0.6B",
        "local-model",
        "org/model.with.dots-and_underscores",
    ],
)
def test_accepts_huggingface_style_ids(value: str) -> None:
    assert ModelId(value) == value
    assert "/" not in ModelId(value).normalize()


@pytest.mark.parametrize(
    "value",
    [
        "..",
        ".",
        "../etc",
        "org/..",
        "org/../other",
        "org//model",
        "/absolute",
        "trailing/",
        "org\\model",
        "org/model\x00",
    ],
)
def test_rejects_ids_that_would_escape_the_models_directory(value: str) -> None:
    with pytest.raises(ValueError):
        ModelId(value)


def test_fresh_id_without_value_is_still_a_uuid() -> None:
    assert len(ModelId()) == 36


class _Payload(FrozenModel):
    model_id: ModelId


def test_pydantic_validation_rejects_traversal_in_payloads() -> None:
    assert _Payload(model_id=ModelId("org/model")).model_id == "org/model"
    with pytest.raises(ValidationError):
        _Payload.model_validate({"model_id": ".."})
