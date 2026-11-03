"""A request that doesn't fit in the model's context window is refused, in OpenAI's words, and
the limit is the most the model itself takes, never less."""

import json
from pathlib import Path

import pytest

from exo.shared.request_errors import generation_error_code
from exo.worker.runner.context_length import (
    ContextLengthExceededError,
    InvalidRequestError,
    check_context_length,
    context_length_error,
    model_context_length,
    read_model_context_length,
)


def test_a_prompt_that_fits_is_not_refused() -> None:
    assert context_length_error(100_000, 1000, 131_072) is None
    assert context_length_error(131_071, None, 131_072) is None


def test_a_prompt_longer_than_the_context_is_refused_in_openais_words() -> None:
    message = context_length_error(160_000, None, 131_072)
    assert message == (
        "This model's maximum context length is 131072 tokens. However, your messages "
        "resulted in 160000 tokens. Please reduce the length of the messages."
    )
    assert generation_error_code(message) == "context_length_exceeded"


def test_max_tokens_counts_towards_the_context() -> None:
    assert context_length_error(130_000, 2000, 131_072) == (
        "This model's maximum context length is 131072 tokens. However, you requested "
        "132000 tokens (130000 in the messages, 2000 in the completion). Please reduce "
        "the length of the messages or completion."
    )


def test_a_prompt_filling_the_whole_context_leaves_no_room_to_answer() -> None:
    assert context_length_error(131_072, None, 131_072) is not None


@pytest.mark.parametrize("unlimited", [-1, 0])
def test_a_max_tokens_meaning_no_limit_is_not_counted(unlimited: int) -> None:
    """Ollama's num_predict=-1 means no limit"""
    message = context_length_error(140_000, unlimited, 131_072)
    assert message is not None
    assert "completion" not in message
    assert context_length_error(100_000, unlimited, 131_072) is None


def test_an_unknown_context_length_refuses_nothing() -> None:
    assert context_length_error(10_000_000, 1000, 0) is None


def test_the_check_raises_for_a_request_that_does_not_fit() -> None:
    check_context_length(1000, 100, 131_072)
    with pytest.raises(ContextLengthExceededError):
        check_context_length(160_000, None, 131_072)


def test_an_invalid_request_is_coded() -> None:
    message = str(
        InvalidRequestError("the model's chat template failed: roles must alternate")
    )
    assert message == (
        "This request can't be run on this model: the model's chat template failed: "
        "roles must alternate."
    )
    assert generation_error_code(message) == "invalid_request"


# Qwen3-30B-A3B's card says 32768; its config.json says 40960
QWEN3_CONFIG = {"max_position_embeddings": 40960, "rope_scaling": None}
# DeepSeek V3.x's cards say 131072
DEEPSEEK_CONFIG = {
    "max_position_embeddings": 163840,
    "rope_scaling": {
        "type": "yarn",
        "factor": 40,
        "original_max_position_embeddings": 4096,
    },
}
# Llama 3.2: rope scaling is already counted in max_position_embeddings
LLAMA_CONFIG = {
    "max_position_embeddings": 131072,
    "rope_scaling": {
        "rope_type": "llama3",
        "factor": 32.0,
        "original_max_position_embeddings": 8192,
    },
}


@pytest.mark.parametrize(
    ("card", "config", "limit"),
    [
        (32768, QWEN3_CONFIG, 40960),
        (131072, DEEPSEEK_CONFIG, 163840),
        (131072, LLAMA_CONFIG, 131072),
        # A model run with YaRN turned on in its config
        (
            32768,
            {
                "max_position_embeddings": 32768,
                "rope_scaling": {
                    "rope_type": "yarn",
                    "factor": 4.0,
                    "original_max_position_embeddings": 32768,
                },
            },
            131072,
        ),
        # A vision model keeps its text model's settings apart
        (0, {"text_config": {"max_position_embeddings": 262144}}, 262144),
        # The card says more than the config
        (262144, {"max_position_embeddings": 131072}, 262144),
        (0, {}, 0),
    ],
)
def test_the_limit_is_the_most_the_card_or_the_config_says(
    card: int, config: dict[str, object], limit: int
) -> None:
    assert model_context_length(card, config) == limit


def test_a_request_the_model_takes_is_not_refused_because_of_its_card() -> None:
    """Qwen3-30B-A3B ran a 30k prompt with max_tokens 8192 before the check existed"""
    assert (
        context_length_error(30_000, 8_192, model_context_length(32768, QWEN3_CONFIG))
        is None
    )


def test_the_limit_is_read_from_the_models_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(QWEN3_CONFIG))
    assert read_model_context_length(tmp_path, 32768) == 40960


def test_without_a_readable_config_the_card_is_the_limit(tmp_path: Path) -> None:
    assert read_model_context_length(tmp_path, 32768) == 32768
    (tmp_path / "config.json").write_text("not json")
    assert read_model_context_length(tmp_path, 32768) == 32768
