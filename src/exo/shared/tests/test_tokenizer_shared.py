from typing import Any
from unittest.mock import MagicMock

from exo.shared.types.common import ModelId


def test_kimi_eos() -> None:
    """Kimi K2 has a known EOS token ID."""
    from exo.shared.tokenizer.eos_tokens import get_eos_token_ids_for_model

    assert get_eos_token_ids_for_model(ModelId("moonshotai/Kimi-K2-Instruct")) == [163586]


def test_glm_flash_eos() -> None:
    """GLM-4.7-flash has specific EOS token IDs."""
    from exo.shared.tokenizer.eos_tokens import get_eos_token_ids_for_model

    result = get_eos_token_ids_for_model(ModelId("THUDM/glm-4.7-flash"))
    assert result == [154820, 154827, 154829]


def test_glm_generic_eos() -> None:
    """Generic GLM models have different EOS tokens than flash."""
    from exo.shared.tokenizer.eos_tokens import get_eos_token_ids_for_model

    result = get_eos_token_ids_for_model(ModelId("THUDM/glm-4-9b"))
    assert result == [151336, 151329, 151338]


def test_unknown_model_returns_none() -> None:
    """Unknown models should return None (use tokenizer default)."""
    from exo.shared.tokenizer.eos_tokens import get_eos_token_ids_for_model

    assert get_eos_token_ids_for_model(ModelId("meta-llama/Llama-3.2-1B")) is None


def test_chat_template_chatml_fallback() -> None:
    """Without apply_chat_template, fall back to ChatML format."""
    from exo.shared.tokenizer.chat_template import apply_chat_template
    from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams

    tokenizer = MagicMock(spec=[])

    task = TextGenerationTaskParams(
        model=ModelId("test-model"),
        input=[InputMessage(role="user", content="Hello")],
    )

    result: Any = apply_chat_template(tokenizer, task)
    assert "<|im_start|>user" in result
    assert "Hello" in result
    assert "<|im_end|>" in result
    assert result.endswith("<|im_start|>assistant\n")  # pyright: ignore[reportAny]


def test_chat_template_delegates_to_tokenizer() -> None:
    """When tokenizer has apply_chat_template, use it."""
    from exo.shared.tokenizer.chat_template import apply_chat_template
    from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams

    tokenizer: Any = MagicMock()
    tokenizer.apply_chat_template.return_value = "<tokenizer output>"  # pyright: ignore[reportAny]

    task = TextGenerationTaskParams(
        model=ModelId("test-model"),
        input=[InputMessage(role="user", content="Hello")],
    )

    result: Any = apply_chat_template(tokenizer, task)
    assert result == "<tokenizer output>"
    tokenizer.apply_chat_template.assert_called_once()  # pyright: ignore[reportAny]


def test_normalize_tool_calls_parses_json_string() -> None:
    """Tool call arguments stored as strings should be parsed to dicts."""
    from exo.shared.tokenizer.chat_template import (
        _normalize_tool_calls,  # pyright: ignore[reportPrivateUsage]
    )

    msg: dict[str, Any] = {
        "role": "assistant",
        "tool_calls": [
            {"function": {"name": "get_weather", "arguments": '{"city": "London"}'}}
        ],
    }
    _normalize_tool_calls(msg)
    tool_calls: Any = msg["tool_calls"]  # pyright: ignore[reportAny]
    assert tool_calls[0]["function"]["arguments"] == {"city": "London"}
