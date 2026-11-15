import pytest
from pydantic import JsonValue, TypeAdapter

from exo.shared.request_errors import generation_error_code
from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.worker.engines.tensorfold.capabilities import LoadedCapabilities, LoadedModel
from exo.worker.engines.tensorfold.requests import (
    UnsupportedRequestError,
    invalid_request,
    request_body,
)
from exo.worker.engines.tensorfold.tests.fixtures import MODEL

_json = TypeAdapter(dict[str, JsonValue])


def loaded() -> LoadedModel:
    return LoadedModel(
        id=MODEL,
        capabilities=LoadedCapabilities(
            text_generation=True,
            reasoning=True,
            thinking_switch=True,
            logprobs=True,
            penalties=True,
        ),
    )


def test_preserves_developer_tool_history_reasoning_and_sampling_without_mutation() -> (
    None
):
    params = TextGenerationTaskParams(
        model=MODEL,
        input=[
            InputMessage(role="user", content=InputMessageContent("ignored fallback"))
        ],
        instructions=InputMessageContent("already in history"),
        chat_template_messages=[
            {"role": "system", "content": "one system"},
            {"role": "developer", "content": "do this"},
            {
                "role": "assistant",
                "reasoning_content": "checked",
                "tool_calls": [
                    {
                        "id": "native-call",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "native-call", "content": "result"},
        ],
        max_output_tokens=100,
        temperature=0.3,
        top_p=0.9,
        stop=["end"],
        tools=[
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
        enable_thinking=False,
        reasoning_effort="none",
        logprobs=True,
        top_logprobs=2,
    )
    before = params.model_dump_json()
    body = _json.validate_json(request_body(params, loaded()))
    assert params.model_dump_json() == before
    messages = body["messages"]
    assert isinstance(messages, list) and len(messages) == 4
    assert messages[1] == {"role": "developer", "content": "do this"}
    assert messages[3] == {
        "role": "tool",
        "tool_call_id": "native-call",
        "content": "result",
    }
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["stream_options"] == {"tensorfold_token_stream_version": 1}
    assert body["return_token_ids"] is True
    assert body["max_tokens"] == 100 and body["top_logprobs"] == 2


@pytest.mark.parametrize(
    "values, reason",
    [
        ({"max_output_tokens": 0}, "at least one"),
        ({"repetition_penalty": 1.1}, "repetition_penalty"),
        ({"repetition_context_size": 50}, "repetition_context_size"),
        ({"prefill_endpoint": "localhost:123"}, "disaggregated"),
    ],
)
def test_refuses_parameters_that_would_be_silently_dropped(
    values: dict[str, object], reason: str
) -> None:
    params = TextGenerationTaskParams.model_validate(
        {"model": MODEL, "input": [], **values}
    )
    with pytest.raises(ValueError, match=reason):
        request_body(params, loaded())


def test_loaded_capabilities_gate_nonzero_penalties_and_logprobs() -> None:
    restricted = LoadedModel(
        id=MODEL, capabilities=LoadedCapabilities(text_generation=True)
    )
    params = TextGenerationTaskParams(model=MODEL, input=[], frequency_penalty=0.5)
    with pytest.raises(ValueError, match="penalties"):
        request_body(params, restricted)
    params = TextGenerationTaskParams(model=MODEL, input=[], logprobs=True)
    with pytest.raises(ValueError, match="log probabilities"):
        request_body(params, restricted)


def test_client_refusals_carry_exos_invalid_request_code() -> None:
    restricted = LoadedModel(
        id=MODEL, capabilities=LoadedCapabilities(text_generation=True)
    )
    params = TextGenerationTaskParams(model=MODEL, input=[], logprobs=True)
    with pytest.raises(UnsupportedRequestError) as refusal:
        request_body(params, restricted)
    message = invalid_request(str(refusal.value))
    assert generation_error_code(message) == "invalid_request"
    assert message.endswith("does not support log probabilities.")
    mismatched = TextGenerationTaskParams(model=ModelId("another/model"), input=[])
    with pytest.raises(ValueError) as routing:
        request_body(mismatched, restricted)
    assert not isinstance(routing.value, UnsupportedRequestError)


def test_prefix_cache_requires_an_enabled_native_cache() -> None:
    params = TextGenerationTaskParams(model=MODEL, input=[], use_prefix_cache=True)
    with pytest.raises(ValueError, match="prompt cache"):
        request_body(params, loaded())
    cached = LoadedModel(
        id=MODEL,
        capabilities=LoadedCapabilities(text_generation=True, prompt_cache=True),
    )
    body = _json.validate_json(request_body(params, cached))
    assert "use_prefix_cache" not in body


def test_greedy_only_native_model_accepts_explicit_zero_temperature() -> None:
    greedy = LoadedModel(
        id=MODEL,
        capabilities=LoadedCapabilities(text_generation=True, greedy_only=True),
    )
    params = TextGenerationTaskParams(model=MODEL, input=[], temperature=0.0)
    assert _json.validate_json(request_body(params, greedy))["temperature"] == 0.0
    with pytest.raises(ValueError, match="greedy"):
        request_body(params.model_copy(update={"temperature": 0.2}), greedy)
