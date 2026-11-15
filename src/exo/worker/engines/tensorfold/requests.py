import json

from pydantic import JsonValue, TypeAdapter

from exo.shared.request_errors import INVALID_REQUEST
from exo.shared.types.text_generation import ChatTemplateValue, TextGenerationTaskParams
from exo.worker.engines.tensorfold.capabilities import LoadedModel

_tools = TypeAdapter(list[dict[str, JsonValue]])


class UnsupportedRequestError(ValueError):
    """A request the loaded model can't run as sent; the client has to change it."""


def invalid_request(reason: str) -> str:
    """EXO's invalid-request sentence for a refusal, which the API answers with HTTP 400."""
    return f"{INVALID_REQUEST}{reason.rstrip('.')}."


def _value(value: ChatTemplateValue) -> JsonValue:
    if isinstance(value, dict):
        return {key: _value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_value(item) for item in value]
    return value


def request_body(params: TextGenerationTaskParams, loaded: LoadedModel) -> bytes:
    if str(params.model) != loaded.id:
        raise ValueError("The request model does not match the loaded TensorFold model")
    if params.images:
        raise UnsupportedRequestError(
            "TensorFold's EXO backend currently supports text input only"
        )
    if params.prefill_endpoint is not None:
        raise ValueError("TensorFold does not accept EXO disaggregated prefill state")
    if params.max_output_tokens is not None and params.max_output_tokens <= 0:
        raise UnsupportedRequestError(
            "TensorFold's EXO backend requires at least one output token"
        )
    if params.repetition_penalty not in (None, 1.0):
        raise UnsupportedRequestError(
            "TensorFold does not implement EXO repetition_penalty"
        )
    if params.repetition_context_size is not None:
        raise UnsupportedRequestError(
            "TensorFold does not implement EXO repetition_context_size"
        )
    capabilities = loaded.capabilities
    if params.use_prefix_cache and not capabilities.prompt_cache:
        raise UnsupportedRequestError(
            "The loaded TensorFold model does not provide a prompt cache"
        )
    if params.logprobs and not capabilities.logprobs:
        raise UnsupportedRequestError(
            "The loaded TensorFold model does not support log probabilities"
        )
    if not capabilities.penalties and any(
        value not in (None, 0.0)
        for value in (params.presence_penalty, params.frequency_penalty)
    ):
        raise UnsupportedRequestError(
            "The loaded TensorFold model does not support sampling penalties"
        )
    if capabilities.greedy_only and (params.temperature or 0.0) > 0:
        raise UnsupportedRequestError(
            "The loaded TensorFold model requires greedy decoding"
        )
    if (
        params.enable_thinking is False
        and not capabilities.thinking_switch
        and capabilities.thinking_applicable
    ):
        raise UnsupportedRequestError(
            "The loaded TensorFold model cannot disable thinking"
        )
    if params.enable_thinking is True and not capabilities.reasoning:
        raise UnsupportedRequestError(
            "The loaded TensorFold model does not support reasoning"
        )

    messages: list[JsonValue]
    if params.chat_template_messages is not None:
        messages = [
            {key: _value(value) for key, value in message.items()}
            for message in params.chat_template_messages
        ]
    else:
        messages = []
        if params.instructions is not None:
            messages.append({"role": "system", "content": str(params.instructions)})
        messages.extend(
            {"role": message.role, "content": str(message.content)}
            for message in params.input
        )
    body: dict[str, JsonValue] = {
        "model": loaded.id,
        "messages": messages,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"tensorfold_token_stream_version": 1},
    }
    optional: dict[str, JsonValue] = {
        "max_tokens": params.max_output_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "min_p": params.min_p,
        "stop": [item for item in params.stop]
        if isinstance(params.stop, list)
        else params.stop,
        "seed": params.seed,
        "presence_penalty": params.presence_penalty,
        "frequency_penalty": params.frequency_penalty,
        "reasoning_effort": params.reasoning_effort,
        "top_logprobs": params.top_logprobs,
    }
    body.update({key: value for key, value in optional.items() if value is not None})
    if params.tools is not None:
        body["tools"] = [tool for tool in _tools.validate_python(params.tools)]
    if params.enable_thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": params.enable_thinking}
    if params.logprobs:
        body["logprobs"] = True
    return json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
