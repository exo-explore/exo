import json

from pydantic import JsonValue

from exo.shared.types.common import ModelId

MODEL = ModelId("test/model")


def raw(tokens: list[int], offset: int = 0) -> str:
    return json.dumps(
        {
            "object": "tensorfold.token.chunk",
            "version": 1,
            "id": "reply",
            "model": MODEL,
            "offset": offset,
            "token_ids": tokens,
        }
    )


def semantic(
    count: int,
    token: int | None,
    delta: dict[str, JsonValue] | None = None,
    *,
    finish: str | None = None,
    tokens: list[int] | None = None,
    completion_tokens: int | None = None,
    logprobs: list[dict[str, JsonValue]] | None = None,
    logprob_positions: list[int] | None = None,
) -> str:
    anchor: dict[str, object] = {"version": 1, "token_count": count, "token_id": token}
    choice: dict[str, object] = {
        "index": 0,
        "delta": delta or {},
        "finish_reason": finish,
    }
    if logprobs is not None:
        choice["logprobs"] = {"content": logprobs}
        if logprob_positions is not None:
            anchor["logprob_positions"] = {"content": logprob_positions}
    event: dict[str, object] = {
        "object": "chat.completion.chunk",
        "id": "reply",
        "model": MODEL,
        "choices": [choice],
        "tensorfold_stream": anchor,
    }
    if tokens is not None:
        completions = len(tokens) if completion_tokens is None else completion_tokens
        event["tensorfold"] = {
            "token_ids": tokens,
            "tokens_per_second": 12.0,
            "prefill_seconds": 0.5,
        }
        event["usage"] = {
            "prompt_tokens": 4,
            "completion_tokens": completions,
            "total_tokens": 4 + completions,
        }
    return json.dumps(event)
