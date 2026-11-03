"""A request that the model can't run is refused before any work is done on it.

A request longer than the model's context window used to be run anyway: a 160k-token prompt
to a 131,072-token model was prefilled for minutes. It now ends at once with OpenAI's
"context_length_exceeded" error and OpenAI's wording, which clients and tools recognise. So
does a request the model's chat template can't render, as an invalid request.

Only the runner can count a prompt's tokens (it has the tokenizer and the chat template). The
check is made where the runner tokenizes the prompt to run it, so it costs nothing more, and it
comes out the same on every rank, which all skip the request before any collective.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast, final

from loguru import logger

from exo.shared.request_errors import CONTEXT_LENGTH_EXCEEDED, INVALID_REQUEST


class RequestRefusedError(Exception):
    """A request the model can't run as it is. Its message is said to the user."""


@final
class ContextLengthExceededError(RequestRefusedError):
    pass


@final
class InvalidRequestError(RequestRefusedError):
    def __init__(self, reason: str):
        super().__init__(f"{INVALID_REQUEST}{reason.rstrip('.')}.")


def context_length_error(
    prompt_tokens: int, max_output_tokens: int | None, context_length: int
) -> str | None:
    """The error for a request that doesn't fit in the context window, or None if it fits.

    The prompt plus max_tokens must fit; with no max_tokens (or "no limit", as Ollama's -1),
    the prompt must leave room for one token. A context length of 0 means it's unknown.
    """
    completion = max_output_tokens if max_output_tokens and max_output_tokens > 0 else 0
    if context_length <= 0 or prompt_tokens + max(completion, 1) <= context_length:
        return None
    if completion:
        return (
            f"{CONTEXT_LENGTH_EXCEEDED}{context_length} tokens. "
            f"However, you requested {prompt_tokens + completion} tokens "
            f"({prompt_tokens} in the messages, {completion} in the completion). "
            "Please reduce the length of the messages or completion."
        )
    return (
        f"{CONTEXT_LENGTH_EXCEEDED}{context_length} tokens. "
        f"However, your messages resulted in {prompt_tokens} tokens. "
        "Please reduce the length of the messages."
    )


def check_context_length(
    prompt_tokens: int, max_output_tokens: int | None, context_length: int
) -> None:
    """Raise ContextLengthExceededError if the request doesn't fit"""
    if (
        message := context_length_error(
            prompt_tokens, max_output_tokens, context_length
        )
    ) is not None:
        raise ContextLengthExceededError(message)


def model_context_length(card_context_length: int, config: Mapping[str, object]) -> int:
    """The longest context the model takes: the most that its card or its config.json says.

    Cards can say less than the model takes (Qwen3-30B-A3B's card says 32,768; its config says
    40,960), and a request the model can run must never be refused. A config gives its own
    limit as max_position_embeddings, for a vision model in its text_config, and a model with
    YaRN rope scaling takes its original length times the factor. (Llama 3's rope scaling is
    already counted in its max_position_embeddings.)
    """
    limits = [card_context_length]
    for section in (config, config.get("text_config")):
        if not isinstance(section, Mapping):
            continue
        section = cast(Mapping[str, object], section)
        limits.append(_positive_int(section.get("max_position_embeddings")))
        rope = section.get("rope_scaling")
        if isinstance(rope, Mapping):
            rope = cast(Mapping[str, object], rope)
            if rope.get("rope_type", rope.get("type")) != "yarn":
                continue
            factor = rope.get("factor")
            original = _positive_int(rope.get("original_max_position_embeddings"))
            if isinstance(factor, (int, float)) and not isinstance(factor, bool):
                limits.append(int(original * factor))
    return max(limits)


def read_model_context_length(model_dir: Path, card_context_length: int) -> int:
    """model_context_length for the model in model_dir"""
    try:
        config = cast(object, json.loads((model_dir / "config.json").read_text()))
    except (OSError, ValueError) as error:
        logger.warning(f"Couldn't read the model's config.json: {error}")
        return card_context_length
    if not isinstance(config, Mapping):
        return card_context_length
    return model_context_length(card_context_length, cast(Mapping[str, object], config))


def _positive_int(value: object) -> int:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else 0
    )
