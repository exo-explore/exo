"""Conversation history for the Llama 3.1-3.3 chat templates.

The templates render one tool call per assistant message and raise "This model only
supports single tool-calls at once!" on more, so a history with parallel calls (another
model's, or a client's) failed the whole request. Such a message is split into one
assistant message per call, each followed by its call's result when the results come
right after it, as the model writes them: a call, then its ipython result.
"""

from typing import Any, cast

from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.models.llama.spec import LLAMA_3_MODEL_ID_SUBSTRINGS

# Llama 3.x checkpoints with a chat template of their own that renders several calls in
# one message (<TOOLCALL>[...]).
_OWN_TOOL_FORMAT = ("nemotron-nano",)


def applies_to(task_params: TextGenerationTaskParams) -> bool:
    model_id = task_params.model.lower()
    return any(
        substring in model_id for substring in LLAMA_3_MODEL_ID_SUBSTRINGS
    ) and not any(substring in model_id for substring in _OWN_TOOL_FORMAT)


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = message.get("tool_calls")
    if message.get("role") != "assistant" or not isinstance(tool_calls, list):
        return []
    return cast(list[dict[str, Any]], tool_calls)


def split_parallel_tool_calls(formatted_messages: list[dict[str, Any]]) -> None:
    messages: list[dict[str, Any]] = []
    index = 0
    while index < len(formatted_messages):
        message = formatted_messages[index]
        index += 1
        tool_calls = _tool_calls(message)
        if len(tool_calls) <= 1:
            messages.append(message)
            continue
        results: list[dict[str, Any]] = []
        while (
            index < len(formatted_messages)
            and formatted_messages[index].get("role") == "tool"
        ):
            results.append(formatted_messages[index])
            index += 1
        for position, tool_call in enumerate(tool_calls):
            split = {**message, "tool_calls": [tool_call]}
            if position > 0:
                # The message's text goes with its first call.
                split["content"] = ""
                split.pop("reasoning_content", None)
            messages.append(split)
            call_id = tool_call.get("id")
            for result in results:
                if call_id is not None and result.get("tool_call_id") == call_id:
                    messages.append(result)
                    results.remove(result)
                    break
        # Results that name no call of the message keep their place after it.
        messages.extend(results)
    formatted_messages[:] = messages
