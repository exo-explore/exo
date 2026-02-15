import contextlib
import json
from typing import Any, cast

from loguru import logger

from exo.shared.types.text_generation import TextGenerationTaskParams


def apply_chat_template(
    tokenizer: Any,  # pyright: ignore[reportAny]
    task_params: TextGenerationTaskParams,
) -> str:
    """Convert TextGenerationTaskParams to a chat template prompt.

    Converts the internal format (input + instructions) to a messages list
    that can be processed by the tokenizer's chat template.

    When chat_template_messages is available (from Chat Completions API),
    uses those directly to preserve tool_calls, thinking, and other fields.
    Otherwise builds messages from the task params input/instructions.
    """
    formatted_messages: list[dict[str, Any]] = []
    if task_params.chat_template_messages is not None:
        formatted_messages = list(task_params.chat_template_messages)
        for msg in formatted_messages:
            _normalize_tool_calls(msg)
    else:
        if task_params.instructions:
            formatted_messages.append(
                {"role": "system", "content": task_params.instructions}
            )

        for msg in task_params.input:
            if not msg.content:
                logger.warning("Received message with empty content, skipping")
                continue
            formatted_messages.append({"role": msg.role, "content": msg.content})

    # For assistant prefilling, append content after templating to avoid a closing turn token.
    partial_assistant_content: str | None = None
    if formatted_messages and formatted_messages[-1].get("role") == "assistant":
        partial_assistant_content = cast(str, formatted_messages[-1].get("content", ""))
        formatted_messages = formatted_messages[:-1]

    if hasattr(tokenizer, "apply_chat_template"):  # pyright: ignore[reportAny]
        prompt: str = tokenizer.apply_chat_template(  # pyright: ignore[reportAny]
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=task_params.tools,
        )
        if partial_assistant_content:
            prompt += partial_assistant_content
        return prompt

    # Fallback: ChatML format
    parts: list[str] = []
    for msg in formatted_messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg.get('content', '')}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    result = "".join(parts)
    if partial_assistant_content:
        result += partial_assistant_content
    return result


def _normalize_tool_calls(msg_dict: dict[str, Any]) -> None:
    tool_calls: list[Any] | None = msg_dict.get("tool_calls")
    if not tool_calls:
        return

    for tool_call in tool_calls:  # pyright: ignore[reportAny]
        if not isinstance(tool_call, dict):
            continue

        func: dict[str, Any] | None = tool_call.get("function")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if not isinstance(func, dict):
            continue

        args: Any = func.get("arguments")  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if isinstance(args, str):
            with contextlib.suppress(json.JSONDecodeError):
                func["arguments"] = json.loads(args)
