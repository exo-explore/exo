import re
from typing import Any, cast

from exo.shared.types.text_generation import TextGenerationTaskParams

# exo's reasoning efforts on the encoder's ("low" = 50, "high" = 75, "max" = 100 on its
# numeric 1-100 scale). None renders the encoder's default, "high". "medium" is what exo
# sends when thinking is on and no effort was asked for, so it gets the default too.
_EFFORT: dict[str, str | None] = {
    "minimal": "low",
    "low": "low",
    "medium": None,
    "high": "high",
    "xhigh": "max",
}


def applies_to(task_params: TextGenerationTaskParams) -> bool:
    """For callers that don't know the loaded model's model_type."""
    return "deepseek-v4.1" in task_params.model.lower()


def reasoning_effort(task_params: TextGenerationTaskParams) -> str | None:
    effort = task_params.reasoning_effort
    if effort is None or effort == "none":
        return None
    return _EFFORT[effort]


def _strip_thinking_markers(content: str) -> str:
    """Remove `<think>...</think>` blocks and stray think tags from an earlier assistant
    turn's content (the encoder renders reasoning from `reasoning_content`)."""
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return cleaned.replace("<think>", "").replace("</think>", "")


def encode_prompt(
    formatted_messages: list[dict[str, Any]], task_params: TextGenerationTaskParams
) -> str:
    """DeepSeek V4.1's prompt, from DeepSeek's encoder (the model card names it as the
    prompt format; the repository's Jinja template is a community port of it)."""
    from exo.worker.engines.mlx.models.deepseek_v41.encoding import encode_messages

    messages = [dict(message) for message in formatted_messages]
    for message in messages:
        if message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = _strip_thinking_markers(content)
    if task_params.tools:
        # The encoder renders tools with a system message: the leading one, or a new
        # empty one (as the repository's chat template does).
        for message in messages:
            if message.get("role") in ("system", "developer"):
                message["tools"] = task_params.tools
                break
        else:
            messages.insert(
                0, {"role": "system", "content": "", "tools": task_params.tools}
            )
    return cast(
        str,
        encode_messages(
            messages,
            thinking_mode="chat"
            if task_params.enable_thinking is False
            else "thinking",
            reasoning_effort=reasoning_effort(task_params),
        ),
    )
