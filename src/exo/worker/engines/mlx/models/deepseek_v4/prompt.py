import re
from typing import Any

from exo.shared.types.text_generation import TextGenerationTaskParams


def applies_to(task_params: TextGenerationTaskParams) -> bool:
    return "deepseek-v4" in task_params.model.lower()


def _v4_reasoning_effort(task_params: TextGenerationTaskParams) -> str | None:
    effort = task_params.reasoning_effort
    if effort == "xhigh":
        return "max"
    if effort == "high":
        return "high"
    return None


def _strip_v4_thinking_markers(content: str) -> str:
    """Remove `<think>…</think>` blocks and any stray `<think>`/`</think>` tags
    from prior-turn assistant content.

    The V4 encoder drops `reasoning_content` for older turns when
    `drop_thinking=True`"""
    block = re.compile(r"<think>.*?</think>", re.DOTALL)
    if not content:
        return content
    cleaned = block.sub("", content)
    return cleaned.replace("<think>", "").replace("</think>", "")


def encode_prompt(
    formatted_messages: list[dict[str, Any]], task_params: TextGenerationTaskParams
) -> str:
    """DeepSeek V4 ships no Jinja chat template: encode with its Python encoder."""
    from exo.worker.engines.mlx.models.deepseek_v4.encoding import (
        encode_messages as encode_messages_v4,
    )

    v4_messages = [dict(m) for m in formatted_messages]
    for msg in v4_messages:
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = _strip_v4_thinking_markers(content)
    if task_params.tools:
        for msg in v4_messages:
            if msg.get("role") in ("system", "developer"):
                msg["tools"] = task_params.tools
                break
        else:
            v4_messages.insert(
                0, {"role": "system", "content": "", "tools": task_params.tools}
            )

    return encode_messages_v4(
        messages=v4_messages,
        thinking_mode="chat" if task_params.enable_thinking is False else "thinking",
        reasoning_effort=_v4_reasoning_effort(task_params),
    )
