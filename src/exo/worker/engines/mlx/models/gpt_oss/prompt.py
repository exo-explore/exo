from typing import Any

from exo.shared.types.text_generation import TextGenerationTaskParams


def applies_to(task_params: TextGenerationTaskParams) -> bool:
    return "gpt-oss" in task_params.model.lower()


def prepare_template_messages(formatted_messages: list[dict[str, Any]]) -> None:
    # Put reasoning content in thinking block for GPT OSS
    for msg in formatted_messages:
        if msg.get("role") == "assistant" and "thinking" not in msg:
            rc = msg.get("reasoning_content")
            if isinstance(rc, str) and rc:
                msg["thinking"] = rc
