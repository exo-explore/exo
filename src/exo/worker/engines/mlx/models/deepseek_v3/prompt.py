from typing import Any

from exo.shared.types.text_generation import TextGenerationTaskParams


def applies_to(task_params: TextGenerationTaskParams) -> bool:
    return "deepseek-v3.2" in task_params.model.lower()


def encode_prompt(
    formatted_messages: list[dict[str, Any]], task_params: TextGenerationTaskParams
) -> str:
    """DeepSeek V3.2 ships no Jinja chat template: encode with the DSML encoder."""
    from exo.worker.engines.mlx.models.deepseek_v3.dsml_encoding import (
        encode_messages,
    )

    return encode_messages(
        messages=formatted_messages,
        # Only use chat mode if enable thinking is explicitly Fakse.
        thinking_mode="chat" if task_params.enable_thinking is False else "thinking",
        tools=task_params.tools,
    )
