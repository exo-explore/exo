"""
llama.cpp generation functions for exo.

Provides the same interface as the MLX engine but uses llama-cpp-python
for cross-platform inference.
"""

from collections.abc import Generator
from typing import cast

from llama_cpp import Llama

from exo.shared.types.api import ChatCompletionMessage, FinishReason
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.llamacpp.constants import MAX_TOKENS
from exo.worker.runner.bootstrap import logger


def apply_chat_template(
    model: Llama,
    chat_task_data: ChatCompletionTaskParams,
) -> str:
    """
    Apply chat template to format messages for the model.
    Uses the model's built-in chat template if available.
    """
    messages: list[dict[str, str]] = []

    for message in chat_task_data.messages:
        content = message.content
        if content is None:
            continue

        if isinstance(content, list):
            if len(content) == 0:
                continue
            content = content[0].text
        elif hasattr(content, "text"):
            content = content.text

        messages.append({"role": message.role, "content": str(content)})

    # llama-cpp-python handles chat template internally via create_chat_completion
    # For raw prompt generation, we'll format manually if no template
    if hasattr(model, "metadata") and model.metadata:
        # Model has metadata, use simple formatting
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
        prompt_parts.append("Assistant: ")
        return "".join(prompt_parts)

    # Fallback: simple concatenation
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nAssistant: "


def warmup_inference(model: Llama) -> int:
    """
    Warm up the inference engine by generating a few tokens.
    Returns the number of tokens generated.
    """
    warmup_prompt = "Hello, how are you today?"
    tokens_generated = 0

    logger.info("Warming up llama.cpp inference")

    try:
        output = model(
            warmup_prompt,
            max_tokens=10,
            echo=False,
            stream=False,
        )

        if isinstance(output, dict) and "choices" in output:
            choices = output["choices"]
            if choices and len(choices) > 0:
                text = choices[0].get("text", "")
                tokens_generated = len(model.tokenize(text.encode()))

        logger.info(f"Warmup complete, generated {tokens_generated} tokens")

    except Exception as e:
        logger.warning(f"Warmup failed (non-fatal): {e}")
        tokens_generated = 0

    return tokens_generated


def llamacpp_generate(
    model: Llama,
    task: ChatCompletionTaskParams,
) -> Generator[GenerationResponse, None, None]:
    """
    Generate text using llama.cpp, yielding GenerationResponse objects.
    Matches the interface of mlx_generate for compatibility.
    """
    logger.info(f"llama.cpp task_params: {task}")

    # Format the prompt
    prompt = apply_chat_template(model, task)
    max_tokens = task.max_tokens or MAX_TOKENS

    logger.info(f"Starting generation with max_tokens={max_tokens}")

    try:
        # Use streaming generation
        stream = model(
            prompt,
            max_tokens=max_tokens,
            echo=False,
            stream=True,
            temperature=task.temperature if task.temperature is not None else 0.7,
            top_p=task.top_p if task.top_p is not None else 0.9,
        )

        token_idx = 0
        for output in stream:
            if not isinstance(output, dict):
                continue

            choices = output.get("choices", [])
            if not choices:
                continue

            choice = choices[0]
            text = choice.get("text", "")
            finish_reason_raw = choice.get("finish_reason")

            # Map finish reasons
            finish_reason: FinishReason | None = None
            if finish_reason_raw == "stop":
                finish_reason = "stop"
            elif finish_reason_raw == "length":
                finish_reason = "length"

            if text:
                logger.debug(f"Generated token {token_idx}: {repr(text)}")

                yield GenerationResponse(
                    text=text,
                    token=token_idx,
                    finish_reason=finish_reason,
                )
                token_idx += 1

            if finish_reason is not None:
                logger.info(f"Generation complete: {finish_reason}")
                break

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise

