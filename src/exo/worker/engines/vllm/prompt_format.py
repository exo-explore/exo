from typing import Any, cast

from vllm.sampling_params import SamplingParams
from vllm.v1.engine.llm_engine import LLMEngine

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.utils_mlx import (
    normalize_tool_calls,
    patch_lossy_chat_template,
    schemas_lost_in_prompt,
)
from exo.worker.runner.bootstrap import logger


def format_vllm_prompt(
    engine: LLMEngine, params: TextGenerationTaskParams
) -> tuple[list[int], str, int]:
    tokenizer = engine.get_tokenizer()

    if params.chat_template_messages is not None:
        formatted_messages: list[dict[str, Any]] = list(params.chat_template_messages)
        for msg in formatted_messages:
            normalize_tool_calls(msg)
    else:
        formatted_messages = []
        if params.instructions:
            formatted_messages.append(
                {"role": "system", "content": params.instructions}
            )
        for msg in params.input:
            if msg.content:
                formatted_messages.append({"role": msg.role, "content": msg.content})

    partial_assistant_content: str | None = None
    if formatted_messages and formatted_messages[-1].get("role") == "assistant":
        last_content = cast(object, formatted_messages[-1].get("content", ""))
        partial_assistant_content = str(last_content)
        formatted_messages = formatted_messages[:-1]

    extra_kwargs: dict[str, bool | str] = {}
    if params.enable_thinking is not None:
        extra_kwargs["enable_thinking"] = params.enable_thinking
        extra_kwargs["thinking"] = params.enable_thinking
    if params.reasoning_effort is not None:
        extra_kwargs["reasoning_effort"] = params.reasoning_effort

    patched_template: str | None = None
    chat_template = getattr(tokenizer, "chat_template", None)
    if params.tools and isinstance(chat_template, str):
        patched_template = patch_lossy_chat_template(chat_template)
        if patched_template is not None:
            logger.info("Patched lossy chat template (removed inner_type length guard)")

    prompt_text: str = tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=params.tools,
        **({"chat_template": patched_template} if patched_template is not None else {}),
        **extra_kwargs,
    )
    assert isinstance(prompt_text, str)

    if params.tools and schemas_lost_in_prompt(prompt_text, params.tools):
        logger.warning("Chat template lost nested tool schemas even after patching")

    if partial_assistant_content:
        prompt_text += partial_assistant_content

    token_ids: list[int] = tokenizer.encode(prompt_text, add_special_tokens=False)  # type: ignore[reportUnknownMemberType]
    special = {200002, 200005, 200006, 200007, 200008}
    annotated = [f"*{t}*" if t in special else str(t) for t in token_ids]
    logger.info(f"prompt tokens ({len(token_ids)}): [{', '.join(annotated)}]")

    return token_ids, prompt_text, len(token_ids)


_VLLM_EXTRA_STOP_TOKEN_IDS: dict[str, list[int]] = {
    "gpt-oss": [200012],
    "gpt_oss": [200012],
    "kimi-k2": [163586],
    "glm-5": [154820, 154827, 154829],
    "glm-4.7": [154820, 154827, 154829],
    "qwen3.5": [248046, 248044],
    "qwen-3.5": [248046, 248044],
}


def make_vllm_sampling_params(
    engine: LLMEngine, params: TextGenerationTaskParams, model_id: ModelId | None = None
) -> SamplingParams:
    kwargs: dict[str, object] = {}

    if params.max_output_tokens is not None:
        kwargs["max_tokens"] = params.max_output_tokens
    else:
        kwargs["max_tokens"] = min(engine.model_config.max_model_len, 32168)
    if params.temperature is not None:
        kwargs["temperature"] = params.temperature
    if params.top_p is not None:
        kwargs["top_p"] = params.top_p
    if params.top_k is not None:
        kwargs["top_k"] = params.top_k
    if params.min_p is not None:
        kwargs["min_p"] = params.min_p
    if params.stop is not None:
        kwargs["stop"] = params.stop
    if params.seed is not None:
        kwargs["seed"] = params.seed
    if params.repetition_penalty is not None:
        kwargs["repetition_penalty"] = params.repetition_penalty
    if params.logprobs:
        kwargs["logprobs"] = params.top_logprobs or 1

    if model_id is not None:
        lower = model_id.lower()
        extra_stop: list[int] = []
        for key, ids in _VLLM_EXTRA_STOP_TOKEN_IDS.items():
            if key in lower:
                extra_stop = ids
                break
        if extra_stop:
            kwargs["stop_token_ids"] = extra_stop

    return SamplingParams(**kwargs)
