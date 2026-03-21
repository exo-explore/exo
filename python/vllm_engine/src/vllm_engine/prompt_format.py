from exo_core.types.common import ModelId
from exo_core.types.text_generation import TextGenerationTaskParams
from mlx_lm.tokenizer_utils import TokenizerWrapper
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.llm_engine import LLMEngine

from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    get_eos_token_ids_for_model,
)


def format_vllm_prompt(
    engine: LLMEngine, params: TextGenerationTaskParams
) -> tuple[list[int], str, int]:
    # we should have our own wrapper
    # (instead of abusing mlx's TokenizerWrapper, use tokenizers Tokenizer)
    tokenizer = TokenizerWrapper(engine.get_tokenizer())
    prompt_text = apply_chat_template(tokenizer, params)
    token_ids: list[int] = tokenizer.encode(prompt_text, add_special_tokens=False)  # type: ignore[reportUnknownMemberType]
    return token_ids, prompt_text, len(token_ids)


def make_vllm_sampling_params(
    engine: LLMEngine,
    params: TextGenerationTaskParams,
    model_id: ModelId | None = None,
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
        extra_stop = get_eos_token_ids_for_model(model_id)
        if extra_stop:
            kwargs["stop_token_ids"] = extra_stop

    return SamplingParams(**kwargs)
