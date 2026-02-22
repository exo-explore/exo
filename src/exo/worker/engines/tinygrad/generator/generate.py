import time
from collections.abc import Callable, Generator
from typing import Any

from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor

from exo.shared.model_config import ModelConfig
from exo.shared.models.model_cards import ModelId
from exo.shared.tokenizer.eos_tokens import get_eos_token_ids_for_model
from exo.shared.types.api import (
    CompletionTokensDetails,
    GenerationStats,
    PromptTokensDetails,
    TopLogprobItem,
    Usage,
)
from exo.shared.types.memory import Memory
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.tinygrad.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)

from ..forward import forward_pass
from ..sampling import sample_token
from ..weight_loader import TransformerWeights


def tinygrad_generate(
    model: TransformerWeights,
    tokenizer: Any,  # pyright: ignore[reportAny]
    task: TextGenerationTaskParams,
    prompt: str,
    kv_prefix_cache: Any = None,  # pyright: ignore[reportAny]
    on_prefill_progress: Callable[[int, int], None] | None = None,
    group: None = None,
) -> Generator[GenerationResponse]:
    input_ids = _encode_prompt(tokenizer, prompt)
    input_tensor = Tensor([input_ids], dtype=dtypes.int32)

    max_tokens = task.max_output_tokens or DEFAULT_MAX_TOKENS
    temperature = task.temperature or DEFAULT_TEMPERATURE
    top_p = task.top_p or DEFAULT_TOP_P

    request_logprobs = task.logprobs
    top_logprobs_count = task.top_logprobs or 0

    eos_ids = _get_eos_ids(tokenizer, model.config)
    print(f"[DEBUG] eos_ids={eos_ids}")
    prompt_tokens = len(input_ids)

    # Prefill
    prefill_start = time.time()
    logits, cache = forward_pass(model, input_tensor, cache=None, position_offset=0)
    prefill_time = time.time() - prefill_start
    prompt_tps = prompt_tokens / max(prefill_time, 1e-9)

    # Decode
    generation_start = time.time()
    for token_idx in range(max_tokens):
        result = sample_token(
            logits, temperature=temperature, top_p=top_p,
            top_logprobs_count=top_logprobs_count,
        )

        token_text: str = tokenizer.decode([result.token_id])  # pyright: ignore[reportAny]

        is_eos = result.token_id in eos_ids
        if is_eos:
            print(f"[DEBUG] EOS detected: token_id={result.token_id}, text={token_text!r}")
        if "<|eot_id|>" in token_text or "<|end" in token_text:
            print(f"[DEBUG] Special token in text but is_eos={is_eos}, token_id={result.token_id}")
        tokens_generated = token_idx + 1
        elapsed = time.time() - generation_start
        generation_tps = tokens_generated / max(elapsed, 1e-9)

        finish_reason = None
        stats = None
        usage = None

        if is_eos:
            finish_reason = "stop"
        elif token_idx == max_tokens - 1:
            finish_reason = "length"

        if finish_reason is not None:
            stats = GenerationStats(
                prompt_tps=prompt_tps, generation_tps=generation_tps,
                prompt_tokens=prompt_tokens,
                generation_tokens=tokens_generated,
                peak_memory_usage=Memory.from_bytes(0),
            )

            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=tokens_generated,
                total_tokens=prompt_tokens + tokens_generated,
                prompt_tokens_details=PromptTokensDetails(),
                completion_tokens_details=CompletionTokensDetails(),
            )

        logprob = result.logprob if request_logprobs else None
        top_lps = None
        if request_logprobs and task.top_logprobs:
            top_lps = [
                TopLogprobItem(
                    token=str(tokenizer.decode([tok_id])),  # pyright: ignore[reportAny]
                    logprob=lp,
                    bytes=list(str(tokenizer.decode([tok_id])).encode("utf-8")),  # pyright: ignore[reportAny]
                )
                for tok_id, lp in result.top_logprobs
            ]

        if is_eos:
            token_text = ""

        yield GenerationResponse(
            text=token_text, token=result.token_id,
            logprob=logprob, top_logprobs=top_lps,
            finish_reason=finish_reason, stats=stats, usage=usage,
        )

        if finish_reason is not None:
            break

        next_input = Tensor([[result.token_id]], dtype=dtypes.int32)
        logits, cache = forward_pass(
            model, next_input, cache,
            position_offset=prompt_tokens + token_idx + 1,
        )

def warmup_inference(model: TransformerWeights, tokenizer: Any, group: None = None) -> int:  # pyright: ignore[reportAny]
    """Run a full generation loop to warm up forward pass, KV cache, and sampling."""
    from exo.shared.tokenizer.chat_template import apply_chat_template
    from exo.shared.types.common import ModelId as CommonModelId
    from exo.shared.types.text_generation import InputMessage

    warmup_task = TextGenerationTaskParams(
        model=CommonModelId("warmup"),
        input=[InputMessage(role="user", content="Time to warm up!")],
    )

    prompt: str = apply_chat_template(tokenizer, warmup_task)
    tokens_generated = 0

    for _ in tinygrad_generate(model, tokenizer, warmup_task, prompt):
        tokens_generated += 1
        if tokens_generated >= 5:
            break

    return tokens_generated

def _encode_prompt(tokenizer: Any, prompt: str) -> list[int]:  # pyright: ignore[reportAny]
    result: Any = tokenizer.encode(prompt)  # pyright: ignore[reportAny]

    return result.ids if hasattr(result, "ids") else result  # pyright: ignore[reportAny]

def _get_eos_ids(tokenizer: Any, config: ModelConfig) -> set[int]:  # pyright: ignore[reportAny]
    eos_ids: set[int] = set()

    model_eos = get_eos_token_ids_for_model(ModelId(config.architecture_spec.name))

    if model_eos:
        eos_ids.update(model_eos)

    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:  # pyright: ignore[reportAny]
        eos_ids.add(int(tokenizer.eos_token_id))  # pyright: ignore[reportAny]

    if not eos_ids:
        eos_ids.add(2)

    return eos_ids
