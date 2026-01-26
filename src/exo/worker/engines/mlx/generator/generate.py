from typing import Any, Callable, Generator, cast, get_args

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import KVCache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

# from exo.engines.mlx.cache import KVPrefixCache
from exo.shared.types.api import (
    BenchChatCompletionTaskParams,
    ChatCompletionMessage,
    FinishReason,
    GenerationStats,
)
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
)
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.constants import KV_BITS, KV_GROUP_SIZE, MAX_TOKENS
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    make_kv_cache,
    mx_barrier,
)
from exo.worker.runner.bootstrap import logger

generation_stream = mx.new_stream(mx.default_device())


def maybe_quantize_kv_cache(
    prompt_cache: list[KVCache | Any],
    quantized_kv_start: int,
    kv_group_size: int,
    kv_bits: int | None,
) -> None:
    if kv_bits is None:
        return
    for e, c in enumerate(prompt_cache):
        if (
            hasattr(c, "to_quantized") and c.offset >= quantized_kv_start  # type: ignore
        ):
            prompt_cache[e] = c.to_quantized(group_size=kv_group_size, bits=kv_bits)


def warmup_inference(
    model: Model,
    tokenizer: TokenizerWrapper,
) -> int:
    content = "Prompt to warm up the inference engine. Repeat this."

    warmup_prompt = apply_chat_template(
        tokenizer=tokenizer,
        chat_task_data=ChatCompletionTaskParams(
            model="",
            messages=[
                ChatCompletionMessage(
                    role="user",
                    content=content,
                )
            ],
        ),
    )

    tokens_generated = 0

    cache = make_kv_cache(
        model=model,
    )

    # Use a default sampler for warmup
    sampler = make_sampler(temp=0.7)

    logger.info("Generating warmup tokens")
    for _r in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=warmup_prompt,
        max_tokens=50,
        sampler=sampler,
        prompt_cache=cache,
        prefill_step_size=2048,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
    ):
        logger.info("Generated warmup token: " + str(_r.text))
        tokens_generated += 1

    logger.info("Generated ALL warmup tokens")

    # TODO: Do we want an mx_barrier?
    #  At least this version is actively incorrect, as it should use mx_barrier(group)
    mx_barrier()

    return tokens_generated


def ban_token_ids(token_ids: list[int]) -> Callable[[mx.array, mx.array], mx.array]:
    token_ids = [int(t) for t in token_ids]

    def proc(_history: mx.array, logits: mx.array) -> mx.array:
        for tid in token_ids:
            logits[..., tid] = -1e9
        return logits

    return proc


def eos_ids_from_tokenizer(tokenizer: TokenizerWrapper) -> list[int]:
    eos: list[int] | None = getattr(tokenizer, "eos_token_ids", None)
    if eos is None:
        return []
    return eos


def mlx_generate(
    model: Model,
    tokenizer: TokenizerWrapper,
    task: ChatCompletionTaskParams,
    prompt: str,
) -> Generator[GenerationResponse]:
    # Ensure that generation stats only contains peak memory for this generation
    mx.reset_peak_memory()
    is_bench: bool = isinstance(task, BenchChatCompletionTaskParams)

    # Currently we support chat-completion tasks only.
    logger.debug(f"task_params: {task}")

    if task.seed is not None:
        mx.random.seed(task.seed)

    caches = make_kv_cache(model=model)

    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] = []
    if is_bench:
        # Only sample length eos tokens
        eos_ids = eos_ids_from_tokenizer(tokenizer)
        logits_processors = [ban_token_ids(eos_ids)]

    sampler = make_sampler(
        temp=task.temperature if task.temperature is not None else 0.7,
        top_p=task.top_p if task.top_p is not None else 1.0,
    )

    max_tokens = task.max_tokens or MAX_TOKENS
    for out in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        prompt_cache=caches,
        # TODO: Dynamically change prefill step size to be the maximum possible without timing out.
        prefill_step_size=2048,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
    ):
        logger.info(out.text)

        stats: GenerationStats | None = None
        if out.finish_reason is not None:
            stats = GenerationStats(
                prompt_tps=float(out.prompt_tps),
                generation_tps=float(out.generation_tps),
                prompt_tokens=int(out.prompt_tokens),
                generation_tokens=int(out.generation_tokens),
                peak_memory_usage=Memory.from_gb(out.peak_memory),
            )

            if out.finish_reason not in get_args(FinishReason):
                # We don't throw here as this failure case is really not all that bad
                # Just log the error and move on
                logger.warning(
                    f"Model generated unexpected finish_reason: {out.finish_reason}"
                )

        yield GenerationResponse(
            text=out.text,
            token=out.token,
            finish_reason=cast(FinishReason | None, out.finish_reason),
            stats=stats,
        )

        if out.finish_reason is not None:
            break

        # TODO: Do we want an mx_barrier?
