import time
from typing import Any, Callable, Generator, cast, get_args

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import trim_prompt_cache
from mlx_lm.sample_utils import (
    make_logits_processors,  # pyright: ignore[reportUnknownVariableType]
    make_sampler,
)
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.api import (
    BenchChatCompletionTaskParams,
    ChatCompletionMessage,
    FinishReason,
    GenerationStats,
)
from exo.shared.types.memory import Memory
from exo.shared.types.mlx import KVCacheType
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
)
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.cache import KVPrefixCache, encode_prompt, make_kv_cache
from exo.worker.engines.mlx.constants import KV_BITS, KV_GROUP_SIZE, MAX_TOKENS
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    mx_barrier,
)
from exo.worker.runner.bootstrap import logger

generation_stream = mx.new_stream(mx.default_device())

_MIN_PREFIX_HIT_TO_UPDATE = 1000


def prefill(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    prompt_tokens: mx.array,
    cache: KVCacheType,
) -> float:
    """Prefill the KV cache with prompt tokens.

    This runs the model over the prompt tokens to populate the cache,
    then trims off the extra generated token.

    Returns:
        tokens_per_sec
    """
    num_tokens = len(prompt_tokens)
    if num_tokens == 0:
        return 0.0

    logger.debug(f"Prefilling {num_tokens} tokens...")
    start_time = time.perf_counter()

    def progress_callback(processed: int, total: int) -> None:
        elapsed = time.time() - start_time
        tok_per_sec = processed / elapsed if elapsed > 0 else 0
        logger.debug(
            f"Prefill progress: {processed}/{total} tokens ({tok_per_sec:.1f} tok/s)"
        )

    # Use max_tokens=1 because max_tokens=0 does not work.
    # We just throw away the generated token - we only care about filling the cache
    for _ in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_tokens,
        max_tokens=1,
        sampler=sampler,
        prompt_cache=cache,
        prefill_step_size=2048,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
        prompt_progress_callback=progress_callback,
    ):
        break  # Stop after first iteration - cache is now filled
    trim_prompt_cache(cast(list[Any], cache), 1)

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0.0
    logger.debug(
        f"Prefill complete: {num_tokens} tokens in {elapsed:.2f}s "
        f"({tokens_per_sec:.1f} tok/s)"
    )
    return tokens_per_sec


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
    kv_prefix_cache: KVPrefixCache | None = None,
) -> Generator[GenerationResponse]:
    # Ensure that generation stats only contains peak memory for this generation
    mx.reset_peak_memory()
    is_bench: bool = isinstance(task, BenchChatCompletionTaskParams)

    # Currently we support chat-completion tasks only.
    logger.debug(f"task_params: {task}")

    if task.seed is not None:
        mx.random.seed(task.seed)

    # Do not use the prefix cache if we are trying to do benchmarks.
    if is_bench:
        kv_prefix_cache = None

    # Use prefix cache if available, otherwise create fresh cache
    prefix_hit_length = 0
    matched_index: int | None = None
    if kv_prefix_cache is None:
        caches = make_kv_cache(model=model)
        prompt_tokens = encode_prompt(tokenizer, prompt)
    else:
        caches, prompt_tokens, matched_index = kv_prefix_cache.get_kv_cache(
            model, prompt
        )
        all_prompt_tokens = encode_prompt(tokenizer, prompt)
        prefix_hit_length = len(all_prompt_tokens) - len(prompt_tokens)

    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] = []
    if is_bench:
        # Only sample length eos tokens
        eos_ids = eos_ids_from_tokenizer(tokenizer)
        logits_processors = [ban_token_ids(eos_ids)]

    # MiniMax M2 models require specific sampling settings per official docs:
    # https://github.com/MiniMax-AI/MiniMax-M2.1
    # - Higher temperature (1.0) for proper reasoning
    # - Light repetition penalty with larger context to prevent conceptual loops
    is_minimax = task.model and "minimax" in task.model.lower()
    if is_minimax:
        logits_processors.extend(
            cast(
                list[Callable[[mx.array, mx.array], mx.array]],
                make_logits_processors(
                    repetition_penalty=1.05,
                    repetition_context_size=256,
                ),
            )
        )

    # Use MiniMax-specific sampling parameters if no user override provided
    if is_minimax:
        sampler = make_sampler(
            temp=task.temperature if task.temperature is not None else 1.0,
            top_p=task.top_p if task.top_p is not None else 0.95,
            top_k=40,
            min_p=0.05,
        )
    else:
        sampler = make_sampler(
            temp=task.temperature if task.temperature is not None else 0.7,
            top_p=task.top_p if task.top_p is not None else 1.0,
        )

    # Prefill cache with all tokens except the last one
    prefill_tps = prefill(model, tokenizer, sampler, prompt_tokens[:-1], caches)

    # stream_generate starts from the last token
    last_token = prompt_tokens[-1:]

    max_tokens = task.max_tokens or MAX_TOKENS
    generated_text_parts: list[str] = []
    generation_start_time = time.perf_counter()
    for out in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=last_token,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        prompt_cache=caches,
        # TODO: Dynamically change prefill step size to be the maximum possible without timing out.
        prefill_step_size=2048,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
    ):
        generated_text_parts.append(out.text)
        logger.info(out.text)

        stats: GenerationStats | None = None
        if out.finish_reason is not None:
            stats = GenerationStats(
                prompt_tps=float(prefill_tps or out.prompt_tps),
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
            # Log generation stats
            generation_elapsed = time.perf_counter() - generation_start_time
            generated_tokens = len(generated_text_parts)
            generation_tps = (
                generated_tokens / generation_elapsed if generation_elapsed > 0 else 0.0
            )
            logger.debug(
                f"Generation complete: prefill {prompt_tokens} tokens @ "
                f"{prefill_tps:.1f} tok/s, generated {generated_tokens} tokens @ "
                f"{generation_tps:.1f} tok/s"
            )
            if kv_prefix_cache is not None:
                full_prompt = prompt + "".join(generated_text_parts)
                if (
                    matched_index is not None
                    and prefix_hit_length >= _MIN_PREFIX_HIT_TO_UPDATE
                ):
                    kv_prefix_cache.update_kv_cache(matched_index, full_prompt, caches)
                else:
                    kv_prefix_cache.add_kv_cache(full_prompt, caches)
            break

        # TODO: Do we want an mx_barrier?
