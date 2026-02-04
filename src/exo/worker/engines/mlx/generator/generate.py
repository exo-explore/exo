import time
from copy import deepcopy
from typing import Callable, Generator, cast, get_args

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.api import (
    CompletionTokensDetails,
    FinishReason,
    GenerationStats,
    PromptTokensDetails,
    Usage,
)
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.mlx import KVCacheType
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
)
from exo.worker.engines.mlx import Model
from mlx_lm.models.cache import ArraysCache

from exo.worker.engines.mlx.cache import (
    KVPrefixCache,
    SSMSnapshot,
    encode_prompt,
    has_ssm_caches,
    make_kv_cache,
    snapshot_ssm_states,
)
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
    capture_ssm_snapshots: bool = False,
) -> tuple[float, int, list[SSMSnapshot]]:
    """Prefill the KV cache with prompt tokens.

    This runs the model over the prompt tokens to populate the cache,
    then trims off the extra generated token.

    Returns:
        tokens_per_sec
    """
    num_tokens = len(prompt_tokens)
    if num_tokens == 0:
        return 0.0, 0, []

    logger.debug(f"Prefilling {num_tokens} tokens...")
    start_time = time.perf_counter()
    has_ssm = has_ssm_caches(cache)
    snapshots: list[SSMSnapshot] = []

    def progress_callback(processed: int, total: int) -> None:
        elapsed = time.perf_counter() - start_time
        tok_per_sec = processed / elapsed if elapsed > 0 else 0
        logger.debug(
            f"Prefill progress: {processed}/{total} tokens ({tok_per_sec:.1f} tok/s)"
        )
        if has_ssm:
            snapshots.append(snapshot_ssm_states(cache))

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

    # stream_generate added 1 extra generated token to the cache, so we should trim ti.
    pre_gen = snapshots[-1] if has_ssm else None
    for i, c in enumerate(cache):
        if has_ssm and isinstance(c, ArraysCache):
            if pre_gen.states[i] is not None:
                c.state = deepcopy(pre_gen.states[i])  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        else:
            c.trim(1)  # pyright: ignore[reportUnknownMemberType]


    elapsed = time.perf_counter() - start_time
    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0.0
    logger.debug(
        f"Prefill complete: {num_tokens} tokens in {elapsed:.2f}s "
        f"({tokens_per_sec:.1f} tok/s)"
    )
    return tokens_per_sec, num_tokens, snapshots


def warmup_inference(
    model: Model,
    tokenizer: TokenizerWrapper,
    group: mx.distributed.Group | None = None,
) -> int:
    content = "Prompt to warm up the inference engine. Repeat this."

    warmup_prompt = apply_chat_template(
        tokenizer=tokenizer,
        task_params=TextGenerationTaskParams(
            model=ModelId(""),
            input=[InputMessage(role="user", content=content)],
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

    mx_barrier(group)

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
    task: TextGenerationTaskParams,
    prompt: str,
    kv_prefix_cache: KVPrefixCache | None = None,
    group: mx.distributed.Group | None = None,
) -> Generator[GenerationResponse]:
    # Ensure that generation stats only contains peak memory for this generation
    mx.reset_peak_memory()
    if task.seed is not None:
        mx.random.seed(task.seed)

    # Do not use the prefix cache if we are trying to do benchmarks.
    is_bench = task.bench
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

    sampler = make_sampler(
        temp=task.temperature if task.temperature is not None else 0.7,
        top_p=task.top_p if task.top_p is not None else 1.0,
        top_k=task.top_k if task.top_k is not None else 0,
    )

    # Normalize stop sequences to a list
    stop_sequences: list[str] = (
        ([task.stop] if isinstance(task.stop, str) else task.stop)
        if task.stop is not None
        else []
    )
    max_stop_len = max((len(s) for s in stop_sequences), default=0)

    # Prefill cache with all tokens except the last one
    capture_snapshots = has_ssm_caches(caches) and kv_prefix_cache is not None
    prefill_tps, prefill_tokens, ssm_snapshots_list = prefill(
        model,
        tokenizer,
        sampler,
        prompt_tokens[:-1],
        caches,
        capture_ssm_snapshots=capture_snapshots,
    )
    ssm_snapshots: list[SSMSnapshot] | None = ssm_snapshots_list or None

    # stream_generate starts from the last token
    last_token = prompt_tokens[:-1]

    max_tokens = task.max_output_tokens or MAX_TOKENS
    accumulated_text = ""
    generated_text_parts: list[str] = []
    generation_start_time = time.perf_counter()
    usage: Usage | None = None
    in_thinking = False
    reasoning_tokens = 0
    think_start = tokenizer.think_start
    think_end = tokenizer.think_end
    for completion_tokens, out in enumerate(
        stream_generate(
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
        ),
        start=1,
    ):
        generated_text_parts.append(out.text)
        logger.info(out.text)
        accumulated_text += out.text

        if think_start is not None and out.text == think_start:
            in_thinking = True
        elif think_end is not None and out.text == think_end:
            in_thinking = False
        if in_thinking:
            reasoning_tokens += 1

        # Check for stop sequences
        text = out.text
        finish_reason: FinishReason | None = cast(
            FinishReason | None, out.finish_reason
        )
        stop_matched = False

        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in accumulated_text:
                    # Trim text to just before the stop sequence
                    stop_index = accumulated_text.find(stop_seq)
                    text_before_stop = accumulated_text[:stop_index]
                    chunk_start = len(accumulated_text) - len(out.text)
                    text = text_before_stop[chunk_start:]
                    finish_reason = "stop"
                    stop_matched = True
                    break

        is_done = finish_reason is not None

        stats: GenerationStats | None = None
        if is_done:
            stats = GenerationStats(
                prompt_tps=float(prefill_tps or out.prompt_tps),
                generation_tps=float(out.generation_tps),
                prompt_tokens=int(prefill_tokens + out.prompt_tokens),
                generation_tokens=int(out.generation_tokens),
                peak_memory_usage=Memory.from_gb(out.peak_memory),
            )
            if not stop_matched and out.finish_reason not in get_args(FinishReason):
                logger.warning(
                    f"Model generated unexpected finish_reason: {out.finish_reason}"
                )

            usage = Usage(
                prompt_tokens=int(out.prompt_tokens),
                completion_tokens=completion_tokens,
                total_tokens=int(out.prompt_tokens) + completion_tokens,
                prompt_tokens_details=PromptTokensDetails(
                    cached_tokens=prefix_hit_length
                ),
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=reasoning_tokens
                ),
            )

        if is_done:
            # Update prefix cache BEFORE yielding the final response.
            # Consumers typically break on finish_reason, which prevents
            # the generator from resuming — so any code after yield
            # would never execute.
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
                    kv_prefix_cache.update_kv_cache(
                        matched_index,
                        full_prompt,
                        caches,
                        ssm_snapshots,
                        restore_pos=prefix_hit_length,
                    )
                else:
                    kv_prefix_cache.add_kv_cache(full_prompt, caches, ssm_snapshots)

        yield GenerationResponse(
            text=text,
            token=out.token,
            finish_reason=finish_reason,
            stats=stats,
            usage=usage,
        )

        if is_done:
            mx_barrier(group)
            break

        # Limit accumulated_text to what's needed for stop sequence detection
        if max_stop_len > 0 and len(accumulated_text) > max_stop_len:
            accumulated_text = accumulated_text[-max_stop_len:]
