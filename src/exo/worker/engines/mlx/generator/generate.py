from typing import Any, Callable, Generator, cast, get_args

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import KVCache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

# from exo.engines.mlx.cache import KVPrefixCache
from exo.shared.types.api import (
    FinishReason,
    GenerationStats,
    TopLogprobItem,
)
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.openai_responses import ResponsesRequest
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
        task_params=ResponsesRequest(
            model=ModelId(""),
            input=content,
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
        prefill_step_size=256,  # Temporarily reduced from 2048 for testing progress bar
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


def extract_top_logprobs(
    logprobs: mx.array,
    tokenizer: TokenizerWrapper,
    top_k: int,
    selected_token: int,
) -> tuple[float, list[TopLogprobItem]]:
    """Extract the selected token's logprob and top-k alternative tokens.

    Args:
        logprobs: Full vocabulary logprobs array from MLX
        tokenizer: Tokenizer for decoding token IDs to strings
        top_k: Number of top alternatives to return
        selected_token: The token ID that was actually sampled

    Returns:
        Tuple of (selected_token_logprob, list of TopLogprobItem for top-k tokens)
    """
    # Get the logprob of the selected token
    selected_logprob = float(logprobs[selected_token].item())

    # Get top-k indices (most probable tokens)
    # mx.argpartition gives indices that would partition the array
    # We negate logprobs since argpartition finds smallest, and we want largest
    top_k = min(top_k, logprobs.shape[0])  # Don't exceed vocab size
    top_indices = mx.argpartition(-logprobs, top_k)[:top_k]

    # Get the actual logprob values for these indices
    top_values = logprobs[top_indices]

    # Sort by logprob (descending) for consistent ordering
    sort_order = mx.argsort(-top_values)
    top_indices = top_indices[sort_order]
    top_values = top_values[sort_order]

    # Convert to list of TopLogprobItem
    top_logprob_items: list[TopLogprobItem] = []
    for i in range(top_k):
        token_id = int(top_indices[i].item())
        token_logprob = float(top_values[i].item())
        # Decode token ID to string
        token_str = tokenizer.decode([token_id])
        # Get byte representation
        token_bytes = list(token_str.encode("utf-8"))
        top_logprob_items.append(
            TopLogprobItem(
                token=token_str,
                logprob=token_logprob,
                bytes=token_bytes,
            )
        )

    return selected_logprob, top_logprob_items


def mlx_generate(
    model: Model,
    tokenizer: TokenizerWrapper,
    task: ResponsesRequest,
    prompt: str,
    is_bench: bool = False,
    on_prefill_progress: Callable[[int, int], None] | None = None,
) -> Generator[GenerationResponse]:
    # Ensure that generation stats only contains peak memory for this generation
    mx.reset_peak_memory()

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
        top_k=task.top_k if task.top_k is not None else 0,
    )

    # Normalize stop sequences to a list
    stop_sequences: list[str] = (
        ([task.stop] if isinstance(task.stop, str) else task.stop)
        if task.stop is not None
        else []
    )
    max_stop_len = max((len(s) for s in stop_sequences), default=0)

    max_tokens = task.max_output_tokens or MAX_TOKENS
    accumulated_text = ""

    for out in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        prompt_cache=caches,
        # TODO: Dynamically change prefill step size to be the maximum possible without timing out.
        prefill_step_size=256,  # Temporarily reduced from 2048 for testing progress bar
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
        prompt_progress_callback=on_prefill_progress,
    ):
        logger.info(out.text)
        accumulated_text += out.text

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
                prompt_tps=float(out.prompt_tps),
                generation_tps=float(out.generation_tps),
                prompt_tokens=int(out.prompt_tokens),
                generation_tokens=int(out.generation_tokens),
                peak_memory_usage=Memory.from_gb(out.peak_memory),
            )
            if not stop_matched and out.finish_reason not in get_args(FinishReason):
                logger.warning(
                    f"Model generated unexpected finish_reason: {out.finish_reason}"
                )

        # Extract logprobs from the full vocabulary logprobs array
        logprob, top_logprobs = extract_top_logprobs(
            logprobs=out.logprobs,
            tokenizer=tokenizer,
            top_k=5,
            selected_token=out.token,
        )

        yield GenerationResponse(
            text=text,
            token=out.token,
            logprob=logprob,
            top_logprobs=top_logprobs,
            finish_reason=finish_reason,
            stats=stats,
        )

        if is_done:
            break

        # Limit accumulated_text to what's needed for stop sequence detection
        if max_stop_len > 0 and len(accumulated_text) > max_stop_len:
            accumulated_text = accumulated_text[-max_stop_len:]

        # TODO: Do we want an mx_barrier?
