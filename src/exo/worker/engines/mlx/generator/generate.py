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
    TopLogprobItem,
)
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
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


def extract_top_logprobs(
    logprobs_array: mx.array,
    selected_token: int,
    tokenizer: TokenizerWrapper,
    top_k: int | None,
) -> tuple[float, list[TopLogprobItem]]:
    """Extract the selected token's logprob and top-k alternatives.

    top k an be set to None to return all the logprobs
    """
    selected_logprob = float(logprobs_array[selected_token].item())

    if top_k == 0:
        return selected_logprob, []

    vocab_size = logprobs_array.shape[0]

    if top_k is None:
        sorted_indices = mx.argsort(-logprobs_array)
        mx.eval(sorted_indices)
        indices_list: list[int] = cast(list[int], sorted_indices.tolist())
    else:
        k = min(top_k, vocab_size)
        top_indices = mx.argpartition(-logprobs_array, kth=k - 1)[:k]
        top_logprobs_values = logprobs_array[top_indices]
        sorted_order = mx.argsort(-top_logprobs_values)
        top_indices = top_indices[sorted_order]
        mx.eval(top_indices)
        indices_list = cast(list[int], top_indices.tolist())

    top_logprob_items: list[TopLogprobItem] = []
    for token_id in indices_list:
        logprob_value = float(logprobs_array[token_id].item())
        token_str = tokenizer.decode([token_id])

        top_logprob_items.append(
            TopLogprobItem(
                token=token_str,
                logprob=logprob_value,
                bytes=list(token_str.encode("utf-8")),
            )
        )

    return selected_logprob, top_logprob_items


def score_tokens(
    model: Model,
    tokenizer: TokenizerWrapper,
    tokens: list[int],
    top_k: int | None = None,
) -> list[tuple[float, list[TopLogprobItem]]]:
    """Score a sequence of tokens, returning logprobs for each token.

    This is used for the completions API with echo=True, where we need
    logprobs for the prompt tokens (not just generated tokens).

    Args:
        model: The MLX model.
        tokenizer: The tokenizer.
        tokens: List of token IDs to score.
        top_k: Number of top logprobs to return per position.
               If None, returns all logprobs.

    Returns:
        List of (token_logprob, top_logprobs) tuples for each token position.
        The first position has no logprob (no previous context), so returns (0.0, []).
    """
    if len(tokens) == 0:
        return []

    # First token has no previous context to condition on
    results: list[tuple[float, list[TopLogprobItem]]] = [(0.0, [])]

    if len(tokens) == 1:
        return results

    # Create an empty KV cache for the forward pass
    cache = make_kv_cache(model=model)

    # Convert to MLX array and run forward pass
    input_tokens = mx.array(tokens[:-1])[None]  # All tokens except last, batched

    # Run the model to get logits for all positions
    # The model returns logits with shape [1, seq_len, vocab_size]
    logits: mx.array = model(input_tokens, cache=cast(list[KVCache], cache))
    logits = logits.squeeze(0)  # Shape: [seq_len, vocab_size]

    # Convert to log probabilities
    logprobs_all: mx.array = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    mx.eval(logprobs_all)

    # For each position, extract the logprob of the actual next token
    for i in range(len(tokens) - 1):
        next_token = tokens[i + 1]
        logprobs_at_position: mx.array = logprobs_all[i]

        logprob, top_logprobs_items = extract_top_logprobs(
            logprobs_array=logprobs_at_position,
            selected_token=next_token,
            tokenizer=tokenizer,
            top_k=top_k,
        )
        results.append((logprob, top_logprobs_items))

    return results


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

    # Determine if we need logprobs
    should_extract_logprobs = task.logprobs is True
    top_k = task.top_logprobs if task.top_logprobs is not None else 0

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

        # Extract logprobs if requested
        logprob: float | None = None
        top_logprobs: list[TopLogprobItem] | None = None
        if should_extract_logprobs:
            logprob, top_logprobs = extract_top_logprobs(
                logprobs_array=out.logprobs,
                selected_token=out.token,
                tokenizer=tokenizer,
                top_k=top_k,
            )

        yield GenerationResponse(
            text=out.text,
            token=out.token,
            logprob=logprob,
            top_logprobs=top_logprobs,
            finish_reason=cast(FinishReason | None, out.finish_reason),
            stats=stats,
        )

        if out.finish_reason is not None:
            break

        # TODO: Do we want an mx_barrier?
