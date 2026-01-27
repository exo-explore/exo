import time
from typing import Any, Callable, Generator, cast, get_args

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import trim_prompt_cache, KVCache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.api import (
    BenchChatCompletionTaskParams,
    ChatCompletionMessage,
    FinishReason,
    GenerationStats,
    TopLogprobItem,
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
    make_kv_cache,
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


def score_tokens_batched(
    model: Model,
    tokenizer: TokenizerWrapper,
    token_sequences: list[list[int]],
    top_k: int | None = None,
) -> list[list[tuple[float, list[TopLogprobItem]]]]:
    """Score multiple token sequences in a single batched forward pass.

    This is significantly faster than calling score_tokens() multiple times
    because it batches the forward pass across all sequences.

    Args:
        model: The MLX model.
        tokenizer: The tokenizer.
        token_sequences: List of token ID sequences to score.
        top_k: Number of top logprobs to return per position.

    Returns:
        List of results for each sequence. Each result is a list of
        (token_logprob, top_logprobs) tuples for each token position.
    """
    if not token_sequences:
        return []

    # Handle empty sequences and single-token sequences
    results: list[list[tuple[float, list[TopLogprobItem]]]] = []
    non_empty_indices: list[int] = []
    non_empty_sequences: list[list[int]] = []

    for i, tokens in enumerate(token_sequences):
        if len(tokens) == 0:
            results.append([])
        elif len(tokens) == 1:
            results.append([(0.0, [])])
        else:
            results.append([])  # Placeholder, will be filled later
            non_empty_indices.append(i)
            non_empty_sequences.append(tokens)

    if not non_empty_sequences:
        return results

    # Find max sequence length (excluding last token since we predict it)
    max_len = max(len(seq) - 1 for seq in non_empty_sequences)

    # Get pad token (use eos_token_id or 0)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0)

    # Pad sequences and create attention mask
    batch_size = len(non_empty_sequences)
    padded_inputs = mx.full((batch_size, max_len), pad_token_id, dtype=mx.int32)
    seq_lengths: list[int] = []

    for i, tokens in enumerate(non_empty_sequences):
        input_len = len(tokens) - 1  # Exclude last token
        padded_inputs[i, :input_len] = mx.array(tokens[:-1], dtype=mx.int32)
        seq_lengths.append(input_len)

    # Run batched forward pass (no KV cache for scoring)
    # The model accepts [batch_size, seq_len] and returns [batch_size, seq_len, vocab_size]
    logits = model(padded_inputs, cache=None)

    # Convert to log probabilities - logits shape: [batch, seq_len, vocab]
    logprobs_all = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(logprobs_all)

    # Extract results for each sequence
    for batch_idx, (orig_idx, tokens, seq_len) in enumerate(
        zip(non_empty_indices, non_empty_sequences, seq_lengths, strict=True)
    ):
        seq_results: list[tuple[float, list[TopLogprobItem]]] = [(0.0, [])]

        for pos in range(seq_len):
            next_token = tokens[pos + 1]
            logprobs_at_position: mx.array = logprobs_all[batch_idx, pos]

            logprob, top_logprobs_items = extract_top_logprobs(
                logprobs_array=logprobs_at_position,
                selected_token=next_token,
                tokenizer=tokenizer,
                top_k=top_k,
            )
            seq_results.append((logprob, top_logprobs_items))

        results[orig_idx] = seq_results

    return results


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

    sampler = make_sampler(
        temp=task.temperature if task.temperature is not None else 0.7,
        top_p=task.top_p if task.top_p is not None else 1.0,
    )

    # Prefill cache with all tokens except the last one
    prefill_tps = prefill(model, tokenizer, sampler, prompt_tokens[:-1], caches)

    # stream_generate starts from the last token
    last_token = prompt_tokens[-1:]

    # Determine if we need logprobs
    should_extract_logprobs = task.logprobs is True
    top_k = task.top_logprobs if task.top_logprobs is not None else 0

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
