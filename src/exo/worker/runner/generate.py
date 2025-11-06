import asyncio
import concurrent.futures
import functools
import time
from collections.abc import AsyncGenerator
from functools import partial
from typing import Any, Callable, Generator, Optional, Tuple

import mlx.core as mx
from mlx.core import array
from mlx_lm.models import cache
from mlx_lm.models.cache import KVCache

from exo.engines.mlx import Model, TokenizerWrapper
from exo.engines.mlx.utils_mlx import (
    apply_chat_template,
    broadcast_from_zero,
    make_kv_cache,
    mx_barrier,
)
from exo.shared.types.api import ChatCompletionMessage
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.commands_runner import (
    GenerationResponse,
    RunnerMessage,
    RunnerResponse,
    TokenizedResponse,
)
from exo.shared.types.worker.communication import (
    AsyncConnection,
    runner_print,
)

generation_stream = mx.new_stream(mx.default_device())


def maybe_quantize_kv_cache(
    prompt_cache: list[Any],
    quantized_kv_start: int,
    kv_group_size: int,
    kv_bits: int | None,
) -> None:
    if kv_bits is None:
        return
    for e, c in enumerate(prompt_cache):  # type: ignore[type-arg]
        if hasattr(c, "to_quantized") and c.offset >= quantized_kv_start:  # type: ignore[type-arg]
            prompt_cache[e] = c.to_quantized(group_size=kv_group_size, bits=kv_bits)  # type: ignore[type-arg]


def generate_step(
    prompt: mx.array,
    model: Model,
    *,
    max_tokens: int = 256,
    sampler: Callable[[mx.array], mx.array],
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] | None = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[list[KVCache]] = None,
    prefill_step_size: int = 2048,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    prompt_progress_callback: Callable[[int, int], None] | None = None,
    input_embeddings: mx.array | None = None,
    group: mx.distributed.Group | None = None,
) -> Generator[Tuple[int, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (Model): The model to use for generation.
        max_tokens (int): The maximum number of tokens. Use``-1`` for an infinite
          generator. Default: ``256``.
        sampler (Callable[mx.array, mx.array]): A sampler for sampling a
          token from a vector of log probabilities.
        logits_processors (List[Callable[[mx.array, mx.array], mx.array]], optional):
          A list of functions that take tokens and logits and return the processed
          logits. Default: ``None``.
        max_kv_size (int, optional): Maximum size of the key-value cache. Old
          entries (except the first 4 tokens) will be overwritten.
        prompt_cache (List[Any], optional): A pre-computed prompt cache. Note, if
          provided, the cache will be updated in place.
        prefill_step_size (int): Step size for processing the prompt.
        kv_bits (int, optional): Number of bits to use for KV cache quantization.
          None implies no cache quantization. Default: ``None``.
        kv_group_size (int): Group size for KV cache quantization. Default: ``64``.
        quantized_kv_start (int): Step to begin using a quantized KV cache.
           when ``kv_bits`` is non-None. Default: ``0``.
        prompt_progress_callback (Callable[[int, int], None]): A call-back which takes the
           prompt tokens processed so far and the total number of prompt tokens.
        input_embeddings (mx.array, optional): Input embeddings to use instead of or in
          conjunction with prompt tokens. Default: ``None``.

    Yields:
        Tuple[int, mx.array]: One token and a vector of log probabilities.
    """
    if input_embeddings is not None:
        if len(prompt) > 0 and len(prompt) != len(input_embeddings):
            raise ValueError(
                f"When providing input_embeddings, their sequence length ({len(input_embeddings)}) "
                f"must match the sequence length of the prompt ({len(prompt)}), or the "
                "prompt must be empty."
            )
    elif len(prompt) == 0:
        raise ValueError(
            "Either input_embeddings or prompt (or both) must be provided."
        )

    tokens = None

    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model,
            max_kv_size=max_kv_size,
        )

    prompt_progress_callback = prompt_progress_callback or (lambda *_: None)  # type: ignore[type-arg]

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    def _model_call(
        input_tokens: mx.array, input_embeddings: mx.array | None
    ) -> mx.array:
        if input_embeddings is not None:
            return model(  # type: ignore[type-arg]
                input_tokens,
                cache=prompt_cache,
                input_embeddings=input_embeddings,  # type: ignore[type-arg]
            )
        else:
            return model(input_tokens, cache=prompt_cache)

    def _step(
        input_tokens: mx.array, input_embeddings: mx.array | None = None
    ) -> tuple[mx.array, mx.array]:
        nonlocal tokens

        with mx.stream(generation_stream):
            logits = _model_call(
                input_tokens=input_tokens[None],
                input_embeddings=(
                    input_embeddings[None] if input_embeddings is not None else None
                ),
            )

            logits = logits[:, -1, :]

            if logits_processors and len(input_tokens) > 0:
                tokens = (
                    mx.concat([tokens, input_tokens])
                    if tokens is not None
                    else input_tokens
                )
                for processor in logits_processors:
                    logits = processor(tokens, logits)

            quantize_cache_fn(prompt_cache)

            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            sampled = sampler(logprobs)
            return sampled, logprobs.squeeze(0)

    with mx.stream(generation_stream):
        total_prompt_tokens = (
            len(input_embeddings) if input_embeddings is not None else len(prompt)
        )
        prompt_processed_tokens = 0
        prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)

        while total_prompt_tokens - prompt_processed_tokens > prefill_step_size:
            runner_print(
                f"Prefilling {min(prefill_step_size, len(prompt))} tokens. Remaining tokens: {len(prompt)}. Peak memory: {mx.get_peak_memory() // 2**30} GB"
            )
            n_to_process = min(prefill_step_size, prompt.size)
            _model_call(
                input_tokens=prompt[:n_to_process][None],
                input_embeddings=(
                    input_embeddings[:n_to_process][None]
                    if input_embeddings is not None
                    else None
                ),
            )
            quantize_cache_fn(prompt_cache)

            start_time = time.time()
            mx.eval([c.state for c in prompt_cache])  # type: ignore
            eval_time = time.time() - start_time
            prompt_processed_tokens += n_to_process

            prompt = prompt[n_to_process:]
            input_embeddings = (
                input_embeddings[n_to_process:]
                if input_embeddings is not None
                else input_embeddings
            )

            mx.clear_cache()
            if eval_time > 7.0:
                prefill_step_size = prefill_step_size // 2
            if group is not None:
                prefill_step_size = broadcast_from_zero(prefill_step_size)
            prefill_step_size = max(1, prefill_step_size)
            prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)

        if prompt_processed_tokens > 0:
            runner_print("finished prefil stage.")

        y, logprobs = _step(input_tokens=prompt, input_embeddings=input_embeddings)

    mx.async_eval(y, logprobs)
    next_y: array | None = None
    next_logprobs: array | None = None
    n = 0
    while True:
        if n != max_tokens:
            assert y is not None
            next_y, next_logprobs = _step(y)
            mx.async_eval(next_y, next_logprobs)
        if n == 0:
            mx.eval(y)  # type: ignore[type-arg]
            prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
        if n == max_tokens:
            break
        yield int(y.item()), logprobs  # type: ignore
        if n % 256 == 0:
            mx.clear_cache()
        y, logprobs = next_y, next_logprobs
        n += 1


def stream_generate(
    model: Model,
    tokenizer: TokenizerWrapper,
    prompt: str,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    conn: AsyncConnection[RunnerResponse, RunnerMessage] | None,
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] | None = None,
    max_kv_size: int | None = None,
    prompt_cache: Optional[list[KVCache]] = None,
    prefill_step_size: int = 2048,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    prompt_progress_callback: Callable[[int, int], None] | None = None,
    input_embeddings: mx.array | None = None,
    group: mx.distributed.Group | None = None,
) -> Generator[GenerationResponse, None, None]:
    # Try to infer if special tokens are needed
    add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
        tokenizer.bos_token
    )
    prompt_array: mx.array = mx.array(
        tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    )
    if conn is not None:
        conn.send_sync(TokenizedResponse(prompt_tokens=len(prompt_array)))

    detokenizer = tokenizer.detokenizer

    token_generator: Generator[Tuple[int, array], None, None] = generate_step(
        prompt_array,
        model,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        max_kv_size=max_kv_size,
        prompt_cache=prompt_cache,
        prefill_step_size=prefill_step_size,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
        prompt_progress_callback=prompt_progress_callback,
        input_embeddings=input_embeddings,
        group=group,
    )

    token = None
    detokenizer.reset()
    for token, _ in token_generator:
        if token in tokenizer.eos_token_ids:
            break

        detokenizer.add_token(token)

        # TODO: We could put more metrics on this GenerationResponse if we wish
        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=token,
            finish_reason=None,
        )

    assert token is not None
    detokenizer.finalize()
    yield GenerationResponse(
        text=detokenizer.last_segment,
        token=token,
        finish_reason="stop" if token in tokenizer.eos_token_ids else "length",
    )


async def warmup_inference(
    mlx_executor: concurrent.futures.ThreadPoolExecutor,
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    group: mx.distributed.Group | None = None,
) -> int:
    loop = asyncio.get_running_loop()

    warmup_prompt = await apply_chat_template(
        mlx_executor=mlx_executor,
        tokenizer=tokenizer,
        chat_task_data=ChatCompletionTaskParams(
            model="warmup",
            messages=[
                ChatCompletionMessage(
                    role="user",
                    content="Prompt to warm up the inference engine. Repeat this.",
                )
            ],
        ),
    )

    tokens_generated = 0

    def _generate_warmup():
        nonlocal tokens_generated
        runner_print("Generating warmup tokens")
        for _r in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=warmup_prompt,
            max_tokens=50,
            sampler=sampler,
            conn=None,
            group=group,
        ):
            runner_print("Generated warmup token: " + str(_r.text))
            tokens_generated += 1

    await loop.run_in_executor(mlx_executor, _generate_warmup)
    runner_print("Generated ALL warmup tokens")
    mx_barrier()

    return tokens_generated


async def mlx_generate(
    mlx_executor: concurrent.futures.ThreadPoolExecutor,
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    task: ChatCompletionTaskParams,
    conn: AsyncConnection[RunnerResponse, RunnerMessage],
) -> AsyncGenerator[GenerationResponse]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[GenerationResponse | Exception | object] = asyncio.Queue()
    sentinel = object()

    def _generate_tokens(prompt: str, max_tokens: int, cache: list[KVCache]) -> None:
        try:
            for generation_response in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                prompt_cache=cache,
                prefill_step_size=1024,
                conn=conn,
            ):
                _ = loop.call_soon_threadsafe(queue.put_nowait, generation_response)
        except Exception as e:
            _ = loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
            _ = loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    # Currently we support chat-completion tasks only.
    runner_print(f"task_params: {task}")

    prompt = await apply_chat_template(
        mlx_executor=mlx_executor,
        tokenizer=tokenizer,
        chat_task_data=task,
    )

    cache_future = loop.run_in_executor(
        mlx_executor,
        lambda: asyncio.run(
            make_kv_cache(
                model=model,
            )
        ),
    )
    cache = await cache_future

    max_tokens = task.max_tokens or 1000
    generation_fn = partial(_generate_tokens, prompt, max_tokens, cache)

    future = loop.run_in_executor(mlx_executor, generation_fn)

    while True:
        item = await queue.get()
        queue.task_done()

        if item is sentinel:
            break

        if isinstance(item, Exception):
            raise item

        assert isinstance(item, GenerationResponse)  # constrain datatype
        runner_print(item.text)
        yield item

    # Wait for the executor thread to complete
    await future
