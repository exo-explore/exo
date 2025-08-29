import asyncio
import concurrent.futures
import time
from collections.abc import AsyncGenerator
from functools import partial
from typing import Callable, Generator, Optional, Tuple

import mlx.core as mx
from mlx.core import array
from mlx_lm.generate import stream_generate as mlx_stream_generate
from mlx_lm.models import cache
from mlx_lm.models.cache import KVCache

from exo.engines.mlx import Model, TokenizerWrapper
from exo.engines.mlx.utils_mlx import (
    apply_chat_template,
    broadcast_from_zero,
    initialize_mlx,
    make_kv_cache,
    mlx_force_oom,
    mlx_setup,
    mx_barrier,
)
from exo.shared.types.api import ChatCompletionMessage
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.commands_runner import (
    ChatTaskMessage,
    ExitMessage,
    FinishedResponse,
    GenerationResponse,
    InitializedResponse,
    SetupMessage,
    TokenizedResponse,
)
from exo.shared.types.worker.communication import (
    runner_print,
    runner_read_message,
    runner_write_error,
    runner_write_response,
)
from exo.shared.utils import ensure_type
from exo.worker.runner.utils import get_weights_size_kb

generation_stream = mx.new_stream(mx.default_device())

def generate_step(
    prompt: mx.array,
    model: Model,
    *,
    max_tokens: int = 256,
    sampler: Callable[[mx.array], mx.array],
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[list[KVCache]] = None,
    prefill_step_size: int = 2048,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (Model): The model to use for generation.
        max_tokens (int): The maximum number of tokens. Use``-1`` for an infinite
          generator. Default: ``256``.
        sampler (Callable[mx.array, mx.array], optional): A sampler for sampling a
          token from a vector of log probabilities. Default: ``None``.
        max_kv_size (int, optional): Maximum size of the key-value cache. Old
          entries (except the first 4 tokens) will be overwritten.
        prompt_cache (List[Any], optional): A pre-computed prompt cache. Note, if
          provided, the cache will be updated in place.
        prefill_step_size (int): Step size for processing the prompt.

    Yields:
        Tuple[mx.array, mx.array]: One token and a vector of log probabilities.
    """
    tokens = None

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model,
            max_kv_size=max_kv_size,
        )

    def _step(input_tokens: mx.array):
        nonlocal tokens

        with mx.stream(generation_stream):
            logits = model(
                input_tokens[None],
                cache=prompt_cache,
            )

            logits = logits[:, -1, :]

            logprobs = logits - mx.logsumexp(logits, keepdims=True)  # pyright: ignore[reportUnknownMemberType]
            sampled = sampler(logprobs)
            return sampled, logprobs.squeeze(0)

    with mx.stream(generation_stream):
        total_prompt_tokens = len(prompt)
        prompt_processed_tokens = 0

        while total_prompt_tokens - prompt_processed_tokens > prefill_step_size:
            runner_print(f'Prefilling {min(prefill_step_size, len(prompt))} tokens. Remaining tokens: {len(prompt)}. Peak memory: {mx.get_peak_memory() // 2**30} GB')
            logits = model(
                prompt[:prefill_step_size][None],
                cache=prompt_cache
            )

            start_time = time.time()
            mx.eval([c.state for c in prompt_cache] + [logits]) # type: ignore
            eval_time = time.time() - start_time
            prompt_processed_tokens += prefill_step_size

            prompt = prompt[prefill_step_size:]

            mx.clear_cache()
            if eval_time > 7.0:
                prefill_step_size = prefill_step_size // 2
            prefill_step_size = broadcast_from_zero(prefill_step_size)
            prefill_step_size = max(1, prefill_step_size)


        runner_print('finished prefil.')
        y, logprobs = _step(input_tokens=prompt)

    mx.async_eval(y, logprobs) # type: ignore
    n = 0
    next_y: array | None = None
    next_logprobs: array | None = None
    while True:
        if n != max_tokens and n > 0:  # Only call _step after first iteration
            next_y, next_logprobs = _step(y)
            mx.async_eval(next_y, next_logprobs) # type: ignore
        if n == 0:
            mx.eval(y) # type: ignore
        if n == max_tokens:
            break
        yield y, logprobs  # y is always defined here, no need for cast
        if n % 256 == 0:
            mx.clear_cache()
        if next_y is not None and next_logprobs is not None:
            y, logprobs = next_y, next_logprobs
        n += 1



def stream_generate(
    model: Model,
    tokenizer: TokenizerWrapper,
    prompt: str,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    prompt_cache: Optional[list[KVCache]] = None,
    prefill_step_size: int = 2048,
) -> Generator[GenerationResponse, None, None]:

    # Try to infer if special tokens are needed
    add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
        tokenizer.bos_token
    )
    prompt_array: mx.array = mx.array(tokenizer.encode(prompt, add_special_tokens=add_special_tokens))
    runner_write_response(TokenizedResponse(prompt_tokens=len(prompt_array)))

    detokenizer = tokenizer.detokenizer

    token_generator: Generator[Tuple[array, array], None, None] = generate_step(
        prompt_array, 
        model, 
        max_tokens=max_tokens, 
        sampler=sampler,
        prompt_cache=prompt_cache,
        prefill_step_size=prefill_step_size,
    )

    token = None
    detokenizer.reset()
    for token, _ in token_generator:
        if token in tokenizer.eos_token_ids:
            break

        detokenizer.add_token(int(token))

        # TODO: We could put more metrics on this GenerationResponse if we wish
        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=int(token),
            finish_reason=None,
        )

    assert token is not None
    detokenizer.finalize()
    yield GenerationResponse(
        text=detokenizer.last_segment,
        token=int(token),
        finish_reason="stop" if token in tokenizer.eos_token_ids else "length",
    )

async def warmup_inference(
    mlx_executor: concurrent.futures.ThreadPoolExecutor,
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
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
        for _ in mlx_stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=warmup_prompt,
            max_tokens=50,
            sampler=sampler,
        ):
            tokens_generated += 1

    await loop.run_in_executor(mlx_executor, _generate_warmup)
    mx_barrier()

    return tokens_generated

async def _mlx_generate(
    mlx_executor: concurrent.futures.ThreadPoolExecutor,
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    task: ChatCompletionTaskParams,
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
        lambda: asyncio.run(make_kv_cache(
            model=model,
        ))
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


async def main():
    try:
        runner_print("hello from the runner")
        # Get setup info from worker
        init_message = await runner_read_message()
        setup_message = ensure_type(init_message, SetupMessage)
        model_shard_meta = setup_message.model_shard_meta
        hosts = setup_message.hosts

        mlx_setup(int(get_weights_size_kb(model_shard_meta) // 2**10), cache_frac_of_mrwss=0.8, wired_frac_of_mrwss=0.8)

        # For testing - these are fake break conditions
        if model_shard_meta.immediate_exception:
            raise Exception("Fake exception - runner failed to spin up.")
        if model_shard_meta.should_timeout:
            await asyncio.sleep(model_shard_meta.should_timeout)

        setup_start_time = time.time()

        mlx_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_running_loop()

        model, tokenizer, sampler = await loop.run_in_executor(
            mlx_executor,
            partial(initialize_mlx, model_shard_meta=model_shard_meta, hosts=hosts),
        )

        toks = await warmup_inference(
            mlx_executor=mlx_executor,
            model=model,
            tokenizer=tokenizer,
            sampler=sampler,
        )
        runner_print(f"Warmed up by generating {toks} tokens")
        runner_write_response(
            InitializedResponse(time_taken=time.time() - setup_start_time)
        )

        while True:
            message = await runner_read_message()
            match message:
                case ChatTaskMessage(task_data=task):
                    runner_print(f"received chat request: {str(task)[:500]}")
                    # Ensure we have a chat-completion task subtype
                    # TODO: this is a hack, why are we only looking at the first message? should have a tokenizer
                    prompt = task.messages[0]
                    if (
                        prompt.content is not None
                        and "EXO RUNNER MUST FAIL" in prompt.content
                    ):
                        runner_print("raising exception")
                        raise Exception(
                            "Artificial runner exception - for testing purposes only."
                        )
                    if (
                        prompt.content is not None
                        and "EXO RUNNER MUST OOM" in prompt.content
                    ):
                        mlx_force_oom()
                    if (
                        prompt.content is not None
                        and "EXO RUNNER MUST TIMEOUT" in prompt.content
                    ):
                        await asyncio.sleep(100)

                    # Generate responses using the actual MLX generation
                    async for generation_response in _mlx_generate(
                        mlx_executor=mlx_executor,
                        model=model,
                        tokenizer=tokenizer,
                        sampler=sampler,
                        task=task,
                    ):
                        runner_write_response(generation_response)

                    runner_write_response(FinishedResponse())
                case ExitMessage():
                    break
                case _:
                    raise ValueError(f"Unknown message: {message}")

    except Exception as e:
        runner_write_error(e)


if __name__ == "__main__":
    asyncio.run(main())
