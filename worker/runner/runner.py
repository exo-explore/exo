import asyncio
import concurrent.futures
from collections.abc import AsyncGenerator
from functools import partial
from typing import Callable, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import stream_generate  # type: ignore
from mlx_lm.tokenizer_utils import TokenizerWrapper

from engines.mlx.utils_mlx import apply_chat_template, initialize_mlx
from shared.openai_compat import FinishReason
from shared.types.tasks import ChatCompletionTaskParams
from shared.types.worker.commands_runner import (
    ChatTaskMessage,
    ExitMessage,
    FinishedResponse,
    GenerationResponse,
    RunnerMessage,
    SetupMessage,
)
from shared.utils import ensure_type
from worker.runner.communication import (
    runner_print,
    runner_read_message,
    runner_write_error,
    runner_write_response,
)


async def _mlx_generate(
    mlx_executor: concurrent.futures.ThreadPoolExecutor,
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    task: ChatCompletionTaskParams,
) -> AsyncGenerator[GenerationResponse]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[GenerationResponse | Exception | object] = asyncio.Queue()
    sentinel = object()

    def _generate_tokens(prompt: str, max_tokens: int) -> None:
        try:
            for generation_response in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
            ):
                response = GenerationResponse(
                    text=generation_response.text,
                    token=generation_response.token,
                    finish_reason=cast(
                        FinishReason | None, generation_response.finish_reason
                    ),  # has to be considered as a FinishReason instead of a str.
                )
                _ = loop.call_soon_threadsafe(queue.put_nowait, response)
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

    max_tokens = task.max_tokens or 100
    generation_fn = partial(_generate_tokens, prompt, max_tokens)

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

        mlx_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_running_loop()

        runner_print(f"got here; {hosts}")

        model, tokenizer, sampler = await loop.run_in_executor(
            mlx_executor,
            partial(initialize_mlx, model_shard_meta=model_shard_meta, hosts=hosts),
        )

        while True:
            message: RunnerMessage = await runner_read_message()
            match message:
                case ChatTaskMessage(task_data=task):
                    runner_print(f"received chat request: {task}")
                    # Ensure we have a chat-completion task subtype
                    # TODO: this is a hack, why are we only looking at the first message? should have a tokenizer
                    prompt = task.messages[0]
                    if prompt.content is not None and 'EXO RUNNER MUST FAIL' in prompt.content:
                        raise Exception('Artificial runner exception - for testing purposes only.')

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
