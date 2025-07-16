import asyncio
import concurrent.futures
from asyncio.events import AbstractEventLoop
from collections.abc import AsyncGenerator
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from typing import Callable, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import stream_generate  # type: ignore
from mlx_lm.tokenizer_utils import TokenizerWrapper

from engines.mlx.utils_mlx import apply_chat_template, initialize_mlx
from shared.openai import FinishReason
from shared.types.tasks.common import (
    TaskData,
)
from shared.types.worker.commands_runner import (
    ChatTaskMessage,
    ExitMessage,
    FinishedResponse,
    GenerationResponse,
    RunnerMessage,
    SetupMessage,
)
from shared.types.worker.mlx import Host
from shared.types.worker.shards import ShardMeta
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
    task: TaskData,
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
    task_data = task.task_data

    runner_print(f"task_data: {task_data}")

    prompt = await apply_chat_template(
        mlx_executor=mlx_executor,
        tokenizer=tokenizer,
        chat_task=task_data,
    )

    max_tokens = task_data.max_tokens or 100
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
        yield item

    assert future.done()


async def main():
    try:
        runner_print("hello from the runner")

        # Get setup info from worker
        init_message: RunnerMessage = await runner_read_message()
        setup_message: SetupMessage = ensure_type(init_message, SetupMessage)
        model_shard_meta: ShardMeta = setup_message.model_shard_meta
        hosts: list[Host] = setup_message.hosts

        mlx_executor: ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        )
        loop: AbstractEventLoop = asyncio.get_running_loop()

        runner_print(f"got here; {model_shard_meta.model_path}")

        model, tokenizer, sampler = await loop.run_in_executor(
            mlx_executor,
            partial(initialize_mlx, model_shard_meta=model_shard_meta, hosts=hosts),
        )

        while True:
            message: RunnerMessage = await runner_read_message()
            match message:
                case ChatTaskMessage(task=task_data):
                    runner_print(f"received chat request: {task_data}")

                    # Ensure we have a chat-completion task subtype
                    messages = task_data.task_data.messages
                    messages_dicts = [msg.model_dump() for msg in messages]
                    runner_print(f"messages_dicts RUNNER: {messages_dicts}")

                    # Generate responses using the actual MLX generation
                    async for generation_response in _mlx_generate(
                        mlx_executor=mlx_executor,
                        model=model,
                        tokenizer=tokenizer,
                        sampler=sampler,
                        task=task_data,
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
