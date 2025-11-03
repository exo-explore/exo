import asyncio
import concurrent.futures
import time
from functools import partial
from multiprocessing.connection import Connection

from exo.engines.mlx.utils_mlx import (
    initialize_mlx,
    mlx_force_oom,
    mlx_setup,
)
from exo.shared.global_conn import set_conn
from exo.shared.types.worker.commands_runner import (
    ChatTaskMessage,
    ExitMessage,
    FinishedResponse,
    InitializedResponse,
    RunnerMessage,
    RunnerResponse,
    SetupMessage,
)
from exo.shared.types.worker.communication import (
    AsyncConnection,
    runner_print,
    runner_write_error,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils import ensure_type
from exo.worker.runner.generate import mlx_generate, warmup_inference
from exo.worker.runner.utils import get_weights_size


async def main(raw_conn: Connection):
    conn = AsyncConnection[RunnerResponse, RunnerMessage](raw_conn)
    set_conn(conn)

    try:
        runner_print("hello from the runner")
        init_message = await conn.recv()
        setup_message = ensure_type(init_message, SetupMessage)
        model_shard_meta: ShardMetadata = setup_message.model_shard_meta
        hosts = setup_message.hosts

        if getattr(model_shard_meta, "immediate_exception", False):
            raise Exception("Fake exception - runner failed to spin up.")
        if timeout := getattr(model_shard_meta, "should_timeout", 0):
            await asyncio.sleep(timeout)

        mlx_setup(
            int(get_weights_size(model_shard_meta).in_kb // 2**10),
            cache_frac_of_mrwss=0.8,
            wired_frac_of_mrwss=0.8,
        )

        setup_start_time = time.time()

        mlx_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_running_loop()

        model, tokenizer, sampler, group = await loop.run_in_executor(  # type: ignore[type-arg]
            mlx_executor,
            partial(
                initialize_mlx,
                model_shard_meta=model_shard_meta,
                hosts=hosts,
            ),
        )

        runner_print(
            f"Warming up inference for model_shard_meta: {model_shard_meta} hosts: {hosts}"
        )
        toks = await warmup_inference(
            mlx_executor=mlx_executor,
            model=model,
            tokenizer=tokenizer,
            sampler=sampler,
        )
        runner_print(f"Warmed up by generating {toks} tokens")
        await conn.send(InitializedResponse(time_taken=time.time() - setup_start_time))

        while True:
            message = await conn.recv()
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
                    async for generation_response in mlx_generate(
                        mlx_executor=mlx_executor,
                        model=model,
                        tokenizer=tokenizer,
                        sampler=sampler,
                        task=task,
                        conn=conn,
                    ):
                        await conn.send(generation_response)

                    await conn.send(FinishedResponse())
                case ExitMessage():
                    break
                case _:
                    raise ValueError(f"Unknown message: {message}")

    except Exception as e:
        runner_write_error(e)
