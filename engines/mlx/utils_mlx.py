# type: ignore


import asyncio
import concurrent.futures
import os
from asyncio import AbstractEventLoop
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper, load_tokenizer
from mlx_lm.utils import load_model
from pydantic import RootModel

from engines.mlx.auto_parallel import auto_parallel
from shared.types.tasks.common import ChatCompletionTaskParams
from shared.types.worker.mlx import Host
from shared.types.worker.shards import ShardMeta
from worker.runner.communication import runner_print


def mx_barrier():
    mx.eval(
        mx.distributed.all_sum(
            mx.array(1.0), stream=mx.default_stream(mx.Device(mx.cpu))
        )
    )


class HostList(RootModel[list[str]]):
    @classmethod
    def from_hosts(cls, hosts: list[Host]) -> "HostList":
        return cls(root=[str(host) for host in hosts])


def mlx_distributed_init(rank: int, hosts: list[Host]) -> mx.distributed.Group:
    """
    Initialize the MLX distributed (runs in thread pool)
    """
    runner_print(f"Starting initialization for rank {rank}")

    # Setup distributed environment
    hostfile = f"./hosts_{rank}.json"  # TODO: this needs to be unique?
    hosts_json = HostList.from_hosts(hosts).model_dump_json()

    runner_print(f"rank {rank} hostfile: {hostfile} hosts: {hosts_json}")

    with open(hostfile, "w") as f:
        _ = f.write(hosts_json)

    os.environ["MLX_HOSTFILE"] = hostfile
    os.environ["MLX_RANK"] = str(rank)
    os.environ["MLX_RING_VERBOSE"] = "1"

    # Initialize distributed
    group = mx.distributed.init(backend="ring", strict=True)
    runner_print(f"Rank {rank} mlx distributed initialization complete")

    return group


def initialize_mlx(
    model_shard_meta: ShardMeta,
    hosts: list[Host],
) -> tuple[nn.Module, TokenizerWrapper, Callable[[mx.array], mx.array]]:
    """
    Initialize the MLX model, tokenizer, and sampler. Runs in the MLX thread.
    """
    mx.random.seed(42)
    if len(hosts) > 1:
        mlx_distributed_init(model_shard_meta.device_rank, hosts)
    sampler: Callable[[mx.array], mx.array] = make_sampler(temp=0.7)

    model, tokenizer = shard_and_load(model_shard_meta)

    return model, tokenizer, sampler


def shard_and_load(model_shard_meta: ShardMeta) -> tuple[nn.Module, TokenizerWrapper]:
    runner_print(f"loading model from {model_shard_meta.model_path}")

    model, config = load_model(model_shard_meta.model_path, lazy=True, strict=False)

    tokenizer = load_tokenizer(model_shard_meta.model_path)
    assert isinstance(tokenizer, TokenizerWrapper)
    model = auto_parallel(model, model_shard_meta)

    # Synchronize processes before generation to avoid timeout
    mx_barrier()

    return model, tokenizer


async def apply_chat_template(
    mlx_executor: concurrent.futures.ThreadPoolExecutor,
    tokenizer: TokenizerWrapper,
    chat_task_data: ChatCompletionTaskParams,
) -> str:
    loop: AbstractEventLoop = asyncio.get_running_loop()

    # Now we can properly access the messages
    messages = chat_task_data.messages
    messages_dicts = [msg.model_dump() for msg in messages]

    # Filter out None values, keeping only 'role' and 'content' keys
    formatted_messages = []
    for message in messages_dicts:
        filtered_message = {k: v for k, v in message.items() if v is not None}
        # Verify we have exactly the expected keys
        assert set(filtered_message.keys()) == {"role", "content"}, (
            f"Expected only 'role' and 'content' keys, got: {filtered_message.keys()}"
        )
        formatted_messages.append(filtered_message)

    messages_dicts = formatted_messages

    prompt: str = await loop.run_in_executor(
        executor=mlx_executor,
        func=lambda: tokenizer.apply_chat_template(
            messages_dicts,
            tokenize=False,
            add_generation_prompt=True,
        ),
    )

    return prompt
