import asyncio
import concurrent.futures
import os
import resource
from asyncio import AbstractEventLoop
from typing import Any, Callable, cast

from mlx_lm.models.cache import KVCache
from mlx_lm.sample_utils import make_sampler

from exo.worker.runner.utils import get_weights_size

try:
    from mlx_lm.tokenizer_utils import load_tokenizer
except ImportError:
    from mlx_lm.tokenizer_utils import load as load_tokenizer  # type: ignore
from mlx_lm.utils import load_model
from mlx.utils import tree_reduce
from pydantic import RootModel

import mlx.core as mx
import mlx.nn as nn
from exo.engines.mlx import Model, TokenizerWrapper
from exo.engines.mlx.auto_parallel import (
    PipelineParallelisationStrategy,
    TensorParallelisationStrategy,
)
from exo.shared.types.memory import Memory
from exo.shared.types.common import Host
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.communication import runner_print
from exo.shared.types.worker.shards import ShardMetadata
from exo.worker.download.download_utils import build_model_path

# Needed for 8 bit model
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, 4096))

mlx_rank: None | int = None
mlx_world_size: None | int = None


def mx_barrier(group: mx.distributed.Group | None = None):
    mx.eval(
        mx.distributed.all_sum(
            mx.array(1.0),
            stream=mx.default_stream(mx.Device(mx.cpu)),
            group=group,
        )
    )


def broadcast_from_zero(value: int, group: mx.distributed.Group | None = None):
    if mlx_rank is None:
        return value

    if mlx_rank == 0:
        a = mx.array([value], dtype=mx.int32)
    else:
        a = mx.array([0], dtype=mx.int32)

    m = mx.distributed.all_sum(a, stream=mx.Device(mx.DeviceType.cpu), group=group)
    mx.eval(m)
    return int(m.item())


class HostList(RootModel[list[str]]):
    @classmethod
    def from_hosts(cls, hosts: list[Host]) -> "HostList":
        return cls(root=[str(host) for host in hosts])


def mlx_distributed_init(
    rank: int,
    hosts: list[Host] | None = None,
    mlx_ibv_devices: list[list[str | None]] | None = None,
    mlx_ibv_coordinator: str | None = None,
    strict: bool = True,
) -> mx.distributed.Group:
    """
    Initialize the MLX distributed (runs in thread pool).

    Either hosts or mlx_ibv_devices must be provided:
    - hosts: traditional host-based connectivity using MLX_HOSTFILE
    - mlx_ibv_devices: RDMA connectivity matrix using MLX_IBV_DEVICES
    - mlx_ibv_coordinator: coordinator address (IP:PORT) for RDMA setup
    - strict: if True, raise an error if the distributed backend is not available
    """
    runner_print(f"Starting initialization for rank {rank}. Strict: {strict}")

    if mlx_ibv_devices is not None and mlx_ibv_devices != []:
        assert mlx_ibv_coordinator is not None, (
            "To use ibv backend must set ibv coordinator"
        )
        import json

        # Use RDMA connectivity matrix
        devices_file = f"./hosts_{rank}.json"
        ibv_devices_json = json.dumps(mlx_ibv_devices)
        runner_print(f"rank {rank} MLX_IBV_DEVICES: {ibv_devices_json}")
        runner_print(f"rank {rank} MLX_IBV_COORDINATOR: {mlx_ibv_coordinator}")

        with open(devices_file, "w") as f:
            _ = f.write(ibv_devices_json)

        os.environ["MLX_IBV_DEVICES"] = devices_file
        os.environ["MLX_RANK"] = str(rank)
        os.environ["MLX_IBV_COORDINATOR"] = mlx_ibv_coordinator
    elif hosts is not None and hosts != []:
        # Traditional host-based connectivity
        hostfile = f"./hosts_{rank}.json"
        hosts_json = HostList.from_hosts(hosts).model_dump_json()

        runner_print(f"rank {rank} hostfile: {hostfile} hosts: {hosts_json}")

        with open(hostfile, "w") as f:
            _ = f.write(hosts_json)

        os.environ["MLX_HOSTFILE"] = hostfile
        os.environ["MLX_RANK"] = str(rank)
        os.environ["MLX_RING_VERBOSE"] = "1"
    else:
        runner_print("No distributed setup, using single device mode")

    group = mx.distributed.init(
        backend="ring" if hosts is not None else "ibv",
        strict=strict,
    )
    runner_print(f"Rank {rank} mlx distributed initialization complete")

    return group


def initialize_mlx(
    model_shard_meta: ShardMetadata,
    hosts: list[Host] | None = None,
    mlx_ibv_devices: list[list[str | None]] | None = None,
    mlx_ibv_coordinator: str | None = None,
) -> tuple[
    Model, TokenizerWrapper, Callable[[mx.array], mx.array], mx.distributed.Group
]:
    """
    Initialize the MLX model, tokenizer, and sampler. Runs in the MLX thread.

    Either hosts or mlx_ibv_devices must be provided for distributed setups:
    - hosts: traditional host-based connectivity
    - mlx_ibv_devices: RDMA connectivity matrix
    """
    mx.random.seed(42)
    group = mlx_distributed_init(
        model_shard_meta.device_rank,
        hosts=hosts,
        mlx_ibv_devices=mlx_ibv_devices,
        mlx_ibv_coordinator=mlx_ibv_coordinator,
        strict=(hosts is not None and len(hosts) > 1)
        or (mlx_ibv_devices is not None and len(mlx_ibv_devices) > 1),
    )

    set_wired_limit_for_model(get_weights_size(model_shard_meta))

    sampler: Callable[[mx.array], mx.array] = make_sampler(temp=0.7)

    model, tokenizer = shard_and_load(model_shard_meta, group=group)
    model = cast(Model, model)

    return model, tokenizer, sampler, group


def shard_and_load(
    model_shard_meta: ShardMetadata,
    group: mx.distributed.Group,
) -> tuple[nn.Module, TokenizerWrapper]:
    model_path = build_model_path(model_shard_meta.model_meta.model_id)

    runner_print(
        f"loading model from {model_path} with strategy {model_shard_meta.strategy}"
    )

    model, config = load_model(model_path, lazy=True, strict=False)
    runner_print(f"{config=}")
    assert isinstance(model, nn.Module)

    tokenizer = cast(TokenizerWrapper, load_tokenizer(model_path))

    runner_print(f"Group size: {group.size()}, group rank: {group.rank()}")

    match model_shard_meta.strategy:
        case "auto":
            strategy = PipelineParallelisationStrategy(group)
        case "pipeline":
            strategy = PipelineParallelisationStrategy(group)
        case "pipeline_rdma":
            strategy = PipelineParallelisationStrategy(group)
        case "tensor":
            strategy = TensorParallelisationStrategy(group)
        case "tensor_rdma":
            strategy = TensorParallelisationStrategy(group)

    model = strategy.auto_parallel(model, model_shard_meta)

    mx.eval(model.parameters())
    mx.eval(model)

    # Synchronize processes before generation to avoid timeout
    mx_barrier(group)

    return model, tokenizer


async def apply_chat_template(
    mlx_executor: concurrent.futures.ThreadPoolExecutor,
    tokenizer: TokenizerWrapper,
    chat_task_data: ChatCompletionTaskParams,
) -> str:
    loop: AbstractEventLoop = asyncio.get_running_loop()

    # Now we can properly access the messages
    messages = chat_task_data.messages
    messages_dicts: list[dict[str, Any]] = [msg.model_dump() for msg in messages]

    # Filter out None values, keeping relevant keys for the model
    formatted_messages: list[dict[str, Any]] = []
    for message in messages_dicts:
        filtered_message: dict[str, Any] = {
            k: v
            for k, v in message.items()  # pyright: ignore[reportAny]
            if v is not None
        }

        # Verify we have required fields
        if "role" not in filtered_message:
            raise ValueError(f"Message missing 'role' field: {filtered_message}")
        if "content" not in filtered_message and "thinking" not in filtered_message:
            # If neither content nor thinking is present, skip this message
            continue

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


class NullKVCache(KVCache):
    """
    A KVCache that pretends to exist but holds zero tokens.
    It satisfies .state/.meta_state and never allocates real keys/values.
    """

    def __init__(self, dtype: mx.Dtype = mx.float16):
        super().__init__()
        # zero-length K/V so shapes/dtypes are defined but empty
        self.keys = mx.zeros((1, 1, 0, 1), dtype=dtype)
        self.values = mx.zeros((1, 1, 0, 1), dtype=dtype)
        self.offset = 0

    @property
    def state(self) -> tuple[mx.array, mx.array]:
        # matches what mx.save_safetensors / mx.eval expect
        return self.keys, self.values

    @state.setter
    def state(self, v: tuple[mx.array, mx.array]) -> None:
        raise NotImplementedError("We should not be setting a NullKVCache.")


async def make_kv_cache(
    model: Model,
    max_kv_size: int | None = None,
) -> list[KVCache]:
    assert hasattr(model, "layers")
    return [KVCache() for _ in model.layers]


def mlx_force_oom(size: int = 40000) -> None:
    """
    Force an Out-Of-Memory (OOM) error in MLX by performing large tensor operations.
    """
    mx.set_default_device(mx.gpu)
    a = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    b = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    mx.eval(a, b)
    c = mx.matmul(a, b)
    d = mx.matmul(a, c)
    e = mx.matmul(b, c)
    f = mx.sigmoid(d + e)
    mx.eval(f)


def set_wired_limit_for_model(model_size: Memory):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    if not mx.metal.is_available():
        return

    model_bytes = model_size.in_bytes
    max_rec_size = int(mx.metal.device_info()["max_recommended_working_set_size"])
    if model_bytes > 0.9 * max_rec_size:
        model_mb = model_bytes // 2**20
        max_rec_mb = max_rec_size // 2**20
        runner_print(
            f"[WARNING] Generating with a model that requires {model_mb} MB "
            f"which is close to the maximum recommended size of {max_rec_mb} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
        )
    kv_bytes = int(0.02 * model_bytes)
    target_cache = int(1.10 * (model_bytes + kv_bytes))
    target_cache = min(target_cache, max_rec_size)
    mx.set_cache_limit(target_cache)
    mx.set_wired_limit(max_rec_size)
    runner_print(f"Wired limit set to {max_rec_size}. Cache limit set to {target_cache}.")
