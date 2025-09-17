import asyncio
import concurrent.futures
import contextlib
import os
import resource
from asyncio import AbstractEventLoop
from typing import Any, Callable, Optional, cast

from mlx_lm.models.cache import KVCache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper as _TokenizerWrapper
from mlx_lm.tokenizer_utils import load_tokenizer  # type: ignore
from mlx_lm.utils import load_model  # type: ignore
from pydantic import RootModel

import mlx.core as mx
import mlx.nn as nn  # pyright: ignore[reportMissingTypeStubs]
from exo.engines.mlx import Model, TokenizerWrapper
from exo.engines.mlx.auto_parallel import IdentityLayer, auto_parallel
from exo.shared.types.common import Host
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.communication import runner_print
from exo.shared.types.worker.shards import ShardMetadata
from exo.worker.download.download_utils import build_model_path

# Needed for 8 bit model
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, 4096))

mlx_rank: None | int = None
mlx_world_size: None | int = None

def mx_barrier():
    mx.eval(  # type: ignore
        mx.distributed.all_sum(
            mx.array(1.0), stream=mx.default_stream(mx.Device(mx.cpu))
        )
    )

def broadcast_from_zero(value: int) -> int:
    if mlx_rank is None:
        return value

    if mlx_rank == 0:
        a = mx.array([value], dtype=mx.int32)
    else:
        a = mx.array([0], dtype=mx.int32)

    m = mx.distributed.all_sum(a, stream=mx.Device(mx.DeviceType.cpu))
    mx.eval(m) # type: ignore
    return int(m.item()) # type: ignore

class HostList(RootModel[list[str]]):
    @classmethod
    def from_hosts(cls, hosts: list[Host]) -> "HostList":
        return cls(root=[str(host) for host in hosts])


def mlx_setup(
    model_size_mb: int,
    cache_frac_of_mrwss: float = 0.65,  # main workhorse
    wired_frac_of_mrwss: float = 0.00,  # start with no wiring
) -> None:
    info = mx.metal.device_info()
    mrwss = int(info["max_recommended_working_set_size"])  # bytes
    memsize = int(info["memory_size"])  # bytes

    runner_print(f"model size mb {model_size_mb}")
    runner_print(f"{mrwss=}")
    runner_print(f"{memsize=}")

    model_bytes = int(model_size_mb * 1024**2)
    kv_bytes = int(0.02 * model_bytes)

    # Cache: keep most of weights+KV “on ice”, but don’t starve the OS.
    target_cache = int(1.10 * (model_bytes + kv_bytes))  # +10% slack
    target_cache = min(target_cache, int(cache_frac_of_mrwss * mrwss))
    target_cache = min(target_cache, memsize)

    runner_print(f"{target_cache=}")
    mx.set_cache_limit(max(target_cache, 0))

    # Wiring: off by default; if you re‑enable, wire at most a small fraction.
    if wired_frac_of_mrwss > 0.0:
        target_wired = int(wired_frac_of_mrwss * mrwss)
        target_wired = min(target_wired, target_cache)  # don’t wire more than cache
        
        runner_print(f"{target_wired=}")
        with contextlib.suppress(Exception):  # older macOS won’t have this
            mx.set_wired_limit(max(target_wired, 0))


def mlx_distributed_init(rank: int, hosts: list[Host]) -> mx.distributed.Group:  # type: ignore
    """
    Initialize the MLX distributed (runs in thread pool)
    """
    global mlx_rank, mlx_world_size
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

    group = mx.distributed.init(backend="ring", strict=True)
    mlx_rank = group.rank()
    mlx_world_size = group.rank()
    runner_print(f"Rank {rank} mlx distributed initialization complete")

    return group


def initialize_mlx(
    model_shard_meta: ShardMetadata,
    hosts: list[Host],
) -> tuple[Model, TokenizerWrapper, Callable[[mx.array], mx.array]]:
    """
    Initialize the MLX model, tokenizer, and sampler. Runs in the MLX thread.
    """
    mx.random.seed(42)
    if len(hosts) > 1:
        mlx_distributed_init(model_shard_meta.device_rank, hosts)
    sampler: Callable[[mx.array], mx.array] = make_sampler(temp=0.7)

    model, tokenizer = shard_and_load(model_shard_meta)
    model = cast(Model, model)

    return model, tokenizer, sampler


def shard_and_load(
    model_shard_meta: ShardMetadata, 
) -> tuple[nn.Module, TokenizerWrapper]:
    model_path = build_model_path(model_shard_meta.model_meta.model_id)

    runner_print(f"loading model from {model_path}")

    model, config = load_model(model_path, lazy=True, strict=False)  # type: ignore
    runner_print(f'{config=}')
    assert isinstance(model, nn.Module)

    tokenizer = load_tokenizer(model_path)
    assert isinstance(tokenizer, _TokenizerWrapper)
    model = auto_parallel(model, model_shard_meta)
    mx.eval(model.parameters())  # type: ignore

    # Synchronize processes before generation to avoid timeout
    mx_barrier()

    return model, tokenizer # type: ignore


async def apply_chat_template(
    mlx_executor: concurrent.futures.ThreadPoolExecutor,
    tokenizer: TokenizerWrapper,
    chat_task_data: ChatCompletionTaskParams,
) -> str:
    loop: AbstractEventLoop = asyncio.get_running_loop()

    # Now we can properly access the messages
    messages = chat_task_data.messages
    messages_dicts = [msg.model_dump() for msg in messages]

    # Filter out None values, keeping relevant keys for the model
    formatted_messages = []
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

        formatted_messages.append(filtered_message)  # type: ignore

    messages_dicts = formatted_messages

    prompt: str = await loop.run_in_executor(
        executor=mlx_executor,
        func=lambda: tokenizer.apply_chat_template(  # type: ignore
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
        self.keys = mx.zeros((1, 1, 0, 1), dtype=dtype)  # pyright: ignore[reportUnknownMemberType]
        self.values = mx.zeros((1, 1, 0, 1), dtype=dtype)  # pyright: ignore[reportUnknownMemberType]
        self.offset = 0

    @property
    def state(self) -> tuple[mx.array, mx.array]:
        # matches what mx.save_safetensors / mx.eval expect
        return self.keys, self.values

    @state.setter
    def state(self, v: tuple[mx.array, mx.array]) -> None:
        raise NotImplementedError('We should not be setting a NullKVCache.')

async def make_kv_cache(
    model: Model,
    max_kv_size: Optional[int] = None,
) -> list[KVCache]:
    assert hasattr(model, 'layers')
    
    return [
        NullKVCache() if isinstance(layer, IdentityLayer) else KVCache()
        for layer in model.layers
    ]

def mlx_force_oom(size: int = 40000) -> None:
    """
    Force an Out-Of-Memory (OOM) error in MLX by performing large tensor operations.
    """
    mx.set_default_device(mx.gpu)  # type: ignore
    a = mx.random.uniform(shape=(size, size), dtype=mx.float32)  # type: ignore
    b = mx.random.uniform(shape=(size, size), dtype=mx.float32)  # type: ignore
    mx.eval(a, b)  # type: ignore
    c = mx.matmul(a, b)  # type: ignore
    d = mx.matmul(a, c)  # type: ignore
    e = mx.matmul(b, c)  # type: ignore
    f = mx.sigmoid(d + e)  # type: ignore
    mx.eval(f)  # type: ignore
