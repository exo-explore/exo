import os
import resource
from typing import Any, Callable, cast

from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.runner.utils import get_weights_size

try:
    from mlx_lm.tokenizer_utils import load_tokenizer
except ImportError:
    from mlx_lm.tokenizer_utils import load as load_tokenizer  # type: ignore
from mlx_lm.utils import load_model
from pydantic import RootModel

import mlx.core as mx
import mlx.nn as nn
from exo.engines.mlx import Model
from exo.engines.mlx.auto_parallel import (
    pipeline_auto_parallel,
    tensor_auto_parallel,
)
from exo.shared.types.common import Host
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.instances import (
    BoundInstance,
    MlxIbvInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    ShardMetadata,
    TensorShardMetadata,
)
from exo.worker.download.download_utils import build_model_path
from exo.worker.runner.bootstrap import logger

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
    bound_instance: BoundInstance,
) -> mx.distributed.Group:
    """
    Initialize the MLX distributed (runs in thread pool).

    Either hosts or mlx_ibv_devices must be provided:
    - hosts: traditional host-based connectivity using MLX_HOSTFILE
    - mlx_ibv_devices: RDMA connectivity matrix using MLX_IBV_DEVICES
    - mlx_ibv_coordinator: coordinator address (IP:PORT) for RDMA setup
    - strict: if True, raise an error if the distributed backend is not available
    """
    rank = bound_instance.bound_shard().device_rank
    logger.info(f"Starting initialization for rank {rank}")

    # TODO: singleton instances
    match bound_instance.instance:
        case MlxRingInstance(hosts=hosts):
            hostfile = f"./hosts_{rank}.json"
            hosts_json = HostList.from_hosts(hosts).model_dump_json()

            with open(hostfile, "w") as f:
                _ = f.write(hosts_json)

            logger.info(f"rank {rank} hostfile: {hostfile} hosts: {hosts_json}")

            os.environ["MLX_HOSTFILE"] = hostfile
            os.environ["MLX_RANK"] = str(rank)
            os.environ["MLX_RING_VERBOSE"] = "1"
            group = mx.distributed.init(backend="ring", strict=True)

        case MlxIbvInstance(ibv_devices=ibv_devices, ibv_coordinator=ibv_coordinator):
            import json

            # Use RDMA connectivity matrix
            devices_file = f"./hosts_{rank}.json"
            ibv_devices_json = json.dumps(ibv_devices)

            with open(devices_file, "w") as f:
                _ = f.write(ibv_devices_json)

            logger.info(f"rank {rank} MLX_IBV_DEVICES: {ibv_devices_json}")
            logger.info(f"rank {rank} MLX_IBV_COORDINATOR: {ibv_coordinator}")
            os.environ["MLX_IBV_DEVICES"] = devices_file
            os.environ["MLX_RANK"] = str(rank)
            os.environ["MLX_IBV_COORDINATOR"] = ibv_coordinator
            group = mx.distributed.init(backend="ibv", strict=True)

    logger.info(f"Rank {rank} mlx distributed initialization complete")

    return group


def initialize_mlx(
    bound_instance: BoundInstance,
) -> tuple[Model, TokenizerWrapper, Callable[[mx.array], mx.array]]:
    """
    Initialize the MLX model, tokenizer, and sampler. Runs in the MLX thread.
    """
    mx.random.seed(42)

    set_wired_limit_for_model(get_weights_size(bound_instance.bound_shard()))

    sampler: Callable[[mx.array], mx.array] = make_sampler(temp=0.7)
    logger.info("Created a sampler")

    if len(bound_instance.instance.shard_assignments.node_to_runner) <= 1:
        logger.info(f"Single device used for {bound_instance.instance}")
        model_path = build_model_path(bound_instance.bound_shard().model_meta.model_id)
        model, _ = load_model(model_path, strict=True)
        # TODO: we should really make this opt-in, but Kimi requires trust_remote_code=True
        tokenizer = cast(
            TokenizerWrapper,
            load_tokenizer(
                model_path, tokenizer_config_extra={"trust_remote_code": True}
            ),
        )
        assert isinstance(tokenizer, TokenizerWrapper)

    else:
        logger.info("Starting distributed init")
        group = mlx_distributed_init(bound_instance)
        model, tokenizer = shard_and_load(bound_instance.bound_shard(), group=group)

    set_wired_limit_for_model(get_weights_size(bound_instance.bound_shard()))

    return cast(Model, model), tokenizer, sampler


def shard_and_load(
    shard_metadata: ShardMetadata,
    group: mx.distributed.Group,
) -> tuple[nn.Module, TokenizerWrapper]:
    model_path = build_model_path(shard_metadata.model_meta.model_id)

    model, config = load_model(model_path, lazy=True, strict=False)
    logger.info(f"{config=}")
    assert isinstance(model, nn.Module)

    # TODO: we should really make this opt-in, but Kimi requires trust_remote_code=True
    tokenizer = cast(
        TokenizerWrapper,
        load_tokenizer(model_path, tokenizer_config_extra={"trust_remote_code": True}),
    )

    logger.info(f"Group size: {group.size()}, group rank: {group.rank()}")

    match shard_metadata:
        case TensorShardMetadata():
            logger.info(f"loading model from {model_path} with tensor parallelism")
            model = tensor_auto_parallel(model, group)
        case PipelineShardMetadata():
            logger.info(f"loading model from {model_path} with pipeline parallelism")
            model = pipeline_auto_parallel(model, group, shard_metadata)

    mx.eval(model.parameters())
    mx.eval(model)

    # Synchronize processes before generation to avoid timeout
    mx_barrier(group)

    return model, tokenizer


def apply_chat_template(
    tokenizer: TokenizerWrapper,
    chat_task_data: ChatCompletionTaskParams,
) -> str:
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

    prompt: str = tokenizer.apply_chat_template(  # type: ignore
        messages_dicts,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt  # type: ignore


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


def make_kv_cache(
    model: Model,
    max_kv_size: int | None = None,
) -> list[KVCache | RotatingKVCache]:
    assert hasattr(model, "layers")
    if max_kv_size is None:
        logger.info("Using default KV cache")
        return [KVCache() for _ in model.layers]
    else:
        logger.info(f"Using rotating KV cache with {max_kv_size=}")
        return [RotatingKVCache(max_size=max_kv_size) for _ in model.layers]


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
        logger.warning(
            f"Generating with a model that requires {model_mb} MB "
            f"which is close to the maximum recommended size of {max_rec_mb} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
        )
    kv_bytes = int(0.02 * model_bytes)
    target_cache = int(1.10 * (model_bytes + kv_bytes))
    target_cache = min(target_cache, max_rec_size)
    mx.set_cache_limit(target_cache)
    mx.set_wired_limit(max_rec_size)
    logger.info(
        f"Wired limit set to {max_rec_size}. Cache limit set to {target_cache}."
    )
