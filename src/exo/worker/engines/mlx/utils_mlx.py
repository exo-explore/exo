import json
import os
import resource
import time
from pathlib import Path
from typing import Any, Callable, cast

from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache
from mlx_lm.models.deepseek_v3 import DeepseekV3Model
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.engines.mlx.constants import (
    CACHE_GROUP_SIZE,
    KV_CACHE_BITS,
    TRUST_REMOTE_CODE,
)

try:
    from mlx_lm.tokenizer_utils import load_tokenizer
except ImportError:
    from mlx_lm.tokenizer_utils import load as load_tokenizer  # type: ignore
import contextlib

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model
from pydantic import RootModel

from exo.shared.types.api import ChatCompletionMessageText
from exo.shared.types.common import Host
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.instances import (
    BoundInstance,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    ShardMetadata,
    TensorShardMetadata,
)
from exo.worker.download.download_utils import build_model_path
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.auto_parallel import (
    pipeline_auto_parallel,
    tensor_auto_parallel,
)
from exo.worker.runner.bootstrap import logger

Group = mx.distributed.Group
# Needed for 8 bit model
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, 4096))


# TODO: Test this
#  ALSO https://github.com/exo-explore/exo/pull/233#discussion_r2549683673
def get_weights_size(model_shard_meta: ShardMetadata) -> Memory:
    return Memory.from_float_kb(
        (model_shard_meta.end_layer - model_shard_meta.start_layer)
        / model_shard_meta.n_layers
        * model_shard_meta.model_meta.storage_size.in_kb
        / (
            1
            if isinstance(model_shard_meta, PipelineShardMetadata)
            else model_shard_meta.world_size
        )
    )


def mx_barrier(group: Group | None = None):
    mx.eval(
        mx.distributed.all_sum(
            mx.array(1.0),
            stream=mx.default_stream(mx.Device(mx.cpu)),
            group=group,
        )
    )


def broadcast_from_zero(value: int, group: Group | None = None):
    if group is None:
        return value

    if group.rank() == 0:
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
) -> Group:
    """
    Initialize MLX distributed.
    """
    rank = bound_instance.bound_shard.device_rank
    logger.info(f"Starting initialization for rank {rank}")

    coordination_file = None
    try:
        # TODO: singleton instances
        match bound_instance.instance:
            case MlxRingInstance(hosts_by_node=hosts_by_node, ephemeral_port=_):
                coordination_file = (
                    f"./hosts_{bound_instance.instance.instance_id}_{rank}.json"
                )
                hosts_for_node = hosts_by_node[bound_instance.bound_node_id]
                hosts_json = HostList.from_hosts(hosts_for_node).model_dump_json()

                with open(coordination_file, "w") as f:
                    _ = f.write(hosts_json)

                logger.info(
                    f"rank {rank} hostfile: {coordination_file} hosts: {hosts_json}"
                )

                os.environ["MLX_HOSTFILE"] = coordination_file
                os.environ["MLX_RANK"] = str(rank)
                os.environ["MLX_RING_VERBOSE"] = "1"
                group = mx.distributed.init(backend="ring", strict=True)

            case MlxJacclInstance(
                ibv_devices=ibv_devices, jaccl_coordinators=jaccl_coordinators
            ):
                # Use RDMA connectivity matrix
                coordination_file = (
                    f"./hosts_{bound_instance.instance.instance_id}_{rank}.json"
                )
                ibv_devices_json = json.dumps(ibv_devices)

                with open(coordination_file, "w") as f:
                    _ = f.write(ibv_devices_json)

                jaccl_coordinator = jaccl_coordinators[bound_instance.bound_node_id]

                logger.info(f"rank {rank} MLX_IBV_DEVICES: {ibv_devices_json}")
                logger.info(f"rank {rank} MLX_JACCL_COORDINATOR: {jaccl_coordinator}")
                os.environ["MLX_IBV_DEVICES"] = coordination_file
                os.environ["MLX_RANK"] = str(rank)
                os.environ["MLX_JACCL_COORDINATOR"] = jaccl_coordinator
                group = mx.distributed.init(backend="jaccl", strict=True)

        logger.info(f"Rank {rank} mlx distributed initialization complete")

        return group
    finally:
        with contextlib.suppress(FileNotFoundError):
            if coordination_file:
                os.remove(coordination_file)


def initialize_mlx(
    bound_instance: BoundInstance,
) -> Group:
    # should we unseed it?
    # TODO: pass in seed from params
    mx.random.seed(42)

    assert len(bound_instance.instance.shard_assignments.node_to_runner) > 1, (
        "Tried to initialize mlx for a single node instance"
    )
    return mlx_distributed_init(bound_instance)


def load_mlx_items(
    bound_instance: BoundInstance, group: Group | None
) -> tuple[Model, TokenizerWrapper, Callable[[mx.array], mx.array]]:
    # TODO: pass temperature
    sampler: Callable[[mx.array], mx.array] = make_sampler(temp=0.7)
    logger.info("Created a sampler")

    if group is None:
        logger.info(f"Single device used for {bound_instance.instance}")
        model_path = build_model_path(bound_instance.bound_shard.model_meta.model_id)
        start_time = time.perf_counter()
        model, _ = load_model(model_path, strict=True)
        end_time = time.perf_counter()
        logger.info(f"Time taken to load model: {(end_time - start_time):.2f}s")
        tokenizer = get_tokenizer(model_path, bound_instance.bound_shard)

    else:
        logger.info("Starting distributed init")
        start_time = time.perf_counter()
        model, tokenizer = shard_and_load(bound_instance.bound_shard, group=group)
        end_time = time.perf_counter()
        logger.info(
            f"Time taken to shard and load model: {(end_time - start_time):.2f}s"
        )

    set_wired_limit_for_model(get_weights_size(bound_instance.bound_shard))

    return cast(Model, model), tokenizer, sampler


def shard_and_load(
    shard_metadata: ShardMetadata,
    group: Group,
) -> tuple[nn.Module, TokenizerWrapper]:
    model_path = build_model_path(shard_metadata.model_meta.model_id)

    model, _ = load_model(model_path, lazy=True, strict=False)
    logger.debug(model)
    if hasattr(model, "model") and isinstance(model.model, DeepseekV3Model):  # type: ignore
        pass
        # TODO: See if we should quantize the model.
        # def is_attention_layer(path: str) -> bool:
        #     path = path.lower()

        #     return "self_attn" in path and "layernorm" not in path

        # def quant_predicate(path: str, module: nn.Module):
        #     if not isinstance(module, nn.Linear):
        #         return False

        #     return is_attention_layer(path)
        # model, config = quantize_model(
        #        model, config, group_size=KV_GROUP_SIZE, bits=ATTENTION_KV_BITS, quant_predicate=quant_predicate, mode=QUANTIZE_MODEL_MODE
        #    )

    assert isinstance(model, nn.Module)

    tokenizer = get_tokenizer(model_path, shard_metadata)

    logger.info(f"Group size: {group.size()}, group rank: {group.rank()}")

    match shard_metadata:
        case TensorShardMetadata():
            logger.info(f"loading model from {model_path} with tensor parallelism")
            model = tensor_auto_parallel(model, group)
        case PipelineShardMetadata():
            logger.info(f"loading model from {model_path} with pipeline parallelism")
            model = pipeline_auto_parallel(model, group, shard_metadata)

    mx.eval(model.parameters())

    # TODO: Do we need this?
    mx.eval(model)

    logger.debug("SHARDED")
    logger.debug(model)

    # Synchronize processes before generation to avoid timeout
    mx_barrier(group)

    return model, tokenizer


def get_tokenizer(model_path: Path, shard_metadata: ShardMetadata):
    # TODO: Let's move away from this custom logic to mlx_lm.load()
    if "kimi-k2" in shard_metadata.model_meta.model_id.lower():
        eos_token_ids = [163586]

    elif "glm" in shard_metadata.model_meta.model_id.lower():
        eos_token_ids = [151336, 151329, 151338]

    else:
        eos_token_ids = None

    tokenizer = cast(
        TokenizerWrapper,
        load_tokenizer(
            model_path,
            tokenizer_config_extra={"trust_remote_code": TRUST_REMOTE_CODE},
            eos_token_ids=eos_token_ids,
        ),
    )
    assert isinstance(tokenizer, TokenizerWrapper)

    return tokenizer


def apply_chat_template(
    tokenizer: TokenizerWrapper,
    chat_task_data: ChatCompletionTaskParams,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    # Now we can properly access the messages
    messages = chat_task_data.messages

    formatted_messages: list[dict[str, Any]] = []
    tool_results_buffer: list[str] = []

    for message in messages:
        if isinstance(message.content, ChatCompletionMessageText):
            message.content = message.content.text
        if isinstance(message.content, list):
            if len(message.content) != 1:
                logger.warning("Received malformed prompt")
                continue

            message.content = message.content[0].text
        if message.content is None and message.thinking is None:
            continue

        # Null values are not valid when applying templates in tokenizer
        msg_dict = {k: v for k, v in message.model_dump().items() if v is not None}  # type: ignore

        # Llama 3.1 doesn't support parallel tool calling - remove tool_calls from history
        # Tool results are sent as separate tool messages, so we don't need them here
        if msg_dict.get("role") == "assistant" and "tool_calls" in msg_dict:
            tool_calls = cast(list[Any], msg_dict.get("tool_calls"))
            if len(tool_calls) > 1:
                # Multiple tool calls - remove them entirely to avoid template error
                logger.info(f"Removing {len(tool_calls)} parallel tool_calls from assistant message (Llama 3.1 limitation)")
                msg_dict.pop("tool_calls", None)
                # If content is None, skip this message entirely
                if msg_dict.get("content") is None:
                    continue

        # Handle tool messages: Llama template only supports single tool-calls
        # Collect consecutive tool results and combine them into one user message
        if msg_dict.get("role") == "tool":
            tool_name = cast(str, msg_dict.get("name")) if "name" in msg_dict else "unknown"
            tool_result = cast(str, msg_dict.get("content")) if "content" in msg_dict else ""
            tool_results_buffer.append(f"Tool '{tool_name}' returned: {tool_result}")
            continue  # Don't append yet, collect all consecutive tool results

        # If we have buffered tool results, flush them as a single user message
        if tool_results_buffer:
            formatted_messages.append({
                "role": "user",
                "content": "\n".join(tool_results_buffer)
            })
            tool_results_buffer = []

        formatted_messages.append(msg_dict)

    # Flush any remaining tool results at the end
    if tool_results_buffer:
        formatted_messages.append({
            "role": "user",
            "content": "\n".join(tool_results_buffer)
        })

    # Pass tools to the tokenizer if provided
    if tools is not None and len(tools) > 0:
        logger.info(f"Applying chat template with {len(tools)} tools")
        prompt: str = tokenizer.apply_chat_template(  # type: ignore
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
        )
    else:
        prompt: str = tokenizer.apply_chat_template(  # type: ignore
            formatted_messages,
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
    model: Model, max_kv_size: int | None = None, keep: int = 0
) -> list[KVCache | RotatingKVCache | QuantizedKVCache]:
    assert hasattr(model, "layers")

    if max_kv_size is None:
        if KV_CACHE_BITS is None:
            logger.info("Using default KV cache")
            return [KVCache() for _ in model.layers]
        else:
            logger.info("Using quantized KV cache")
            return [
                QuantizedKVCache(group_size=CACHE_GROUP_SIZE, bits=KV_CACHE_BITS)
                for _ in model.layers
            ]
    else:
        logger.info(f"Using rotating KV cache with {max_kv_size=} with {keep=}")
        return [RotatingKVCache(max_size=max_kv_size, keep=keep) for _ in model.layers]


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
    mx.set_wired_limit(max_rec_size)
    logger.info(f"Wired limit set to {max_rec_size}.")
