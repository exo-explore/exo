import json
import os
import resource
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

# Monkey-patch for transformers 5.x compatibility
# Kimi's tokenization_kimi.py imports bytes_to_unicode from the old location
# which was moved in transformers 5.0.0rc2
try:
    import transformers.models.gpt2.tokenization_gpt2 as gpt2_tokenization
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    if not hasattr(gpt2_tokenization, "bytes_to_unicode"):
        gpt2_tokenization.bytes_to_unicode = bytes_to_unicode  # type: ignore[attr-defined]
except ImportError:
    pass  # transformers < 5.0 or bytes_to_unicode not available

from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache
from mlx_lm.models.deepseek_v3 import DeepseekV3Model
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.models.model_cards import ModelId
from exo.worker.engines.mlx.constants import (
    CACHE_GROUP_SIZE,
    KV_CACHE_BITS,
    MTP_ENABLED,
    TRUST_REMOTE_CODE,
)

try:
    from mlx_lm.tokenizer_utils import load_tokenizer
except ImportError:
    from mlx_lm.tokenizer_utils import load as load_tokenizer
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
    TimeoutCallback,
    eval_with_timeout,
    pipeline_auto_parallel,
    tensor_auto_parallel,
)
from exo.worker.runner.bootstrap import logger

Group = mx.distributed.Group
# Needed for 8 bit model
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, 4096))


# MTP (Multi-Token Prediction) support for DeepSeek V3
MTP_LAYER_INDEX = 61
_original_deepseek_sanitize: Callable[..., dict[str, Any]] | None = None


def _is_deepseek_v3_model(model: nn.Module) -> bool:
    """Check if the model is DeepSeek V3."""
    return hasattr(model, "model") and isinstance(model.model, DeepseekV3Model)


def _patch_deepseek_sanitize_for_mtp() -> None:
    """Patch DeepSeek V3 Model.sanitize to preserve MTP layer weights.

    The original sanitize() method filters out layer 61 (MTP layer) weights.
    This patch keeps them so we can extract and use the MTP module.
    """
    global _original_deepseek_sanitize
    from mlx_lm.models.deepseek_v3 import Model as DeepSeekV3Model

    if _original_deepseek_sanitize is not None:
        # Already patched
        return

    _original_deepseek_sanitize = DeepSeekV3Model.sanitize

    def sanitize_with_mtp(
        self: DeepSeekV3Model, weights: dict[str, Any]
    ) -> dict[str, Any]:
        """Modified sanitize that keeps MTP layer weights."""
        # First, call the original sanitize to handle all the weight transformations
        # (dequantization, expert stacking, etc.)
        if _original_deepseek_sanitize is None:
            raise RuntimeError(
                "_original_deepseek_sanitize is None - patch not applied correctly"
            )
        original_result: dict[str, Any] = _original_deepseek_sanitize(self, weights)

        # Re-add the MTP layer weights that were filtered out
        mtp_weights = {
            k: v
            for k, v in weights.items()
            if k.startswith(f"model.layers.{MTP_LAYER_INDEX}")
        }

        return {**original_result, **mtp_weights}

    DeepSeekV3Model.sanitize = sanitize_with_mtp


def _restore_deepseek_sanitize() -> None:
    """Restore the original DeepSeek V3 sanitize method."""
    global _original_deepseek_sanitize
    if _original_deepseek_sanitize is None:
        return

    from mlx_lm.models.deepseek_v3 import Model as DeepSeekV3Model

    DeepSeekV3Model.sanitize = _original_deepseek_sanitize
    _original_deepseek_sanitize = None


# TODO: Test this
#  ALSO https://github.com/exo-explore/exo/pull/233#discussion_r2549683673
def get_weights_size(model_shard_meta: ShardMetadata) -> Memory:
    return Memory.from_float_kb(
        (model_shard_meta.end_layer - model_shard_meta.start_layer)
        / model_shard_meta.n_layers
        * model_shard_meta.model_card.storage_size.in_kb
        / (
            1
            if isinstance(model_shard_meta, PipelineShardMetadata)
            else model_shard_meta.world_size
        )
    )


class ModelLoadingTimeoutError(Exception):
    pass


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
                jaccl_devices=jaccl_devices, jaccl_coordinators=jaccl_coordinators
            ):
                assert all(
                    jaccl_devices[i][i] is None for i in range(len(jaccl_devices))
                )
                # Use RDMA connectivity matrix
                coordination_file = (
                    f"./hosts_{bound_instance.instance.instance_id}_{rank}.json"
                )
                jaccl_devices_json = json.dumps(jaccl_devices)

                with open(coordination_file, "w") as f:
                    _ = f.write(jaccl_devices_json)

                jaccl_coordinator = jaccl_coordinators[bound_instance.bound_node_id]

                # TODO: update once upstream fixes
                logger.info(
                    f"rank {rank} MLX_JACCL_DEVICES: {coordination_file} with devices: {jaccl_devices_json}"
                )
                logger.info(f"rank {rank} MLX_JACCL_COORDINATOR: {jaccl_coordinator}")
                os.environ["MLX_JACCL_DEVICES"] = coordination_file
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
    bound_instance: BoundInstance,
    group: Group | None,
    on_timeout: TimeoutCallback | None = None,
) -> tuple[Model, TokenizerWrapper]:
    """Load MLX model and tokenizer.

    Returns:
        Tuple of (model, tokenizer)
    """
    model_id = bound_instance.bound_shard.model_meta.model_id
    mtp_module = None

    # Patch sanitize for MTP if this might be DeepSeek V3
    should_try_mtp = MTP_ENABLED and _might_be_deepseek_v3(model_id)
    if should_try_mtp:
        logger.info("Patching DeepSeek V3 sanitize for MTP weight preservation")
        _patch_deepseek_sanitize_for_mtp()

    try:
        if group is None:
            logger.info(f"Single device used for {bound_instance.instance}")
            model_path = build_model_path(model_id)
            start_time = time.perf_counter()
            model, _ = load_model(model_path, strict=not should_try_mtp)
            end_time = time.perf_counter()
            logger.info(f"Time taken to load model: {(end_time - start_time):.2f}s")
            tokenizer = get_tokenizer(model_path, bound_instance.bound_shard)

        else:
            logger.info("Starting distributed init")
            start_time = time.perf_counter()
            model, tokenizer = shard_and_load(
                bound_instance.bound_shard, group=group, on_timeout=on_timeout
            )
            end_time = time.perf_counter()
            logger.info(
                f"Time taken to shard and load model: {(end_time - start_time):.2f}s"
            )

        # Extract MTP module if available
        if should_try_mtp and _is_deepseek_v3_model(model):
            mtp_module = _extract_mtp_module(model)
            if mtp_module is not None:
                logger.info("Successfully extracted MTP module from DeepSeek V3")

    finally:
        # Restore original sanitize
        if should_try_mtp:
            _restore_deepseek_sanitize()

    set_wired_limit_for_model(get_weights_size(bound_instance.bound_shard))

    # Store MTP module on the model for later access
    if mtp_module is not None:
        model.mtp_module = mtp_module  # noqa: B010

    return cast(Model, model), tokenizer


def _might_be_deepseek_v3(model_id: str) -> bool:
    """Check if model ID suggests this might be DeepSeek V3."""
    model_id_lower = model_id.lower()
    return "deepseek" in model_id_lower and (
        "v3" in model_id_lower or "r1" in model_id_lower
    )


def _flatten_params(
    params: dict[str, Any],
    prefix: str = "",
) -> dict[str, mx.array]:
    """Flatten nested parameter dict to flat dict with dot-separated keys."""
    result: dict[str, mx.array] = {}
    for key, value in params.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, mx.array):
            result[full_key] = value
        elif isinstance(value, dict):
            result.update(_flatten_params(value, full_key))
    return result


def _extract_mtp_module(model: nn.Module) -> Any | None:
    """Extract MTP module from a loaded DeepSeek V3 model.

    The MTP weights are stored in model.model.layers at index 61 (if preserved).
    This function extracts them and creates an MTPModule.

    Returns:
        MTPModule if MTP weights were found and extracted, None otherwise.
    """
    from exo.worker.engines.mlx.mtp.module import (
        MTPModule,
        extract_mtp_weights,
        load_mtp_weights_into_module,
    )

    try:
        # Check if the model has the MTP layer
        inner_model = getattr(model, "model", None)
        if inner_model is None or not hasattr(inner_model, "layers"):
            logger.debug("Model doesn't have expected structure for MTP extraction")
            return None

        layers: list[nn.Module] = inner_model.layers  # type: ignore[assignment]
        if len(layers) <= MTP_LAYER_INDEX:
            logger.debug(
                f"Model has {len(layers)} layers, MTP layer {MTP_LAYER_INDEX} not found"
            )
            return None

        # Get model config
        config = getattr(model, "args", None)
        if config is None:
            logger.debug("Could not get model config for MTP module")
            return None

        # Create MTP module with shared weights
        embed_tokens = getattr(inner_model, "embed_tokens", None)
        lm_head = getattr(model, "lm_head", None)
        norm = getattr(inner_model, "norm", None)

        if embed_tokens is None or lm_head is None or norm is None:
            logger.debug("Could not get required model components for MTP")
            return None

        mtp_module = MTPModule(
            config=config,
            shared_embedding=embed_tokens,
            shared_lm_head=lm_head,
            output_norm=norm,
        )

        # Extract MTP layer weights from the model's parameters
        # The weights should be at model.model.layers.61.*
        # model.parameters() returns a nested dict, we need to flatten it
        raw_params: dict[str, Any] = dict(model.parameters())  # type: ignore[arg-type]
        model_weights = _flatten_params(raw_params)
        mtp_weights = extract_mtp_weights(model_weights)

        if not mtp_weights:
            logger.debug("No MTP weights found in model parameters")
            return None

        # Load weights into MTP module
        load_mtp_weights_into_module(mtp_module, mtp_weights)

        # Remove MTP layer from main model to avoid double computation
        # Create new layers list without the MTP layer
        new_layers = [layer for i, layer in enumerate(layers) if i != MTP_LAYER_INDEX]
        inner_model.layers = new_layers  # noqa: B010

        logger.info(
            f"Extracted MTP module, main model now has {len(new_layers)} layers"
        )
        return mtp_module

    except Exception as e:
        logger.warning(f"Failed to extract MTP module: {e}")
        return None


def shard_and_load(
    shard_metadata: ShardMetadata,
    group: Group,
    on_timeout: TimeoutCallback | None = None,
) -> tuple[nn.Module, TokenizerWrapper]:
    model_path = build_model_path(shard_metadata.model_card.model_id)

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

    # Estimate timeout based on model size
    base_timeout = float(os.environ.get("EXO_MODEL_LOAD_TIMEOUT", "60"))
    model_size_gb = get_weights_size(shard_metadata).in_bytes / (1024**3)
    timeout_seconds = base_timeout + model_size_gb / 5
    logger.info(
        f"Evaluating model parameters with timeout of {timeout_seconds:.0f}s "
        f"(model size: {model_size_gb:.1f}GB)"
    )

    match shard_metadata:
        case TensorShardMetadata():
            logger.info(f"loading model from {model_path} with tensor parallelism")
            model = tensor_auto_parallel(model, group, timeout_seconds, on_timeout)
        case PipelineShardMetadata():
            logger.info(f"loading model from {model_path} with pipeline parallelism")
            model = pipeline_auto_parallel(model, group, shard_metadata)
            eval_with_timeout(model.parameters(), timeout_seconds, on_timeout)

    # TODO: Do we need this?
    mx.eval(model)

    logger.debug("SHARDED")
    logger.debug(model)

    # Synchronize processes before generation to avoid timeout
    mx_barrier(group)

    return model, tokenizer


def get_tokenizer(model_path: Path, shard_metadata: ShardMetadata) -> TokenizerWrapper:
    """Load tokenizer for a model shard. Delegates to load_tokenizer_for_model_id."""
    return load_tokenizer_for_model_id(shard_metadata.model_card.model_id, model_path)


def get_eos_token_ids_for_model(model_id: ModelId) -> list[int] | None:
    """
    Get the EOS token IDs for a model based on its ID.

    Some models require explicit EOS token configuration that isn't in their
    tokenizer config. This function returns the known EOS token IDs for such models.

    Args:
        model_id: The HuggingFace model ID

    Returns:
        List of EOS token IDs, or None if the model uses standard tokenizer config
    """
    model_id_lower = model_id.lower()
    if "kimi-k2" in model_id_lower:
        return [163586]
    elif "glm-4.7-flash" in model_id_lower:
        # 154820: <|endoftext|>, 154827: <|user|>, 154829: <|observation|>
        return [154820, 154827, 154829]
    elif "glm" in model_id_lower:
        return [151336, 151329, 151338]
    return None


def load_tokenizer_for_model_id(
    model_id: ModelId, model_path: Path
) -> TokenizerWrapper:
    """
    Load tokenizer for a model given its ID and local path.

    This is the core tokenizer loading logic, handling special cases for different
    model families (Kimi, GLM, etc.) and transformers 5.x compatibility.

    Args:
        model_id: The HuggingFace model ID (e.g., "moonshotai/Kimi-K2-Instruct")
        model_path: Local path where the model/tokenizer files are stored

    Returns:
        TokenizerWrapper instance configured for the model
    """
    model_id_lower = model_id.lower()
    eos_token_ids = get_eos_token_ids_for_model(model_id)

    # Kimi uses a custom TikTokenTokenizer that transformers 5.x can't load via AutoTokenizer
    if "kimi-k2" in model_id_lower:
        sys.path.insert(0, str(model_path))
        from tokenization_kimi import TikTokenTokenizer  # type: ignore[import-not-found]  # noqa: I001

        hf_tokenizer: Any = TikTokenTokenizer.from_pretrained(model_path)  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]

        # Patch encode to use internal tiktoken model directly
        # transformers 5.x has a bug in the encode->pad path for slow tokenizers
        def _patched_encode(text: str, **_kwargs: object) -> list[int]:
            # Pass allowed_special="all" to handle special tokens like <|im_user|>
            return list(hf_tokenizer.model.encode(text, allowed_special="all"))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

        hf_tokenizer.encode = _patched_encode
        return TokenizerWrapper(hf_tokenizer, eos_token_ids=eos_token_ids)

    tokenizer = load_tokenizer(
        model_path,
        tokenizer_config_extra={"trust_remote_code": TRUST_REMOTE_CODE},
        eos_token_ids=eos_token_ids,
    )

    return tokenizer


def apply_chat_template(
    tokenizer: TokenizerWrapper,
    chat_task_data: ChatCompletionTaskParams,
) -> str:
    # Now we can properly access the messages
    messages = chat_task_data.messages

    formatted_messages: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message.content, ChatCompletionMessageText):
            message.content = message.content.text
        if isinstance(message.content, list):
            if len(message.content) == 0:
                logger.warning("Received prompt with no content, skipping")
                continue

            message.content = "\n".join(c.text for c in message.content).strip()
        if message.content is None and message.thinking is None:
            continue

        # Null values are not valid when applying templates in tokenizer
        formatted_messages.append(
            {k: v for k, v in message.model_dump().items() if v is not None}  # type: ignore
        )

    prompt: str = tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=chat_task_data.tools,
    )

    logger.info(prompt)

    return prompt


def detect_thinking_prompt_suffix(prompt: str, tokenizer: TokenizerWrapper) -> bool:
    """
    Detect if prompt ends with a thinking opening tag that should be
    prepended to the output stream.
    """
    think_token = tokenizer.think_start

    return think_token is not None and prompt.rstrip().endswith(think_token)


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

    # TODO: Do this for all models
    if hasattr(model, "make_cache") and isinstance(model, GptOssModel):
        logger.info("Using MLX LM's make cache")
        return model.make_cache()  # type: ignore

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


def mlx_cleanup(
    model: Model | None, tokenizer: TokenizerWrapper | None, group: Group | None
) -> None:
    del model, tokenizer, group
    mx.clear_cache()
    import gc

    gc.collect()
