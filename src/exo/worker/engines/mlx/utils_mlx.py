import json
import os
import re
import sys
import time
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

from mlx_lm.models.cache import KVCache
from mlx_lm.models.deepseek_v3 import DeepseekV3Model
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.models.model_cards import ModelId
from exo.worker.engines.mlx.constants import TRUST_REMOTE_CODE

try:
    from mlx_lm.tokenizer_utils import load_tokenizer
except ImportError:
    from mlx_lm.tokenizer_utils import load as load_tokenizer
import contextlib

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model
from pydantic import RootModel

from exo.download.download_utils import build_model_path
from exo.shared.types.common import Host
from exo.shared.types.memory import Memory
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import (
    BoundInstance,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    ShardMetadata,
    TensorShardMetadata,
)
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.auto_parallel import (
    LayerLoadedCallback,
    TimeoutCallback,
    eval_with_timeout,
    get_inner_model,
    get_layers,
    pipeline_auto_parallel,
    tensor_auto_parallel,
)
from exo.worker.runner.bootstrap import logger

Group = mx.distributed.Group


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

                logger.info(
                    f"rank {rank} MLX_IBV_DEVICES: {coordination_file} with devices: {jaccl_devices_json}"
                )
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
    bound_instance: BoundInstance,
    group: Group | None,
    on_timeout: TimeoutCallback | None,
    on_layer_loaded: LayerLoadedCallback | None,
) -> tuple[Model, TokenizerWrapper]:
    if group is None:
        logger.info(f"Single device used for {bound_instance.instance}")
        model_path = build_model_path(bound_instance.bound_shard.model_card.model_id)
        start_time = time.perf_counter()
        model, _ = load_model(model_path, lazy=True, strict=False)
        # Eval layers one by one for progress reporting
        try:
            inner = get_inner_model(model)
            layers = get_layers(inner)
            total = len(layers)
            for i, layer in enumerate(layers):
                mx.eval(layer)  # type: ignore
                if on_layer_loaded is not None:
                    on_layer_loaded(i, total)
        except ValueError as e:
            logger.opt(exception=e).debug(
                "Model architecture doesn't support layer-by-layer progress tracking",
            )
        mx.eval(model)
        end_time = time.perf_counter()
        logger.info(f"Time taken to load model: {(end_time - start_time):.2f}s")
        tokenizer = get_tokenizer(model_path, bound_instance.bound_shard)

    else:
        logger.info("Starting distributed init")
        start_time = time.perf_counter()
        model, tokenizer = shard_and_load(
            bound_instance.bound_shard,
            group=group,
            on_timeout=on_timeout,
            on_layer_loaded=on_layer_loaded,
        )
        end_time = time.perf_counter()
        logger.info(
            f"Time taken to shard and load model: {(end_time - start_time):.2f}s"
        )

    set_wired_limit_for_model(get_weights_size(bound_instance.bound_shard))

    return cast(Model, model), tokenizer


def shard_and_load(
    shard_metadata: ShardMetadata,
    group: Group,
    on_timeout: TimeoutCallback | None,
    on_layer_loaded: LayerLoadedCallback | None,
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

    # Estimate timeout based on model size (5x default for large queued workloads)
    base_timeout = float(os.environ.get("EXO_MODEL_LOAD_TIMEOUT", "300"))
    model_size = get_weights_size(shard_metadata)
    timeout_seconds = base_timeout + model_size.in_gb
    logger.info(
        f"Evaluating model parameters with timeout of {timeout_seconds:.0f}s "
        f"(model size: {model_size.in_gb:.1f}GB)"
    )

    match shard_metadata:
        case TensorShardMetadata():
            logger.info(f"loading model from {model_path} with tensor parallelism")
            model = tensor_auto_parallel(
                model, group, timeout_seconds, on_timeout, on_layer_loaded
            )
        case PipelineShardMetadata():
            logger.info(f"loading model from {model_path} with pipeline parallelism")
            model = pipeline_auto_parallel(
                model, group, shard_metadata, on_layer_loaded=on_layer_loaded
            )
            eval_with_timeout(model.parameters(), timeout_seconds, on_timeout)
        case CfgShardMetadata():
            raise ValueError(
                "CfgShardMetadata is not supported for text model loading - "
                "this metadata type is only for image generation models"
            )

    # TODO: Do we need this?
    mx.eval(model)

    logger.debug("SHARDED")
    logger.debug(model)

    # Synchronize processes before generation to avoid timeout
    mx_barrier(group)

    return model, tokenizer


def get_tokenizer(model_path: Path, shard_metadata: ShardMetadata) -> TokenizerWrapper:
    """Load tokenizer for a model shard. Delegates to load_tokenizer_for_model_id."""
    return load_tokenizer_for_model_id(
        shard_metadata.model_card.model_id,
        model_path,
        trust_remote_code=shard_metadata.model_card.trust_remote_code,
    )


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
    elif "glm-5" in model_id_lower or "glm-4.7" in model_id_lower:
        # For GLM-5 and GLM-4.7
        # 154820: <|endoftext|>, 154827: <|user|>, 154829: <|observation|>
        return [154820, 154827, 154829]
    elif "glm" in model_id_lower:
        # For GLM-4.5 and older
        return [151336, 151329, 151338]
    elif "gpt-oss" in model_id_lower:
        return [200002, 200012]
    return None


def load_tokenizer_for_model_id(
    model_id: ModelId, model_path: Path, *, trust_remote_code: bool = TRUST_REMOTE_CODE
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
        import importlib.util
        import types

        sys.path.insert(0, str(model_path))

        # Load tool_declaration_ts first (tokenization_kimi imports it with relative import)
        tool_decl_path = model_path / "tool_declaration_ts.py"
        if tool_decl_path.exists():
            spec = importlib.util.spec_from_file_location(
                "tool_declaration_ts", tool_decl_path
            )
            if spec and spec.loader:
                tool_decl_module = importlib.util.module_from_spec(spec)
                sys.modules["tool_declaration_ts"] = tool_decl_module
                spec.loader.exec_module(tool_decl_module)

        # Load tokenization_kimi with patched source (convert relative to absolute import)
        tok_path = model_path / "tokenization_kimi.py"
        source = tok_path.read_text()
        source = source.replace("from .tool_declaration_ts", "from tool_declaration_ts")
        spec = importlib.util.spec_from_file_location("tokenization_kimi", tok_path)
        if spec:
            tok_module = types.ModuleType("tokenization_kimi")
            tok_module.__file__ = str(tok_path)
            sys.modules["tokenization_kimi"] = tok_module
            exec(compile(source, tok_path, "exec"), tok_module.__dict__)  # noqa: S102
            TikTokenTokenizer = tok_module.TikTokenTokenizer  # type: ignore[attr-defined]  # noqa: N806
        else:
            from tokenization_kimi import TikTokenTokenizer  # type: ignore[import-not-found]  # noqa: I001

        hf_tokenizer: Any = TikTokenTokenizer.from_pretrained(model_path)  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]

        # Patch encode to use internal tiktoken model directly
        # transformers 5.x has a bug in the encode->pad path for slow tokenizers
        def _patched_encode(text: str, **_kwargs: object) -> list[int]:
            # Pass allowed_special="all" to handle special tokens like <|im_user|>
            return list(hf_tokenizer.model.encode(text, allowed_special="all"))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

        hf_tokenizer.encode = _patched_encode
        return TokenizerWrapper(
            hf_tokenizer,
            eos_token_ids=eos_token_ids,
            tool_call_start="<|tool_calls_section_begin|>",
            tool_call_end="<|tool_calls_section_end|>",
            tool_parser=_parse_kimi_tool_calls,
        )

    tokenizer = load_tokenizer(
        model_path,
        tokenizer_config_extra={"trust_remote_code": trust_remote_code},
        eos_token_ids=eos_token_ids,
    )

    if "gemma-3" in model_id_lower:
        gemma_3_eos_id = 1
        gemma_3_end_of_turn_id = 106
        if tokenizer.eos_token_ids is not None:
            if gemma_3_end_of_turn_id not in tokenizer.eos_token_ids:
                tokenizer.eos_token_ids = list(tokenizer.eos_token_ids) + [
                    gemma_3_end_of_turn_id
                ]
        else:
            tokenizer.eos_token_ids = [gemma_3_eos_id, gemma_3_end_of_turn_id]

    return tokenizer


def _normalize_tool_calls(msg_dict: dict[str, Any]) -> None:
    """Normalize tool_calls in a message dict.

    OpenAI format has tool_calls[].function.arguments as a JSON string,
    but some chat templates (e.g., GLM) expect it as a dict.
    """
    tool_calls = msg_dict.get("tool_calls")
    if not tool_calls or not isinstance(tool_calls, list):
        return

    for tc in tool_calls:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(tc, dict):
            continue
        func = tc.get("function")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if not isinstance(func, dict):
            continue
        args = func.get("arguments")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if isinstance(args, str):
            with contextlib.suppress(json.JSONDecodeError):
                func["arguments"] = json.loads(args)


def _collect_nested_property_names(schema: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    properties: dict[str, Any] = schema.get("properties", {})  # type: ignore[reportAny]
    for prop_spec in properties.values():  # pyright: ignore[reportAny]
        if not isinstance(prop_spec, dict):
            continue
        if prop_spec.get("type") == "array":  # type: ignore[reportAny]
            items: dict[str, Any] | None = prop_spec.get("items")  # type: ignore[reportAny]
            if isinstance(items, dict) and items.get("type") == "object":  # type: ignore[reportAny]
                inner_props: dict[str, Any] = items.get("properties", {})  # type: ignore[reportAny]
                for k in inner_props:  # pyright: ignore[reportUnknownVariableType]
                    names.add(str(k))  # pyright: ignore[reportUnknownArgumentType]
                names.update(_collect_nested_property_names(items))  # pyright: ignore[reportUnknownArgumentType]
    return names


def _schemas_lost_in_prompt(prompt: str, tools: list[dict[str, Any]]) -> bool:
    """Return True if nested property names from any tool schema are absent."""
    for tool in tools:
        fn: dict[str, Any] = tool.get("function", {})  # type: ignore
        params: dict[str, Any] = fn.get("parameters", {})  # type: ignore
        nested = _collect_nested_property_names(params)
        if nested and not all(name in prompt for name in nested):
            return True
    return False


_LOSSY_TEMPLATE_PATTERN = re.compile(
    r"""inner_type\s*==\s*["']object \| object["']\s*or\s*inner_type\|length\s*>\s*\d+""",
)


def _patch_lossy_chat_template(template: str) -> str | None:
    """Patch chat templates that collapse nested object schemas to ``any[]``.

    Some templates (e.g., GPT-OSS) have a guard like::

        inner_type == "object | object" or inner_type|length > 50

    The length check silently drops complex array-of-object schemas.
    We remove the length guard, keeping only the object-union check.
    Returns the patched template, or *None* if no patch was needed.
    """
    patched, n = _LOSSY_TEMPLATE_PATTERN.subn(
        lambda m: m.group(0).split(" or ")[0],  # keep only the object-union check
        template,
    )
    return patched if n > 0 else None


def _needs_dsml_encoding(task_params: TextGenerationTaskParams) -> bool:
    if "deepseek-v3.2" not in task_params.model.lower():
        return False
    # Use DSML encoding when tools are provided or tool results are in the conversation
    if task_params.tools:
        return True
    if task_params.chat_template_messages:
        return any(
            msg.get("role") == "tool" for msg in task_params.chat_template_messages
        )
    return False


def apply_chat_template(
    tokenizer: TokenizerWrapper,
    task_params: TextGenerationTaskParams,
) -> str:
    """Convert TextGenerationTaskParams to a chat template prompt.

    Converts the internal format (input + instructions) to a messages list
    that can be processed by the tokenizer's chat template.

    When chat_template_messages is available (from Chat Completions API),
    uses those directly to preserve tool_calls, thinking, and other fields.
    """
    formatted_messages: list[dict[str, Any]] = []
    if task_params.chat_template_messages is not None:
        # Use pre-formatted messages that preserve tool_calls, thinking, etc.
        formatted_messages = list(task_params.chat_template_messages)
        for msg in formatted_messages:
            _normalize_tool_calls(msg)
    else:
        # Add system message (instructions) if present
        if task_params.instructions:
            formatted_messages.append(
                {"role": "system", "content": task_params.instructions}
            )

        # Convert input to messages
        for msg in task_params.input:
            if not msg.content:
                logger.warning("Received message with empty content, skipping")
                continue
            formatted_messages.append({"role": msg.role, "content": msg.content})

    # For assistant prefilling, append content after templating to avoid a closing turn token.
    partial_assistant_content: str | None = None
    if formatted_messages and formatted_messages[-1].get("role") == "assistant":
        partial_assistant_content = cast(str, formatted_messages[-1].get("content", ""))
        formatted_messages = formatted_messages[:-1]

    if _needs_dsml_encoding(task_params):
        from exo.worker.engines.mlx.dsml_encoding import encode_messages

        prompt = encode_messages(
            messages=formatted_messages,
            thinking_mode="thinking" if task_params.enable_thinking else "chat",
            tools=task_params.tools,
        )
        if partial_assistant_content:
            prompt += partial_assistant_content
        logger.info(prompt)
        return prompt

    extra_kwargs: dict[str, Any] = {}
    if task_params.enable_thinking is not None:
        # Qwen3 and GLM use "enable_thinking"; DeepSeek uses "thinking".
        # Jinja ignores unknown variables, so passing both is safe.
        extra_kwargs["enable_thinking"] = task_params.enable_thinking
        extra_kwargs["thinking"] = task_params.enable_thinking

    patched_template: str | None = None
    if task_params.tools:
        original_template: str | None = getattr(tokenizer, "chat_template", None)
        if isinstance(original_template, str):
            patched_template = _patch_lossy_chat_template(original_template)
            if patched_template is not None:
                logger.info(
                    "Patched lossy chat template (removed inner_type length guard)"
                )

    prompt: str = tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=task_params.tools,
        **({"chat_template": patched_template} if patched_template is not None else {}),
        **extra_kwargs,
    )

    if task_params.tools and _schemas_lost_in_prompt(prompt, task_params.tools):
        logger.warning("Chat template lost nested tool schemas even after patching")

    if partial_assistant_content:
        prompt += partial_assistant_content

    logger.info(prompt)

    return prompt


def detect_thinking_prompt_suffix(prompt: str, tokenizer: TokenizerWrapper) -> bool:
    """
    Detect if prompt ends with a thinking opening tag that should be
    prepended to the output stream.
    """
    think_token = tokenizer.think_start

    return think_token is not None and prompt.rstrip().endswith(think_token)


def fix_unmatched_think_end_tokens(
    tokens: mx.array, tokenizer: TokenizerWrapper
) -> mx.array:
    if not tokenizer.has_thinking:
        return tokens
    assert tokenizer.think_start_id
    assert tokenizer.think_end_id
    think_start_id: int = tokenizer.think_start_id
    think_end_id: int = tokenizer.think_end_id
    token_list: list[int] = cast(list[int], tokens.tolist())
    result: list[int] = []
    depth = 0
    for token in token_list:
        if token == think_start_id:
            depth += 1
        elif token == think_end_id:
            if depth == 0:
                result.append(think_start_id)
            else:
                depth -= 1
        result.append(token)
    return mx.array(result)


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


def mlx_force_oom(size: int = 200000) -> None:
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

    max_rec_size = Memory.from_bytes(
        int(mx.device_info()["max_recommended_working_set_size"])
    )
    if model_size > 0.9 * max_rec_size:
        logger.warning(
            f"Generating with a model that requires {model_size.in_float_mb:.1f} MB "
            f"which is close to the maximum recommended size of {max_rec_size.in_float_mb:.1f} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
        )
    mx.set_wired_limit(max_rec_size.in_bytes)
    logger.info(f"Wired limit set to {max_rec_size}.")


def mlx_cleanup(
    model: Model | None, tokenizer: TokenizerWrapper | None, group: Group | None
) -> None:
    del model, tokenizer, group
    mx.clear_cache()
    import gc

    gc.collect()


def mx_any(bool_: bool, group: Group | None) -> bool:
    if group is None:
        return bool_
    num_true = mx.distributed.all_sum(
        mx.array(bool_), group=group, stream=mx.default_stream(mx.Device(mx.cpu))
    )
    mx.eval(num_true)
    return num_true.item() > 0


def mx_barrier(group: Group | None):
    if group is None:
        return
    mx.eval(
        mx.distributed.all_sum(
            mx.array(1.0), group=group, stream=mx.default_stream(mx.Device(mx.cpu))
        )
    )


def _parse_kimi_tool_calls(text: str):
    import regex as re

    # kimi has a fixed function naming scheme, with a json formatted arg
    #   functions.multiply:0<|tool_call_argument_begin|>{"a": 2, "b": 3}
    _func_name_regex = re.compile(
        r"^\s*((?:functions\.)?(.+?):\d+)\s*<\|tool_call_argument_begin\|>", re.DOTALL
    )
    _func_arg_regex = re.compile(r"<\|tool_call_argument_begin\|>\s*(.*)\s*", re.DOTALL)
    _tool_call_split_regex = re.compile(
        r"<\|tool_call_begin\|>(.*?)<\|tool_call_end\|>", re.DOTALL
    )

    def _parse_single_tool(text: str) -> dict[str, Any]:
        func_name_match = _func_name_regex.search(text)
        if func_name_match is None:
            raise ValueError("No tool call found.")
        tool_call_id = func_name_match.group(1)  # e.g. "functions.get_weather:0"
        func_name = func_name_match.group(2)  # e.g. "get_weather"

        func_args_match = _func_arg_regex.search(text)
        if func_args_match is None:
            raise ValueError("No tool call arguments found.")
        func_args = func_args_match.group(1)
        try:
            arg_dct = json.loads(func_args)  # pyright: ignore[reportAny]
        except Exception:
            arg_dct = None

        return dict(id=tool_call_id, name=func_name, arguments=arg_dct)

    tool_matches = _tool_call_split_regex.findall(text)
    if tool_matches:
        return [_parse_single_tool(match) for match in tool_matches]  # pyright: ignore[reportAny]
    else:
        return [_parse_single_tool(text)]
