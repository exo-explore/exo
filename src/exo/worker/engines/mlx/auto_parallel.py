import os
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from inspect import signature
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import (
    compute_shard_sizes,
    shard_inplace,
    shard_linear,
    sum_gradients,
)
from mlx_lm.models.base import (
    scaled_dot_product_attention,  # pyright: ignore[reportUnknownVariableType]
)
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.deepseek_v3 import DeepseekV3MLP
from mlx_lm.models.deepseek_v3 import Model as DeepseekV3Model
from mlx_lm.models.deepseek_v32 import DeepseekV32MLP
from mlx_lm.models.deepseek_v32 import Model as DeepseekV32Model
from mlx_lm.models.glm4_moe import Model as Glm4MoeModel
from mlx_lm.models.glm4_moe import MoE
from mlx_lm.models.glm4_moe_lite import Glm4MoeLiteDecoderLayer, Glm4MoeLiteMLP
from mlx_lm.models.glm4_moe_lite import Model as GLM4MoeLiteModel
from mlx_lm.models.gpt_oss import GptOssMoeModel
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.models.kimi_k25 import Model as KimiK25Model
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.minimax import MiniMaxAttention
from mlx_lm.models.minimax import Model as MiniMaxModel
from mlx_lm.models.ministral3 import Model as Ministral3Model
from mlx_lm.models.nemotron_h import Model as NemotronHModel
from mlx_lm.models.nemotron_h import (
    NemotronHAttention,
    NemotronHMamba2Mixer,
    NemotronHMoE,
)
from mlx_lm.models.nemotron_h import NemotronHModel as NemotronHInnerModel
from mlx_lm.models.qwen3_5 import DecoderLayer as Qwen3_5DecoderLayer
from mlx_lm.models.qwen3_5 import Model as Qwen3_5TextModel
from mlx_lm.models.qwen3_5 import Qwen3_5TextModel as Qwen3_5TextModelInner
from mlx_lm.models.qwen3_5 import SparseMoeBlock as Qwen3_5SparseMoeBlock
from mlx_lm.models.qwen3_5_moe import Model as Qwen3_5MoeModel
from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel
from mlx_lm.models.qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeSparseMoeBlock
from mlx_lm.models.qwen3_next import Model as Qwen3NextModel
from mlx_lm.models.qwen3_next import (
    Qwen3NextDecoderLayer,
    Qwen3NextGatedDeltaNet,
    Qwen3NextSparseMoeBlock,
)
from mlx_lm.models.qwen3_next import Qwen3NextModel as Qwen3NextInnerModel
from mlx_lm.models.step3p5 import Model as Step35Model
from mlx_lm.models.step3p5 import Step3p5MLP as Step35MLP
from mlx_lm.models.step3p5 import Step3p5Model as Step35InnerModel

from exo.shared.types.worker.shards import PipelineShardMetadata, TensorShardMode
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from mlx_lm.models.cache import Cache

TimeoutCallback = Callable[[], None]
LayerLoadedCallback = Callable[[int, int], None]  # (layers_loaded, total_layers)


_pending_prefill_sends: list[tuple[mx.array, int, mx.distributed.Group]] = []


def flush_prefill_sends() -> None:
    for output, dst, group in _pending_prefill_sends:
        sent = mx.distributed.send(output, dst, group=group)
        mx.async_eval(sent)
    _pending_prefill_sends.clear()


def clear_prefill_sends() -> None:
    # Discard pending sends (e.g. on cancellation).
    _pending_prefill_sends.clear()


def eval_with_timeout(
    mlx_item: Any,  # pyright: ignore[reportAny]
    timeout_seconds: float = 60.0,
    on_timeout: TimeoutCallback | None = None,
) -> None:
    """Evaluate MLX item with a hard timeout.

    If on_timeout callback is provided, it will be called before terminating
    the process. This allows the runner to send a failure event before exit.
    """
    completed = threading.Event()

    def watchdog() -> None:
        if not completed.wait(timeout=timeout_seconds):
            logger.error(
                f"mlx_item evaluation timed out after {timeout_seconds:.0f}s. "
                "This may indicate an issue with FAST_SYNCH and tensor parallel sharding. "
                "Terminating process."
            )
            if on_timeout is not None:
                on_timeout()
            os._exit(1)

    watchdog_thread = threading.Thread(target=watchdog, daemon=True)
    watchdog_thread.start()

    try:
        mx.eval(mlx_item)  # pyright: ignore[reportAny]
    finally:
        completed.set()


class _LayerCallable(Protocol):
    """Structural type that any compatible layer must satisfy.

    We require a single positional input of type ``mx.array`` and an
    ``mx.array`` output, while permitting arbitrary *args / **kwargs so this
    protocol matches the vast majority of `mlx.nn.Module` subclasses.
    """

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array: ...


class CustomMlxLayer(nn.Module):
    """Base class for replacing an MLX layer with a custom implementation."""

    def __init__(self, original_layer: _LayerCallable):
        super().__init__()
        dict.__setitem__(self, "_original_layer", original_layer)  # pyright: ignore[reportUnknownMemberType]

    @property
    def original_layer(self) -> _LayerCallable:
        return cast(_LayerCallable, self["_original_layer"])

    # Calls __getattr__ for any attributes not found on nn.Module (e.g. use_sliding)
    if not TYPE_CHECKING:

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                original_layer = cast(_LayerCallable, self["_original_layer"])
                return getattr(original_layer, name)


class PipelineFirstLayer(CustomMlxLayer):
    def __init__(
        self,
        original_layer: _LayerCallable,
        r: int,
        group: mx.distributed.Group,
    ):
        super().__init__(original_layer)
        self.r: int = r
        self.group = group
        self.is_prefill: bool = False

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        if self.r != 0:
            # We want to avoid GPU timeout errors by evalling the distributed operation
            # so that it stays on CPU, which does not have a timeout.
            mx.eval(x)
            x = mx.distributed.recv_like(x, (self.r - 1), group=self.group)
            mx.eval(x)
        return self.original_layer(x, *args, **kwargs)


class PipelineLastLayer(CustomMlxLayer):
    def __init__(
        self,
        original_layer: _LayerCallable,
        r: int,
        s: int,
        group: mx.distributed.Group,
    ):
        super().__init__(original_layer)
        self.r: int = r
        self.s: int = s
        self.group = group
        self.original_layer_signature = signature(self.original_layer.__call__)
        self.is_prefill: bool = False
        self.queue_sends: bool = False

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        cache = self.original_layer_signature.bind_partial(
            x, *args, **kwargs
        ).arguments.get("cache", None)

        output: mx.array = self.original_layer(x, *args, **kwargs)

        # Eval layer output to materialize it before send — this splits the graph
        # so the send is isolated and the receiving rank's recv can complete.
        mx.eval(output)

        if self.r != self.s - 1:
            if self.queue_sends:
                _pending_prefill_sends.append(
                    (output, (self.r + 1) % self.s, self.group)
                )
            else:
                output = mx.distributed.send(
                    output, (self.r + 1) % self.s, group=self.group
                )
            if cache is not None:
                # CacheList (used by MLA models like DeepSeekV32, GLM MoE DSA)
                # doesn't have .keys directly; access via first sub-cache.
                _cache = cache[0] if hasattr(cache, "caches") else cache  # type: ignore
                if hasattr(_cache, "keys"):  # pyright: ignore[reportAny]
                    _cache.keys = mx.depends(_cache.keys, output)  # type: ignore
            mx.eval(output)
            if cache is not None and hasattr(_cache, "keys"):  # type: ignore
                mx.eval(_cache.keys)  # type: ignore

        if not self.is_prefill:
            output = mx.distributed.all_gather(output, group=self.group)[
                -output.shape[0] :
            ]
            mx.eval(output)

        return output


def set_pipeline_prefill(model: nn.Module, is_prefill: bool) -> None:
    for layer in model.layers:  # type: ignore
        if isinstance(layer, (PipelineFirstLayer, PipelineLastLayer)):
            layer.is_prefill = is_prefill


def set_pipeline_queue_sends(model: nn.Module, queue_sends: bool) -> None:
    for layer in model.layers:  # type: ignore
        if isinstance(layer, PipelineLastLayer):
            layer.queue_sends = queue_sends


def get_inner_model(model: nn.Module) -> nn.Module:
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module):
        return inner

    inner = getattr(model, "transformer", None)
    if isinstance(inner, nn.Module):
        return inner

    inner = getattr(model, "language_model", None)
    if isinstance(inner, nn.Module):
        inner_inner = getattr(inner, "model", None)
        if isinstance(inner_inner, nn.Module):
            return inner_inner

    inner = getattr(model, "backbone", None)
    if isinstance(inner, nn.Module):
        return inner

    raise ValueError(
        "Model must either have a 'model', 'transformer', or 'backbone' attribute"
    )


def get_layers(inner_model_instance: nn.Module) -> list[_LayerCallable]:
    # Handle both model.layers and model.h cases
    layers: list[_LayerCallable]
    if hasattr(inner_model_instance, "layers"):
        layers = cast(list[_LayerCallable], inner_model_instance.layers)
    elif hasattr(inner_model_instance, "h"):
        layers = cast(list[_LayerCallable], inner_model_instance.h)
    else:
        raise ValueError("Model must have either a 'layers' or 'h' attribute")

    return layers


def _patch_hybrid_cache(
    model: Qwen3_5TextModel | Qwen3NextModel | NemotronHModel,
    fa_idx: int,
    has_full_attn: bool,
    ssm_idx: int,
    has_linear: bool,
) -> None:
    # Hacks to make make_mask happy.
    original = model.make_cache

    def patched() -> list[ArraysCache | KVCache]:
        cache = original()
        if not has_full_attn:
            entry = cache[fa_idx]
            orig_make_mask = entry.make_mask
            entry.make_mask = lambda n, **_kw: orig_make_mask(n)  # type: ignore
        if not has_linear:
            orig_ssm_make_mask = cache[ssm_idx].make_mask

            def _ssm_mask(
                n: int, **kw: bool | int | None
            ) -> mx.array | Literal["causal"] | None:
                return orig_ssm_make_mask(n, **kw) if kw else None

            cache[ssm_idx].make_mask = _ssm_mask  # type: ignore
        return cache

    model.make_cache = patched


def pipeline_auto_parallel(
    model: nn.Module,
    group: mx.distributed.Group,
    model_shard_meta: PipelineShardMetadata,
    on_layer_loaded: LayerLoadedCallback | None,
) -> nn.Module:
    """
    Automatically parallelize a model across multiple devices.
    Args:
    model: The model to parallelize (must have a 'layers' or 'h' property)
    model_shard_meta: The metadata for the model shard
    Returns:
    The parallelized model
    """
    inner_model_instance: nn.Module = get_inner_model(model)

    layers = get_layers(inner_model_instance)

    start_layer, end_layer = model_shard_meta.start_layer, model_shard_meta.end_layer
    device_rank, world_size = model_shard_meta.device_rank, model_shard_meta.world_size

    layers = layers[start_layer:end_layer]
    total = len(layers)
    for i, layer in enumerate(layers):
        mx.eval(layer)  # type: ignore
        if on_layer_loaded is not None:
            on_layer_loaded(i, total)

    layers[0] = PipelineFirstLayer(layers[0], device_rank, group=group)
    layers[-1] = PipelineLastLayer(
        layers[-1],
        device_rank,
        world_size,
        group=group,
    )

    if isinstance(inner_model_instance, GptOssMoeModel):
        inner_model_instance.layer_types = inner_model_instance.layer_types[  # type: ignore
            start_layer:end_layer
        ]
        # We can assume the model has at least one layer thanks to placement.
        # If a layer type doesn't exist, we can set it to 0.
        inner_model_instance.swa_idx = (
            0
            if "sliding_attention" not in inner_model_instance.layer_types  # type: ignore
            else inner_model_instance.layer_types.index(  # type: ignore
                "sliding_attention"
            )
        )
        inner_model_instance.ga_idx = (
            0
            if "full_attention" not in inner_model_instance.layer_types  # type: ignore
            else inner_model_instance.layer_types.index(  # type: ignore
                "full_attention"
            )
        )

    if isinstance(inner_model_instance, Step35InnerModel):
        inner_model_instance.num_layers = len(layers)
        sliding_layers = [
            i for i, layer in enumerate(layers) if getattr(layer, "is_sliding", False)
        ]
        full_layers = [
            i
            for i, layer in enumerate(layers)
            if not getattr(layer, "is_sliding", True)
        ]
        inner_model_instance._swa_idx = 0 if not sliding_layers else sliding_layers[0]
        inner_model_instance._full_idx = 0 if not full_layers else full_layers[0]

    if isinstance(inner_model_instance, (Qwen3_5TextModelInner, Qwen3NextInnerModel)):
        full_attn_layers = [
            i for i, layer in enumerate(layers) if not getattr(layer, "is_linear", True)
        ]
        linear_layers = [
            i for i, layer in enumerate(layers) if getattr(layer, "is_linear", False)
        ]
        inner_model_instance.fa_idx = full_attn_layers[0] if full_attn_layers else 0
        inner_model_instance.ssm_idx = linear_layers[0] if linear_layers else 0
        if not full_attn_layers or not linear_layers:
            _patch_hybrid_cache(
                cast(Qwen3_5TextModel | Qwen3NextModel, model),
                fa_idx=inner_model_instance.fa_idx,
                has_full_attn=bool(full_attn_layers),
                ssm_idx=inner_model_instance.ssm_idx,
                has_linear=bool(linear_layers),
            )

    if isinstance(inner_model_instance, NemotronHInnerModel):
        # NemotronH uses block_type: "M" (Mamba/SSM), "*" (Attention), "E" (MoE), "-" (MLP)
        # Only "M" and "*" blocks have cache entries.
        # Recompute fa_idx and ssm_idx as cache-array indices for the shard's layers.
        cache_idx = 0
        fa_idx: int | None = None
        ssm_idx: int | None = None
        for layer in layers:
            block_type = getattr(layer, "block_type", None)
            if block_type == "*":
                if fa_idx is None:
                    fa_idx = cache_idx
                cache_idx += 1
            elif block_type == "M":
                if ssm_idx is None:
                    ssm_idx = cache_idx
                cache_idx += 1
        has_attn = fa_idx is not None
        has_mamba = ssm_idx is not None
        inner_model_instance.fa_idx = fa_idx if fa_idx is not None else 0
        inner_model_instance.ssm_idx = ssm_idx if ssm_idx is not None else 0
        if not has_attn or not has_mamba:
            _patch_hybrid_cache(
                cast(NemotronHModel, model),
                fa_idx=inner_model_instance.fa_idx,
                has_full_attn=has_attn,
                ssm_idx=inner_model_instance.ssm_idx,
                has_linear=has_mamba,
            )

    _set_layers(model, layers)

    assert isinstance(layers, list), (
        "Expected a list of layers after auto-parallel initialisation"
    )

    return patch_pipeline_model(model, group)


def patch_pipeline_model[T](model: T, group: mx.distributed.Group) -> T:
    # Patch __call__ on the model's class
    cls = model.__class__
    original_call = cls.__call__  # type :ignore
    call_signature = signature(original_call)  # type :ignore

    def patched_call(
        self: T,
        *args: object,
        **kwargs: object,
    ) -> mx.array:
        logits: mx.array = original_call(self, *args, **kwargs)  # type: ignore
        cache = call_signature.bind_partial(self, *args, **kwargs).arguments.get(
            "cache", None
        )

        # Add dependency to last cache entry to ensure distributed ops are evaluated
        if cache is not None:
            last = cache[-1]  # type: ignore
            dep_cache = last[0] if hasattr(last, "caches") else last  # type: ignore
            if hasattr(dep_cache, "keys") and dep_cache.keys is not None:  # type: ignore
                dep_cache.keys = mx.depends(dep_cache.keys, logits)  # type: ignore

        return logits

    cls.__call__ = patched_call
    return model


def patch_tensor_model[T](model: T) -> T:
    """Patch model's __call__ to ensure distributed ops sync during inference."""
    cls = model.__class__
    original_call = cls.__call__
    call_signature = signature(original_call)

    def patched_call(
        self: T,
        *args: object,
        **kwargs: object,
    ) -> mx.array:
        logits: mx.array = original_call(self, *args, **kwargs)  # pyright: ignore[reportAny]
        cache = call_signature.bind_partial(self, *args, **kwargs).arguments.get(
            "cache", None
        )

        # Add dependency to last cache entry to ensure distributed ops are evaluated
        if cache is not None and len(cache) > 0:  # pyright: ignore[reportAny]
            last = cache[-1]  # pyright: ignore[reportAny]
            dep_cache = last[0] if hasattr(last, "caches") else last  # pyright: ignore[reportAny]
            if hasattr(dep_cache, "keys"):  # type: ignore
                dep_cache.keys = mx.depends(dep_cache.keys, logits)  # pyright: ignore[reportAny]

        return logits

    cls.__call__ = patched_call
    return model


def _kv_balanced_q_sizes(
    q_dim: int, N: int, gqa_repeat: int, head_unit: int, num_kv: int
) -> list[int]:
    group_elems = gqa_repeat * head_unit
    if q_dim // group_elems >= N:
        return compute_shard_sizes(q_dim, N, group_elems)
    best_m = 1
    best_diff = float("inf")
    for m in range(1, gqa_repeat):
        side = (gqa_repeat - m) * head_unit
        mid = m * num_kv * head_unit
        if abs(side - mid) < best_diff:
            best_diff = abs(side - mid)
            best_m = m
    side = (gqa_repeat - best_m) * head_unit
    mid = best_m * num_kv * head_unit
    return [side, mid, side]


def _slice_kv_proj(module: nn.Module, kv_start: int, kv_end: int, head_dim: int) -> None:
    params = module.parameters()
    row_start = kv_start * head_dim
    row_end = kv_end * head_dim
    gs = getattr(module, "group_size", 0)
    if gs > 0 and "scales" in params:
        module.weight = params["weight"][row_start:row_end]
        module.scales = params["scales"][row_start:row_end]
        if "biases" in params:
            module.biases = params["biases"][row_start:row_end]
    else:
        module.weight = params["weight"][row_start:row_end]
    if "bias" in params and params["bias"].ndim > 0:
        module.bias = params["bias"][row_start:row_end]


def tensor_auto_parallel(
    model: nn.Module,
    group: mx.distributed.Group,
    timeout_seconds: float,
    on_timeout: TimeoutCallback | None,
    on_layer_loaded: LayerLoadedCallback | None,
    shard_weights: list[float] | None = None,
    shard_mode: TensorShardMode | None = None,
) -> nn.Module:
    resolved_shard_mode = shard_mode or TensorShardMode.Constant
    all_to_sharded_linear = partial(
        shard_linear,
        sharding="all-to-sharded",
        group=group,
        weights=shard_weights,
    )
    sharded_to_all_linear = partial(
        shard_linear,
        sharding="sharded-to-all",
        group=group,
        weights=shard_weights,
    )

    segments: int = 1

    def _all_to_sharded(path: str, weight: mx.array):
        if path.endswith("bias"):
            logger.info(f"Sharding bias for {path} - all to sharded")
            return weight.ndim - 1, segments
        return max(weight.ndim - 2, 0), segments

    all_to_sharded_linear_in_place = partial(
        shard_inplace,
        sharding=_all_to_sharded,  # type: ignore
        group=group,
        weights=shard_weights,
    )

    n = group.size()

    def _sharded_to_all(path: str, weight: mx.array):
        if path.endswith("bias"):
            logger.info(f"Sharding bias for {path} - sharded to all")
            weight /= n
            return None
        return -1, segments

    _base_sharded_to_all_in_place = partial(
        shard_inplace,
        sharding=_sharded_to_all,  # type: ignore
        group=group,
        weights=shard_weights,
    )

    _base_all_to_sharded_in_place = all_to_sharded_linear_in_place

    def _quantized_moe_shard_inplace(
        module: nn.Module,
        sharding: Literal["all-to-sharded", "sharded-to-all"],
        weights: list[float] | None = None,
    ) -> None:
        N = group.size()
        r = group.rank()
        gs = module.group_size  # pyright: ignore[reportAttributeAccessIssue]
        bits = module.bits  # pyright: ignore[reportAttributeAccessIssue]
        params = module.parameters()
        scales = params["scales"]

        if sharding == "all-to-sharded":
            dim = params["weight"].shape[max(params["weight"].ndim - 2, 0)]
            sizes = compute_shard_sizes(dim, N, gs, weights)
            result: dict[str, Any] = {}
            for key, param in params.items():
                if not isinstance(param, mx.array):
                    result[key] = param
                    continue
                axis = max(param.ndim - 2, 0)
                indices = [sum(sizes[:i]) for i in range(1, len(sizes))]
                result[key] = mx.contiguous(mx.split(param, indices, axis=axis)[r])
        else:
            num_groups = scales.shape[-1]
            group_counts = compute_shard_sizes(num_groups, N, 1, weights)
            weight_ppg = gs * bits // 32
            result = {}
            for key, param in params.items():
                if not isinstance(param, mx.array):
                    result[key] = param
                    continue
                if key == "weight":
                    s = [gc * weight_ppg for gc in group_counts]
                elif key in ("scales", "biases"):
                    s = list(group_counts)
                else:
                    result[key] = param
                    continue
                indices = [sum(s[:i]) for i in range(1, len(s))]
                result[key] = mx.contiguous(mx.split(param, indices, axis=-1)[r])
        module.update(result)

    def all_to_sharded_linear_in_place(module: nn.Module, **kwargs: Any) -> None:
        if getattr(module, "group_size", 0) > 0 and getattr(module, "bits", 0) > 0 and "scales" in module.parameters():
            _quantized_moe_shard_inplace(module, "all-to-sharded", weights=kwargs.get("weights"))
        else:
            _base_all_to_sharded_in_place(module, **kwargs)

    def sharded_to_all_linear_in_place(module: nn.Module, **kwargs: Any) -> None:
        if getattr(module, "group_size", 0) > 0 and getattr(module, "bits", 0) > 0 and "scales" in module.parameters():
            _quantized_moe_shard_inplace(module, "sharded-to-all", weights=kwargs.get("weights"))
        else:
            _base_sharded_to_all_in_place(module, **kwargs)

    if isinstance(model, (LlamaModel, Ministral3Model)):
        tensor_parallel_sharding_strategy = LlamaShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
            shard_weights=shard_weights,
            shard_mode=resolved_shard_mode,
        )
    elif isinstance(model, (DeepseekV3Model, DeepseekV32Model, KimiK25Model)):
        tensor_parallel_sharding_strategy = DeepSeekShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
            shard_weights=shard_weights,
            shard_mode=resolved_shard_mode,
        )
    elif isinstance(model, MiniMaxModel):
        tensor_parallel_sharding_strategy = MiniMaxShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
            shard_weights=shard_weights,
            shard_mode=resolved_shard_mode,
        )
    elif isinstance(model, GLM4MoeLiteModel):
        tensor_parallel_sharding_strategy = GLM4MoeLiteShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
            shard_weights=shard_weights,
            shard_mode=resolved_shard_mode,
        )
    elif isinstance(model, Glm4MoeModel):
        tensor_parallel_sharding_strategy = Glm4MoeShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
            shard_weights=shard_weights,
            shard_mode=resolved_shard_mode,
        )
    elif isinstance(
        model, (Qwen3MoeModel, Qwen3NextModel, Qwen3_5TextModel, Qwen3_5MoeModel)
    ):
        tensor_parallel_sharding_strategy = QwenShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
            shard_weights=shard_weights,
            shard_mode=resolved_shard_mode,
        )
    elif isinstance(model, GptOssModel):
        tensor_parallel_sharding_strategy = GptOssShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
            shard_weights=shard_weights,
            shard_mode=resolved_shard_mode,
        )
    elif isinstance(model, Step35Model):
        tensor_parallel_sharding_strategy = Step35ShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
            shard_weights=shard_weights,
            shard_mode=resolved_shard_mode,
        )
    elif isinstance(model, NemotronHModel):
        tensor_parallel_sharding_strategy = NemotronHShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
            shard_weights=shard_weights,
            shard_mode=resolved_shard_mode,
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    model = tensor_parallel_sharding_strategy.shard_model(
        model, timeout_seconds, on_timeout, on_layer_loaded
    )
    return patch_tensor_model(model)


class TensorParallelShardingStrategy(ABC):
    def __init__(
        self,
        group: mx.distributed.Group,
        all_to_sharded_linear: Callable[..., nn.Linear],
        sharded_to_all_linear: Callable[..., nn.Linear],
        all_to_sharded_linear_in_place: Callable[..., None],
        sharded_to_all_linear_in_place: Callable[..., None],
        shard_weights: list[float] | None = None,
        shard_mode: TensorShardMode = TensorShardMode.Constant,
    ):
        self._base_all_to_sharded_linear = all_to_sharded_linear
        self._base_sharded_to_all_linear = sharded_to_all_linear
        self._base_all_to_sharded_linear_in_place = all_to_sharded_linear_in_place
        self._base_sharded_to_all_linear_in_place = sharded_to_all_linear_in_place
        self.shard_weights = shard_weights
        self.shard_mode = shard_mode
        self.group = group
        self.N = group.size()
        self._greedy_trackers: dict[str, list[list[float]] | list[float]] | None = None
        if shard_weights is not None and shard_mode == TensorShardMode.Greedy:
            self._greedy_trackers = {}

    def _greedy_weights_for(
        self, key: str, dim: int, unit: int = 1
    ) -> list[float] | None:
        """Get adjusted weights for a specific projection type, and record the allocation."""
        if self.shard_weights is None or self._greedy_trackers is None:
            return self.shard_weights
        n = len(self.shard_weights)
        total_w = sum(self.shard_weights)
        target = [dim * self.shard_weights[i] / total_w for i in range(n)]
        if key not in self._greedy_trackers:
            self._greedy_trackers[key] = [[0.0] * n, [0.0] * n, [0] * n]
        tracker = cast(list[list[float]], self._greedy_trackers[key])
        cum_target, cum_actual, last_sizes = tracker[0], tracker[1], tracker[2]
        desired: list[float] = [
            target[i] + (cum_target[i] - cum_actual[i]) for i in range(n)
        ]
        min_d = min(desired)
        if min_d <= 0:
            desired = [d - min_d + 0.01 for d in desired]
        actual_sizes = compute_shard_sizes(dim, n, unit, desired)
        for i in range(n):
            cum_target[i] += target[i]
            cum_actual[i] += actual_sizes[i]
            last_sizes[i] = actual_sizes[i]
        self._greedy_trackers[key + "_last_weights"] = desired
        return desired

    def _greedy_last_sizes(self, key: str) -> list[int]:
        if self._greedy_trackers is None or key not in self._greedy_trackers:
            return []
        tracker = self._greedy_trackers[key]
        assert isinstance(tracker, list) and len(tracker) == 3  # noqa: S101
        return cast(list[int], tracker[2])

    def _greedy_last_weights(self, key: str) -> list[float] | None:
        if self._greedy_trackers is None:
            return self.shard_weights
        w = self._greedy_trackers.get(key + "_last_weights")
        return cast(list[float] | None, w) if w is not None else self.shard_weights

    @property
    def all_to_sharded_linear(self) -> Callable[..., nn.Linear]:
        return self._base_all_to_sharded_linear

    @property
    def sharded_to_all_linear(self) -> Callable[..., nn.Linear]:
        return self._base_sharded_to_all_linear

    @property
    def all_to_sharded_linear_in_place(self) -> Callable[..., None]:
        return self._base_all_to_sharded_linear_in_place

    @property
    def sharded_to_all_linear_in_place(self) -> Callable[..., None]:
        return self._base_sharded_to_all_linear_in_place

    @abstractmethod
    def shard_model(
        self,
        model: nn.Module,
        timeout_seconds: float,
        on_timeout: TimeoutCallback | None,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module: ...


class LlamaShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        timeout_seconds: float,
        on_timeout: TimeoutCallback | None,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(LlamaModel, model)
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            # Force load weights before sharding to avoid FAST_SYNCH deadlock
            eval_with_timeout(layer.parameters(), timeout_seconds / total, on_timeout)
            head_dim = layer.self_attn.head_dim
            n_kv = layer.self_attn.n_kv_heads or layer.self_attn.n_heads
            assert layer.self_attn.n_heads % n_kv == 0, "Breaks assumptions"
            q_dim = layer.self_attn.q_proj.weight.shape[0]
            k_dim = layer.self_attn.k_proj.weight.shape[0]
            intermediate = layer.mlp.gate_proj.weight.shape[0]
            if n_kv >= self.N:
                gqa_unit = head_dim * (layer.self_attn.n_heads // n_kv)
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj,
                    unit=gqa_unit,
                    weights=self._greedy_weights_for("q", q_dim, gqa_unit),
                )
                layer.self_attn.k_proj = self.all_to_sharded_linear(
                    layer.self_attn.k_proj,
                    unit=head_dim,
                    weights=self._greedy_weights_for("k", k_dim, head_dim),
                )
                layer.self_attn.v_proj = self.all_to_sharded_linear(
                    layer.self_attn.v_proj,
                    unit=head_dim,
                    weights=self._greedy_weights_for("v", k_dim, head_dim),
                )
                layer.self_attn.o_proj = self.sharded_to_all_linear(
                    layer.self_attn.o_proj,
                    unit=gqa_unit,
                    weights=self._greedy_weights_for("o", q_dim, gqa_unit),
                )
                layer.self_attn.n_kv_heads = (
                    layer.self_attn.k_proj.weight.shape[0] // head_dim
                )
            else:
                q_unit = head_dim
                gqa_repeat = layer.self_attn.n_heads // n_kv
                kv_bal = _kv_balanced_q_sizes(q_dim, self.N, gqa_repeat, head_dim, n_kv)
                kv_bal_w = [float(s) for s in kv_bal]
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj,
                    unit=q_unit,
                    weights=kv_bal_w,
                )
                layer.self_attn.o_proj = self.sharded_to_all_linear(
                    layer.self_attn.o_proj,
                    unit=q_unit,
                    weights=kv_bal_w,
                )
                local_q_heads = layer.self_attn.q_proj.weight.shape[0] // head_dim
                q_offset = sum(kv_bal[: self.group.rank()])
                start_head = q_offset // head_dim
                kv_start = start_head // gqa_repeat
                kv_end = (start_head + local_q_heads - 1) // gqa_repeat + 1
                _slice_kv_proj(layer.self_attn.k_proj, kv_start, kv_end, head_dim)
                _slice_kv_proj(layer.self_attn.v_proj, kv_start, kv_end, head_dim)
                layer.self_attn.n_kv_heads = kv_end - kv_start
            layer.self_attn.n_heads = layer.self_attn.q_proj.weight.shape[0] // head_dim

            mlp_unit = getattr(layer.mlp.gate_proj, "group_size", 1)
            layer.mlp.gate_proj = self.all_to_sharded_linear(
                layer.mlp.gate_proj,
                unit=mlp_unit,
                weights=self._greedy_weights_for("gate", intermediate),
            )
            layer.mlp.down_proj = self.sharded_to_all_linear(
                layer.mlp.down_proj,
                unit=mlp_unit,
                weights=self._greedy_weights_for("down", intermediate),
            )
            layer.mlp.up_proj = self.all_to_sharded_linear(
                layer.mlp.up_proj,
                unit=mlp_unit,
                weights=self._greedy_weights_for("up", intermediate),
            )
            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(i, total)
        return model


def _set_layers(model: nn.Module, layers: list[_LayerCallable]) -> None:
    inner_model_instance = get_inner_model(model)
    if hasattr(inner_model_instance, "layers"):
        inner_model_instance.layers = layers

        # Update DeepSeek V3 specific parameters when layers are shrunk
        if isinstance(
            model, (DeepseekV3Model, DeepseekV32Model, Glm4MoeModel, KimiK25Model)
        ) and hasattr(inner_model_instance, "num_layers"):
            logger.info(
                f"Setting num_layers to {len(layers)} for model {model.model.__class__.__name__}"
            )
            inner_model_instance.start_idx = 0
            inner_model_instance.end_idx = len(layers)
            inner_model_instance.num_layers = len(layers)
        elif isinstance(model, Qwen3MoeModel):
            logger.info(
                f"Setting num_hidden_layers to {len(layers)} for model {model.model.__class__.__name__}"
            )
            inner_model_instance.num_hidden_layers = len(layers)
    elif hasattr(inner_model_instance, "h"):
        inner_model_instance.h = layers
    else:
        raise ValueError("Model must have either a 'layers' or 'h' attribute")


class DeepSeekShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        timeout_seconds: float,
        on_timeout: TimeoutCallback | None,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(DeepseekV3Model, model)
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            eval_with_timeout(layer.parameters(), timeout_seconds / total, on_timeout)

            # Shard the self attention
            original_num_heads = layer.self_attn.num_heads
            q_head_dim = (
                layer.self_attn.q_b_proj.weight.shape[0] // original_num_heads
                if layer.self_attn.q_lora_rank is not None
                else layer.self_attn.q_proj.weight.shape[0] // original_num_heads
            )
            q_dim = (
                layer.self_attn.q_proj.weight.shape[0]
                if layer.self_attn.q_lora_rank is None
                else layer.self_attn.q_b_proj.weight.shape[0]
            )
            o_dim = q_dim
            if layer.self_attn.q_lora_rank is None:
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj,
                    unit=q_head_dim,
                    weights=self._greedy_weights_for("q", q_dim, q_head_dim),
                )
            else:
                layer.self_attn.q_b_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_b_proj,
                    unit=q_head_dim,
                    weights=self._greedy_weights_for("q", q_dim, q_head_dim),
                )

            layer.self_attn.o_proj = self.sharded_to_all_linear(
                layer.self_attn.o_proj,
                unit=q_head_dim,
                weights=self._greedy_weights_for("o", o_dim, q_head_dim),
            )
            q_actual = self._greedy_last_sizes("q")
            head_sizes = (
                [s // q_head_dim for s in q_actual]
                if q_actual
                else compute_shard_sizes(
                    original_num_heads, self.N, weights=self.shard_weights
                )
            )
            layer.self_attn.num_heads = head_sizes[self.group.rank()]

            # Logic from upstream mlx
            sh = sum(head_sizes[: self.group.rank()])
            eh = sh + head_sizes[self.group.rank()]

            def shard_heads(w: mx.array, sh: int = sh, eh: int = eh) -> mx.array:
                return w[sh:eh]

            layer.self_attn.embed_q.apply(shard_heads)
            layer.self_attn.unembed_out.apply(shard_heads)

            # Shard the MLP
            if isinstance(layer.mlp, (DeepseekV3MLP, DeepseekV32MLP)):
                intermediate = layer.mlp.gate_proj.weight.shape[0]
                mlp_unit = getattr(layer.mlp.gate_proj, "group_size", 1)
                layer.mlp.gate_proj = self.all_to_sharded_linear(
                    layer.mlp.gate_proj,
                    unit=mlp_unit,
                    weights=self._greedy_weights_for("gate", intermediate),
                )
                layer.mlp.down_proj = self.sharded_to_all_linear(
                    layer.mlp.down_proj,
                    unit=mlp_unit,
                    weights=self._greedy_weights_for("down", intermediate),
                )
                layer.mlp.up_proj = self.all_to_sharded_linear(
                    layer.mlp.up_proj,
                    unit=mlp_unit,
                    weights=self._greedy_weights_for("up", intermediate),
                )

            # Shard the MoE.
            else:
                if getattr(layer.mlp, "shared_experts", None) is not None:
                    shared_gate_dim = layer.mlp.shared_experts.gate_proj.weight.shape[0]
                    shared_down_dim = layer.mlp.shared_experts.down_proj.weight.shape[
                        -1
                    ]
                    shared_up_dim = layer.mlp.shared_experts.up_proj.weight.shape[0]
                    shared_greedy = self._greedy_weights_for("shared_gate", shared_gate_dim)
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.gate_proj,
                        weights=shared_greedy,
                    )
                    self.sharded_to_all_linear_in_place(
                        layer.mlp.shared_experts.down_proj,
                        weights=shared_greedy,
                    )
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.up_proj,
                        weights=shared_greedy,
                    )
                moe_gate_dim = int(layer.mlp.switch_mlp.gate_proj.weight.shape[1])
                moe_greedy = self._greedy_weights_for("moe_gate", moe_gate_dim)
                self.all_to_sharded_linear_in_place(
                    layer.mlp.switch_mlp.gate_proj,
                    weights=moe_greedy,
                )
                self.sharded_to_all_linear_in_place(
                    layer.mlp.switch_mlp.down_proj,
                    weights=moe_greedy,
                )
                self.all_to_sharded_linear_in_place(
                    layer.mlp.switch_mlp.up_proj,
                    weights=moe_greedy,
                )
                layer.mlp = ShardedMoE(layer.mlp)  # type: ignore
                layer.mlp.sharding_group = self.group

            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(i, total)

        return model


class ShardedMoE(CustomMlxLayer):
    """Wraps any MoE layer with distributed sum_gradients / all_sum."""

    def __init__(self, layer: _LayerCallable):
        super().__init__(layer)
        self.sharding_group: mx.distributed.Group | None = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        y = self.original_layer.__call__(x)
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


class GLM4MoeLiteShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        timeout_seconds: float,
        on_timeout: TimeoutCallback | None,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(GLM4MoeLiteModel, model)
        total = len(model.layers)  # type: ignore
        for i, layer in enumerate(model.layers):  # type: ignore
            layer = cast(Glm4MoeLiteDecoderLayer, layer)
            eval_with_timeout(
                layer.parameters(),
                timeout_seconds / total,
                on_timeout,
            )
            original_num_heads = layer.self_attn.num_heads
            q_head_dim = (
                layer.self_attn.q_b_proj.weight.shape[0] // original_num_heads
                if layer.self_attn.q_lora_rank is not None  # pyright: ignore[reportUnnecessaryComparison]
                else layer.self_attn.q_proj.weight.shape[0] // original_num_heads
            )
            q_dim = (
                layer.self_attn.q_proj.weight.shape[0]
                if layer.self_attn.q_lora_rank is None  # pyright: ignore[reportUnnecessaryComparison]
                else layer.self_attn.q_b_proj.weight.shape[0]
            )
            o_dim = q_dim
            if layer.self_attn.q_lora_rank is None:  # pyright: ignore[reportUnnecessaryComparison]
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj,
                    unit=q_head_dim,
                    weights=self._greedy_weights_for("q", q_dim, q_head_dim),
                )
            else:
                layer.self_attn.q_b_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_b_proj,
                    unit=q_head_dim,
                    weights=self._greedy_weights_for("q", q_dim, q_head_dim),
                )

            layer.self_attn.o_proj = self.sharded_to_all_linear(
                layer.self_attn.o_proj,
                unit=q_head_dim,
                weights=self._greedy_weights_for("o", o_dim, q_head_dim),
            )
            q_actual = self._greedy_last_sizes("q")
            head_sizes = (
                [s // q_head_dim for s in q_actual]
                if q_actual
                else compute_shard_sizes(
                    original_num_heads, self.N, weights=self.shard_weights
                )
            )
            layer.self_attn.num_heads = head_sizes[self.group.rank()]

            # Logic from upstream mlx
            sh = sum(head_sizes[: self.group.rank()])
            eh = sh + head_sizes[self.group.rank()]

            def shard_heads(w: mx.array, sh: int = sh, eh: int = eh) -> mx.array:
                return w[sh:eh]

            layer.self_attn.embed_q.apply(shard_heads)
            layer.self_attn.unembed_out.apply(shard_heads)

            if isinstance(layer.mlp, Glm4MoeLiteMLP):
                intermediate = layer.mlp.gate_proj.weight.shape[0]
                mlp_unit = getattr(layer.mlp.gate_proj, "group_size", 1)
                layer.mlp.gate_proj = self.all_to_sharded_linear(
                    layer.mlp.gate_proj,
                    unit=mlp_unit,
                    weights=self._greedy_weights_for("gate", intermediate),
                )
                layer.mlp.down_proj = self.sharded_to_all_linear(
                    layer.mlp.down_proj,
                    unit=mlp_unit,
                    weights=self._greedy_weights_for("down", intermediate),
                )
                layer.mlp.up_proj = self.all_to_sharded_linear(
                    layer.mlp.up_proj,
                    unit=mlp_unit,
                    weights=self._greedy_weights_for("up", intermediate),
                )

            else:
                if getattr(layer.mlp, "shared_experts", None) is not None:
                    shared_gate_dim = layer.mlp.shared_experts.gate_proj.weight.shape[0]
                    shared_down_dim = layer.mlp.shared_experts.down_proj.weight.shape[
                        -1
                    ]
                    shared_up_dim = layer.mlp.shared_experts.up_proj.weight.shape[0]
                    shared_greedy = self._greedy_weights_for("shared_gate", shared_gate_dim)
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.gate_proj,
                        weights=shared_greedy,
                    )
                    self.sharded_to_all_linear_in_place(
                        layer.mlp.shared_experts.down_proj,
                        weights=shared_greedy,
                    )
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.up_proj,
                        weights=shared_greedy,
                    )
                moe_gate_dim = int(layer.mlp.switch_mlp.gate_proj.weight.shape[1])
                moe_greedy = self._greedy_weights_for("moe_gate", moe_gate_dim)
                self.all_to_sharded_linear_in_place(
                    layer.mlp.switch_mlp.gate_proj,
                    weights=moe_greedy,
                )
                self.sharded_to_all_linear_in_place(
                    layer.mlp.switch_mlp.down_proj,
                    weights=moe_greedy,
                )
                self.all_to_sharded_linear_in_place(
                    layer.mlp.switch_mlp.up_proj,
                    weights=moe_greedy,
                )
                layer.mlp = ShardedMoE(layer.mlp)  # type: ignore
                layer.mlp.sharding_group = self.group  # type: ignore
            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(i, total)

        return model


class WrappedMiniMaxAttention(CustomMlxLayer):
    def __init__(self, layer: _LayerCallable, group: mx.distributed.Group):
        super().__init__(layer)
        self.group = group

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: "Cache | None" = None,
    ) -> mx.array:
        batch_dim, seq_dim, _ = x.shape

        self._original_layer = cast(MiniMaxAttention, self.original_layer)  # type: ignore

        queries: mx.array = self._original_layer.q_proj(x)
        keys: mx.array = self._original_layer.k_proj(x)
        values: mx.array = self._original_layer.v_proj(x)

        if getattr(self, "use_qk_norm", False):
            q_dim = queries.shape[-1]
            k_dim = keys.shape[-1]
            n = self.group.size()

            qk = mx.concatenate(
                [queries, keys], axis=-1
            )  # (batch_dim, seq_dim, q_dim + k_dim)
            qk = mx.distributed.all_gather(
                qk, group=self.group
            )  # (n*batch_dim, seq_dim, q_dim + k_dim)

            qk = qk.reshape(n, batch_dim, seq_dim, q_dim + k_dim).transpose(1, 2, 0, 3)
            queries = qk[..., :q_dim].reshape(
                batch_dim, seq_dim, -1
            )  # (batch_dim, seq_dim, n * q_dim)
            keys = qk[..., q_dim:].reshape(
                batch_dim, seq_dim, -1
            )  # (batch_dim, seq_dim, n * k_dim)

            queries = self._original_layer.q_norm(queries)
            keys = self._original_layer.k_norm(keys)

            # Split back and take this rank's portion
            queries = mx.split(queries, n, axis=-1)[self.group.rank()]
            keys = mx.split(keys, n, axis=-1)[self.group.rank()]

        queries = queries.reshape(
            batch_dim, seq_dim, self._original_layer.num_attention_heads, -1
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            batch_dim, seq_dim, self._original_layer.num_key_value_heads, -1
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_dim, seq_dim, self._original_layer.num_key_value_heads, -1
        ).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self._original_layer.rope(queries, offset=cache.offset)
            keys = self._original_layer.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self._original_layer.rope(queries)
            keys = self._original_layer.rope(keys)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self._original_layer.scale,  # type: ignore
            mask=mask,
        )

        output = output.transpose(0, 2, 1, 3).reshape(batch_dim, seq_dim, -1)

        return self._original_layer.o_proj(output)


class MiniMaxShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        timeout_seconds: float,
        on_timeout: TimeoutCallback | None,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(MiniMaxModel, model)
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            eval_with_timeout(layer.parameters(), timeout_seconds / total, on_timeout)
            # Shard the self attention
            head_dim = layer.self_attn.head_dim
            n_kv = layer.self_attn.num_key_value_heads
            q_dim = layer.self_attn.q_proj.weight.shape[0]
            k_dim = layer.self_attn.k_proj.weight.shape[0]
            if n_kv >= self.N:
                gqa_unit = head_dim * (
                    layer.self_attn.num_attention_heads // n_kv
                )
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj,
                    unit=gqa_unit,
                    weights=self._greedy_weights_for("q", q_dim, gqa_unit),
                )
                layer.self_attn.k_proj = self.all_to_sharded_linear(
                    layer.self_attn.k_proj,
                    unit=head_dim,
                    weights=self._greedy_weights_for("k", k_dim, head_dim),
                )
                layer.self_attn.v_proj = self.all_to_sharded_linear(
                    layer.self_attn.v_proj,
                    unit=head_dim,
                    weights=self._greedy_weights_for("v", k_dim, head_dim),
                )
                layer.self_attn.o_proj = self.sharded_to_all_linear(
                    layer.self_attn.o_proj,
                    unit=gqa_unit,
                    weights=self._greedy_weights_for("o", q_dim, gqa_unit),
                )
                layer.self_attn.num_key_value_heads = (
                    layer.self_attn.k_proj.weight.shape[0] // head_dim
                )
            else:
                q_unit = head_dim
                gqa_repeat = layer.self_attn.num_attention_heads // n_kv
                kv_bal = _kv_balanced_q_sizes(q_dim, self.N, gqa_repeat, head_dim, n_kv)
                kv_bal_w = [float(s) for s in kv_bal]
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj,
                    unit=q_unit,
                    weights=kv_bal_w,
                )
                layer.self_attn.o_proj = self.sharded_to_all_linear(
                    layer.self_attn.o_proj,
                    unit=q_unit,
                    weights=kv_bal_w,
                )
                local_q_heads = layer.self_attn.q_proj.weight.shape[0] // head_dim
                q_offset = sum(kv_bal[: self.group.rank()])
                start_head = q_offset // head_dim
                kv_start = start_head // gqa_repeat
                kv_end = (start_head + local_q_heads - 1) // gqa_repeat + 1
                _slice_kv_proj(layer.self_attn.k_proj, kv_start, kv_end, head_dim)
                _slice_kv_proj(layer.self_attn.v_proj, kv_start, kv_end, head_dim)
                layer.self_attn.num_key_value_heads = kv_end - kv_start
            layer.self_attn.num_attention_heads = (
                layer.self_attn.q_proj.weight.shape[0] // head_dim
            )

            layer.self_attn = WrappedMiniMaxAttention(layer.self_attn, self.group)  # pyright: ignore[reportAttributeAccessIssue,reportArgumentType]

            # Shard the MoE.
            moe_gate_dim = int(
                layer.block_sparse_moe.switch_mlp.gate_proj.weight.shape[1]
            )
            moe_greedy = self._greedy_weights_for("moe_gate", moe_gate_dim)
            self.all_to_sharded_linear_in_place(
                layer.block_sparse_moe.switch_mlp.gate_proj,
                weights=moe_greedy,
            )
            self.sharded_to_all_linear_in_place(
                layer.block_sparse_moe.switch_mlp.down_proj,
                weights=moe_greedy,
            )
            self.all_to_sharded_linear_in_place(
                layer.block_sparse_moe.switch_mlp.up_proj,
                weights=moe_greedy,
            )
            layer.block_sparse_moe = ShardedMoE(layer.block_sparse_moe)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
            layer.block_sparse_moe.sharding_group = self.group  # pyright: ignore[reportAttributeAccessIssue]
            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(i, total)
        return model


class QwenShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        timeout_seconds: float,
        on_timeout: TimeoutCallback | None,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(
            Qwen3MoeModel | Qwen3NextModel | Qwen3_5TextModel | Qwen3_5MoeModel, model
        )
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            eval_with_timeout(layer.parameters(), timeout_seconds / total, on_timeout)
            # Shard the self attention
            if isinstance(layer, Qwen3MoeDecoderLayer):
                head_dim = (
                    layer.self_attn.q_proj.weight.shape[0] // layer.self_attn.n_heads
                )
                n_kv = layer.self_attn.n_kv_heads
                q_dim = layer.self_attn.q_proj.weight.shape[0]
                k_dim = layer.self_attn.k_proj.weight.shape[0]
                if n_kv >= self.N:
                    gqa_unit = head_dim * (layer.self_attn.n_heads // n_kv)
                    layer.self_attn.q_proj = self.all_to_sharded_linear(
                        layer.self_attn.q_proj,
                        unit=gqa_unit,
                        weights=self._greedy_weights_for("q", q_dim, gqa_unit),
                    )
                    layer.self_attn.k_proj = self.all_to_sharded_linear(
                        layer.self_attn.k_proj,
                        unit=head_dim,
                        weights=self._greedy_weights_for("k", k_dim, head_dim),
                    )
                    layer.self_attn.v_proj = self.all_to_sharded_linear(
                        layer.self_attn.v_proj,
                        unit=head_dim,
                        weights=self._greedy_weights_for("v", k_dim, head_dim),
                    )
                    layer.self_attn.o_proj = self.sharded_to_all_linear(
                        layer.self_attn.o_proj,
                        unit=gqa_unit,
                        weights=self._greedy_weights_for("o", q_dim, gqa_unit),
                    )
                    layer.self_attn.n_kv_heads = (
                        layer.self_attn.k_proj.weight.shape[0] // head_dim
                    )
                else:
                    q_unit = head_dim * n_kv
                    gqa_repeat = layer.self_attn.n_heads // n_kv
                    kv_bal = _kv_balanced_q_sizes(q_dim, self.N, gqa_repeat, head_dim, n_kv)
                    kv_bal_w = [s // q_unit for s in kv_bal]
                    layer.self_attn.q_proj = self.all_to_sharded_linear(
                        layer.self_attn.q_proj,
                        unit=q_unit,
                        weights=kv_bal_w,
                    )
                    layer.self_attn.o_proj = self.sharded_to_all_linear(
                        layer.self_attn.o_proj,
                        unit=q_unit,
                        weights=kv_bal_w,
                    )
                    local_q_heads = layer.self_attn.q_proj.weight.shape[0] // head_dim
                    q_offset = sum(kv_bal[:self.group.rank()])
                    start_head = q_offset // head_dim
                    kv_s = start_head // gqa_repeat
                    kv_e = (start_head + local_q_heads - 1) // gqa_repeat + 1
                    _slice_kv_proj(layer.self_attn.k_proj, kv_s, kv_e, head_dim)
                    _slice_kv_proj(layer.self_attn.v_proj, kv_s, kv_e, head_dim)
                    layer.self_attn.n_kv_heads = kv_e - kv_s
                layer.self_attn.n_heads = (
                    layer.self_attn.q_proj.weight.shape[0] // head_dim
                )
            else:
                assert isinstance(layer, (Qwen3NextDecoderLayer, Qwen3_5DecoderLayer))
                if hasattr(layer, "linear_attn"):
                    linear_attn = layer.linear_attn

                    k_greedy: list[float] | None = None
                    v_greedy: list[float] | None = None
                    if isinstance(linear_attn, Qwen3NextGatedDeltaNet):
                        qkvz_dim = linear_attn.in_proj_qkvz.weight.shape[0]
                        ba_dim = linear_attn.in_proj_ba.weight.shape[0]
                        linear_attn.in_proj_qkvz = self.all_to_sharded_linear(
                            linear_attn.in_proj_qkvz,
                            weights=self._greedy_weights_for("linear_qkvz", qkvz_dim),
                        )
                        linear_attn.in_proj_ba = self.all_to_sharded_linear(
                            linear_attn.in_proj_ba,
                            weights=self._greedy_weights_for("linear_ba", ba_dim),
                        )
                    else:
                        head_k_dim = linear_attn.head_k_dim
                        head_v_dim = linear_attn.head_v_dim
                        key_dim = linear_attn.key_dim
                        value_dim = linear_attn.value_dim
                        b_dim = linear_attn.in_proj_b.weight.shape[0]
                        a_dim = linear_attn.in_proj_a.weight.shape[0]
                        v_per_k = linear_attn.num_v_heads // linear_attn.num_k_heads
                        k_sizes = compute_shard_sizes(key_dim, self.N, head_k_dim)
                        v_sizes = [(s // head_k_dim) * v_per_k * head_v_dim for s in k_sizes]
                        qkv_sizes = [k_sizes[r] * 2 + v_sizes[r] for r in range(self.N)]
                        qkv_w = [float(s) for s in qkv_sizes]
                        v_w = [float(s) for s in v_sizes]
                        k_greedy = self._greedy_weights_for(
                            "linear_k_dim", key_dim, head_k_dim
                        )
                        linear_attn.in_proj_qkv = shard_linear(
                            linear_attn.in_proj_qkv,
                            "all-to-sharded",
                            segments=[key_dim, key_dim + key_dim],
                            unit=head_k_dim,
                            group=self.group,
                            weights=qkv_w,
                        )
                        linear_attn.in_proj_z = self.all_to_sharded_linear(
                            linear_attn.in_proj_z,
                            unit=head_v_dim,
                            weights=v_w,
                        )
                        linear_attn.in_proj_b = self.all_to_sharded_linear(
                            linear_attn.in_proj_b,
                            weights=v_w,
                        )
                        linear_attn.in_proj_a = self.all_to_sharded_linear(
                            linear_attn.in_proj_a,
                            weights=v_w,
                        )
                    is_qwen3next = isinstance(linear_attn, Qwen3NextGatedDeltaNet)
                    out_dim = linear_attn.out_proj.weight.shape[-1]
                    if not is_qwen3next:
                        linear_attn.out_proj = self.sharded_to_all_linear(
                            linear_attn.out_proj,
                            unit=linear_attn.head_v_dim,
                            weights=v_w,
                        )
                    else:
                        linear_attn.out_proj = self.sharded_to_all_linear(
                            linear_attn.out_proj,
                            unit=linear_attn.head_k_dim,
                            weights=self._greedy_weights_for(
                                "linear_out", out_dim, linear_attn.head_k_dim
                            ),
                        )

                    rank = self.group.rank()
                    key_dim = linear_attn.key_dim
                    value_dim = linear_attn.value_dim
                    head_k_dim = linear_attn.head_k_dim
                    head_v_dim = linear_attn.head_v_dim
                    if not is_qwen3next:
                        key_shard_sizes = k_sizes
                        value_shard_sizes = v_sizes
                    else:
                        k_w = self.shard_weights
                        key_shard_sizes = compute_shard_sizes(
                            key_dim, self.N, unit=head_k_dim, weights=k_w
                        )
                        value_shard_sizes = compute_shard_sizes(
                            value_dim, self.N, unit=head_k_dim, weights=k_w
                        )
                    key_dim_shard = key_shard_sizes[rank]
                    value_dim_shard = value_shard_sizes[rank]
                    key_dim_offset = sum(key_shard_sizes[:rank])
                    value_dim_offset = sum(value_shard_sizes[:rank])

                    q_idx = mx.arange(key_dim_offset, key_dim_offset + key_dim_shard)
                    k_idx = mx.arange(
                        key_dim + key_dim_offset,
                        key_dim + key_dim_offset + key_dim_shard,
                    )
                    v_idx = mx.arange(
                        2 * key_dim + value_dim_offset,
                        2 * key_dim + value_dim_offset + value_dim_shard,
                    )
                    conv_indices = mx.concatenate([q_idx, k_idx, v_idx])
                    linear_attn.conv1d.weight = linear_attn.conv1d.weight[conv_indices]
                    linear_attn.conv1d.groups = len(conv_indices)

                    num_k_per_rank = key_dim_shard // head_k_dim
                    num_v_per_rank = value_dim_shard // head_v_dim
                    v_offset = value_dim_offset // head_v_dim
                    linear_attn.A_log = linear_attn.A_log[
                        v_offset : v_offset + num_v_per_rank
                    ]
                    linear_attn.dt_bias = linear_attn.dt_bias[
                        v_offset : v_offset + num_v_per_rank
                    ]

                    linear_attn.num_k_heads = num_k_per_rank
                    linear_attn.num_v_heads = num_v_per_rank
                    linear_attn.key_dim = (
                        linear_attn.head_k_dim * linear_attn.num_k_heads
                    )
                    linear_attn.value_dim = (
                        linear_attn.head_v_dim * linear_attn.num_v_heads
                    )
                    linear_attn.conv_dim = int(
                        linear_attn.in_proj_qkv.weight.shape[0]
                    )
                else:
                    n_kv = layer.self_attn.num_key_value_heads
                    kv_head_dim = (
                        layer.self_attn.k_proj.weight.shape[0] // n_kv
                    )
                    gqa_repeat = (
                        layer.self_attn.num_attention_heads // n_kv
                    )
                    q_dim = layer.self_attn.q_proj.weight.shape[0]
                    k_dim = layer.self_attn.k_proj.weight.shape[0]
                    if n_kv >= self.N:
                        q_unit = kv_head_dim * 2 * gqa_repeat
                        qo_greedy = self._greedy_weights_for(
                            "qwen_qo", q_dim, q_unit
                        )
                        layer.self_attn.q_proj = self.all_to_sharded_linear(
                            layer.self_attn.q_proj,
                            unit=q_unit,
                            weights=qo_greedy,
                        )
                        layer.self_attn.k_proj = self.all_to_sharded_linear(
                            layer.self_attn.k_proj,
                            unit=kv_head_dim,
                            weights=self._greedy_weights_for("k", k_dim, kv_head_dim),
                        )
                        layer.self_attn.v_proj = self.all_to_sharded_linear(
                            layer.self_attn.v_proj,
                            unit=kv_head_dim,
                            weights=self._greedy_weights_for("v", k_dim, kv_head_dim),
                        )
                        layer.self_attn.o_proj = self.sharded_to_all_linear(
                            layer.self_attn.o_proj,
                            unit=kv_head_dim * gqa_repeat,
                            weights=qo_greedy,
                        )
                        layer.self_attn.num_key_value_heads = (
                            layer.self_attn.k_proj.weight.shape[0] // kv_head_dim
                        )
                    else:
                        q_unit = kv_head_dim * 2
                        gqa_repeat_q = layer.self_attn.num_attention_heads // n_kv
                        kv_bal = _kv_balanced_q_sizes(q_dim, self.N, gqa_repeat_q, kv_head_dim * 2, n_kv)
                        kv_bal_w = [float(s) for s in kv_bal]
                        layer.self_attn.q_proj = self.all_to_sharded_linear(
                            layer.self_attn.q_proj,
                            unit=q_unit,
                            weights=kv_bal_w,
                        )
                        o_bal = [s // 2 for s in kv_bal]
                        o_bal_w = [float(s) for s in o_bal]
                        layer.self_attn.o_proj = self.sharded_to_all_linear(
                            layer.self_attn.o_proj,
                            unit=kv_head_dim,
                            weights=o_bal_w,
                        )
                        local_q_heads = layer.self_attn.q_proj.weight.shape[0] // (kv_head_dim * 2)
                        q_offset = sum(kv_bal[:self.group.rank()])
                        start_head = q_offset // (kv_head_dim * 2)
                        kv_s = start_head // gqa_repeat_q
                        kv_e = (start_head + local_q_heads - 1) // gqa_repeat_q + 1
                        _slice_kv_proj(layer.self_attn.k_proj, kv_s, kv_e, kv_head_dim)
                        _slice_kv_proj(layer.self_attn.v_proj, kv_s, kv_e, kv_head_dim)
                        layer.self_attn.num_key_value_heads = kv_e - kv_s
                    layer.self_attn.num_attention_heads = (
                        layer.self_attn.q_proj.weight.shape[0] // (kv_head_dim * 2)
                    )

            # Shard the MoE.
            if isinstance(
                layer.mlp,
                (
                    Qwen3MoeSparseMoeBlock,
                    Qwen3NextSparseMoeBlock,
                    Qwen3_5SparseMoeBlock,
                ),
            ):
                moe_gate_dim = int(layer.mlp.switch_mlp.gate_proj.weight.shape[1])
                moe_greedy = self._greedy_weights_for("moe_gate", moe_gate_dim)
                self.all_to_sharded_linear_in_place(
                    layer.mlp.switch_mlp.gate_proj,
                    weights=moe_greedy,
                )
                self.sharded_to_all_linear_in_place(
                    layer.mlp.switch_mlp.down_proj,
                    weights=moe_greedy,
                )
                self.all_to_sharded_linear_in_place(
                    layer.mlp.switch_mlp.up_proj,
                    weights=moe_greedy,
                )
                if isinstance(
                    layer.mlp, (Qwen3NextSparseMoeBlock, Qwen3_5SparseMoeBlock)
                ):
                    shared_gate_dim = layer.mlp.shared_expert.gate_proj.weight.shape[0]
                    shared_down_dim = layer.mlp.shared_expert.down_proj.weight.shape[-1]
                    shared_up_dim = layer.mlp.shared_expert.up_proj.weight.shape[0]
                    shared_greedy = self._greedy_weights_for("shared_gate", shared_gate_dim)
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_expert.gate_proj,
                        weights=shared_greedy,
                    )
                    self.sharded_to_all_linear_in_place(
                        layer.mlp.shared_expert.down_proj,
                        weights=shared_greedy,
                    )
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_expert.up_proj,
                        weights=shared_greedy,
                    )
                layer.mlp = ShardedMoE(layer.mlp)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
                layer.mlp.sharding_group = self.group

            # Shard the MLP
            else:
                intermediate = layer.mlp.gate_proj.weight.shape[0]
                mlp_unit = getattr(layer.mlp.gate_proj, "group_size", 1)
                layer.mlp.gate_proj = self.all_to_sharded_linear(
                    layer.mlp.gate_proj,
                    unit=mlp_unit,
                    weights=self._greedy_weights_for("gate", intermediate),
                )
                layer.mlp.down_proj = self.sharded_to_all_linear(
                    layer.mlp.down_proj,
                    unit=mlp_unit,
                    weights=self._greedy_weights_for("down", intermediate),
                )
                layer.mlp.up_proj = self.all_to_sharded_linear(
                    layer.mlp.up_proj,
                    unit=mlp_unit,
                    weights=self._greedy_weights_for("up", intermediate),
                )

            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(i, total)
        return model


class Glm4MoeShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        timeout_seconds: float,
        on_timeout: TimeoutCallback | None,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(Glm4MoeModel, model)
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            eval_with_timeout(layer.parameters(), timeout_seconds / total, on_timeout)

            head_dim = layer.self_attn.q_proj.weight.shape[0] // layer.self_attn.n_heads
            n_kv = layer.self_attn.n_kv_heads
            q_dim = layer.self_attn.q_proj.weight.shape[0]
            k_dim = layer.self_attn.k_proj.weight.shape[0]
            if n_kv >= self.N:
                gqa_unit = head_dim * (layer.self_attn.n_heads // n_kv)
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj,
                    unit=gqa_unit,
                    weights=self._greedy_weights_for("q", q_dim, gqa_unit),
                )
                layer.self_attn.k_proj = self.all_to_sharded_linear(
                    layer.self_attn.k_proj,
                    unit=head_dim,
                    weights=self._greedy_weights_for("k", k_dim, head_dim),
                )
                layer.self_attn.v_proj = self.all_to_sharded_linear(
                    layer.self_attn.v_proj,
                    unit=head_dim,
                    weights=self._greedy_weights_for("v", k_dim, head_dim),
                )
                layer.self_attn.o_proj = self.sharded_to_all_linear(
                    layer.self_attn.o_proj,
                    unit=gqa_unit,
                    weights=self._greedy_weights_for("o", q_dim, gqa_unit),
                )
                layer.self_attn.n_kv_heads = (
                    layer.self_attn.k_proj.weight.shape[0] // head_dim
                )
            else:
                q_unit = head_dim
                gqa_repeat = layer.self_attn.n_heads // n_kv
                kv_bal = _kv_balanced_q_sizes(q_dim, self.N, gqa_repeat, head_dim, n_kv)
                kv_bal_w = [float(s) for s in kv_bal]
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj,
                    unit=q_unit,
                    weights=kv_bal_w,
                )
                layer.self_attn.o_proj = self.sharded_to_all_linear(
                    layer.self_attn.o_proj,
                    unit=q_unit,
                    weights=kv_bal_w,
                )
                local_q_heads = layer.self_attn.q_proj.weight.shape[0] // head_dim
                q_offset = sum(kv_bal[:self.group.rank()])
                start_head = q_offset // head_dim
                kv_s = start_head // gqa_repeat
                kv_e = (start_head + local_q_heads - 1) // gqa_repeat + 1
                _slice_kv_proj(layer.self_attn.k_proj, kv_s, kv_e, head_dim)
                _slice_kv_proj(layer.self_attn.v_proj, kv_s, kv_e, head_dim)
                layer.self_attn.n_kv_heads = kv_e - kv_s
            layer.self_attn.n_heads = layer.self_attn.q_proj.weight.shape[0] // head_dim

            if isinstance(layer.mlp, MoE):
                moe_gate_dim = int(layer.mlp.switch_mlp.gate_proj.weight.shape[1])
                moe_greedy = self._greedy_weights_for("moe_gate", moe_gate_dim)
                self.all_to_sharded_linear_in_place(
                    layer.mlp.switch_mlp.gate_proj,
                    weights=moe_greedy,
                )
                self.sharded_to_all_linear_in_place(
                    layer.mlp.switch_mlp.down_proj,
                    weights=moe_greedy,
                )
                self.all_to_sharded_linear_in_place(
                    layer.mlp.switch_mlp.up_proj,
                    weights=moe_greedy,
                )
                if getattr(layer.mlp, "shared_experts", None) is not None:
                    shared_gate_dim = layer.mlp.shared_experts.gate_proj.weight.shape[0]
                    shared_down_dim = layer.mlp.shared_experts.down_proj.weight.shape[
                        -1
                    ]
                    shared_up_dim = layer.mlp.shared_experts.up_proj.weight.shape[0]
                    shared_greedy = self._greedy_weights_for("shared_gate", shared_gate_dim)
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.gate_proj,
                        weights=shared_greedy,
                    )
                    self.sharded_to_all_linear_in_place(
                        layer.mlp.shared_experts.down_proj,
                        weights=shared_greedy,
                    )
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.up_proj,
                        weights=shared_greedy,
                    )
                layer.mlp = ShardedMoE(layer.mlp)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
                layer.mlp.sharding_group = self.group

            else:
                intermediate = layer.mlp.gate_proj.weight.shape[0]
                mlp_unit = getattr(layer.mlp.gate_proj, "group_size", 1)
                layer.mlp.gate_proj = self.all_to_sharded_linear(
                    layer.mlp.gate_proj,
                    unit=mlp_unit,
                    weights=self._greedy_weights_for("gate", intermediate),
                )
                layer.mlp.down_proj = self.sharded_to_all_linear(
                    layer.mlp.down_proj,
                    unit=mlp_unit,
                    weights=self._greedy_weights_for("down", intermediate),
                )
                layer.mlp.up_proj = self.all_to_sharded_linear(
                    layer.mlp.up_proj,
                    unit=mlp_unit,
                    weights=self._greedy_weights_for("up", intermediate),
                )

            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(i, total)
        return model


class GptOssShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        timeout_seconds: float,
        on_timeout: TimeoutCallback | None,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(GptOssMoeModel, model)
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            eval_with_timeout(layer.parameters(), timeout_seconds / total, on_timeout)
            head_dim = layer.self_attn.head_dim
            original_num_heads = layer.self_attn.num_attention_heads
            n_kv = layer.self_attn.num_key_value_heads
            q_dim = layer.self_attn.q_proj.weight.shape[0]
            k_dim = layer.self_attn.k_proj.weight.shape[0]
            if n_kv >= self.N:
                gqa_unit = head_dim * (original_num_heads // n_kv)
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj,
                    unit=gqa_unit,
                    weights=self._greedy_weights_for("q", q_dim, gqa_unit),
                )
                layer.self_attn.k_proj = self.all_to_sharded_linear(
                    layer.self_attn.k_proj,
                    unit=head_dim,
                    weights=self._greedy_weights_for("k", k_dim, head_dim),
                )
                layer.self_attn.v_proj = self.all_to_sharded_linear(
                    layer.self_attn.v_proj,
                    unit=head_dim,
                    weights=self._greedy_weights_for("v", k_dim, head_dim),
                )
                layer.self_attn.o_proj = self.sharded_to_all_linear(
                    layer.self_attn.o_proj,
                    unit=gqa_unit,
                    weights=self._greedy_weights_for("o", q_dim, gqa_unit),
                )
                layer.self_attn.num_key_value_heads = (
                    layer.self_attn.k_proj.weight.shape[0] // head_dim
                )
                q_unit_for_sinks = gqa_unit
            else:
                q_unit = head_dim
                gqa_repeat = original_num_heads // n_kv
                kv_bal = _kv_balanced_q_sizes(q_dim, self.N, gqa_repeat, head_dim, n_kv)
                kv_bal_w = [float(s) for s in kv_bal]
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj,
                    unit=q_unit,
                    weights=kv_bal_w,
                )
                layer.self_attn.o_proj = self.sharded_to_all_linear(
                    layer.self_attn.o_proj,
                    unit=q_unit,
                    weights=kv_bal_w,
                )
                local_q_heads = layer.self_attn.q_proj.weight.shape[0] // head_dim
                q_offset = sum(kv_bal[:self.group.rank()])
                start_head = q_offset // head_dim
                kv_s = start_head // gqa_repeat
                kv_e = (start_head + local_q_heads - 1) // gqa_repeat + 1
                _slice_kv_proj(layer.self_attn.k_proj, kv_s, kv_e, head_dim)
                _slice_kv_proj(layer.self_attn.v_proj, kv_s, kv_e, head_dim)
                layer.self_attn.num_key_value_heads = kv_e - kv_s
                q_unit_for_sinks = q_unit
            layer.self_attn.num_attention_heads = (
                layer.self_attn.q_proj.weight.shape[0] // head_dim
            )
            layer.self_attn.num_key_value_groups = (
                layer.self_attn.num_attention_heads
                // layer.self_attn.num_key_value_heads
            )

            rank = self.group.rank()
            q_actual = self._greedy_last_sizes("q")
            q_head_sizes = (
                [s // head_dim for s in q_actual]
                if q_actual
                else compute_shard_sizes(
                    original_num_heads,
                    self.N,
                    unit=q_unit_for_sinks // head_dim,
                    weights=self.shard_weights,
                )
            )
            sink_start = sum(q_head_sizes[:rank])
            sink_end = sink_start + q_head_sizes[rank]
            layer.self_attn.sinks = layer.self_attn.sinks[sink_start:sink_end]

            moe_gate_dim = int(layer.mlp.experts.gate_proj.weight.shape[1])
            moe_down_dim = int(layer.mlp.experts.down_proj.weight.shape[-1])
            moe_up_dim = int(layer.mlp.experts.up_proj.weight.shape[1])
            moe_greedy = self._greedy_weights_for("moe_gate", moe_gate_dim)
            self.all_to_sharded_linear_in_place(
                layer.mlp.experts.gate_proj,
                weights=moe_greedy,
            )
            self.sharded_to_all_linear_in_place(
                layer.mlp.experts.down_proj,
                weights=moe_greedy,
            )
            self.all_to_sharded_linear_in_place(
                layer.mlp.experts.up_proj,
                weights=moe_greedy,
            )

            layer.mlp = ShardedMoE(layer.mlp)  # type: ignore
            layer.mlp.sharding_group = self.group  # pyright: ignore[reportAttributeAccessIssue]
            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(i, total)
        return model


class Step35ShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        timeout_seconds: float,
        on_timeout: TimeoutCallback | None,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(Step35Model, model)
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            eval_with_timeout(layer.parameters(), timeout_seconds / total, on_timeout)
            head_dim = layer.self_attn.head_dim
            n_kv = layer.self_attn.num_kv_heads
            q_dim = layer.self_attn.q_proj.weight.shape[0]
            k_dim = layer.self_attn.k_proj.weight.shape[0]
            if n_kv >= self.N:
                gqa_unit = head_dim * (layer.self_attn.num_heads // n_kv)
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj,
                    unit=gqa_unit,
                    weights=self._greedy_weights_for("q", q_dim, gqa_unit),
                )
                layer.self_attn.k_proj = self.all_to_sharded_linear(
                    layer.self_attn.k_proj,
                    unit=head_dim,
                    weights=self._greedy_weights_for("k", k_dim, head_dim),
                )
                layer.self_attn.v_proj = self.all_to_sharded_linear(
                    layer.self_attn.v_proj,
                    unit=head_dim,
                    weights=self._greedy_weights_for("v", k_dim, head_dim),
                )
                layer.self_attn.o_proj = self.sharded_to_all_linear(
                    layer.self_attn.o_proj,
                    unit=gqa_unit,
                    weights=self._greedy_weights_for("o", q_dim, gqa_unit),
                )
                layer.self_attn.num_kv_heads = (
                    layer.self_attn.k_proj.weight.shape[0] // head_dim
                )
                q_unit_for_g = gqa_unit
            else:
                q_unit = head_dim
                gqa_repeat = layer.self_attn.num_heads // n_kv
                kv_bal = _kv_balanced_q_sizes(q_dim, self.N, gqa_repeat, head_dim, n_kv)
                kv_bal_w = [float(s) for s in kv_bal]
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj,
                    unit=q_unit,
                    weights=kv_bal_w,
                )
                layer.self_attn.o_proj = self.sharded_to_all_linear(
                    layer.self_attn.o_proj,
                    unit=q_unit,
                    weights=kv_bal_w,
                )
                local_q_heads = layer.self_attn.q_proj.weight.shape[0] // head_dim
                q_offset = sum(kv_bal[:self.group.rank()])
                start_head = q_offset // head_dim
                kv_s = start_head // gqa_repeat
                kv_e = (start_head + local_q_heads - 1) // gqa_repeat + 1
                _slice_kv_proj(layer.self_attn.k_proj, kv_s, kv_e, head_dim)
                _slice_kv_proj(layer.self_attn.v_proj, kv_s, kv_e, head_dim)
                layer.self_attn.num_kv_heads = kv_e - kv_s
                q_unit_for_g = q_unit
            layer.self_attn.num_heads = (
                layer.self_attn.q_proj.weight.shape[0] // head_dim
            )

            if getattr(layer.self_attn, "use_head_wise_attn_gate", False):
                g_dim = layer.self_attn.g_proj.weight.shape[0]
                g_unit = q_unit_for_g // head_dim
                layer.self_attn.g_proj = self.all_to_sharded_linear(
                    layer.self_attn.g_proj,
                    unit=g_unit,
                    weights=self._greedy_weights_for("g", g_dim, g_unit),
                )

            if isinstance(layer.mlp, Step35MLP):
                intermediate = layer.mlp.gate_proj.weight.shape[0]
                mlp_unit = getattr(layer.mlp.gate_proj, "group_size", 1)
                layer.mlp.gate_proj = self.all_to_sharded_linear(
                    layer.mlp.gate_proj,
                    unit=mlp_unit,
                    weights=self._greedy_weights_for("gate", intermediate),
                )
                layer.mlp.up_proj = self.all_to_sharded_linear(
                    layer.mlp.up_proj,
                    unit=mlp_unit,
                    weights=self._greedy_weights_for("up", intermediate),
                )
                layer.mlp.down_proj = self.sharded_to_all_linear(
                    layer.mlp.down_proj,
                    unit=mlp_unit,
                    weights=self._greedy_weights_for("down", intermediate),
                )
            else:
                layer.mlp.sharding_group = self.group
                shared_gate_dim = layer.mlp.share_expert.gate_proj.weight.shape[0]
                shared_up_dim = layer.mlp.share_expert.up_proj.weight.shape[0]
                shared_down_dim = layer.mlp.share_expert.down_proj.weight.shape[-1]
                shared_greedy = self._greedy_weights_for("shared_gate", shared_gate_dim)
                self.all_to_sharded_linear_in_place(
                    layer.mlp.share_expert.gate_proj,
                    weights=shared_greedy,
                )
                self.all_to_sharded_linear_in_place(
                    layer.mlp.share_expert.up_proj,
                    weights=shared_greedy,
                )
                self.sharded_to_all_linear_in_place(
                    layer.mlp.share_expert.down_proj,
                    weights=shared_greedy,
                )
                moe_gate_dim = int(layer.mlp.switch_mlp.gate_proj.weight.shape[1])
                moe_greedy = self._greedy_weights_for("moe_gate", moe_gate_dim)
                self.all_to_sharded_linear_in_place(
                    layer.mlp.switch_mlp.gate_proj,
                    weights=moe_greedy,
                )
                self.all_to_sharded_linear_in_place(
                    layer.mlp.switch_mlp.up_proj,
                    weights=moe_greedy,
                )
                self.sharded_to_all_linear_in_place(
                    layer.mlp.switch_mlp.down_proj,
                    weights=moe_greedy,
                )

            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(i, total)
        return model


class NemotronHShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        timeout_seconds: float,
        on_timeout: TimeoutCallback | None,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(NemotronHModel, model)
        rank = self.group.rank()
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            eval_with_timeout(layer.parameters(), timeout_seconds / total, on_timeout)

            mixer = layer.mixer

            if isinstance(mixer, NemotronHAttention):
                attn_head_dim = mixer.head_dim
                n_kv = mixer.num_key_value_heads
                q_dim = mixer.q_proj.weight.shape[0]
                k_dim = mixer.k_proj.weight.shape[0]
                if n_kv >= self.N:
                    gqa_unit = attn_head_dim * (mixer.num_heads // n_kv)
                    mixer.q_proj = self.all_to_sharded_linear(
                        mixer.q_proj,
                        unit=gqa_unit,
                        weights=self._greedy_weights_for("q", q_dim, gqa_unit),
                    )
                    mixer.k_proj = self.all_to_sharded_linear(
                        mixer.k_proj,
                        unit=attn_head_dim,
                        weights=self._greedy_weights_for("k", k_dim, attn_head_dim),
                    )
                    mixer.v_proj = self.all_to_sharded_linear(
                        mixer.v_proj,
                        unit=attn_head_dim,
                        weights=self._greedy_weights_for("v", k_dim, attn_head_dim),
                    )
                    mixer.o_proj = self.sharded_to_all_linear(
                        mixer.o_proj,
                        unit=gqa_unit,
                        weights=self._greedy_weights_for("o", q_dim, gqa_unit),
                    )
                    mixer.num_key_value_heads = (
                        mixer.k_proj.weight.shape[0] // attn_head_dim
                    )
                else:
                    q_unit = attn_head_dim
                    gqa_repeat = mixer.num_heads // n_kv
                    kv_bal = _kv_balanced_q_sizes(q_dim, self.N, gqa_repeat, attn_head_dim, n_kv)
                    kv_bal_w = [float(s) for s in kv_bal]
                    mixer.q_proj = self.all_to_sharded_linear(
                        mixer.q_proj,
                        unit=q_unit,
                        weights=kv_bal_w,
                    )
                    mixer.o_proj = self.sharded_to_all_linear(
                        mixer.o_proj,
                        unit=q_unit,
                        weights=kv_bal_w,
                    )
                    local_q_heads = mixer.q_proj.weight.shape[0] // attn_head_dim
                    q_offset = sum(kv_bal[:self.group.rank()])
                    start_head = q_offset // attn_head_dim
                    kv_s = start_head // gqa_repeat
                    kv_e = (start_head + local_q_heads - 1) // gqa_repeat + 1
                    _slice_kv_proj(mixer.k_proj, kv_s, kv_e, attn_head_dim)
                    _slice_kv_proj(mixer.v_proj, kv_s, kv_e, attn_head_dim)
                    mixer.num_key_value_heads = kv_e - kv_s
                mixer.num_heads = mixer.q_proj.weight.shape[0] // attn_head_dim

            elif isinstance(mixer, NemotronHMamba2Mixer):
                self._shard_mamba2_mixer(mixer, rank)

            elif isinstance(mixer, NemotronHMoE):
                # Shard routed experts (SwitchMLP uses fc1/fc2)
                moe_fc1_dim = int(mixer.switch_mlp.fc1.weight.shape[1])
                moe_greedy = self._greedy_weights_for("moe_gate", moe_fc1_dim)
                self.all_to_sharded_linear_in_place(
                    mixer.switch_mlp.fc1,
                    weights=moe_greedy,
                )
                self.sharded_to_all_linear_in_place(
                    mixer.switch_mlp.fc2,
                    weights=moe_greedy,
                )
                if hasattr(mixer, "shared_experts"):
                    shared_gate_dim = mixer.shared_experts.up_proj.weight.shape[0]
                    shared_greedy = self._greedy_weights_for("shared_gate", shared_gate_dim)
                    self.all_to_sharded_linear_in_place(
                        mixer.shared_experts.up_proj,
                        weights=shared_greedy,
                    )
                    self.sharded_to_all_linear_in_place(
                        mixer.shared_experts.down_proj,
                        weights=shared_greedy,
                    )
                mixer = ShardedMoE(mixer)  # pyright: ignore[reportArgumentType]
                mixer.sharding_group = self.group
                layer.mixer = mixer  # pyright: ignore[reportAttributeAccessIssue]

            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(i, total)
        return model

    def _shard_mamba2_mixer(self, mixer: NemotronHMamba2Mixer, rank: int) -> None:
        """Shard the Mamba2 mixer along the head dimension."""
        world_size = self.N
        num_heads = mixer.num_heads
        head_dim = mixer.head_dim
        n_groups = mixer.n_groups
        ssm_state_size = mixer.ssm_state_size
        intermediate_size = mixer.intermediate_size  # = num_heads * head_dim

        # === out_proj first — determines the actual per-rank intermediate sizes ===
        heads_per_group = num_heads // n_groups
        out_unit = heads_per_group * head_dim
        mixer.out_proj = self.sharded_to_all_linear(
            mixer.out_proj,
            unit=out_unit,
            weights=self._greedy_weights_for("mamba_out", intermediate_size, out_unit),
        )
        out_actual = self._greedy_last_sizes("mamba_out")
        if out_actual:
            head_sizes = [s // head_dim for s in out_actual]
            group_sizes = [s // heads_per_group for s in head_sizes]
        else:
            group_sizes = compute_shard_sizes(
                n_groups, world_size, weights=self.shard_weights
            )
            head_sizes = [g * heads_per_group for g in group_sizes]

        groups_per_rank = group_sizes[rank]
        heads_per_rank = head_sizes[rank]
        is_per_rank = heads_per_rank * head_dim
        bc_per_rank = groups_per_rank * ssm_state_size

        # Cumulative offsets
        is_offset = sum(head_sizes[:rank]) * head_dim
        bc_offset = sum(group_sizes[:rank]) * ssm_state_size
        head_offset = sum(head_sizes[:rank])

        # === in_proj: output layout is [gate:IS | conv_ssm:IS | B:NG*SS | C:NG*SS | dt:NH] ===
        gate_start = 0
        conv_ssm_start = intermediate_size
        b_start = 2 * intermediate_size
        c_start = b_start + n_groups * ssm_state_size
        dt_start = c_start + n_groups * ssm_state_size

        # Build index tensor for this rank's slice of each section
        gate_idx = mx.arange(
            gate_start + is_offset, gate_start + is_offset + is_per_rank
        )
        conv_ssm_idx = mx.arange(
            conv_ssm_start + is_offset,
            conv_ssm_start + is_offset + is_per_rank,
        )
        b_idx = mx.arange(b_start + bc_offset, b_start + bc_offset + bc_per_rank)
        c_idx = mx.arange(c_start + bc_offset, c_start + bc_offset + bc_per_rank)
        dt_idx = mx.arange(
            dt_start + head_offset, dt_start + head_offset + heads_per_rank
        )

        indices = mx.concatenate([gate_idx, conv_ssm_idx, b_idx, c_idx, dt_idx])
        mixer.in_proj.weight = mixer.in_proj.weight[indices]

        # === conv1d: depthwise conv on conv_dim channels ===
        # conv_dim layout: [ssm_hidden:IS | B:NG*SS | C:NG*SS]
        conv_ssm_idx_local = mx.arange(is_offset, is_offset + is_per_rank)
        conv_b_idx = mx.arange(
            intermediate_size + bc_offset,
            intermediate_size + bc_offset + bc_per_rank,
        )
        conv_c_idx = mx.arange(
            intermediate_size + n_groups * ssm_state_size + bc_offset,
            intermediate_size + n_groups * ssm_state_size + bc_offset + bc_per_rank,
        )
        conv_indices = mx.concatenate([conv_ssm_idx_local, conv_b_idx, conv_c_idx])
        mixer.conv1d.weight = mixer.conv1d.weight[conv_indices]
        new_conv_dim = is_per_rank + 2 * bc_per_rank
        mixer.conv1d.groups = new_conv_dim
        if mixer.conv1d.bias is not None:
            mixer.conv1d.bias = mixer.conv1d.bias[conv_indices]

        # === Per-head parameters ===
        h_start = head_offset
        h_end = h_start + heads_per_rank
        mixer.dt_bias = mixer.dt_bias[h_start:h_end]
        mixer.A_log = mixer.A_log[h_start:h_end]
        mixer.D = mixer.D[h_start:h_end]

        # === Norm: weight is intermediate_size ===
        mixer.norm.weight = mixer.norm.weight[is_offset : is_offset + is_per_rank]
        mixer.norm.group_size = is_per_rank // groups_per_rank

        # === Update dimensions ===
        mixer.num_heads = heads_per_rank
        mixer.n_groups = groups_per_rank
        mixer.intermediate_size = is_per_rank
        mixer.conv_dim = new_conv_dim
        mixer.heads_per_group = heads_per_rank // groups_per_rank
