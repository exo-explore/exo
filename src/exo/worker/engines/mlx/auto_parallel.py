from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from inspect import signature
from typing import TYPE_CHECKING, Literal, Protocol, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import (
    ShardedToAllLinear,
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
from mlx_lm.models.gemma4 import Model as Gemma4Model
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
from mlx_lm.models.qwen3 import Model as Qwen3Model
from mlx_lm.models.qwen3 import TransformerBlock as Qwen3TransformerBlock
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
from mlx_lm.models.qwen3_vl import Model as Qwen3VLModel
from mlx_lm.models.step3p5 import Model as Step35Model
from mlx_lm.models.step3p5 import Step3p5MLP as Step35MLP
from mlx_lm.models.step3p5 import Step3p5Model as Step35InnerModel
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear

from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from mlx_lm.models.cache import Cache

def _fp32_reducing_sharded_to_all_call(
    self: ShardedToAllLinear, x: mx.array
) -> mx.array:
    weight = cast(mx.array, self["weight"])
    group = cast(mx.distributed.Group, self.group)
    y = mx.matmul(x, weight.T, output_dtype=mx.float32)  # pyright: ignore[reportCallIssue]
    y = mx.distributed.all_sum(y, group=group)
    if "bias" in self:
        y = y + cast(mx.array, self["bias"]).astype(mx.float32)
    return y.astype(x.dtype)


ShardedToAllLinear.__call__ = _fp32_reducing_sharded_to_all_call


from mlx.nn.layers.distributed import AllToShardedLinear  # noqa: E402


def _splitk_override_for_unsharded(M: int, N_full: int, K: int) -> int:
    """Return the override value that forces a per-rank matmul to match the
    unsharded kernel's K-reduction shape.

    The updated mlx heuristic returns 0 when splitk wouldn't dispatch for the
    unsharded shape; the caller must then set the override to -1 so the
    per-rank matmul also skips splitk. If the heuristic returns a positive n,
    set the override to n so the per-rank matmul uses the same partition
    count as the unsharded call.
    """
    n = mx.compute_splitk_partitions(M, N_full, K)
    return n if n > 0 else -1


def _splitk_override_all_to_sharded_call(
    self: AllToShardedLinear, x: mx.array
) -> mx.array:
    """All-to-sharded matmul that matches the unsharded kernel's K-reduction.

    mx.eval is forced inside the override block because the override is
    thread-local at kernel-dispatch time; without the eval the matmul is
    deferred past the clear and the dispatch falls back to the per-rank
    heuristic.
    """
    x = sum_gradients(self.group)(x)  # pyright: ignore[reportAttributeAccessIssue]
    weight = cast(mx.array, self["weight"])
    per_rank_N, K = weight.shape
    N_full = per_rank_N * self.group.size()  # pyright: ignore[reportAttributeAccessIssue]
    M = x.shape[-2] if x.ndim >= 2 else 1
    mx.set_splitk_partitions_override(_splitk_override_for_unsharded(M, N_full, K))
    try:
        if "bias" in self:
            y = mx.addmm(cast(mx.array, self["bias"]), x, weight.T)
        else:
            y = mx.matmul(x, weight.T)
    finally:
        mx.set_splitk_partitions_override(0)
    return y


AllToShardedLinear.__call__ = _splitk_override_all_to_sharded_call


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
        if cache is not None and len(cache) > 0:  # type: ignore
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


def tensor_auto_parallel(
    model: nn.Module,
    group: mx.distributed.Group,
    on_layer_loaded: LayerLoadedCallback | None,
) -> nn.Module:
    all_to_sharded_linear = partial(
        shard_linear,
        sharding="all-to-sharded",
        group=group,
    )
    sharded_to_all_linear = partial(
        shard_linear,
        sharding="sharded-to-all",
        group=group,
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
    )

    n = group.size()

    def _sharded_to_all(path: str, weight: mx.array):
        if path.endswith("bias"):
            logger.info(f"Sharding bias for {path} - sharded to all")
            weight /= n
            return None
        return -1, segments

    sharded_to_all_linear_in_place = partial(
        shard_inplace,
        sharding=_sharded_to_all,  # type: ignore
        group=group,
    )

    if isinstance(model, (LlamaModel, Ministral3Model)):
        tensor_parallel_sharding_strategy = LlamaShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, (DeepseekV3Model, DeepseekV32Model, KimiK25Model)):
        tensor_parallel_sharding_strategy = DeepSeekShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, MiniMaxModel):
        tensor_parallel_sharding_strategy = MiniMaxShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, GLM4MoeLiteModel):
        tensor_parallel_sharding_strategy = GLM4MoeLiteShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, Glm4MoeModel):
        tensor_parallel_sharding_strategy = Glm4MoeShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(
        model,
        (
            Qwen3Model,
            Qwen3MoeModel,
            Qwen3NextModel,
            Qwen3_5TextModel,
            Qwen3_5MoeModel,
            Qwen3VLModel,
        ),
    ):
        tensor_parallel_sharding_strategy = QwenShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, GptOssModel):
        tensor_parallel_sharding_strategy = GptOssShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, Step35Model):
        tensor_parallel_sharding_strategy = Step35ShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, NemotronHModel):
        tensor_parallel_sharding_strategy = NemotronHShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, Gemma4Model):
        tensor_parallel_sharding_strategy = Gemma4ShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    model = tensor_parallel_sharding_strategy.shard_model(model, on_layer_loaded)
    return patch_tensor_model(model)


class TensorParallelShardingStrategy(ABC):
    def __init__(
        self,
        group: mx.distributed.Group,
        all_to_sharded_linear: Callable[..., nn.Linear],
        sharded_to_all_linear: Callable[..., nn.Linear],
        all_to_sharded_linear_in_place: Callable[..., None],
        sharded_to_all_linear_in_place: Callable[..., None],
    ):
        self.all_to_sharded_linear = all_to_sharded_linear
        self.sharded_to_all_linear = sharded_to_all_linear
        self.all_to_sharded_linear_in_place = all_to_sharded_linear_in_place
        self.sharded_to_all_linear_in_place = sharded_to_all_linear_in_place
        self.group = group
        self.N = group.size()

    @abstractmethod
    def shard_model(
        self,
        model: nn.Module,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module: ...


class LlamaShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(LlamaModel, model)
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            # Force load weights before sharding to avoid FAST_SYNCH deadlock
            mx.eval(layer.parameters())
            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = NShardedLinear.from_linear(layer.self_attn.o_proj, self.group)
            layer.self_attn.n_heads //= self.N
            if layer.self_attn.n_kv_heads is not None:
                layer.self_attn.n_kv_heads //= self.N

            layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
            layer.mlp.down_proj = NShardedLinear.from_linear(layer.mlp.down_proj, self.group)
            layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)
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
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(DeepseekV3Model, model)
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            # Shard the self attention
            if layer.self_attn.q_lora_rank is None:
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj
                )
            else:
                layer.self_attn.q_b_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_b_proj
                )

            layer.self_attn.o_proj = NShardedLinear.from_linear(layer.self_attn.o_proj, self.group)
            layer.self_attn.num_heads //= self.N

            # Logic from upstream mlx
            num_heads = layer.self_attn.num_heads
            sh = self.group.rank() * num_heads
            eh = sh + num_heads

            def shard_heads(w: mx.array, sh: int = sh, eh: int = eh) -> mx.array:
                return w[sh:eh]

            layer.self_attn.embed_q.apply(shard_heads)
            layer.self_attn.unembed_out.apply(shard_heads)

            # Shard the MLP
            if isinstance(layer.mlp, (DeepseekV3MLP, DeepseekV32MLP)):
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = NShardedLinear.from_linear(layer.mlp.down_proj, self.group)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            # Shard the MoE with column-sharded down_proj for bit-exactness.
            else:
                if getattr(layer.mlp, "shared_experts", None) is not None:
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.gate_proj
                    )
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.up_proj
                    )
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.down_proj
                    )
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.down_proj)
                layer.mlp = ShardedMoE(layer.mlp)  # type: ignore
                layer.mlp.sharding_group = self.group

            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(i, total)

        return model


class NShardedLinear(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, bias: bool, group: mx.distributed.Group):
        super().__init__()
        N = group.size()
        self.group = group
        self.weight = mx.zeros((out_dims // N, in_dims))
        if bias:
            self.bias = mx.zeros((out_dims // N,))

    def __call__(self, x: mx.array) -> mx.array:
        x_full = _all_gather_last(x, self.group)
        weight = cast(mx.array, self["weight"])
        M = x_full.shape[-2] if x_full.ndim >= 2 else 1
        per_rank_N, K = weight.shape
        N_full = per_rank_N * self.group.size()
        mx.set_splitk_partitions_override(_splitk_override_for_unsharded(M, N_full, K))
        try:
            if "bias" in self:
                y = mx.addmm(cast(mx.array, self["bias"]), x_full, weight.T)
            else:
                y = mx.matmul(x_full, weight.T)
        finally:
            mx.set_splitk_partitions_override(0)
        return _all_gather_last(y, self.group)

    @classmethod
    def from_linear(cls, linear: nn.Linear, group: mx.distributed.Group) -> "NShardedLinear":
        out_dims, in_dims = linear.weight.shape  # pyright: ignore[reportAttributeAccessIssue]
        N = group.size()
        rank = group.rank()
        per_rank = out_dims // N
        instance = cls(in_dims, out_dims, hasattr(linear, "bias"), group)
        new_weight = cast(mx.array, linear["weight"])[rank * per_rank : (rank + 1) * per_rank]
        instance.update({"weight": new_weight})
        if hasattr(linear, "bias"):
            new_bias = cast(mx.array, linear["bias"])[rank * per_rank : (rank + 1) * per_rank]
            instance.update({"bias": new_bias})
        return instance


def _all_gather_last(x: mx.array, group: mx.distributed.Group) -> mx.array:
    """all_gather over the last axis.

    ``mx.distributed.all_gather`` concatenates along axis 0 and its internal
    ``ensure_row_contiguous`` forces a strided memcpy if the input view isn't
    contiguous. Naively wrapping with ``moveaxis`` produces two strided full-
    tensor memcpys per call (inside all_gather + at the next consumer). We
    sidestep that by (1) flattening all leading axes so the tensor becomes 2D
    ``(prefix, last_shard)``, (2) forcing a single contiguous transpose to
    ``(last_shard, prefix)``, (3) running the contiguous all_gather (no
    internal copy), and (4) transposing+reshaping back to
    ``(*leading, last_shard * N)``. One explicit memcpy instead of two
    strided ones.
    """
    leading = x.shape[:-1]
    last = x.shape[-1]
    x2 = x.reshape(-1, last)
    xt = mx.contiguous(x2.T)
    g = mx.distributed.all_gather(xt, group=group)
    return mx.contiguous(g.T).reshape(*leading, last * group.size())


class ShardedInputNorm(CustomMlxLayer):
    def __init__(self, norm: _LayerCallable, group: mx.distributed.Group):
        super().__init__(norm)
        self.group = group

    def __call__(self, x: mx.array) -> mx.array:
        return cast(mx.array, self.original_layer(_all_gather_last(x, self.group)))


class ShardedEmbedding(CustomMlxLayer):
    def __init__(self, embed: _LayerCallable, group: mx.distributed.Group):
        super().__init__(embed)
        self.group = group

    def __call__(self, ids: mx.array) -> mx.array:
        y = cast(mx.array, self.original_layer(ids))
        N = self.group.size()
        per_rank = y.shape[-1] // N
        rank = self.group.rank()
        return y[..., rank * per_rank : (rank + 1) * per_rank]


def _wrap_block_entry_norms(layer: nn.Module, group: mx.distributed.Group) -> None:
    children = layer.children() if hasattr(layer, "children") else {}
    to_wrap: list[str] = []
    for name, child in (children.items() if isinstance(children, dict) else children):  # pyright: ignore[reportGeneralTypeIssues]
        if isinstance(child, (nn.RMSNorm, nn.LayerNorm)):
            to_wrap.append(name)
    for name in to_wrap:
        orig = getattr(layer, name)
        setattr(layer, name, ShardedInputNorm(orig, group))


def _switch_mlp_activation(switch_mlp: object, x_gate: mx.array, x_up: mx.array) -> mx.array:
    activation = getattr(switch_mlp, "activation", None)
    if activation is not None:
        return cast(mx.array, activation(x_up, x_gate))
    return nn.silu(x_gate) * x_up


def _switch_mlp_n_sharded_sharded_out(
    switch_mlp: object,
    x: mx.array,
    indices: mx.array,
    scores: mx.array,
    group: mx.distributed.Group,
) -> mx.array:
    """Run a SwitchGLU-shaped MoE expert block with column-sharded down_proj
    and return the output STILL SHARDED on the hidden dim (H/N per rank).

    gate_proj/up_proj are all-to-sharded (output dim split on expert_hidden).
    down_proj is all-to-sharded (output dim split on model_dim). The
    intermediate is all_gathered to full before down_proj so each rank runs
    the full K reduction over intermediate_full. The per-rank down_proj
    output is weighted by the routing scores and summed across the top-k
    dimension, but NOT gathered to full — the caller is expected to combine
    sharded outputs and issue a single final all_gather.
    """
    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

    x_exp = mx.expand_dims(x, (-2, -3))
    do_sort = indices.size >= 64
    idx = indices
    inv_order = None
    if do_sort:
        x_exp, idx, inv_order = _gather_sort(x_exp, indices)

    gp = switch_mlp.gate_proj  # pyright: ignore[reportAttributeAccessIssue]
    up = switch_mlp.up_proj  # pyright: ignore[reportAttributeAccessIssue]
    dp = switch_mlp.down_proj  # pyright: ignore[reportAttributeAccessIssue]

    x_up = mx.gather_mm(
        x_exp,
        cast(mx.array, up["weight"]).swapaxes(-1, -2),
        rhs_indices=idx,
        sorted_indices=do_sort,
    )
    if "bias" in up:
        x_up = x_up + mx.expand_dims(cast(mx.array, up["bias"])[idx], -2)
    x_gate = mx.gather_mm(
        x_exp,
        cast(mx.array, gp["weight"]).swapaxes(-1, -2),
        rhs_indices=idx,
        sorted_indices=do_sort,
    )
    if "bias" in gp:
        x_gate = x_gate + mx.expand_dims(cast(mx.array, gp["bias"])[idx], -2)
    hidden_shard = _switch_mlp_activation(switch_mlp, x_gate, x_up)

    hidden_full = _all_gather_last(hidden_shard, group)

    out_shard = mx.gather_mm(
        hidden_full,
        cast(mx.array, dp["weight"]).swapaxes(-1, -2),
        rhs_indices=idx,
        sorted_indices=do_sort,
    )
    if "bias" in dp:
        out_shard = out_shard + mx.expand_dims(cast(mx.array, dp["bias"])[idx], -2)

    if do_sort:
        out_shard = _scatter_unsort(out_shard, inv_order, indices.shape)
    out_shard = out_shard.squeeze(-2)
    return (out_shard * scores[..., None]).sum(axis=-2).astype(out_shard.dtype)


def _switch_mlp_n_sharded(
    switch_mlp: object,
    x: mx.array,
    indices: mx.array,
    scores: mx.array,
    group: mx.distributed.Group,
) -> mx.array:
    out_shard = _switch_mlp_n_sharded_sharded_out(
        switch_mlp, x, indices, scores, group
    )
    return _all_gather_last(out_shard, group)


def _switch_fc_n_sharded(
    switch_mlp: object,
    x: mx.array,
    indices: mx.array,
    group: mx.distributed.Group,
    activation: Callable[[mx.array], mx.array],
) -> mx.array:
    """SwitchMLP variant (NemotronH): fc1 -> activation -> fc2, both SwitchLinears.

    Both fc1 and fc2 are column-sharded (output dim). fc1 output is
    all_gathered before fc2, and fc2 output is all_gathered afterward.
    """
    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

    x_exp = mx.expand_dims(x, (-2, -3))
    do_sort = indices.size >= 64
    idx = indices
    inv_order = None
    if do_sort:
        x_exp, idx, inv_order = _gather_sort(x_exp, indices)

    fc1 = switch_mlp.fc1  # pyright: ignore[reportAttributeAccessIssue]
    fc2 = switch_mlp.fc2  # pyright: ignore[reportAttributeAccessIssue]

    h_shard = mx.gather_mm(
        x_exp,
        cast(mx.array, fc1["weight"]).swapaxes(-1, -2),
        rhs_indices=idx,
        sorted_indices=do_sort,
    )
    h_shard = activation(h_shard)
    h_full = _all_gather_last(h_shard, group)

    out_shard = mx.gather_mm(
        h_full,
        cast(mx.array, fc2["weight"]).swapaxes(-1, -2),
        rhs_indices=idx,
        sorted_indices=do_sort,
    )
    out_full = _all_gather_last(out_shard, group)

    if do_sort:
        out_full = _scatter_unsort(out_full, inv_order, indices.shape)
    return out_full.squeeze(-2)


def _matmul_with_unsharded_splitk(
    x: mx.array, weight: mx.array, per_rank_N: int, K: int, group: mx.distributed.Group
) -> mx.array:
    """Per-rank matmul with splitk override forced to the unsharded count."""
    N_full = per_rank_N * group.size()
    M = x.shape[-2] if x.ndim >= 2 else 1
    mx.set_splitk_partitions_override(_splitk_override_for_unsharded(M, N_full, K))
    try:
        y = mx.matmul(x, weight.T)
    finally:
        mx.set_splitk_partitions_override(0)
    return y


def _mlp_n_sharded_sharded_out(
    mlp: object, x: mx.array, group: mx.distributed.Group
) -> mx.array:
    """Like _mlp_n_sharded but returns the output still sharded on H/N."""
    up = mlp.up_proj  # pyright: ignore[reportAttributeAccessIssue]
    dp = mlp.down_proj  # pyright: ignore[reportAttributeAccessIssue]
    up_w = cast(mx.array, up["weight"])
    dp_w = cast(mx.array, dp["weight"])

    x_up = _matmul_with_unsharded_splitk(x, up_w, up_w.shape[0], up_w.shape[1], group)
    if hasattr(mlp, "gate_proj"):
        gp = mlp.gate_proj  # pyright: ignore[reportAttributeAccessIssue]
        gp_w = cast(mx.array, gp["weight"])
        x_gate = _matmul_with_unsharded_splitk(
            x, gp_w, gp_w.shape[0], gp_w.shape[1], group
        )
        hidden_shard = nn.silu(x_gate) * x_up
    else:
        hidden_shard = nn.relu2(x_up)

    hidden_full = _all_gather_last(hidden_shard, group)

    return _matmul_with_unsharded_splitk(
        hidden_full, dp_w, dp_w.shape[0], dp_w.shape[1], group
    )


def _mlp_n_sharded(mlp: object, x: mx.array, group: mx.distributed.Group) -> mx.array:
    """SwiGLU MLP (gate_proj/up_proj/down_proj) with N-sharded down_proj.

    Falls back to up_proj + activation + down_proj for MLPs that don't have a
    gate_proj (e.g. NemotronHMLP). The sharded linears here were sharded
    via ``shard_inplace`` so they are still plain ``nn.Linear`` instances;
    we run their matmuls directly with the splitk override so each per-rank
    matmul produces a bf16 output that's bit-exact per column to what the
    unsharded kernel would produce.
    """
    out_shard = _mlp_n_sharded_sharded_out(mlp, x, group)
    return _all_gather_last(out_shard, group)


class ShardedMoE(CustomMlxLayer):
    """Wraps a MoE block to use N-sharded (column) down_proj + all_gather.

    Each sharded down_proj output element is bit-exact per column to the
    unsharded tp=1 kernel's output (every (m, n) goes through the full K-fold
    the unsharded kernel uses). all_gather is a pure bf16 byte-shuffle with
    no rounding introduced. The MoE's internal routing (gate + softmax +
    argpartition) is unchanged.

    Dispatches by inspecting the wrapped MoE's attributes to handle the
    different MoE block APIs across model families.
    """

    def __init__(self, layer: _LayerCallable):
        super().__init__(layer)
        self.sharding_group: mx.distributed.Group | None = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is None:
            return cast(mx.array, self.original_layer.__call__(x))

        moe = self.original_layer
        if hasattr(moe, "switch_mlp") and hasattr(moe, "shared_expert") and hasattr(
            moe, "shared_expert_gate"
        ):
            return self._qwen_style(x)
        if hasattr(moe, "switch_mlp") and hasattr(moe.switch_mlp, "fc1"):
            return self._nemotron_h_style(x)
        if hasattr(moe, "switch_mlp") and hasattr(moe, "shared_experts"):
            return self._deepseek_style(x)
        if hasattr(moe, "switch_mlp") and hasattr(moe, "e_score_correction_bias"):
            return self._minimax_style(x)
        if hasattr(moe, "switch_mlp") and hasattr(moe, "share_expert"):
            return self._nemotronh_style(x)
        if hasattr(moe, "switch_mlp"):
            return self._generic_switch_mlp_style(x)
        if hasattr(moe, "experts") and hasattr(moe, "router"):
            return self._gpt_oss_style(x)

        x = sum_gradients(self.sharding_group)(x)
        return cast(mx.array, moe.__call__(x))

    def _route_softmax_topk(self, moe: object, x: mx.array) -> tuple[mx.array, mx.array]:
        gates = moe.gate(x)  # pyright: ignore[reportAttributeAccessIssue]
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = getattr(moe, "top_k", None) or moe.num_experts_per_tok  # pyright: ignore[reportAttributeAccessIssue]
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if getattr(moe, "norm_topk_prob", False):
            scores = scores / scores.sum(axis=-1, keepdims=True)
        return inds, scores

    def _qwen_style(self, x: mx.array) -> mx.array:
        assert self.sharding_group is not None
        x = sum_gradients(self.sharding_group)(x)
        moe = self.original_layer
        inds, scores = self._route_softmax_topk(moe, x)
        y_shard = _switch_mlp_n_sharded_sharded_out(
            moe.switch_mlp, x, inds, scores, self.sharding_group  # pyright: ignore[reportAttributeAccessIssue]
        )
        shared_shard = _mlp_n_sharded_sharded_out(moe.shared_expert, x, self.sharding_group)  # pyright: ignore[reportAttributeAccessIssue]
        shared_shard = mx.sigmoid(moe.shared_expert_gate(x)) * shared_shard  # pyright: ignore[reportAttributeAccessIssue]
        return _all_gather_last(y_shard + shared_shard, self.sharding_group)

    def _deepseek_style(self, x: mx.array) -> mx.array:
        assert self.sharding_group is not None
        x = sum_gradients(self.sharding_group)(x)
        moe = self.original_layer
        inds, scores = moe.gate(x)  # pyright: ignore[reportAttributeAccessIssue]
        y_shard = _switch_mlp_n_sharded_sharded_out(
            moe.switch_mlp, x, inds, scores, self.sharding_group  # pyright: ignore[reportAttributeAccessIssue]
        )
        if getattr(moe.config, "n_shared_experts", None) is not None:  # pyright: ignore[reportAttributeAccessIssue]
            y_shard = y_shard + _mlp_n_sharded_sharded_out(
                moe.shared_experts, x, self.sharding_group  # pyright: ignore[reportAttributeAccessIssue]
            )
        return _all_gather_last(y_shard, self.sharding_group)

    def _minimax_style(self, x: mx.array) -> mx.array:
        assert self.sharding_group is not None
        x = sum_gradients(self.sharding_group)(x)
        moe = self.original_layer
        gates = moe.gate(x.astype(mx.float32))  # pyright: ignore[reportAttributeAccessIssue]
        scores = mx.sigmoid(gates)
        orig_scores = scores
        scores = scores + moe.e_score_correction_bias  # pyright: ignore[reportAttributeAccessIssue]
        k = moe.num_experts_per_tok  # pyright: ignore[reportAttributeAccessIssue]
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(orig_scores, inds, axis=-1)
        scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
        scores = scores.astype(x.dtype)
        y = _switch_mlp_n_sharded(moe.switch_mlp, x, inds, scores, self.sharding_group)  # pyright: ignore[reportAttributeAccessIssue]
        return y

    def _nemotronh_style(self, x: mx.array) -> mx.array:
        """Handles both NemotronH and Step35 (gate returns indices+weights directly)."""
        assert self.sharding_group is not None
        x = sum_gradients(self.sharding_group)(x)
        moe = self.original_layer
        gate_out = moe.gate(x)  # pyright: ignore[reportAttributeAccessIssue]
        if isinstance(gate_out, tuple):
            inds, scores = gate_out
        else:
            inds, scores = self._route_softmax_topk(moe, x)
        y = _switch_mlp_n_sharded(moe.switch_mlp, x, inds, scores, self.sharding_group)  # pyright: ignore[reportAttributeAccessIssue]
        if getattr(moe, "share_expert", None) is not None:
            y = y + _mlp_n_sharded(moe.share_expert, x, self.sharding_group)  # pyright: ignore[reportAttributeAccessIssue]
        return y

    def _nemotron_h_style(self, x: mx.array) -> mx.array:
        assert self.sharding_group is not None
        moe = self.original_layer
        residuals = x
        inds, scores = moe.gate(x)  # pyright: ignore[reportAttributeAccessIssue]
        if moe.moe_latent_size is not None:  # pyright: ignore[reportAttributeAccessIssue]
            x = moe.fc1_latent_proj(x)  # pyright: ignore[reportAttributeAccessIssue]
        y = _switch_fc_n_sharded(
            moe.switch_mlp,  # pyright: ignore[reportAttributeAccessIssue]
            x,
            inds,
            self.sharding_group,
            moe.switch_mlp.activation,  # pyright: ignore[reportAttributeAccessIssue]
        )
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if moe.moe_latent_size is not None:  # pyright: ignore[reportAttributeAccessIssue]
            y = moe.fc2_latent_proj(y)  # pyright: ignore[reportAttributeAccessIssue]
        if moe.config.n_shared_experts is not None:  # pyright: ignore[reportAttributeAccessIssue]
            y = y + _mlp_n_sharded(moe.shared_experts, residuals, self.sharding_group)  # pyright: ignore[reportAttributeAccessIssue]
        return y

    def _gpt_oss_style(self, x: mx.array) -> mx.array:
        from mlx_lm.models.gpt_oss import mlx_topk  # pyright: ignore[reportUnknownVariableType]

        assert self.sharding_group is not None
        x = sum_gradients(self.sharding_group)(x)
        moe = self.original_layer
        g = moe.router(x)  # pyright: ignore[reportAttributeAccessIssue]
        expert_weights, indices = mlx_topk(g, k=moe.num_experts_per_tok, axis=-1)  # pyright: ignore[reportAttributeAccessIssue]
        expert_weights = mx.softmax(expert_weights, axis=-1, precise=True)
        y = _switch_mlp_n_sharded(moe.experts, x, indices, expert_weights, self.sharding_group)  # pyright: ignore[reportAttributeAccessIssue]
        return y

    def _generic_switch_mlp_style(self, x: mx.array) -> mx.array:
        assert self.sharding_group is not None
        x = sum_gradients(self.sharding_group)(x)
        moe = self.original_layer
        gate_out = moe.gate(x)  # pyright: ignore[reportAttributeAccessIssue]
        if isinstance(gate_out, tuple):
            inds, scores = gate_out
        else:
            inds, scores = self._route_softmax_topk(moe, x)
        y = _switch_mlp_n_sharded(moe.switch_mlp, x, inds, scores, self.sharding_group)  # pyright: ignore[reportAttributeAccessIssue]
        return y


class GLM4MoeLiteShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(GLM4MoeLiteModel, model)
        total = len(model.layers)  # type: ignore
        for i, layer in enumerate(model.layers):  # type: ignore
            layer = cast(Glm4MoeLiteDecoderLayer, layer)
            mx.eval(layer.parameters())
            if layer.self_attn.q_lora_rank is None:  # type: ignore
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj
                )
            else:
                layer.self_attn.q_b_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_b_proj
                )

            layer.self_attn.o_proj = NShardedLinear.from_linear(layer.self_attn.o_proj, self.group)
            layer.self_attn.num_heads //= self.N

            # Logic from upstream mlx
            num_heads = layer.self_attn.num_heads
            sh = self.group.rank() * num_heads
            eh = sh + num_heads

            def shard_heads(w: mx.array, sh: int = sh, eh: int = eh) -> mx.array:
                return w[sh:eh]

            layer.self_attn.embed_q.apply(shard_heads)
            layer.self_attn.unembed_out.apply(shard_heads)

            if isinstance(layer.mlp, Glm4MoeLiteMLP):
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = NShardedLinear.from_linear(layer.mlp.down_proj, self.group)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            else:
                if getattr(layer.mlp, "shared_experts", None) is not None:
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.gate_proj
                    )
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.up_proj
                    )
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.down_proj
                    )
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.down_proj)
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
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(MiniMaxModel, model)
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())
            # Shard the self attention
            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = NShardedLinear.from_linear(layer.self_attn.o_proj, self.group)

            layer.self_attn.num_attention_heads //= self.N
            layer.self_attn.num_key_value_heads //= self.N

            layer.self_attn = WrappedMiniMaxAttention(layer.self_attn, self.group)  # pyright: ignore[reportAttributeAccessIssue,reportArgumentType]

            # Shard the MoE.
            self.all_to_sharded_linear_in_place(
                layer.block_sparse_moe.switch_mlp.gate_proj
            )
            self.all_to_sharded_linear_in_place(
                layer.block_sparse_moe.switch_mlp.up_proj
            )
            self.all_to_sharded_linear_in_place(
                layer.block_sparse_moe.switch_mlp.down_proj
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
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(
            Qwen3Model
            | Qwen3MoeModel
            | Qwen3NextModel
            | Qwen3_5TextModel
            | Qwen3_5MoeModel
            | Qwen3VLModel,
            model,
        )
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())
            # Shard the self attention
            if isinstance(layer, (Qwen3MoeDecoderLayer, Qwen3TransformerBlock)):
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj
                )
                layer.self_attn.k_proj = self.all_to_sharded_linear(
                    layer.self_attn.k_proj
                )
                layer.self_attn.v_proj = self.all_to_sharded_linear(
                    layer.self_attn.v_proj
                )
                layer.self_attn.o_proj = NShardedLinear.from_linear(
                    layer.self_attn.o_proj, self.group
                )
                layer.self_attn.n_heads //= self.N
                layer.self_attn.n_kv_heads //= self.N
            else:
                assert isinstance(layer, (Qwen3NextDecoderLayer, Qwen3_5DecoderLayer))
                if hasattr(layer, "linear_attn"):
                    linear_attn = layer.linear_attn

                    if isinstance(linear_attn, Qwen3NextGatedDeltaNet):
                        # Qwen3-Next: combined projections
                        linear_attn.in_proj_qkvz = self.all_to_sharded_linear(
                            linear_attn.in_proj_qkvz
                        )
                        linear_attn.in_proj_ba = self.all_to_sharded_linear(
                            linear_attn.in_proj_ba
                        )
                    else:
                        # Qwen3.5: separate projections
                        # in_proj_qkv has sections [q(key_dim), k(key_dim), v(value_dim)]
                        # that must be split section-aware, not as a contiguous block
                        key_dim = linear_attn.key_dim
                        value_dim = linear_attn.value_dim
                        linear_attn.in_proj_qkv = shard_linear(
                            linear_attn.in_proj_qkv,
                            "all-to-sharded",
                            segments=[key_dim, key_dim + key_dim],
                            group=self.group,
                        )
                        linear_attn.in_proj_z = self.all_to_sharded_linear(
                            linear_attn.in_proj_z
                        )
                        linear_attn.in_proj_b = self.all_to_sharded_linear(
                            linear_attn.in_proj_b
                        )
                        linear_attn.in_proj_a = self.all_to_sharded_linear(
                            linear_attn.in_proj_a
                        )
                    linear_attn.out_proj = NShardedLinear.from_linear(
                        linear_attn.out_proj, self.group
                    )

                    # Shard conv1d: depthwise conv with non-contiguous channel slicing.
                    # Channel layout is [q(key_dim), k(key_dim), v(value_dim)].
                    # Each rank takes its head-slice from each of the three sections.
                    rank = self.group.rank()
                    key_dim = linear_attn.key_dim
                    value_dim = linear_attn.value_dim
                    key_dim_shard = key_dim // self.N
                    value_dim_shard = value_dim // self.N

                    q_idx = mx.arange(rank * key_dim_shard, (rank + 1) * key_dim_shard)
                    k_idx = mx.arange(
                        key_dim + rank * key_dim_shard,
                        key_dim + (rank + 1) * key_dim_shard,
                    )
                    v_idx = mx.arange(
                        2 * key_dim + rank * value_dim_shard,
                        2 * key_dim + (rank + 1) * value_dim_shard,
                    )
                    conv_indices = mx.concatenate([q_idx, k_idx, v_idx])
                    linear_attn.conv1d.weight = linear_attn.conv1d.weight[conv_indices]
                    new_conv_dim = key_dim_shard * 2 + value_dim_shard
                    linear_attn.conv1d.groups = new_conv_dim

                    num_v_shard = linear_attn.num_v_heads // self.N
                    v_start = rank * num_v_shard
                    v_end = v_start + num_v_shard
                    linear_attn.A_log = linear_attn.A_log[v_start:v_end]
                    linear_attn.dt_bias = linear_attn.dt_bias[v_start:v_end]

                    linear_attn.num_k_heads //= self.N
                    linear_attn.num_v_heads //= self.N
                    linear_attn.key_dim = (
                        linear_attn.head_k_dim * linear_attn.num_k_heads
                    )
                    linear_attn.value_dim = (
                        linear_attn.head_v_dim * linear_attn.num_v_heads
                    )
                    linear_attn.conv_dim = (
                        linear_attn.key_dim * 2 + linear_attn.value_dim
                    )
                else:
                    layer.self_attn.q_proj = self.all_to_sharded_linear(
                        layer.self_attn.q_proj
                    )
                    layer.self_attn.k_proj = self.all_to_sharded_linear(
                        layer.self_attn.k_proj
                    )
                    layer.self_attn.v_proj = self.all_to_sharded_linear(
                        layer.self_attn.v_proj
                    )
                    layer.self_attn.o_proj = NShardedLinear.from_linear(
                        layer.self_attn.o_proj, self.group
                    )
                    layer.self_attn.num_attention_heads //= self.N
                    layer.self_attn.num_key_value_heads //= self.N

            # Shard the MoE. Down_proj is column-sharded (output dim) so each
            # per-rank matmul runs the full K-reduction and its bf16 output is
            # bit-exact per column to tp=1. ShardedMoE.__call__ inserts the
            # required all_gather of the intermediate before down_proj and of
            # the output after.
            if isinstance(
                layer.mlp,
                (
                    Qwen3MoeSparseMoeBlock,
                    Qwen3NextSparseMoeBlock,
                    Qwen3_5SparseMoeBlock,
                ),
            ):
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.down_proj)
                if isinstance(
                    layer.mlp, (Qwen3NextSparseMoeBlock, Qwen3_5SparseMoeBlock)
                ):
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_expert.gate_proj
                    )
                    self.all_to_sharded_linear_in_place(layer.mlp.shared_expert.up_proj)
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_expert.down_proj
                    )
                layer.mlp = ShardedMoE(layer.mlp)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
                layer.mlp.sharding_group = self.group

            # Shard the MLP
            else:
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = NShardedLinear.from_linear(layer.mlp.down_proj, self.group)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(i, total)
        return model


class Glm4MoeShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(Glm4MoeModel, model)
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = NShardedLinear.from_linear(layer.self_attn.o_proj, self.group)
            layer.self_attn.n_heads //= self.N
            layer.self_attn.n_kv_heads //= self.N

            if isinstance(layer.mlp, MoE):
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.down_proj)
                if getattr(layer.mlp, "shared_experts", None) is not None:
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.gate_proj
                    )
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.up_proj
                    )
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.down_proj
                    )
                layer.mlp = ShardedMoE(layer.mlp)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
                layer.mlp.sharding_group = self.group

            else:
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = NShardedLinear.from_linear(layer.mlp.down_proj, self.group)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(i, total)
        return model


class GptOssShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(GptOssMoeModel, model)
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())
            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = NShardedLinear.from_linear(layer.self_attn.o_proj, self.group)

            layer.self_attn.num_attention_heads //= self.N
            layer.self_attn.num_key_value_heads //= self.N
            layer.self_attn.num_key_value_groups = (
                layer.self_attn.num_attention_heads
                // layer.self_attn.num_key_value_heads
            )

            layer.self_attn.sinks = layer.self_attn.sinks[
                layer.self_attn.num_attention_heads
                * self.group.rank() : layer.self_attn.num_attention_heads
                * (self.group.rank() + 1)
            ]

            self.all_to_sharded_linear_in_place(layer.mlp.experts.gate_proj)
            self.all_to_sharded_linear_in_place(layer.mlp.experts.up_proj)
            self.all_to_sharded_linear_in_place(layer.mlp.experts.down_proj)

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
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(Step35Model, model)
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())
            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = NShardedLinear.from_linear(layer.self_attn.o_proj, self.group)

            layer.self_attn.num_heads //= self.N
            layer.self_attn.num_kv_heads //= self.N

            if getattr(layer.self_attn, "use_head_wise_attn_gate", False):
                layer.self_attn.g_proj = self.all_to_sharded_linear(
                    layer.self_attn.g_proj
                )

            if isinstance(layer.mlp, Step35MLP):
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)
                layer.mlp.down_proj = NShardedLinear.from_linear(layer.mlp.down_proj, self.group)
            else:
                self.all_to_sharded_linear_in_place(layer.mlp.share_expert.gate_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.share_expert.up_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.share_expert.down_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.down_proj)
                layer.mlp = ShardedMoE(layer.mlp)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
                layer.mlp.sharding_group = self.group

            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(i, total)
        return model


class NemotronHShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(NemotronHModel, model)
        rank = self.group.rank()
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            mixer = layer.mixer

            if isinstance(mixer, NemotronHAttention):
                mixer.q_proj = self.all_to_sharded_linear(mixer.q_proj)
                mixer.k_proj = self.all_to_sharded_linear(mixer.k_proj)
                mixer.v_proj = self.all_to_sharded_linear(mixer.v_proj)
                mixer.o_proj = NShardedLinear.from_linear(mixer.o_proj, self.group)
                mixer.num_heads //= self.N
                mixer.num_key_value_heads //= self.N

            elif isinstance(mixer, NemotronHMamba2Mixer):
                self._shard_mamba2_mixer(mixer, rank)

            elif isinstance(mixer, NemotronHMoE):
                # N-shard both fc1 and fc2 so each per-rank matmul runs the
                # full K reduction (bit-exact per column). ShardedMoE does the
                # all_gather of the intermediate between fc1 and fc2 and of the
                # fc2 output.
                self.all_to_sharded_linear_in_place(mixer.switch_mlp.fc1)
                self.all_to_sharded_linear_in_place(mixer.switch_mlp.fc2)
                if hasattr(mixer, "shared_experts"):
                    self.all_to_sharded_linear_in_place(mixer.shared_experts.gate_proj)
                    self.all_to_sharded_linear_in_place(mixer.shared_experts.up_proj)
                    self.all_to_sharded_linear_in_place(mixer.shared_experts.down_proj)
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

        # Per-rank sizes
        heads_per_rank = num_heads // world_size
        groups_per_rank = n_groups // world_size
        is_per_rank = heads_per_rank * head_dim
        bc_per_rank = groups_per_rank * ssm_state_size

        # === in_proj: output layout is [gate:IS | conv_ssm:IS | B:NG*SS | C:NG*SS | dt:NH] ===
        gate_start = 0
        conv_ssm_start = intermediate_size
        b_start = 2 * intermediate_size
        c_start = b_start + n_groups * ssm_state_size
        dt_start = c_start + n_groups * ssm_state_size

        # Build index tensor for this rank's slice of each section
        gate_idx = mx.arange(
            gate_start + rank * is_per_rank, gate_start + (rank + 1) * is_per_rank
        )
        conv_ssm_idx = mx.arange(
            conv_ssm_start + rank * is_per_rank,
            conv_ssm_start + (rank + 1) * is_per_rank,
        )
        b_idx = mx.arange(
            b_start + rank * bc_per_rank, b_start + (rank + 1) * bc_per_rank
        )
        c_idx = mx.arange(
            c_start + rank * bc_per_rank, c_start + (rank + 1) * bc_per_rank
        )
        dt_idx = mx.arange(
            dt_start + rank * heads_per_rank, dt_start + (rank + 1) * heads_per_rank
        )

        indices = mx.concatenate([gate_idx, conv_ssm_idx, b_idx, c_idx, dt_idx])
        mixer.in_proj.weight = mixer.in_proj.weight[indices]

        # === out_proj: input is intermediate_size (sharded) → hidden_size (reduce) ===
        mixer.out_proj = NShardedLinear.from_linear(mixer.out_proj, self.group)

        # === conv1d: depthwise conv on conv_dim channels ===
        # conv_dim layout: [ssm_hidden:IS | B:NG*SS | C:NG*SS]
        conv_ssm_idx_local = mx.arange(rank * is_per_rank, (rank + 1) * is_per_rank)
        conv_b_idx = mx.arange(
            intermediate_size + rank * bc_per_rank,
            intermediate_size + (rank + 1) * bc_per_rank,
        )
        conv_c_idx = mx.arange(
            intermediate_size + n_groups * ssm_state_size + rank * bc_per_rank,
            intermediate_size + n_groups * ssm_state_size + (rank + 1) * bc_per_rank,
        )
        conv_indices = mx.concatenate([conv_ssm_idx_local, conv_b_idx, conv_c_idx])
        mixer.conv1d.weight = mixer.conv1d.weight[conv_indices]
        new_conv_dim = is_per_rank + 2 * bc_per_rank
        mixer.conv1d.groups = new_conv_dim
        if mixer.conv1d.bias is not None:
            mixer.conv1d.bias = mixer.conv1d.bias[conv_indices]

        # === Per-head parameters ===
        h_start = rank * heads_per_rank
        h_end = h_start + heads_per_rank
        mixer.dt_bias = mixer.dt_bias[h_start:h_end]
        mixer.A_log = mixer.A_log[h_start:h_end]
        mixer.D = mixer.D[h_start:h_end]

        # === Norm: weight is intermediate_size ===
        mixer.norm.weight = mixer.norm.weight[
            rank * is_per_rank : (rank + 1) * is_per_rank
        ]

        # === Update dimensions ===
        mixer.num_heads = heads_per_rank
        mixer.n_groups = groups_per_rank
        mixer.intermediate_size = is_per_rank
        mixer.conv_dim = new_conv_dim
        mixer.heads_per_group = heads_per_rank // groups_per_rank


class WrappedGemma4Experts(CustomMlxLayer):
    def __init__(self, layer: _LayerCallable):
        super().__init__(layer)
        self.sharding_group: mx.distributed.Group | None = None

    def __call__(
        self, x: mx.array, top_k_indices: mx.array, top_k_weights: mx.array
    ) -> mx.array:
        if self.sharding_group is None:
            return cast(mx.array, self.original_layer(x, top_k_indices, top_k_weights))
        x = sum_gradients(self.sharding_group)(x)
        switch_glu = self.original_layer.switch_glu  # pyright: ignore[reportAttributeAccessIssue]
        return _switch_mlp_n_sharded(
            switch_glu, x, top_k_indices, top_k_weights, self.sharding_group
        )


class Gemma4ShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(Gemma4Model, model)
        layers = model.language_model.model.layers
        total = len(layers)
        for i, layer in enumerate(layers):
            mx.eval(layer.parameters())

            attn = layer.self_attn
            attn.q_proj = self.all_to_sharded_linear(attn.q_proj)
            attn.k_proj = self.all_to_sharded_linear(attn.k_proj)
            if not attn.use_k_eq_v:
                attn.v_proj = self.all_to_sharded_linear(attn.v_proj)
            attn.o_proj = NShardedLinear.from_linear(attn.o_proj, self.group)
            attn.n_heads //= self.N
            attn.n_kv_heads //= self.N

            layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
            layer.mlp.down_proj = NShardedLinear.from_linear(layer.mlp.down_proj, self.group)
            layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            if layer.enable_moe:
                self.all_to_sharded_linear_in_place(layer.experts.switch_glu.gate_proj)
                self.all_to_sharded_linear_in_place(layer.experts.switch_glu.up_proj)
                self.all_to_sharded_linear_in_place(layer.experts.switch_glu.down_proj)
                layer.experts = WrappedGemma4Experts(layer.experts)  # pyright: ignore[reportAttributeAccessIssue,reportArgumentType]
                layer.experts.sharding_group = self.group

            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(i, total)
        return model
