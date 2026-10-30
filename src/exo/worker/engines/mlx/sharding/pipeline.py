from inspect import signature
from typing import Literal, Protocol, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache, KVCache

from exo.worker.engines.mlx.sharding.blocks import CustomMlxLayer, LayerCallable

_pending_prefill_sends: list[tuple[mx.array, int, mx.distributed.Group]] = []


def flush_prefill_sends() -> None:
    for output, dst, group in _pending_prefill_sends:
        sent = mx.distributed.send(output, dst, group=group)
        mx.async_eval(sent)
    _pending_prefill_sends.clear()


def clear_prefill_sends() -> None:
    # Discard pending sends (e.g. on cancellation).
    _pending_prefill_sends.clear()


# Decode-step receives this rank's first layer has posted and its last layer has not yet
# waited for (see PipelineFirstLayer).
_pending_recvs: list[mx.array] = []

# Model modules whose forward reads on the host (`.tolist()`, `.item()`, ...) a value
# computed from a layer's input. Such a read in the middle of a shard would evaluate a
# pending receive together with GPU work, so their ranks keep waiting for the receive in
# PipelineFirstLayer. Audited over mlx-lm's models and exo's model families: DBRX's MoE
# lists each token's experts on the host (`inds.tolist()`). test_pipeline_decode_receive.py
# fails when a forward with a host read appears that is not listed here.
EAGER_RECEIVE_MODEL_MODULES = frozenset({"mlx_lm.models.dbrx"})


def receive_can_be_deferred(model: nn.Module) -> bool:
    """Whether a pipeline rank of `model` may build its decode graph on a pending receive."""
    return type(model).__module__ not in EAGER_RECEIVE_MODEL_MODULES


def wait_for_pending_recvs() -> None:
    """Wait on the CPU for the activations this rank's first layer is receiving. Only the
    receives are evaluated here: no GPU work depends on an unfinished collective."""
    if _pending_recvs:
        mx.eval(*_pending_recvs)
        _pending_recvs.clear()


class PipelineFirstLayer(CustomMlxLayer):
    def __init__(
        self,
        original_layer: LayerCallable,
        r: int,
        group: mx.distributed.Group,
        defer_receive: bool = True,
    ):
        super().__init__(original_layer)
        self.r: int = r
        self.group = group
        self.is_prefill: bool = False
        # False for models that read layer values on the host mid-shard (see
        # EAGER_RECEIVE_MODEL_MODULES).
        self.defer_receive = defer_receive

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        if self.r != 0:
            # We want to avoid GPU timeout errors by evalling the distributed operation
            # so that it stays on CPU, which does not have a timeout.
            mx.eval(x)
            wait_for_pending_recvs()
            x = mx.distributed.recv_like(x, (self.r - 1), group=self.group)
            if self.is_prefill or not self.defer_receive:
                mx.eval(x)
            else:
                # Decode: build the rest of this shard's graph while the activation is
                # still on its way, rather than after it arrives. PipelineLastLayer waits
                # for the receive on the CPU, on its own, before it evaluates anything
                # on the GPU, so the GPU still never waits on the network.
                _pending_recvs.append(x)
        return self.original_layer(x, *args, **kwargs)


class PipelineLastLayer(CustomMlxLayer):
    def __init__(
        self,
        original_layer: LayerCallable,
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

        # The receive this shard's graph is built on, on the CPU first (see
        # PipelineFirstLayer).
        wait_for_pending_recvs()
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


def get_layers(inner_model_instance: nn.Module) -> list[LayerCallable]:
    # Handle both model.layers and model.h cases
    layers: list[LayerCallable]
    if hasattr(inner_model_instance, "layers"):
        layers = cast(list[LayerCallable], inner_model_instance.layers)
    elif hasattr(inner_model_instance, "h"):
        layers = cast(list[LayerCallable], inner_model_instance.h)
    else:
        raise ValueError("Model must have either a 'layers' or 'h' attribute")

    return layers


class PipelineShardFixup(Protocol):
    """A model family's adjustments after its layers were cut to [start_layer, end_layer):
    e.g. recomputing which cache entry its attention mask is built from."""

    def __call__(
        self,
        model: nn.Module,
        inner_model_instance: nn.Module,
        layers: list[LayerCallable],
        start_layer: int,
        end_layer: int,
    ) -> None: ...


class HybridCacheModel(Protocol):
    """A model whose cache mixes attention (KV) and linear-attention/SSM entries."""

    def make_cache(self) -> list[ArraysCache | KVCache]: ...


def patch_hybrid_cache(
    model: HybridCacheModel,
    fa_idx: int,
    has_full_attn: bool,
    ssm_idx: int,
    has_linear: bool,
) -> None:
    """For a pipeline shard of a hybrid model that holds only one kind of layer, keep
    make_mask working on the cache entry the model reads its mask from."""
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
