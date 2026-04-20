"""Install the attention/MoE split DecoderLayer.__call__ on a Qwen3.5 MoE model.

Unlike patches/qwen3_5_moe/apply.py (which installs fused local kernels),
this patch replaces DecoderLayer.__call__ with a two-rank split: rank 0 runs
attention + first residual, rank 1 runs post_attention_layernorm + MoE +
second residual, with one cross-rank send/recv pair per layer.

Must be called instead of apply_qwen35_batched_fused_patches — they both
replace DecoderLayer.__call__. In exo's current distributed load path
(utils_mlx.shard_and_load) maybe_apply_patches is only called in the
single-device branch, so there is no conflict for distributed AttnMoeSplit
runs — we call this directly from attn_moe_split_auto_parallel.
"""

import os

import mlx.core as mx
import mlx.nn as nn
from loguru import logger
from mlx_lm.models import cache as cache_module
from mlx_lm.models.qwen3_5 import DecoderLayer, Qwen3_5TextModel

from .decoder import ATTN_RANK, MOE_RANK, make_split_decoder_call
from .model_forward import (
    make_pipelined_model_call,
    make_pipelined_speculative_forward,
)

_EMPTY_KV_PLACEHOLDER = mx.zeros((1, 1, 0, 1))


def _patch_caches_for_moe_rank() -> None:
    """Return zero-length placeholder arrays when caches are empty.

    Rank 1 (MoE) never runs attention, so its per-layer KVCache /
    BatchKVCache / BatchRotatingKVCache / ArraysCache never have their keys
    populated. mlx_lm.generate and BatchGenerator both call
    ``mx.eval([c.state for c in prompt_cache])`` once per step, which
    invokes the cache's state property and crashes on
    ``self.keys.shape[2]`` (AttributeError: 'NoneType' object has no
    attribute 'shape'). Patch each used cache class's state property on
    rank 1 only, so that an unpopulated cache reports empty placeholder
    arrays whose eval is a no-op.
    """

    # KVCache.state → (k, v)
    def _kv_state(self):  # type: ignore[no-untyped-def]
        if self.keys is None:
            return (_EMPTY_KV_PLACEHOLDER, _EMPTY_KV_PLACEHOLDER)
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        return (
            self.keys[..., : self.offset, :],
            self.values[..., : self.offset, :],
        )

    def _kv_state_setter(self, v):  # type: ignore[no-untyped-def]
        self.keys, self.values = v
        self.offset = self.keys.shape[2]

    cache_module.KVCache.state = property(_kv_state, _kv_state_setter)  # type: ignore[method-assign]

    # BatchKVCache.state → (k, v, offset, left_padding)
    def _batch_kv_state(self):  # type: ignore[no-untyped-def]
        if self.keys is None:
            return (
                _EMPTY_KV_PLACEHOLDER,
                _EMPTY_KV_PLACEHOLDER,
                self.offset,
                self.left_padding,
            )
        k, v = self.keys, self.values
        if self._idx < k.shape[2]:
            k = k[..., : self._idx, :]
            v = v[..., : self._idx, :]
        return k, v, self.offset, self.left_padding

    def _batch_kv_state_setter(self, v):  # type: ignore[no-untyped-def]
        self.keys, self.values, self.offset, self.left_padding = v
        self._idx = self.keys.shape[2]

    cache_module.BatchKVCache.state = property(  # type: ignore[method-assign]
        _batch_kv_state, _batch_kv_state_setter
    )

    # BatchRotatingKVCache.state → (k, v, offset, left_padding)
    def _batch_rot_state(self):  # type: ignore[no-untyped-def]
        if self.keys is None:
            return (
                _EMPTY_KV_PLACEHOLDER,
                _EMPTY_KV_PLACEHOLDER,
                self.offset,
                self.left_padding,
            )
        k, v = self.keys, self.values
        if self._offset < k.shape[2]:
            k, v = k[..., : self._offset, :], v[..., : self._offset, :]
        return k, v, self.offset, self.left_padding

    def _batch_rot_state_setter(self, v):  # type: ignore[no-untyped-def]
        self.keys, self.values, self.offset, self.left_padding = v

    cache_module.BatchRotatingKVCache.state = property(  # type: ignore[method-assign]
        _batch_rot_state, _batch_rot_state_setter
    )

    # ArraysCache.state → list that may contain None entries
    original_arrays_state_getter = cache_module.ArraysCache.state.fget  # type: ignore[attr-defined]
    original_arrays_state_setter = cache_module.ArraysCache.state.fset  # type: ignore[attr-defined]

    def _arrays_state(self):  # type: ignore[no-untyped-def]
        raw = original_arrays_state_getter(self)
        return [_EMPTY_KV_PLACEHOLDER if c is None else c for c in raw]

    cache_module.ArraysCache.state = property(  # type: ignore[method-assign]
        _arrays_state, original_arrays_state_setter
    )

    # Batching methods (extract / filter / extend) short-circuit when the
    # cache is still in its uninitialised state on rank 1. Every method
    # defers to the stock implementation once keys/values are populated,
    # which would only happen if future code actually ran attention on
    # rank 1 — for now that never occurs, but the guards keep the patches
    # safe under that change.

    def _arrays_is_empty(self) -> bool:  # type: ignore[no-untyped-def]
        return all(c is None for c in self.cache)

    _arrays_extract_orig = cache_module.ArraysCache.extract

    def _arrays_extract(self, idx):  # type: ignore[no-untyped-def]
        if _arrays_is_empty(self):
            return cache_module.ArraysCache(len(self.cache))
        return _arrays_extract_orig(self, idx)

    cache_module.ArraysCache.extract = _arrays_extract  # type: ignore[method-assign]

    _arrays_filter_orig = cache_module.ArraysCache.filter

    def _arrays_filter(self, batch_indices):  # type: ignore[no-untyped-def]
        if _arrays_is_empty(self):
            return
        _arrays_filter_orig(self, batch_indices)

    cache_module.ArraysCache.filter = _arrays_filter  # type: ignore[method-assign]

    _arrays_extend_orig = cache_module.ArraysCache.extend

    def _arrays_extend(self, other):  # type: ignore[no-untyped-def]
        if _arrays_is_empty(self) and _arrays_is_empty(other):
            return
        _arrays_extend_orig(self, other)

    cache_module.ArraysCache.extend = _arrays_extend  # type: ignore[method-assign]

    # BatchKVCache: keys/values stay None until attention runs.
    _bkv_extract_orig = cache_module.BatchKVCache.extract

    def _bkv_extract(self, idx):  # type: ignore[no-untyped-def]
        if self.keys is None:
            return cache_module.KVCache()
        return _bkv_extract_orig(self, idx)

    cache_module.BatchKVCache.extract = _bkv_extract  # type: ignore[method-assign]

    _bkv_filter_orig = cache_module.BatchKVCache.filter

    def _bkv_filter(self, batch_indices):  # type: ignore[no-untyped-def]
        if self.keys is None:
            return
        _bkv_filter_orig(self, batch_indices)

    cache_module.BatchKVCache.filter = _bkv_filter  # type: ignore[method-assign]

    _bkv_extend_orig = cache_module.BatchKVCache.extend

    def _bkv_extend(self, other):  # type: ignore[no-untyped-def]
        if self.keys is None and other.keys is None:
            return
        _bkv_extend_orig(self, other)

    cache_module.BatchKVCache.extend = _bkv_extend  # type: ignore[method-assign]

    # BatchRotatingKVCache: same story.
    _brkv_extract_orig = cache_module.BatchRotatingKVCache.extract

    def _brkv_extract(self, idx):  # type: ignore[no-untyped-def]
        if self.keys is None:
            return cache_module.RotatingKVCache(self.max_size)
        return _brkv_extract_orig(self, idx)

    cache_module.BatchRotatingKVCache.extract = _brkv_extract  # type: ignore[method-assign]

    _brkv_filter_orig = cache_module.BatchRotatingKVCache.filter

    def _brkv_filter(self, batch_indices):  # type: ignore[no-untyped-def]
        if self.keys is None:
            return
        _brkv_filter_orig(self, batch_indices)

    cache_module.BatchRotatingKVCache.filter = _brkv_filter  # type: ignore[method-assign]

    _brkv_extend_orig = cache_module.BatchRotatingKVCache.extend

    def _brkv_extend(self, other):  # type: ignore[no-untyped-def]
        if self.keys is None and other.keys is None:
            return
        _brkv_extend_orig(self, other)

    cache_module.BatchRotatingKVCache.extend = _brkv_extend  # type: ignore[method-assign]


def apply_qwen35_attn_moe_split_patches(
    model: nn.Module, group: mx.distributed.Group
) -> nn.Module:
    """Install the split DecoderLayer.__call__ on a Qwen3.5 MoE model."""
    if group.size() != 2:
        raise ValueError(
            f"Qwen3.5 attn/moe split requires world_size==2, got {group.size()}"
        )

    inner = model
    for attr in ("model", "language_model"):
        if hasattr(inner, attr):
            inner = getattr(inner, attr)
    if hasattr(inner, "model"):
        inner = inner.model
    n_layers = len(inner.layers) if hasattr(inner, "layers") else 48

    # S == 1 serial split lives in DecoderLayer.__call__.
    DecoderLayer.__call__ = make_split_decoder_call(group, n_layers=n_layers)  # type: ignore[method-assign]

    # S > 1 pipelined split lives in Qwen3_5TextModel.__call__.
    Qwen3_5TextModel.__call__ = make_pipelined_model_call(group)  # type: ignore[method-assign]

    # Speculative verify path gets its own pipelined forward.
    try:
        from exo.worker.engines.mlx.speculative import mtp_module

        mtp_module.speculative_forward = make_pipelined_speculative_forward(group)
    except ImportError:
        pass  # speculative not used

    if group.rank() == MOE_RANK:
        _patch_caches_for_moe_rank()

    # Skip weight dropping under speculative — speculative_forward accesses
    # linear_attn/input_layernorm on both ranks in its post-loop.
    if os.environ.get("EXO_SPECULATIVE") != "1":
        _drop_unused_weights(model, group)

    logger.info(
        f"Qwen3.5 attn/moe split patch applied on rank {group.rank()}/{group.size()}"
    )
    return model


def _drop_unused_weights(model: nn.Module, group: mx.distributed.Group) -> None:
    """Free weights each rank doesn't need.

    ATTN_RANK keeps attention + input_layernorm, drops MLP + post_attention_layernorm.
    MOE_RANK keeps MLP + post_attention_layernorm, drops attention + input_layernorm.
    embed_tokens, norm, and lm_head stay on both ranks.
    """
    import gc

    inner = model
    for attr in ("model", "language_model"):
        if hasattr(inner, attr):
            inner = getattr(inner, attr)
    if hasattr(inner, "model"):
        inner = inner.model

    layers = inner.layers if hasattr(inner, "layers") else []

    for layer in layers:
        if group.rank() == ATTN_RANK:
            layer.mlp = None  # type: ignore[assignment]
            layer.post_attention_layernorm = None  # type: ignore[assignment]
        else:
            layer.self_attn = None  # type: ignore[assignment]
            layer.linear_attn = None  # type: ignore[assignment]
            layer.input_layernorm = None  # type: ignore[assignment]

    gc.collect()
    mx.clear_cache()
    logger.info(
        f"Rank {group.rank()}: dropped unused weights from {len(layers)} layers"
    )
