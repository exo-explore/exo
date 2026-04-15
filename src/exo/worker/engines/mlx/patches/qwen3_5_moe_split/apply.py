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

import mlx.core as mx
import mlx.nn as nn
from loguru import logger
from mlx_lm.models import cache as cache_module
from mlx_lm.models.qwen3_5 import DecoderLayer

from .decoder import MOE_RANK, make_split_decoder_call

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


def apply_qwen35_attn_moe_split_patches(
    model: nn.Module, group: mx.distributed.Group
) -> nn.Module:
    """Install the split DecoderLayer.__call__ on a Qwen3.5 MoE model."""
    if group.size() != 2:
        raise ValueError(
            f"Qwen3.5 attn/moe split requires world_size==2, got {group.size()}"
        )

    DecoderLayer.__call__ = make_split_decoder_call(group)  # type: ignore[method-assign]

    if group.rank() == MOE_RANK:
        _patch_caches_for_moe_rank()

    logger.info(
        f"Qwen3.5 attn/moe split patch applied on rank {group.rank()}/{group.size()}"
    )
    return model
