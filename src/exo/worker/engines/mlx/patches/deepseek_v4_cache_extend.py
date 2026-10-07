from mlx_lm.models.cache import BatchRotatingKVCache, RotatingKVCache
from mlx_lm.models.deepseek_v4 import DeepseekV4Cache

_original_extend = DeepseekV4Cache.extend


def _compact_local_buffer(local: RotatingKVCache) -> None:
    """Shrink the buffer to its temporally ordered contents so ``_idx`` equals its length."""
    if isinstance(local, BatchRotatingKVCache):
        return
    if local.keys is None or local.values is None:
        return
    local.keys = local._temporal_order(local.keys)
    local.values = local._temporal_order(local.values)
    local._idx = local.keys.shape[2]


def _patched_extend(self: DeepseekV4Cache, other: DeepseekV4Cache) -> None:
    """Make upstream's fast path safe for buffers grown ahead of the write index.

    Upstream concatenates the raw local buffers whenever ``offset`` and
    ``_idx`` match, but a RotatingKVCache grows its buffer ahead of ``_idx``
    during single-token updates, so equal positions do not imply equal buffer
    lengths. After compacting, equal ``_idx`` does imply equal lengths, and
    anything else falls through to upstream's BatchRotatingKVCache path.
    """
    _compact_local_buffer(self.local)
    _compact_local_buffer(other.local)
    _original_extend(self, other)


def patch_deepseek_v4_cache_extend() -> None:
    DeepseekV4Cache.extend = _patched_extend
