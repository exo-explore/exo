# pyright: reportUnknownMemberType=false
from tinygrad.tensor import Tensor


def test_cache_initial_seq_len():
    """Fresh cache should have seq_len == 0."""
    from exo.worker.engines.tinygrad.cache import KVCache

    cache = KVCache(num_layers=4)
    assert cache.seq_len == 0

def test_cache_update_first_call():
    """First update should set the cached tensors."""
    from exo.worker.engines.tinygrad.cache import KVCache

    cache = KVCache(num_layers=4)
    k = Tensor.randn(1, 8, 3, 64)  # (batch, kv_heads, seq=3, head_dim)
    v = Tensor.randn(1, 8, 3, 64)
    k_out, _v_out = cache.update(0, k, v)
    assert k_out.shape == (1, 8, 3, 64)
    assert cache.seq_len == 3

def test_cache_update_concatenates():
    """Subsequent updates should concatenate along dim=2 (sequence)."""
    from exo.worker.engines.tinygrad.cache import KVCache

    cache = KVCache(num_layers=4)
    k1 = Tensor.randn(1, 8, 3, 64)
    v1 = Tensor.randn(1, 8, 3, 64)
    cache.update(0, k1, v1)

    k2 = Tensor.randn(1, 8, 1, 64)  # decode step: seq=1
    v2 = Tensor.randn(1, 8, 1, 64)
    k_out, _v_out = cache.update(0, k2, v2)
    assert k_out.shape == (1, 8, 4, 64)  # 3 + 1
    assert cache.seq_len == 4

def test_cache_layers_are_independent():
    """Updates to layer 0 should not affect layer 1."""
    from exo.worker.engines.tinygrad.cache import KVCache

    cache = KVCache(num_layers=4)
    k = Tensor.randn(1, 8, 3, 64)
    v = Tensor.randn(1, 8, 3, 64)
    cache.update(0, k, v)
    assert cache._keys[1] is None  # layer 1 untouched  # pyright: ignore[reportPrivateUsage]
