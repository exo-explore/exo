# pyright: reportUnknownMemberType=false
import math

from tinygrad.tensor import Tensor


def test_cache_initial_seq_len():
    """Fresh cache should have seq_len equal to max_seq_len (pre-allocated)."""
    from exo.worker.engines.tinygrad.cache import KVCache

    cache = KVCache(num_layers=4, num_kv_heads=8, head_dim=64, max_seq_len=128)
    assert cache.seq_len == 128


def test_cache_update_writes_to_position():
    """Update should write keys/values at the specified position."""
    from exo.worker.engines.tinygrad.cache import KVCache

    cache = KVCache(num_layers=4, num_kv_heads=8, head_dim=64, max_seq_len=128)
    k = Tensor.ones(1, 8, 3, 64)
    v = Tensor.ones(1, 8, 3, 64)
    k_out, _v_out = cache.update(0, k, v, position=0)

    # Output shape is always the full pre-allocated buffer
    assert k_out.shape == (1, 8, 128, 64)
    # The written positions should be non-zero
    assert float(k_out[0, 0, 0, 0].item()) != 0.0
    assert float(k_out[0, 0, 2, 0].item()) != 0.0
    # Positions beyond the write should remain zero
    assert float(k_out[0, 0, 3, 0].item()) == 0.0


def test_cache_update_tensor_position():
    """Update with Tensor position should write a single token."""
    from exo.worker.engines.tinygrad.cache import KVCache

    cache = KVCache(num_layers=4, num_kv_heads=8, head_dim=64, max_seq_len=128)
    k = Tensor.ones(1, 8, 1, 64) * 5.0
    v = Tensor.ones(1, 8, 1, 64) * 5.0
    pos = Tensor([10]).reshape(1, 1, 1, 1)
    k_out, _v_out = cache.update(0, k, v, position=pos)

    # Position 10 should have the value we wrote
    assert math.isclose(float(k_out[0, 0, 10, 0].item()), 5.0, rel_tol=1e-2)
    # Other positions should remain zero
    assert float(k_out[0, 0, 0, 0].item()) == 0.0


def test_cache_layers_are_independent():
    """Updates to layer 0 should not affect layer 1."""
    from exo.worker.engines.tinygrad.cache import KVCache

    cache = KVCache(num_layers=4, num_kv_heads=8, head_dim=64, max_seq_len=128)
    k = Tensor.ones(1, 8, 3, 64)
    v = Tensor.ones(1, 8, 3, 64)
    cache.update(0, k, v, position=0)
    # Layer 1 should still be all zeros (pre-allocated but untouched)
    assert float(cache._keys[1].sum().item()) == 0.0  # pyright: ignore[reportPrivateUsage]
