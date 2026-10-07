# pyright: reportPrivateUsage=false
"""Regression test for extending DeepseekV4Cache with mismatched buffer sizes.

A RotatingKVCache grows its buffer ahead of the write index during
single-token updates, so two caches can share ``offset`` and ``_idx`` while
holding buffers of different lengths. Extending one with the other used to
concatenate the raw buffers and crash with a shape mismatch.
"""

import mlx.core as mx
from mlx_lm.models.cache import RotatingKVCache
from mlx_lm.models.deepseek_v4 import DeepseekV4Cache

from exo.worker.engines.mlx.patches import apply_mlx_patches

SLIDING_WINDOW = 128
HEAD_DIM = 8


def _kv(length: int, seed: int) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    keys = mx.random.normal((1, 1, length, HEAD_DIM))
    values = mx.random.normal((1, 1, length, HEAD_DIM))
    return keys, values


def _logical_keys(cache: DeepseekV4Cache, row: int) -> mx.array:
    local = cache.extract(row).local
    assert isinstance(local, RotatingKVCache)
    assert local.keys is not None
    return local._temporal_order(local.keys)


def test_extend_with_different_buffer_sizes() -> None:
    apply_mlx_patches()

    # Prompt of 20 tokens, then one decode step: the buffer grows to the window.
    grown = DeepseekV4Cache(SLIDING_WINDOW)
    grown_keys, grown_values = _kv(20, seed=0)
    grown.update_and_fetch(grown_keys, grown_values)
    step_keys, step_values = _kv(1, seed=1)
    grown.update_and_fetch(step_keys, step_values)

    # Prompt of 21 tokens in one go: the buffer is exactly 21 long.
    exact = DeepseekV4Cache(SLIDING_WINDOW)
    exact_keys, exact_values = _kv(21, seed=2)
    exact.update_and_fetch(exact_keys, exact_values)

    assert grown.local.offset == exact.local.offset
    assert grown.local._idx == exact.local._idx
    assert grown.local.keys is not None and exact.local.keys is not None
    assert grown.local.keys.shape != exact.local.keys.shape

    expected_grown = mx.concatenate([grown_keys, step_keys], axis=2)
    grown.extend(exact)

    assert mx.array_equal(_logical_keys(grown, 0), expected_grown)
    assert mx.array_equal(_logical_keys(grown, 1), exact_keys)


def test_extend_wrapped_window_with_short_prompt() -> None:
    apply_mlx_patches()

    # 200-token prompt then two decode steps: the window has wrapped around.
    wrapped = DeepseekV4Cache(SLIDING_WINDOW)
    prompt_keys, prompt_values = _kv(200, seed=3)
    wrapped.update_and_fetch(prompt_keys, prompt_values)
    decoded_keys = [prompt_keys]
    for seed in (4, 5):
        step_keys, step_values = _kv(1, seed=seed)
        wrapped.update_and_fetch(step_keys, step_values)
        decoded_keys.append(step_keys)

    short = DeepseekV4Cache(SLIDING_WINDOW)
    short_keys, short_values = _kv(23, seed=6)
    short.update_and_fetch(short_keys, short_values)

    expected_wrapped = _logical_keys(wrapped, 0)
    all_wrapped_keys = mx.concatenate(decoded_keys, axis=2)
    assert mx.array_equal(
        expected_wrapped[..., -SLIDING_WINDOW:, :],
        all_wrapped_keys[..., -SLIDING_WINDOW:, :],
    )

    wrapped.extend(short)

    assert mx.array_equal(
        _logical_keys(wrapped, 0)[..., -SLIDING_WINDOW:, :],
        expected_wrapped[..., -SLIDING_WINDOW:, :],
    )
    assert mx.array_equal(_logical_keys(wrapped, 1)[..., -23:, :], short_keys)
