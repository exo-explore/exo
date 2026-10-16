# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportPrivateUsage=false
"""Sliding-window caches updated with ``update_and_fetch(kv, kv)`` keep their
values as the keys, so no unevaluated chain of value writes builds up."""

import io
from collections.abc import Callable

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import BatchRotatingKVCache, RotatingKVCache

from exo.worker.engines.mlx.patches import shared_window_values


def _graph_lines(array: mx.array) -> int:
    dot = io.StringIO()
    mx.export_to_dot(dot, array)
    return dot.getvalue().count("\n")


def _set_update(monkeypatch: pytest.MonkeyPatch, patched: bool) -> None:
    # Other tests may have applied every exo patch already.
    monkeypatch.setattr(
        RotatingKVCache,
        "update_and_fetch",
        shared_window_values._rotating_update
        if patched
        else shared_window_values._original_rotating_update,
    )
    monkeypatch.setattr(
        BatchRotatingKVCache,
        "update_and_fetch",
        shared_window_values._batch_rotating_update
        if patched
        else shared_window_values._original_batch_rotating_update,
    )


def _updates(
    cache: RotatingKVCache | BatchRotatingKVCache, steps: int, check_values: bool
) -> list[mx.array]:
    """A prefill chunk then single-token decode steps, past the window size."""
    rng = np.random.default_rng(9)
    fetched: list[mx.array] = []
    for length in [5, 3, *([1] * steps)]:
        kv = mx.array(rng.standard_normal((2, 1, length, 8)), dtype=mx.float16)
        keys, values = cache.update_and_fetch(kv, kv)
        mx.eval(keys)
        if check_values:
            # Evaluating the values also collapses their pending writes.
            assert mx.array_equal(keys, values).item()
        fetched.append(keys)
    return fetched


_CACHES: list[Callable[[], RotatingKVCache | BatchRotatingKVCache]] = [
    lambda: RotatingKVCache(max_size=16),
    lambda: BatchRotatingKVCache(16, [2, 0]),
]


@pytest.mark.parametrize("make_cache", _CACHES)
def test_values_do_not_chain_over_a_long_decode(
    monkeypatch: pytest.MonkeyPatch,
    make_cache: Callable[[], RotatingKVCache | BatchRotatingKVCache],
) -> None:
    _set_update(monkeypatch, patched=False)
    unpatched = make_cache()
    _updates(unpatched, 300, check_values=False)
    assert unpatched.values is not None
    assert _graph_lines(unpatched.values) > 1000

    _set_update(monkeypatch, patched=True)
    patched = make_cache()
    _updates(patched, 300, check_values=False)
    assert patched.values is patched.keys


@pytest.mark.parametrize("make_cache", _CACHES)
def test_fetched_keys_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    make_cache: Callable[[], RotatingKVCache | BatchRotatingKVCache],
) -> None:
    _set_update(monkeypatch, patched=False)
    expected = _updates(make_cache(), 60, check_values=True)
    _set_update(monkeypatch, patched=True)
    got = _updates(make_cache(), 60, check_values=True)
    for a, b in zip(expected, got, strict=True):
        assert mx.array_equal(a, b).item()


def test_distinct_keys_and_values_are_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_update(monkeypatch, patched=True)
    cache = RotatingKVCache(max_size=16)
    keys = mx.ones((1, 1, 1, 8))
    values = mx.zeros((1, 1, 1, 8))
    _, fetched_values = cache.update_and_fetch(keys, values)
    assert cache.values is not cache.keys
    assert mx.array_equal(fetched_values, values).item()
