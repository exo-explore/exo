"""Decoding doesn't build up an ever-growing graph behind cache bookkeeping that the model
doesn't read: linear attention layers' lengths, sparse-attention index caches' offsets."""

import io
from types import SimpleNamespace
from typing import cast

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, BatchKVCache, CacheList

from exo.worker.engines.mlx.patches.opt_batch_gen import (
    _patched_step,  # pyright: ignore[reportPrivateUsage]
)

VOCAB = 8


class LinearAttentionModel:
    """Stands in for a model whose linear attention layers each move their cache on by the
    tokens they process, while only one of them is read to build the mask."""

    def __call__(self, inputs: mx.array, cache: list[ArraysCache]) -> mx.array:
        for layer_cache in cache:
            layer_cache.advance(inputs.shape[1])
        return mx.zeros((inputs.shape[0], inputs.shape[1], VOCAB))


def greedy(logprobs: mx.array) -> mx.array:
    return mx.argmax(logprobs, axis=-1)


def graph_edges(array: mx.array) -> int:
    dot = io.StringIO()
    mx.export_to_dot(dot, array)  # pyright: ignore[reportUnknownMemberType]
    return dot.getvalue().count("->")


def test_linear_attention_caches_do_not_grow_a_graph_while_decoding() -> None:
    caches = [ArraysCache(2, left_padding=[0, 3]) for _ in range(4)]
    for layer_cache in caches:
        layer_cache.prepare(lengths=[5, 5])
    batch = SimpleNamespace(
        model=LinearAttentionModel(),
        prompt_cache=caches,
        # mlx-lm gives every sequence a sampler and a (possibly empty) processor list
        logits_processors=[(), ()],
        samplers=[greedy, greedy],
        fallback_sampler=greedy,
        uids=[0, 1],
        tokens=[[], []],
        _next_tokens=mx.array([1, 2]),
        _next_logprobs=mx.zeros((2, VOCAB)),
    )

    for _ in range(256):
        _patched_step(batch)  # pyright: ignore[reportArgumentType]

    for layer_cache in caches:
        assert layer_cache.left_padding is not None
        assert layer_cache.lengths is not None
        # Each step's update was evaluated, so nothing chains back to earlier steps
        assert graph_edges(layer_cache.left_padding) == 0
        assert graph_edges(layer_cache.lengths) == 0
        assert layer_cache.left_padding.tolist() == [-256, -253]
        assert layer_cache.lengths.tolist() == [-251, -251]


class IndexedAttentionModel:
    """Stands in for MiniMax M3: each layer caches its attention keys and, alongside, the
    keys of a sparse-attention indexer. Both caches move on at every step, but only the
    attention cache's offsets are read (to position the keys); the index cache's are not."""

    def __call__(self, inputs: mx.array, cache: list[CacheList]) -> mx.array:
        batch, length = inputs.shape
        logits = mx.zeros((batch, length, VOCAB))
        for layer_cache in cache:
            attention = cast(BatchKVCache, layer_cache[0])
            index = cast(BatchKVCache, layer_cache[1])
            positions = attention.offset
            new_keys = mx.ones((batch, 1, length, 2))
            keys, _ = attention.update_and_fetch(new_keys, new_keys)
            index_keys, _ = index.update_and_fetch(new_keys, new_keys)
            used = keys.sum(axis=(1, 2, 3)) + index_keys.sum(axis=(1, 2, 3))
            logits = logits + (used + positions)[:, None, None] * 0
        return logits


def test_unread_index_cache_offsets_do_not_grow_a_graph_while_decoding() -> None:
    sub_caches = [BatchKVCache([0, 3]) for _ in range(8)]
    for sub_cache in sub_caches:
        sub_cache.update_and_fetch(mx.ones((2, 1, 5, 2)), mx.ones((2, 1, 5, 2)))
    batch = SimpleNamespace(
        model=IndexedAttentionModel(),
        prompt_cache=[CacheList(*sub_caches[i : i + 2]) for i in range(0, 8, 2)],
        logits_processors=[(), ()],
        samplers=[greedy, greedy],
        fallback_sampler=greedy,
        uids=[0, 1],
        tokens=[[], []],
        _next_tokens=mx.array([1, 2]),
        _next_logprobs=mx.zeros((2, VOCAB)),
    )

    for _ in range(256):
        _patched_step(batch)  # pyright: ignore[reportArgumentType]

    # Each step's offsets were evaluated, so nothing chains back to earlier steps
    assert [graph_edges(sub_cache.offset) for sub_cache in sub_caches] == [0] * 8
    for sub_cache in sub_caches:
        assert sub_cache.offset.tolist() == [261, 258]
