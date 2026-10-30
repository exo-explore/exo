"""GLM-5.x attention fast paths against mlx-lm's own GlmMoeDsaAttention, and the kernels
they use against plain MLX: dense and sparse prefill, single-token and batched decode."""

from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm.generate import _merge_caches  # pyright: ignore[reportPrivateUsage]
from mlx_lm.models.cache import CacheList, KVCache
from mlx_lm.models.glm_moe_dsa import Model as MlxLmModel

from exo.worker.engines.mlx.models.glm.glm_moe_dsa import (
    INDEXER_KEY_NORM_EPS,
    GlmAttention,
    Model,
    ModelArgs,
)
from exo.worker.engines.mlx.models.glm.kernels import (
    gather_rows,
    indexer_scores,
    metal_kernels_available,
)
from exo.worker.engines.mlx.models.glm.tests.test_glm_moe_dsa import CONFIG

needs_metal = pytest.mark.skipif(
    not metal_kernels_available(), reason="needs a Metal GPU as the default device"
)


def _models() -> tuple[nn.Module, nn.Module]:
    """mlx-lm's model and exo's, float32, with the same weights and indexer epsilon."""
    args = cast(ModelArgs, ModelArgs.from_dict(CONFIG))  # pyright: ignore[reportUnknownMemberType]
    mx.random.seed(7)
    upstream = MlxLmModel(args)
    for layer in cast(list[nn.Module], upstream.model.layers):  # pyright: ignore[reportUnknownMemberType]
        indexer = getattr(cast(nn.Module, layer.self_attn), "indexer", None)
        if indexer is not None:
            cast(
                nn.LayerNorm, cast(nn.Module, indexer).k_norm
            ).eps = INDEXER_KEY_NORM_EPS
    ours = Model(args)
    weights = cast(
        list[tuple[str, mx.array]],
        tree_flatten(upstream.parameters()),  # pyright: ignore[reportUnknownMemberType]
    )
    ours.load_weights(weights)  # pyright: ignore[reportUnknownMemberType]
    mx.eval([w for _, w in weights])
    return upstream, ours


def _attentions(model: nn.Module) -> list[GlmAttention]:
    layers = cast(list[nn.Module], model.model.layers)  # pyright: ignore[reportUnknownMemberType]
    return [cast(GlmAttention, layer.self_attn) for layer in layers]


def _run(
    model: nn.Module, prompts: list[list[int]], steps: list[list[int]], chunk: int
) -> list[mx.array]:
    """Prefill each prompt in chunks, merge the caches into one batch, decode steps."""
    outputs: list[mx.array] = []
    caches: list[list[Any]] = []
    for prompt in prompts:
        cache = cast(list[Any], model.make_cache())  # pyright: ignore[reportUnknownMemberType]
        for start in range(0, len(prompt), chunk):
            piece = mx.array(prompt[start : start + chunk])[None]
            outputs.append(model(piece, cache=cache))
        caches.append(cache)
    batch_cache = _merge_caches(caches) if len(caches) > 1 else caches[0]
    for step in steps:
        outputs.append(model(mx.array(step)[:, None], cache=batch_cache))
    return outputs


def _assert_close(expected: list[mx.array], got: list[mx.array]) -> None:
    assert len(expected) == len(got)
    for a, b in zip(expected, got, strict=True):
        assert a.shape == b.shape
        assert mx.allclose(a, b, atol=2e-4, rtol=2e-4).item(), (
            mx.abs(a - b).max().item()
        )


VOCAB_SIZE = cast(int, CONFIG["vocab_size"])


def _tokens(n: int, seed: int) -> list[int]:
    return cast(
        list[int],
        mx.random.randint(0, VOCAB_SIZE, (n,), key=mx.random.key(seed)).tolist(),
    )


@needs_metal
def test_dense_prefill_and_decode_match_mlx_lm() -> None:
    upstream, ours = _models()
    # 30 prompt tokens in chunks of 13: below and past index_topk (8), then 3 decode steps
    # that each run the indexer over more than 8 keys.
    prompt = _tokens(30, 1)
    steps = [[t] for t in _tokens(3, 2)]
    _assert_close(_run(upstream, [prompt], steps, 13), _run(ours, [prompt], steps, 13))


@needs_metal
def test_sparse_prefill_matches_mlx_lm() -> None:
    upstream, ours = _models()
    for attention in _attentions(upstream) + _attentions(ours):
        attention.sparse_prefill_min_context = 16
    prompt = _tokens(40, 3)
    steps = [[t] for t in _tokens(2, 4)]
    _assert_close(_run(upstream, [prompt], steps, 12), _run(ours, [prompt], steps, 12))


@needs_metal
def test_batched_decode_with_left_padding_matches_mlx_lm() -> None:
    upstream, ours = _models()
    prompts = [_tokens(25, 5), _tokens(11, 6), _tokens(4, 7)]
    steps = [
        cast(
            list[int],
            mx.random.randint(0, 64, (3,), key=mx.random.key(10 + i)).tolist(),
        )
        for i in range(4)
    ]
    _assert_close(_run(upstream, prompts, steps, 32), _run(ours, prompts, steps, 32))


def test_falls_back_to_mlx_lm_off_metal() -> None:
    upstream, ours = _models()
    prompt = _tokens(20, 8)
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        assert not metal_kernels_available()
        expected = _run(upstream, [prompt], [[3]], 20)
        got = _run(ours, [prompt], [[3]], 20)
    finally:
        mx.set_default_device(previous)
    for a, b in zip(expected, got, strict=True):
        assert mx.array_equal(a, b).item()


def test_router_weight_is_stored_in_float32() -> None:
    args = cast(ModelArgs, ModelArgs.from_dict(CONFIG))  # pyright: ignore[reportUnknownMemberType]
    model = Model(args)
    weights = {
        "model.layers.3.mlp.gate.weight": mx.zeros((4, 32), mx.bfloat16),
        "model.layers.3.mlp.gate.e_score_correction_bias": mx.zeros((4,), mx.float32),
        "model.layers.3.self_attn.o_proj.weight": mx.zeros((32, 32), mx.bfloat16),
    }
    out = model.sanitize(weights)
    assert out["model.layers.3.mlp.gate.weight"].dtype == mx.float32
    assert out["model.layers.3.self_attn.o_proj.weight"].dtype == mx.bfloat16


@needs_metal
@pytest.mark.parametrize("masked", [False, True])
def test_indexer_scores_sum_heads_in_float32(masked: bool) -> None:
    key = mx.random.key(0)
    k1, k2, k3 = mx.random.split(key, 3)
    scores = mx.random.normal((2, 4, 5, 37), key=k1).astype(mx.bfloat16)
    weights = mx.random.normal((2, 4, 5, 1), key=k2).astype(mx.bfloat16)
    mask = mx.random.uniform(shape=(5, 37), key=k3) > 0.3 if masked else None
    got = indexer_scores(scores, weights, mask)
    expected = (
        mx.maximum(scores.astype(mx.float32), 0) * weights.astype(mx.float32)
    ).sum(axis=1, keepdims=True)
    if mask is not None:
        expected = mx.where(mask, expected, -mx.inf)
    assert got.dtype == mx.float32
    assert got.shape == (2, 1, 5, 37)
    finite = mx.isfinite(expected)
    assert mx.array_equal(finite, mx.isfinite(got)).item()
    assert mx.allclose(
        mx.where(finite, got, 0), mx.where(finite, expected, 0), atol=1e-5
    ).item()


@needs_metal
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_gather_rows_matches_take(dtype: mx.Dtype) -> None:
    source = mx.random.normal((50, 72)).astype(dtype)
    indices = mx.random.randint(0, 50, (3, 9)).astype(mx.uint32)
    assert mx.array_equal(gather_rows(source, indices), source[indices]).item()


def test_decode_cache_is_unchanged_in_layout() -> None:
    # The fast paths read and write the same caches as mlx-lm's model.
    _, ours = _models()
    caches = cast(list[CacheList], ours.make_cache())  # pyright: ignore[reportUnknownMemberType]
    assert all(isinstance(cache[0], KVCache) for cache in caches)
