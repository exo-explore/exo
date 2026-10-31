"""Inkling's fused Metal paths against the op-by-op paths they replace: the short
convolution, the relative-position bias (with a mask passed in, and with the causal and
window masks worked out from the key distances), the decoding path of the shared experts,
and the whole model with and without the kernels."""

from typing import cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.base import (
    create_causal_mask,  # pyright: ignore[reportUnknownVariableType]
)
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from exo.worker.engines.mlx.models.inkling import model as inkling
from exo.worker.engines.mlx.models.inkling.model import (
    Attention,
    ShortConvolution,
    TextArgs,
)
from exo.worker.engines.mlx.models.inkling.tests.test_inkling_model import (
    _QUANTIZABLE_CONFIG,  # pyright: ignore[reportPrivateUsage]
    _SEQUENCE,  # pyright: ignore[reportPrivateUsage]
    _TEXT_CONFIG,  # pyright: ignore[reportPrivateUsage]
    ARGS,
    tiny_model,
)

_GLOBAL_CONFIG: dict[str, object] = dict(_TEXT_CONFIG)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the fused paths need Metal"
)


def _conv(channels: int, kernel: int = 4) -> ShortConvolution:
    conv = ShortConvolution(channels, kernel)
    conv.weight = mx.random.normal((channels, kernel, 1))
    return conv


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("length", [1, 2, 5, 33])
@pytest.mark.parametrize("with_state", [False, True])
@pytest.mark.parametrize("with_skip", [False, True])
def test_the_fused_short_convolution_matches_the_op_path(
    dtype: mx.Dtype, length: int, with_state: bool, with_skip: bool
) -> None:
    mx.random.seed(length)
    conv = _conv(48)
    x = mx.random.normal((2, length, 48)).astype(dtype)
    skip = mx.random.normal((2, length, 48)).astype(dtype) if with_skip else None
    caches = [ArraysCache(4), ArraysCache(4)]
    if with_state:
        state = mx.random.normal((2, 3, 48))
        for cache in caches:
            cache[1] = state
    fused = conv._fused(x, caches[0], 1, skip)  # pyright: ignore[reportPrivateUsage]
    by_ops = conv._by_ops(x, caches[1], 1, None)  # pyright: ignore[reportPrivateUsage]
    if skip is not None:
        by_ops = skip + by_ops
    assert fused.dtype == dtype
    np.testing.assert_array_equal(
        np.array(fused.astype(mx.float32)), np.array(by_ops.astype(mx.float32))
    )
    np.testing.assert_array_equal(
        np.array(cast(mx.array, caches[0][1])), np.array(cast(mx.array, caches[1][1]))
    )


def test_the_fused_short_convolution_without_a_cache_matches() -> None:
    conv = _conv(16)
    x = mx.random.normal((1, 9, 16))
    fused = conv._fused(x, None, 0)  # pyright: ignore[reportPrivateUsage]
    by_ops = conv._by_ops(x, None, 0, None)  # pyright: ignore[reportPrivateUsage]
    np.testing.assert_array_equal(np.array(fused), np.array(by_ops))


def _attention(sliding: bool) -> Attention:
    args: TextArgs = ARGS.text
    layer = 0 if sliding else 3
    attention = Attention(args, layer)
    attention.rel_proj = mx.random.normal(attention.rel_proj.shape)
    return attention


def _bias_pair(
    attention: Attention,
    cache: KVCache | RotatingKVCache,
    queries: int,
    mask: mx.array | None,
    blocks: list[tuple[int, int, int, int]],
) -> list[tuple[mx.array, mx.array]]:
    """For each block, the kernel's bias (given ``mask``) and the op path's bias (given
    the explicit causal mask of the cache)."""
    heads = attention.num_heads
    width = 8
    keys = mx.random.normal((1, attention.num_kv_heads, queries, width))
    returned, _ = cast(
        tuple[mx.array, mx.array],
        cache.update_and_fetch(keys, keys),  # pyright: ignore[reportUnknownMemberType]
    )
    num_keys = returned.shape[2]
    profiles = mx.random.normal((1, queries, heads, attention.rel_extent))
    offset = cache.offset - queries
    explicit = (
        None
        if queries == 1
        else cast(
            mx.array,
            create_causal_mask(
                queries,
                min(offset, num_keys - queries),
                window_size=attention.window,
            ),
        )
    )
    fused = inkling._RelativeBias(  # pyright: ignore[reportPrivateUsage]
        attention, profiles, mask, cache, queries, num_keys
    )
    by_ops = inkling._RelativeBias(  # pyright: ignore[reportPrivateUsage]
        attention, profiles, explicit, cache, queries, num_keys
    )
    by_ops.fused = False
    out: list[tuple[mx.array, mx.array]] = []
    for q0, q1, k0, k1 in blocks or [(0, queries, 0, num_keys)]:
        out.append((fused.block(q0, q1, k0, k1), by_ops.block(q0, q1, k0, k1)))
    return out


@pytest.mark.parametrize("sliding", [True, False])
def test_the_bias_kernel_works_out_the_causal_and_window_masks(sliding: bool) -> None:
    attention = _attention(sliding)
    cache: KVCache | RotatingKVCache = (
        RotatingKVCache(max_size=6) if sliding else KVCache()
    )
    # a prefill, a second prefill chunk, then decode steps past the window and extent
    for queries, blocks in [
        (9, [(0, 4, 0, 9), (4, 9, 0, 9)]),
        (7, [(0, 3, 0, 8), (3, 7, 2, 12)] if sliding else []),
        *[(1, [])] * 12,
    ]:
        for fused, by_ops in _bias_pair(attention, cache, queries, None, blocks):
            np.testing.assert_array_equal(np.array(fused), np.array(by_ops))


def test_the_bias_kernel_applies_a_mask_passed_in() -> None:
    attention = _attention(sliding=False)
    cache = KVCache()
    queries = 6
    mask = mx.random.uniform(shape=(2, 1, queries, queries)) > 0.4
    profiles = mx.random.normal((2, queries, attention.num_heads, attention.rel_extent))
    keys = mx.zeros((2, attention.num_kv_heads, queries, 8))
    cache.update_and_fetch(keys, keys)  # pyright: ignore[reportUnknownMemberType]
    fused = inkling._RelativeBias(  # pyright: ignore[reportPrivateUsage]
        attention, profiles, mask, cache, queries, queries
    )
    by_ops = inkling._RelativeBias(  # pyright: ignore[reportPrivateUsage]
        attention, profiles, mask, cache, queries, queries
    )
    by_ops.fused = False
    for block in [(0, queries, 0, queries), (2, 5, 1, 6)]:
        np.testing.assert_array_equal(
            np.array(fused.block(*block)), np.array(by_ops.block(*block))
        )


def _decode_logits(model: inkling.Model, steps: int) -> np.ndarray:
    cache = model.make_cache()
    rows = [np.array(model(mx.array(_SEQUENCE[:17])[None], cache=cache)[0, -1])]
    for token in _SEQUENCE[17 : 17 + steps]:
        rows.append(np.array(model(mx.array([[token]]), cache=cache)[0, -1]))
    return np.stack(rows)


def test_the_model_gives_the_same_logits_without_the_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fused kernels and derived masks against the op-by-op paths with built masks."""
    model = tiny_model()
    with_kernels = _decode_logits(model, 15)
    monkeypatch.setattr(inkling, "_metal_kernels_available", lambda: False)
    without = _decode_logits(model, 15)
    np.testing.assert_allclose(with_kernels, without, rtol=1e-5, atol=1e-5)


def test_the_shared_experts_decode_path_gives_the_sorted_paths_bits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decoding gathers quantized shared experts directly (unquantized ones keep the
    sorted layout); both take MLX's quantized matrix-vector kernel, so the bits match."""
    args = inkling.ModelArgs(
        model_type="inkling_mm_model", text_config=_QUANTIZABLE_CONFIG
    )
    mx.random.seed(5)
    model = inkling.Model(args)
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: (
            hasattr(module, "to_quantized") and ".mlp." in path
        ),
    )
    mx.eval(model.parameters())
    moe = model.layers[1].mlp
    assert isinstance(moe, inkling.MoE)
    assert moe.shared_experts.few(mx.zeros((1, 1, 64)))
    with_decode_path = _decode_logits(model, 10)
    monkeypatch.setattr(inkling, "_SORTED_SHARED_ROWS", 0)
    sorted_only = _decode_logits(model, 10)
    np.testing.assert_array_equal(with_decode_path, sorted_only)
    unquantized = tiny_model().layers[1].mlp
    assert isinstance(unquantized, inkling.MoE)
    assert not unquantized.shared_experts.few(mx.zeros((1, 1, 32)))


def test_the_compiled_decoding_steps_compute_what_the_plain_ones_do() -> None:
    mx.random.seed(7)
    x = mx.random.normal((3, 1, 64)).astype(mx.bfloat16)
    weight = (0.2 * mx.random.normal((258, 64))).astype(mx.bfloat16)
    correction = 0.1 * mx.random.normal((256,))
    scale = mx.array([1.3])
    plain = inkling._route(x, weight, correction, scale, 256, 6, 2.5)  # pyright: ignore[reportPrivateUsage]
    compiled = inkling._route_compiled(x, weight, correction, scale, 256, 6, 2.5)  # pyright: ignore[reportPrivateUsage]
    for a, b in zip(plain, compiled, strict=True):
        np.testing.assert_array_equal(np.array(a), np.array(b))
    routed = mx.random.normal((3, 1, 6, 64)).astype(mx.bfloat16)
    shared = mx.random.normal((3, 1, 64))
    np.testing.assert_array_equal(
        np.array(inkling._combine_experts(routed, plain[1], shared).astype(mx.float32)),  # pyright: ignore[reportPrivateUsage]
        np.array(
            inkling._combine_experts_compiled(routed, plain[1], shared).astype(  # pyright: ignore[reportPrivateUsage]
                mx.float32
            )
        ),
    )
    gate, up = (
        mx.random.normal((3, 1, 2, 1, 32)).astype(mx.bfloat16) for _ in range(2)
    )
    gammas = mx.random.uniform(shape=(3, 1, 2, 1, 1))
    np.testing.assert_array_equal(
        np.array(inkling._gate_activations(gate, up, gammas).astype(mx.float32)),  # pyright: ignore[reportPrivateUsage]
        np.array(
            inkling._gate_activations_compiled(gate, up, gammas).astype(mx.float32)  # pyright: ignore[reportPrivateUsage]
        ),
    )


@pytest.mark.parametrize(
    ("routed", "shared", "top_k", "tokens"),
    [(256, 2, 6, 4096), (8, 2, 2, 300), (64, 1, 4, 17), (256, 2, 6, 1)],
)
def test_the_router_kernel_gives_the_same_bits_as_the_ops(
    routed: int, shared: int, top_k: int, tokens: int
) -> None:
    mx.random.seed(routed + tokens)
    x = mx.random.normal((1, tokens, 64)).astype(mx.bfloat16)
    weight = (0.4 * mx.random.normal((routed + shared, 64))).astype(mx.bfloat16)
    correction = 0.05 * mx.random.normal((routed,))
    global_scale = mx.array([0.9])
    expected = inkling._route(  # pyright: ignore[reportPrivateUsage]
        x, weight, correction, global_scale, routed, top_k, 2.5
    )
    got = inkling._route_by_kernel(  # pyright: ignore[reportPrivateUsage]
        x @ weight.T, correction, global_scale, routed, top_k, 2.5
    )
    for a, b in zip(expected, got, strict=True):
        assert a.shape == b.shape
        np.testing.assert_array_equal(np.array(a), np.array(b))


@pytest.mark.parametrize("tokens", [1, 3, 8])
def test_the_combine_kernel_gives_the_same_bits_as_the_ops(tokens: int) -> None:
    mx.random.seed(tokens)
    routed = mx.random.normal((tokens, 1, 6, 256)).astype(mx.bfloat16)
    weights = mx.softmax(mx.random.normal((tokens, 1, 6)), axis=-1) * 2.5
    shared = mx.random.normal((tokens, 1, 2, 256)).astype(mx.bfloat16)
    expected = inkling._combine_experts(  # pyright: ignore[reportPrivateUsage]
        routed, weights, shared.astype(mx.float32).sum(axis=-2)
    )
    got = inkling._combine_by_kernel(routed, weights, shared)  # pyright: ignore[reportPrivateUsage]
    summed = inkling._combine_by_kernel(  # pyright: ignore[reportPrivateUsage]
        routed, weights, shared.astype(mx.float32).sum(axis=-2), summed=True
    )
    for result in (got, summed):
        np.testing.assert_array_equal(
            np.array(expected.astype(mx.float32)), np.array(result.astype(mx.float32))
        )


@pytest.mark.parametrize(
    ("length", "num_keys"),
    [
        (1025, 1025),
        (513, 1113),
        (4096, 16384),
        (4096, 65536),
        (100, 100),
        (4000, 300000),
    ],
)
def test_global_attention_spans_keep_each_querys_kernel(
    length: int, num_keys: int
) -> None:
    """Global layers regroup long-prefill blocks into bigger calls, but keep a trailing
    block of at most 8 queries alone (MLX attends that few queries with its vector
    kernel), and every call starts at the first key and stops at its last query's key."""
    args = TextArgs.from_text_config(
        {**_GLOBAL_CONFIG, "num_attention_heads": 32, "swa_num_attention_heads": 32}
    )
    attention = Attention(args, 3)
    spans = attention._spans(length, num_keys)  # pyright: ignore[reportPrivateUsage]
    first = num_keys - length
    old_block = max(1, inkling._BIAS_ELEMENTS_PER_CALL // (32 * num_keys))  # pyright: ignore[reportPrivateUsage]
    old = [(s, min(s + old_block, length)) for s in range(0, length, old_block)]
    if length <= old_block:
        old = [(0, length)]
    assert [q for a, b, _, _ in spans for q in range(a, b)] == list(range(length))
    if len(spans) > 1:
        assert all(k0 == 0 and k1 == first + b for _, b, k0, k1 in spans)
    old_small = {(a, b) for a, b in old if b - a <= 8}
    new_small = {(a, b) for a, b, _, _ in spans if b - a <= 8}
    assert old_small == new_small
