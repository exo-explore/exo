"""exo's MiniMax M3 model (minimax_m3.py) against mlx-lm's, on the same weights.

Everything but the sparse decode step must be bitwise mlx-lm's: prefill at every chunk
length (dense and sparse attention), decode below the block budget, batched decode, and
the parts on their own (norms, routing, index-key cache). The sparse decode step attends
to the same selected keys, summed in another order, so it is compared with a tolerance
and its block selection with mlx-lm's.
"""

import math
from typing import Any, cast

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map_with_path
from mlx_lm.models import minimax_m3_vl as upstream
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import BatchKVCache, CacheList, KVCache

from exo.worker.engines.mlx.models.minimax import minimax_m3

_TEXT_CONFIG: dict[str, Any] = {
    "hidden_size": 64,
    "intermediate_size": 32,
    "dense_intermediate_size": 128,
    "shared_intermediate_size": 64,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    # 32 lanes x 1: the decode kernels' smallest head size.
    "head_dim": 32,
    "rotary_dim": 16,
    "num_hidden_layers": 3,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "vocab_size": 128,
    "moe_layer_freq": [0, 1, 1],
    "sparse_attention_config": {
        "sparse_index_dim": 32,
        "sparse_num_index_heads": 2,
        "sparse_topk_blocks": 2,
        "sparse_block_size": 4,
        "sparse_local_block": 1,
        "sparse_attention_freq": [0, 1, 1],
    },
}


def _models(dtype: mx.Dtype) -> tuple[upstream.Model, minimax_m3.Model]:
    """mlx-lm's model and exo's, with the same random weights (non-zero norms and
    routing biases, which mlx-lm initialises to zero)."""
    mx.random.seed(0)
    args = upstream.ModelArgs(text_config=dict(_TEXT_CONFIG))
    reference = upstream.Model(args)
    params = cast(dict[str, Any], reference.parameters())
    for layer in params["language_model"]["model"]["layers"]:  # pyright: ignore[reportAny]
        for norm in ("input_layernorm", "post_attention_layernorm"):
            layer[norm]["weight"] = 0.1 * mx.random.normal((64,))
        for norm in ("q_norm", "k_norm", "index_q_norm", "index_k_norm"):
            if norm in layer["self_attn"]:
                layer["self_attn"][norm]["weight"] = 0.1 * mx.random.normal((32,))
        if "block_sparse_moe" in layer:
            layer["block_sparse_moe"]["e_score_correction_bias"] = 0.1 * (
                mx.random.normal((8,))
            )
    reference.update(params)
    keep_dtype = reference.cast_predicate  # the router stays float32, as when loading

    def to_dtype(path: str, value: mx.array) -> mx.array:
        return value.astype(dtype) if keep_dtype(path) else value

    reference.update(
        cast(dict[str, Any], tree_map_with_path(to_dtype, reference.parameters()))
    )
    fast = minimax_m3.Model(upstream.ModelArgs(text_config=dict(_TEXT_CONFIG)))
    fast.update(cast(dict[str, Any], reference.parameters()))
    mx.eval(reference.parameters(), fast.parameters())
    return reference, fast


def _tokens(n: int, seed: int = 1) -> mx.array:
    return mx.random.randint(0, 128, (1, n), key=mx.random.key(seed))


def _prefill(
    model: upstream.Model, tokens: mx.array, chunk: int
) -> tuple[mx.array, list[Any]]:
    cache = model.make_cache()
    outs: list[mx.array] = []
    for start in range(0, tokens.shape[1], chunk):
        outs.append(model(tokens[:, start : start + chunk], cache=cache))
    logits = mx.concatenate(outs, axis=1)
    mx.eval(logits)
    return logits, cache


def _same(a: mx.array, b: mx.array) -> bool:
    return bool(mx.array_equal(a, b).item())


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("chunk", [2, 3, 4, 5, 7, 8, 9, 13, 16, 31, 64])
def test_prefill_is_bitwise_mlx_lms_at_every_chunk_length(
    dtype: mx.Dtype, chunk: int
) -> None:
    """Dense up to 8 keys, sparse block selection from 9 (4-key blocks, 2 kept). 62
    tokens, so no chunk is a single token (the decode step, tested below)."""
    reference, fast = _models(dtype)
    tokens = _tokens(62)
    expected, _ = _prefill(reference, tokens, chunk)
    got, _ = _prefill(fast, tokens, chunk)
    assert _same(got, expected)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_decode_is_bitwise_below_the_block_budget_and_close_above_it(
    dtype: mx.Dtype,
) -> None:
    """Above the budget each step attends in another order, so in bf16 the caches part
    after the first sparse step; float32 keeps them within its rounding for 30 steps."""
    reference, fast = _models(dtype)
    tokens = _tokens(5)
    _, ref_cache = _prefill(reference, tokens, 5)
    _, fast_cache = _prefill(fast, tokens, 5)
    step_ids = mx.array([[3]])
    steps = 30 if dtype == mx.float32 else 4
    for step in range(steps):
        expected = reference(step_ids, cache=ref_cache)
        got = fast(step_ids, cache=fast_cache)
        mx.eval(expected, got)
        keys = 5 + step + 1
        if keys <= 8:
            assert _same(got, expected), step
        else:
            np.testing.assert_allclose(
                np.array(got.astype(mx.float32)),
                np.array(expected.astype(mx.float32)),
                rtol=0,
                atol=1e-4 if dtype == mx.float32 else 0.1,
            )
        step_ids = mx.argmax(expected[:, -1], axis=-1, keepdims=True)


def _attention_inputs(
    layer: upstream.Attention, length: int
) -> tuple[mx.array, CacheList]:
    """A layer's attention after a prefill of `length` tokens, and its cache."""
    x = mx.random.normal((1, length, 64), key=mx.random.key(2))
    cache = CacheList(KVCache(), KVCache())
    causal = mx.less_equal(mx.arange(length)[None], mx.arange(length)[:, None])
    mx.eval(layer(x, causal, cache))
    return mx.random.normal((1, 1, 64), key=mx.random.key(3)), cache


@pytest.mark.parametrize("length", [8, 9, 11, 12, 13, 40, 41])
def test_sparse_decode_attends_to_mlx_lms_selected_keys(length: int) -> None:
    """The gathered attention against mlx-lm's dense attention with its block mask, in
    float32: the same keys, so the same result up to the order of the sums."""
    reference, fast = _models(mx.float32)
    ref_attn = reference.layers[1].self_attn
    fast_attn = fast.layers[1].self_attn
    x, ref_cache = _attention_inputs(ref_attn, length)
    _, fast_cache = _attention_inputs(fast_attn, length)
    expected = ref_attn(x, None, ref_cache)
    got = fast_attn(x, None, fast_cache)
    mx.eval(expected, got)
    np.testing.assert_allclose(np.array(got), np.array(expected), rtol=1e-5, atol=1e-6)


def _batched_decode(model: upstream.Model, lengths: list[int]) -> mx.array:
    """A left-padded batch (as exo's batch generator merges requests): prefill, then
    three decode steps. Returns the logits of the steps."""
    prompts = [_tokens(n, seed=10 + i) for i, n in enumerate(lengths)]
    longest = max(lengths)
    padding = [longest - n for n in lengths]
    batch = mx.concatenate(
        [
            mx.pad(p, [(0, 0), (pad, 0)])
            for p, pad in zip(prompts, padding, strict=True)
        ],
        axis=0,
    )
    caches = [
        CacheList(BatchKVCache(padding), BatchKVCache(padding))
        if layer.self_attn.is_sparse_attn
        else CacheList(BatchKVCache(padding))
        for layer in model.layers
    ]
    model(batch, cache=caches)
    steps: list[mx.array] = []
    step_ids = mx.array([[5]] * len(lengths))
    for _ in range(3):
        logits = model(step_ids, cache=caches)
        steps.append(logits)
        step_ids = mx.argmax(logits[:, -1], axis=-1, keepdims=True)
    out = mx.concatenate(steps, axis=1)
    mx.eval(out)
    return out


def test_batched_sparse_decode_masks_the_padding_as_mlx_lm_does() -> None:
    """Padding keys are never chosen nor attended to; each sequence selects its own
    blocks (over the batch's cache slots, as mlx-lm does)."""
    reference, fast = _models(mx.float32)
    lengths = [23, 14, 9]
    expected = _batched_decode(reference, lengths)
    got = _batched_decode(fast, lengths)
    np.testing.assert_allclose(np.array(got), np.array(expected), rtol=1e-4, atol=1e-5)


_TOKEN_COUNTS = [*range(1, 34), 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513]


@pytest.mark.parametrize("tokens", _TOKEN_COUNTS)
def test_routing_and_combine_are_bitwise_mlx_lms(tokens: int) -> None:
    """MiniMax M3's router and expert outputs: 128 experts, 4 per token, float32 scores,
    bf16 activations."""
    gates = mx.random.normal((tokens, 128), key=mx.random.key(tokens)) * 3
    bias = mx.random.normal((128,), key=mx.random.key(1000)) * 0.1
    scores = mx.sigmoid(gates)
    expected_inds = mx.argpartition(-(scores + bias), kth=3, axis=-1)[..., :4]
    expected = mx.take_along_axis(scores, expected_inds, axis=-1)
    expected = expected / (mx.sum(expected, axis=-1, keepdims=True) + 1e-20)
    expected = (expected * 2.0).astype(mx.bfloat16)
    for route in (minimax_m3._route, minimax_m3._route_ops):  # pyright: ignore[reportPrivateUsage]
        inds, weights = route(gates, bias, 4, 2.0, mx.bfloat16)
        assert _same(inds, expected_inds.astype(inds.dtype))
        assert _same(weights, expected)

    y = mx.random.normal((tokens, 4, 256), key=mx.random.key(tokens + 1)).astype(
        mx.bfloat16
    )
    shared = mx.random.normal((tokens, 256), key=mx.random.key(tokens + 2)).astype(
        mx.bfloat16
    )
    combined_expected = (y * expected[..., None]).sum(axis=-2) + shared
    parts: list[mx.array] = []
    for k in range(4):
        parts += [y[:, k], expected[:, k : k + 1]]
    assert _same(minimax_m3._combine(shared, *parts), combined_expected)  # pyright: ignore[reportPrivateUsage]


def test_routing_ties_go_to_the_lower_expert_as_in_mlxs_sort() -> None:
    """Equal scores (here: identical gates and no bias) at the top-4 boundary: mlx's
    argpartition sorts stably, so the lower expert wins; the routing kernel too."""
    gates = mx.zeros((5, 128))
    gates = mx.where(mx.arange(128) % 7 == 3, 1.5, gates)  # 18 experts tie at the top
    gates[1] = mx.arange(128).astype(mx.float32) * 0  # every expert ties
    bias = mx.zeros((128,))
    choice = -(mx.sigmoid(gates) + bias)
    expected = mx.argpartition(choice, kth=3, axis=-1)[..., :4]
    inds, _ = minimax_m3._route(gates, bias, 4, 2.0, mx.bfloat16)  # pyright: ignore[reportPrivateUsage]
    assert _same(inds, expected.astype(inds.dtype))


def test_gemma_norm_is_bitwise_mlx_lms() -> None:
    reference = upstream.GemmaRMSNorm(64)
    fast = minimax_m3.GemmaRMSNorm(64)
    reference.weight = (0.1 * mx.random.normal((64,))).astype(mx.bfloat16)
    fast.weight = reference.weight
    x = mx.random.normal((3, 5, 64)).astype(mx.bfloat16)
    for _ in range(2):
        assert _same(fast(x), reference(x))


def _upstream_sparse_decode(
    q: mx.array,
    keys: mx.array,
    values: mx.array,
    index_q: mx.array,
    index_keys: mx.array,
    mask: mx.array | None,
    topk: int,
) -> tuple[mx.array, mx.array]:
    """mlx-lm's sparse decode step: float32 scores of every key, its block mask over
    every head and key, dense attention. Returns the output and the kept blocks."""
    batch, q_heads, _, head_dim = q.shape
    heads = keys.shape[1]
    n_keys = keys.shape[2]
    scores = index_q @ index_keys.astype(mx.float32).swapaxes(-1, -2)
    if mask is not None:
        scores = mx.where(mask, scores, -mx.inf)
    full, tail = divmod(n_keys, 128)
    block_scores = (
        scores[..., : full * 128].reshape(batch, heads, 1, full, 128).max(axis=-1)
    )
    if tail:
        block_scores = mx.concatenate(
            [block_scores, scores[..., full * 128 :].max(axis=-1, keepdims=True)], -1
        )
    n_blocks = block_scores.shape[-1]
    block_scores = mx.where(
        mx.arange(n_blocks) == (n_keys - 1) // 128, mx.inf, block_scores
    )
    inds = mx.argpartition(-block_scores, kth=topk - 1, axis=-1)[..., :topk]
    keep_blocks = mx.put_along_axis(
        mx.zeros(block_scores.shape, dtype=mx.bool_), inds, mx.array(True), axis=-1
    )
    keep = mx.repeat(keep_blocks, 128, axis=-1)[..., :n_keys]
    if mask is not None:
        keep = keep & mask
    keep = mx.repeat(keep, q_heads // heads, axis=1)
    out = scaled_dot_product_attention(
        q, keys, values, cache=None, scale=1 / math.sqrt(head_dim), mask=keep
    )
    return out, keep_blocks


_SCALE = 1 / math.sqrt(128)


def _decode_inputs(
    batch: int, q_heads: int, heads: int, n_keys: int, capacity: int, seed: int
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    k1, k2, k3, k4, k5 = mx.random.split(mx.random.key(seed), 5)
    q = mx.random.normal((batch, q_heads, 1, 128), key=k1).astype(mx.bfloat16)
    keys = mx.random.normal((batch, heads, capacity, 128), key=k2).astype(mx.bfloat16)
    values = mx.random.normal((batch, heads, capacity, 128), key=k3).astype(mx.bfloat16)
    index_q = mx.random.normal((batch, heads, 1, 128), key=k4).astype(mx.bfloat16)
    index_keys = mx.random.normal((batch, 1, capacity, 128), key=k5).astype(mx.bfloat16)
    return q, keys, values, index_q.astype(mx.float32), index_keys


@pytest.mark.parametrize(
    ("batch", "q_heads", "heads", "n_keys"),
    [
        (1, 64, 4, 2049),  # first step past the block budget
        (1, 64, 4, 2176),  # a whole number of blocks
        (1, 64, 4, 16_000),
        (1, 32, 2, 16_000),  # one rank of 2x tensor parallelism
        (1, 16, 1, 70_001),  # one rank of 4x
        (3, 64, 4, 5_000),
    ],
)
def test_the_decode_kernels_select_and_attend_as_mlx_lm(
    batch: int, q_heads: int, heads: int, n_keys: int
) -> None:
    capacity = n_keys + 200  # the cache buffers are larger than their contents
    q, keys, values, index_q, index_keys = _decode_inputs(
        batch, q_heads, heads, n_keys, capacity, n_keys
    )
    mask = None
    if batch > 1:
        # left padding of 0, 700 and 2900 slots
        starts = mx.array([0, 700, 2900])[:, None, None, None]
        mask = mx.greater_equal(mx.arange(n_keys)[None, None, None, :], starts)
    expected, _ = _upstream_sparse_decode(
        q,
        keys[..., :n_keys, :],
        values[..., :n_keys, :],
        index_q,
        index_keys[..., :n_keys, :],
        mask,
        16,
    )
    got = minimax_m3._sparse_decode_kernels(  # pyright: ignore[reportPrivateUsage]
        q, index_q, keys, values, index_keys, mask, n_keys, 128, 16, 1, _SCALE
    )
    mx.eval(expected, got)
    np.testing.assert_allclose(
        np.array(got.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        rtol=0,
        atol=2e-2,  # one bf16 rounding step at these magnitudes
    )


def test_block_selection_breaks_ties_as_mlx_does() -> None:
    """Equal block scores (e.g. a repeated haystack): mlx's argpartition sorts, so the
    lower block wins a tie at the boundary; so do the kernels. Here 30 blocks hold the
    same best key and 15 of them (+ the query's own block) are kept."""
    n_keys = 128 * 40
    q, keys, values, index_q, index_keys = _decode_inputs(1, 64, 4, n_keys, n_keys, 7)
    best = mx.ones((128,), dtype=mx.bfloat16)
    index_keys = mx.where(
        mx.logical_and(
            mx.equal(mx.arange(n_keys) % 128, 5), mx.less(mx.arange(n_keys), 128 * 30)
        )[None, None, :, None],
        best,
        index_keys * 0.01,
    )
    index_q = mx.abs(index_q)
    expected, _ = _upstream_sparse_decode(
        q, keys, values, index_q, index_keys, None, 16
    )
    got = minimax_m3._sparse_decode_kernels(  # pyright: ignore[reportPrivateUsage]
        q, index_q, keys, values, index_keys, None, n_keys, 128, 16, 1, _SCALE
    )
    np.testing.assert_allclose(
        np.array(got.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        rtol=0,
        atol=2e-2,
    )
