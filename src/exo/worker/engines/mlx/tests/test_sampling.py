# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
"""exo's sampler and logits processors give mlx-lm's results exactly."""

import random
from collections.abc import Callable

import mlx.core as mx
import pytest
from mlx_lm.models.cache import BatchKVCache
from mlx_lm.sample_utils import apply_top_k, apply_top_p
from mlx_lm.sample_utils import make_logits_processors as mlx_make_logits_processors
from mlx_lm.sample_utils import make_sampler as mlx_make_sampler

from exo.worker.engines.mlx.patches.opt_batch_gen import apply_batch_gen_patch
from exo.worker.engines.mlx.radix_sort import can_sort_rows, sort_rows
from exo.worker.engines.mlx.sampling import (
    RecentTokensProcessor,
    can_select_top_k,
    history_window,
    make_logits_processors,
    make_sampler,
    recent_tokens,
    select_top_k,
    sorted_top_p,
)

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal kernels")


def bf16_logprobs(rows: int, vocab: int, seed: int, scale: float = 3.0) -> mx.array:
    """Log-probs normalised in float32 from bfloat16 logits, as exo's batch step makes them:
    many tokens share a value."""
    logits = mx.random.normal((rows, vocab), key=mx.random.key(seed)) * scale
    logits = logits.astype(mx.bfloat16).astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def special_rows(vocab: int) -> list[mx.array]:
    rows: list[mx.array] = []
    # a handful of values: every cut falls inside a group of ties, across chunks
    few = mx.random.randint(0, 5, (2, vocab), key=mx.random.key(1)).astype(mx.float32)
    rows.append(few * -1.25)
    # all equal
    rows.append(mx.zeros((1, vocab)))
    # fewer finite tokens than k (as after a tight top-p)
    sparse = mx.full((1, vocab), -float("inf"))
    sparse[0, mx.array([3, vocab // 2, vocab - 1])] = mx.array([-0.5, -2.0, -2.0])
    rows.append(sparse)
    # -0.0 ties 0.0
    signed = mx.full((1, vocab), -30.0)
    signed[0, 9] = -0.0
    signed[0, 4] = 0.0
    signed[0, vocab - 2] = -0.0
    rows.append(signed)
    # NaN ranks last
    with_nan = bf16_logprobs(1, vocab, 3)
    with_nan[0, 11] = float("nan")
    rows.append(with_nan)
    return rows


def bits(x: mx.array) -> mx.array:
    return x.view(mx.uint32)


@metal
@pytest.mark.parametrize("vocab", [33, 4096, 4097, 151936, 248320])
@pytest.mark.parametrize("top_k", [1, 2, 20, 64, 1000])
def test_select_top_k_is_apply_top_k_bit_for_bit(vocab: int, top_k: int) -> None:
    if top_k >= vocab:
        pytest.skip("top-k must be below the vocabulary size")
    inputs = [bf16_logprobs(rows, vocab, seed) for seed, rows in enumerate([1, 3, 1])]
    inputs.append(mx.random.normal((2, vocab), key=mx.random.key(9)) * 5.0)
    inputs.extend(special_rows(vocab))
    for x in inputs:
        assert can_select_top_k(x, top_k)
        expected = apply_top_k(x, top_k)
        assert mx.array_equal(bits(select_top_k(x, top_k)), bits(expected)).item()


@metal
@pytest.mark.parametrize("vocab", [1, 33, 4095, 4096, 4097, 151936, 248320])
def test_sort_rows_sorts_like_mx_sort(vocab: int) -> None:
    inputs = [bf16_logprobs(rows, vocab, seed) for seed, rows in enumerate([1, 3])]
    inputs.append(mx.random.normal((2, vocab), key=mx.random.key(5)) * 1e30)
    inputs.extend(special_rows(vocab) if vocab > 12 else [])
    for x in inputs:
        assert can_sort_rows(x)
        # equal values: -0.0 and 0.0 may swap places between themselves, NaN keeps its place
        assert mx.array_equal(sort_rows(x), mx.sort(x, axis=-1), equal_nan=True).item()


@metal
@pytest.mark.parametrize("vocab", [4097, 151936, 248320])
@pytest.mark.parametrize("top_p", [0.1, 0.8, 0.9, 0.95, 0.999])
def test_sorted_top_p_is_apply_top_p_bit_for_bit(vocab: int, top_p: float) -> None:
    inputs = [
        bf16_logprobs(rows, vocab, seed, 1.0 + seed)
        for seed, rows in enumerate([1, 3, 1])
    ]
    inputs.append(mx.random.normal((2, vocab), key=mx.random.key(9)) * 5.0)
    inputs.extend(special_rows(vocab))
    for x in inputs:
        expected = apply_top_p(x, top_p)
        assert mx.array_equal(bits(sorted_top_p(x, top_p)), bits(expected)).item()


def test_select_top_k_only_where_the_kernels_apply() -> None:
    x = bf16_logprobs(1, 1000, 0)
    assert not can_select_top_k(x.astype(mx.bfloat16), 20)
    assert not can_select_top_k(x[0], 20)
    assert not can_select_top_k(x, 0)
    assert not can_select_top_k(x, 1000)
    assert not can_select_top_k(x, 1025)


SAMPLER_SETTINGS: list[dict[str, float | int]] = [
    {"temp": 1.0, "top_k": 20},
    {"temp": 0.7, "top_k": 20},
    {"temp": 1.0, "top_p": 0.95, "top_k": 20},
    {"temp": 0.6, "top_p": 0.8, "top_k": 40},
    {"temp": 1.0, "top_p": 0.95, "min_p": 0.05, "top_k": 64},
    {"temp": 1.0, "top_k": 2000},
    {"temp": 0.6, "top_p": 0.9, "min_p": 0.05},
    {"temp": 1.0},
    {"temp": 0.0, "top_k": 20},
]


def _settings(s: dict[str, float | int]) -> tuple[float, float, float, int]:
    return (
        float(s.get("temp", 0.0)),
        float(s.get("top_p", 0.0)),
        float(s.get("min_p", 0.0)),
        int(s.get("top_k", 0)),
    )


@pytest.mark.parametrize("settings", SAMPLER_SETTINGS)
def test_sampler_draws_mlx_lm_tokens_and_leaves_the_same_random_state(
    settings: dict[str, float | int],
) -> None:
    temp, top_p, min_p, top_k = _settings(settings)
    reference = mlx_make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k)
    sampler = make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k)
    inputs = [
        bf16_logprobs(1, 151936, seed, scale) for seed, scale in enumerate([1, 3, 8])
    ]
    inputs.append(bf16_logprobs(4, 248320, 7))
    for seed, x in enumerate(inputs * 3):
        mx.random.seed(seed)
        expected = reference(x)
        mx.eval(expected)
        expected_state = mx.array(mx.random.state[0])
        mx.random.seed(seed)
        drawn = sampler(x)
        mx.eval(drawn)
        assert drawn.dtype == expected.dtype
        assert mx.array_equal(drawn, expected).item()
        assert mx.array_equal(mx.random.state[0], expected_state).item()


def _random_logits(seed: int) -> mx.array:
    return mx.random.normal((1, 50), key=mx.random.key(seed)).astype(mx.bfloat16)


def _processor_outputs(
    processors: list[Callable[[mx.array, mx.array], mx.array]],
    history: list[int],
    logits: mx.array,
) -> mx.array:
    tokens = mx.array(recent_tokens(history, history_window(processors)))
    for processor in processors:
        logits = processor(tokens, logits)
    return logits


@pytest.mark.parametrize(
    "penalties",
    [
        (1.0, 20, None, None),
        (1.0, 20, 1.5, None),
        (1.1, 64, 0.5, 0.25),
        (None, 20, 0.0, 0.3),
        (0.9, 0, None, None),
        (1.0, 20, 0.0, 0.0),
    ],
)
def test_logits_processors_match_mlx_lm(
    penalties: tuple[float | None, int, float | None, float | None],
) -> None:
    repetition, context, presence, frequency = penalties
    reference = mlx_make_logits_processors(
        repetition_penalty=repetition,
        repetition_context_size=context,
        presence_penalty=presence,
        frequency_penalty=frequency,
    )
    processors = make_logits_processors(
        repetition_penalty=repetition,
        repetition_context_size=context,
        presence_penalty=presence,
        frequency_penalty=frequency,
    )
    assert len(processors) == len(reference) - (1 if repetition == 1.0 else 0)
    for length in [0, 1, 7, 300]:
        draw = random.Random(length)
        history = [draw.randrange(50) for _ in range(length)]
        # mlx-lm's processors update the logits they are given in place: one copy each
        expected = _random_logits(length)
        for processor in reference:
            expected = processor(mx.array(history), expected)
        got = _processor_outputs(processors, history, _random_logits(length))
        assert mx.array_equal(got.view(mx.uint16), expected.view(mx.uint16)).item()


def test_history_window() -> None:
    def passthrough(_tokens: mx.array, logits: mx.array) -> mx.array:
        return logits

    assert history_window([]) == 0
    assert history_window([RecentTokensProcessor(passthrough, 0)]) == 0
    assert (
        history_window(
            [
                RecentTokensProcessor(passthrough, 20),
                RecentTokensProcessor(passthrough, 5),
            ]
        )
        == 20
    )
    assert history_window([RecentTokensProcessor(passthrough, 20), passthrough]) is None
    assert recent_tokens([1, 2, 3], None) == [1, 2, 3]
    assert recent_tokens([1, 2, 3], 5) == [1, 2, 3]
    assert recent_tokens([1, 2, 3], 2) == [2, 3]
    assert recent_tokens([1, 2, 3], 0) == []
    # a context size of 0 reads the whole history in mlx-lm: no window
    (unbounded,) = make_logits_processors(1.2, 0, None, None)
    assert not isinstance(unbounded, RecentTokensProcessor)


def _batch_cache(left_padding: list[int], length: int) -> BatchKVCache:
    cache = BatchKVCache(left_padding)
    keys = mx.random.normal(
        (len(left_padding), 2, length, 8), key=mx.random.key(length)
    )
    cache.update_and_fetch(keys, keys)
    return cache


def test_one_token_steps_skip_an_all_true_mask() -> None:
    apply_batch_gen_patch()
    unpadded = _batch_cache([0, 0], 10)
    assert unpadded.make_mask(1) is None
    assert isinstance(unpadded.make_mask(1, return_array=True), mx.array)
    assert isinstance(unpadded.make_mask(1, window_size=4), mx.array)
    assert isinstance(unpadded.make_mask(3), mx.array)
    padded = _batch_cache([0, 2], 10)
    mask = padded.make_mask(1)
    assert isinstance(mask, mx.array)
    assert not mask.all().item()
    # once the padding is gone (the padded sequence left), the mask goes too
    padded.filter([0])
    assert padded.make_mask(1) is None


@metal
def test_attention_without_the_all_true_mask_is_identical() -> None:
    queries = mx.random.normal((2, 4, 1, 64), key=mx.random.key(0)).astype(mx.bfloat16)
    keys = mx.random.normal((2, 2, 700, 64), key=mx.random.key(1)).astype(mx.bfloat16)
    values = mx.random.normal((2, 2, 700, 64), key=mx.random.key(2)).astype(mx.bfloat16)
    mask = mx.ones((2, 1, 1, 700), dtype=mx.bool_)
    with_mask = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=0.125, mask=mask
    )
    without = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=0.125, mask=None
    )
    assert mx.array_equal(with_mask.view(mx.uint16), without.view(mx.uint16)).item()
