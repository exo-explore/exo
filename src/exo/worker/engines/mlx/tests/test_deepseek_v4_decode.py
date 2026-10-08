# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportPrivateUsage=false
"""DeepSeek V4 decode optimizations produce the same results as mlx_lm.

Uses a tiny randomly initialized DeepSeek V4 (8-bit dense weights, mxfp4
experts, compressed and uncompressed layers); no model download required.
"""

import copy
from functools import partial
from typing import cast

import mlx.core as mx
import numpy as np
import pytest
from mlx.nn.layers.distributed import shard_inplace
from mlx_lm.models import deepseek_v4 as dv4
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

from exo.worker.engines.mlx.auto_parallel import (
    _shard_v4_attention_groups,
)
from exo.worker.engines.mlx.patches import deepseek_v4_decode_kernels as kernels
from exo.worker.engines.mlx.patches import deepseek_v4_indexer as indexer_patch
from exo.worker.engines.mlx.patches import deepseek_v4_moe_gate as gate_patch
from exo.worker.engines.mlx.patches import switch_lhs_indices
from exo.worker.engines.mlx.tests.tiny_deepseek_v4 import (
    CONFIG,
    SLIDING_WINDOW,
    VOCAB_SIZE,
    tiny_model,
)

_PROMPT_LENGTH = 41
_DECODE_STEPS = 10


class _FakeGroup:
    def __init__(self, rank: int, size: int) -> None:
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _tokens(length: int, seed: int) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.integers(0, VOCAB_SIZE, (1, length)))


def _decode_logits(model: dv4.Model) -> mx.array:
    cache = model.make_cache()
    mx.eval(model(_tokens(_PROMPT_LENGTH, 1), cache=cache))
    steps = []
    for token in cast(list[int], _tokens(_DECODE_STEPS, 2).reshape(-1).tolist()):
        logits = model(mx.array([[token]]), cache=cache)
        mx.eval(logits)
        steps.append(logits.astype(mx.float32))
    return mx.concatenate(steps, axis=1)


def _relative_error(reference: mx.array, actual: mx.array) -> float:
    reference = reference.astype(mx.float32)
    actual = actual.astype(mx.float32)
    error = mx.sqrt(mx.sum(mx.square(reference - actual)))
    return (error / mx.sqrt(mx.sum(mx.square(reference)))).item()


def test_hc_post_is_correctly_rounded() -> None:
    mx.random.seed(1)
    f_out = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    residual = mx.random.normal((1, 1, 4, 1024)).astype(mx.bfloat16)
    post = mx.random.uniform(shape=(1, 1, 4))
    comb = mx.random.uniform(shape=(1, 1, 4, 4))
    f64, r64, p64, c64 = (
        np.array(a.astype(mx.float32)).astype(np.float64)
        for a in (f_out, residual, post, comb)
    )
    exact = p64[..., None] * f64[:, :, None, :] + np.einsum("bsij,bsid->bsjd", c64, r64)
    rounded = mx.array(exact.astype(np.float32)).astype(mx.bfloat16)
    fused = kernels.hc_post(f_out, residual, post, comb)
    original = kernels._original_hc_expand_ops(f_out, residual, post, comb)
    fused_misses = mx.sum(mx.not_equal(fused, rounded)).item()
    original_misses = mx.sum(mx.not_equal(original, rounded)).item()
    # Only double-rounding ties may differ; mlx_lm's fp32 matmul misses ~10%.
    assert fused_misses <= 4
    assert fused_misses < original_misses


def test_hc_mix_matches_norm_then_matmul() -> None:
    mx.random.seed(2)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    fn = mx.random.normal((24, 1024)) * 0.05
    reference = mx.fast.rms_norm(x.astype(mx.float32), None, 1e-6) @ fn.T
    assert mx.allclose(kernels.hc_mix(x, fn, 1e-6), reference, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("with_weight", [False, True])
def test_norm_rope_is_bit_exact(with_weight: bool) -> None:
    mx.random.seed(3)
    x = (mx.random.normal((8, 512)) * 3).astype(mx.bfloat16)
    weight = mx.random.normal((512,)).astype(mx.bfloat16) if with_weight else None
    freqs = mx.random.uniform(shape=(32,)) * 1000 + 1
    normed = mx.fast.rms_norm(x, weight, 1e-6)
    rotated = mx.fast.rope(
        normed[..., -64:][:, None, None, :],
        64,
        traditional=True,
        base=None,
        scale=1.0,
        offset=300,
        freqs=freqs,
    ).reshape(8, 64)
    reference = mx.concatenate([normed[..., :-64], rotated], axis=-1)
    fused = kernels.norm_rope(x, weight, 1e-6, [300.0], freqs, 64)
    assert mx.array_equal(fused, reference)


def test_inverse_rope_is_bit_exact() -> None:
    mx.random.seed(4)
    x = mx.random.normal((8, 512)).astype(mx.bfloat16)
    freqs = mx.random.uniform(shape=(32,)) * 1000 + 1
    rotated = mx.fast.rope(
        x[..., -64:][:, None, None, :],
        64,
        traditional=True,
        base=None,
        scale=-1.0,
        offset=77,
        freqs=freqs,
    ).reshape(8, 64)
    reference = mx.concatenate([x[..., :-64], rotated], axis=-1)
    assert mx.array_equal(
        kernels.norm_rope(x, None, None, [-77.0], freqs, 64), reference
    )


@pytest.mark.parametrize("keys", [1, 130, 640])
def test_decode_attention_matches_float32_sdpa(keys: int) -> None:
    mx.random.seed(5)
    q = mx.random.normal((1, 1, 16, 512)).astype(mx.bfloat16)
    kv = mx.random.normal((1, keys, 512)).astype(mx.bfloat16)
    sinks = mx.random.normal((16,)).astype(mx.bfloat16)
    reference = mx.fast.scaled_dot_product_attention(
        q.transpose(0, 2, 1, 3).astype(mx.float32),
        kv[:, None].astype(mx.float32),
        kv[:, None].astype(mx.float32),
        scale=512**-0.5,
        sinks=sinks.astype(mx.float32),
    ).transpose(0, 2, 1, 3)
    fused = kernels.decode_attention(
        q, kv, sinks, 512**-0.5, offset=keys - 1, window_len=keys, window=keys, ratio=0
    )
    assert _relative_error(reference, fused) < 4e-3


@pytest.mark.parametrize("sequence", [2, 3, 4])
@pytest.mark.parametrize("offset", [14, 17, 130])
def test_multi_query_decode_attention_masks_per_query(
    sequence: int, offset: int
) -> None:
    # Queries at offset .. offset + sequence - 1 over a 16-token sliding window
    # and ratio-4 compressed rows, each of which becomes visible once its
    # window of tokens is complete.
    window, ratio, heads, d = 16, 4, 8, 128
    window_len = min(window + sequence - 1, offset + sequence)
    compressed = (offset + sequence) // ratio
    mx.random.seed(9)
    q = mx.random.normal((1, sequence, heads, d)).astype(mx.bfloat16)
    kv = mx.random.normal((1, window_len + compressed, d)).astype(mx.bfloat16)
    sinks = mx.random.normal((heads,)).astype(mx.bfloat16)

    mask = np.full((sequence, window_len + compressed), -np.inf, dtype=np.float32)
    for query in range(sequence):
        # Window key j holds position offset + sequence - window_len + j.
        position = offset + query
        for j in range(window_len):
            key_position = offset + sequence - window_len + j
            if position - window < key_position <= position:
                mask[query, j] = 0.0
        mask[query, window_len : window_len + (position + 1) // ratio] = 0.0
    reference = mx.fast.scaled_dot_product_attention(
        q.transpose(0, 2, 1, 3).astype(mx.float32),
        kv[:, None].astype(mx.float32),
        kv[:, None].astype(mx.float32),
        scale=d**-0.5,
        mask=mx.array(mask),
        sinks=sinks.astype(mx.float32),
    ).transpose(0, 2, 1, 3)
    fused = kernels.decode_attention(
        q, kv, sinks, d**-0.5, offset, window_len, window, ratio
    )
    assert _relative_error(reference, fused) < 4e-3


def test_patched_decode_matches_mlx_lm(monkeypatch: pytest.MonkeyPatch) -> None:
    model = tiny_model()
    reference = _decode_logits(model)

    monkeypatch.setattr(dv4.HyperConnection, "hc_pre", kernels._patched_hc_pre)
    monkeypatch.setattr(dv4, "_hc_expand_ops", kernels._patched_hc_expand_ops)
    monkeypatch.setattr(dv4.V4Attention, "__call__", kernels._patched_attention)
    monkeypatch.setattr(dv4.MoEGate, "__call__", gate_patch._patched_call)
    monkeypatch.setattr(
        QuantizedSwitchLinear, "__call__", switch_lhs_indices._patched_call
    )
    patched = _decode_logits(model)

    assert _relative_error(reference, patched) < 2e-2
    agreement = mx.mean(mx.equal(reference.argmax(-1), patched.argmax(-1))).item()
    assert agreement >= 0.9


def _verify_logits(model: dv4.Model, prompt_length: int, sequence: int) -> mx.array:
    """A multi-token (speculative verification) step and the decode step after it."""
    cache = model.make_cache()
    tokens = _tokens(prompt_length + sequence + 1, prompt_length)
    mx.eval(model(tokens[:, :prompt_length], cache=cache))
    verify = model(tokens[:, prompt_length : prompt_length + sequence], cache=cache)
    after = model(tokens[:, prompt_length + sequence :], cache=cache)
    return mx.concatenate([verify, after], axis=1).astype(mx.float32)


@pytest.mark.parametrize("sequence", [2, 3])
@pytest.mark.parametrize("prompt_length", [14, 15, 16, 17, 31, 33])
def test_patched_verification_matches_mlx_lm(
    monkeypatch: pytest.MonkeyPatch, prompt_length: int, sequence: int
) -> None:
    model = tiny_model()
    reference = _verify_logits(model, prompt_length, sequence)

    monkeypatch.setattr(dv4.HyperConnection, "hc_pre", kernels._patched_hc_pre)
    monkeypatch.setattr(dv4, "_hc_expand_ops", kernels._patched_hc_expand_ops)
    monkeypatch.setattr(dv4.V4Attention, "__call__", kernels._patched_attention)
    monkeypatch.setattr(dv4.Indexer, "__call__", indexer_patch._patched_call)
    patched = _verify_logits(model, prompt_length, sequence)

    assert _relative_error(reference, patched) < 2e-2


@pytest.mark.parametrize("n_routed", [64, 256])
def test_parallel_gate_matches_mlx_lm(n_routed: int) -> None:
    mx.random.seed(6)
    args = dv4.ModelArgs.from_dict(
        {**CONFIG, "n_routed_experts": n_routed, "num_experts_per_tok": 6}
    )
    gate = dv4.MoEGate(args, layer_id=2)
    gate.weight = (mx.random.normal(gate.weight.shape) * 0.05).astype(mx.bfloat16)
    gate.e_score_correction_bias = mx.random.normal((n_routed,)) * 0.05
    for sequence in (1, 9):
        x = mx.random.normal((1, sequence, args.hidden_size)).astype(mx.bfloat16)
        reference_indices, reference_weights = gate_patch._original_call(gate, x)
        indices, weights = gate_patch._patched_call(gate, x)
        reference_order = mx.argsort(reference_indices, axis=-1)
        order = mx.argsort(indices, axis=-1)
        assert mx.array_equal(
            mx.take_along_axis(reference_indices, reference_order, -1),
            mx.take_along_axis(indices, order, -1),
        )
        assert mx.array_equal(
            mx.take_along_axis(reference_weights, reference_order, -1),
            mx.take_along_axis(weights, order, -1),
        )


@pytest.mark.parametrize(("index_topk", "exact"), [(64, False), (4, True)])
def test_indexer_shortcut_attends_to_the_same_keys(
    monkeypatch: pytest.MonkeyPatch, index_topk: int, exact: bool
) -> None:
    # index_topk 64 > pool rows: shortcut taken. index_topk 4 < pool rows: the
    # patched Indexer must score exactly like mlx_lm.
    model = tiny_model(index_topk=index_topk)
    attention = model.model.layers[1].attn
    assert attention.compress_ratio == 4
    mx.random.seed(7)
    prompt = (mx.random.normal((1, _PROMPT_LENGTH, 256)) * 0.5).astype(mx.bfloat16)
    steps = [
        (mx.random.normal((1, 1, 256)) * 0.5).astype(mx.bfloat16)
        for _ in range(_DECODE_STEPS)
    ]

    def run() -> mx.array:
        cache = dv4.DeepseekV4Cache(SLIDING_WINDOW)
        attention(prompt, cache=cache)
        return mx.concatenate([attention(step, cache=cache) for step in steps], 1)

    reference = run()
    monkeypatch.setattr(dv4.Indexer, "__call__", indexer_patch._patched_call)
    patched = run()
    if exact:
        assert mx.array_equal(reference, patched)
    else:
        assert _relative_error(reference, patched) < 1e-2


def test_cached_lhs_indices_match_default() -> None:
    mx.random.seed(8)
    layer = QuantizedSwitchLinear(
        256, 64, 32, bias=False, mode="mxfp4", group_size=32, bits=4
    )
    quantized = mx.quantize(
        mx.random.normal((32, 64, 256)) * 0.05, group_size=32, bits=4, mode="mxfp4"
    )
    layer.weight, layer.scales = quantized[0], quantized[1]
    x = mx.random.normal((1, 1, 1, 1, 256)).astype(mx.bfloat16)
    indices = mx.array([[[3, 17]]], dtype=mx.uint32)
    reference = QuantizedSwitchLinear.__call__(layer, x, indices)
    assert mx.array_equal(
        switch_lhs_indices._patched_call(layer, x, indices), reference
    )


@pytest.mark.parametrize("sequence", [1, 9])
def test_group_sharded_attention_sums_to_unsharded(
    monkeypatch: pytest.MonkeyPatch, sequence: int
) -> None:
    monkeypatch.setattr(mx.distributed, "all_sum", lambda x, group=None, stream=None: x)
    model = tiny_model()
    reference_attention = model.model.layers[0].attn
    mx.random.seed(9)
    x = (mx.random.normal((1, sequence, 256)) * 0.5).astype(mx.bfloat16)
    reference = reference_attention(x, cache=dv4.DeepseekV4Cache(16))

    world_size = 4
    total = mx.zeros(reference.shape, dtype=mx.float32)
    for rank in range(world_size):
        attention = copy.copy(reference_attention)
        for name in ("wq_b", "wo_a", "wo_b"):
            setattr(attention, name, copy.copy(getattr(reference_attention, name)))
        group = _FakeGroup(rank, world_size)
        _shard_v4_attention_groups(
            attention,
            group,  # pyright: ignore[reportArgumentType]
            partial(shard_inplace, sharding="all-to-sharded", group=group),  # pyright: ignore[reportArgumentType]
            partial(shard_inplace, sharding="sharded-to-all", group=group),  # pyright: ignore[reportArgumentType]
        )
        total = total + attention(x, cache=dv4.DeepseekV4Cache(16)).astype(mx.float32)

    assert _relative_error(reference, total) < 1e-2
