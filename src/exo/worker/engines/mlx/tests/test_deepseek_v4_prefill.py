# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportPrivateUsage=false
"""DeepSeek V4 prefill optimizations produce the same results as mlx_lm.

Uses a tiny randomly initialized DeepSeek V4 (8-bit dense weights, mxfp4
experts, compressed and uncompressed layers); no model download required.
"""

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models import deepseek_v4 as dv4

from exo.worker.engines.mlx.patches import deepseek_v4_decode_kernels as kernels
from exo.worker.engines.mlx.patches import deepseek_v4_indexer as indexer_patch
from exo.worker.engines.mlx.patches import deepseek_v4_prefill_attention as prefill
from exo.worker.engines.mlx.tests.tiny_deepseek_v4 import VOCAB_SIZE, tiny_model


def _relative_error(reference: mx.array, actual: mx.array) -> float:
    reference = reference.astype(mx.float32)
    actual = actual.astype(mx.float32)
    error = mx.sqrt(mx.sum(mx.square(reference - actual)))
    return (error / mx.sqrt(mx.sum(mx.square(reference)))).item()


def _masked_sdpa(
    q: mx.array,
    window_kv: mx.array,
    compressed: mx.array | None,
    topk: mx.array | None,
    sinks: mx.array,
    scale: float,
    offset: int,
    window: int,
    ratio: int,
) -> mx.array:
    """mlx_lm's prefill attention (masks over the whole chunk) in float32."""
    batch, _, sequence, _ = q.shape
    mask = dv4._build_window_mask(batch, sequence, offset, window, window_kv.shape[1])
    kv = window_kv
    if compressed is not None:
        pool = compressed.shape[1]
        visible = dv4._compressed_visibility(batch, sequence, offset, pool, ratio)
        if topk is not None:
            selected = mx.put_along_axis(
                mx.zeros((batch, sequence, pool), dtype=mx.bool_),
                topk,
                mx.array(True),
                axis=-1,
            )
            visible = visible & selected[:, None]
        mask = mx.concatenate([mask, visible], axis=-1)
        kv = mx.concatenate([window_kv, compressed], axis=1)
    kv = kv[:, None].astype(mx.float32)
    return mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32),
        kv,
        kv,
        scale=scale,
        mask=mask,
        sinks=sinks.astype(mx.float32),
    )


@pytest.mark.parametrize(
    ("heads", "head_dim", "sequence", "offset", "ratio", "pool", "topk"),
    [
        (16, 512, 70, 0, 0, 0, 0),
        (16, 512, 70, 300, 128, 2, 0),
        (16, 512, 70, 300, 4, 92, 24),
        (32, 128, 50, 0, 4, 12, 8),
        (8, 128, 40, 17, 4, 14, 14),
    ],
)
def test_prefill_attention_matches_masked_sdpa(
    heads: int,
    head_dim: int,
    sequence: int,
    offset: int,
    ratio: int,
    pool: int,
    topk: int,
) -> None:
    window = 16
    window_len = sequence + min(offset, window - 1)
    mx.random.seed(10)
    q = mx.random.normal((2, heads, sequence, head_dim)).astype(mx.bfloat16)
    window_kv = mx.random.normal((2, window_len, head_dim)).astype(mx.bfloat16)
    compressed = (
        mx.random.normal((2, pool, head_dim)).astype(mx.bfloat16) if pool else None
    )
    # Top-k over the whole pool, as mlx_lm selects it: rows completed later in
    # the chunk are selected too and must stay masked.
    selection = (
        mx.argpartition(-mx.random.normal((2, sequence, pool)), kth=topk - 1, axis=-1)[
            ..., :topk
        ].astype(mx.int32)
        if topk
        else None
    )
    sinks = mx.random.normal((heads,)).astype(mx.bfloat16)
    args = (q, window_kv, compressed, selection, sinks, head_dim**-0.5, offset)
    reference = _masked_sdpa(*args, window, ratio)
    fused = prefill.prefill_attention(*args, window, ratio)
    assert _relative_error(reference, fused) < 4e-3


def test_relu_weighted_sum_is_correctly_rounded() -> None:
    mx.random.seed(11)
    scores = mx.random.normal((1, 8, 64, 300)).astype(mx.bfloat16)
    weights = (mx.random.normal((1, 8, 64)) * 0.1).astype(mx.bfloat16)
    s64 = np.array(scores.astype(mx.float32)).astype(np.float64)
    w64 = np.array(weights.astype(mx.float32)).astype(np.float64)
    exact = np.einsum("bsh,bsht->bst", w64, np.maximum(s64, 0))
    rounded = mx.array(exact.astype(np.float32)).astype(mx.bfloat16)
    fused = indexer_patch.relu_weighted_sum(scores, weights)
    original = mx.matmul(weights[:, :, None, :], mx.maximum(scores, 0)).squeeze(2)
    fused_misses = mx.sum(mx.not_equal(fused, rounded)).item()
    assert fused_misses <= 4
    assert fused_misses <= mx.sum(mx.not_equal(original, rounded)).item()


def test_hc_mix_many_rows_matches_float64() -> None:
    mx.random.seed(12)
    x = mx.random.normal((1, 130, 1024)).astype(mx.bfloat16)
    fn = mx.random.normal((24, 1024)) * 0.05
    x64 = np.array(x.astype(mx.float32)).astype(np.float64)
    normed = x64 / np.sqrt(np.mean(x64 * x64, axis=-1, keepdims=True) + 1e-6)
    exact = normed @ np.array(fn).astype(np.float64).T
    fused = kernels.hc_mix(x, fn, 1e-6)
    assert np.allclose(np.array(fused), exact, rtol=1e-4, atol=1e-5)


def _chunked_attention(attention: dv4.V4Attention, x: mx.array, chunk: int) -> mx.array:
    cache = dv4.DeepseekV4Cache(16)
    outputs: list[mx.array] = []
    for start in range(0, x.shape[1], chunk):
        output = attention(x[:, start : start + chunk], cache=cache)
        mx.eval(output)
        outputs.append(output)
    return mx.concatenate(outputs, axis=1).astype(mx.float32)


@pytest.mark.parametrize("layer", [0, 1, 2, 3])
@pytest.mark.parametrize("chunk", [64, 320])
def test_patched_prefill_attention_matches_mlx_lm(
    monkeypatch: pytest.MonkeyPatch, layer: int, chunk: int
) -> None:
    # 320 tokens: past the sliding window, the ratio-128 pool's first rows and
    # the ratio-4 pool outgrowing index_topk (64 rows).
    model = tiny_model()
    attention = model.model.layers[layer].attn
    mx.random.seed(13)
    x = (mx.random.normal((1, 320, 256)) * 0.5).astype(mx.bfloat16)
    reference = _chunked_attention(attention, x, chunk)

    monkeypatch.setattr(dv4.Indexer, "__call__", indexer_patch._patched_call)
    prefill.patch_deepseek_v4_prefill_attention()
    patched = _chunked_attention(attention, x, chunk)

    per_position = mx.abs(reference - patched).max(-1) / mx.abs(reference).max(-1)
    assert per_position.max().item() < 1e-2


def _prefill_logits(model: dv4.Model, chunk: int) -> mx.array:
    """Chunked prefill of 300 tokens, then a few decode steps."""
    rng = np.random.default_rng(13)
    tokens = mx.array(rng.integers(0, VOCAB_SIZE, (1, 306)))
    cache = model.make_cache()
    steps: list[mx.array] = []
    for start in range(0, 300, chunk):
        logits = model(tokens[:, start : min(start + chunk, 300)], cache=cache)
        mx.eval(logits)
        steps.append(logits.astype(mx.float32))
    for position in range(300, 306):
        logits = model(tokens[:, position : position + 1], cache=cache)
        mx.eval(logits)
        steps.append(logits.astype(mx.float32))
    return mx.concatenate(steps, axis=1)


@pytest.mark.parametrize("chunk", [64, 300])
def test_patched_prefill_matches_mlx_lm(
    monkeypatch: pytest.MonkeyPatch, chunk: int
) -> None:
    model = tiny_model()
    reference = _prefill_logits(model, chunk)

    monkeypatch.setattr(dv4.HyperConnection, "hc_pre", kernels._patched_hc_pre)
    monkeypatch.setattr(dv4, "_hc_expand_ops", kernels._patched_hc_expand_ops)
    monkeypatch.setattr(dv4.Indexer, "__call__", indexer_patch._patched_call)
    monkeypatch.setattr(dv4.V4Attention, "__call__", kernels._patched_attention)
    prefill.patch_deepseek_v4_prefill_attention()
    patched = _prefill_logits(model, chunk)

    # The random tiny model routes tokens between near-tied experts, so a
    # rounding difference occasionally flips a route and that token's later
    # context; nearly every position must still match.
    per_position = mx.sqrt(mx.sum(mx.square(reference - patched), -1)) / mx.sqrt(
        mx.sum(mx.square(reference), -1)
    )
    assert mx.mean(per_position < 1e-2).item() >= 0.9
    agreement = mx.mean(mx.equal(reference.argmax(-1), patched.argmax(-1))).item()
    assert agreement >= 0.95


def test_prefill_attention_skips_batched_caches() -> None:
    model = tiny_model()
    attention = model.model.layers[1].attn
    x = mx.zeros((2, 8, 256), dtype=mx.bfloat16)
    uniform = dv4.DeepseekV4Cache(16)
    assert prefill._prefill_supported(attention, x, uniform)
    assert not prefill._prefill_supported(attention, x[:, :4], uniform)
    batched = dv4.DeepseekV4Cache(16)
    batched.prepare(lengths=[8, 5], right_padding=[0, 3])
    assert not prefill._prefill_supported(attention, x, batched)
