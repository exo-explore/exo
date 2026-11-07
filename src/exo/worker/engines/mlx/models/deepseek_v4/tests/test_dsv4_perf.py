"""DeepSeek V4's performance paths against the code they replace: the FP8 projections kept
as mxfp8 hold exactly the bf16 expansion's values; the fused rounding kernel matches the
rounding ops bit for bit; the hyper-connection and indexer-scoring kernels match their
ops; the decode attention kernels are no further from a float32 computation than MLX's
fallback; whole-head RoPE matches RoPE on the trailing dims; and the prefill attention
over each query's own keys matches the masked attention over all keys (and its query
split), on a tiny model in float32."""

import math
from collections.abc import Iterator

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from exo.worker.engines.mlx.models.deepseek_v4 import model as dsv4
from exo.worker.engines.mlx.models.deepseek_v4.model import Model, ModelArgs

PROMPT_TOKENS = 160

needs_metal = pytest.mark.skipif(
    dsv4._block_round_kernel is None,  # pyright: ignore[reportPrivateUsage]
    reason="the fused kernels need the Metal GPU",
)


def _fp8_weight(shape: tuple[int, int], seed: int) -> tuple[mx.array, mx.array]:
    """Random e4m3 bytes (no NaN encodings) and ue8m0 128x128 block scales."""
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 256, size=shape).astype(np.uint8) & 0xF7
    scales = rng.integers(
        110, 135, size=((shape[0] + 127) // 128, (shape[1] + 127) // 128)
    )
    return mx.array(w), mx.array(scales.astype(np.uint8))


def _e4m3(byte: int) -> float:
    """An e4m3 byte's value (bias 7, subnormals below 2**-6)."""
    sign = -1.0 if byte >> 7 else 1.0
    exponent = (byte >> 3) & 0xF
    mantissa = (byte & 7) / 8.0
    if exponent == 0:
        return sign * mantissa * 2.0**-6
    return sign * (1.0 + mantissa) * 2.0 ** (exponent - 7)


def _bf16_expansion(weight: mx.array, scale: mx.array) -> mx.array:
    """FP8 times its 128x128 block's power-of-two scale, from value tables (not MLX's FP8
    conversion): the values #2516 loads as bf16 (exactly representable, so the cast is
    exact)."""
    values = mx.array([_e4m3(b) for b in range(256)], dtype=mx.float32)
    powers = mx.array([2.0 ** (b - 127) for b in range(256)], dtype=mx.float32)
    w = mx.take(values, weight.astype(mx.uint32))
    s = mx.take(powers, scale.astype(mx.uint32))
    m, n = weight.shape
    full = w.reshape(m // 128, 128, n // 128, 128) * s[:, None, :, None]
    return full.reshape(m, n).astype(mx.bfloat16)


@pytest.mark.parametrize("shape", [(256, 512), (1536, 4096), (512, 1024)])
def test_fp8_weights_as_mxfp8_hold_the_same_values(shape: tuple[int, int]) -> None:
    weight, scale = _fp8_weight(shape, seed=shape[0])
    packed, scales = dsv4._fp8_block_to_mxfp8(weight, scale)  # pyright: ignore[reportPrivateUsage]
    dequantized = mx.dequantize(packed, scales, group_size=32, bits=8, mode="mxfp8")
    dequantized = dequantized.astype(mx.bfloat16)
    assert mx.array_equal(dequantized, _bf16_expansion(weight, scale))


# Row counts around the switch to the bf16 GEMM (512) and MLX's quantized-matmul kernel
# choices for small inputs (1, 2-8 and more rows), as one sequence or a batch of them.
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 8),
        (1, 9),
        (8, 1),
        (1, 16),
        (1, 64),
        (1, 129),
        (1, 511),
        (1, 512),
        (1, 513),
        (2, 256),
        (3, 171),
        (1, 600),
        (1, 2048),
    ],
)
def test_fp8_linear_matches_the_bf16_linear(shape: tuple[int, int]) -> None:
    weight, scale = _fp8_weight((512, 1024), seed=3)
    packed, scales = dsv4._fp8_block_to_mxfp8(weight, scale)  # pyright: ignore[reportPrivateUsage]
    fp8 = dsv4.Fp8BlockLinear(1024, 512)
    fp8.weight, fp8.scales = packed, scales
    dense = nn.Linear(1024, 512, bias=False)
    dense.weight = _bf16_expansion(weight, scale)
    mx.random.seed(0)
    x = (mx.random.normal((*shape, 1024)) * 0.5).astype(mx.bfloat16)
    if shape[0] * shape[1] >= 512:
        # prefill-sized inputs dequantize: the bf16 GEMM's result exactly
        assert mx.array_equal(fp8(x), dense(x))
    else:
        # decode-sized inputs use the quantized matmul: the same values, another summation
        # order
        a = fp8(x).astype(mx.float32)
        b = dense(x).astype(mx.float32)
        assert float(mx.max(mx.abs(a - b)) / mx.max(mx.abs(b))) < 1e-2


@pytest.mark.parametrize("batch", [1, 2, 5, 8])
def test_grouped_fp8_output_projection_reads_each_weight_once(batch: int) -> None:
    """wo_a's groups as one quantized matmul with the batch's rows as its rows: one row
    gives each group's own quantized matmul bit for bit; in a batch every row stays as
    close to the bf16 projection as alone."""
    groups, rank, feat = 4, 128, 1024
    args = ModelArgs(
        vocab_size=256,
        hidden_size=1024,
        num_hidden_layers=1,
        num_attention_heads=8,
        q_lora_rank=64,
        o_lora_rank=rank,
        o_groups=groups,
        head_dim=512,
        qk_rope_head_dim=64,
        num_nextn_predict_layers=0,
        quantization_config={
            "quant_method": "fp8",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    )
    attention = dsv4.V4Attention(args, 0)
    weight, scale = _fp8_weight((groups * rank, feat), seed=5)
    packed, scales = dsv4._fp8_block_to_mxfp8(weight, scale)  # pyright: ignore[reportPrivateUsage]
    attention.wo_a.weight, attention.wo_a.scales = packed, scales
    dense = _bf16_expansion(weight, scale).reshape(groups, rank, feat)
    mx.random.seed(batch)
    out = (mx.random.normal((batch, 1, groups * feat)) * 0.5).astype(mx.bfloat16)
    got = attention._grouped_output_projection(out)

    def per_group(row: mx.array) -> mx.array:
        return mx.concatenate(
            [
                mx.quantized_matmul(
                    row[..., g * feat : (g + 1) * feat],
                    packed[g * rank : (g + 1) * rank],
                    scales[g * rank : (g + 1) * rank],
                    transpose=True,
                    group_size=32,
                    bits=8,
                    mode="mxfp8",
                )
                for g in range(groups)
            ],
            axis=-1,
        )

    def error(x: mx.array, row: mx.array) -> float:
        rows = row.reshape(groups, 1, feat).astype(mx.float32)
        exact = (rows @ dense.astype(mx.float32).swapaxes(-1, -2)).reshape(
            1, 1, groups * rank
        )
        diff = x.astype(mx.float32) - exact
        return float(mx.sqrt(mx.sum(diff * diff) / mx.sum(exact * exact)))

    if batch == 1:
        assert mx.array_equal(got, per_group(out))
    for b in range(batch):
        row = out[b : b + 1]
        assert error(got[b : b + 1], row) <= error(per_group(row), row) * 1.5 + 1e-3


def _same_values(a: mx.array, b: mx.array) -> bool:
    return bool(
        np.array_equal(
            np.array(a.astype(mx.float32)),
            np.array(b.astype(mx.float32)),
            equal_nan=True,
        )
    )


@needs_metal
@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 300.0, 1e5])
def test_rounding_kernel_matches_the_ops(scale: float) -> None:
    mx.random.seed(int(math.log10(scale) + 10))
    kv = (mx.random.normal((3, 7, 512)) * scale).astype(mx.bfloat16)
    assert _same_values(
        dsv4._fp8_kv_sim(kv, 64),  # pyright: ignore[reportPrivateUsage]
        dsv4._fp8_kv_sim_ops(kv, 64),  # pyright: ignore[reportPrivateUsage]
    )
    q = (mx.random.normal((2, 5, 64, 128)) * scale).astype(mx.bfloat16)
    assert _same_values(
        dsv4._fp4_indexer_sim(q),  # pyright: ignore[reportPrivateUsage]
        dsv4._fp4_rotate_round(q, dsv4._hadamard(128)),  # pyright: ignore[reportPrivateUsage]
    )


@needs_metal
def test_rounding_kernel_matches_the_ops_on_every_bf16_value() -> None:
    bits = np.arange(0, 65536, dtype=np.uint32)
    values = (bits << 16).view(np.float32)
    values = values[np.isfinite(values)]
    rows = np.concatenate([values, np.zeros((-len(values)) % 448, np.float32)]).reshape(
        -1, 448
    )
    kv = mx.array(np.concatenate([rows, np.zeros((rows.shape[0], 64), np.float32)], 1))
    kv = kv.astype(mx.bfloat16)
    assert _same_values(
        dsv4._fp8_kv_sim(kv, 64),  # pyright: ignore[reportPrivateUsage]
        dsv4._fp8_kv_sim_ops(kv, 64),  # pyright: ignore[reportPrivateUsage]
    )
    x = mx.array(rows.reshape(-1)[: (rows.size // 128) * 128].reshape(-1, 128))
    x = x.astype(mx.bfloat16)
    fused = dsv4._block_round_fused(  # pyright: ignore[reportPrivateUsage]
        x.astype(mx.float32), 32, True, 128, mx.bfloat16, via_bf16=True
    )
    assert _same_values(fused, dsv4._block_round(x, 32, True))  # pyright: ignore[reportPrivateUsage]


@needs_metal
@pytest.mark.parametrize("rows", [1, 3, 300])
def test_hyper_connection_kernels_match_the_ops(rows: int) -> None:
    mx.random.seed(rows)
    hc = dsv4.HyperConnection(256, 4, 1e-6, 20, 1e-6)
    hc.fn = mx.random.normal((24, 1024)) * 0.05
    hc.base = mx.random.normal((24,)) * 0.1
    hc.scale = mx.array([0.5, 0.3, 0.2])
    x = (mx.random.normal((1, rows, 4, 256)) * 2).astype(mx.bfloat16)
    f = mx.random.normal((1, rows, 256)).astype(mx.bfloat16)
    flat = x.reshape(1, rows, 1024).astype(mx.float32)
    flat = flat * mx.rsqrt(mx.mean(flat * flat, axis=-1, keepdims=True) + 1e-6)
    mixes = flat @ hc.fn.T
    _, post_ref, comb_ref = dsv4.hc_sinkhorn_collapse(
        mixes, hc.scale, hc.base, x, 4, 20, 1e-6
    )
    y, post, comb = hc.hc_pre(x)
    np.testing.assert_allclose(np.array(post), np.array(post_ref), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.array(comb), np.array(comb_ref), rtol=1e-5, atol=1e-6)
    assert mx.array_equal(
        hc.hc_post(f, x, post, comb),
        dsv4._hc_expand_ops(f, x, post, comb),  # pyright: ignore[reportPrivateUsage]
    )
    assert y.shape == (1, rows, 256)


@needs_metal
@pytest.mark.parametrize(("rows", "pool"), [(1, 600), (3, 1000)])
def test_indexer_decode_kernel_matches_the_matmuls(rows: int, pool: int) -> None:
    mx.random.seed(pool)
    q = dsv4._fp4_indexer_sim(mx.random.normal((rows, 1, 64, 128)).astype(mx.bfloat16))  # pyright: ignore[reportPrivateUsage]
    k = dsv4._fp4_indexer_sim(mx.random.normal((rows, pool, 128)).astype(mx.bfloat16))  # pyright: ignore[reportPrivateUsage]
    w = (mx.random.normal((rows, 1, 64)) * -0.02).astype(mx.bfloat16)
    kernel = dsv4._indexer_score_kernel  # pyright: ignore[reportPrivateUsage]
    assert kernel is not None
    fused = kernel(
        inputs=[q, k, w],
        template=[("H", 64), ("D", 128)],
        grid=(rows * pool * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, 1, pool)],
        output_dtypes=[mx.bfloat16],
    )[0]
    scores = mx.maximum(
        (q.reshape(rows, 64, 128) @ k.swapaxes(-1, -2)).reshape(rows, 1, 64, pool), 0
    )
    reference = (w[:, :, None, :] @ scores).squeeze(2)
    # float32 summation order only: equal here, and within one bf16 step in general
    diff = mx.abs(fused.astype(mx.float32) - reference.astype(mx.float32))
    assert (
        float(mx.max(diff))
        <= float(mx.max(mx.abs(reference.astype(mx.float32)))) * 2**-7
    )


@needs_metal
@pytest.mark.parametrize(
    ("window", "pool", "selected"), [(128, 1500, 512), (37, 300, None), (128, 0, None)]
)
def test_decode_attention_kernels_match_the_mlx_fallback(
    window: int, pool: int, selected: int | None
) -> None:
    """The decode kernels against a float32 computation of the same attention, next to MLX's
    fallback: no further from it (they keep the softmax weights in float32)."""
    mx.random.seed(window + pool)
    heads, dim, scale = 64, 512, 1.0 / math.sqrt(512)
    q = (mx.random.normal((1, heads, 1, dim)) * 1.5).astype(mx.bfloat16)
    win = mx.random.normal((1, window, dim)).astype(mx.bfloat16)
    rows = mx.random.normal((1, max(pool, 1), dim)).astype(mx.bfloat16)
    sinks = (mx.random.normal((heads,)) * 2).astype(mx.bfloat16)
    idx = None
    keys = win
    if pool:
        if selected is not None:
            idx = (
                mx.random.permutation(pool)[:selected]
                .astype(mx.uint32)
                .reshape(1, 1, selected)
            )
            keys = mx.concatenate([win, mx.take(rows, idx.reshape(-1), axis=1)], axis=1)
        else:
            keys = mx.concatenate([win, rows], axis=1)
    fused = dsv4._decode_attention(  # pyright: ignore[reportPrivateUsage]
        q, win, rows if pool else None, idx, scale, sinks
    )
    assert fused is not None
    fallback = dsv4.scaled_dot_product_attention(
        q, keys[:, None], keys[:, None], cache=None, scale=scale, mask=None, sinks=sinks
    )
    k32 = keys.astype(mx.float32)[:, None]
    scores = (q.astype(mx.float32) * scale) @ k32.swapaxes(-1, -2)
    sink_col = mx.broadcast_to(
        sinks.astype(mx.float32)[None, :, None, None], (1, heads, 1, 1)
    )
    probs = mx.softmax(mx.concatenate([sink_col, scores], axis=-1), axis=-1)[..., 1:]
    exact = probs @ k32

    def error(x: mx.array) -> float:
        diff = x.astype(mx.float32) - exact
        return float(mx.sqrt(mx.sum(diff * diff) / mx.sum(exact * exact)))

    assert error(fused) <= error(fallback) * 1.1 + 1e-4


@needs_metal
@pytest.mark.parametrize("selected", [64, None])
def test_decode_attention_kernels_skip_each_rows_padding(selected: int | None) -> None:
    """Rows of different lengths in one batch: a short row's padded window slots (left
    padding, or a wrapped range once the window rotates) and its compressed rows past its
    own pool length are skipped, like the masked fallback does."""
    mx.random.seed(7)
    batch, heads, dim, window, pool = 3, 64, 512, 128, 300
    scale = 1.0 / math.sqrt(dim)
    q = (mx.random.normal((batch, heads, 1, dim)) * 1.5).astype(mx.bfloat16)
    win = mx.random.normal((batch, window, dim)).astype(mx.bfloat16)
    rows = mx.random.normal((batch, pool, dim)).astype(mx.bfloat16)
    sinks = (mx.random.normal((heads,)) * 2).astype(mx.bfloat16)
    slots = np.arange(window)
    valid_np = np.stack([slots >= 0, slots >= 100, (slots < 10) | (slots >= 50)])
    window_valid = mx.array(valid_np)[:, None, None, :]
    lengths = [pool, 120, 7]
    idx = None
    if selected is not None:
        idx = mx.stack([mx.random.permutation(pool)[:selected] for _ in range(batch)])
        idx = idx.astype(mx.uint32).reshape(batch, 1, selected)
    fused = dsv4._decode_attention(  # pyright: ignore[reportPrivateUsage]
        q, win, rows, idx, scale, sinks, window_valid, lengths
    )
    assert fused is not None
    chosen = (
        rows if idx is None else mx.take_along_axis(rows, idx[:, 0, :, None], axis=1)
    )
    keys = mx.concatenate([win, chosen], axis=1)
    mask = dsv4._decode_padding_mask(  # pyright: ignore[reportPrivateUsage]
        batch, window_valid, window, lengths, pool, idx
    )
    fallback = dsv4.scaled_dot_product_attention(
        q, keys[:, None], keys[:, None], cache=None, scale=scale, mask=mask, sinks=sinks
    )
    assert mask is not None
    k32 = keys.astype(mx.float32)[:, None]
    scores = (q.astype(mx.float32) * scale) @ k32.swapaxes(-1, -2)
    scores = mx.where(mask, scores, -mx.inf)
    sink_col = mx.broadcast_to(
        sinks.astype(mx.float32)[None, :, None, None], (batch, heads, 1, 1)
    )
    probs = mx.softmax(mx.concatenate([sink_col, scores], axis=-1), axis=-1)[..., 1:]
    exact = probs @ k32

    def error(x: mx.array, b: int) -> float:
        diff = x[b].astype(mx.float32) - exact[b]
        return float(mx.sqrt(mx.sum(diff * diff) / mx.sum(exact[b] * exact[b])))

    for b in range(batch):
        assert error(fused, b) <= error(fallback, b) * 1.1 + 1e-4


def test_whole_head_rope_matches_rope_on_the_trailing_dims() -> None:
    rope = dsv4.DeepseekV4RoPE(64, 10000.0)
    mx.random.seed(2)
    q = mx.random.normal((1, 8, 37, 512)).astype(mx.bfloat16)
    kv = mx.random.normal((1, 37, 512)).astype(mx.bfloat16)
    for offset in (0, 100, 5000):
        got_q, got_kv = dsv4._attn_qkv_partial_rope(  # pyright: ignore[reportPrivateUsage]
            q, kv, offset, 64, rope.head_freqs(512)
        )
        for got, x in ((got_q, q), (got_kv, kv)):
            tail = rope(x[..., -64:], offset=offset)
            expected = mx.concatenate([x[..., :-64], tail], axis=-1)
            assert mx.array_equal(got.view(mx.uint16), expected.view(mx.uint16))
        # and back: the inverse rotation on the output
        back = dsv4._attn_inv_rope_flatten(  # pyright: ignore[reportPrivateUsage]
            q, offset, 64, rope.head_freqs(512), 8 * 512
        )
        tail = rope(q[..., -64:], offset=offset, inverse=True)
        expected = mx.concatenate([q[..., :-64], tail], axis=-1)
        expected = expected.transpose(0, 2, 1, 3).reshape(1, 37, 8 * 512)
        assert mx.array_equal(back.view(mx.uint16), expected.view(mx.uint16))


def _tiny_model() -> Model:
    mx.random.seed(0)
    args = ModelArgs(
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        q_lora_rank=64,
        o_lora_rank=32,
        o_groups=2,
        head_dim=128,
        qk_rope_head_dim=64,
        sliding_window=16,
        compress_ratios=[0, 4, 128, 4],
        index_n_heads=4,
        index_head_dim=64,
        index_topk=8,
        moe_intermediate_size=64,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hash_layers=1,
        num_nextn_predict_layers=0,
        max_position_embeddings=4096,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


def _no_kv_rounding(kv: mx.array, _rope_dims: int) -> mx.array:
    return kv


def _no_indexer_rounding(x: mx.array) -> mx.array:
    return x


@pytest.fixture(scope="module")
def tiny_model() -> Iterator[Model]:
    """With the FP8 / FP4 roundings off, as in test_dsv4_model: they are discontinuous,
    so float32 summation-order differences could flip one and the comparison would
    measure that instead of the attention."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dsv4, "_fp8_kv_sim", _no_kv_rounding)
        patch.setattr(dsv4, "_fp4_indexer_sim", _no_indexer_rounding)
        yield _tiny_model()


def _logits(model: Model, step: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    tokens = mx.array(rng.integers(0, 256, size=(1, PROMPT_TOKENS)).astype(np.int32))
    cache = model.make_cache()
    out: list[mx.array] = []
    for start in range(0, PROMPT_TOKENS, step):
        logits = model(tokens[:, start : start + step], cache=cache).astype(mx.float32)
        out.append(logits - mx.logsumexp(logits, axis=-1, keepdims=True))
        mx.eval(out[-1])
    return np.array(mx.concatenate(out, axis=1)[0])


@pytest.mark.parametrize("step", [PROMPT_TOKENS, 48, 7])
def test_prefill_attention_matches_the_masked_attention(
    tiny_model: Model, step: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past 32 tokens the indexer layers select 8 of their rows (the per-query gather);
    the other layers attend to a block's window rows and all compressed rows. Both must
    give what the masked attention over every key gives, and so must the masked
    attention split into query slices (the MLX sinks bug work-around)."""
    monkeypatch.setattr(dsv4, "_PREFILL_Q_BLOCK", 32)
    new = _logits(tiny_model, step)
    # the indexer scoring a few queries at a time (long contexts) selects the same rows
    monkeypatch.setattr(dsv4, "_INDEXER_MAX_SCORES", 2**9)
    np.testing.assert_allclose(_logits(tiny_model, step), new, atol=1e-5)
    monkeypatch.setattr(dsv4, "_INDEXER_MAX_SCORES", 2**30)
    monkeypatch.setattr(dsv4, "_PREFILL_PER_QUERY_KEYS", False)
    masked = _logits(tiny_model, step)
    np.testing.assert_allclose(new, masked, atol=1e-4)
    monkeypatch.setattr(dsv4, "_SDPA_MAX_SCORES", 2**10)
    split = _logits(tiny_model, step)
    np.testing.assert_allclose(split, masked, atol=1e-4)
