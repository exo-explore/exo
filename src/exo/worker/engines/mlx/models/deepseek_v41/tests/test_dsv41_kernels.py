"""DeepSeek V4.1's fused Metal kernels against the plain MLX ops they replace (and the
fallback when they are off). The kernels keep the ops' float32 arithmetic and roundings;
accumulation orders differ, so outputs agree to float32 noise, which can flip the last
bit of a bf16 result."""

import math
from collections.abc import Callable, Iterator
from typing import cast

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.models.deepseek_v41 import kernels
from exo.worker.engines.mlx.models.deepseek_v41.model import (
    DeepseekV41SwiGLU,
    HyperConnection,
    ModelArgs,
    RoPE,
    _apply_rope,  # pyright: ignore[reportPrivateUsage]
    _collapse,  # pyright: ignore[reportPrivateUsage]
    _expand,  # pyright: ignore[reportPrivateUsage]
    _sparse_attention,  # pyright: ignore[reportPrivateUsage]
    _window_indices,  # pyright: ignore[reportPrivateUsage]
    fake_quant_fp8,
)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the fused kernels need Metal"
)


@pytest.fixture(autouse=True)
def fused_on() -> Iterator[None]:
    kernels.set_fused(True)
    yield
    kernels.set_fused(True)


def _plain[T](fn: Callable[[], T]) -> T:
    kernels.set_fused(False)
    try:
        return fn()
    finally:
        kernels.set_fused(True)


def _rel(a: mx.array, b: mx.array) -> float:
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    diff = mx.sqrt(mx.sum(mx.square(a32 - b32)))
    return float(diff / (mx.sqrt(mx.sum(mx.square(b32))) + 1e-30))


def _same(a: mx.array, b: mx.array) -> float:
    """The fraction of identical elements."""
    return float(mx.mean(mx.equal(a, b)))


def _rope() -> RoPE:
    scaling: dict[str, float | int | str] = {
        "original_max_position_embeddings": 65536,
        "factor": 16,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    return RoPE(64, 160000.0, scaling)


@pytest.mark.parametrize(
    "length,previous,compressed,start",
    [(1, 128, 1024, 3000), (6, 128, 1024, 777), (1, 20, 0, 20), (40, 128, 512, 100)],
)
def test_attention_with_rope_matches_the_ops(
    length: int, previous: int, compressed: int, start: int
) -> None:
    mx.random.seed(0)
    heads, dim = 16, 512
    rope = _rope()
    q = (mx.random.normal((1, length, heads, dim)) * 2).astype(mx.bfloat16)
    window = fake_quant_fp8(
        (mx.random.normal((1, previous + length, dim)) * 3).astype(mx.bfloat16)
    )
    width = min(128, previous + length)
    widx = mx.broadcast_to(
        _window_indices(previous=previous, length=length, window=128),
        (1, length, width),
    )
    comp = cidx = None
    if compressed:
        comp = mx.random.normal((1, compressed, dim)).astype(mx.bfloat16)
        cidx = mx.random.randint(-1, compressed, (1, length, 256)).astype(mx.int32)
    sinks = mx.random.normal((heads,))
    scale = 1.0 / math.sqrt(dim)

    def plain() -> mx.array:
        qr = _apply_rope(q, rope, rope_dim=64, offset=start)
        kvs, idx = [window], [widx]
        if comp is not None and cidx is not None:
            kvs.append(comp)
            idx.append(cidx)
        out = _sparse_attention(qr, kvs, idx, sinks=sinks, scale=scale)
        return _apply_rope(out, rope, rope_dim=64, offset=start, inverse=True)

    reference = _plain(plain)
    fused = kernels.attention_with_rope(
        q,
        window,
        widx,
        comp,
        cidx,
        sinks=sinks,
        scale=scale,
        periods=rope.periods,
        rope_dim=64,
        start_pos=start,
    )
    assert fused is not None
    assert _rel(fused, reference) < 3e-4
    assert _same(fused, reference) > 0.99


def test_kv_epilogue_is_bit_identical() -> None:
    mx.random.seed(1)
    rope = _rope()
    weight = (mx.random.normal((512,)) * 0.1 + 1).astype(mx.bfloat16)
    for length, start in ((1, 4000), (7, 100), (300, 0)):
        kv = (mx.random.normal((1, length, 512)) * 5).astype(mx.bfloat16)
        reference = _plain(
            lambda kv=kv, start=start: fake_quant_fp8(
                _apply_rope(
                    mx.fast.rms_norm(kv, weight, 1e-20), rope, rope_dim=64, offset=start
                ),
                32,
            )
        )
        fused = kernels.kv_epilogue(
            kv, weight, eps=1e-20, periods=rope.periods, rope_dim=64, start_pos=start
        )
        assert fused is not None
        assert bool(mx.array_equal(fused, reference))


def test_route_and_swiglu_are_bit_identical() -> None:
    mx.random.seed(2)
    logits = mx.random.normal((5, 384)) * 3
    bias = mx.random.normal((384,)) * 0.1
    routed = kernels.route(
        logits, bias, top_k=6, gate_temp=1.0, norm_topk_prob=True, route_scale=1.5
    )
    assert routed is not None
    ids, weights = routed
    scores = mx.sqrt(mx.logaddexp(logits, mx.array(0.0)))
    chosen = mx.argpartition(-(scores + bias), 5, axis=-1)[..., :6]
    expected = mx.take_along_axis(scores, chosen, axis=-1)
    expected = expected / (mx.sum(expected, axis=-1, keepdims=True) + 1e-20) * 1.5
    got_rows = cast(list[list[int]], ids.tolist())
    want_rows = cast(list[list[int]], chosen.tolist())
    for got_ids, want_ids in zip(got_rows, want_rows, strict=True):
        assert sorted(got_ids) == sorted(want_ids)
    order_got = mx.argsort(ids, axis=-1)
    order_want = mx.argsort(chosen, axis=-1)
    assert bool(
        mx.allclose(
            mx.take_along_axis(weights, order_got, -1),
            mx.take_along_axis(expected, order_want, -1),
            rtol=1e-6,
            atol=0,
        )
    )
    up = (mx.random.normal((6, 2304)) * 8).astype(mx.bfloat16)
    gate = (mx.random.normal((6, 2304)) * 8).astype(mx.bfloat16)
    activation = DeepseekV41SwiGLU(10.0)
    reference = _plain(lambda: activation(up, gate))
    assert bool(mx.array_equal(activation(up, gate), reference))


def test_head_logits_are_float32_sums_of_bf16_products() -> None:
    mx.random.seed(3)
    weight = (mx.random.normal((3000, 1024)) * 0.02).astype(mx.bfloat16)
    for rows in (1, 3, 6):
        x = mx.random.normal((1, rows, 1024)).astype(mx.bfloat16)
        reference = x.astype(mx.float32) @ weight.astype(mx.float32).T
        fused = kernels.head_logits(x, weight)
        assert fused.dtype == mx.float32
        assert _rel(fused, reference) < 1e-6


def test_hyper_connection_boundary_matches_the_ops() -> None:
    mx.random.seed(4)
    args = ModelArgs(hidden_size=1024, hc_mult=4)
    attn_hc, ffn_hc = HyperConnection(args), HyperConnection(args)
    for hc in (attn_hc, ffn_hc):
        hc.fn = mx.random.normal(hc.fn.shape) * 0.02
        hc.base = mx.random.normal(hc.base.shape) * 0.5
        hc.scale = mx.random.uniform(shape=(3,)) + 0.5
    norm_weight = (mx.random.normal((1024,)) * 0.1 + 1).astype(mx.bfloat16)
    for rows in (1, 6):
        streams = (mx.random.normal((1, rows, 4, 1024)) * 4).astype(mx.bfloat16)
        x = (mx.random.normal((1, rows, 1024)) * 2).astype(mx.bfloat16)
        pre, post, comb = _plain(lambda streams=streams: attn_hc.mixes(streams))

        def plain(
            x: mx.array = x,
            streams: mx.array = streams,
            pre: mx.array = pre,
            post: mx.array = post,
            comb: mx.array = comb,
        ) -> tuple[mx.array, ...]:
            new = _expand(x, streams, post, comb)
            p, q, c = ffn_hc.mixes(new)
            h = mx.fast.rms_norm(_collapse(new, pre), norm_weight, 1e-20)
            return new, p, q, c, h

        new_ref, p_ref, q_ref, c_ref, h_ref = _plain(plain)
        new, partials = kernels.hc_partials(streams, ffn_hc.fn, (x, post, comb))
        p, q, c, h = kernels.hc_finish(
            partials,
            new,
            pre,
            scale=ffn_hc.scale,
            base=ffn_hc.base,
            norm_weight=norm_weight,
            norm_eps=1e-20,
            hc_eps=args.hc_eps,
            iters=args.hc_sinkhorn_iters,
        )
        assert _same(new, new_ref) > 0.999
        for got, want in ((p, p_ref), (q, q_ref), (c, c_ref)):
            assert _rel(got, want) < 1e-5
        assert _same(h, h_ref) > 0.999


def test_kernels_off_fall_back_to_the_ops() -> None:
    kernels.set_fused(False)
    q = mx.zeros((1, 1, 16, 512), dtype=mx.bfloat16)
    window = mx.zeros((1, 4, 512), dtype=mx.bfloat16)
    idx = mx.zeros((1, 1, 4), dtype=mx.int32)
    assert (
        kernels.attention_with_rope(
            q,
            window,
            idx,
            None,
            None,
            sinks=mx.zeros((16,)),
            scale=1.0,
            periods=_rope().periods,
            rope_dim=64,
            start_pos=0,
        )
        is None
    )
    assert (
        kernels.route(
            mx.zeros((1, 8)),
            mx.zeros((8,)),
            top_k=2,
            gate_temp=1.0,
            norm_topk_prob=True,
            route_scale=1.0,
        )
        is None
    )
    assert not kernels.hc_fusable(mx.zeros((1, 1, 4, 1024), dtype=mx.bfloat16))
    weight = mx.ones((8, 256), dtype=mx.bfloat16)
    logits = kernels.head_logits(mx.ones((1, 256), dtype=mx.bfloat16), weight)
    assert logits.dtype == mx.float32 and float(logits[0, 0]) == 256.0
