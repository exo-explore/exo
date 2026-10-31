"""DeepSeek V4.1's fused Metal kernels against the plain MLX ops they replace (and the
fallback when they are off). The kernels keep the ops' float32 arithmetic and roundings;
accumulation orders differ, so outputs agree to float32 noise, which can flip the last
bit of a bf16 result."""

import math
from collections.abc import Callable, Iterator
from typing import cast

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from exo.worker.engines.mlx.models.deepseek_v41 import kernels
from exo.worker.engines.mlx.models.deepseek_v41.model import (
    DeepseekV41SwiGLU,
    HyperConnection,
    LayerCache,
    Model,
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
    [
        (1, 128, 1024, 3000),
        (6, 128, 1024, 777),
        (1, 20, 0, 20),
        (16, 128, 1024, 500),
        (17, 128, 1024, 500),
        (40, 128, 512, 100),
    ],
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


@pytest.mark.parametrize("batch", [1, 2, 3, 4, 5, 8, 9, 16])
def test_kernels_give_each_row_the_same_bits_at_any_row_count(batch: int) -> None:
    """MLX picks some kernels by row count, so prefill chunk length or decode batch size can
    change results. The fused kernels must not add such switches: within the row counts each
    one takes, every row's output equals the one-row call's bit for bit."""
    mx.random.seed(5)

    def rows(value: mx.array) -> mx.array:
        return mx.broadcast_to(value, (batch, *value.shape[1:]))

    def same_rows(got: mx.array | None, want: mx.array | None) -> bool:
        assert got is not None and want is not None
        return bool(mx.array_equal(got, mx.broadcast_to(want, got.shape)))

    rope = _rope()
    weight = (mx.random.normal((3000, 1024)) * 0.02).astype(mx.bfloat16)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    if batch <= 8:
        assert same_rows(
            kernels.head_logits(rows(x), weight), kernels.head_logits(x, weight)
        )

    logits = mx.random.normal((1, 384)) * 3
    bias = mx.random.normal((384,)) * 0.1

    def route(value: mx.array) -> tuple[mx.array, mx.array]:
        routed = kernels.route(
            value, bias, top_k=6, gate_temp=1.0, norm_topk_prob=True, route_scale=1.5
        )
        assert routed is not None
        return routed

    for got, want in zip(route(rows(logits)), route(logits), strict=True):
        assert same_rows(got, want)

    activation = DeepseekV41SwiGLU(10.0)
    up = (mx.random.normal((1, 2048)) * 8).astype(mx.bfloat16)
    gate = (mx.random.normal((1, 2048)) * 8).astype(mx.bfloat16)
    assert same_rows(activation(rows(up), rows(gate)), activation(up, gate))

    routed_rows = mx.random.normal((1, 6, 1024)).astype(mx.bfloat16)
    weights = mx.random.uniform(shape=(1, 6))
    shared = mx.random.normal((1, 1024)).astype(mx.bfloat16)
    assert same_rows(
        kernels.moe_combine(
            rows(routed_rows), rows(weights), rows(shared), mx.bfloat16
        ),
        kernels.moe_combine(routed_rows, weights, shared, mx.bfloat16),
    )

    norm_weight = (mx.random.normal((512,)) * 0.1 + 1).astype(mx.bfloat16)
    kv = (mx.random.normal((1, 1, 512)) * 5).astype(mx.bfloat16)

    def epilogue(value: mx.array) -> mx.array | None:
        return kernels.kv_epilogue(
            value,
            norm_weight,
            eps=1e-20,
            periods=rope.periods,
            rope_dim=64,
            start_pos=777,
        )

    assert same_rows(epilogue(rows(kv)), epilogue(kv))

    heads, dim = 16, 512
    q = (mx.random.normal((1, 1, heads, dim)) * 2).astype(mx.bfloat16)
    window = (mx.random.normal((1, 128, dim)) * 3).astype(mx.bfloat16)
    window_idx = mx.arange(128, dtype=mx.int32).reshape(1, 1, 128)
    compressed = mx.random.normal((1, 1024, dim)).astype(mx.bfloat16)
    compressed_idx = mx.random.randint(-1, 1024, (1, 1, 256)).astype(mx.int32)
    sinks = mx.random.normal((heads,))

    def attention(count: int) -> mx.array | None:
        def widen(value: mx.array) -> mx.array:
            return mx.broadcast_to(value, (count, *value.shape[1:]))

        return kernels.attention_with_rope(
            widen(q),
            widen(window),
            widen(window_idx),
            widen(compressed),
            widen(compressed_idx),
            sinks=sinks,
            scale=1.0 / math.sqrt(dim),
            periods=rope.periods,
            rope_dim=64,
            start_pos=3000,
        )

    assert same_rows(attention(batch), attention(1))

    args = ModelArgs(hidden_size=1024, hc_mult=4)
    hc = HyperConnection(args)
    hc.fn = mx.random.normal(hc.fn.shape) * 0.02
    hc.base = mx.random.normal(hc.base.shape) * 0.5
    hc.scale = mx.random.uniform(shape=(3,)) + 0.5
    streams = (mx.random.normal((1, 1, 4, 1024)) * 4).astype(mx.bfloat16)
    h = (mx.random.normal((1, 1, 1024)) * 2).astype(mx.bfloat16)
    post = mx.random.uniform(shape=(1, 1, 4))
    comb = mx.softmax(mx.random.normal((1, 1, 4, 4)), axis=-1)
    pre = mx.softmax(mx.random.normal((1, 1, 4)), axis=-1)

    def boundary(count: int) -> tuple[mx.array, ...]:
        def widen(value: mx.array) -> mx.array:
            return mx.broadcast_to(value, (count, *value.shape[1:]))

        assert kernels.hc_fusable(widen(streams))
        new, partials = kernels.hc_partials(
            widen(streams), hc.fn, (widen(h), widen(post), widen(comb))
        )
        return (
            new,
            *kernels.hc_finish(
                partials,
                new,
                widen(pre),
                scale=hc.scale,
                base=hc.base,
                norm_weight=mx.ones((1024,), dtype=mx.bfloat16),
                norm_eps=1e-20,
                hc_eps=args.hc_eps,
                iters=args.hc_sinkhorn_iters,
            ),
        )

    for got, want in zip(boundary(batch), boundary(1), strict=True):
        assert same_rows(got, want)


def test_fused_paths_hand_over_to_the_ops_at_their_row_limits() -> None:
    """The head kernel takes up to 8 rows, attention and the hyper-connection boundary up to
    16; one row more and the plain ops run."""
    rows_16 = mx.zeros((1, 16, 4, 1024), dtype=mx.bfloat16)
    rows_17 = mx.zeros((1, 17, 4, 1024), dtype=mx.bfloat16)
    assert kernels.hc_fusable(rows_16)
    assert not kernels.hc_fusable(rows_17)
    mx.random.seed(6)
    weight = (mx.random.normal((3000, 1024)) * 0.02).astype(mx.bfloat16)
    for count in (8, 9):
        x = mx.random.normal((1, count, 1024)).astype(mx.bfloat16)
        reference = x.astype(mx.float32) @ weight.astype(mx.float32).T
        assert _rel(kernels.head_logits(x, weight), reference) < 1e-6


def test_a_pipeline_seam_expands_as_one_device_does() -> None:
    """On one device a layer's fused expand hands the next layer its mix partials. At a
    pipeline seam the stage's last layer expands with the same kernel and drops them, and
    the next stage's first layer computes them from the streams: bit-identical."""
    args = ModelArgs(
        vocab_size=256,
        hidden_size=1024,
        num_hidden_layers=2,
        num_attention_heads=8,
        head_dim=512,
        q_lora_rank=128,
        qk_rope_head_dim=64,
        o_groups=8,
        o_lora_rank=64,
        moe_intermediate_size=64,
        n_routed_experts=8,
        num_experts_per_tok=2,
        sliding_window=8,
        compress_ratios=(0, 0),
        load_draft=False,
    )
    # a seed where the plain expand rounds some elements differently from the kernel's
    mx.random.seed(0)
    model = Model(args)
    keep_float32 = model.cast_predicate
    leaves = cast(list[tuple[str, mx.array]], tree_flatten(model.parameters()))
    updates: list[tuple[str, mx.array]] = []
    for name, value in leaves:
        if name.endswith(("norm.weight", "layernorm.weight", "q_weight", "k_weight")):
            weight = mx.ones(value.shape)
        elif name.endswith("hc.scale"):
            weight = mx.random.uniform(shape=value.shape) + 0.5
        else:
            weight = mx.random.normal(value.shape) * 0.02
        updates.append(
            (name, weight.astype(mx.bfloat16) if keep_float32(name) else weight)
        )
    model.load_weights(updates)
    first, second = model.model.layers[0], model.model.layers[1]
    assert first is not None and second is not None
    # 16 rows (the most the fused path takes), so the two expands' roundings would differ
    streams = (mx.random.normal((16, 1, 4, 1024)) * 8).astype(mx.bfloat16)
    pre_mix = mx.softmax(mx.random.normal((16, 1, 4)), axis=-1)
    window_idx = mx.broadcast_to(
        _window_indices(previous=0, length=1, window=8), (16, 1, 1)
    )
    inputs = mx.arange(16).reshape(16, 1)
    assert kernels.hc_fusable(streams)

    def run(seam: bool) -> tuple[mx.array, mx.array]:
        caches = [LayerCache(8), LayerCache(8)]
        state, pre, partials = first.forward(
            streams,
            pre_mix,
            inputs=inputs,
            start_pos=0,
            window_idx=window_idx,
            cache=caches[0],
            shared={},
            next_hc=None if seam else second.attn_hc,
            fused_expand=seam,
        )
        assert (partials is None) == seam
        state, pre, _ = second.forward(
            state,
            pre,
            inputs=inputs,
            start_pos=0,
            window_idx=window_idx,
            cache=caches[1],
            shared={},
            partials=partials,
        )
        return state, pre

    one_device, seam = run(False), run(True)
    for got, want in zip(seam, one_device, strict=True):
        assert bool(mx.array_equal(got, want))
