"""GLM-5.3-Flash's fused paths (glm5_next.py) against the plain MLX ops they replace, on a
small random model with the dimensions the Metal kernels need (KDA head_dim 128, an indexer
of 32 heads, a hidden size divisible by 256) in bfloat16:

- the fused projections give bitwise the separate layers' outputs;
- the KDA decode kernel, the hyper-connection kernels and the indexer's pool-score kernel
  agree with the ops to float32 rounding (bf16 outputs within a unit in the last place);
- the model with every kernel matches itself across chunked prefill and decode, and a
  layer called alone (as pipeline-parallel wrappers do) computes bitwise what the fused
  layer-to-layer path computes.

Without Metal the kernels don't exist and the model runs the ops: the tests are skipped."""

from collections.abc import Callable
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import numpy.typing as npt
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.cache import ArraysCache, BatchKVCache, CacheList, KVCache

from exo.worker.engines.mlx.models.glm import glm5_next
from exo.worker.engines.mlx.models.glm.glm5_next import (
    DecoderLayer,
    Glm5NextAttention,
    KimiDeltaAttention,
    Model,
    ModelArgs,
    MoE,
)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the fused kernels are Metal kernels"
)

TEXT_CONFIG: dict[str, Any] = {
    "model_type": "glm5_next_text",
    "vocab_size": 128,
    "hidden_size": 256,
    "num_hidden_layers": 5,
    "rms_norm_eps": 1e-5,
    "layer_types": [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "deepseek_sparse_attention",
        "linear_attention",
    ],
    "mlp_layer_types": ["dense", "dense", "sparse", "sparse", "sparse"],
    "indexer_types": ["full"] * 5,
    "linear_attn_config": {
        "num_heads": 2,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
    },
    "intermediate_size": 192,
    "moe_intermediate_size": 64,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "routed_scaling_factor": 2.5,
    "norm_topk_prob": True,
    "n_group": 1,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "swiglu_limit": 10.0,
    "num_attention_heads": 2,
    "q_lora_rank": 64,
    "kv_lora_rank": 64,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 0,
    "v_head_dim": 32,
    "mla_use_nope": True,
    "index_n_heads": 32,
    "index_head_dim": 32,
    "index_topk": 16,
    "index_kpool": 4,
    "index_kpool_compress": True,
    "index_kpool_always_select_tail": True,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
}
CONFIG: dict[str, Any] = {"model_type": "glm5_next", "text_config": TEXT_CONFIG}
# parameters the model keeps in float32 (sanitize stores the last two in float32 too)
_FLOAT32 = (
    "A_log",
    "dt_bias",
    "e_score_correction_bias",
    "_hc.base",
    "_hc.scale",
    "_hc.fn",
    "mlp.gate.weight",
)


def _model(seed: int = 0) -> Model:
    mx.random.seed(seed)
    model = Model(cast(ModelArgs, ModelArgs.from_dict(CONFIG)))  # pyright: ignore[reportUnknownMemberType]
    params: list[tuple[str, mx.array]] = []
    for name, value in cast(
        list[tuple[str, mx.array]], tree_flatten(model.parameters())
    ):
        scale = 0.02 if name.endswith("_hc.fn") else 0.1
        weight = mx.random.normal(value.shape) * scale
        if name.endswith(("norm.weight", "layernorm.weight")):
            weight = weight + 1
        dtype = mx.float32 if name.endswith(_FLOAT32) else mx.bfloat16
        params.append((name, weight.astype(dtype)))
    model.load_weights(params)
    model.eval()
    mx.eval(model.parameters())
    return model


def _tokens(n: int, seed: int) -> mx.array:
    return mx.array(np.random.default_rng(seed).integers(0, 128, size=(1, n)))


def _f32(a: mx.array) -> npt.NDArray[np.float32]:
    return np.asarray(a.astype(mx.float32), dtype=np.float32)


def _close_bf16(actual: mx.array, expected: mx.array, flips: float = 0.01) -> None:
    """bf16 outputs from float32 math summed in another order: a few values may round to
    the neighbouring bf16 value."""
    a, e = _f32(actual), _f32(expected)
    differ = float(np.mean(np.not_equal(a, e)))
    assert differ <= flips, differ
    np.testing.assert_allclose(a, e, rtol=2**-7, atol=1e-3)


@pytest.mark.parametrize("rows", [1, 3, 16, 17, 40])
@pytest.mark.parametrize(
    ("sizes", "quantized"),
    [
        # the groupings the model fuses: KDA q/k/v and f_a/g_a (bf16), MLA q_a/kv_a and
        # gate/up (quantized)
        ((2048, 2048, 2048), False),
        ((128, 128), False),
        ((1536, 512), True),
        ((2048, 2048), True),
    ],
)
def test_fused_projections_are_the_separate_layers(
    rows: int, sizes: tuple[int, ...], quantized: bool
) -> None:
    mx.random.seed(1)
    layers = [nn.Linear(1024, n, bias=False) for n in sizes]
    for layer in layers:
        layer.weight = (mx.random.normal(layer.weight.shape) * 0.05).astype(mx.bfloat16)
    parts: list[nn.Module] = list(layers)
    if quantized:
        parts = [nn.QuantizedLinear.from_linear(layer, 64, 4) for layer in layers]
    x = (mx.random.normal((1, rows, 1024)) * 0.5).astype(mx.bfloat16)
    expected = [cast(Callable[[mx.array], mx.array], part)(x) for part in parts]
    holder = nn.Module()
    names = tuple("abc"[: len(parts)])
    for name, part in zip(names, parts, strict=True):
        holder[name] = part
    fused = glm5_next._fuse(holder, names)  # pyright: ignore[reportPrivateUsage]
    assert fused is not None and "a" not in holder
    for got, want in zip(fused(x), expected, strict=True):
        assert mx.array_equal(got, want).item()
    (joined,) = fused(x, groups=(len(parts),))
    assert mx.array_equal(joined, mx.concatenate(expected, axis=-1)).item()


def _kda(model: Model) -> KimiDeltaAttention:
    attention = model.layers[0].self_attn
    assert isinstance(attention, KimiDeltaAttention)
    return attention


@pytest.mark.parametrize("batch", [1, 3])
def test_kda_decode_kernel_matches_the_ops(
    batch: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    attention = _kda(_model())
    with_kernel, with_ops = ArraysCache(size=2), ArraysCache(size=2)
    prompt = (mx.random.normal((batch, 9, 256)) * 0.5).astype(mx.bfloat16)
    mx.eval(attention(prompt, with_kernel))
    with_ops[0], with_ops[1] = with_kernel[0], with_kernel[1]
    kernel = glm5_next._kda_decode_kernel  # pyright: ignore[reportPrivateUsage]
    for step in range(3):
        if step == 1 and batch > 1:
            # a left-padded row the cache masks out keeps its state and outputs 0
            for cache in (with_kernel, with_ops):
                cache.left_padding = mx.array([0, 1] + [0] * (batch - 2))
        else:
            for cache in (with_kernel, with_ops):
                cache.left_padding = None
        x = (mx.random.normal((batch, 1, 256)) * 0.5).astype(mx.bfloat16)
        got = attention(x, with_kernel)
        monkeypatch.setattr(glm5_next, "_kda_decode_kernel", None)
        want = attention(x, with_ops)
        monkeypatch.setattr(glm5_next, "_kda_decode_kernel", kernel)
        conv_k, state_k = cast(mx.array, with_kernel[0]), cast(mx.array, with_kernel[1])
        conv_o, state_o = cast(mx.array, with_ops[0]), cast(mx.array, with_ops[1])
        mx.eval(got, want, state_k, state_o)
        _close_bf16(got, want)
        assert mx.array_equal(conv_k, conv_o).item()
        np.testing.assert_allclose(_f32(state_k), _f32(state_o), rtol=1e-5, atol=1e-6)


def test_hyper_connection_kernels_match_the_ops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _model().layers[1]
    x = (mx.random.normal((2, 5, 4, 256))).astype(mx.bfloat16)
    out = (mx.random.normal((2, 5, 256)) * 0.5).astype(mx.bfloat16)
    post = mx.random.uniform(0, 2, (2, 5, 4))
    comb = mx.softmax(mx.random.normal((2, 5, 4, 4)), axis=-1)
    hc, norm = layer.ffn_hc, layer.post_attention_layernorm
    fused = glm5_next._hc_post_then_pre(out, x, post, comb, hc, norm)  # pyright: ignore[reportPrivateUsage]
    pre = glm5_next._hc_pre(x, hc, norm)  # pyright: ignore[reportPrivateUsage]
    post_only = glm5_next._hc_post(out, x, post, comb, hc)  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(glm5_next, "_hc_update_kernel", None)
    fused_ops = glm5_next._hc_post_then_pre(out, x, post, comb, hc, norm)  # pyright: ignore[reportPrivateUsage]
    pre_ops = glm5_next._hc_pre(x, hc, norm)  # pyright: ignore[reportPrivateUsage]
    mx.eval(fused, pre, post_only, fused_ops, pre_ops)
    # the same new streams whichever path writes them
    assert mx.array_equal(fused[0], post_only).item()
    _close_bf16(fused[0], fused_ops[0], flips=1e-3)
    for got, want in ((fused[1], fused_ops[1]), (pre, pre_ops)):
        np.testing.assert_allclose(_f32(got[0]), _f32(want[0]), rtol=1e-4)
        np.testing.assert_allclose(_f32(got[1]), _f32(want[1]), rtol=1e-4, atol=1e-6)
        _close_bf16(got[2], want[2])


@pytest.mark.parametrize("padding", [None, [0, 7]])
@pytest.mark.parametrize("length", [1, 3])
def test_pool_score_kernel_selects_like_the_ops(
    padding: list[int] | None, length: int
) -> None:
    attention = _model().layers[3].self_attn
    assert isinstance(attention, Glm5NextAttention)
    indexer = attention.indexer
    batch = 1 if padding is None else len(padding)
    n = 300
    rng = np.random.default_rng(3)
    x = mx.array(rng.normal(size=(batch, n, 256)).astype(np.float32)).astype(
        mx.bfloat16
    )
    qr = mx.array(rng.normal(size=(batch, n, 64)).astype(np.float32)).astype(
        mx.bfloat16
    )
    cache = BatchKVCache(padding) if padding else KVCache()
    indexer(x[:, : n - length], qr[:, : n - length], cache)
    indexer(x[:, n - length :], qr[:, n - length :], cache)
    assert cache.keys is not None and cache.values is not None
    keys, gates = cache.keys[:, 0, :n], cache.values[:, 0, :n]
    _, _, weights = indexer._x_projections(x[:, n - length :])
    first = None if padding is None else mx.array(padding, dtype=mx.int32)
    q_rows = qr[:, n - length :]
    fast = indexer._select_few(q_rows, keys, gates, weights, first, n)
    slow = indexer._select(x[:, n - length :], q_rows, keys, gates, weights, first, n)
    mx.eval(fast, slow)
    assert fast.shape == slow.shape
    overlaps: list[float] = []
    for b in range(batch):
        for i in range(length):
            got = set(cast(list[int], fast[b, i].tolist())) - {-1}
            want = set(cast(list[int], slow[b, i].tolist())) - {-1}
            assert len(got) == len(want)
            overlaps.append(len(got & want) / len(want))
    # float32 sums in another order can only swap pools whose scores tie to the last bits
    assert min(overlaps) >= 0.95 and float(np.mean(overlaps)) >= 0.99


def _chunked_vs_one_shot(model: Model) -> tuple[float, float]:
    """(mean |logit difference|, argmax agreement) between chunked prefill plus decode and
    one forward over the prompt."""
    prompt = _tokens(40, seed=2)
    full = model(prompt)
    cache = model.make_cache()
    pieces = [model(prompt[:, :13], cache), model(prompt[:, 13:30], cache)]
    for i in range(30, 40):
        pieces.append(model(prompt[:, i : i + 1], cache))
    a, b = _f32(mx.concatenate(pieces, axis=1)), _f32(full)
    top_a = cast(npt.NDArray[np.intp], np.argmax(a, axis=-1))
    top_b = cast(npt.NDArray[np.intp], np.argmax(b, axis=-1))
    agreement = np.equal(top_a, top_b)
    return float(np.mean(np.abs(a - b))), float(np.mean(agreement))


def test_kernels_keep_chunked_prefill_and_decode_consistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In bf16, chunks and single tokens run other kernels than one shot: the kernels may
    not widen that gap beyond what the plain ops show."""
    error, agreement = _chunked_vs_one_shot(_model())
    for name in ("_kda_decode_kernel", "_hc_update_kernel", "_pool_scores_kernel"):
        monkeypatch.setattr(glm5_next, name, None)
    ops_error, ops_agreement = _chunked_vs_one_shot(_model())
    assert error <= 2 * ops_error + 1e-3, (error, ops_error)
    assert agreement >= ops_agreement - 0.05, (agreement, ops_agreement)


def test_a_layer_called_alone_computes_what_the_fused_path_computes() -> None:
    """Pipeline-parallel wrappers call each layer as layer(x, cache): the hyper-connection
    step at a rank boundary runs alone. It must give bitwise what the fused path gives, so
    pipeline parallelism stays bit-identical to one device."""
    model = _model()
    prompt = _tokens(24, seed=5)
    caches = (model.make_cache(), model.make_cache())
    inner = model.model
    for step in (prompt[:, :20], prompt[:, 20:21], prompt[:, 21:22]):
        fused = model(step, caches[0])
        h = mx.repeat(inner.embed_tokens(step)[:, :, None, :], inner.hc_mult, axis=2)
        for layer, cache in zip(inner.layers, caches[1], strict=True):
            assert isinstance(layer, DecoderLayer)
            h = layer(h, cache)
        h = h.astype(mx.float32).mean(axis=2).astype(h.dtype)
        alone = model.lm_head(inner.norm(h))
        mx.eval(fused, alone)
        assert mx.array_equal(fused, alone).item()


@pytest.mark.parametrize("rows", [1, 5, 64])
def test_router_kernel_selects_like_group_expert_select(
    rows: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    mlp = _model().layers[3].mlp
    assert isinstance(mlp, MoE)
    gate = mlp.gate
    gate.e_score_correction_bias = mx.random.normal((8,)) * 0.5
    x = (mx.random.normal((1, rows, 256))).astype(mx.bfloat16)
    inds, scores = gate(x)
    monkeypatch.setattr(glm5_next, "_router_select_kernel", None)
    want_inds, want_scores = gate(x)
    mx.eval(inds, scores, want_inds, want_scores)
    # the same experts with the same weights, possibly listed in another order
    order = mx.argsort(inds, axis=-1)
    want_order = mx.argsort(want_inds, axis=-1)
    assert mx.array_equal(
        mx.take_along_axis(inds, order, axis=-1),
        mx.take_along_axis(want_inds, want_order, axis=-1),
    ).item()
    np.testing.assert_allclose(
        _f32(mx.take_along_axis(scores, order, axis=-1)),
        _f32(mx.take_along_axis(want_scores, want_order, axis=-1)),
        rtol=1e-5,
    )


def _batched_vs_single(model: Model, steps: int = 4) -> float:
    """Max |logit difference| between decoding two left-padded sequences as one batch and
    decoding each alone, over a few steps past the sparse-attention point."""
    sequences = (_tokens(30 + steps, seed=3), _tokens(19 + steps, seed=4))
    singles: list[list[mx.array]] = []
    to_merge: list[list[ArraysCache | CacheList]] = []
    for sequence in sequences:
        n = sequence.shape[1] - steps
        single = model.make_cache()
        model(sequence[:, :n], single)
        singles.append(
            [model(sequence[:, n + i : n + i + 1], single) for i in range(steps)]
        )
        batched = model.make_cache()
        model(sequence[:, :n], batched)
        to_merge.append(batched)
    merged: list[ArraysCache | CacheList] = []
    for a, b in zip(to_merge[0], to_merge[1], strict=True):
        if isinstance(a, ArraysCache):
            assert isinstance(b, ArraysCache)
            merged.append(ArraysCache.merge([a, b]))
        else:
            assert isinstance(b, CacheList)
            merged.append(CacheList.merge([a, b]))
    worst = 0.0
    for i in range(steps):
        tokens = mx.concatenate(
            [
                s[:, s.shape[1] - steps + i : s.shape[1] - steps + i + 1]
                for s in sequences
            ],
            axis=0,
        )
        out = model(tokens, merged)
        for row in range(2):
            diff = mx.abs(
                out[row].astype(mx.float32) - singles[row][i][0].astype(mx.float32)
            )
            worst = max(worst, diff.max().item())
    return worst


def test_kernels_keep_batched_decode_like_single_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The batch generator decodes left-padded sequences together: with the kernels, a
    batch row may not drift from its sequence alone more than with the ops."""
    with_kernels = _batched_vs_single(_model())
    for name in (
        "_kda_decode_kernel",
        "_hc_update_kernel",
        "_pool_scores_kernel",
        "_router_select_kernel",
    ):
        monkeypatch.setattr(glm5_next, name, None)
    with_ops = _batched_vs_single(_model())
    assert with_kernels <= 2 * with_ops + 0.05, (with_kernels, with_ops)


_KERNELS = (
    "_kda_decode_kernel",
    "_hc_update_kernel",
    "_pool_scores_kernel",
    "_router_select_kernel",
)


def _logits_by_chunk_length(lengths: list[int]) -> list[mx.array]:
    model = _model()
    prompt = _tokens(max(lengths), seed=11)
    outs: list[mx.array] = []
    for length in lengths:
        out = model(prompt[:, :length], model.make_cache())
        mx.eval(out)
        outs.append(out)
    return outs


def _logits_by_decode_batch(sizes: list[int]) -> list[mx.array]:
    model = _model()
    outs: list[mx.array] = []
    for size in sizes:
        caches: list[list[ArraysCache | CacheList]] = []
        for row in range(size):
            cache = model.make_cache()
            model(_tokens(20 + 3 * row, seed=20 + row), cache)
            caches.append(cache)
        merged: list[ArraysCache | CacheList] = []
        for layer in range(len(caches[0])):
            parts = [cache[layer] for cache in caches]
            if isinstance(parts[0], ArraysCache):
                merged.append(ArraysCache.merge([cast(ArraysCache, p) for p in parts]))
            else:
                merged.append(CacheList.merge([cast(CacheList, p) for p in parts]))
        for step in range(2):
            out = model(_tokens(size, seed=40 + step).reshape(size, 1), merged)
            mx.eval(out)
            outs.append(out)
    return outs


def test_fused_projections_are_exact_across_kernel_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fused projections and expert gathers change the row counts MLX's matmul,
    quantized-matmul and gather kernels see, and those kernels pick their variant by row
    counts. At chunk lengths around their thresholds, and for decode batches of 1-9 rows,
    the model with fused projections must give bitwise the separate layers' logits (the
    float32-reordering kernels are off on both sides)."""
    for name in _KERNELS:
        monkeypatch.setattr(glm5_next, name, None)
    # 2..8 rows: the dense quantized matmul's wide-row variant on M3-class GPUs; 15-17
    # tokens: this model's gather flip (tokens x top_k / experts >= 4) and the fused
    # matmuls' 16-row limit
    lengths = list(range(1, 10)) + [15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129]
    sizes = list(range(1, 10))
    fused_prefill = _logits_by_chunk_length(lengths)
    fused_decode = _logits_by_decode_batch(sizes)
    monkeypatch.setattr(glm5_next, "_FUSE_PROJECTIONS", False)
    separate_prefill = _logits_by_chunk_length(lengths)
    separate_decode = _logits_by_decode_batch(sizes)
    for length, a, b in zip(lengths, fused_prefill, separate_prefill, strict=True):
        assert mx.array_equal(a, b).item(), f"prefill chunk of {length} tokens"
    for i, (a, b) in enumerate(zip(fused_decode, separate_decode, strict=True)):
        assert mx.array_equal(a, b).item(), f"decode batch of {sizes[i // 2]}"
