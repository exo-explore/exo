"""The fused Laguna forward pass (fast.py, kernels.py) against mlx-lm's, bit for bit.

Each fused kernel is checked against the MLX ops it replaces, and whole models (bf16, 4-bit
affine and NVFP4) are checked end to end: chunked prefill longer than the sliding window
and the windowed prefill block, decode steps, and batched decode with left padding.
"""

import copy
from collections.abc import Callable
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models import base as mlx_base
from mlx_lm.models.cache import (
    BatchKVCache,
    BatchRotatingKVCache,
    KVCache,
    RotatingKVCache,
)
from mlx_lm.models.laguna import Model, ModelArgs, MoE

from exo.worker.engines.mlx.models.laguna import fast, kernels

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the fused kernels are Metal kernels"
)

_LAYER_TYPES = [
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
]
_N_LAYERS = len(_LAYER_TYPES)


def _args(top_k: int = 4) -> ModelArgs:
    return ModelArgs(
        model_type="laguna",
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=_N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        num_attention_heads_per_layer=[4, 6, 6, 4, 6],
        layer_types=_LAYER_TYPES,
        mlp_layer_types=["dense"] + ["sparse"] * (_N_LAYERS - 1),
        sliding_window=64,
        gating="per-head",
        rope_parameters={
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500000.0,
                "factor": 32.0,
                "original_max_position_embeddings": 8192,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            },
        },
        num_experts=16,
        num_experts_per_tok=top_k,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        moe_routed_scaling_factor=2.5,
        rms_norm_eps=1e-6,
    )


def _to_bfloat16(module: nn.Module) -> None:
    module.set_dtype(mx.bfloat16)  # pyright: ignore[reportUnknownMemberType]


def _model(quantization: str | None, top_k: int = 4) -> Model:
    mx.random.seed(0)
    model = Model(_args(top_k))
    for layer in model.layers[1:]:
        gate = cast(MoE, layer.mlp).gate
        gate.e_score_correction_bias = mx.random.uniform(
            low=-0.3, high=0.3, shape=(16,)
        )
    _to_bfloat16(model)
    if quantization == "affine":
        nn.quantize(model, group_size=64, bits=4)
    elif quantization == "nvfp4":
        nn.quantize(model, group_size=16, bits=4, mode="nvfp4")
    mx.eval(model.parameters())
    return model


def _fused(model: Model) -> Model:
    fused = copy.deepcopy(model)
    fast.fuse_for_inference(fused, cast(list[nn.Module], fused.layers))
    return fused


def _assert_same(a: mx.array, b: mx.array) -> None:
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert mx.array_equal(a, b).item()


@pytest.mark.parametrize("quantization", [None, "affine", "nvfp4"])
def test_fused_model_is_bit_identical(quantization: str | None) -> None:
    model = _model(quantization)
    fused = _fused(model)
    assert all(isinstance(layer, fast.FastTransformerBlock) for layer in fused.layers)
    moe = cast(object, fused.layers[1].mlp)
    assert isinstance(moe, fast.FastMoE)
    assert moe.shared_stacked == (quantization is not None)
    tokens = [(7 * i + 3) % 256 for i in range(700)]
    outputs: list[list[mx.array]] = []
    for candidate in (model, fused):
        cache = candidate.make_cache()
        logits: list[mx.array] = []
        # Chunks: longer than the sliding window (windowed prefill), a single token, short,
        # and 16 and 17 tokens: with 16 experts and top-4 routing, 16 tokens is where MLX
        # switches its gathered-matmul kernel (at 4 tokens per expert), which must not move
        # when the shared expert is stacked as a 17th.
        for start, end in (
            (0, 600),
            (600, 601),
            (601, 640),
            (640, 656),
            (656, 673),
            (673, 700),
        ):
            logits.append(candidate(mx.array([tokens[start:end]]), cache=cache))
        for step in range(4):
            logits.append(candidate(mx.array([[step + 5]]), cache=cache))
        mx.eval(*logits)
        outputs.append(logits)
    for a, b in zip(outputs[0], outputs[1], strict=True):
        _assert_same(a, b)


@pytest.mark.parametrize("quantization", [None, "nvfp4"])
def test_fused_model_is_bit_identical_in_a_padded_batch(
    quantization: str | None,
) -> None:
    model = _model(quantization)
    fused = _fused(model)
    pads = [0, 3, 9]
    outputs: list[list[mx.array]] = []
    for candidate in (model, fused):
        cache: list[Any] = [
            BatchRotatingKVCache(64, pads) if layer.use_sliding else BatchKVCache(pads)
            for layer in candidate.layers
        ]
        logits = [candidate(mx.array([[1 + i for i in range(80)]] * 3), cache=cache)]
        for step in range(5):
            logits.append(candidate(mx.array([[step + 2]] * 3), cache=cache))
        mx.eval(*logits)
        outputs.append(logits)
    for a, b in zip(outputs[0], outputs[1], strict=True):
        _assert_same(a, b)


def test_fused_linear_matches_separate_layers_at_any_row_count() -> None:
    mx.random.seed(1)
    for quantization in (None, "affine"):
        layers: list[nn.Module] = [
            nn.Linear(512, n, bias=False) for n in (1024, 256, 256)
        ]
        for layer in layers:
            _to_bfloat16(layer)
        if quantization is not None:
            layers = [nn.QuantizedLinear.from_linear(layer) for layer in layers]
        fused = fast.FusedLinear(layers)
        for rows in (1, 2, 5, 8, 9, 16, 33, 200):
            x = mx.random.normal((1, rows, 512)).astype(mx.bfloat16)
            separate = [layer(x) for layer in layers]
            if fused.use_fused(x):
                out = fused(x)
                pieces = mx.split(out, [1024, 1280], axis=-1)
            else:
                pieces = fused.parts(x)
            for a, b in zip(separate, pieces, strict=True):
                _assert_same(a, b)


@pytest.mark.parametrize("yarn", [True, False])
def test_qk_norm_rope_matches_mlx(yarn: bool) -> None:
    model = _model(None)
    layer = model.layers[0 if yarn else 1]
    attention = layer.self_attn
    fast_attention = fast.FastAttention(attention)
    mx.random.seed(2)
    for seq_len, offset in ((1, 0), (1, 37), (9, 100), (300, 5)):
        x = mx.random.normal((2, seq_len, 256)).astype(mx.bfloat16)
        q = attention.q_proj(x).reshape(2, seq_len, attention.n_heads, -1)
        k = attention.k_proj(x).reshape(2, seq_len, attention.n_kv_heads, -1)
        v = attention.v_proj(x).reshape(2, seq_len, attention.n_kv_heads, -1)
        offsets = mx.array([offset, offset + 11], dtype=mx.int32)
        q_ref = attention.rope(
            attention.q_norm(q).transpose(0, 2, 1, 3), offset=offsets
        )
        k_ref = attention.rope(
            attention.k_norm(k).transpose(0, 2, 1, 3), offset=offsets
        )
        queries, keys, values = kernels.qk_norm_rope(
            (
                q.reshape(2, seq_len, -1),
                k.reshape(2, seq_len, -1),
                v.reshape(2, seq_len, -1),
            ),
            (0, 0, 0),
            attention.q_norm.weight,
            attention.k_norm.weight,
            n_q_heads=attention.n_heads,
            n_kv_heads=attention.n_kv_heads,
            head_dim=128,
            eps=fast_attention.eps,
            rotary_dims=fast_attention.rotary_dims,
            freqs=fast_attention._freqs,  # pyright: ignore[reportPrivateUsage]
            base=fast_attention._base,  # pyright: ignore[reportPrivateUsage]
            magnitude_scale=fast_attention._magnitude_scale,  # pyright: ignore[reportPrivateUsage]
            offsets=offsets,
        )
        _assert_same(q_ref, queries)
        _assert_same(k_ref, keys)
        _assert_same(mx.contiguous(v.transpose(0, 2, 1, 3)), values)


@pytest.mark.parametrize("top_k", [4, 8, 10])
def test_route_matches_mlx_including_ties(top_k: int) -> None:
    args = _args(top_k)
    args.num_experts = 64
    moe = MoE(args)
    mx.random.seed(3)
    moe.gate.e_score_correction_bias = mx.random.uniform(
        low=-0.3, high=0.3, shape=(64,)
    )
    x = mx.random.normal((3, 7, 256)).astype(mx.bfloat16)
    logits = moe.gate.proj(x)
    # Ties: whole rows of equal logits, and repeated values.
    logits = mx.concatenate(
        [logits, mx.zeros_like(logits[:, :1]), mx.round(logits[:, :2] * 2) / 2], axis=1
    )
    reference = moe.gate.__class__.__call__
    del reference
    scores = mx.sigmoid(logits.astype(mx.float32))
    selection = scores + moe.gate.e_score_correction_bias
    indices_ref = mx.argpartition(-selection, kth=top_k - 1, axis=-1)[..., :top_k]
    weights_ref = mx.take_along_axis(scores, indices_ref, axis=-1)
    weights_ref = weights_ref / weights_ref.sum(axis=-1, keepdims=True)
    indices, weights = kernels.route(logits, moe.gate.e_score_correction_bias, top_k)
    _assert_same(indices_ref.astype(mx.uint32), indices)
    _assert_same(weights_ref, weights)
    extended, weights = kernels.route(
        logits, moe.gate.e_score_correction_bias, top_k, extra_index=64
    )
    _assert_same(indices_ref.astype(mx.uint32), extended[..., :top_k])
    assert mx.all(mx.equal(extended[..., top_k], 64)).item()
    _assert_same(weights_ref, weights)


@pytest.mark.parametrize("top_k", [4, 8, 10])
def test_combine_matches_mlx(top_k: int) -> None:
    mx.random.seed(4)
    y = mx.random.normal((2, 5, top_k, 512)).astype(mx.bfloat16)
    w = mx.random.uniform(shape=(2, 5, top_k))
    shared = mx.random.normal((2, 5, 512)).astype(mx.bfloat16)
    residual = mx.random.normal((2, 5, 512)).astype(mx.bfloat16)
    routed = (y * w[..., None]).sum(axis=-2).astype(mx.bfloat16) * 2.5 + shared
    _assert_same(routed, kernels.combine_experts(y, w, shared, 2.5, None)[0])
    _assert_same(
        residual + routed, kernels.combine_experts(y, w, shared, 2.5, residual)[0]
    )
    # The shared expert stacked after the routed ones, and the next layer's norm fused.
    stacked = mx.concatenate([y, shared[..., None, :]], axis=-2)
    norm_weight = (mx.random.normal((512,)) * 0.1 + 1).astype(mx.bfloat16)
    out, normed = kernels.combine_experts(
        stacked, w, None, 2.5, residual, (norm_weight, 1e-6)
    )
    _assert_same(residual + routed, out)
    assert normed is not None
    _assert_same(mx.fast.rms_norm(residual + routed, norm_weight, 1e-6), normed)


@pytest.mark.parametrize("dim", [256, 2048, 3072])
def test_add_rms_norm_matches_mlx(dim: int) -> None:
    mx.random.seed(5)
    x = mx.random.normal((3, 4, dim)).astype(mx.bfloat16)
    r = mx.random.normal((3, 4, dim)).astype(mx.bfloat16)
    weight = (mx.random.normal((dim,)) * 0.1 + 1).astype(mx.bfloat16)
    h, normed = kernels.add_rms_norm(x, r, weight, 1e-6)
    _assert_same(x + r, h)
    _assert_same(mx.fast.rms_norm(x + r, weight, 1e-6), normed)


def _update(
    cache: RotatingKVCache, keys: mx.array, values: mx.array
) -> tuple[mx.array, mx.array]:
    update = cast(
        Callable[[mx.array, mx.array], tuple[mx.array, mx.array]],
        getattr(cache, "update_and_fetch"),  # noqa: B009
    )
    return update(keys, values)


def test_windowed_prefill_attention_matches_mlx() -> None:
    mx.random.seed(6)
    for prefix in (0, 63, 200):
        cache = RotatingKVCache(max_size=64)
        if prefix:
            keys = mx.random.normal((1, 2, prefix, 128)).astype(mx.bfloat16)
            _ = _update(cache, keys, keys)
        n = 900
        q = mx.random.normal((1, 4, n, 128)).astype(mx.bfloat16)
        k = mx.random.normal((1, 2, n, 128)).astype(mx.bfloat16)
        create_attention_mask = cast(
            Callable[..., object],
            getattr(mlx_base, "create_attention_mask"),  # noqa: B009
        )
        mask = create_attention_mask(q.transpose(0, 2, 1, 3), cache, window_size=64)
        assert isinstance(mask, mx.array)
        keys, values = _update(cache, k, k * 0.5)
        reference = mx.fast.scaled_dot_product_attention(
            q, keys, values, scale=0.088, mask=mask
        )
        windowed = fast._sliding_window_attention(  # pyright: ignore[reportPrivateUsage]
            q, keys, values, mask, 0.088, 64
        )
        _assert_same(reference, windowed)


def test_layers_without_a_supported_shape_keep_mlx_lm_code() -> None:
    """head_dim 32 has no fused attention kernel: the block keeps mlx-lm's attention and
    still matches exactly."""
    args = _args()
    args.head_dim = 32
    mx.random.seed(0)
    model = Model(args)
    _to_bfloat16(model)
    mx.eval(model.parameters())
    fused = _fused(model)
    assert not isinstance(fused.layers[1].self_attn, fast.FastAttention)
    assert isinstance(fused.layers[1].mlp, fast.FastMoE)
    tokens = mx.array([[(5 * i) % 256 for i in range(90)]])
    a = model(tokens, cache=[KVCache() for _ in range(_N_LAYERS)])
    b = fused(tokens, cache=[KVCache() for _ in range(_N_LAYERS)])
    _assert_same(a, b)
