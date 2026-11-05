# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false
"""Qwen3.8-Flash-Next's inference path (qwen4_exp_kernels.py and
Model.pack_for_inference): each fused kernel against the MLX ops it replaces, and the
packed model against the plain one. Both must agree bit for bit."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_map
from mlx_lm.models.activations import swiglu
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import compute_g, normalize_qk

from exo.worker.engines.mlx.models.qwen3 import qwen4_exp_kernels as kernels
from exo.worker.engines.mlx.models.qwen3.qwen4_exp import (
    GatedDeltaNet,
    GatedResidual,
    Model,
    ModelArgs,
    TextArgs,
    ZeroCenteredRMSNorm,
    add_to_streams,
)
from exo.worker.engines.mlx.models.qwen3.tests.test_qwen4_exp import (
    EOS,
    PLE_PREFIX,
    _write_table,  # pyright: ignore[reportPrivateUsage]
)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the fused kernels are Metal kernels"
)

BF16 = mx.bfloat16

# Dimensions the kernels take (multiples of their thread splits), at a small scale
PACKABLE: dict[str, object] = {
    "model_type": "qwen4_exp_text",
    "vocab_size": 256,
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 128,
    "layer_types": [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ],
    "num_experts": 32,
    "num_experts_per_tok": 4,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64,
    "linear_num_key_heads": 1,
    "linear_num_value_heads": 2,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "hc_count": 4,
    "hc_lowrank": 64,
    "indexer_n_heads": 4,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 32,
    "indexer_budget": 16,
    "indexer_compress_ratio": 4,
    "ngram_size": 3,
    "heads_per_ngram": 2,
    "ngram_vocab_size_base": 1000,
    "ple_embed_dim": 128,
    "ple_layer_ids": [2],
    "eos_token_id": EOS,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
    },
}


def _same(a: mx.array, b: mx.array) -> bool:
    return bool(
        np.array_equal(np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32)))
    )


def _quantized_residual(dims: int) -> GatedResidual:
    args = TextArgs.from_dict({**PACKABLE, "hidden_size": dims})
    hyper = GatedResidual(args)
    hyper.set_dtype(BF16)
    hyper.hc_norm.weight = (mx.random.normal((4 * dims,)) * 0.1).astype(BF16)
    for linear in (
        hyper.input_mix_weight_down,
        hyper.input_mix_weight_up,
        hyper.block_inject_weight,
    ):
        assert linear is not None
        linear.weight = (mx.random.normal(linear.weight.shape) * 0.05).astype(BF16)
    nn.quantize(hyper, group_size=32, bits=4)
    mx.eval(hyper.parameters())
    return hyper


@pytest.mark.parametrize("tokens", [1, 3])
def test_hyper_connection_kernels_match_the_mlx_ops(tokens: int) -> None:
    mx.random.seed(0)
    dims = 128
    hyper = _quantized_residual(dims)
    streams = (mx.random.normal((1, tokens, 4 * dims)) * 2).astype(BF16)
    block_out = mx.random.normal((1, tokens, dims)).astype(BF16)

    normed = hyper.hc_norm(streams)
    assert _same(kernels.hc_norm(streams, hyper.hc_norm.weight, 1e-6, 4), normed)

    mixed, inject = hyper.mix(streams)
    assert inject is not None
    assert hyper.pack()
    packed_mixed, projection, offset = hyper.packed_mix(hyper.packed_norm(streams))
    assert _same(packed_mixed, mixed)

    updated = add_to_streams(streams, block_out, inject)
    new_streams, new_normed = kernels.hc_add_norm(
        streams, block_out, projection, offset, 4, hyper.hc_norm.weight, 1e-6
    )
    assert new_normed is not None
    assert _same(new_streams, updated)
    assert _same(new_normed, hyper.hc_norm(updated))
    only, none = kernels.hc_add_norm(
        streams, block_out, projection, offset, 4, None, 0.0
    )
    assert none is None and _same(only, updated)


@pytest.mark.parametrize(("tokens", "dims"), [(1, 256), (3, 128), (40, 256)])
def test_attention_kernels_match_the_mlx_ops(tokens: int, dims: int) -> None:
    mx.random.seed(5)
    heads = 3
    norm = ZeroCenteredRMSNorm(dims, 1e-6)
    norm.weight = (mx.random.normal((dims,)) * 0.1).astype(BF16)
    # queries with each head's gate after them, as the query projection lays them out
    projection = (mx.random.normal((1, tokens, heads * 2 * dims)) * 2).astype(BF16)
    queries, gate = mx.split(projection.reshape(1, tokens, heads, -1), 2, axis=-1)
    got = kernels.head_norm(projection, norm.weight, 1e-6, heads, 2 * dims)
    assert _same(got, norm(queries))

    attention = mx.random.normal((1, heads, tokens, dims)).astype(BF16)
    want = attention.transpose(0, 2, 1, 3).reshape(1, tokens, -1) * mx.sigmoid(
        gate.reshape(1, tokens, -1)
    )
    assert _same(kernels.gate_heads(attention, projection), want)


@pytest.mark.parametrize(("tokens", "top_k"), [(1, 4), (5, 10), (70, 10)])
def test_router_and_combine_match_the_mlx_ops(tokens: int, top_k: int) -> None:
    mx.random.seed(1)
    experts = 64
    logits = (mx.random.normal((1, tokens, experts + 1)) * 2).astype(BF16)
    probs = mx.softmax(logits[..., :experts].astype(mx.float32), axis=-1)
    reference = mx.argpartition(-probs, kth=top_k - 1, axis=-1)[..., :top_k]
    scores = mx.take_along_axis(probs, reference, axis=-1)
    scores = (scores / scores.sum(axis=-1, keepdims=True)).astype(BF16)

    indices, weights, rows = kernels.moe_route(logits, experts, top_k, True)
    np.testing.assert_array_equal(
        np.array(rows),
        np.concatenate([np.array(indices), np.array(indices) + experts + 1], axis=-1),
    )
    got = np.array(indices[..., :top_k])
    want = np.array(reference)
    np.testing.assert_array_equal(np.sort(got, axis=-1), np.sort(want, axis=-1))
    got_weights = np.take_along_axis(
        np.array(weights[..., :top_k].astype(mx.float32)), np.argsort(got, -1), -1
    )
    want_weights = np.take_along_axis(
        np.array(scores.astype(mx.float32)), np.argsort(want, -1), -1
    )
    np.testing.assert_array_equal(got_weights, want_weights)
    assert np.all(np.array(indices[..., top_k]) == experts)
    assert _same(weights[..., top_k], mx.sigmoid(logits[..., experts]))

    outputs = mx.random.normal((1, tokens, top_k + 1, 128)).astype(BF16)
    routed = (outputs[..., :top_k, :] * weights[..., :top_k, None]).sum(axis=-2)
    shared = weights[..., top_k:] * outputs[..., top_k, :]
    assert _same(kernels.moe_combine(outputs, weights), routed + shared)


def test_router_normalises_in_the_order_mlx_softmax_sums() -> None:
    """512 experts, top 10: rows on which a normaliser summed in another order than
    MLX's softmax gives a different bf16 routing weight."""
    experts, top_k = 512, 10
    noise = mx.random.normal((1, 20000, experts + 1), key=mx.random.key(0))
    for scale, row in ((1.0, 17849), (2.0, 9791), (2.0, 11087), (4.0, 19953)):
        logits = (noise[:, row : row + 1] * scale).astype(BF16)
        probs = mx.softmax(logits[..., :experts].astype(mx.float32), axis=-1)
        reference = mx.argpartition(-probs, kth=top_k - 1, axis=-1)[..., :top_k]
        scores = mx.take_along_axis(probs, reference, axis=-1)
        scores = (scores / scores.sum(axis=-1, keepdims=True)).astype(BF16)
        indices, weights, _ = kernels.moe_route(logits, experts, top_k, True)
        assert _same(indices[..., :top_k], reference.astype(mx.uint32))
        assert _same(weights[..., :top_k], scores)


@pytest.mark.parametrize(("tokens", "inner"), [(1, 640), (3, 64)])
def test_expert_down_kernel_matches_the_mlx_ops(tokens: int, inner: int) -> None:
    mx.random.seed(6)
    experts, top_k, dims = 16, 5, 256
    weight = (mx.random.normal((experts * dims, inner)) * 0.05).astype(BF16)
    down_w, down_s, down_b = (
        part.reshape(experts, dims, -1) for part in mx.quantize(weight, 32, 4)
    )
    gate_up = mx.random.normal((1, tokens, 2 * top_k, inner)).astype(BF16)
    indices = mx.array(
        np.random.default_rng(0)
        .integers(0, experts, (1, tokens, top_k))
        .astype(np.uint32)
    )
    gate, up = gate_up[..., :top_k, :], gate_up[..., top_k:, :]
    want = mx.gather_qmm(
        swiglu(gate, up)[..., None, :],
        down_w,
        down_s,
        down_b,
        rhs_indices=indices,
        transpose=True,
        group_size=32,
        bits=4,
    ).squeeze(-2)
    assert _same(kernels.moe_down(gate_up, indices, down_w, down_s, down_b, 32), want)


@pytest.mark.parametrize("tokens", [1, 4])
def test_gated_deltanet_kernels_match_the_mlx_ops(tokens: int) -> None:
    mx.random.seed(2)
    args = TextArgs.from_dict(PACKABLE)
    net = GatedDeltaNet(args)
    net.set_dtype(BF16)
    net.A_log = mx.log(mx.random.uniform(1, 16, (2,))).astype(mx.float32)
    net.dt_bias = mx.random.normal((2,)).astype(BF16)
    net.conv1d.weight = (mx.random.normal(net.conv1d.weight.shape) * 0.3).astype(BF16)
    net.norm.weight = (1 + 0.1 * mx.random.normal((128,))).astype(BF16)
    heads_k, heads_v, dim = 1, 2, 128
    conv_dim = 2 * heads_k * dim + heads_v * dim
    width = conv_dim + heads_v * dim + 2 * heads_v
    projection = mx.random.normal((1, tokens, width)).astype(BF16)
    state = mx.random.normal((1, 3, conv_dim)).astype(BF16)

    qkv = projection[..., :conv_dim]
    b = projection[..., conv_dim + heads_v * dim : conv_dim + heads_v * dim + heads_v]
    a = projection[..., conv_dim + heads_v * dim + heads_v :]
    conv_input = mx.concatenate([state, qkv], axis=1)
    conv_out = nn.silu(net.conv1d(conv_input))
    q, k, v = (
        part.reshape(1, tokens, heads, dim)
        for part, heads in zip(
            mx.split(conv_out, [dim * heads_k, 2 * dim * heads_k], -1),
            (heads_k, heads_k, heads_v),
            strict=True,
        )
    )
    q, k = normalize_qk(q, k, inv_scale=dim**-0.5, eps=1e-6)
    got = kernels.gdn_prepare(
        projection,
        state,
        net.conv1d.weight.reshape(conv_dim, -1),
        net.A_log,
        net.dt_bias,
        heads_k,
        heads_v,
        dim,
        dim,
    )
    want = (
        q,
        k,
        v,
        compute_g(net.A_log, a, net.dt_bias),
        mx.sigmoid(b),
        conv_input[:, -3:],
    )
    for got_part, want_part in zip(got, want, strict=True):
        assert _same(got_part, want_part)

    out = mx.random.normal((1, tokens, heads_v, dim)).astype(BF16)
    z = projection[..., conv_dim : conv_dim + heads_v * dim]
    gated = net.norm(out, z.reshape(1, tokens, heads_v, dim)).reshape(1, tokens, -1)
    assert _same(
        kernels.gdn_gated_norm(out, projection, conv_dim, net.norm.weight, 1e-6), gated
    )


def _packable_model(tmp_path: Path) -> Model:
    mx.random.seed(3)
    model = Model(
        ModelArgs.from_dict({"model_type": "qwen4_exp", "text_config": PACKABLE})
    )
    model.update(
        tree_map(
            lambda p: mx.random.normal(p.shape) * 0.1 if p.dtype == mx.float32 else p,
            model.parameters(),
        )
    )
    model.set_dtype(BF16, lambda dtype: mx.issubdtype(dtype, mx.floating))
    model.language_model.model.layers[0].linear_attn.A_log = mx.zeros((2,))

    def quantize_as_the_checkpoint(
        path: str, module: nn.Module
    ) -> bool | dict[str, int]:
        if not hasattr(module, "to_quantized"):
            return False
        if path.endswith(("mlp.gate", "mlp.shared_expert_gate")):
            return {"group_size": 64, "bits": 8}
        return True

    nn.quantize(
        model, group_size=32, bits=4, class_predicate=quantize_as_the_checkpoint
    )
    model.eval()  # as mlx-lm's load_model leaves it
    mx.eval(model.parameters())
    ple = model.layers[1].ple
    assert ple is not None
    rows = ple.ple_embedding.hasher.rows
    table = mx.random.normal((rows, ple.ple_embedding.width)) * 0.5
    _write_table(tmp_path, table, 2, quantized=True, prefix=PLE_PREFIX)
    model.attach_ngram_table(tmp_path, prewarm=False)
    return model


def test_packed_model_matches_the_plain_one(tmp_path: Path) -> None:
    plain = _packable_model(tmp_path)
    packed = _packable_model(tmp_path)
    packed.pack_for_inference()
    layer = packed.layers[0]
    assert layer.packed and layer.linear_attn.in_proj is not None
    assert packed.language_model.model.defer_ngram_reads

    rng = np.random.default_rng(4)
    tokens = rng.integers(0, 240, size=(1, 60))
    tokens[0, 31] = EOS
    caches = (plain.make_cache(), packed.make_cache())
    # a prefill chunk past the indexer budget, then decode steps of 1 and 3 tokens
    start = 0
    for length in (40, 1, 1, 3, 1, 1, 9, 1):
        chunk = mx.array(tokens[:, start : start + length])
        want = plain(chunk, caches[0])
        got = packed(chunk, caches[1])
        assert _same(got, want), (start, length)
        start += length
    for plain_cache, packed_cache in zip(*caches, strict=True):
        if isinstance(plain_cache, ArraysCache):
            assert isinstance(packed_cache, ArraysCache)
            for want_state, got_state in zip(
                plain_cache.cache, packed_cache.cache, strict=True
            ):
                assert want_state is not None and got_state is not None
                assert _same(got_state, want_state)


@pytest.mark.parametrize(
    "length", [1, 2, 3, 5, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128]
)
def test_packed_model_matches_the_plain_one_across_kernel_thresholds(
    tmp_path: Path, length: int
) -> None:
    """Chunk lengths around MLX's kernel choices: qmv for one row, qmv_wide for a few,
    and the sorted expert gather whose kernel depends on rows per expert (4 experts per
    token over 32 experts: 32 tokens). Then decode steps of 1 to 9 tokens."""
    plain = _packable_model(tmp_path)
    packed = _packable_model(tmp_path)
    packed.pack_for_inference()
    rng = np.random.default_rng(length)
    tokens = mx.array(rng.integers(0, 240, size=(1, length + 45)))
    caches = (plain.make_cache(), packed.make_cache())
    start = 0
    for size in (length, *range(1, 10)):
        chunk = tokens[:, start : start + size]
        assert _same(packed(chunk, caches[1]), plain(chunk, caches[0])), (start, size)
        start += size


def test_packing_can_be_turned_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EXO_QWEN4_EXP_PACK", "0")
    model = _packable_model(tmp_path)
    model.pack_for_inference()
    assert not model.layers[0].packed
    assert not model.language_model.model.defer_ngram_reads
