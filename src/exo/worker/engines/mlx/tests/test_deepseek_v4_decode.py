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
import mlx.nn as nn
import numpy as np
import pytest
from mlx.nn.layers.distributed import shard_inplace
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models import deepseek_v4 as dv4
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

from exo.worker.engines.mlx.auto_parallel import (
    _shard_v4_attention_groups,
)
from exo.worker.engines.mlx.patches import deepseek_v4_decode_kernels as kernels
from exo.worker.engines.mlx.patches import deepseek_v4_indexer as indexer_patch
from exo.worker.engines.mlx.patches import deepseek_v4_moe_gate as gate_patch
from exo.worker.engines.mlx.patches import switch_lhs_indices

_VOCAB_SIZE = 512
_SLIDING_WINDOW = 16
_CONFIG: dict[str, object] = {
    "model_type": "deepseek_v4",
    "vocab_size": _VOCAB_SIZE,
    "hidden_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "q_lora_rank": 128,
    "o_lora_rank": 64,
    "o_groups": 4,
    "head_dim": 128,
    "qk_rope_head_dim": 64,
    "sliding_window": _SLIDING_WINDOW,
    "compress_ratios": [0, 4, 128, 4],
    "index_n_heads": 4,
    "index_head_dim": 64,
    "index_topk": 64,
    "compress_rope_theta": 160000.0,
    "moe_intermediate_size": 64,
    "n_routed_experts": 32,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "num_hash_layers": 1,
    "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc",
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.5,
    "swiglu_limit": 10.0,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "num_nextn_predict_layers": 0,
    "max_position_embeddings": 4096,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "rms_norm_eps": 1e-6,
}
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


def _tiny_model(index_topk: int = 64) -> dv4.Model:
    mx.random.seed(0)
    args = dv4.ModelArgs.from_dict({**_CONFIG, "index_topk": index_topk})
    model = dv4.Model(args)
    nn.quantize(
        model,
        group_size=64,
        bits=8,
        class_predicate=lambda _, module: type(module) is nn.Linear,
    )
    keep_fp32 = model.cast_predicate
    params = []
    for path, leaf in tree_flatten(model.parameters()):
        value = cast(mx.array, leaf)
        if path.endswith("tid2eid"):
            value = mx.random.randint(0, args.n_routed_experts, value.shape)
        elif mx.issubdtype(value.dtype, mx.floating):
            value = mx.random.normal(value.shape) * 0.05
            if path.endswith("norm.weight"):
                value = value + 1.0
            if keep_fp32(path):
                value = value.astype(mx.float32)
            else:
                value = value.astype(mx.bfloat16)
        params.append((path, value))
    model.update(tree_unflatten(params))
    for _, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear) and module.mode == "affine":
            out_dims = module.weight.shape[0]
            in_dims = module.scales.shape[1] * module.group_size
            weight = mx.random.normal((out_dims, in_dims)) * 0.05
            quantized = mx.quantize(weight.astype(mx.bfloat16), group_size=64, bits=8)
            module.weight, module.scales, module.biases = quantized
        elif isinstance(module, QuantizedSwitchLinear):
            experts, out_dims, _ = module.weight.shape
            in_dims = module.scales.shape[-1] * module.group_size
            weight = mx.random.normal((experts, out_dims, in_dims)) * 0.05
            quantized = mx.quantize(weight, group_size=32, bits=4, mode="mxfp4")
            module.weight, module.scales = quantized[0], quantized[1]
    mx.eval(model.parameters())
    return model


def _tokens(length: int, seed: int) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.integers(0, _VOCAB_SIZE, (1, length)))


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
    fused = kernels.norm_rope(x, weight, 1e-6, 300.0, freqs, 64)
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
    assert mx.array_equal(kernels.norm_rope(x, None, None, -77.0, freqs, 64), reference)


@pytest.mark.parametrize("keys", [1, 130, 640])
def test_decode_attention_matches_float32_sdpa(keys: int) -> None:
    mx.random.seed(5)
    q = mx.random.normal((1, 16, 1, 512)).astype(mx.bfloat16)
    kv = mx.random.normal((1, keys, 512)).astype(mx.bfloat16)
    sinks = mx.random.normal((16,)).astype(mx.bfloat16)
    reference = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32),
        kv[:, None].astype(mx.float32),
        kv[:, None].astype(mx.float32),
        scale=512**-0.5,
        sinks=sinks.astype(mx.float32),
    )
    fused = kernels.decode_attention(q, kv, sinks, 512**-0.5)
    assert _relative_error(reference, fused) < 4e-3


def test_patched_decode_matches_mlx_lm(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny_model()
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


@pytest.mark.parametrize("n_routed", [64, 256])
def test_parallel_gate_matches_mlx_lm(n_routed: int) -> None:
    mx.random.seed(6)
    args = dv4.ModelArgs.from_dict(
        {**_CONFIG, "n_routed_experts": n_routed, "num_experts_per_tok": 6}
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
    model = _tiny_model(index_topk=index_topk)
    attention = model.model.layers[1].attn
    assert attention.compress_ratio == 4
    mx.random.seed(7)
    prompt = (mx.random.normal((1, _PROMPT_LENGTH, 256)) * 0.5).astype(mx.bfloat16)
    steps = [
        (mx.random.normal((1, 1, 256)) * 0.5).astype(mx.bfloat16)
        for _ in range(_DECODE_STEPS)
    ]

    def run() -> mx.array:
        cache = dv4.DeepseekV4Cache(_SLIDING_WINDOW)
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
    model = _tiny_model()
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
