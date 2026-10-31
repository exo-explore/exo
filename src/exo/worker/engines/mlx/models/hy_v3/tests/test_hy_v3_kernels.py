# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Hy3's fused kernels and stacked projections against the MLX ops they replace, for bit
equality, at Hy3's shapes and at MLX's kernel-choice thresholds; and the whole fused model
against #2520's code path (the fallback), logits bit for bit."""

import copy
from collections.abc import Callable, Iterator
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.activations import swiglu
from mlx_lm.models.cache import KVCache
from mlx_lm.models.switch_layers import SwitchGLU

from exo.worker.engines.mlx.models.hy_v3 import kernels
from exo.worker.engines.mlx.models.hy_v3.fused import (
    ExpertStack,
    StackedProjection,
    qmv_batch_limit,
)
from exo.worker.engines.mlx.models.hy_v3.model import (
    MLP,
    Model,
    ModelArgs,
    MoE,
    _select_experts,  # pyright: ignore[reportPrivateUsage]
)

pytestmark = pytest.mark.skipif(
    not kernels.fused_kernels_available(), reason="the fused kernels need Metal"
)

_EPS = 1e-5
_THETA = 11_158_840.0


def _same(a: mx.array, b: mx.array) -> bool:
    return (
        a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a, b).item())
    )


@pytest.fixture(autouse=True)
def seed_mlx() -> Iterator[None]:
    mx.random.seed(0)
    yield


@pytest.mark.parametrize("tokens", [1, 2, 9, 513])
def test_add_rms_norm(tokens: int) -> None:
    x = (mx.random.normal((1, tokens, 4096)) * 3).astype(mx.bfloat16)
    y = (mx.random.normal((1, tokens, 4096)) * 2).astype(mx.bfloat16)
    weight = (1 + 0.3 * mx.random.normal((4096,))).astype(mx.bfloat16)
    h, normed, normed_float32 = kernels.add_rms_norm(
        x, y, weight, _EPS, float32_copy=True
    )
    reference = mx.fast.rms_norm(x + y, weight, _EPS)
    assert _same(h, x + y)
    assert _same(normed, reference)
    assert normed_float32 is not None
    assert _same(normed_float32, reference.astype(mx.float32))
    h2, normed2, none = kernels.add_rms_norm(x, y, weight, _EPS)
    assert none is None and _same(h2, h) and _same(normed2, normed)


@pytest.mark.parametrize(
    "batch,length,offset",
    [(1, 1, 700), (1, 1, 0), (1, 2, 511), (1, 37, 0), (3, 1, None), (2, 5, None)],
)
def test_qk_norm_rope(batch: int, length: int, offset: int | None) -> None:
    q_norm, k_norm = nn.RMSNorm(128, eps=_EPS), nn.RMSNorm(128, eps=_EPS)
    q_norm.weight = (1 + 0.5 * mx.random.normal((128,))).astype(mx.bfloat16)
    k_norm.weight = (1 + 0.5 * mx.random.normal((128,))).astype(mx.bfloat16)
    rope = nn.RoPE(128, traditional=False, base=_THETA)
    q = (mx.random.normal((batch, length, 64 * 128)) * 4).astype(mx.bfloat16)
    k = (mx.random.normal((batch, length, 8 * 128)) * 4).astype(mx.bfloat16)
    v = mx.random.normal((batch, length, 8 * 128)).astype(mx.bfloat16)
    # exo's batch caches hand over one offset per row
    positions: int | mx.array = (
        offset
        if offset is not None
        else mx.array([100 * i + 7 for i in range(batch)], dtype=mx.int32)
    )
    for given in (
        [positions, mx.array([positions], dtype=mx.int32)]
        if isinstance(positions, int)
        else [positions]
    ):
        queries, keys, values = kernels.qk_norm_rope(
            q,
            k,
            v,
            q_weight=q_norm.weight,
            k_weight=k_norm.weight,
            eps=_EPS,
            n_heads=64,
            n_kv_heads=8,
            offset=given,
            rope_base=_THETA,
        )
        assert _same(
            queries,
            rope(
                q_norm(q.reshape(batch, length, 64, 128)).transpose(0, 2, 1, 3),
                offset=given,
            ),
        )
        assert _same(
            keys,
            rope(
                k_norm(k.reshape(batch, length, 8, 128)).transpose(0, 2, 1, 3),
                offset=given,
            ),
        )
        assert _same(values, v.reshape(batch, length, 8, 128).transpose(0, 2, 1, 3))


@pytest.mark.parametrize("experts,top_k", [(192, 8), (16, 4)])
@pytest.mark.parametrize("tokens", [1, 2, 9, 100, 777])
@pytest.mark.parametrize("ties", [False, True])
def test_route(experts: int, top_k: int, tokens: int, ties: bool) -> None:
    logits = (mx.random.normal((1, tokens, experts)) * 2.5).astype(mx.float32)
    bias = (mx.random.normal((experts,)) * 0.05).astype(mx.float32)
    if ties:
        # exact ties: argpartition (MLX's stable merge sort) keeps the lower index first
        logits = mx.round(logits * 2) / 2
        bias = mx.zeros((experts,), dtype=mx.float32)
    indices, weights = _select_experts(logits, bias, top_k, 2.826)
    fused_indices, fused_weights = kernels.route(
        logits, bias, top_k, mx.array(2.826, dtype=mx.float32)
    )
    assert _same(fused_indices, indices.astype(mx.uint32))
    assert _same(fused_weights, weights)


@pytest.mark.parametrize("tokens", [1, 2, 64, 300])
@pytest.mark.parametrize("float32_combine", [False, True])
def test_moe_combine(tokens: int, float32_combine: bool) -> None:
    y = (mx.random.normal((1, tokens, 8, 4096)) * 2).astype(mx.bfloat16)
    weights = mx.softmax(mx.random.normal((1, tokens, 8)), axis=-1) * 2.826
    shared = mx.random.normal((1, tokens, 4096)).astype(mx.bfloat16)
    residual = (mx.random.normal((1, tokens, 4096)) * 5).astype(mx.bfloat16)
    routed = (y * weights[..., None]).sum(axis=-2)
    if float32_combine:
        reference = (routed + shared.astype(mx.float32)).astype(mx.bfloat16)
    else:
        reference = routed.astype(mx.bfloat16) + shared
    fused = kernels.moe_combine(y, weights, shared, float32_combine=float32_combine)
    assert _same(fused, reference)
    fused_residual = kernels.moe_combine(
        y, weights, shared, residual, float32_combine=float32_combine
    )
    assert _same(fused_residual, residual + reference)


def _quantized(
    shape: tuple[int, ...], bits: int = 4
) -> tuple[mx.array, mx.array, mx.array]:
    weight = (mx.random.normal(shape) * 0.02).astype(mx.bfloat16)
    quantized, scales, biases = mx.quantize(weight, 64, bits)
    mx.eval(quantized, scales, biases)
    return quantized, scales, biases


@pytest.mark.parametrize("tokens", [1, 3])
def test_swiglu_down(tokens: int) -> None:
    hidden, intermediate, experts = 1024, 1536, 12
    down = _quantized((experts, hidden, intermediate))
    indices = mx.random.randint(0, experts, (tokens, 8)).astype(mx.uint32)
    gate_up = mx.random.normal((tokens, 8, 1, 2 * intermediate)).astype(mx.bfloat16)
    assert kernels.swiglu_down_fusable(gate_up, down, 64, 4)
    activation = swiglu(gate_up[..., :intermediate], gate_up[..., intermediate:])
    reference = mx.gather_qmm(
        activation, *down, rhs_indices=indices, transpose=True, group_size=64, bits=4
    )
    fused = kernels.swiglu_down(gate_up, down, indices)
    assert _same(fused.reshape(*reference.shape), reference)
    # a prefix of a larger stack of experts
    prefix = tuple(a[: experts - 2] for a in down)
    small = indices % (experts - 2)
    reference = mx.gather_qmm(
        activation, *prefix, rhs_indices=small, transpose=True, group_size=64, bits=4
    )
    fused = kernels.swiglu_down(gate_up, (prefix[0], prefix[1], prefix[2]), small)
    assert _same(fused.reshape(*reference.shape), reference)
    # dense, one row
    row = gate_up[:1, 0]
    reference = mx.quantized_matmul(
        swiglu(row[..., :intermediate], row[..., intermediate:]),
        *(a[0] for a in down),
        transpose=True,
        group_size=64,
        bits=4,
    )
    fused = kernels.swiglu_down(row, (down[0][0], down[1][0], down[2][0]))
    assert _same(fused.reshape(*reference.shape), reference)


def _linears(in_dims: int, outs: list[int]) -> list[nn.Module]:
    parts: list[nn.Module] = []
    for out in outs:
        linear = nn.Linear(in_dims, out, bias=False)
        linear.weight = (mx.random.normal((out, in_dims)) * 0.02).astype(mx.bfloat16)
        parts.append(nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4))
    mx.eval([p.parameters() for p in parts])
    return parts


def _caller(module: nn.Module) -> "Callable[[mx.array], mx.array]":
    def call(x: mx.array) -> mx.array:
        return module(x)

    return call


def test_stacked_projection_matches_its_parts_at_every_row_count() -> None:
    """Stacked q/k/v-like and gate/up-like projections, one matmul up to MLX's batch
    limit and per-part matmuls above it: always the parts' own bits."""
    for outs in ([8192, 1024, 1024], [1536, 1536]):
        parts = _linears(4096, outs)
        originals = [copy.copy(p) for p in parts]
        references_fn = [_caller(p) for p in originals]
        stack = StackedProjection.build(parts)
        assert stack is not None
        limit = min(qmv_batch_limit(4096, n) or 0 for n in [*outs, sum(outs)])
        assert stack.max_rows == limit - 1
        for rows in range(1, limit + 3):
            x = mx.random.normal((1, rows, 4096)).astype(mx.bfloat16)
            references = [f(x) for f in references_fn]
            # the parts now point into the stack: still their own results
            for part, reference in zip(parts, references, strict=True):
                assert _same(part(x), reference)
            stacked = stack(x)
            if rows > stack.max_rows:
                assert stacked is None
                continue
            assert stacked is not None
            for start, stop, reference in zip(
                stack.bounds[:-1], stack.bounds[1:], references, strict=True
            ):
                assert _same(stacked[..., start:stop], reference)


def _switch(experts: int, hidden: int, intermediate: int) -> SwitchGLU:
    glu = SwitchGLU(hidden, intermediate, experts)
    for name in ("gate_proj", "up_proj", "down_proj"):
        linear = getattr(glu, name)
        linear.weight = (mx.random.normal(linear.weight.shape) * 0.02).astype(
            mx.bfloat16
        )
    nn.quantize(glu, group_size=64, bits=4)
    mx.eval(glu.parameters())
    return glu


@pytest.mark.parametrize(
    "tokens",
    # mlx-lm sorts from 64 index pairs (8 tokens); MLX's gather switches to its sorted
    # kernel at 16 pairs and 4 per expert (96 tokens for 192 experts, top 8)
    [1, 2, 7, 8, 9, 63, 64, 95, 96, 97, 128, 129],
)
def test_experts_with_stacked_gate_up_match_switch_glu(tokens: int) -> None:
    experts, hidden, intermediate = 192, 512, 512
    reference_glu = _switch(experts, hidden, intermediate)
    args = ModelArgs(
        model_type="hy_v3",
        vocab_size=16,
        hidden_size=hidden,
        intermediate_size=hidden,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=128,
        rms_norm_eps=_EPS,
        num_experts=experts,
        num_experts_per_tok=8,
        num_shared_experts=1,
        moe_intermediate_size=intermediate,
        router_scaling_factor=2.826,
    )
    moe = MoE(args)
    moe.switch_mlp = copy.deepcopy(reference_glu)
    stack = moe._stack()  # pyright: ignore[reportPrivateUsage]
    assert stack is not None
    x = mx.random.normal((1, tokens, hidden)).astype(mx.bfloat16)
    indices = mx.argsort(mx.random.normal((1, tokens, experts)), axis=-1)[
        ..., :8
    ].astype(mx.uint32)
    reference = reference_glu(x, indices)
    fused = moe._experts(x, indices, stack)  # pyright: ignore[reportPrivateUsage]
    assert _same(fused, reference)


def test_expert_stack_points_the_modules_into_it() -> None:
    glu = _switch(8, 512, 256)
    gate_before, up_before = glu.gate_proj.weight, glu.up_proj.weight
    stack = ExpertStack.build(glu.gate_proj, glu.up_proj)
    assert stack is not None
    assert _same(glu.gate_proj.weight, gate_before)
    assert _same(glu.up_proj.weight, up_before)
    assert stack.gate_up[0].shape == (8, 512, 512 // 8)


def _config(quantized: bool) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model_type": "hy_v3",
        "vocab_size": 256,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 3,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "rms_norm_eps": _EPS,
        "num_experts": 16,
        "num_experts_per_tok": 4,
        "num_shared_experts": 1,
        "moe_intermediate_size": 512,
        "router_scaling_factor": 2.826,
        "rope_parameters": {"rope_theta": _THETA, "rope_type": "default"},
        "first_k_dense_replace": 1,
        "enable_moe_fp32_combine": False,
    }
    if quantized:
        config["quantization"] = {"group_size": 64, "bits": 4}
    return config


def _models(quantized: bool) -> tuple[Model, Model]:
    model = Model(ModelArgs.from_dict(_config(quantized)))
    for layer in model.layers:
        layer.input_layernorm.weight = 1 + 0.2 * mx.random.normal((512,))
        layer.post_attention_layernorm.weight = 1 + 0.2 * mx.random.normal((512,))
        layer.self_attn.q_norm.weight = 1 + 0.2 * mx.random.normal((128,))
        layer.self_attn.k_norm.weight = 1 + 0.2 * mx.random.normal((128,))
        if isinstance(layer.mlp, MoE):
            layer.mlp.router.expert_bias = mx.random.normal((16,)) * 0.1
    model.set_dtype(mx.bfloat16)
    for layer in model.layers:
        if isinstance(layer.mlp, MoE):
            layer.mlp.router.expert_bias = layer.mlp.router.expert_bias.astype(
                mx.float32
            )
    if quantized:

        def predicate(path: str, module: nn.Module) -> bool | dict[str, int]:
            if path.endswith("mlp.router.gate"):
                return {"group_size": 64, "bits": 8}
            return hasattr(module, "to_quantized")

        nn.quantize(model, group_size=64, bits=4, class_predicate=predicate)
    mx.eval(model.parameters())
    return model, copy.deepcopy(model)


@pytest.mark.parametrize("quantized", [False, True])
def test_fused_model_matches_the_reference_path(
    quantized: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prefill at several lengths, then decode steps of 1 and 2 tokens, batch 1 and 3:
    the fused path's logits are the reference path's, bit for bit."""
    fused, reference = _models(quantized)
    tokens = mx.random.randint(0, 256, (3, 200))

    def run(model: Model, batch: int) -> list[mx.array]:
        cache = [KVCache() for _ in model.layers]
        outputs = [model(tokens[:batch, :130], cache=cache)]
        outputs.append(model(tokens[:batch, 130:131], cache=cache))
        outputs.append(model(tokens[:batch, 131:133], cache=cache))
        outputs.append(model(tokens[:batch, 133:134], cache=cache))
        mx.eval(*outputs)
        return outputs

    for batch in (1, 3):
        fused_out = run(fused, batch)
        with monkeypatch.context() as patch:
            patch.setattr(kernels, "fused_kernels_available", lambda: False)
            reference_out = run(reference, batch)
        for a, b in zip(fused_out, reference_out, strict=True):
            assert _same(a, b)
    if quantized:
        # the fused path did run: the experts' gate and up, and the dense MLP's, stacked
        moe = fused.layers[1].mlp
        assert isinstance(moe, MoE)
        assert moe.__dict__.get("_expert_stack") is not None
        dense = fused.layers[0].mlp
        assert isinstance(dense, MLP)
        assert dense.__dict__.get("_gate_up_stack") is not None
