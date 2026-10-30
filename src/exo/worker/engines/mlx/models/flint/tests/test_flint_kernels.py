"""Flint's fused kernels compute exactly what the MLX ops they replace compute."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import cast

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.flint import Attention, ModelArgs
from mlx_lm.models.flint import Model as MlxLmModel

from exo.worker.engines.mlx.models.flint import kernels
from exo.worker.engines.mlx.models.flint.model import (
    FlintAttention,
    FlintSparseMoeBlock,
    Model,
    StackedQuantizedLinear,
)

needs_metal = pytest.mark.skipif(
    not kernels.fused_kernels_available(), reason="the fused kernels need Metal"
)


def _bf16(shape: tuple[int, ...], seed: int, scale: float = 1.0) -> mx.array:
    """Heavy-tailed values, like a residual stream."""
    base = mx.random.normal(shape, key=mx.random.key(seed))
    tails = mx.random.normal(shape, key=mx.random.key(seed + 1000)) ** 3
    return ((base + 0.2 * tails) * scale).astype(mx.bfloat16)


def _weight(dim: int, seed: int) -> mx.array:
    return (mx.random.normal((dim,), key=mx.random.key(seed)) * 0.3 + 1.0).astype(
        mx.bfloat16
    )


def _reference_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    return mx.fast.rms_norm(
        x.astype(mx.float32), weight.astype(mx.float32), eps
    ).astype(x.dtype)


def _identical(a: mx.array, b: mx.array) -> bool:
    return (
        a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a, b).item())
    )


_ROWS = [(1, 1, 2048), (4, 9, 2048), (1, 512, 2048), (1, 7, 32, 128), (3, 5, 2049)]


@needs_metal
@pytest.mark.parametrize("shape", _ROWS)
@pytest.mark.parametrize("scale", [0.01, 1.0, 3000.0])
def test_rms_norm_float32_is_the_float32_formula_bit_for_bit(
    shape: tuple[int, ...], scale: float
) -> None:
    x = _bf16(shape, 1, scale)
    weight = _weight(shape[-1], 2)
    assert _identical(
        kernels.rms_norm_float32(x, weight, 1e-6), _reference_norm(x, weight, 1e-6)
    )


@needs_metal
@pytest.mark.parametrize("shape", _ROWS)
def test_add_rms_norm_float32_is_add_then_norm_bit_for_bit(
    shape: tuple[int, ...],
) -> None:
    x, y = _bf16(shape, 3, 30.0), _bf16(shape, 4, 30.0)
    weight = _weight(shape[-1], 5)
    h, normed = kernels.add_rms_norm_float32(x, y, weight, 1e-5)
    assert _identical(h, x + y)
    assert _identical(normed, _reference_norm(x + y, weight, 1e-5))
    h, normed, normed_float32 = kernels.add_rms_norm_float32_with_copy(
        x, y, weight, 1e-5
    )
    assert _identical(h, x + y)
    assert _identical(normed, _reference_norm(x + y, weight, 1e-5))
    assert _identical(normed_float32, normed.astype(mx.float32))


@needs_metal
@pytest.mark.parametrize("batch,length", [(1, 1), (2, 1), (1, 7), (3, 600)])
@pytest.mark.parametrize("rope", [True, False])
def test_qk_norm_rope_is_norm_then_rope_bit_for_bit(
    batch: int, length: int, rope: bool
) -> None:
    heads, kv_heads, dim = 32, 4, 128
    q = _bf16((batch, length, heads * dim), 6, 3.0)
    k = _bf16((batch, length, kv_heads * dim), 7, 3.0)
    q_weight, k_weight = _weight(dim, 8), _weight(dim, 9)
    offsets: list[int | mx.array] = [0, 4093]
    if batch > 1:
        offsets.append(mx.array([3, 900, 12000][:batch], dtype=mx.int32))
    for offset in offsets:
        expected_q = _reference_norm(
            q.reshape(batch, length, heads, dim), q_weight, 1e-6
        ).transpose(0, 2, 1, 3)
        expected_k = _reference_norm(
            k.reshape(batch, length, kv_heads, dim), k_weight, 1e-6
        ).transpose(0, 2, 1, 3)
        if rope:
            expected_q = mx.fast.rope(
                expected_q,
                dim,
                traditional=False,
                base=10000.0,
                scale=1.0,
                offset=offset,
            )
            expected_k = mx.fast.rope(
                expected_k,
                dim,
                traditional=False,
                base=10000.0,
                scale=1.0,
                offset=offset,
            )
        queries, keys = kernels.qk_norm_rope(
            q,
            k,
            q_weight=q_weight,
            k_weight=k_weight,
            eps=1e-6,
            n_heads=heads,
            n_kv_heads=kv_heads,
            offset=offset,
            rope_base=10000.0 if rope else None,
        )
        assert _identical(queries, expected_q)
        assert _identical(keys, expected_k)


@needs_metal
@pytest.mark.parametrize("lead", [(1, 1), (8, 1), (1, 9), (2, 300)])
@pytest.mark.parametrize("residual", [True, False])
def test_moe_combine_is_mlx_lms_weighted_sum_bit_for_bit(
    lead: tuple[int, ...], residual: bool
) -> None:
    y = _bf16((*lead, 6, 2048), 10, 2.0)
    weights = mx.random.uniform(shape=(*lead, 6), key=mx.random.key(11))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    shared = _bf16((*lead, 2048), 12)
    stream = _bf16((*lead, 2048), 13, 10.0)
    expected = (y * weights[..., None]).sum(axis=-2).astype(y.dtype) + shared
    if residual:
        expected = stream + expected
    out = kernels.moe_combine(y, weights, shared, stream if residual else None)
    assert _identical(out, expected)


_ARGS = ModelArgs(
    model_type="flint",
    hidden_size=256,
    num_hidden_layers=4,
    intermediate_size=384,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=128,
    num_experts=8,
    num_experts_per_tok=3,
    moe_intermediate_size=64,
    shared_expert_intermediate_size=128,
    mlp_only_layers=[0],
    layer_types=[
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ],
    sliding_window=5,
    rms_norm_eps=1e-6,
    vocab_size=96,
)


class _ReferenceNorm(nn.RMSNorm):
    """The float32 norm through MLX's ops: what exo ran before the fused kernels."""

    def __call__(self, x: mx.array) -> mx.array:
        return _reference_norm(x, self.weight, self.eps)


def _reference_model() -> MlxLmModel:
    model = MlxLmModel(_ARGS)

    def ref(norm: nn.RMSNorm) -> _ReferenceNorm:
        return _ReferenceNorm(norm.weight.shape[0], eps=norm.eps)

    for layer in model.layers:
        layer.input_layernorm = ref(layer.input_layernorm)
        layer.post_attention_layernorm = ref(layer.post_attention_layernorm)
        layer.self_attn.q_norm = ref(layer.self_attn.q_norm)
        layer.self_attn.k_norm = ref(layer.self_attn.k_norm)
    model.model.norm = ref(model.model.norm)
    return model


def _quantizable(path: str, module: nn.Module) -> bool:
    """As the checkpoints are: everything but the router."""
    return hasattr(module, "to_quantized") and not path.endswith("mlp.gate")


def _quantized_pair() -> tuple[Model, MlxLmModel]:
    """exo's model and the reference, bf16 with 4-bit weights, sharing every weight."""
    mx.random.seed(3)
    fused = Model(_ARGS)
    fused.set_dtype(mx.bfloat16)  # pyright: ignore[reportUnknownMemberType]
    nn.quantize(fused, group_size=64, bits=4, class_predicate=_quantizable)
    reference = _reference_model()
    reference.set_dtype(mx.bfloat16)  # pyright: ignore[reportUnknownMemberType]
    nn.quantize(reference, group_size=64, bits=4, class_predicate=_quantizable)
    reference.load_weights(tree_flatten(fused.parameters()))  # pyright: ignore[reportArgumentType]
    mx.eval(fused.parameters(), reference.parameters())
    return fused, reference


def _run(
    model: nn.Module,
    prompt: mx.array,
    steps: int,
    make_cache: Callable[[], Sequence[object]],
) -> list[mx.array]:
    cache = list(make_cache())
    logits = model(prompt, cache=cache)
    outputs = [logits]
    token = mx.argmax(logits[:, -1:], axis=-1)
    for _ in range(steps):
        logits = model(token, cache=cache)
        outputs.append(logits)
        token = mx.argmax(logits[:, -1:], axis=-1)
    mx.eval(*outputs)
    return outputs


@needs_metal
def test_fused_model_logits_equal_the_unfused_models_bit_for_bit() -> None:
    """Prefill past the sliding window, then decode: every logit equal."""
    fused, reference = _quantized_pair()
    prompt = mx.random.randint(0, _ARGS.vocab_size, (2, 11), key=mx.random.key(4))
    got = _run(fused, prompt, 6, fused.make_cache)
    expected = _run(reference, prompt, 6, reference.make_cache)
    for a, b in zip(got, expected, strict=True):
        assert _identical(a, b)


def test_router_weight_is_kept_in_float32() -> None:
    weights = {
        "model.layers.1.mlp.gate.weight": mx.ones((8, 256), dtype=mx.bfloat16),
        "model.layers.1.mlp.shared_expert.up_proj.weight": mx.ones(
            (128, 256), dtype=mx.bfloat16
        ),
    }
    out = Model(_ARGS).sanitize(weights)
    assert out["model.layers.1.mlp.gate.weight"].dtype == mx.float32
    assert out["model.layers.1.mlp.shared_expert.up_proj.weight"].dtype == mx.bfloat16


def test_layers_use_the_fused_classes() -> None:
    model = Model(_ARGS)
    assert all(isinstance(layer.self_attn, FlintAttention) for layer in model.layers)
    assert not isinstance(model.layers[0].mlp, FlintSparseMoeBlock)  # dense layer
    assert all(isinstance(layer.mlp, FlintSparseMoeBlock) for layer in model.layers[1:])


def test_off_metal_the_layer_runs_mlx_lms_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the kernels (CUDA, CPU) every path falls back to the reference ops."""
    monkeypatch.setattr(kernels, "fused_kernels_available", lambda: False)
    fused, reference = _quantized_pair()
    prompt = mx.random.randint(0, _ARGS.vocab_size, (1, 8), key=mx.random.key(5))
    got = _run(fused, prompt, 3, fused.make_cache)
    expected = _run(reference, prompt, 3, reference.make_cache)
    for a, b in zip(got, expected, strict=True):
        assert _identical(a, b)


def test_attention_with_other_head_dims_runs_the_original_attention() -> None:
    """The fused query/key kernel covers 128-value heads; others take mlx-lm's path."""
    args = replace(_ARGS, head_dim=64)
    attention = FlintAttention(args, 1)
    attention.set_dtype(mx.bfloat16)  # pyright: ignore[reportUnknownMemberType]
    x = _bf16((1, 3, args.hidden_size), 14)
    assert not attention._fusable(x)  # pyright: ignore[reportPrivateUsage]
    original = Attention(args, 1)
    original.q_norm = _ReferenceNorm(64, eps=args.rms_norm_eps)
    original.k_norm = _ReferenceNorm(64, eps=args.rms_norm_eps)
    original.set_dtype(mx.bfloat16)  # pyright: ignore[reportUnknownMemberType]
    original.load_weights(tree_flatten(attention.parameters()))  # pyright: ignore[reportArgumentType]
    assert _identical(attention(x), original(x))


@needs_metal
def test_projections_are_stacked_at_the_first_call() -> None:
    """q/k/v and every gate/up pair become one stacked projection; nothing else changes."""
    fused, _ = _quantized_pair()
    prompt = mx.random.randint(0, _ARGS.vocab_size, (1, 3), key=mx.random.key(6))
    mx.eval(fused(prompt, cache=fused.make_cache()))
    names = {name for name, _ in tree_flatten(fused.parameters())}
    for i in range(_ARGS.num_hidden_layers):
        attention = f"model.layers.{i}.self_attn"
        assert f"{attention}.qkv_proj.weight" in names
        assert f"{attention}.q_proj.weight" not in names
        assert f"{attention}.o_proj.weight" in names
    assert "model.layers.0.mlp.gate_up_proj.weight" in names  # dense layer
    assert "model.layers.1.mlp.shared_expert.gate_up_proj.weight" in names
    assert "model.layers.1.mlp.switch_mlp.gate_up_proj.weight" in names
    assert "model.layers.1.mlp.switch_mlp.down_proj.weight" in names


@needs_metal
def test_stacked_projection_runs_each_part_alone_beyond_a_few_rows() -> None:
    parts = [nn.QuantizedLinear(256, rows, bias=False) for rows in (512, 256, 256)]
    stacked = StackedQuantizedLinear(list[nn.Module](parts))
    for tokens in (1, 4, 5, 64):
        x = _bf16((1, tokens, 256), 15)
        outputs = stacked.split(x)
        assert (stacked.stacked(x) is None) == (tokens > 4)
        for part, out in zip(parts, outputs, strict=True):
            assert _identical(out, part(x))


@needs_metal
@pytest.mark.parametrize("cached", [0, 7, 37])
def test_blocked_sliding_prefill_equals_attention_over_all_keys(
    monkeypatch: pytest.MonkeyPatch, cached: int
) -> None:
    """Sliding layers prefill one block of queries at a time over the key tiles in reach;
    the logits equal attention over every key, after any number of cached tokens."""
    from exo.worker.engines.mlx.models.flint import model as flint_model

    fused, reference = _quantized_pair()
    prompt = mx.random.randint(
        0, _ARGS.vocab_size, (1, cached + 150), key=mx.random.key(8)
    )

    def prefill(
        model: nn.Module, make_cache: Callable[[], Sequence[object]]
    ) -> mx.array:
        cache = list(make_cache())
        if cached:
            mx.eval(model(prompt[:, :cached], cache=cache))
        logits = model(prompt[:, cached:], cache=cache)
        mx.eval(logits)
        return logits

    monkeypatch.setattr(flint_model, "_SLIDING_QUERY_BLOCK", 32)
    blocked = prefill(fused, fused.make_cache)
    assert _identical(blocked, prefill(reference, reference.make_cache))


def test_sliding_window_is_known_to_sliding_layers_only() -> None:
    model = Model(_ARGS)
    windows = [
        cast(FlintAttention, layer.self_attn).sliding_window for layer in model.layers
    ]
    assert windows == [None, 5, 5, None]
