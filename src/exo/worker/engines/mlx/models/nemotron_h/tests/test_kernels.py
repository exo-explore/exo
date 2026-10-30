"""Nemotron-H's Metal kernels against the MLX ops they replace, and the model's fused
paths against its unfused ones (the CPU fallback, block-by-block calls, an unfolded
shared expert)."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.nemotron_h import NemotronHMamba2Mixer, group_expert_select
from mlx_lm.models.ssm import ssm_update

from exo.worker.engines.mlx.models.nemotron_h import kernels
from exo.worker.engines.mlx.models.nemotron_h import model as family
from exo.worker.engines.mlx.models.nemotron_h.model import (
    Float32MoE,
    Model,
    ModelArgs,
    fold_shared_experts,
)
from exo.worker.engines.mlx.models.nemotron_h.tests.tiny import (
    CONFIG,
    NANO_CONFIG,
    PROMPT,
)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the kernels are Metal kernels"
)

# The SSM state of every Nemotron-H checkpoint has 128 columns; the decode kernels (this
# family's and mlx-lm's) split it over 32 lanes, so the test models use 32 rather than
# tiny.py's 16 (at 16 mlx-lm's own decode kernel updates nothing).
MODEL_CONFIGS = [
    {**CONFIG, "ssm_state_size": 32},
    {**NANO_CONFIG, "ssm_state_size": 32},
]

# Lightning's Mamba-2 shapes, scaled down: heads, head dim, groups, state, conv kernel
HEADS, HEAD_DIM, GROUPS, STATE, KERNEL = 8, 16, 2, 64, 4
INNER = HEADS * HEAD_DIM
CONV = INNER + 2 * GROUPS * STATE
PROJ = INNER + CONV + HEADS


def _relative(a: mx.array, b: mx.array) -> float:
    a32 = a.astype(mx.float32)
    b32 = b.astype(mx.float32)
    return float((mx.abs(a32 - b32).max() / mx.abs(b32).max()).item())


@contextmanager
def _on_cpu() -> Iterator[None]:
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(mx.gpu)


def _mamba_inputs(batch: int, length: int) -> dict[str, mx.array]:
    mx.random.seed(length)
    return {
        "proj": mx.random.normal((batch, length, PROJ)).astype(mx.bfloat16),
        "conv_state": mx.random.normal((batch, KERNEL - 1, CONV)).astype(mx.bfloat16),
        "ssm_state": mx.random.normal((batch, HEADS, HEAD_DIM, STATE)) * 0.5,
        "weight": (mx.random.normal((CONV, KERNEL, 1)) * 0.3).astype(mx.bfloat16),
        "bias": (mx.random.normal((CONV,)) * 0.1).astype(mx.bfloat16),
        "dt_bias": mx.random.uniform(-4, 0, (HEADS,)).astype(mx.bfloat16),
        "A_log": mx.log(mx.random.uniform(1, 16, (HEADS,))),
        "D": mx.random.normal((HEADS,)).astype(mx.bfloat16),
    }


@pytest.mark.parametrize(("batch", "length"), [(1, 1), (3, 1), (1, 2), (2, 37)])
def test_mamba_kernels_match_mlx_ops(batch: int, length: int) -> None:
    p = _mamba_inputs(batch, length)
    xbc, conv_state = kernels.causal_conv_silu(
        p["proj"], INNER, CONV, p["conv_state"], p["weight"], p["bias"]
    )
    ref_xbc, ref_conv_state = kernels.causal_conv_silu_reference(
        p["proj"], INNER, CONV, p["conv_state"], p["weight"], p["bias"]
    )
    assert _relative(xbc, ref_xbc) < 1e-6
    assert mx.array_equal(conv_state, ref_conv_state)

    limits = (0.0, float("inf"))
    y, ssm_state = kernels.selective_scan(
        xbc, p["proj"], INNER + CONV, p["dt_bias"], p["A_log"], p["D"], limits,
        p["ssm_state"], HEADS, HEAD_DIM, GROUPS, STATE,
    )  # fmt: skip
    x, b, c = mx.split(ref_xbc, [INNER, INNER + GROUPS * STATE], axis=-1)
    ref_y, ref_state = ssm_update(
        x.reshape(batch, length, HEADS, HEAD_DIM),
        p["A_log"],
        b.reshape(batch, length, GROUPS, STATE),
        c.reshape(batch, length, GROUPS, STATE),
        p["D"].astype(mx.float32),
        p["proj"][..., INNER + CONV :],
        p["dt_bias"],
        p["ssm_state"],
        limits,
        None,
    )
    gate = p["proj"][..., :INNER].astype(mx.float32)
    ref_y = ref_y.reshape(batch, length, INNER) * nn.silu(gate)
    assert y.dtype == mx.float32
    assert _relative(y, ref_y) < 1e-5
    assert _relative(ssm_state, ref_state) < 1e-5

    weight = (1 + 0.1 * mx.random.normal((INNER,))).astype(mx.bfloat16)
    normed = kernels.grouped_rms_norm(y, weight, INNER // GROUPS, 1e-5, mx.float32)
    grouped = mx.unflatten(ref_y, -1, (-1, INNER // GROUPS))
    ref_normed = weight * mx.fast.rms_norm(grouped, None, 1e-5).flatten(-2)
    assert _relative(normed, ref_normed) < 1e-5


@pytest.mark.parametrize("always_on", [0, 2])
def test_select_experts_matches_group_expert_select(always_on: int) -> None:
    mx.random.seed(1)
    logits = mx.random.normal((2, 5, 128)) * 2
    bias = mx.random.normal((128,)) * 0.1
    inds, scores = kernels.select_experts(logits, bias, 6, 2.5, True, always_on)
    ref_inds, ref_scores = group_expert_select(logits, bias, 6, 1, 1, 2.5, True)
    ref_inds, ref_scores = kernels.with_always_on_experts(
        ref_inds, ref_scores, 128, always_on
    )
    assert inds.shape == ref_inds.shape == (2, 5, 6 + always_on)
    order = mx.argsort(inds, axis=-1)
    ref_order = mx.argsort(ref_inds, axis=-1)
    assert mx.array_equal(
        mx.take_along_axis(inds, order, -1), mx.take_along_axis(ref_inds, ref_order, -1)
    )
    assert mx.allclose(
        mx.take_along_axis(scores, order, -1),
        mx.take_along_axis(ref_scores, ref_order, -1),
        rtol=1e-6,
        atol=0,
    )


def test_combine_and_residual_norm_round_as_mlx_ops() -> None:
    mx.random.seed(2)
    y = mx.random.normal((3, 8, 160)).astype(mx.bfloat16)
    scores = mx.random.uniform(0, 1, (3, 8))
    shared = mx.random.normal((3, 160)).astype(mx.bfloat16)
    residual = mx.random.normal((3, 160)).astype(mx.bfloat16)
    expected = residual + (
        (y * scores[..., None]).sum(axis=-2).astype(y.dtype) + shared
    )

    assert mx.array_equal(
        kernels.combine_experts(y, scores, shared, residual), expected
    )

    weight = (1 + 0.1 * mx.random.normal((160,))).astype(mx.bfloat16)
    hidden, normed, normed32 = kernels.residual_norm(
        residual, weight, 1e-5, y=y, scores=scores, shared=shared, want_float32=True
    )
    assert mx.array_equal(hidden, expected)
    assert normed32 is not None and mx.array_equal(normed32, normed.astype(mx.float32))
    # MLX's RMSNorm sums the squares in another order: at most one bf16 rounding apart
    assert _relative(normed, mx.fast.rms_norm(expected, weight, 1e-5)) < 2**-7
    # A block output without scores is added as is, and the norm alone gives the same bits
    block_output = mx.random.normal((3, 160)).astype(mx.bfloat16)
    hidden, normed, _ = kernels.residual_norm(residual, weight, 1e-5, y=block_output)
    assert mx.array_equal(hidden, residual + block_output)
    assert mx.array_equal(normed, kernels.residual_norm(hidden, weight, 1e-5)[1])


def _model(config: dict[str, Any], quantize: bool) -> Model:
    mx.random.seed(0)
    model = Model(ModelArgs.from_dict(config))
    for layer in model.layers:
        mixer = layer.mixer
        if isinstance(mixer, NemotronHMamba2Mixer):
            heads = mixer.num_heads
            mixer.update(
                {
                    "A_log": mx.random.normal((heads,)),
                    "D": mx.random.normal((heads,)),
                    "dt_bias": mx.random.normal((heads,)),
                }
            )
    if quantize:
        nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())
    return model


def _generate(model: Model, steps: int = 4) -> mx.array:
    """Logits of a prefill in two chunks and a few decode steps through the cache (with
    given tokens: an argmax would turn a near-tie into a different continuation)."""
    cache = model.make_cache()
    prompt = mx.array(PROMPT)
    rows = [model(prompt[:, :5], cache=cache), model(prompt[:, 5:], cache=cache)]
    for token in PROMPT[0][:steps]:
        rows.append(model(mx.array([[token]]), cache=cache))
    return mx.concatenate(rows, axis=1)


@pytest.mark.parametrize("config", MODEL_CONFIGS)
def test_kernels_compute_the_mlx_op_path_in_float32(config: dict[str, Any]) -> None:
    model = _model(config, quantize=False)
    with _on_cpu():
        expected = _generate(model)
        mx.eval(expected)
    assert _relative(_generate(model), expected) < 1e-4


@pytest.mark.parametrize("config", MODEL_CONFIGS)
def test_fused_block_boundaries_give_the_bits_of_whole_blocks(
    config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A pipeline shard calls its edge blocks whole; the rest of the model is fused.
    model = _model(config, quantize=True)
    model.set_dtype(mx.bfloat16)  # pyright: ignore[reportUnknownMemberType]
    fused = _generate(model)
    monkeypatch.setattr(
        family.Backbone, "__call__", family.NemotronHModel.__call__, raising=True
    )
    assert mx.array_equal(_generate(model), fused)


def test_folded_shared_expert_computes_the_same_moe() -> None:
    model = _model(MODEL_CONFIGS[1], quantize=True)
    unfolded = _generate(model)
    moes = [
        layer.mixer for layer in model.layers if isinstance(layer.mixer, Float32MoE)
    ]
    for moe in moes:
        fold_shared_experts(moe)
    assert all("shared_experts" not in moe for moe in moes)
    assert all(moe.router.always_on == 2 for moe in moes)
    assert all(moe.switch_mlp.fc1.weight.shape[0] == 4 + 2 for moe in moes)
    assert _relative(_generate(model), unfolded) < 1e-4


@pytest.mark.parametrize("length", [1, 5, 6, 7, 15, 16, 17, 31, 32, 33, 64])
def test_folded_shared_expert_at_gather_kernel_switch_points(length: int) -> None:
    # gather_qmm switches kernels with the number of rows (sorted indices from 64 index
    # pairs, a per-expert path from 4 rows per expert): with the shared expert's two
    # slices, a token has 4 index pairs over 6 experts here, so the switches fall at
    # other token counts than without folding. Every count around them computes the
    # same MoE.
    model = _model(MODEL_CONFIGS[1], quantize=True)
    prompt = mx.array([(PROMPT[0] * (length // len(PROMPT[0]) + 1))[:length]])
    unfolded = model(prompt)
    for layer in model.layers:
        if isinstance(layer.mixer, Float32MoE):
            fold_shared_experts(layer.mixer)
    assert _relative(model(prompt), unfolded) < 1e-4


def test_folding_skips_shared_experts_quantized_differently() -> None:
    model = _model(MODEL_CONFIGS[1], quantize=False)
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: hasattr(module, "to_quantized")
        and "shared_experts" not in path,
    )
    moe = next(
        layer.mixer for layer in model.layers if isinstance(layer.mixer, Float32MoE)
    )
    fold_shared_experts(moe)
    assert "shared_experts" in moe
    assert moe.router.always_on == 0


def test_caches_keep_the_layout_of_mlx_lm() -> None:
    # exo snapshots, restores and merges SSM caches: the fused mixer keeps their layout.
    model = _model(MODEL_CONFIGS[1], quantize=True)
    model.set_dtype(mx.bfloat16)  # pyright: ignore[reportUnknownMemberType]
    cache = model.make_cache()
    model(mx.array(PROMPT), cache=cache)
    with _on_cpu():
        ref_cache = model.make_cache()
        model(mx.array(PROMPT), cache=ref_cache)
    for entry, ref in zip(cache, ref_cache, strict=True):
        if isinstance(entry, ArraysCache):
            assert isinstance(ref, ArraysCache)
            for state, ref_state in zip(entry.cache, ref.cache, strict=True):
                assert state is not None and ref_state is not None
                assert state.shape == ref_state.shape
                assert state.dtype == ref_state.dtype
        else:
            assert isinstance(entry, KVCache)
