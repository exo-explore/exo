"""GLM-5.3-Flash's fused paths (glm5_next.py) against the plain MLX ops they replace, on a
small random model with the dimensions the Metal kernels need (KDA head_dim 128, an indexer
of 32 heads, a hidden size divisible by 256) in bfloat16:

- the fused projections give bitwise the separate layers' outputs, on the GPU the tests run
  on: one matmul over the fused weight only up to the row count where MLX sums it like the
  parts (its dispatch rule, checked here for every GPU generation MLX can emulate), and the
  parts' own matmuls past it or on GPUs and MLX releases the model doesn't know;
- the KDA decode kernel, the hyper-connection kernels and the indexer's pool-score kernel
  agree with the ops to float32 rounding (bf16 outputs within a unit in the last place);
- the model with every kernel matches itself across chunked prefill and decode, and a
  layer called alone (as pipeline-parallel wrappers do) computes bitwise what the fused
  layer-to-layer path computes.

Without Metal the kernels don't exist and the model runs the ops: the tests are skipped."""

import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from importlib.metadata import version
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import numpy.typing as npt
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.cache import ArraysCache, BatchKVCache, CacheList, KVCache
from mlx_lm.models.switch_layers import SwitchLinear

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
# Wide enough that MLX's quantized matmuls change kernel or K split with the output width
# within 16 rows on every GPU generation: 1024-wide inputs, dense gate/up of 3072 (6144
# fused), shared-expert gate/up of 1024 and MLA q_a/kv_a of 1024 and 512.
QUANTIZED_CONFIG: dict[str, Any] = {
    "model_type": "glm5_next",
    "text_config": {
        **TEXT_CONFIG,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "moe_intermediate_size": 1024,
        "q_lora_rank": 1024,
        "kv_lora_rank": 512,
    },
}
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


def _model(seed: int = 0, quantized: bool = False) -> Model:
    """The random model; ``quantized`` is a wider one (QUANTIZED_CONFIG) that stores the
    MLPs, experts and MLA down projections in 4 bits with float16 scales, as the
    mlx-community checkpoint does (the KDA and indexer projections stay bf16, as there)."""
    mx.random.seed(seed)
    config = QUANTIZED_CONFIG if quantized else CONFIG
    model = Model(cast(ModelArgs, ModelArgs.from_dict(config)))  # pyright: ignore[reportUnknownMemberType]
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
    if quantized:

        def is_quantized(path: str, module: nn.Module) -> bool:
            if not isinstance(module, (nn.Linear, SwitchLinear)):
                return False
            projection = ".mlp." in path or path.endswith(
                ("q_a_proj", "kv_a_proj_with_mqa")
            )
            return projection and cast(mx.array, module.weight).shape[-1] % 64 == 0

        nn.quantize(model, group_size=64, bits=4, class_predicate=is_quantized)
        model.load_weights(
            [
                (name, value.astype(mx.float16))
                for name, value in cast(
                    list[tuple[str, mx.array]], tree_flatten(model.parameters())
                )
                if name.endswith((".scales", ".biases"))
            ],
            strict=False,
        )
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
    if glm5_next._gpu_architecture() is None:  # pyright: ignore[reportPrivateUsage]
        # MLX's dispatch on this GPU or MLX release isn't known: the layers stay as they are
        assert fused is None and all(name in holder for name in names)
        return
    assert fused is not None and "a" not in holder
    # One matmul up to max_rows rows (at least 5 for these quantized shapes on every known
    # GPU, 1 for bf16), the parts' row slices past it: the separate layers' outputs either way.
    assert fused.max_rows >= (5 if quantized else 1)
    path = "one matmul" if rows <= fused.max_rows else "one matmul per part"
    for got, want in zip(fused(x), expected, strict=True):
        assert mx.array_equal(got, want).item(), path
    (joined,) = fused(x, groups=(len(parts),))
    assert mx.array_equal(joined, mx.concatenate(expected, axis=-1)).item(), path


# (name, input width, part widths, bits): the model's fused quantized projections, alone and
# under 2x tensor parallelism, and the tests' shapes above
_QUANTIZED_GROUPS = [
    ("MLA q_a/kv_a", 4096, (1536, 512), 4),
    ("shared experts' gate/up", 4096, (2048, 2048), 6),
    ("shared experts' gate/up, 2x TP", 4096, (1024, 1024), 6),
    ("dense MLP gate/up", 4096, (12288, 12288), 4),
    ("dense MLP gate/up, 2x TP", 4096, (6144, 6144), 4),
    ("test q_a/kv_a", 1024, (1536, 512), 4),
    ("test gate/up", 1024, (2048, 2048), 4),
]


@pytest.mark.parametrize(("name", "in_dims", "sizes", "bits"), _QUANTIZED_GROUPS)
def test_fused_quantized_matmuls_are_their_parts_on_this_gpu(
    name: str, in_dims: int, sizes: tuple[int, ...], bits: int
) -> None:
    """At every row count the model runs one fused quantized matmul for on this GPU, its
    outputs are bitwise the parts' matmuls'. In float16 (the model's matmul dtype) and in
    float32, where any other summation order shows."""
    max_rows = glm5_next._fused_max_rows(True, in_dims, sizes)  # pyright: ignore[reportPrivateUsage]
    if glm5_next._gpu_architecture() is None:  # pyright: ignore[reportPrivateUsage]
        assert max_rows == 0
        return
    mx.random.seed(len(name))
    parts = [
        mx.quantize(
            (mx.random.normal((n, in_dims)) * 0.05).astype(mx.bfloat16),
            group_size=64,
            bits=bits,
        )
        for n in sizes
    ]
    fused = [mx.concatenate([part[i] for part in parts], axis=0) for i in range(3)]
    mx.eval(*fused, *(array for part in parts for array in part))
    for dtype in (mx.float16, mx.float32):
        for rows in range(1, max_rows + 1):
            x = (mx.random.normal((1, rows, in_dims)) * 0.5).astype(dtype)
            out, start = _quantized_matmul(x, fused, bits), 0
            for n, part in zip(sizes, parts, strict=True):
                want = _quantized_matmul(x, part, bits)
                got = out[..., start : start + n]
                assert mx.array_equal(got, want).item(), (name, dtype, rows, n)
                start += n


def _quantized_matmul(x: mx.array, weight: Sequence[mx.array], bits: int) -> mx.array:
    w, scales, biases = weight
    return mx.quantized_matmul(
        x,
        w,
        scales.astype(x.dtype),
        biases.astype(x.dtype),
        transpose=True,
        group_size=64,
        bits=bits,
    )


@pytest.mark.parametrize(
    ("name", "in_dims", "sizes", "fused"),
    [
        ("KDA q/k/v", 4096, (8192, 8192, 8192), True),
        ("KDA q/k/v, 2x TP", 4096, (4096, 4096, 4096), True),
        ("KDA f_a/g_a, indexer wk/gate", 4096, (128, 128), True),
        # b_proj (64 rows) splits its dot products across 8 simdgroups, a 320-row GEMV
        # across 1: not fused
        ("KDA b/f_a/g_a", 4096, (64, 128, 128), False),
        ("test q/k/v", 1024, (2048, 2048, 2048), True),
    ],
)
def test_fused_bf16_matmuls_are_their_parts_on_this_gpu(
    name: str, in_dims: int, sizes: tuple[int, ...], fused: bool
) -> None:
    """The bf16 projections fuse for one input row where MLX's GEMV reduces each output
    like the parts' GEMVs; then the outputs are bitwise the parts', also in float32."""
    max_rows = glm5_next._fused_max_rows(False, in_dims, sizes)  # pyright: ignore[reportPrivateUsage]
    if glm5_next._gpu_architecture() is None:  # pyright: ignore[reportPrivateUsage]
        assert max_rows == 0
        return
    assert max_rows == (1 if fused else 0), name
    if not fused:
        return
    mx.random.seed(len(name))
    weights = [
        (mx.random.normal((n, in_dims)) * 0.05).astype(mx.bfloat16) for n in sizes
    ]
    joined = mx.concatenate(weights, axis=0)
    for dtype in (mx.bfloat16, mx.float32):
        x = (mx.random.normal((1, 1, in_dims)) * 0.5).astype(dtype)
        out, start = x @ joined.T, 0
        for n, w in zip(sizes, weights, strict=True):
            got = out[..., start : start + n]
            assert mx.array_equal(got, x @ w.T).item(), (name, dtype, n)
            start += n


def test_fused_rows_per_gpu_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Up to how many rows the model's quantized projections run as one matmul, per GPU (MLX
    0.32.4's qmv row limits): the MLA and shared-expert groupings, then the dense MLP's."""
    groups = [(in_dims, sizes) for _, in_dims, sizes, _ in _QUANTIZED_GROUPS[:5]]
    expected = {
        (13, "g"): [9, 9, 9, 5, 5],  # M1 / M1 Pro
        (13, "s"): [9, 9, 9, 6, 6],  # M1 Max
        (14, "g"): [9, 9, 9, 5, 5],  # M2 / M2 Pro
        (14, "d"): [16, 16, 16, 11, 11],  # M2 Ultra
        (15, "d"): [16, 16, 16, 11, 11],  # M3 Ultra
        (16, "s"): [14, 14, 14, 12, 12],  # M4 Max
        (17, "g"): [16, 16, 16, 12, 12],  # M5
    }
    for architecture, rows in expected.items():
        monkeypatch.setattr(glm5_next, "_gpu_architecture", _gpu_is(architecture))
        got = [
            glm5_next._fused_max_rows(True, in_dims, sizes)  # pyright: ignore[reportPrivateUsage]
            for in_dims, sizes in groups
        ]
        assert got == rows, architecture


def _gpu_is(architecture: tuple[int, str]) -> Callable[[], tuple[int, str]]:
    return lambda: architecture


def test_expert_gathers_fuse_on_m1_to_m4_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The routed experts' gate/up gathers fuse where MLX's gather kernels are known not to
    depend on the output width (M1-M4); M5 GPUs (matrix-unit gathers) and unknown GPUs keep
    the separate gathers."""
    for architecture, fuses in [
        ((13, "g"), True),
        ((15, "d"), True),
        ((16, "s"), True),
        ((17, "s"), False),
        (None, False),
    ]:
        monkeypatch.setattr(
            glm5_next, "_gpu_architecture", _gpu_is_or_none(architecture)
        )
        moe = _model(quantized=True).model.layers[3].mlp
        assert isinstance(moe, MoE)
        assert (moe._fuse_gate_up() is not None) == fuses, architecture


def _gpu_is_or_none(
    architecture: tuple[int, str] | None,
) -> Callable[[], tuple[int, str] | None]:
    return lambda: architecture


# Run with MLX_METAL_GPU_ARCH set, so that MLX dispatches as on that GPU generation: for
# each (input width, output width, rows), whether MLX ran the matrix-vector kernel. It did if
# the first two rows' outputs are bitwise those of a two-row matmul (always a matrix-vector
# kernel); float32 inputs make any other summation order show (and keep MLX off the M5
# matrix units, which an emulated M5 doesn't have).
_DISPATCH_PROBE = """
import json, sys
import mlx.core as mx
results = []
for in_dims, out_dims, rows in json.loads(sys.argv[1]):
    mx.random.seed(0)
    w = mx.random.normal((out_dims, in_dims)) * 0.05
    wq, scales, biases = mx.quantize(w, group_size=64, bits=4)
    x = mx.random.normal((rows, in_dims))
    def matmul(v):
        return mx.quantized_matmul(v, wq, scales, biases, transpose=True)
    results.append(mx.array_equal(matmul(x)[:2], matmul(x[:2])).item())
print(json.dumps({"architecture": mx.device_info()["architecture"], "qmv": results}))
"""


@pytest.mark.parametrize(
    "architecture",
    [f"applegpu_g{generation}{size}" for generation in range(13, 18) for size in "gsd"],
)
def test_qmv_row_limits_are_mlx_dispatch(architecture: str) -> None:
    """_qmv_row_limit against the running MLX, dispatching as on each GPU generation (MLX
    reads MLX_METAL_GPU_ARCH): one row below the limit MLX runs the matrix-vector kernel, at
    the limit it doesn't. An MLX release that moves the limits fails here."""
    match = glm5_next._KNOWN_ARCHITECTURE.fullmatch(architecture)  # pyright: ignore[reportPrivateUsage]
    assert match is not None
    gpu = (int(match[1]), match[2])
    cases: list[tuple[int, int, int]] = []
    for in_dims, out_dims in [(1024, 2048), (4096, 2048), (4096, 6144), (2048, 4096)]:
        limit = glm5_next._qmv_row_limit(gpu, in_dims, out_dims)  # pyright: ignore[reportPrivateUsage]
        cases += [(in_dims, out_dims, limit - 1), (in_dims, out_dims, limit)]
    env = {**os.environ, "MLX_METAL_GPU_ARCH": architecture, "MLX_ENABLE_TF32": "0"}
    run = subprocess.run(
        [sys.executable, "-c", _DISPATCH_PROBE, json.dumps(cases)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    probe = cast(dict[str, Any], json.loads(run.stdout))
    assert probe["architecture"] == architecture
    qmv = cast(list[bool], probe["qmv"])
    expected = [rows < glm5_next._qmv_row_limit(gpu, k, n) for k, n, rows in cases]  # pyright: ignore[reportPrivateUsage]
    assert qmv == expected, list(zip(cases, qmv, strict=True))


# GPU generations whose dispatch the machine running the tests can stand in for (an M5's
# matrix-unit kernels need an M5): M1, M1 Max, an Ultra, M3, M4 Max.
_EMULATED_GPUS = [
    "applegpu_g13g",
    "applegpu_g13s",
    "applegpu_g14d",
    "applegpu_g15g",
    "applegpu_g16s",
]


@pytest.mark.parametrize("architecture", _EMULATED_GPUS)
def test_fused_paths_with_each_gpu_generations_dispatch(architecture: str) -> None:
    """The fused-projection tests above again, on this GPU but with MLX dispatching as on
    another generation (MLX_METAL_GPU_ARCH): the fused paths take that generation's rules,
    on real hardware."""
    tests = "fused_projections or fused_quantized or fused_bf16 or unknown_gpus"
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            __file__,
            "-q",
            "-p",
            "no:cacheprovider",
            "--import-mode=importlib",
            "-k",
            tests,
        ],
        env={**os.environ, "MLX_METAL_GPU_ARCH": architecture},
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout[-4000:]
    assert "skipped" not in run.stdout.splitlines()[-1], run.stdout[-1000:]


def test_mlx_release_is_one_the_fusion_rules_were_checked_on() -> None:
    """The fused projections run only on the MLX releases in _CHECKED_MLX. After an MLX
    upgrade, check its dispatch rules (quantized.cpp's get_qmv_batch_limit and qmv kernels,
    matmul.cpp's GEMV tiling) against glm5_next.py's, then add the release."""
    assert glm5_next._CHECKED_MLX.fullmatch(version("mlx")), version("mlx")  # pyright: ignore[reportPrivateUsage]


def test_unknown_gpus_run_the_separate_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a GPU or MLX release whose dispatch the model doesn't know, nothing fuses: the
    model computes bitwise what it computes with fusing switched off."""
    for name in _KERNELS:
        monkeypatch.setattr(glm5_next, name, None)
    monkeypatch.setattr(glm5_next, "_gpu_architecture", lambda: None)
    prompt = _tokens(12, seed=3)
    model = _model(quantized=True)
    unknown = model(prompt, model.make_cache())
    layers = model.model.layers
    assert "q_proj" in layers[0].self_attn and "q_a_proj" in layers[3].self_attn
    assert "gate_proj" in cast(MoE, layers[3].mlp).switch_mlp
    monkeypatch.setattr(glm5_next, "_FUSE_PROJECTIONS", False)
    off = _model(quantized=True)
    assert mx.array_equal(unknown, off(prompt, off.make_cache())).item()


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


def _logits_by_chunk_length(lengths: list[int], quantized: bool) -> list[mx.array]:
    model = _model(quantized=quantized)
    prompt = _tokens(max(lengths), seed=11)
    outs: list[mx.array] = []
    for length in lengths:
        out = model(prompt[:, :length], model.make_cache())
        mx.eval(out)
        outs.append(out)
    return outs


def _logits_by_decode_batch(sizes: list[int], quantized: bool) -> list[mx.array]:
    model = _model(quantized=quantized)
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


@pytest.mark.parametrize("quantized", [False, True])
def test_fused_projections_are_exact_across_kernel_thresholds(
    monkeypatch: pytest.MonkeyPatch, quantized: bool
) -> None:
    """The fused projections and expert gathers change the row counts MLX's matmul,
    quantized-matmul and gather kernels see, and those kernels pick their variant by row
    counts. At chunk lengths around their thresholds, and for decode batches of 1-9 rows,
    the model with fused projections must give bitwise the separate layers' logits (the
    float32-reordering kernels are off on both sides)."""
    for name in _KERNELS:
        monkeypatch.setattr(glm5_next, name, None)
    # 2..8 rows: the quantized matmuls' wide-row variant on M3-class GPUs; 12-17 tokens:
    # MLX's quantized matrix-vector limit for these widths (13 to 33 rows, by GPU), this
    # model's gather flip (tokens x top_k / experts >= 4) and the fused matmuls' 16-row cap
    lengths = list(range(1, 18)) + [31, 32, 33, 63, 64, 65, 127, 128, 129]
    sizes = list(range(1, 10))
    fused_prefill = _logits_by_chunk_length(lengths, quantized)
    fused_decode = _logits_by_decode_batch(sizes, quantized)
    monkeypatch.setattr(glm5_next, "_FUSE_PROJECTIONS", False)
    separate_prefill = _logits_by_chunk_length(lengths, quantized)
    separate_decode = _logits_by_decode_batch(sizes, quantized)
    for length, a, b in zip(lengths, fused_prefill, separate_prefill, strict=True):
        assert mx.array_equal(a, b).item(), f"prefill chunk of {length} tokens"
    for i, (a, b) in enumerate(zip(fused_decode, separate_decode, strict=True)):
        assert mx.array_equal(a, b).item(), f"decode batch of {sizes[i // 2]}"
