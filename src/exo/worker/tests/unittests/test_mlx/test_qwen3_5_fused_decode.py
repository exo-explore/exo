# type: ignore
"""The fused Qwen3.5 decode paths against mlx-lm's own code, on a small random 4-bit model.

Exact where the fused code reproduces MLX's arithmetic (fused projections, attention prologue,
add + RMSNorm); within float tolerance for the Gated DeltaNet step, whose normalisation, gate
and beta are float32 in the kernel."""

import copy

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_map
from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache
from mlx_lm.models.qwen3_5 import GatedDeltaNet, Model, ModelArgs
from mlx_lm.models.qwen3_next import Qwen3NextAttention, Qwen3NextMLP

from exo.worker.engines.mlx import qwen3_5_decode
from exo.worker.engines.mlx.qwen3_5_decode import (
    AddNormDecoderLayer,
    FusedDecodeAttention,
    FusedDecodeGatedDeltaNet,
    FusedGateUpMLP,
    add_rms_norm,
    prepare_qwen3_5_decode,
)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the fused decode kernels are Metal kernels"
)

# hidden_size > 4096 so that the add + RMSNorm kernel (MLX's looped rms_norm) is exercised
HIDDEN = 4160


def _tiny_model() -> Model:
    mx.random.seed(3)
    args = ModelArgs.from_dict(
        {
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "vocab_size": 384,
                "hidden_size": HIDDEN,
                "intermediate_size": 512,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "max_position_embeddings": 4096,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": False,
                "attention_bias": False,
                "full_attention_interval": 4,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "rope_parameters": {
                    "type": "default",
                    "rope_theta": 10000000.0,
                    "partial_rotary_factor": 0.25,
                    "mrope_section": [11, 11, 10],
                },
            },
        }
    )
    model = Model(args)
    # scale the random init so activations stay in a realistic range
    model.update(tree_map(lambda p: p * 0.5, model.parameters()))
    model.set_dtype(mx.bfloat16)
    for layer in model.layers:
        if layer.is_linear:
            layer.linear_attn.A_log = mx.log(
                mx.random.uniform(low=1, high=16, shape=layer.linear_attn.A_log.shape)
            )
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    return model


def _snapshot(cache):
    out = []
    for c in cache:
        d = copy.copy(c)
        if isinstance(c, ArraysCache):
            d.cache = list(c.cache)
        out.append(d)
    return out


def _decode_logits(model, prompt, steps):
    cache = model.make_cache()
    logits = model(mx.array(prompt)[None], cache=cache)
    tokens = [int(mx.argmax(logits[0, -1]).item())]
    out = []
    for _ in range(steps):
        logits = model(mx.array([[tokens[-1]]]), cache=cache)[:, -1, :].astype(
            mx.float32
        )
        mx.eval(logits)
        out.append(logits)
        tokens.append(int(mx.argmax(logits).item()))
    return out, tokens


def test_installs_on_every_layer_without_extra_memory():
    model = _tiny_model()
    before = mx.get_active_memory()
    assert prepare_qwen3_5_decode(model)
    mx.clear_cache()
    assert mx.get_active_memory() <= before + 1024
    for layer in model.layers:
        assert isinstance(layer, AddNormDecoderLayer)
        assert isinstance(layer.mlp, FusedGateUpMLP)
        if layer.is_linear:
            assert isinstance(layer.linear_attn, FusedDecodeGatedDeltaNet)
        else:
            assert isinstance(layer.self_attn, FusedDecodeAttention)


def test_fused_projections_are_bitwise_the_separate_ones():
    model = _tiny_model()
    gdn = model.layers[0].linear_attn
    x1 = (mx.random.normal((1, 1, HIDDEN)) * 0.5).astype(mx.bfloat16)
    xs = (mx.random.normal((1, 9, HIDDEN)) * 0.5).astype(mx.bfloat16)
    modules = [gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a]
    reference = [[m(x) for m in modules] for x in (x1, xs)]
    mx.eval(reference)
    prepare_qwen3_5_decode(model)
    fused = gdn.__dict__["_exo_fused_projection"]
    for x, expected in zip((x1, xs), reference, strict=True):
        # the originals (views of the fused weights) are unchanged, for prefill too
        for module, value in zip(modules, expected, strict=True):
            assert mx.array_equal(module(x), value)
    # the fused matmul is used for single tokens
    out = fused(x1)
    start = 0
    for value in reference[0]:
        n = value.shape[-1]
        assert mx.array_equal(out[..., start : start + n], value)
        start += n


def test_decode_matches_the_original_model():
    prompt = list(range(5, 40))
    reference = _tiny_model()
    expected, tokens = _decode_logits(reference, prompt, 12)
    model = _tiny_model()
    prepare_qwen3_5_decode(model)
    cache = model.make_cache()
    model(mx.array(prompt)[None], cache=cache)
    for step, want in enumerate(expected):
        got = model(mx.array([[tokens[step]]]), cache=cache)[:, -1, :].astype(
            mx.float32
        )
        log_want = want - mx.logsumexp(want, axis=-1, keepdims=True)
        log_got = got - mx.logsumexp(got, axis=-1, keepdims=True)
        kl = (mx.exp(log_want) * (log_want - log_got)).sum()
        assert kl.item() < 1e-3, (step, kl.item())
        assert mx.argmax(got).item() == mx.argmax(want).item()


def test_attention_decode_is_exact():
    model = _tiny_model()
    layer = next(layer for layer in model.layers if not layer.is_linear)
    prompt = (mx.random.normal((1, 33, HIDDEN)) * 0.5).astype(mx.bfloat16)
    x = (mx.random.normal((1, 1, HIDDEN)) * 0.5).astype(mx.bfloat16)
    prepare_qwen3_5_decode(model)
    attention = layer.self_attn
    for cache_kind in ("kv", "batch"):
        caches = []
        for _ in range(2):
            c = KVCache() if cache_kind == "kv" else BatchKVCache([0])
            Qwen3NextAttention.__call__(
                attention,
                prompt,
                "causal" if cache_kind == "kv" else c.make_mask(33),
                c,
            )
            caches.append(c)
        mask = None if cache_kind == "kv" else caches[0].make_mask(1)
        want = Qwen3NextAttention.__call__(attention, x, mask, caches[0])
        got = attention(x, mask, caches[1])
        assert mx.array_equal(want, got), cache_kind
        assert mx.array_equal(caches[0].keys[..., :34, :], caches[1].keys[..., :34, :])


def test_add_rms_norm_is_exact():
    weight = (mx.random.normal((HIDDEN,)) * 0.3 + 1).astype(mx.bfloat16)
    for rows in (1, 3):
        x = (mx.random.normal((1, rows, HIDDEN)) * 4).astype(mx.bfloat16)
        r = (mx.random.normal((1, rows, HIDDEN)) * 4).astype(mx.bfloat16)
        h, out = add_rms_norm(x, r, weight, 1e-6)
        assert mx.array_equal(h, x + r)
        assert mx.array_equal(out, mx.fast.rms_norm(x + r, weight, 1e-6))


def test_gated_delta_decode_masked_rows():
    model = _tiny_model()
    gdn = model.layers[0].linear_attn
    prepare_qwen3_5_decode(model)
    batch_state = (
        mx.random.normal((2, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim)) * 0.05
    )
    conv_state = (mx.random.normal((2, 3, gdn.conv_dim)) * 0.5).astype(mx.bfloat16)
    x = (mx.random.normal((2, 1, HIDDEN)) * 0.5).astype(mx.bfloat16)
    mask = mx.array([True, False])[:, None]

    def run(call):
        cache = ArraysCache(2)
        cache[0], cache[1] = conv_state, batch_state
        out = call(gdn, x, mask, cache)
        mx.eval(out, cache.cache)
        return out, cache

    want, want_cache = run(GatedDeltaNet.__call__)
    got, got_cache = run(type(gdn).__call__)
    assert mx.array_equal(got_cache[0], want_cache[0])
    assert mx.allclose(got[0], want[0], atol=2e-2, rtol=2e-2)
    assert mx.allclose(got_cache[1][0], want_cache[1][0], atol=1e-3, rtol=1e-2)
    # the padded row: no output, state passes through
    assert mx.abs(got[1]).max().item() == 0
    assert mx.array_equal(got_cache[1][1], batch_state[1])


def test_cpu_falls_back_to_the_original_code():
    prompt = list(range(5, 30))
    reference = _tiny_model()
    model = _tiny_model()
    prepare_qwen3_5_decode(model)
    with mx.stream(mx.cpu):
        mx.set_default_device(mx.cpu)
        try:
            expected, tokens = _decode_logits(reference, prompt, 4)
            cache = model.make_cache()
            model(mx.array(prompt)[None], cache=cache)
            for step, want in enumerate(expected):
                got = model(mx.array([[tokens[step]]]), cache=cache)[:, -1, :]
                assert mx.array_equal(got.astype(mx.float32), want)
        finally:
            mx.set_default_device(mx.gpu)


def test_disabled_by_environment(monkeypatch):
    monkeypatch.setenv("EXO_QWEN3_5_FUSED_DECODE", "0")
    model = _tiny_model()
    assert not prepare_qwen3_5_decode(model)
    assert type(model.layers[0].linear_attn) is GatedDeltaNet
    assert type(model.layers[0].mlp) is Qwen3NextMLP


def test_other_models_are_left_alone():
    assert not prepare_qwen3_5_decode(nn.Linear(4, 4))
    assert qwen3_5_decode._unwrap(nn.Linear(4, 4)) is not None
