"""Inkling's MLX model on a tiny random configuration: loading the official checkpoint
layout, and caches (rotating sliding-window caches that wrap, short-convolution states,
batches of different lengths) against a full forward pass.

The window (6) and the relative-bias extents (6 and 10) are much shorter than the
sequences, so the bias band, the window and the ring buffer all matter.
"""

from typing import cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import RotatingKVCache

from exo.worker.engines.mlx.models.inkling.model import (
    Model,
    ModelArgs,
    key_distances,
)

_TEXT_CONFIG: dict[str, object] = {
    "hidden_size": 32,
    "num_hidden_layers": 4,
    "vocab_size": 96,
    "unpadded_vocab_size": 90,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 2,
    "swa_head_dim": 8,
    "sliding_window_size": 6,
    "d_rel": 4,
    "rel_extent": 10,
    "log_scaling_n_floor": 12,
    "log_scaling_alpha": 0.1,
    "rms_norm_eps": 1e-6,
    "local_layer_ids": [0, 1, 2],
    "dense_mlp_idx": 1,
    "dense_intermediate_size": 48,
    "intermediate_size": 16,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 2,
    "route_scale": 8.0,
    "logits_mup_width_multiplier": 16.0,
    "sconv_kernel_size": 4,
}
ARGS = ModelArgs(model_type="inkling_mm_model", text_config=_TEXT_CONFIG)
_SEQUENCE = [(11 * i + 5) % 90 for i in range(40)]


def tiny_model(seed: int = 0) -> Model:
    """Random weights at scales where every part (bias profiles, convolutions, scales,
    routing bias) changes the output."""
    mx.random.seed(seed)
    model = Model(ARGS)
    params: list[tuple[str, mx.array]] = []
    for name, value in cast(
        list[tuple[str, mx.array]], tree_flatten(model.parameters())
    ):
        if name.endswith(("global_scale", "norm.weight")):
            params.append((name, 0.5 + mx.random.uniform(shape=value.shape)))
        elif name.endswith("e_score_correction_bias"):
            params.append((name, 0.2 * mx.random.normal(value.shape)))
        elif name.endswith("rel_proj") or "sconv" in name:
            params.append((name, mx.random.normal(value.shape)))
        else:
            scale = value.shape[-1] ** -0.5 if value.ndim >= 2 else 1.0
            params.append((name, scale * mx.random.normal(value.shape)))
    model.load_weights(params)
    model.set_dtype(mx.float32)  # pyright: ignore[reportUnknownMemberType]
    mx.eval(model.parameters())
    return model


def _logits(model: Model, tokens: list[int], chunks: list[int]) -> np.ndarray:
    """Logits of every position: the given prefill chunks, then one token at a time."""
    cache = model.make_cache()
    rows: list[np.ndarray] = []
    position = 0
    for size in chunks:
        piece = mx.array(tokens[position : position + size])[None]
        rows.append(np.array(model(piece, cache=cache)[0]))
        position += size
    while position < len(tokens):
        piece = mx.array(tokens[position : position + 1])[None]
        rows.append(np.array(model(piece, cache=cache)[0]))
        position += 1
    return np.concatenate(rows)


@pytest.mark.parametrize("chunks", [[40], [3, 7, 11], [1], [13], [6, 6]])
def test_cached_generation_matches_a_full_forward(chunks: list[int]) -> None:
    model = tiny_model()
    full = np.array(model(mx.array(_SEQUENCE)[None])[0])
    cached = _logits(model, _SEQUENCE, chunks)
    np.testing.assert_allclose(cached, full, rtol=1e-4, atol=1e-4)


def test_long_prefills_split_into_query_blocks_give_the_same_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exo.worker.engines.mlx.models.inkling import model as inkling_model

    model = tiny_model()
    whole = np.array(model(mx.array(_SEQUENCE)[None])[0])
    monkeypatch.setattr(inkling_model, "_BIAS_ELEMENTS_PER_CALL", 64)
    monkeypatch.setattr(inkling_model, "_GLOBAL_BIAS_ELEMENTS_PER_CALL", 64)
    blocked = np.array(model(mx.array(_SEQUENCE)[None])[0])
    np.testing.assert_allclose(blocked, whole, rtol=1e-5, atol=1e-5)
    cached = _logits(model, _SEQUENCE, [17])
    np.testing.assert_allclose(cached, whole, rtol=1e-4, atol=1e-4)


def test_key_distances_follow_a_rotating_cache_that_wraps() -> None:
    cache = RotatingKVCache(max_size=5)
    positions: list[int] = []  # the position of the key in each slot the cache returns
    seen = 0
    for step in [3, 4, 1, 1, 1, 1, 1, 1, 6, 1, 1]:
        keys = mx.arange(seen, seen + step, dtype=mx.float32).reshape(1, 1, step, 1)
        returned, _ = cast(
            tuple[mx.array, mx.array],
            cache.update_and_fetch(keys, keys),  # pyright: ignore[reportUnknownMemberType]
        )
        positions = [int(p) for p in cast(list[float], returned[0, 0, :, 0].tolist())]
        seen += step
        distances = np.array(key_distances(cache, step, len(positions)))
        queries = np.arange(seen - step, seen)
        np.testing.assert_array_equal(
            distances, queries[:, None] - np.array(positions)[None, :]
        )


def _official_layout(model: Model) -> dict[str, mx.array]:
    """The model's weights as the official checkpoint stores them: ``model.llm.*``, fused
    and row-interleaved gate/up projections, convolutions as (channels, 1, kernel), the
    LM head padded to the full vocabulary, plus towers the text model doesn't load."""
    flat = dict(cast(list[tuple[str, mx.array]], tree_flatten(model.parameters())))
    # the fused input projections, split back into the reference's four
    for index, layer in enumerate(model.layers):
        fused = flat.pop(f"model.layers.{index}.self_attn.qkvr_proj.weight")
        start = 0
        widths = layer.self_attn.projection_widths()
        for projection, width in zip(("q", "k", "v", "r"), widths, strict=True):
            name = f"model.layers.{index}.self_attn.{projection}_proj.weight"
            flat[name] = fused[start : start + width]
            start += width
    # the experts' fused gate/up projections (gate rows, then up rows), split in two
    for name in [n for n in flat if n.endswith("experts.gate_up_proj.weight")]:
        fused = flat.pop(name)
        width = fused.shape[-2] // 2
        flat[name.replace("gate_up_proj", "gate_proj")] = fused[..., :width, :]
        flat[name.replace("gate_up_proj", "up_proj")] = fused[..., width:, :]
    out: dict[str, mx.array] = {}
    renames = {
        "self_attn.q_proj": "attn.wq_du",
        "self_attn.k_proj": "attn.wk_dv",
        "self_attn.v_proj": "attn.wv_dv",
        "self_attn.r_proj": "attn.wr_du",
        "self_attn.o_proj": "attn.wo_ud",
        "self_attn.rel_proj": "attn.rel_logits_proj.proj",
        "self_attn.": "attn.",
        "input_layernorm": "attn_norm",
        "post_attention_layernorm": "mlp_norm",
        "mlp.gate.e_score_correction_bias": "mlp.gate.bias",
        "mlp.down_proj": "mlp.w2_md",
        # The shared experts' names contain the routed experts': rename them first.
        "shared_experts.down_proj.weight": "shared_experts.shared_w2_weight",
        "experts.down_proj.weight": "experts.w2_weight",
    }
    for name, value in flat.items():
        if "up_proj" in name:
            continue
        if "gate_proj" in name:
            up = flat[name.replace("gate_proj", "up_proj")]
            fused = mx.stack([value, up], axis=-2).reshape(
                *value.shape[:-2], 2 * value.shape[-2], value.shape[-1]
            )
            name = name.replace(
                "shared_experts.gate_proj.weight", "shared_experts.shared_w13_weight"
            )
            name = name.replace("experts.gate_proj.weight", "experts.w13_weight")
            name = name.replace("mlp.gate_proj.weight", "mlp.w13_dn.weight")
            value = fused
        if name.endswith("sconv.weight"):
            value = value.swapaxes(1, 2)
        if name == "lm_head.weight":
            padding = mx.zeros((6, value.shape[1]), dtype=value.dtype)
            value = mx.concatenate([value, padding])
        for new, old in renames.items():
            name = name.replace(new, old)
        name = name.replace("model.embed_tokens", "embed").replace("lm_head", "unembed")
        name = name.removeprefix("model.")
        out[f"model.llm.{name}"] = value
    out["model.mtp.layers.0.input_proj.weight"] = mx.zeros((4, 4))
    out["model.visual.final_norm.weight"] = mx.zeros((4,))
    out["model.audio.encoder.weight"] = mx.zeros((4, 4))
    return out


def test_the_official_checkpoint_layout_loads_every_parameter() -> None:
    source = tiny_model(seed=1)
    loaded = Model(ARGS)
    weights = loaded.sanitize(_official_layout(source))
    loaded.load_weights(list(weights.items()), strict=True)
    expected = dict(cast(list[tuple[str, mx.array]], tree_flatten(source.parameters())))
    got = dict(cast(list[tuple[str, mx.array]], tree_flatten(loaded.parameters())))
    assert expected.keys() == got.keys()
    for name, value in expected.items():
        assert mx.array_equal(value, got[name]).item(), name


def test_the_lm_head_keeps_only_the_real_vocabulary() -> None:
    model = tiny_model()
    assert model(mx.array([[1, 2, 3]])).shape == (1, 3, 90)


def _greedy_alone(model: Model, prompt: list[int], steps: int) -> list[int]:
    cache = model.make_cache()
    logits = model(mx.array(prompt)[None], cache=cache)[:, -1]
    tokens: list[int] = []
    for _ in range(steps):
        token = int(mx.argmax(logits, axis=-1).item())
        tokens.append(token)
        logits = model(mx.array([[token]]), cache=cache)[:, -1]
    return tokens


def test_a_batch_of_different_lengths_matches_each_prompt_alone() -> None:
    model = tiny_model()
    prompts = [_SEQUENCE[:23], _SEQUENCE[5:9], _SEQUENCE[2:16]]
    steps = 12
    generator = BatchGenerator(model, max_tokens=steps, prefill_step_size=7)
    uids = generator.insert(prompts)
    produced: dict[int, list[int]] = {uid: [] for uid in uids}
    while any(len(tokens) < steps for tokens in produced.values()):
        for response in generator.next_generated():
            produced[response.uid].append(response.token)
    generator.close()
    for uid, prompt in zip(uids, prompts, strict=True):
        assert produced[uid] == _greedy_alone(model, prompt, steps)


def test_unsupported_settings_are_refused() -> None:
    config = dict(_TEXT_CONFIG, gate_activation="softmax")
    with pytest.raises(ValueError, match="gate_activation"):
        Model(ModelArgs(model_type="inkling_mm_model", text_config=config))


def test_dense_and_sparse_layers_and_attention_kinds_follow_the_config() -> None:
    model = Model(ARGS)
    kinds = [(layer.is_sliding, type(layer.mlp).__name__) for layer in model.layers]
    assert kinds == [
        (True, "MLP"),
        (True, "MoE"),
        (True, "MoE"),
        (False, "MoE"),
    ]
    assert isinstance(model.layers[0].mlp, nn.Module)


def test_a_layout_it_cannot_read_is_refused_rather_than_run_on_initial_weights() -> (
    None
):
    # e.g. a conversion in another library's layout (language_model.model.layers.*)
    foreign = {
        "language_model.model.embed_tokens.weight": mx.zeros((96, 32)),
        "language_model.lm_head.weight": mx.zeros((90, 32)),
    }
    with pytest.raises(ValueError, match="Unrecognised Inkling checkpoint layout"):
        Model(ARGS).sanitize(foreign)


_QUANTIZABLE_CONFIG: dict[str, object] = dict(
    _TEXT_CONFIG,
    hidden_size=64,
    dense_intermediate_size=64,
    intermediate_size=32,
    head_dim=16,
    swa_head_dim=16,
)


def test_quantized_experts_compute_what_their_dequantized_weights_do() -> None:
    """The MLX conversions quantize the routed and shared experts and the dense MLPs (the
    SwitchLinear gathers and plain linears): run on those weights, the model must give
    what it gives with the dequantized values."""
    args = ModelArgs(model_type="inkling_mm_model", text_config=_QUANTIZABLE_CONFIG)
    mx.random.seed(3)
    quantized = Model(args)
    quantized.set_dtype(mx.float32)  # pyright: ignore[reportUnknownMemberType]
    nn.quantize(
        quantized,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: (
            hasattr(module, "to_quantized") and (".mlp." in path)
        ),
    )
    mx.eval(quantized.parameters())
    flat = dict(cast(list[tuple[str, mx.array]], tree_flatten(quantized.parameters())))
    for kind in (
        "mlp.gate_proj",
        "mlp.experts.down_proj",
        "mlp.shared_experts.gate_up_proj",
    ):
        assert any(n.endswith(f"{kind}.scales") for n in flat), kind
    dequantized_weights: list[tuple[str, mx.array]] = []
    for name, value in flat.items():
        if name.endswith((".scales", ".biases")):
            continue
        prefix = name.removesuffix(".weight")
        if f"{prefix}.scales" in flat:
            value = mx.dequantize(
                value, flat[f"{prefix}.scales"], flat[f"{prefix}.biases"], 32, 4
            )
        dequantized_weights.append((name, value))
    dequantized = Model(args)
    dequantized.load_weights(dequantized_weights, strict=True)
    dequantized.set_dtype(mx.float32)  # pyright: ignore[reportUnknownMemberType]
    tokens = mx.array(_SEQUENCE)[None]
    np.testing.assert_allclose(
        np.array(quantized(tokens)), np.array(dequantized(tokens)), rtol=1e-4, atol=1e-4
    )


def test_mlx_conversions_load_with_fused_projections() -> None:
    """MLX conversions keep the official names but split (and quantize) the experts' gate
    and up projections: the loader stacks them as gate rows then up rows, and stacks each
    attention layer's q/k/v/r projections."""
    source = tiny_model(seed=2)
    official = _official_layout(source)
    converted: dict[str, mx.array] = {}
    for name, value in official.items():
        if name.endswith(("experts.w13_weight", "shared_experts.shared_w13_weight")):
            base = name.removesuffix(".w13_weight").removesuffix(".shared_w13_weight")
            converted[f"{base}.gate_proj.weight"] = value[..., 0::2, :]
            converted[f"{base}.up_proj.weight"] = value[..., 1::2, :]
        else:
            converted[name] = value
    loaded = Model(ARGS)
    loaded.load_weights(list(loaded.sanitize(converted).items()), strict=True)
    expected = dict(cast(list[tuple[str, mx.array]], tree_flatten(source.parameters())))
    got = dict(cast(list[tuple[str, mx.array]], tree_flatten(loaded.parameters())))
    assert expected.keys() == got.keys()
    for name, value in expected.items():
        assert mx.array_equal(value, got[name]).item(), name
