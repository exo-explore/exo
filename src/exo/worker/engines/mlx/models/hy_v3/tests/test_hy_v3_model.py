# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""exo's Hy3 port against Hugging Face transformers' ``hy_v3`` (the reference it was ported
from), on a tiny random model in float32, loaded from the official checkpoint layout."""

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache

from exo.worker.engines.mlx.models.hy_v3.model import (
    Model,
    ModelArgs,
    MoE,
    float32_router_logits,
)
from exo.worker.engines.mlx.models.registry import get_model_classes

_CONFIG: dict[str, Any] = {
    "model_type": "hy_v3",
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 3,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "rms_norm_eps": 1e-5,
    "num_experts": 16,
    "num_experts_per_tok": 4,
    "num_shared_experts": 1,
    "moe_intermediate_size": 32,
    "router_scaling_factor": 2.826,
    "rope_parameters": {"rope_theta": 11158840.0, "rope_type": "default"},
    "first_k_dense_replace": 1,
    "enable_moe_fp32_combine": False,
    "num_nextn_predict_layers": 1,
    "tie_word_embeddings": False,
}
_PROMPT = [(11 * i + 5) % 128 for i in range(19)]


def test_hy3_checkpoints_load_with_exos_model() -> None:
    assert get_model_classes({"model_type": "hy_v3"}) == (Model, ModelArgs)


def test_first_layer_dense_then_sparse() -> None:
    model = Model(ModelArgs.from_dict(_CONFIG))
    assert [isinstance(layer.mlp, MoE) for layer in model.layers] == [
        False,
        True,
        True,
    ]


def _official_layout(n_layers: int, args: ModelArgs) -> dict[str, mx.array]:
    """Random weights under the official checkpoint's names, with an MTP layer."""
    rng = np.random.default_rng(0)

    def tensor(*shape: int) -> mx.array:
        return mx.array(rng.standard_normal(shape).astype(np.float32) * 0.05)

    hidden, heads, kv, head_dim = (
        args.hidden_size,
        args.num_attention_heads,
        args.num_key_value_heads,
        args.head_dim,
    )
    weights: dict[str, mx.array] = {
        "model.embed_tokens.weight": tensor(args.vocab_size, hidden),
        "model.norm.weight": 1 + tensor(hidden),
        "lm_head.weight": tensor(args.vocab_size, hidden),
    }
    for i in range(n_layers + 1):  # + the MTP layer
        p = f"model.layers.{i}"
        weights |= {
            f"{p}.input_layernorm.weight": 1 + tensor(hidden),
            f"{p}.post_attention_layernorm.weight": 1 + tensor(hidden),
            f"{p}.self_attn.q_proj.weight": tensor(heads * head_dim, hidden),
            f"{p}.self_attn.k_proj.weight": tensor(kv * head_dim, hidden),
            f"{p}.self_attn.v_proj.weight": tensor(kv * head_dim, hidden),
            f"{p}.self_attn.o_proj.weight": tensor(hidden, heads * head_dim),
            f"{p}.self_attn.q_norm.weight": 1 + tensor(head_dim),
            f"{p}.self_attn.k_norm.weight": 1 + tensor(head_dim),
        }
        if i == 0:
            for name, shape in (
                ("gate_proj", (args.intermediate_size, hidden)),
                ("up_proj", (args.intermediate_size, hidden)),
                ("down_proj", (hidden, args.intermediate_size)),
            ):
                weights[f"{p}.mlp.{name}.weight"] = tensor(*shape)
            continue
        inter = args.moe_intermediate_size
        weights[f"{p}.mlp.router.gate.weight"] = tensor(args.num_experts, hidden)
        weights[f"{p}.mlp.expert_bias"] = tensor(args.num_experts) * 10
        for e in range(args.num_experts):
            weights[f"{p}.mlp.experts.{e}.gate_proj.weight"] = tensor(inter, hidden)
            weights[f"{p}.mlp.experts.{e}.up_proj.weight"] = tensor(inter, hidden)
            weights[f"{p}.mlp.experts.{e}.down_proj.weight"] = tensor(hidden, inter)
        for name, shape in (
            ("gate_proj", (inter, hidden)),
            ("up_proj", (inter, hidden)),
            ("down_proj", (hidden, inter)),
        ):
            weights[f"{p}.mlp.shared_mlp.{name}.weight"] = tensor(*shape)
    weights[f"model.layers.{n_layers}.eh_proj.weight"] = tensor(hidden, 2 * hidden)
    return weights


def _loaded_model() -> tuple[Model, dict[str, mx.array]]:
    args = ModelArgs.from_dict(_CONFIG)
    official = _official_layout(args.num_hidden_layers, args)
    model = Model(args)
    model.load_weights(list(model.sanitize(dict(official)).items()), strict=True)
    mx.eval(model.parameters())
    return model, official


def test_sanitize_maps_the_official_layout_onto_every_parameter() -> None:
    model, official = _loaded_model()
    names = {name for name, _ in tree_flatten(model.parameters())}
    sanitized = model.sanitize(dict(official))
    assert set(sanitized) == names
    assert not any(name.startswith("model.layers.3.") for name in sanitized)
    stacked = sanitized["model.layers.2.mlp.switch_mlp.down_proj.weight"]
    assert mx.array_equal(
        stacked[5], official["model.layers.2.mlp.experts.5.down_proj.weight"]
    )
    assert mx.array_equal(
        sanitized["model.layers.1.mlp.router.expert_bias"],
        official["model.layers.1.mlp.expert_bias"],
    )


def _transformers_logits(official: dict[str, mx.array]) -> np.ndarray:
    torch = pytest.importorskip("torch")
    hy_v3 = pytest.importorskip("transformers.models.hy_v3")
    config = hy_v3.HYV3Config(
        **{k: v for k, v in _CONFIG.items() if k != "model_type"},
    )
    config._attn_implementation = "eager"
    reference = hy_v3.HYV3ForCausalLM(config).to(torch.float32).eval()
    state: dict[str, Any] = {}
    n_layers = int(_CONFIG["num_hidden_layers"])
    for name, value in official.items():
        if name.startswith(f"model.layers.{n_layers}."):
            continue
        tensor = torch.from_numpy(np.array(value))
        name = name.replace("mlp.router.gate.", "mlp.gate.")
        name = name.replace("mlp.expert_bias", "mlp.e_score_correction_bias")
        name = name.replace("mlp.shared_mlp.", "mlp.shared_experts.")
        state[name] = tensor
    for i in range(1, n_layers):
        p = f"model.layers.{i}.mlp.experts"
        experts = range(int(_CONFIG["num_experts"]))
        gate = [state.pop(f"{p}.{e}.gate_proj.weight") for e in experts]
        up = [state.pop(f"{p}.{e}.up_proj.weight") for e in experts]
        down = [state.pop(f"{p}.{e}.down_proj.weight") for e in experts]
        state[f"{p}.gate_up_proj"] = torch.stack(
            [torch.cat([g, u], dim=0) for g, u in zip(gate, up, strict=True)]
        )
        state[f"{p}.down_proj"] = torch.stack(down)
    missing, unexpected = reference.load_state_dict(state, strict=False)
    assert not unexpected and not [m for m in missing if "rotary" not in m]
    with torch.no_grad():
        logits = reference(input_ids=torch.tensor([_PROMPT])).logits
    return logits[0].numpy()


def test_matches_transformers_in_float32() -> None:
    model, official = _loaded_model()
    expected = _transformers_logits(official)
    got = np.array(model(mx.array([_PROMPT]))[0])
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
    assert (got.argmax(-1) == expected.argmax(-1)).all()


def test_incremental_decode_matches_one_forward() -> None:
    model, _ = _loaded_model()
    full = np.array(model(mx.array([_PROMPT]))[0])
    cache = [KVCache() for _ in model.layers]
    steps = [np.array(model(mx.array([_PROMPT[:7]]), cache=cache)[0])]
    for token in _PROMPT[7:]:
        steps.append(np.array(model(mx.array([[token]]), cache=cache)[0]))
    np.testing.assert_allclose(np.concatenate(steps), full, rtol=1e-5, atol=1e-5)


def test_router_logits_are_float32_like_the_reference() -> None:
    """transformers: F.linear(hidden_states.float(), weight.float())."""
    mx.random.seed(3)
    x = mx.random.normal((5, 128)).astype(mx.bfloat16)
    gate = nn.Linear(128, 32, bias=False)
    gate.set_dtype(mx.bfloat16)
    expected = x.astype(mx.float32) @ gate.weight.astype(mx.float32).T
    got = float32_router_logits(gate, x)
    assert got.dtype == mx.float32
    assert mx.array_equal(got, expected)

    quantized = nn.QuantizedLinear.from_linear(gate, group_size=64, bits=8)
    weight = mx.dequantize(
        quantized.weight,
        quantized.scales.astype(mx.float32),
        quantized.biases.astype(mx.float32),
        group_size=64,
        bits=8,
    )
    got = float32_router_logits(quantized, x)
    assert got.dtype == mx.float32
    np.testing.assert_allclose(
        np.array(got), np.array(x.astype(mx.float32) @ weight.T), rtol=1e-5, atol=1e-5
    )
