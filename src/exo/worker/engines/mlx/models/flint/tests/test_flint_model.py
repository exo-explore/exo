"""exo's Flint model: mlx-lm's, with RMSNorms computed as Flint's reference does."""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.flint import Model as MlxLmModel

from exo.worker.engines.mlx.models.flint.model import Float32RMSNorm, Model, ModelArgs
from exo.worker.engines.mlx.models.registry import get_model_classes

_ARGS = ModelArgs(
    model_type="flint",
    hidden_size=64,
    num_hidden_layers=3,
    intermediate_size=96,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    mlp_only_layers=[0],
    layer_types=["full_attention", "sliding_attention", "full_attention"],
    sliding_window=5,
    rms_norm_eps=1e-6,
    vocab_size=64,
)


def test_flint_checkpoints_load_with_exos_model() -> None:
    assert get_model_classes({"model_type": "flint"}) == (Model, ModelArgs)


def test_every_norm_is_float32_and_parameters_keep_their_names() -> None:
    model = Model(_ARGS)
    norms: list[nn.Module] = [model.model.norm]
    for layer in model.model.layers:
        norms += [
            layer.input_layernorm,
            layer.post_attention_layernorm,
            layer.self_attn.q_norm,
            layer.self_attn.k_norm,
        ]
    assert all(isinstance(norm, Float32RMSNorm) for norm in norms)
    exo_names = sorted(name for name, _ in tree_flatten(model.parameters()))
    mlx_lm_names = sorted(
        name for name, _ in tree_flatten(MlxLmModel(_ARGS).parameters())
    )
    assert exo_names == mlx_lm_names
    # Every norm weight in the model is one of those modules'.
    assert len([n for n in exo_names if n.endswith("norm.weight")]) == len(norms)


def test_float32_norm_rounds_once_like_the_reference() -> None:
    """transformers' FlintRMSNorm: (weight * normalised float32).to(bf16)."""
    mx.random.seed(0)
    x = (mx.random.normal((3, 7, 128)) * 5).astype(mx.bfloat16)
    norm = Float32RMSNorm(128, eps=1e-6)
    norm.weight = (mx.random.normal((128,)) * 0.3 + 1).astype(mx.bfloat16)

    x32 = x.astype(mx.float32)
    normalised = x32 * mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + 1e-6)
    expected = (norm.weight.astype(mx.float32) * normalised).astype(mx.bfloat16)

    out = norm(x)
    assert out.dtype == mx.bfloat16
    # float32 rsqrt/mean may differ in the last float32 bit, which can move a bf16
    # rounding by one step on rare elements
    differing = mx.mean(mx.not_equal(out, expected).astype(mx.float32)).item()
    assert differing < 0.01
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=2**-7
    ).item()
