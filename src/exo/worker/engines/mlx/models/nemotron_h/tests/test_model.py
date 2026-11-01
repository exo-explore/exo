"""The Nemotron-H model classes exo loads: mlx-lm's, in float32 where NVIDIA's reference is."""

from typing import Any, cast

import mlx.core as mx
from mlx_lm.models.nemotron_h import Model as MlxLmModel
from mlx_lm.models.nemotron_h import ModelArgs as MlxLmModelArgs
from mlx_lm.models.nemotron_h import (
    NemotronHMamba2Mixer,
    NemotronHMoE,
    group_expert_select,
)

from exo.worker.engines.mlx.models import registry
from exo.worker.engines.mlx.models.nemotron_h.model import (
    Float32Mamba2Mixer,
    Float32RouterGate,
    Model,
    ModelArgs,
)
from exo.worker.engines.mlx.models.nemotron_h.tests.tiny import CONFIG, PROMPT


def test_nemotron_h_loads_with_the_family_model_classes() -> None:
    assert registry.get_model_classes(CONFIG) == (Model, ModelArgs)


def test_every_moe_layer_routes_in_float32() -> None:
    model = Model(ModelArgs.from_dict(CONFIG))
    moe_layers = [
        layer.mixer for layer in model.layers if isinstance(layer.mixer, NemotronHMoE)
    ]
    assert moe_layers
    assert all(isinstance(moe.gate, Float32RouterGate) for moe in moe_layers)


def test_router_logits_are_computed_in_float32() -> None:
    mx.random.seed(3)
    args = ModelArgs.from_dict(CONFIG)
    gate = Float32RouterGate(args)
    gate.update(
        {
            "weight": mx.random.normal(gate.weight.shape).astype(mx.bfloat16),
            "e_score_correction_bias": mx.random.normal((gate.n_routed_experts,)),
        }
    )
    x = mx.random.normal((16, args.hidden_size)).astype(mx.bfloat16)

    indices, scores = gate(x)

    expected_indices, expected_scores = group_expert_select(
        x.astype(mx.float32) @ gate.weight.astype(mx.float32).T,
        gate.e_score_correction_bias,
        gate.top_k,
        gate.n_group,
        gate.topk_group,
        gate.routed_scaling_factor,
        gate.norm_topk_prob,
    )
    assert mx.array_equal(indices, expected_indices)
    assert mx.array_equal(scores, expected_scores)


def test_mamba_mixers_run_in_float32_between_in_proj_and_out_proj() -> None:
    model = Model(ModelArgs.from_dict(CONFIG))
    mixers = [
        layer.mixer
        for layer in model.layers
        if isinstance(layer.mixer, NemotronHMamba2Mixer)
    ]
    assert mixers
    assert all(isinstance(mixer, Float32Mamba2Mixer) for mixer in mixers)


def test_float32_weights_give_mlx_lm_outputs() -> None:
    # In float32 the family's model computes what mlx-lm's does.
    mx.random.seed(4)
    reference = MlxLmModel(MlxLmModelArgs.from_dict(CONFIG))
    model = Model(ModelArgs.from_dict(CONFIG))
    model.update(cast(dict[str, Any], reference.parameters()))
    inputs = mx.array(PROMPT)

    assert mx.allclose(model(inputs), reference(inputs), atol=1e-5, rtol=1e-5)


def test_bf16_mixer_rounds_only_for_out_proj() -> None:
    mx.random.seed(5)
    model = Model(ModelArgs.from_dict(CONFIG))
    model.set_dtype(mx.bfloat16)  # pyright: ignore[reportUnknownMemberType]
    mixer = next(
        layer.mixer
        for layer in model.layers
        if isinstance(layer.mixer, Float32Mamba2Mixer)
    )
    hidden = cast(int, CONFIG["hidden_size"])
    x = mx.random.normal((1, 8, hidden)).astype(mx.bfloat16)
    conv_input = mx.random.normal((1, 8, mixer.conv_dim)).astype(mx.bfloat16)

    assert mixer.conv1d(conv_input).dtype == mx.float32
    assert mixer(x, mask=None).dtype == mx.bfloat16
