"""Text-path loading of Nemotron-H Omni checkpoints (model_type nemotron_h_omni)."""

import json
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.utils import load_model

from exo.worker.engines.mlx.models import registry
from exo.worker.engines.mlx.models.nemotron_h import omni
from exo.worker.engines.mlx.models.nemotron_h.model import Model as NemotronHModel
from exo.worker.engines.mlx.models.nemotron_h.model import (
    ModelArgs as NemotronHModelArgs,
)
from exo.worker.engines.mlx.models.nemotron_h.tests.tiny import CONFIG, PROMPT

# An omni config as NVIDIA ships it: the language model's config under `llm_config`, with
# the infinite time-step limit written the way transformers serializes it.
OMNI_CONFIG: dict[str, Any] = {
    "architectures": ["NemotronH_Omni_Reasoning_V3"],
    "model_type": "nemotron_h_omni",
    "llm_config": {
        **CONFIG,
        "time_step_limit": [0.0, {"__float__": "Infinity"}],
    },
    "vision_config": {"model_type": "radio"},
}


def test_omni_checkpoints_load_with_the_family_model_classes() -> None:
    assert registry.get_model_classes(OMNI_CONFIG) == (omni.Model, omni.ModelArgs)


def test_omni_model_shards_like_nemotron_h() -> None:
    architecture = registry.architecture_for_model_class(omni.Model)
    assert architecture is not None
    assert architecture.model_class is NemotronHModel


def test_omni_checkpoint_loads_its_language_model_only(tmp_path: Path) -> None:
    mx.random.seed(0)
    language_model = NemotronHModel(NemotronHModelArgs.from_dict(CONFIG))
    weights: dict[str, mx.array] = {
        f"{omni.LANGUAGE_MODEL_PREFIX}{name}": value
        for name, value in cast(
            list[tuple[str, mx.array]], tree_flatten(language_model.parameters())
        )
    }
    # What the omni checkpoint also holds: the vision tower, its projector and the MTP layer.
    weights["vision_model.radio_model.model.blocks.0.attn.qkv.weight"] = mx.zeros(
        (8, 8)
    )
    weights["mlp1.1.weight"] = mx.zeros((8, 8))
    weights["language_model.mtp.layers.0.norm.weight"] = mx.zeros((8,))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)  # pyright: ignore[reportUnknownMemberType]
    (tmp_path / "config.json").write_text(json.dumps(OMNI_CONFIG))

    loaded, _ = load_model(
        tmp_path, strict=True, get_model_classes=registry.get_model_classes
    )

    assert isinstance(loaded, omni.Model)
    assert loaded.args.time_step_limit is not None
    assert tuple(loaded.args.time_step_limit) == (0.0, float("inf"))
    inputs = mx.array(PROMPT)
    assert mx.array_equal(loaded(inputs), language_model(inputs))
