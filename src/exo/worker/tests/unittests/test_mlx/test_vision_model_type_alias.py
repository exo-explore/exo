# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportPrivateUsage=false
"""Qwen3.8-Flash-Next's checkpoint names its vision config ``qwen4_exp``. Its vision tower is
Qwen3.5's, and mlx-vlm's Qwen3.5 vision model refuses any other type name, so exo gives an
aliased model's vision config its alias's name (models/qwen3/spec.py)."""

import pytest
from mlx_vlm.models.qwen3_5.config import VisionConfig
from mlx_vlm.models.qwen3_5.vision import VisionModel

from exo.worker.engines.mlx.vision import _alias_vision_model_type, _filter_config

# The checkpoint's vision_config (mlx-community/Qwen3.8-Flash-Next-4bit), at tiny sizes
CHECKPOINT_VISION_CONFIG: dict[str, object] = {
    "deepstack_visual_indexes": [],
    "depth": 1,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 32,
    "in_channels": 3,
    "intermediate_size": 64,
    "model_type": "qwen4_exp",
    "num_heads": 2,
    "num_position_embeddings": 16,
    "out_hidden_size": 32,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
}


def test_qwen4_exp_vision_config_loads_as_qwen3_5() -> None:
    config = VisionConfig(**_filter_config(VisionConfig, CHECKPOINT_VISION_CONFIG))
    with pytest.raises(ValueError, match="Unsupported model type"):
        VisionModel(config)
    _alias_vision_model_type(config, "qwen4_exp")
    assert config.model_type == "qwen3_5"
    VisionModel(config)


def test_a_model_without_an_alias_keeps_its_type_name() -> None:
    config = VisionConfig(
        **_filter_config(
            VisionConfig, {**CHECKPOINT_VISION_CONFIG, "model_type": "qwen3_5"}
        )
    )
    _alias_vision_model_type(config, "qwen3_5")
    assert config.model_type == "qwen3_5"
