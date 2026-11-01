"""load_model_checked refuses checkpoints that leave model parameters unset."""

import json
from pathlib import Path
from typing import cast

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.llama import Model, ModelArgs

from exo.worker.engines.mlx.utils_mlx import MissingWeightsError, load_model_checked

CONFIG = {
    "model_type": "llama",
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-5,
    "vocab_size": 64,
    "tie_word_embeddings": False,
}


def _write_checkpoint(directory: Path, drop: str | None = None) -> None:
    model = Model(ModelArgs.from_dict(CONFIG))  # pyright: ignore[reportUnknownMemberType]
    weights = dict(cast(list[tuple[str, mx.array]], tree_flatten(model.parameters())))
    if drop is not None:
        del weights[drop]
    weights["model.layers.9.extra.weight"] = mx.zeros((2,))  # unused tensor: tolerated
    mx.save_safetensors(str(directory / "model.safetensors"), weights)  # pyright: ignore[reportUnknownMemberType]
    (directory / "config.json").write_text(json.dumps(CONFIG))


def test_complete_checkpoint_loads(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    model = load_model_checked(tmp_path)
    assert len(tree_flatten(model.parameters())) > 0


def test_missing_parameter_is_refused(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path, drop="model.layers.1.self_attn.q_proj.weight")
    with pytest.raises(MissingWeightsError, match="model.layers.1.self_attn.q_proj"):
        load_model_checked(tmp_path)
