import json
from pathlib import Path

import pytest

from exo.backends.tinygrad_weights import (
    SafetensorEntry,
    TinygradModelSupportError,
    checkpoint_tensor_entries,
    load_architecture,
    select_shard_tensor_names,
    shard_parameter_byte_count,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import PipelineShardMetadata

_TENSOR_NAMES = [
    "model.embed_tokens.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.1.self_attn.q_proj.weight",
    "model.layers.2.mlp.down_proj.weight",
    "model.layers.3.post_attention_layernorm.weight",
    "model.norm.weight",
    "lm_head.weight",
]


def test_middle_rank_keeps_only_its_layers() -> None:
    selected = select_shard_tensor_names(
        _TENSOR_NAMES,
        _shard(start_layer=1, end_layer=3, layer_count=4),
        embeddings_are_tied=False,
    )
    assert selected == frozenset(
        {
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.2.mlp.down_proj.weight",
        }
    )


def test_first_rank_keeps_embeddings_and_not_the_head() -> None:
    selected = select_shard_tensor_names(
        _TENSOR_NAMES,
        _shard(start_layer=0, end_layer=1, layer_count=4),
        embeddings_are_tied=True,
    )
    assert "model.embed_tokens.weight" in selected
    assert "model.layers.0.input_layernorm.weight" in selected
    assert "model.norm.weight" not in selected
    assert "lm_head.weight" not in selected


def test_last_rank_keeps_norm_and_language_model_head() -> None:
    selected = select_shard_tensor_names(
        _TENSOR_NAMES,
        _shard(start_layer=3, end_layer=4, layer_count=4),
        embeddings_are_tied=True,
    )
    assert selected == frozenset(
        {
            "model.layers.3.post_attention_layernorm.weight",
            "model.norm.weight",
            "lm_head.weight",
        }
    )


def test_tied_language_model_head_loads_embeddings_on_the_last_rank() -> None:
    names = [name for name in _TENSOR_NAMES if name != "lm_head.weight"]
    selected = select_shard_tensor_names(
        names,
        _shard(start_layer=3, end_layer=4, layer_count=4),
        embeddings_are_tied=True,
    )
    assert "model.embed_tokens.weight" in selected
    assert "model.norm.weight" in selected
    assert "model.layers.3.post_attention_layernorm.weight" in selected


def test_selected_header_bytes_ignore_other_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path, tie_word_embeddings=False)
    _write_safetensors(
        tmp_path / "model.safetensors",
        {
            "model.layers.0.input_layernorm.weight": ("F32", [4], b"\x00" * 16),
            "model.layers.1.input_layernorm.weight": ("F16", [4], b"\x00" * 8),
            "model.norm.weight": ("F32", [4], b"\x00" * 16),
        },
    )

    def use_temporary_directory(model_id: ModelId) -> Path:
        _ = model_id
        return tmp_path

    monkeypatch.setattr(
        "exo.backends.tinygrad_weights.build_model_path",
        use_temporary_directory,
    )
    byte_count = shard_parameter_byte_count(
        _shard(start_layer=0, end_layer=1, layer_count=2)
    )
    assert byte_count == 16


def test_index_uses_the_global_name_set_for_tied_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path, tie_word_embeddings=True)
    _write_safetensors(
        tmp_path / "a.safetensors",
        {"model.embed_tokens.weight": ("F32", [2], b"\x00" * 8)},
    )
    _write_safetensors(
        tmp_path / "b.safetensors",
        {
            "model.layers.1.input_layernorm.weight": ("F32", [2], b"\x00" * 8),
            "model.norm.weight": ("F32", [2], b"\x00" * 8),
        },
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "a.safetensors",
                    "model.layers.1.input_layernorm.weight": "b.safetensors",
                    "model.norm.weight": "b.safetensors",
                }
            }
        )
    )

    def use_temporary_directory(model_id: ModelId) -> Path:
        _ = model_id
        return tmp_path

    monkeypatch.setattr(
        "exo.backends.tinygrad_weights.build_model_path",
        use_temporary_directory,
    )
    entries = checkpoint_tensor_entries(tmp_path)
    assert entries["model.embed_tokens.weight"] == SafetensorEntry("a.safetensors", 8)
    byte_count = shard_parameter_byte_count(
        _shard(start_layer=1, end_layer=2, layer_count=2)
    )
    assert byte_count == 8 + 8 + 8


def test_unsupported_model_type_is_rejected(tmp_path: Path) -> None:
    _write_config(tmp_path, tie_word_embeddings=False, model_type="mistral")
    with pytest.raises(TinygradModelSupportError, match="mistral"):
        load_architecture(tmp_path)


def _shard(
    *, start_layer: int, end_layer: int, layer_count: int
) -> PipelineShardMetadata:
    model_card = ModelCard(
        model_id=ModelId("tinygrad-model"),
        storage_size=Memory.from_mb(16),
        n_layers=layer_count,
        hidden_size=32,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.TinygradCpu],
    )
    return PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=start_layer,
        end_layer=end_layer,
        n_layers=layer_count,
    )


def _write_config(
    directory: Path, *, tie_word_embeddings: bool, model_type: str = "llama"
) -> None:
    payload = {
        "model_type": model_type,
        "hidden_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "intermediate_size": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000,
        "vocab_size": 8,
        "num_hidden_layers": 2,
        "tie_word_embeddings": tie_word_embeddings,
    }
    (directory / "config.json").write_text(json.dumps(payload))


def _write_safetensors(
    path: Path, tensors: dict[str, tuple[str, list[int], bytes]]
) -> None:
    header: dict[str, object] = {}
    offset = 0
    body = bytearray()
    for name, (dtype_name, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype_name,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        body.extend(raw)
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + bytes(body))
