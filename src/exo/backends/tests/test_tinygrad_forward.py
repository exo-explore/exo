from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest

if TYPE_CHECKING:
    from tinygrad.tensor import Tensor

from exo.backends.tinygrad_hidden_state import (
    HiddenStateBuffer,
    TokenIdBuffer,
    hidden_state_to_tensor,
    tensor_to_hidden_state,
)
from exo.backends.tinygrad_weights import (
    RopeScalingConfig,
    TinygradShardRoleError,
    TinygradWeightError,
    TransformerArchitecture,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.instances import (
    BoundInstance,
    InstanceId,
    TinygradInstance,
)
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata


def test_hidden_state_length_mismatch_is_rejected() -> None:
    with pytest.raises(TinygradWeightError, match="bytes"):
        hidden_state_to_tensor(
            HiddenStateBuffer(dtype="float32", shape=(2,), data=b"\x00\x00")
        )


def test_hidden_state_round_trip_preserves_bytes() -> None:
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor, dtypes
    from tinygrad.dtype import DType
    from tinygrad.helpers import DEV

    from exo.backends.tinygrad_hidden_state import HiddenStateDTypeName

    DEV.value = "CPU"
    cases: tuple[tuple[HiddenStateDTypeName, DType, tuple[int, ...]], ...] = (
        ("float16", dtypes.float16, (2, 3)),
        ("bfloat16", dtypes.bfloat16, (2, 2)),
        ("float32", dtypes.float32, (1, 4)),
    )
    for dtype_name, tinygrad_dtype, shape in cases:
        source = Tensor.randn(*shape, dtype=tinygrad_dtype).contiguous().realize()
        raw = source.bitcast(dtypes.uint8).contiguous().realize().numpy().tobytes()
        buffer = HiddenStateBuffer(dtype=dtype_name, shape=shape, data=raw)
        restored = tensor_to_hidden_state(hidden_state_to_tensor(buffer))
        assert restored.dtype == dtype_name
        assert restored.shape == shape
        assert restored.data == raw


def test_llama_block_preserves_hidden_state_shape() -> None:
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor

    from exo.backends.tinygrad_engine import TinygradEngine
    from exo.backends.tinygrad_llama import LocalKeyValueCache, assemble_loaded_shard

    architecture = _architecture("llama", rope_scaling=_llama3_scaling())
    loaded = assemble_loaded_shard(
        _random_parameters(
            include_query_key_norm=False, include_language_model_head=True
        ),
        architecture,
        _shard(start_layer=0, end_layer=1, layer_count=1),
    )
    engine = TinygradEngine(device_name="CPU")
    engine.loaded_shard = loaded
    engine.key_value_cache = LocalKeyValueCache(len(loaded.layers))

    hidden = tensor_to_hidden_state(Tensor.randn(1, 3, 16).realize())
    output = engine.forward_hidden_state(hidden)
    assert output.shape == (1, 3, 16)
    decoded = engine.forward_hidden_state(
        tensor_to_hidden_state(Tensor.randn(1, 1, 16).realize())
    )
    assert decoded.shape == (1, 1, 16)
    logits = engine.project_logits(output)
    assert logits.shape == (1, 3, 20)

    from tinygrad import dtypes

    token_ids = Tensor([1, 0], dtype=dtypes.int32).reshape(1, 2).contiguous().realize()
    token_buffer = TokenIdBuffer(
        shape=(1, 2),
        data=token_ids.bitcast(dtypes.uint8).contiguous().realize().numpy().tobytes(),
    )
    embedded = engine.embed_token_ids(token_buffer)
    assert embedded.shape == (1, 2, 16)


def test_qwen3_block_preserves_hidden_state_shape() -> None:
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor

    from exo.backends.tinygrad_engine import TinygradEngine
    from exo.backends.tinygrad_llama import LocalKeyValueCache, assemble_loaded_shard

    architecture = _architecture("qwen3", rope_scaling=None)
    loaded = assemble_loaded_shard(
        _random_parameters(
            include_query_key_norm=True, include_language_model_head=False
        ),
        architecture,
        _shard(start_layer=0, end_layer=1, layer_count=2),
    )
    engine = TinygradEngine(device_name="CPU")
    engine.loaded_shard = loaded
    engine.key_value_cache = LocalKeyValueCache(len(loaded.layers))
    hidden = tensor_to_hidden_state(Tensor.randn(1, 2, 16).realize())
    output = engine.forward_hidden_state(hidden)
    assert output.shape == (1, 2, 16)
    with pytest.raises(TinygradShardRoleError):
        engine.project_logits(output)


def test_loader_realizes_one_layer_and_forwards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_save

    from exo.backends.tinygrad_engine import TinygradEngine

    _write_runtime_config(tmp_path, tie_word_embeddings=False)
    parameters = _random_parameters(
        include_query_key_norm=False, include_language_model_head=True
    )
    parameters.update(_random_layer(layer_index=1))
    safe_save(
        {name: tensor.cast(dtypes.float32) for name, tensor in parameters.items()},
        str(tmp_path / "model.safetensors"),
    )

    def use_temporary_directory(model_id: ModelId) -> Path:
        _ = model_id
        return tmp_path

    monkeypatch.setattr(
        "exo.backends.tinygrad_weights.build_model_path",
        use_temporary_directory,
    )
    engine = TinygradEngine(device_name="CPU")
    progress = list(engine.iter_load_model(_bound_instance(start_layer=0, end_layer=1)))
    assert [(item.layers_loaded, item.total) for item in progress] == [(1, 1)]
    assert engine.parameter_byte_count is None
    engine.allocate_weights(_bound_instance(start_layer=0, end_layer=1))
    assert engine.parameter_byte_count is not None
    assert engine.parameter_byte_count > 0
    hidden = tensor_to_hidden_state(Tensor.randn(1, 2, 16).realize())
    assert engine.forward_hidden_state(hidden).shape == (1, 2, 16)


def test_pinned_interval_replaces_the_previous_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_save

    from exo.backends.tinygrad_engine import TinygradEngine

    layer_count = 3
    _write_runtime_config(tmp_path, tie_word_embeddings=False, layer_count=layer_count)
    parameters = _random_parameters(
        include_query_key_norm=False, include_language_model_head=True
    )
    parameters["model.layers.0.input_layernorm.weight"] = Tensor.full(
        (16,), 0.0, dtype=dtypes.float32
    ).realize()
    middle_layer = _random_layer(1)
    middle_layer["model.layers.1.input_layernorm.weight"] = Tensor.full(
        (16,), 1.0, dtype=dtypes.float32
    ).realize()
    parameters.update(middle_layer)
    parameters.update(_random_layer(2))
    safe_save(
        {name: tensor.cast(dtypes.float32) for name, tensor in parameters.items()},
        str(tmp_path / "model.safetensors"),
    )

    def use_temporary_directory(model_id: ModelId) -> Path:
        _ = model_id
        return tmp_path

    monkeypatch.setattr(
        "exo.backends.tinygrad_weights.build_model_path",
        use_temporary_directory,
    )
    engine = TinygradEngine(device_name="CPU")
    middle = _bound_instance(start_layer=1, end_layer=2, layer_count=layer_count)
    engine.allocate_weights(middle)
    allocated_bytes = engine.parameter_byte_count
    list(engine.iter_load_model(middle))
    assert engine.parameter_byte_count == allocated_bytes
    loaded = engine.loaded_shard
    assert loaded is not None
    assert len(loaded.layers) == 1
    assert loaded.token_embedding is None
    assert loaded.final_norm_weight is None
    assert loaded.language_model_head is None
    assert (
        loaded.layers[0].input_norm_weight.numpy().tobytes()
        == Tensor.full((16,), 1.0, dtype=dtypes.float32).realize().numpy().tobytes()
    )

    hidden = tensor_to_hidden_state(Tensor.randn(1, 2, 16).realize())
    assert engine.forward_hidden_state(hidden).shape == (1, 2, 16)
    filled_cache = engine.key_value_cache
    assert filled_cache is not None
    assert filled_cache.cached(0)[0] is not None

    first = _bound_instance(start_layer=0, end_layer=1, layer_count=layer_count)
    list(engine.iter_load_model(first))
    replaced = engine.loaded_shard
    assert replaced is not None
    assert replaced.token_embedding is not None
    assert replaced.final_norm_weight is None
    assert (
        replaced.layers[0].input_norm_weight.numpy().tobytes()
        == Tensor.full((16,), 0.0, dtype=dtypes.float32).realize().numpy().tobytes()
    )
    replaced_cache = engine.key_value_cache
    assert replaced_cache is not None
    assert replaced_cache.cached(0) == (None, None)


def _architecture(
    model_type: Literal["llama", "qwen2", "qwen3"],
    *,
    rope_scaling: RopeScalingConfig | None,
) -> TransformerArchitecture:
    return TransformerArchitecture(
        model_type=model_type,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        vocab_size=20,
        num_hidden_layers=2,
        head_dim=4,
        rope_scaling=rope_scaling,
        tie_word_embeddings=False,
    )


def _llama3_scaling() -> RopeScalingConfig:
    return RopeScalingConfig(
        rope_type="llama3",
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=32,
    )


def _random_parameters(
    *, include_query_key_norm: bool, include_language_model_head: bool
) -> dict[str, Tensor]:
    from tinygrad import Tensor

    parameters: dict[str, Tensor] = {}
    parameters.update(_random_layer(0))
    if include_query_key_norm:
        parameters["model.layers.0.self_attn.q_norm.weight"] = Tensor.randn(4).realize()
        parameters["model.layers.0.self_attn.k_norm.weight"] = Tensor.randn(4).realize()
    parameters["model.embed_tokens.weight"] = Tensor.randn(20, 16).realize()
    parameters["model.norm.weight"] = Tensor.randn(16).realize()
    if include_language_model_head:
        parameters["lm_head.weight"] = Tensor.randn(20, 16).realize()
    return parameters


def _random_layer(layer_index: int) -> dict[str, Tensor]:
    from tinygrad import Tensor

    prefix = f"model.layers.{layer_index}"

    def matrix(rows: int, columns: int) -> Tensor:
        return Tensor.randn(rows, columns).realize()

    return {
        f"{prefix}.input_layernorm.weight": Tensor.randn(16).realize(),
        f"{prefix}.self_attn.q_proj.weight": matrix(16, 16),
        f"{prefix}.self_attn.k_proj.weight": matrix(8, 16),
        f"{prefix}.self_attn.v_proj.weight": matrix(8, 16),
        f"{prefix}.self_attn.o_proj.weight": matrix(16, 16),
        f"{prefix}.post_attention_layernorm.weight": Tensor.randn(16).realize(),
        f"{prefix}.mlp.gate_proj.weight": matrix(32, 16),
        f"{prefix}.mlp.up_proj.weight": matrix(32, 16),
        f"{prefix}.mlp.down_proj.weight": matrix(16, 32),
    }


def _write_runtime_config(
    directory: Path, *, tie_word_embeddings: bool, layer_count: int = 2
) -> None:
    payload = {
        "model_type": "llama",
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 32,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000,
        "vocab_size": 20,
        "num_hidden_layers": layer_count,
        "head_dim": 4,
        "tie_word_embeddings": tie_word_embeddings,
    }
    (directory / "config.json").write_text(json.dumps(payload))


def _shard(
    *, start_layer: int, end_layer: int, layer_count: int
) -> PipelineShardMetadata:
    model_card = ModelCard(
        model_id=ModelId("tinygrad-forward"),
        storage_size=Memory.from_mb(16),
        n_layers=layer_count,
        hidden_size=16,
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


def _bound_instance(
    *, start_layer: int, end_layer: int, layer_count: int = 2
) -> BoundInstance:
    shard = _shard(
        start_layer=start_layer, end_layer=end_layer, layer_count=layer_count
    )
    node_id = NodeId("node-cpu")
    runner_id = RunnerId("runner-cpu")
    instance = TinygradInstance(
        instance_id=InstanceId("tinygrad-forward"),
        shard_assignments=ShardAssignments(
            model_id=shard.model_card.model_id,
            node_to_runner={node_id: runner_id},
            runner_to_shard={runner_id: shard},
        ),
        device_backend_by_node={node_id: Backend.TinygradCpu},
    )
    return BoundInstance(
        instance=instance,
        bound_runner_id=runner_id,
        bound_node_id=node_id,
    )
