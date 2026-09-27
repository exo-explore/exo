from __future__ import annotations

import struct
from pathlib import Path

import pytest

from exo.backends.tinygrad_checkpoint import (
    architecture_from_gguf,
    assigned_gguf_tensors,
    canonical_tensor_name,
    permute_gguf_rotary_weight,
    read_gguf_checkpoint,
)
from exo.backends.tinygrad_weights import (
    TinygradModelSupportError,
    TinygradWeightError,
    select_shard_tensor_names,
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

_HIDDEN = 16
_HEADS = 4
_KEY_VALUE_HEADS = 2
_INTERMEDIATE = 32
_VOCAB = 20
_HEAD_DIM = 4


def test_source_names_share_one_canonical_query_weight() -> None:
    expected = "model.layers.1.self_attn.q_proj.weight"
    assert canonical_tensor_name("model.layers.1.self_attn.q_proj.weight") == expected
    assert canonical_tensor_name("layers.1.attention.wq.weight") == expected
    assert canonical_tensor_name("blk.1.attn_q.weight") == expected
    assert canonical_tensor_name("layers.1.feed_forward.w1.weight") == (
        "model.layers.1.mlp.gate_proj.weight"
    )
    assert canonical_tensor_name("layers.1.feed_forward.w3.weight") == (
        "model.layers.1.mlp.up_proj.weight"
    )
    assert canonical_tensor_name("layers.1.feed_forward.w2.weight") == (
        "model.layers.1.mlp.down_proj.weight"
    )
    assert canonical_tensor_name("token_embd.weight") == "model.embed_tokens.weight"
    assert canonical_tensor_name("output.weight") == "lm_head.weight"


def test_assigned_interval_keeps_only_the_translated_layer() -> None:
    names = [
        canonical_tensor_name("model.layers.0.self_attn.q_proj.weight"),
        canonical_tensor_name("layers.1.attention.wq.weight"),
        canonical_tensor_name("blk.2.attn_q.weight"),
        canonical_tensor_name("token_embd.weight"),
    ]
    assert all(name is not None for name in names)
    selected = select_shard_tensor_names(
        [name for name in names if name is not None],
        _shard(start_layer=1, end_layer=2, layer_count=4),
        embeddings_are_tied=False,
    )
    assert selected == frozenset({"model.layers.1.self_attn.q_proj.weight"})


def test_gguf_header_selects_only_the_assigned_byte_ranges(tmp_path: Path) -> None:
    _write_gguf(
        tmp_path / "model.gguf",
        [
            _kv_string("general.architecture", "llama"),
            _kv_uint32("general.file_type", 15),
        ],
        [
            ("blk.0.attn_q.weight", (16, 16), 1, _zeros(1, 256)),
            ("blk.1.attn_q.weight", (16, 16), 8, _zeros(8, 256)),
            ("blk.2.attn_q.weight", (16, 16), 12, _zeros(12, 256)),
            ("token_embd.weight", (16, 20), 1, _zeros(1, 320)),
        ],
    )
    checkpoint = read_gguf_checkpoint(tmp_path)
    assert checkpoint.file_type_name == "Q4_K_M"
    by_source = {record.source_name: record for record in checkpoint.tensors}
    assert by_source["blk.0.attn_q.weight"].ggml_type_name() == "F16"
    assert by_source["blk.1.attn_q.weight"].ggml_type_name() == "Q8_0"
    assert by_source["blk.2.attn_q.weight"].ggml_type_name() == "Q4_K"
    assigned = assigned_gguf_tensors(
        checkpoint,
        _shard(start_layer=0, end_layer=1, layer_count=3),
        embeddings_are_tied=False,
    )
    assigned_names = {record.source_name for record in assigned}
    assert assigned_names == {"blk.0.attn_q.weight", "token_embd.weight"}
    skipped = by_source["blk.1.attn_q.weight"].absolute_start
    assert skipped not in {record.absolute_start for record in assigned}


def test_unsupported_gguf_architecture_is_rejected(tmp_path: Path) -> None:
    _write_gguf(
        tmp_path / "model.gguf",
        [_kv_string("general.architecture", "gpt2")],
        [],
    )
    checkpoint = read_gguf_checkpoint(tmp_path)
    with pytest.raises(TinygradModelSupportError, match="gpt2"):
        architecture_from_gguf(checkpoint)


def test_q8_0_block_scales_signed_values() -> None:
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor, dtypes
    from tinygrad.helpers import DEV
    from tinygrad.llm.gguf import ggml_data_to_tensor

    DEV.value = "CPU"
    signed = [1, -2, 3, 0, *([0] * 28)]
    payload = struct.pack("<e32b", 0.5, *signed)
    tensor = Tensor(list(payload), dtype=dtypes.uint8)
    converted = ggml_data_to_tensor(tensor, 32, 8).reshape(32).contiguous().realize()
    expected = struct.pack("<32f", *[0.5 * value for value in signed])
    assert converted.float().numpy().tobytes() == expected


def test_gguf_rotary_permutation_splits_pairs() -> None:
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor, dtypes
    from tinygrad.helpers import DEV

    DEV.value = "CPU"
    weight = Tensor([0.0, 1.0, 2.0, 3.0], dtype=dtypes.float32).reshape(4, 1)
    permuted = permute_gguf_rotary_weight(weight, 1).contiguous().realize()
    assert permuted.reshape(4).numpy().tobytes() == struct.pack(
        "<4f", 0.0, 2.0, 1.0, 3.0
    )


def test_gguf_load_realizes_one_layer_and_skips_the_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor

    from exo.backends import tinygrad_checkpoint as checkpoint_module
    from exo.backends.tinygrad_engine import TinygradEngine
    from exo.backends.tinygrad_hidden_state import tensor_to_hidden_state

    _write_llama_gguf(tmp_path / "weights.gguf", query_columns=_HIDDEN)
    checkpoint = read_gguf_checkpoint(tmp_path)
    shard = _shard(start_layer=0, end_layer=1, layer_count=1)
    assigned = assigned_gguf_tensors(checkpoint, shard, embeddings_are_tied=False)
    assigned_starts = {record.absolute_start for record in assigned}
    skipped = next(
        record.absolute_start
        for record in checkpoint.tensors
        if record.source_name == "blk.1.attn_norm.weight"
    )
    assert skipped not in assigned_starts

    seen: list[int] = []
    original = checkpoint_module.ggml_tensor_from_file

    def record_range(
        path: Path,
        start: int,
        length: int,
        element_count: int,
        ggml_type: int,
    ) -> Tensor:
        seen.append(start)
        return original(path, start, length, element_count, ggml_type)

    def use_temporary_directory(model_id: ModelId) -> Path:
        _ = model_id
        return tmp_path

    monkeypatch.setattr(checkpoint_module, "ggml_tensor_from_file", record_range)
    monkeypatch.setattr(
        "exo.backends.tinygrad_weights.build_model_path",
        use_temporary_directory,
    )

    engine = TinygradEngine(device_name="CPU")
    bound = _bound_instance(start_layer=0, end_layer=1, layer_count=1)
    engine.allocate_weights(bound)
    assert engine.parameter_byte_count == sum(
        record.element_count * 2 for record in assigned
    )
    progress = list(engine.iter_load_model(bound))
    assert [(item.layers_loaded, item.total) for item in progress] == [(1, 1)]
    assert set(seen) == assigned_starts
    assert skipped not in seen
    loaded = engine.loaded_shard
    assert loaded is not None
    assert len(loaded.layers) == 1
    assert loaded.token_embedding is not None
    hidden = tensor_to_hidden_state(Tensor.randn(1, 2, _HIDDEN).realize())
    assert engine.forward_hidden_state(hidden).shape == (1, 2, _HIDDEN)


def test_wrong_gguf_query_shape_is_rejected_before_the_shard_is_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("tinygrad")

    from exo.backends.tinygrad_engine import TinygradEngine

    _write_llama_gguf(tmp_path / "weights.gguf", query_columns=8)

    def use_temporary_directory(model_id: ModelId) -> Path:
        _ = model_id
        return tmp_path

    monkeypatch.setattr(
        "exo.backends.tinygrad_weights.build_model_path",
        use_temporary_directory,
    )
    engine = TinygradEngine(device_name="CPU")
    with pytest.raises(TinygradWeightError, match="shape"):
        list(
            engine.iter_load_model(
                _bound_instance(start_layer=0, end_layer=1, layer_count=1)
            )
        )
    assert engine.loaded_shard is None


def _write_llama_gguf(path: Path, *, query_columns: int) -> None:
    metadata = [
        _kv_string("general.architecture", "llama"),
        _kv_uint32("general.file_type", 1),
        _kv_uint32("llama.block_count", 1),
        _kv_uint32("llama.embedding_length", _HIDDEN),
        _kv_uint32("llama.attention.head_count", _HEADS),
        _kv_uint32("llama.attention.head_count_kv", _KEY_VALUE_HEADS),
        _kv_uint32("llama.feed_forward_length", _INTERMEDIATE),
        _kv_float32("llama.attention.layer_norm_rms_epsilon", 1e-5),
        _kv_float32("llama.rope.freq_base", 10000.0),
        _kv_uint32("llama.attention.key_length", _HEAD_DIM),
        _kv_uint32("llama.vocab_size", _VOCAB),
    ]
    query_rows = _HEADS * _HEAD_DIM
    key_rows = _KEY_VALUE_HEADS * _HEAD_DIM
    tensors: list[tuple[str, tuple[int, ...], int, bytes]] = [
        _named("token_embd.weight", (_VOCAB, _HIDDEN)),
        _named("blk.0.attn_norm.weight", (_HIDDEN,)),
        _named("blk.0.attn_q.weight", (query_rows, query_columns)),
        _named("blk.0.attn_k.weight", (key_rows, _HIDDEN)),
        _named("blk.0.attn_v.weight", (key_rows, _HIDDEN)),
        _named("blk.0.attn_output.weight", (_HIDDEN, query_rows)),
        _named("blk.0.ffn_norm.weight", (_HIDDEN,)),
        _named("blk.0.ffn_gate.weight", (_INTERMEDIATE, _HIDDEN)),
        _named("blk.0.ffn_up.weight", (_INTERMEDIATE, _HIDDEN)),
        _named("blk.0.ffn_down.weight", (_HIDDEN, _INTERMEDIATE)),
        _named("output_norm.weight", (_HIDDEN,)),
        _named("output.weight", (_VOCAB, _HIDDEN)),
        _named("blk.1.attn_norm.weight", (_HIDDEN,)),
    ]
    _write_gguf(path, metadata, tensors)


def _named(
    name: str, logical_shape: tuple[int, ...]
) -> tuple[str, tuple[int, ...], int, bytes]:
    count = 1
    for dimension in logical_shape:
        count *= dimension
    return (name, tuple(reversed(logical_shape)), 1, _zeros(1, count))


def _zeros(ggml_type: int, element_count: int) -> bytes:
    if ggml_type == 1:
        return b"\x00" * (element_count * 2)
    if ggml_type == 8:
        if element_count % 32 != 0:
            raise ValueError("Q8_0 element count must be a multiple of 32")
        return b"\x00" * ((element_count // 32) * 34)
    if ggml_type == 12:
        if element_count % 256 != 0:
            raise ValueError("Q4_K element count must be a multiple of 256")
        return b"\x00" * ((element_count // 256) * 144)
    raise ValueError(f"test writer has no payload for ggml type {ggml_type}")


def _write_gguf(
    path: Path,
    metadata: list[bytes],
    tensors: list[tuple[str, tuple[int, ...], int, bytes]],
) -> None:
    payload = bytearray()
    infos = bytearray()
    offset = 0
    for name, dimensions, ggml_type, raw in tensors:
        aligned = ((offset + 31) // 32) * 32
        payload.extend(b"\x00" * (aligned - offset))
        infos.extend(_tensor_info(name, dimensions, ggml_type, aligned))
        payload.extend(raw)
        offset = aligned + len(raw)
    header = bytearray(b"GGUF")
    header.extend(struct.pack("<I", 3))
    header.extend(struct.pack("<q", len(tensors)))
    header.extend(struct.pack("<q", len(metadata)))
    for item in metadata:
        header.extend(item)
    header.extend(infos)
    data_start = ((len(header) + 31) // 32) * 32
    header.extend(b"\x00" * (data_start - len(header)))
    header.extend(payload)
    path.write_bytes(header)


def _tensor_info(
    name: str, dimensions: tuple[int, ...], ggml_type: int, offset: int
) -> bytes:
    body = _pack_string(name)
    body += struct.pack("<I", len(dimensions))
    for dimension in dimensions:
        body += struct.pack("<Q", dimension)
    body += struct.pack("<I", ggml_type)
    body += struct.pack("<Q", offset)
    return body


def _kv_string(key: str, value: str) -> bytes:
    return _pack_string(key) + struct.pack("<I", 8) + _pack_string(value)


def _kv_uint32(key: str, value: int) -> bytes:
    return _pack_string(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_float32(key: str, value: float) -> bytes:
    return _pack_string(key) + struct.pack("<I", 6) + struct.pack("<f", value)


def _pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _shard(
    *, start_layer: int, end_layer: int, layer_count: int
) -> PipelineShardMetadata:
    model_card = ModelCard(
        model_id=ModelId("tinygrad-gguf"),
        storage_size=Memory.from_mb(16),
        n_layers=layer_count,
        hidden_size=_HIDDEN,
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
    *, start_layer: int, end_layer: int, layer_count: int
) -> BoundInstance:
    shard = _shard(
        start_layer=start_layer, end_layer=end_layer, layer_count=layer_count
    )
    node_id = NodeId("node-cpu")
    runner_id = RunnerId("runner-cpu")
    instance = TinygradInstance(
        instance_id=InstanceId("tinygrad-gguf"),
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
