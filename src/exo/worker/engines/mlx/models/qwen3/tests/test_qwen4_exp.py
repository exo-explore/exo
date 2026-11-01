# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false
"""Qwen3.8-Flash-Next (qwen4_exp): the model class exo loads for it (qwen4_exp.py).

Its numerics against Hugging Face transformers are checked outside the unit tests (real
weights); these tests cover the parts that are easy to get silently wrong: the n-gram hash
and its EOS resets, the n-gram table readers for both checkpoint layouts, the sparse
attention selection, cached and chunked prefill, and the key mapping of official
checkpoints.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map

from exo.worker.engines.mlx.models import registry
from exo.worker.engines.mlx.models.qwen3.qwen4_exp import (
    IntArray,
    Model,
    ModelArgs,
    NGramHasher,
    NGramTable,
    QSAIndexer,
    QSAKVCache,
    TextArgs,
)

EOS = 250
TINY: dict[str, object] = {
    "model_type": "qwen4_exp_text",
    "vocab_size": 256,
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "layer_types": [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ],
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 8,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "hc_count": 4,
    "hc_lowrank": 16,
    "indexer_n_heads": 4,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 16,
    "indexer_budget": 16,
    "indexer_compress_ratio": 4,
    "ngram_size": 3,
    "heads_per_ngram": 2,
    "ngram_vocab_size_base": 1000,
    "ple_embed_dim": 128,
    "ple_layer_ids": [2],
    "eos_token_id": EOS,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
    },
}
PLE_PREFIX = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"


def _tiny_model() -> Model:
    mx.random.seed(0)
    model = Model(ModelArgs.from_dict({"model_type": "qwen4_exp", "text_config": TINY}))
    # Random gains everywhere (they start at zero) so that (1 + w) is exercised
    model.update(
        tree_map(
            lambda p: mx.random.normal(p.shape) * 0.1 if p.dtype == mx.float32 else p,
            model.parameters(),
        )
    )
    mx.eval(model.parameters())
    return model


def _write_table(
    path: Path, table: mx.array, parts: int, quantized: bool, prefix: str
) -> None:
    """A table split in ``parts`` row shards, in the mlx-community 4-bit layout
    (``shards.N.{weight,scales,biases}``) or the official one (``shard_N.weight``)."""
    tensors: dict[str, mx.array] = {}
    for index, rows in enumerate(mx.split(table, parts, axis=0)):
        if quantized:
            weight, scales, biases = mx.quantize(
                rows.astype(mx.bfloat16), group_size=32, bits=4
            )
            tensors[f"{prefix}.shards.{index}.weight"] = weight
            tensors[f"{prefix}.shards.{index}.scales"] = scales
            tensors[f"{prefix}.shards.{index}.biases"] = biases
        else:
            tensors[f"{prefix}.shard_{index}.weight"] = rows.astype(mx.bfloat16)
    path.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path / "table.safetensors"), tensors)
    weight_map = dict.fromkeys(tensors, "table.safetensors")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )


def _attached_tiny_model(tmp_path: Path) -> Model:
    model = _tiny_model()
    hasher = model.layers[1].ple
    assert hasher is not None
    rows = hasher.ple_embedding.hasher.rows
    table = mx.random.normal((rows, hasher.ple_embedding.width)) * 0.5
    _write_table(tmp_path, table, 4, quantized=False, prefix=PLE_PREFIX)
    model.attach_ngram_table(tmp_path, prewarm=False)
    return model


def test_qwen4_exp_loads_with_the_family_model_classes() -> None:
    assert registry.get_model_classes({"model_type": "qwen4_exp"}) == (Model, ModelArgs)


def test_hash_matches_the_buffers_stored_in_the_released_checkpoints() -> None:
    """Qwen/Qwen3.8-Flash-Next and mlx-community/Qwen3.8-Flash-Next-4bit store these for
    layer 1 (seed 1234, which config.json does not carry)."""
    args = TextArgs.from_dict({"vocab_size": 248320})
    hasher = NGramHasher(args, 0)
    assert hasher.multipliers == [23703573157769, 20109073645365, 8052911324071]
    assert hasher.sizes.tolist()[:4] == [20000003, 20000023, 20000033, 20000047]
    assert hasher.sizes.tolist()[-1] == 20000171
    assert hasher.offsets.tolist()[-1] == 300001275
    assert hasher.rows == 320001536
    hasher.check_buffers(
        hasher.multipliers, hasher.sizes.tolist(), hasher.offsets.tolist()
    )
    with pytest.raises(ValueError, match="hash buffers"):
        hasher.check_buffers([1, 2, 3], hasher.sizes.tolist(), hasher.offsets.tolist())


def _slow_ngram_ids(hasher: NGramHasher, history: list[int]) -> list[list[int]]:
    """The reference hash written out per position: n-grams never reach back past an
    EOS (positions before the segment read as EOS)."""
    mask = (1 << 64) - 1

    def wrap(value: int) -> int:
        value &= mask
        return value - (1 << 64) if value >= 1 << 63 else value

    out: list[list[int]] = []
    segment_start = 0
    for position, token in enumerate(history):
        ids: list[int] = []
        for ngram in range(2, hasher.ngram_size + 1):
            mixed = 0
            for shift in range(ngram):
                source = position - shift
                value = history[source] if source >= segment_start else hasher.eos
                mixed = wrap(mixed ^ wrap(value * hasher.multipliers[shift]))
            for head in range(hasher.heads_per_ngram):
                index = (ngram - 2) * hasher.heads_per_ngram + head
                size = int(hasher.sizes[index])
                ids.append(mixed % size + int(hasher.offsets[index]))
        out.append(ids)
        if token == hasher.eos:
            segment_start = position + 1
    return out


def test_ngram_ids_follow_the_reference_and_reset_at_eos() -> None:
    hasher = NGramHasher(TextArgs.from_dict(TINY), 0)
    rng = np.random.default_rng(0)
    history = rng.integers(0, 240, size=40).tolist()
    history[11] = EOS
    history[12] = EOS
    history[30] = EOS
    expected = _slow_ngram_ids(hasher, history)
    got: IntArray = hasher(np.array([history], dtype=np.int64))
    assert got[0].tolist() == expected


@pytest.mark.parametrize("quantized", [False, True])
def test_table_reader_reads_both_checkpoint_layouts(
    tmp_path: Path, quantized: bool
) -> None:
    table = mx.random.normal((512, 32))
    official_prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    prefix = PLE_PREFIX if quantized else official_prefix
    _write_table(tmp_path, table, 4, quantized=quantized, prefix=prefix)
    reader = NGramTable(tmp_path, prefix, 32)
    assert reader.rows == 512
    ids: IntArray = np.array([0, 511, 128, 127, 300, 0], dtype=np.int64)
    rows = reader(ids)
    if quantized:
        # The reader dequantises in float32 (the model then casts to its own dtype)
        weight, scales, biases = mx.quantize(
            table.astype(mx.bfloat16), group_size=32, bits=4
        )
        expected = np.array(
            mx.dequantize(
                weight,
                scales.astype(mx.float32),
                biases.astype(mx.float32),
                group_size=32,
                bits=4,
            )
        )
        np.testing.assert_allclose(rows, expected[ids], rtol=0, atol=1e-6)
    else:
        expected = np.array(table.astype(mx.bfloat16).astype(mx.float32))
        np.testing.assert_array_equal(rows, expected[ids])


def test_indexer_is_dense_within_the_budget_and_causal_past_it() -> None:
    args = TextArgs.from_dict(TINY)
    indexer = QSAIndexer(args, rotary_dims=8)
    x = mx.random.normal((1, 16, 64))
    assert indexer.select(x, None) is None  # 4 complete blocks: all kept

    x = mx.random.normal((1, 41, 64))
    mask = indexer.select(x, None)
    assert mask is not None
    keep = np.array(mask[0, 0])
    # nothing after the query, its own block always, at most 4 blocks + the partial one
    assert not np.triu(keep, k=1).any()
    for query in range(41):
        own_start = (query + 1) // 4 * 4
        assert keep[query, own_start : query + 1].all()
        assert keep[query].sum() <= 16 + 3
        if query < 16:
            assert keep[query, : query + 1].all()


def test_chunked_prefill_and_decode_match_one_forward(tmp_path: Path) -> None:
    model = _attached_tiny_model(tmp_path)
    rng = np.random.default_rng(1)
    tokens = rng.integers(0, 240, size=(1, 70))
    tokens[0, 23] = EOS
    full = model(mx.array(tokens))

    for chunks in ([70], [9, 30, 1, 30], [24] + [1] * 46):
        cache = model.make_cache()
        outputs: list[mx.array] = []
        start = 0
        for length in chunks:
            outputs.append(model(mx.array(tokens[:, start : start + length]), cache))
            start += length
        chunked = mx.concatenate(outputs, axis=1)
        assert mx.abs(chunked - full).max().item() < 1e-4, chunks


def test_quantized_kv_cache_is_refused() -> None:
    with pytest.raises(ValueError, match="quantized KV cache"):
        QSAKVCache().to_quantized()


def test_sanitize_maps_the_official_layout(tmp_path: Path) -> None:
    model = _tiny_model()
    flat = dict(tree_flatten(model.parameters()))
    official: dict[str, mx.array] = {}
    for name, value in flat.items():
        key = name.replace("language_model.model.", "model.language_model.").replace(
            "language_model.lm_head", "lm_head"
        )
        if ".switch_mlp." in key:
            continue
        if key.endswith("conv1d.weight"):
            value = value.swapaxes(1, 2)
        official[key] = value
    for index in range(4):
        layer = f"model.language_model.layers.{index}.mlp"
        gate = flat[
            f"language_model.model.layers.{index}.mlp.switch_mlp.gate_proj.weight"
        ]
        up = flat[f"language_model.model.layers.{index}.mlp.switch_mlp.up_proj.weight"]
        down = flat[
            f"language_model.model.layers.{index}.mlp.switch_mlp.down_proj.weight"
        ]
        official[f"{layer}.experts.gate_up_proj"] = mx.concatenate([gate, up], axis=1)
        official[f"{layer}.experts.down_proj"] = down
    hasher = NGramHasher(TextArgs.from_dict(TINY), 0)
    ple = "model.language_model.layers.1.ple.ple_embedding"
    official[f"{ple}.layer_multipliers"] = mx.array(hasher.multipliers)
    official[f"{ple}.ngram_heads_vocab_sizes"] = mx.array(hasher.sizes.tolist())
    official[f"{ple}.ngram_heads_offsets"] = mx.array(hasher.offsets.tolist())
    official[f"{ple}.ngram_embedding.shard_0.weight"] = mx.zeros((4, 4))
    official["model.visual.blocks.0.attn.qkv.weight"] = mx.zeros((4, 4))
    official["mtp.fc_hidden.weight"] = mx.zeros((4, 4))

    sanitized = model.sanitize(official)
    assert set(sanitized) == set(flat)
    for name, value in sanitized.items():
        assert mx.array_equal(value, flat[name]), name

    official[f"{ple}.layer_multipliers"] = mx.array([1, 2, 3])
    with pytest.raises(ValueError, match="hash buffers"):
        model.sanitize(official)


def test_every_parameter_is_a_checkpoint_tensor() -> None:
    """The mlx-community 4-bit layout: quantising the model gives the same tensor names
    as the conversion (no parameter is left to its initial value by a strict=False load)."""
    model = _tiny_model()
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        class_predicate=lambda _, m: isinstance(m, (nn.Linear, nn.Embedding))
        and m.weight.shape[-1] % 32 == 0,
    )
    names = {name for name, _ in tree_flatten(model.parameters())}
    assert (
        "language_model.model.layers.0.attn_hyper_connection.block_inject_weight.scales"
        in names
    )
    assert "language_model.model.hyper_connection_mixer.hc_norm.weight" in names
    assert (
        "language_model.model.layers.3.self_attn.indexer.index_qk_proj.weight" in names
    )
    assert not any("ngram_embedding" in name for name in names)
