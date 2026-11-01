"""DeepSeek V4.1's MLX model on a tiny random configuration: the engram hashing against the
reference algorithm, the engram layout against the released config, the caches (chunked
prefill and one-token decode give one-shot results), the quantizers and the checkpoint
mapping. Real-weight comparisons with DeepSeek's reference are in the PR's report."""

import json
import struct
from dataclasses import replace
from pathlib import Path
from typing import cast

import mlx.core as mx
import numpy as np
import numpy.typing as npt
import pytest
from mlx.utils import tree_flatten

from exo.worker.engines.mlx.models.deepseek_v41.model import (
    EngramLayout,
    HostEngramTable,
    LayerCache,
    Model,
    ModelArgs,
    NgramHasher,
    fake_quant_fp4,
    fake_quant_fp8,
)

# The released config's engram fields (deepseek-ai/DeepSeek-V4.1-Flash config.json).
RELEASED_ENGRAM = ModelArgs(
    engram_layer_ids=(1, 14),
    engram_num_embeddings=(384006168, 384016682),
    engram_max_ngram_size=4,
    engram_vocab_size=16000000,
    engram_n_heads=8,
    engram_head_dim=256,
    engram_compressed_vocab_size=99092,
)


def test_engram_layout_matches_the_released_table_sizes() -> None:
    layout = EngramLayout.from_args(RELEASED_ENGRAM)
    for layer, rows in enumerate(RELEASED_ENGRAM.engram_num_embeddings):
        primes = [p for per_ngram in layout.primes[layer] for p in per_ngram]
        assert len(primes) == 3 * 8
        assert sum(primes) == rows
    first = layout.primes[0][0]
    assert first[0] > 16000000 - 1
    assert len({p for layer in layout.primes for per in layer for p in per}) == 48


def _reference_hash(
    tokens: list[list[int]], hasher: NgramHasher, max_ngram: int
) -> list[list[list[int]]]:
    """NgramHashState.forward of the reference, for one layer, text only, written
    position by position."""
    token_map = cast(list[int], hasher.token_map.tolist())
    multipliers = cast(list[int], hasher.multipliers.tolist())
    primes = cast(list[list[int]], hasher.primes.tolist())
    offsets = cast(list[int], hasher.offsets.tolist())
    out: list[list[list[int]]] = []
    for row in tokens:
        compressed = [token_map[t] for t in row]
        out_row: list[list[int]] = []
        for p in range(len(compressed)):
            window: list[int] = []
            blocked = False
            for shift in range(max_ngram):
                blocked = blocked or p < shift
                window.append(hasher.pad if blocked else compressed[p - shift])
            products = [t * m for t, m in zip(window, multipliers, strict=True)]
            rolling = products[0]
            hashes: list[int] = []
            for i in range(1, max_ngram):
                rolling ^= products[i]
                hashes.extend(rolling % prime for prime in primes[i - 1])
            out_row.append([h + o for h, o in zip(hashes, offsets, strict=True)])
        out.append(out_row)
    return out


def _hasher(layer_index: int = 1) -> NgramHasher:
    layout = EngramLayout.from_args(RELEASED_ENGRAM)
    vocab = 1000
    token_map = [(i * 7919) % 400 for i in range(vocab)]
    return NgramHasher(layout, layer_index, token_map, 400, pad_token_id=2)


def test_ngram_hash_matches_the_reference_in_chunks_and_on_the_gpu() -> None:
    hasher = _hasher()
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 1000, size=(2, 23), dtype=np.int64)
    expected = _reference_hash(cast(list[list[int]], tokens.tolist()), hasher, 4)

    history = hasher.initial_history(2)
    pieces: list[np.ndarray] = []
    for start, end in ((0, 1), (1, 6), (6, 7), (7, 23)):
        ids, history = hasher.hash_host(tokens[:, start:end], history)
        pieces.append(ids)
    assert np.concatenate(pieces, axis=1).tolist() == expected

    device_ids, device_history = hasher.hash_device(
        mx.array(tokens), mx.array(hasher.initial_history(2))
    )
    assert device_ids.tolist() == expected
    assert device_history.tolist() == history.tolist()


def test_fp4_rounds_ties_to_even_and_scales_by_powers_of_two() -> None:
    # 0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4 (block max 6)
    x = mx.array([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 4])
    q = fake_quant_fp4(x, block_size=32)
    assert np.array(q)[0, :8].tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0]
    # a block whose max is just above 6 x 8 gets scale 16, not 8: 4.0 is then 0.25 of a
    # step and rounds to 0 (with scale 8 it would stay 4.0)
    y = mx.array([[6.0 * 8 * (1 + 2**-20), 4.0] + [0.0] * 30])
    assert np.array(fake_quant_fp4(y, block_size=32))[0, 1] == 0.0


def test_fp8_keeps_exactly_representable_values() -> None:
    x = mx.array([[448.0, 1.0, -2.0, 0.5] * 8])
    assert np.array_equal(np.array(fake_quant_fp8(x)), np.array(x))


def _tiny_args(**overrides: object) -> ModelArgs:
    args = ModelArgs(
        vocab_size=64,
        hidden_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        head_dim=64,
        q_lora_rank=32,
        qk_rope_head_dim=16,
        o_groups=2,
        o_lora_rank=16,
        moe_intermediate_size=32,
        n_routed_experts=4,
        num_experts_per_tok=2,
        sliding_window=8,
        compress_ratios=(0, 0, 2, 2, 1, 1),
        kv_source_layer_ids=(2, 4),
        index_source_layer_ids=(2, 4, 5),
        candidate_source_layer_id=4,
        candidate_topk_blocks=2,
        candidate_block_size=4,
        index_n_heads=2,
        index_head_dim=32,
        index_topk=4,
        rope_scaling={
            "factor": 16,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        engram_layer_ids=(1, 3),
        engram_num_embeddings=(0, 0),
        engram_max_ngram_size=3,
        engram_vocab_size=50,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_compressed_vocab_size=40,
        engram_tables="device",
    )
    return replace(args, **overrides)


def _tiny_model() -> Model:
    args = _tiny_args()
    layout = EngramLayout.from_args(args)
    rows = tuple(sum(p for per in layer for p in per) for layer in layout.primes)
    args = _tiny_args(engram_num_embeddings=rows)
    mx.random.seed(0)
    model = Model(args)
    # random weights everywhere, small enough to keep activations sane
    leaves = cast(list[tuple[str, mx.array]], tree_flatten(model.parameters()))
    updates: list[tuple[str, mx.array]] = []
    for name, value in leaves:
        if value.dtype == mx.uint8:  # engram tables: E4M3 codes 0..0x77 and scales ~2^0
            if name.endswith("scale"):
                updates.append((name, mx.full(value.shape, 127, dtype=mx.uint8)))
            else:
                updates.append(
                    (name, mx.random.randint(0, 0x78, value.shape).astype(mx.uint8))
                )
        elif name.endswith(("norm.weight", "layernorm.weight", "q_weight", "k_weight")):
            updates.append((name, mx.ones(value.shape)))
        else:
            updates.append((name, mx.random.normal(value.shape) * 0.05))
    model.load_weights(updates)
    mx.eval(model.parameters())
    token_map = [i % 40 for i in range(64)]
    for layer in model.layers:
        if layer.engram is not None:
            hasher = NgramHasher(layout, layer.engram.layer_index, token_map, 40, 2)
            layer.engram.attach(hasher, None)
    return model


def _logits(
    model: Model, tokens: list[int], chunks: list[int]
) -> npt.NDArray[np.float32]:
    cache = model.make_cache()
    rows: list[npt.NDArray[np.float32]] = []
    start = 0
    for size in chunks:
        out = model(mx.array([tokens[start : start + size]]), cache)
        rows.append(np.array(out[0], dtype=np.float32))
        start += size
    assert start == len(tokens)
    return np.concatenate(rows, axis=0)


def test_chunked_prefill_and_decode_match_one_shot() -> None:
    model = _tiny_model()
    tokens = cast(list[int], np.random.default_rng(1).integers(0, 64, size=29).tolist())
    one_shot = _logits(model, tokens, [29])
    chunked = _logits(model, tokens, [5, 3, 1, 7, 13])
    decode = _logits(model, tokens, [9] + [1] * 20)
    # same function, different matmul shapes: agreement to float noise
    np.testing.assert_allclose(chunked, one_shot, rtol=0, atol=2e-3)
    np.testing.assert_allclose(decode, one_shot, rtol=0, atol=2e-3)


def test_layer_cache_snapshot_is_independent() -> None:
    cache = LayerCache(window_size=4)
    cache.update_window(mx.ones((1, 3, 8)))
    snapshot = cache.prefix_cache_snapshot()
    cache.update_window(mx.zeros((1, 3, 8)))
    assert snapshot.offset == 3 and cache.offset == 6
    assert snapshot.window is not None and snapshot.window.shape[1] == 3
    assert float(snapshot.window.sum()) == 24.0


def test_sanitize_maps_the_released_names() -> None:
    args = _tiny_args(engram_num_embeddings=(10, 10), engram_tables="host")
    model = Model(args)
    fp8 = mx.zeros((64, 64), dtype=mx.uint8)
    e8m0 = mx.full((2, 2), 127, dtype=mx.uint8)
    weights = {
        "embed.weight": mx.zeros((64, 64)),
        "head.weight": mx.zeros((64, 64), dtype=mx.bfloat16),
        "layers.0.attn.wq_a.weight": fp8[:32],
        "layers.0.attn.wq_a.scale": e8m0[:1],
        "layers.0.attn.attn_sink": mx.zeros((4,)),
        "layers.0.hc_attn_fn": mx.zeros((24, 256)),
        "layers.0.ffn.gate.bias": mx.zeros((4,)),
        "layers.0.ffn.gate.bias_vl": mx.zeros((4,)),
        "layers.1.engram.embed.weight": mx.zeros((10, 32), dtype=mx.uint8),
        "layers.1.engram.embed.scale": mx.zeros((10, 1), dtype=mx.uint8),
        "layers.1.engram.wkv.weight": fp8,
        "layers.1.engram.wkv.scale": e8m0,
        "layers.2.attn.compressor.norm.weight": mx.ones((64,)),
        "layers.2.attn.indexer.wk.weight": mx.zeros((32, 64)),
        "layers.9.attn.attn_sink": mx.zeros((4,)),
        "mtp.0.attn.attn_sink": mx.zeros((4,)),
        "vision.blocks.0.norm1.weight": mx.zeros((8,)),
    }
    clean = model.sanitize(weights)
    assert set(clean) == {
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.q_a_proj.scales",
        "model.layers.0.self_attn.sinks",
        "model.layers.0.attn_hc.fn",
        "model.layers.0.mlp.gate.e_score_correction_bias",
        "model.layers.1.engram.wkv.weight",
        "model.layers.1.engram.wkv.scales",
        "model.layers.2.self_attn.compressor.kv_norm.weight",
        "model.layers.2.self_attn.indexer.k_proj.weight",
    }
    # the head is float32; a 32x32 fp8 block scale becomes one scale per row
    assert clean["lm_head.weight"].dtype == mx.float32
    assert clean["model.layers.1.engram.wkv.scales"].shape == (64, 2)
    assert clean["model.layers.1.engram.wkv.weight"].shape == (64, 16)

    device = Model(replace(args, engram_tables="device"))
    assert "model.layers.1.engram.embed.weight" in device.sanitize(weights)


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, object] = {}
    offset = 0
    blobs: list[bytes] = []
    for name, array in tensors.items():
        data = array.tobytes()
        dtype = "F8_E4M3" if name.endswith("weight") else "F8_E8M0"
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        blobs.append(data)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def test_host_engram_table_reads_rows_from_the_checkpoint(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    weight = rng.integers(0, 0x78, size=(50, 32), dtype=np.uint8)
    scale = rng.integers(120, 130, size=(50, 1), dtype=np.uint8)
    _write_safetensors(
        tmp_path / "model-00001.safetensors",
        {"layers.1.engram.embed.weight": weight, "layers.1.engram.embed.scale": scale},
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layers.1.engram.embed.weight": "model-00001.safetensors",
                    "layers.1.engram.embed.scale": "model-00001.safetensors",
                }
            }
        )
    )
    table = HostEngramTable.open(
        tmp_path, "layers.1.engram.embed.weight", "layers.1.engram.embed.scale"
    )
    ids = np.array([[3, 49, 0], [7, 7, 21]], dtype=np.int64)
    rows, scales = table.gather(ids)
    assert np.array_equal(np.array(rows), weight[ids])
    assert np.array_equal(np.array(scales), scale[ids])


@pytest.mark.parametrize("ratio_layer", [2, 4])
def test_compressed_layers_share_their_source(ratio_layer: int) -> None:
    model = _tiny_model()
    source = model.layers[ratio_layer].self_attn
    assert source.is_kv_source and source.is_index_source
    reuse = model.layers[ratio_layer + 1].self_attn
    assert reuse.ratio == source.ratio and not reuse.is_kv_source
