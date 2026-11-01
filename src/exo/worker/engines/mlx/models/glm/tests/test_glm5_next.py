"""GLM-5.3-Flash (glm5_next.py) on a tiny random config: loading, the DSA indexer's k-pool
selection against a direct transcription of Hugging Face transformers' algorithm, cached
(chunked prefill + decode, batched with left padding) against one-shot forwards, and the
checkpoint key mapping."""

import math
from typing import cast

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.cache import ArraysCache, BatchKVCache, CacheList, KVCache

from exo.worker.engines.mlx.models import registry
from exo.worker.engines.mlx.models.glm.glm5_next import (
    Glm5NextAttention,
    Indexer,
    Model,
    ModelArgs,
)
from exo.worker.engines.mlx.models.glm.tests.glm5_next_tiny import (
    CONFIG,
    TEXT_CONFIG,
    tiny_model,
    tokens,
)


def _model() -> Model:
    return tiny_model()


def _tokens(n: int, seed: int) -> mx.array:
    return tokens(n, seed)


def test_glm5_next_loads_with_the_family_model_classes() -> None:
    assert registry.get_model_classes(CONFIG) == (Model, ModelArgs)


def test_args_read_the_text_config() -> None:
    args = cast(ModelArgs, ModelArgs.from_dict(CONFIG))  # pyright: ignore[reportUnknownMemberType]
    assert args.model_type == "glm5_next"
    assert args.hidden_size == 32
    assert args.num_hidden_layers == 5
    # transformers' name for the MLA + DSA layers is accepted as well
    renamed = dict(TEXT_CONFIG)
    layer_types = cast(list[str], TEXT_CONFIG["layer_types"])
    renamed["layer_types"] = [
        "indexed_attention" if t != "linear_attention" else t for t in layer_types
    ]
    other = cast(ModelArgs, ModelArgs.from_dict({**CONFIG, "text_config": renamed}))  # pyright: ignore[reportUnknownMemberType]
    assert other.layer_types == args.layer_types


def test_shared_indexer_layers_are_refused() -> None:
    text = dict(TEXT_CONFIG)
    text["indexer_types"] = ["full", "full", "full", "shared", "full"]
    with pytest.raises(ValueError, match="shared indexer"):
        ModelArgs.from_dict({**CONFIG, "text_config": text})  # pyright: ignore[reportUnknownMemberType]


def _softmax(values: list[float]) -> list[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def _reference_selection(
    keys: list[list[float]],
    gates: list[list[float]],
    ape: list[list[float]],
    q: list[list[list[float]]],
    weights: list[list[float]],
    topk: int,
    kpool: int,
    first: int,
) -> list[tuple[set[int], set[int], int]]:
    """transformers' Glm5NextTextIndexer.forward for one sequence, from its pooled states
    on, written out with plain loops: for each of the last len(q) queries, the key
    positions it must select and those it may select (pools tied with the last selected
    one). keys/gates: [N][D], q: [L][H][D], weights: [L][H], already scaled."""
    n, dim = len(keys), len(keys[0])
    pools: list[list[int]] = []
    pooled: list[list[float]] = []
    start = first
    while (
        start + kpool <= n
    ):  # complete pools only; incomplete ones are never selectable
        members = list(range(start, start + kpool))
        pools.append(members)
        out = [0.0] * dim
        for d in range(dim):
            probs = _softmax([gates[t][d] + ape[j][d] for j, t in enumerate(members)])
            out[d] = sum(p * keys[t][d] for p, t in zip(probs, members, strict=True))
        pooled.append(out)
        start += kpool
    selections: list[tuple[set[int], set[int], int]] = []
    for row, (q_row, w_row) in enumerate(zip(q, weights, strict=True)):
        position = n - len(q) + row
        scored: list[tuple[float, int]] = []
        for p, members in enumerate(pools):
            if members[-1] > position:
                continue
            score = 0.0
            for q_head, w_head in zip(q_row, w_row, strict=True):
                dot = sum(a * b for a, b in zip(q_head, pooled[p], strict=True))
                score += w_head * max(dot, 0.0)
            scored.append((score, p))
        scored.sort(reverse=True)
        k = topk // kpool
        tail_count = (position - first + 1) % kpool
        tail = set(range(position + 1 - tail_count, position + 1))
        if len(scored) <= k:
            chosen = {t for _, p in scored for t in pools[p]} | tail
            selections.append((chosen, chosen, len(chosen)))
            continue
        # Pools tied with the k-th best score may be selected either way.
        kth = scored[k - 1][0]
        must = {t for score, p in scored if score > kth + 1e-6 for t in pools[p]}
        may = {t for score, p in scored if score >= kth - 1e-6 for t in pools[p]}
        selections.append((must | tail, may | tail, k * kpool + tail_count))
    return selections


def _floats2(x: mx.array) -> list[list[float]]:
    return cast(list[list[float]], x.tolist())


@pytest.mark.parametrize("first", [0, 3])
def test_indexer_selects_like_transformers(first: int) -> None:
    model = _model()
    attention = model.layers[3].self_attn
    assert isinstance(attention, Glm5NextAttention)
    indexer: Indexer = attention.indexer
    rng = np.random.default_rng(1)
    n, length, hidden = 47, 9, 32
    x = mx.array(rng.normal(size=(1, n, hidden)).astype(np.float32))
    qr = mx.array(rng.normal(size=(1, n, 16)).astype(np.float32))
    cache = BatchKVCache([first]) if first else KVCache()
    # Fill the cache with the first n - length positions, then select for the rest.
    indexer(x[:, : n - length], qr[:, : n - length], cache)
    topk = indexer(x[:, n - length :], qr[:, n - length :], cache)
    assert topk is not None
    assert cache.keys is not None and cache.values is not None
    q = indexer.wq_b(qr[0, n - length :]).reshape(length, 2, 32)
    # weights_proj * n_heads^-0.5, with the scores' softmax_scale (head_dim^-0.5) folded in
    weights = indexer.weights_proj(x[0, n - length :]) * (1 / math.sqrt(2 * 32))
    expected = _reference_selection(
        _floats2(cache.keys[0, 0, :n]),
        _floats2(cache.values[0, 0, :n]),
        _floats2(indexer.index_kpool_compress_ape),
        cast(list[list[list[float]]], q.tolist()),
        _floats2(weights),
        indexer.index_topk,
        indexer.kpool,
        first,
    )
    rows = cast(list[list[int]], topk[0].tolist())
    for row, (must, may, count) in zip(rows, expected, strict=True):
        got = [t for t in row if t >= 0]
        assert len(got) == len(set(got)) == count
        assert must <= set(got) <= may


def _caches(model: Model) -> list[ArraysCache | CacheList]:
    return model.make_cache()


def test_chunked_prefill_and_decode_match_one_shot() -> None:
    model = _model()
    prompt = _tokens(41, seed=2)
    full = model(prompt)
    cache = _caches(model)
    pieces = [model(prompt[:, :13], cache), model(prompt[:, 13:30], cache)]
    for i in range(30, 41):
        pieces.append(model(prompt[:, i : i + 1], cache))
    chunked = mx.concatenate(pieces, axis=1)
    # Past index_topk + index_kpool - 1 = 11 keys the attention is sparse.
    np.testing.assert_allclose(np.array(chunked), np.array(full), rtol=2e-4, atol=2e-4)


def test_batched_decode_with_left_padding_matches_single_sequences() -> None:
    model = _model()
    first, second = _tokens(30, seed=3), _tokens(19, seed=4)
    singles: list[mx.array] = []
    to_merge: list[list[ArraysCache | CacheList]] = []
    for sequence in (first, second):
        single = _caches(model)
        model(sequence[:, :-1], single)
        singles.append(model(sequence[:, -1:], single))
        batched = _caches(model)
        model(sequence[:, :-1], batched)
        to_merge.append(batched)
    merged: list[ArraysCache | CacheList] = []
    for a, b in zip(to_merge[0], to_merge[1], strict=True):
        if isinstance(a, ArraysCache):
            assert isinstance(b, ArraysCache)
            merged.append(ArraysCache.merge([a, b]))
        else:
            assert isinstance(b, CacheList)
            merged.append(CacheList.merge([a, b]))
    batch = mx.concatenate([first[:, -1:], second[:, -1:]], axis=0)
    out = model(batch, merged)
    # the second sequence is 11 tokens shorter: its rows are left-padded
    np.testing.assert_allclose(
        np.array(out[0]), np.array(singles[0][0]), rtol=2e-4, atol=2e-4
    )
    np.testing.assert_allclose(
        np.array(out[1]), np.array(singles[1][0]), rtol=2e-4, atol=2e-4
    )


def test_sanitize_maps_the_hugging_face_names() -> None:
    model = _model()
    args = model.args
    params = dict(cast(list[tuple[str, mx.array]], tree_flatten(model.parameters())))
    hf: dict[str, mx.array] = {}
    for name, value in params.items():
        key = name.replace("model.", "model.language_model.", 1)
        if ".switch_mlp." in key:
            layer_prefix, rest = key.split(".mlp.switch_mlp.")
            for e in range(args.n_routed_experts):
                hf[f"{layer_prefix}.mlp.experts.{e}.{rest}"] = value[e]
            continue
        for site in ("attn", "ffn"):
            key = key.replace(f".{site}_hc.", f".hc_{site}_")
        for proj in ("q", "k", "v"):
            if key.endswith(f".{proj}_conv.weight"):
                key = key.replace(f"{proj}_conv.weight", f"{proj}_conv1d.weight")
                value = value.moveaxis(1, 2)
        if key.endswith((".embed_q.weight", ".unembed_out.weight")):
            continue
        hf[key] = value
    for i, layer in enumerate(model.layers):
        attention = layer.self_attn
        if isinstance(attention, Glm5NextAttention):
            wk = attention.embed_q.weight.swapaxes(-1, -2)  # [H, nope, kv_lora]
            wv = attention.unembed_out.weight  # [H, v, kv_lora]
            hf[f"model.language_model.layers.{i}.self_attn.kv_b_proj.weight"] = (
                mx.concatenate([wk, wv], axis=1).reshape(-1, args.kv_lora_rank)
            )
    hf["model.language_model.layers.5.mlp.gate.weight"] = mx.zeros((4, 32))
    hf["model.visual.blocks.0.attn.qkv.weight"] = mx.zeros((3, 3))
    sanitized = model.sanitize(hf)
    assert set(sanitized) == set(params)
    for name, value in params.items():
        np.testing.assert_allclose(
            np.array(sanitized[name]), np.array(value), rtol=1e-6, atol=1e-6
        )
