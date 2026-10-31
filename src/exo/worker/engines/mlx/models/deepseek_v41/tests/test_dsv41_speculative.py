"""Speculative decoding with DeepSeek V4.1's DSpark draft head: the rejection sampling keeps
the sampler's distribution, a recorded verify pass cut to its accepted tokens leaves the
caches as if only those tokens had run, greedy speculative decoding gives the plain greedy
tokens, and the checkpoint's mtp.* names map onto the draft head."""

import math
from dataclasses import dataclass, replace
from typing import cast

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from exo.worker.engines.mlx.models.deepseek_v41.model import (
    EngramLayout,
    LayerCache,
    Model,
    ModelArgs,
    NgramHasher,
)
from exo.worker.engines.mlx.models.deepseek_v41.speculative import (
    _Draft,  # pyright: ignore[reportPrivateUsage]
    _verify,  # pyright: ignore[reportPrivateUsage]
    speculative_generate_step,
)


@dataclass(frozen=True)
class _Sampling:
    temperature: float
    top_p: float = 1.0
    min_p: float = 0.0
    top_k: int = 0


def test_rejection_sampling_keeps_the_target_distribution() -> None:
    rng = np.random.default_rng(0)
    vocab, drafts, trials = 5, 3, 6000
    p_logits = mx.array(rng.normal(size=(drafts + 1, vocab)) * 1.5, dtype=mx.float32)
    q = mx.softmax(
        mx.array(rng.normal(size=(drafts, vocab)) * 1.5, dtype=mx.float32), axis=-1
    )
    p = cast(list[list[float]], mx.softmax(p_logits, axis=-1).tolist())
    logprobs = p_logits - mx.logsumexp(p_logits, axis=-1, keepdims=True)
    mx.random.seed(1)
    counts = [[0.0] * vocab for _ in range(drafts + 1)]
    for _ in range(trials // 100):
        rows: list[mx.array] = []
        for _ in range(100):
            tokens = [
                mx.random.categorical(mx.log(q[i : i + 1])).astype(mx.int32)
                for i in range(drafts)
            ]
            accepted, after = _verify(
                logprobs, _Draft(tokens, [q[i] for i in range(drafts)]), _Sampling(1.0)
            )
            rows.append(mx.concatenate([accepted.astype(mx.int32), after, *tokens]))
        for row in cast(list[list[int]], mx.stack(rows).tolist()):
            n, after_token, tokens_drafted = row[0], row[1], row[2:]
            for i in range(n):
                counts[i][tokens_drafted[i]] += 1
            counts[n][after_token] += 1
    # the token emitted at position i follows p[i] (given that position is reached)
    for i in range(drafts + 1):
        row = counts[i]
        expected = p[i]
        total = sum(row)
        assert total > 300
        for seen, want in zip(row, expected, strict=True):
            z = abs(seen / total - want) / math.sqrt(want * (1 - want) / total)
            assert z < 5.0, (i, row, expected)


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
        compress_ratios=(0, 0, 2, 2, 1, 1, 0, 0),
        kv_source_layer_ids=(2, 4),
        index_source_layer_ids=(2, 4, 5),
        candidate_source_layer_id=4,
        candidate_topk_blocks=2,
        candidate_block_size=4,
        index_n_heads=2,
        index_topk=4,
        index_head_dim=32,
        engram_layer_ids=(1, 3),
        engram_num_embeddings=(0, 0),
        engram_max_ngram_size=3,
        engram_vocab_size=50,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_compressed_vocab_size=40,
        engram_tables="device",
        num_nextn_predict_layers=2,
        dspark_block_size=4,
        dspark_noise_token_id=63,
        dspark_target_layer_ids=(3, 4, 5),
        dspark_markov_rank=8,
        dspark_n_routed_experts=4,
        dspark_num_experts_per_tok=2,
    )
    return replace(args, **overrides)


def _tiny_model() -> Model:
    layout = EngramLayout.from_args(_tiny_args())
    rows = tuple(sum(p for per in layer for p in per) for layer in layout.primes)
    args = _tiny_args(engram_num_embeddings=rows)
    mx.random.seed(0)
    model = Model(args)
    leaves = cast(list[tuple[str, mx.array]], tree_flatten(model.parameters()))
    updates: list[tuple[str, mx.array]] = []
    for name, value in leaves:
        if value.dtype == mx.uint8:
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


def _shapes(cache: LayerCache) -> list[tuple[int, ...] | None]:
    arrays = [
        cache.window,
        cache.compress_kv,
        cache.index_k,
        cache.pending_kv,
        cache.pending_gate,
        cache.ngram_history,
    ]
    if cache.draft is not None:
        arrays += cache.draft.windows
    return [None if a is None else tuple(a.shape) for a in arrays]


def test_keep_cuts_a_recorded_forward_to_its_first_tokens() -> None:
    model = _tiny_model()
    rng = np.random.default_rng(3)
    prompt = cast(list[int], rng.integers(0, 63, size=13).tolist())
    block = cast(list[int], rng.integers(0, 63, size=5).tolist())
    probe = cast(list[int], rng.integers(0, 63, size=1).tolist())
    for n in range(6):
        cut = model.make_cache()
        model(mx.array([prompt]), cut)
        for c in cut:
            c.begin_record()
        model(mx.array([block]), cut)
        for c in cut:
            c.keep(n)
        fresh = model.make_cache()
        model(mx.array([prompt]), fresh)
        if n:
            model(mx.array([block[:n]]), fresh)
        assert [_shapes(c) for c in cut] == [_shapes(c) for c in fresh]
        assert [c.offset for c in cut] == [c.offset for c in fresh]
        last_cut, last_fresh = cut[-1].draft, fresh[-1].draft
        assert last_cut is not None and last_fresh is not None
        assert last_cut.offset == last_fresh.offset == len(prompt) + n
        np.testing.assert_allclose(
            np.array(model(mx.array([probe]), cut)),
            np.array(model(mx.array([probe]), fresh)),
            rtol=0,
            atol=2e-3,
        )


def test_greedy_speculative_decoding_gives_the_greedy_tokens() -> None:
    model = _tiny_model()
    prompt = mx.array(
        cast(list[int], np.random.default_rng(5).integers(0, 63, size=11).tolist())
    )
    cache = model.make_cache()
    model(prompt[None, :-1], cache)
    token = prompt[-1:]
    plain: list[int] = []
    for _ in range(24):
        token = mx.argmax(model(token[None], cache)[0, -1:], axis=-1)
        plain.append(int(token.item()))
    speculative = [
        token
        for token, _ in speculative_generate_step(
            prompt,
            model,
            max_tokens=24,
            settings=_Sampling(0.0),
            prompt_cache=model.make_cache(),
        )
    ]
    assert speculative == plain


def test_draft_weights_map_from_the_mtp_names() -> None:
    model = _tiny_model()
    fp8 = mx.zeros((64, 64), dtype=mx.uint8)
    e8m0 = mx.full((2, 2), 127, dtype=mx.uint8)
    weights = {
        "embed.weight": mx.zeros((64, 64)),
        "mtp.0.attn.wq_a.weight": fp8[:32],
        "mtp.0.attn.wq_a.scale": e8m0[:1],
        "mtp.0.hc_attn_fn": mx.zeros((24, 256)),
        "mtp.0.main_proj.weight": fp8,
        "mtp.0.main_proj.scale": e8m0,
        "mtp.0.ffn.gate.bias_vl": mx.zeros((4,)),
        "mtp.1.norm.weight": mx.ones((64,)),
        "mtp.1.markov_head.embed.weight": mx.zeros((64, 8), dtype=mx.bfloat16),
        "mtp.1.markov_head.head.weight": mx.zeros((64, 8), dtype=mx.bfloat16),
        "mtp.1.confidence_head.proj.weight": mx.zeros((1, 72), dtype=mx.bfloat16),
        "mtp.1.ffn.experts.0.w1.weight": mx.zeros((32, 32), dtype=mx.uint8),
        "mtp.1.ffn.experts.1.w1.weight": mx.zeros((32, 32), dtype=mx.uint8),
        "mtp.2.attn.attn_sink": mx.zeros((4,)),
    }
    clean = model.sanitize(weights)
    draft = {k for k in clean if k.startswith("draft.")}
    assert draft == {
        "draft.layers.0.self_attn.q_a_proj.weight",
        "draft.layers.0.self_attn.q_a_proj.scales",
        "draft.layers.0.attn_hc.fn",
        "draft.main_proj.weight",
        "draft.main_proj.scales",
        "draft.norm.weight",
        "draft.markov_embed.weight",
        "draft.markov_head.weight",
        "draft.confidence",
        "draft.layers.1.mlp.experts.switch_mlp.gate_proj.weight",
    }
    assert clean["draft.confidence"].dtype == mx.float32
    assert clean["draft.layers.1.mlp.experts.switch_mlp.gate_proj.weight"].shape[0] == 2
    without = Model(_tiny_args(load_draft=False))
    assert not any(k.startswith("draft.") for k in without.sanitize(weights))
