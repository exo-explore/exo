# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportPrivateUsage=false, reportUnusedFunction=false
"""Speculative decoding with Qwen3.8-Flash-Next's MTP head (qwen4_exp_mtp.py and
generator/mtp_speculative.py), on the tiny model of test_qwen4_exp.py:
- a verify step's rollback leaves the recurrent states exactly as without the rejected
  drafts (whatever they were), and decoding goes on as from plain one-token steps;
- the converted head loads, and attaches only when its file is present;
- the head's cache follows the trunk's through chunked prefill;
- speculative greedy decoding gives the plain greedy tokens whether every draft is kept,
  none is, or some are.
"""

from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map
from mlx_lm.models.cache import ArraysCache
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.engines.mlx.generator import mtp_speculative
from exo.worker.engines.mlx.models.qwen3 import qwen4_exp_mtp
from exo.worker.engines.mlx.models.qwen3.qwen4_exp import (
    Model,
    QSAKVCache,
    TextArgs,
)
from exo.worker.engines.mlx.models.qwen3.qwen4_exp_mtp import (
    MTPTracker,
    Qwen4ExpMTP,
    attach_mtp,
    load_mtp_head,
)
from exo.worker.engines.mlx.models.qwen3.tests.test_qwen4_exp import (
    _attached_tiny_model,
)
from exo.worker.engines.mlx.types import KVCacheType
from exo.worker.engines.mlx.types import Model as ExoModel


@pytest.fixture(autouse=True)
def _on_the_cpu() -> Generator[None]:
    """On the CPU: a GPU's float32 matmul of 2 rows can differ from one row's by ~1e-3
    here (another kernel), which would flip near-tied greedy picks of this tiny model;
    the CPU's differ by rounding only, so the tests can check exactness."""
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(device)


def _same(a: mx.array, b: mx.array) -> bool:
    return bool(mx.array_equal(a, b).item())


def _close(a: mx.array, b: mx.array, rel: float = 1e-4) -> bool:
    scale = max(float(mx.abs(b).max().item()), 1e-6)
    return float(mx.abs(a - b).max().item()) <= rel * scale


def _verify(
    model: Model, cache: list[ArraysCache | QSAKVCache], tokens: list[int]
) -> mx.array:
    model.spec_begin_verify(cache, len(tokens))
    try:
        out = model(mx.array([tokens]), cache)
        mx.eval(out)
    finally:
        model.spec_end_verify()
    return out


def _same_states(
    a: list[ArraysCache | QSAKVCache], b: list[ArraysCache | QSAKVCache]
) -> bool:
    for a_cache, b_cache in zip(a, b, strict=True):
        if isinstance(a_cache, ArraysCache):
            assert isinstance(b_cache, ArraysCache)
            for x, y in zip(a_cache.cache, b_cache.cache, strict=True):
                if x is None or y is None or not _same(x, y):
                    return False
        elif a_cache.offset != cast(QSAKVCache, b_cache).offset:
            return False
    return True


def _prefilled(
    model: Model, tokens: list[int], copies: int
) -> list[list[ArraysCache | QSAKVCache]]:
    caches = [model.make_cache() for _ in range(copies)]
    for cache in caches:  # past the indexer's budget, so attention is sparse
        mx.eval(model(mx.array([tokens]), cache))
    return caches


def test_rollback_keeps_the_state_after_the_first_token(tmp_path: Path) -> None:
    model = _attached_tiny_model(tmp_path)
    tokens = [int(t) for t in np.random.default_rng(5).integers(0, 240, size=60)]
    plain, wrong_a, wrong_b, kept = _prefilled(model, tokens[:40], 4)
    t, nxt = tokens[40], tokens[41]

    one = model(mx.array([[t]]), plain)
    mx.eval(one)
    rows_a = _verify(model, wrong_a, [t, 7])
    rows_b = _verify(model, wrong_b, [t, 9])
    # The first row doesn't depend on the draft, and is a one-token step's
    assert _same(rows_a[:, :1], rows_b[:, :1])
    assert _close(rows_a[:, :1], one)
    model.spec_rollback(wrong_a, 1)
    model.spec_rollback(wrong_b, 1)
    # What a rejected draft rolls back to doesn't depend on it, bit for bit
    assert _same_states(wrong_a, wrong_b)
    # ... and decoding goes on from it as from a plain step
    after_plain = model(mx.array([[nxt]]), plain)
    after_rollback = model(mx.array([[nxt]]), wrong_a)
    assert _close(after_rollback, after_plain)
    assert model.spec_offsets(wrong_a)[0] == model.spec_offsets(plain)[0]
    # A kept draft: its row is the next one-token step's, and decoding goes on from it
    rows = _verify(model, kept, [t, nxt])
    assert _close(rows[:, 1:], after_plain)
    after_kept = model(mx.array([[tokens[42]]]), kept)
    want = model(mx.array([[tokens[42]]]), plain)
    assert _close(after_kept, want)


def test_three_row_steps_roll_back_one_or_two_drafts(tmp_path: Path) -> None:
    model = _attached_tiny_model(tmp_path)
    tokens = [int(t) for t in np.random.default_rng(6).integers(0, 240, size=60)]
    plain, two_a, two_b, one_a, one_b = _prefilled(model, tokens[:40], 5)
    t, u, w = tokens[40], tokens[41], tokens[42]
    want_t = model(mx.array([[t]]), plain)
    want_u = model(mx.array([[u]]), plain)
    mx.eval(want_t, want_u)
    # Both drafts rejected
    rows_a = _verify(model, two_a, [t, 5, 6])
    rows_b = _verify(model, two_b, [t, 9, 3])
    assert _same(rows_a[:, :1], rows_b[:, :1]) and _close(rows_a[:, :1], want_t)
    model.spec_rollback(two_a, 2)
    model.spec_rollback(two_b, 2)
    assert _same_states(two_a, two_b)
    # The second draft rejected: the first stays, whatever the second was
    rows_c = _verify(model, one_a, [t, u, 5])
    rows_d = _verify(model, one_b, [t, u, 8])
    assert _same(rows_c[:, :2], rows_d[:, :2]) and _close(rows_c[:, 1:2], want_u)
    model.spec_rollback(one_a, 1)
    model.spec_rollback(one_b, 1)
    assert _same_states(one_a, one_b)
    # Decoding goes on from each as from plain steps
    assert _close(model(mx.array([[u]]), two_a), want_u)  # two_a holds t only
    want_w = model(mx.array([[w]]), plain)
    assert _close(model(mx.array([[w]]), one_a), want_w)  # one_a holds t, u
    with pytest.raises(RuntimeError, match="no state to roll back"):
        model.spec_rollback(one_a, 1)  # its kept states were used


def test_verify_steps_take_two_or_three_rows(tmp_path: Path) -> None:
    model = _attached_tiny_model(tmp_path)
    cache = model.make_cache()
    for rows in (1, 4):
        with pytest.raises(ValueError, match="2 to 3 rows"):
            model.spec_begin_verify(cache, rows)


def _random_head(args: TextArgs) -> Qwen4ExpMTP:
    mx.random.seed(3)
    head = Qwen4ExpMTP(args)
    head.update(
        tree_map(
            lambda p: mx.random.normal(p.shape) * 0.1 if p.dtype == mx.float32 else p,
            head.parameters(),
        )
    )
    mx.eval(head.parameters())
    return head


def test_the_head_attaches_only_with_its_converted_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _attached_tiny_model(tmp_path / "model")
    model_path = tmp_path / "mlx-community--tiny"
    monkeypatch.setattr(qwen4_exp_mtp, "MTP_ROOT", tmp_path / "mtp")
    layers = len(model.layers)
    assert not attach_mtp(model, model_path)
    assert model.language_model.mtp is None and len(model.make_cache()) == layers

    # Converted as the script does: 4-bit linears (8-bit router gates), "mtp." names
    args = model.language_model.args
    head = _random_head(args)
    nn.quantize(
        head,
        class_predicate=lambda name, module: (
            {"group_size": 64, "bits": 8}
            if name.endswith(("mlp.gate", "shared_expert_gate"))
            else {"group_size": 32, "bits": 4}
        )
        if isinstance(module, nn.Linear) and module.weight.shape[-1] % 32 == 0
        else False,
    )
    weights = dict(
        ("mtp." + k, v)
        for k, v in cast(list[tuple[str, mx.array]], tree_flatten(head.parameters()))
    )
    path = qwen4_exp_mtp.mtp_weights_path(model_path)
    path.parent.mkdir(parents=True)
    mx.save_safetensors(str(path), weights)
    loaded = load_mtp_head(args, path)
    streams = mx.random.normal((1, 5, args.hidden_size * args.hc_count))
    embeddings = mx.random.normal((1, 5, args.hidden_size))
    assert _same(
        loaded(streams, embeddings, QSAKVCache()),
        head(streams, embeddings, QSAKVCache()),
    )

    monkeypatch.setenv("EXO_QWEN4_EXP_MTP", "0")
    assert not attach_mtp(model, model_path)
    monkeypatch.delenv("EXO_QWEN4_EXP_MTP")
    assert attach_mtp(model, model_path)
    caches = model.make_cache()
    assert len(caches) == layers + 1 and isinstance(caches[-1], QSAKVCache)
    assert model.spec_ready(caches) and not model.spec_ready(caches[:layers])


def _with_head(tmp_path: Path) -> Model:
    model = _attached_tiny_model(tmp_path)
    head = _random_head(model.language_model.args)
    object.__setattr__(model.language_model, "mtp", MTPTracker(head, None))
    object.__setattr__(model.model, "keep_streams", True)
    return model


def test_the_heads_cache_follows_chunked_prefill(tmp_path: Path) -> None:
    model = _with_head(tmp_path)
    tracker = model.language_model.mtp
    assert tracker is not None
    tokens = mx.array(
        [[int(t) for t in np.random.default_rng(2).integers(0, 240, size=70)]]
    )
    # Every pair at once: the trunk's streams h_i with x_{i+1}
    model(tokens)
    streams = model.model.last_streams
    assert streams is not None
    want = QSAKVCache()
    embedded = model.model.embed_tokens(tokens)
    tracker.head(streams[:, :-1], embedded[:, 1:], want)
    mx.eval(want.keys, want.values)

    for chunks in ([70], [9, 30, 1, 30], [24] + [1] * 46):
        cache = model.make_cache()
        start = 0
        for length in chunks:
            model(tokens[:, start : start + length], cache)
            start += length
        head_cache = cache[-1]
        assert isinstance(head_cache, QSAKVCache) and head_cache.offset == 69, chunks
        assert want.keys is not None and head_cache.keys is not None
        assert _close(head_cache.keys[..., :69, :], want.keys[..., :69, :]), chunks

    # exo trims two positions after prefill (restoring the recurrent states from a
    # snapshot); the next forward continues the pairs
    restored = _prefilled(model, cast(list[int], tokens[0, :58].tolist()), 1)[0]
    cache = model.make_cache()
    model(tokens[:, :60], cache)
    for c in cache:
        if isinstance(c, QSAKVCache):
            c.trim(2)  # the trunk's attention layers and the head
    for i, c in enumerate(cache):
        if isinstance(c, ArraysCache):
            cache[i] = restored[
                i
            ]  # the recurrent states as exo's snapshot restores them
    model(tokens[:, 58:62], cache)
    head_cache = cache[-1]
    assert isinstance(head_cache, QSAKVCache) and head_cache.offset == 61
    assert head_cache.keys is not None and want.keys is not None
    assert _close(head_cache.keys[..., :61, :], want.keys[..., :61, :])


class _Detokenizer:
    def __init__(self) -> None:
        self.last_segment = ""

    def add_token(self, token: int) -> None:
        self.last_segment = f"<{token}>"

    def finalize(self) -> None:
        self.last_segment = ""


class _Tokenizer:
    eos_token_ids: Sequence[int] = ()

    def __init__(self) -> None:
        self.detokenizer = _Detokenizer()


def _greedy(logprobs: mx.array) -> mx.array:
    return mx.argmax(logprobs, axis=-1)


def _plain_greedy(model: Model, prompt: list[int], count: int) -> list[int]:
    cache = model.make_cache()
    model(mx.array([prompt[:-2]]), cache)
    logits = model(mx.array([prompt[-2:]]), cache)
    out = [int(_greedy(logits[:, -1]).item())]
    while len(out) < count:
        logits = model(mx.array([[out[-1]]]), cache)
        out.append(int(_greedy(logits[:, -1]).item()))
    return out


@pytest.mark.parametrize("oracle", ["all kept", "none kept", "some kept"])
@pytest.mark.parametrize("depth2", [None, 0.3], ids=["one draft", "two drafts"])
def test_speculative_greedy_decoding_gives_the_plain_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, oracle: str, depth2: float | None
) -> None:
    model = _with_head(tmp_path)
    prompt = [int(t) for t in np.random.default_rng(4).integers(0, 240, size=44)]
    count = 40
    plain = _plain_greedy(model, prompt, count + 4)
    # Drafts every step (two with depth2): the oracle decides what they are
    monkeypatch.setattr(mtp_speculative, "_MIN_P1", 0.0)
    monkeypatch.setattr(mtp_speculative, "_CHAIN_P1", 0.0)
    monkeypatch.setattr(mtp_speculative, "_DEPTH2", depth2)
    rng = np.random.default_rng(7)
    start = len(prompt)  # the trunk's position of the first generated token
    last_rows = [1]

    def guess(position: int) -> int:
        right = plain[position - start]
        keep = oracle == "all kept" or (oracle == "some kept" and rng.random() < 0.7)
        return right if keep else (right + 1) % 240

    real_draft = model.spec_draft
    real_chain = model.spec_chain

    def spec_draft(
        next_tokens: mx.array, caches: Sequence[object], first: int = 0
    ) -> tuple[mx.array, mx.array]:
        _, probs = real_draft(
            next_tokens, caches, first
        )  # keeps the head's cache in step
        rows = int(next_tokens.shape[0])
        last_rows[0] = rows
        trunk = model.spec_offsets(caches)[0]
        # these are the forward's last rows; row i's sample is the token at trunk - rows
        # + i + 1, and its draft guesses the one after
        drafts = [guess(trunk - rows + i + 2) for i in range(rows)]
        return mx.array(drafts), mx.ones_like(probs) * 0.99

    def spec_chain(
        draft: int, row: int, caches: Sequence[object]
    ) -> tuple[mx.array, mx.array]:
        _, probs = real_chain(draft, row, caches)
        # the token after row ``row``'s draft (the trunk hasn't moved since spec_draft)
        position = model.spec_offsets(caches)[0] - last_rows[0] + row + 3
        return mx.array([guess(position)]), mx.ones_like(probs) * 0.99

    monkeypatch.setattr(model, "spec_draft", spec_draft)
    monkeypatch.setattr(model, "spec_chain", spec_chain)

    cache = model.make_cache()
    model(mx.array([prompt[:-2]]), cache)
    tokens = [
        out.token
        for out in mtp_speculative.mtp_stream_generate(
            cast(ExoModel, cast(object, model)),
            cast(TokenizerWrapper, cast(object, _Tokenizer())),
            prompt[-2:],
            count,
            cast(Callable[[mx.array], mx.array], _greedy),
            [],
            cast(KVCacheType, cache),
            None,
        )
    ]
    assert tokens[:count] == plain[:count]
    trunk, head = model.spec_offsets(cache)
    assert head == trunk or head == trunk - 1


@pytest.mark.parametrize("quantized", [False, True])
def test_the_draft_vocabulary_reads_its_rows_of_the_lm_head(
    tmp_path: Path, quantized: bool
) -> None:
    model = _attached_tiny_model(tmp_path)
    text = model.language_model
    if quantized:
        nn.quantize(
            text,
            group_size=32,
            bits=4,
            class_predicate=lambda name, _: name == "lm_head",
        )
        assert isinstance(text.lm_head, nn.QuantizedLinear)
    ids = mx.array([0, 3, 17, 100, 250, 255], dtype=mx.int32)
    out = qwen4_exp_mtp._draft_output_layer(text, ids)
    x = mx.random.normal((1, 3, text.args.hidden_size))
    # a matmul over fewer rows: the same up to rounding
    assert _close(out(x), mx.take(text.lm_head(x), ids, axis=-1), 1e-5)
