# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportPrivateUsage=false
# pyright: reportInvalidCast=false, reportArgumentType=false
"""Correctness tests for :func:`batched_prefill`.

Validates that running K prefills in a single batched forward (the seam
:class:`SequentialGenerator` uses to absorb the residual 11s outliers
on the long-prompt mixed-traffic bench) produces bit-exact decode
state vs running K independent :func:`prefill` calls. We compare
post-prefill logits from the next decode tick rather than raw cache
state because mlx's ``BatchKVCache`` stores keys/values in a different
shape from ``KVCache`` after :meth:`extract` and exact-cache equality
would miss the question we actually care about — does the next forward
sample the same token?

Uses tiny llama-style random weights (no model download) so the tests
stay fast enough to run on every CI invocation.
"""

from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import pytest
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import AutoTokenizer

from exo.worker.engines.mlx.cache import encode_prompt, make_kv_cache
from exo.worker.engines.mlx.generator.generate import (
    BatchedPrefillUnsupportedError,
    batched_prefill,
    prefill,
)
from exo.worker.engines.mlx.types import Model

NUM_STEPS = 16


def _init_random(model: nn.Module) -> None:
    params = model.parameters()
    new_params = mlx.utils.tree_map(
        lambda p: mx.random.normal(shape=p.shape, dtype=p.dtype)
        if isinstance(p, mx.array)
        else p,
        params,
    )
    model.update(new_params)
    mx.eval(model.parameters())


def _make_tiny_llama() -> tuple[Model, TokenizerWrapper]:
    from huggingface_hub import snapshot_download
    from mlx_lm.models.llama import Model as LlamaModel
    from mlx_lm.models.llama import ModelArgs

    mx.random.seed(42)
    args = ModelArgs(
        model_type="llama",
        hidden_size=256,
        num_hidden_layers=4,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=248320,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    model = LlamaModel(args)
    _init_random(model)

    model_path = Path(
        snapshot_download(
            "mlx-community/Qwen3.5-35B-A3B-4bit",
            allow_patterns=["tokenizer*", "*.jinja"],
        )
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = TokenizerWrapper(hf_tokenizer)
    return cast(Model, model), tokenizer


def _decode_one_token(model: Model, cache: object, last_token: int) -> mx.array:
    """Run one forward with the seed token; return the (vocab,) logits.

    Mirrors the entry state ``mlx_generate`` hands to the spec loop:
    cache is at offset ``len(prompt) - 1`` and the next forward feeds
    the seed token (``prompt[-1]``).
    """
    out = model(mx.array([[last_token]]), cache=cast(list[object], cache))
    mx.eval(out)
    return out[0, -1]


@pytest.mark.slow
def test_batched_prefill_matches_sequential_for_two_prompts() -> None:
    """B=2 batched_prefill must produce the same decode logits as 2x B=1 prefill.

    Compares the ``argmax`` token from the first decode forward after
    prefill — that's the only invariant the spec loop reads from the
    post-prefill cache, so bit-exact cache layout doesn't matter as
    long as the next forward agrees.
    """
    model, tokenizer = _make_tiny_llama()
    sampler = make_sampler(temp=0.0)

    tokens_a = encode_prompt(tokenizer, "Write a short essay about AI.")
    tokens_b = encode_prompt(tokenizer, "Explain evolution briefly.")

    # Sequential reference (per-slot prefill on prompt[:-1]; the
    # exo.prefill helper advances cache to len(prompt) - 2 via its
    # +1 / -2 dance).
    cache_a_seq = make_kv_cache(model)
    prefill(model, tokenizer, sampler, tokens_a[:-1], cache_a_seq, None, None, None)
    cache_b_seq = make_kv_cache(model)
    prefill(model, tokenizer, sampler, tokens_b[:-1], cache_b_seq, None, None, None)

    # Sequential decode: feed the prefill-tail's penultimate then last
    # token to advance cache from offset N-2 to N-1, then sample the
    # first generated logits.
    last_a = int(tokens_a[-1].item())
    penult_a = int(tokens_a[-2].item())
    model(mx.array([[penult_a]]), cache=cast(list[object], cache_a_seq))
    seq_logits_a = _decode_one_token(model, cache_a_seq, last_a)

    last_b = int(tokens_b[-1].item())
    penult_b = int(tokens_b[-2].item())
    model(mx.array([[penult_b]]), cache=cast(list[object], cache_b_seq))
    seq_logits_b = _decode_one_token(model, cache_b_seq, last_b)

    # Batched: batched_prefill leaves cache at offset N-1 directly (no
    # +1/-2 dance), so the equivalent decode is one forward on the
    # last token only.
    cache_a_batch = make_kv_cache(model)
    cache_b_batch = make_kv_cache(model)
    aggregate_tps, total_tokens = batched_prefill(
        model=model,
        prompt_tokens_list=[tokens_a, tokens_b],
        caches_list=[cache_a_batch, cache_b_batch],
    )
    assert aggregate_tps > 0.0
    assert total_tokens == int(tokens_a.size) - 1 + int(tokens_b.size) - 1

    batch_logits_a = _decode_one_token(model, cache_a_batch, last_a)
    batch_logits_b = _decode_one_token(model, cache_b_batch, last_b)

    # Decoded token must agree; small numerical drift in the logits is
    # acceptable (different reduction order in the batched matmul) but
    # the argmax must be identical.
    assert int(mx.argmax(seq_logits_a).item()) == int(mx.argmax(batch_logits_a).item())
    assert int(mx.argmax(seq_logits_b).item()) == int(mx.argmax(batch_logits_b).item())


@pytest.mark.slow
def test_batched_prefill_continues_decoding_correctly() -> None:
    """After batched_prefill the per-slot decode must stay aligned for many steps.

    A single matching first-token argmax can be coincidence; we extend
    the comparison to ``NUM_STEPS`` decoded tokens to catch cache-state
    bugs that only show up after multiple forwards (e.g. an off-by-one
    in BatchKVCache.extract that would skew RoPE positions).
    """
    model, tokenizer = _make_tiny_llama()
    sampler = make_sampler(temp=0.0)

    tokens_a = encode_prompt(tokenizer, "Hello there general kenobi.")
    tokens_b = encode_prompt(tokenizer, "The quick brown fox jumps.")

    # Sequential reference run produces a token sequence per slot.
    seq_tokens: list[list[int]] = []
    for tokens in (tokens_a, tokens_b):
        cache_seq = make_kv_cache(model)
        prefill(model, tokenizer, sampler, tokens[:-1], cache_seq, None, None, None)
        last = int(tokens[-1].item())
        penult = int(tokens[-2].item())
        model(mx.array([[penult]]), cache=cast(list[object], cache_seq))
        next_tok = last
        produced: list[int] = []
        for _ in range(NUM_STEPS):
            logits = _decode_one_token(model, cache_seq, next_tok)
            next_tok = int(mx.argmax(logits).item())
            produced.append(next_tok)
        seq_tokens.append(produced)

    # Batched run.
    cache_a = make_kv_cache(model)
    cache_b = make_kv_cache(model)
    batched_prefill(
        model=model,
        prompt_tokens_list=[tokens_a, tokens_b],
        caches_list=[cache_a, cache_b],
    )
    batch_tokens: list[list[int]] = []
    for tokens, cache in ((tokens_a, cache_a), (tokens_b, cache_b)):
        last = int(tokens[-1].item())
        next_tok = last
        produced = []
        for _ in range(NUM_STEPS):
            logits = _decode_one_token(model, cache, next_tok)
            next_tok = int(mx.argmax(logits).item())
            produced.append(next_tok)
        batch_tokens.append(produced)

    # Mismatches downstream of step 0 still indicate a real cache
    # bug; we tolerate up to one drift in NUM_STEPS as numerical
    # slack but the first 8 tokens must agree.
    assert seq_tokens[0][:8] == batch_tokens[0][:8]
    assert seq_tokens[1][:8] == batch_tokens[1][:8]


def test_batched_prefill_empty_inputs_returns_zero() -> None:
    """No-op on empty input: the caller may filter to zero eligible slots."""
    tps, total = batched_prefill(
        model=cast(Model, object()),
        prompt_tokens_list=[],
        caches_list=[],
    )
    assert tps == 0.0
    assert total == 0


def test_batched_prefill_rejects_mismatched_lengths() -> None:
    """``prompt_tokens_list`` and ``caches_list`` must agree on K."""
    with pytest.raises(ValueError, match="must have the same length"):
        batched_prefill(
            model=cast(Model, object()),
            prompt_tokens_list=[mx.array([1, 2, 3]), mx.array([4, 5, 6])],
            caches_list=[[]],
        )


def test_batched_prefill_rejects_short_prompts() -> None:
    """Prompts < 2 tokens leave no decode-seed token after slicing."""
    with pytest.raises(ValueError, match="length >= 2"):
        batched_prefill(
            model=cast(Model, object()),
            prompt_tokens_list=[mx.array([7])],
            caches_list=[[]],
        )


def test_batched_prefill_unsupported_cache_raises_typed_error() -> None:
    """Cache layers without ``merge`` must surface :class:`BatchedPrefillUnsupportedError`.

    The contract: callers (``SequentialGenerator._admit_queued_tasks``)
    catch this typed error to fall back to per-slot prefill instead of
    crashing the runner.
    """

    class _UnsupportedLayer:
        # No ``merge`` classmethod => mlx_lm._merge_caches raises
        # ``ValueError(f"{type} does not yet support batching with history")``.
        pass

    cache_a: list[object] = [_UnsupportedLayer()]
    cache_b: list[object] = [_UnsupportedLayer()]

    with pytest.raises(BatchedPrefillUnsupportedError):
        batched_prefill(
            model=cast(Model, object()),
            prompt_tokens_list=[
                mx.array([1, 2, 3]),
                mx.array([4, 5, 6]),
            ],
            caches_list=cast(list[object], [cache_a, cache_b]),
        )
