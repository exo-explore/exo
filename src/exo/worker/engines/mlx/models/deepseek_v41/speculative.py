"""Speculative decoding with DeepSeek-V4.1's DSpark draft head, distribution-preserving.

Each step drafts up to `block_size` tokens with the draft head, runs the main model once
over the last sampled token and the drafts, and keeps the drafts by rejection sampling:
draft i (drawn from the draft distribution q_i) is accepted with probability
min(1, p_i / q_i), where p_i is exactly the distribution exo's sampler draws from (the
logits processors, top-p / min-p / top-k filters and the temperature applied to the main
model's logits). At the first rejection the token comes from the normalized residual
max(p_i - q_i, 0); when every draft is accepted, one more comes from the next p. Every
emitted token is therefore distributed exactly as without speculation. At temperature 0
both sides are greedy and a draft is kept when it equals the main model's argmax.

The main model's caches record the verify pass and keep only its accepted tokens.
"""

import os
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

from exo.worker.engines.mlx.models.deepseek_v41 import kernels
from exo.worker.engines.mlx.models.deepseek_v41.model import LayerCache, Model

LogitsProcessor = Callable[[mx.array, mx.array], mx.array]


class SamplingSettings(Protocol):
    """The parameters of exo's sampler (mlx-lm's make_sampler): the family module's
    SamplingParameters."""

    @property
    def temperature(self) -> float: ...
    @property
    def top_p(self) -> float: ...
    @property
    def min_p(self) -> float: ...
    @property
    def top_k(self) -> int: ...


@dataclass
class SpeculativeStats:
    """Counts of the last speculative generation (for logs and measurements)."""

    steps: int = 0
    drafted: int = 0
    accepted: int = 0


LAST_STATS = SpeculativeStats()


class DraftLength:
    """Picks each step's number of drafts from the recent acceptance (it depends only on
    past steps, so the output distribution is unchanged). With per-draft acceptance a,
    k drafts give (1 - a^(k+1)) / (1 - a) tokens per step at a cost that grows linearly
    with k (relative to one plain step: draft_cost + verify_cost * k); it takes the k with
    the most tokens per unit cost."""

    def __init__(
        self,
        most: int,
        *,
        draft_cost: float = 0.17,
        verify_cost: float = 0.18,
        prior: float = 0.65,
        memory: float = 0.85,
    ):
        self.most = most
        self.draft_cost = draft_cost
        self.verify_cost = verify_cost
        self.memory = memory
        self.tried = 1.0
        self.kept = prior

    def update(self, drafted: int, accepted: int) -> None:
        # drafts are tried up to the first rejection
        tried = min(accepted + 1, drafted)
        self.tried = self.memory * self.tried + tried
        self.kept = self.memory * self.kept + accepted

    def next(self) -> int:
        rate = min(max(self.kept / max(self.tried, 1e-9), 0.0), 0.99)
        best, best_value = 1, 0.0
        for k in range(1, self.most + 1):
            tokens = (1 - rate ** (k + 1)) / (1 - rate)
            value = tokens / (1 + self.draft_cost + self.verify_cost * k)
            if value > best_value:
                best, best_value = k, value
        return best


def speculative_enabled() -> bool:
    return os.environ.get("EXO_SPECULATIVE_DECODING", "1") != "0"


def draft_length() -> int | None:
    value = os.environ.get("EXO_SPECULATIVE_DRAFT_TOKENS")
    return int(value) if value else None


def filtered(logprobs: mx.array, settings: SamplingSettings) -> mx.array:
    """The sampler's filters, in make_sampler's order (removed tokens at -inf)."""
    if 0 < settings.top_p < 1.0:
        logprobs = apply_top_p(logprobs, settings.top_p)
    if settings.min_p != 0.0:
        logprobs = apply_min_p(logprobs, settings.min_p, 1)
    if settings.top_k > 0:
        logprobs = apply_top_k(logprobs, settings.top_k)
    return logprobs


def distribution(logprobs: mx.array, settings: SamplingSettings) -> mx.array:
    """The probabilities the sampler draws from: categorical(filtered * (1 / T))."""
    scaled = filtered(logprobs, settings) * (1.0 / settings.temperature)
    return mx.softmax(scaled.astype(mx.float32), axis=-1, precise=True)


def _process(
    processors: Sequence[LogitsProcessor],
    history: mx.array,
    logits: mx.array,
) -> mx.array:
    for processor in processors:
        logits = processor(history, logits)
    return logits


def _log_softmax(logits: mx.array) -> mx.array:
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


@dataclass
class _Draft:
    tokens: list[mx.array]  # [1] each
    probs: list[mx.array]  # [V] each (empty when greedy)


def _draft(
    model: Model,
    token: mx.array,
    start_pos: int,
    cache: Sequence[LayerCache],
    length: int,
    settings: SamplingSettings,
    processors: Sequence[LogitsProcessor],
    history: mx.array,
) -> _Draft:
    draft = model.draft
    state = cache[-1].draft
    assert draft is not None and state is not None
    hidden, _ = draft.block(token, start_pos, state, model.model.embed_tokens)
    base = kernels.head_logits(hidden[0, :length], model.lm_head.weight)  # [n, V]
    tokens: list[mx.array] = []
    probs: list[mx.array] = []
    previous = token
    for i in range(length):
        bias, _ = draft.markov(previous)
        logits = base[i : i + 1] + bias
        if processors:
            logits = _process(
                processors, mx.concatenate([history, token, *tokens]), logits
            )
        logprobs = _log_softmax(logits)
        if settings.temperature == 0:
            chosen = mx.argmax(logprobs, axis=-1)
        else:
            q = distribution(logprobs, settings)
            chosen = mx.random.categorical(mx.log(q))
            probs.append(q[0])
        tokens.append(chosen.astype(mx.int32))
        previous = chosen
    return _Draft(tokens, probs)


def _verify(
    logprobs: mx.array,
    draft: _Draft,
    settings: SamplingSettings,
) -> tuple[mx.array, mx.array]:
    """(the number of accepted drafts, the token after them), both [1] arrays."""
    count = len(draft.tokens)
    drafted = mx.concatenate(draft.tokens)  # [n]
    if settings.temperature == 0:
        target = mx.argmax(logprobs, axis=-1).astype(mx.int32)  # [n + 1]
        accepted = mx.equal(drafted, target[:count])
        n_accepted = mx.sum(mx.cumprod(accepted.astype(mx.int32)))
        return n_accepted.reshape(1), target[n_accepted].reshape(1)
    p = distribution(logprobs, settings)  # [n + 1, V]
    q = mx.stack(draft.probs)  # [n, V]
    rows = mx.arange(count)
    p_d = p[rows, drafted]
    q_d = q[rows, drafted]
    u = mx.random.uniform(shape=(count,))
    accepted = u * q_d < p_d
    n_accepted = mx.sum(mx.cumprod(accepted.astype(mx.int32)))
    residual = mx.maximum(p[:count] - q, 0.0)
    total = mx.sum(residual, axis=-1, keepdims=True)
    # a rejection implies q > p at the draft, so the residual has mass; guard anyway
    residual = mx.where(total > 0, residual / mx.maximum(total, 1e-30), p[:count])
    choices = mx.concatenate([residual, p[count:]], axis=0)  # [n + 1, V]
    sampled = mx.random.categorical(mx.log(choices)).astype(mx.int32)  # [n + 1]
    return n_accepted.reshape(1), sampled[n_accepted].reshape(1)


def speculative_generate_step(
    prompt: mx.array,
    model: Model,
    *,
    max_tokens: int,
    settings: SamplingSettings,
    prompt_cache: Sequence[LayerCache],
    logits_processors: Sequence[LogitsProcessor] = (),
    draft_tokens: int | None = None,
) -> Generator[tuple[int, mx.array]]:
    """Like mlx-lm's generate_step: yields (token, log-probs of the distribution it came
    from, after the logits processors) until max_tokens."""
    draft = model.draft
    assert draft is not None and prompt_cache[-1].draft is not None
    fixed = min(draft_tokens, draft.block_size) if draft_tokens else None
    lengths = DraftLength(draft.block_size)
    processors = list(logits_processors)
    stats = LAST_STATS
    stats.steps = stats.drafted = stats.accepted = 0
    history = prompt.astype(mx.int32)
    if prompt.size > 1:
        model(prompt[None, :-1], list(prompt_cache))
    logits = model(prompt[None, -1:], list(prompt_cache))[:, -1]
    if processors:
        logits = _process(processors, history, logits)
    logprobs = _log_softmax(logits)
    if settings.temperature == 0:
        token = mx.argmax(logprobs, axis=-1).astype(mx.int32)
    else:
        token = mx.random.categorical(mx.log(distribution(logprobs, settings))).astype(
            mx.int32
        )
    mx.eval(token, logprobs)
    yield int(token.item()), logprobs[0]
    produced = 1
    while produced < max_tokens:
        start = prompt_cache[0].offset
        n = min(fixed or lengths.next(), max_tokens - produced)
        proposal = _draft(
            model, token, start, prompt_cache, n, settings, processors, history
        )
        # the GPU drafts while the verify pass is built
        mx.async_eval(*proposal.tokens, *proposal.probs)
        for c in prompt_cache:
            c.begin_record()
        inputs = mx.concatenate([token, *proposal.tokens])
        logits = model(inputs[None], list(prompt_cache))[0]  # [n + 1, V]
        if processors:
            rows = [
                _process(
                    processors,
                    mx.concatenate([history, inputs[: i + 1]]),
                    logits[i : i + 1],
                )
                for i in range(n + 1)
            ]
            logits = mx.concatenate(rows, axis=0)
        logprobs = _log_softmax(logits)
        n_accepted, after = _verify(logprobs, proposal, settings)
        mx.eval(n_accepted, after, inputs, logprobs)
        accepted = int(n_accepted.item())
        stats.steps += 1
        stats.drafted += n
        stats.accepted += accepted
        lengths.update(n, accepted)
        for c in prompt_cache:
            c.keep(accepted + 1)
        emitted = inputs[1 : accepted + 1].tolist() + [int(after.item())]
        if processors:
            history = mx.concatenate([history, inputs[: accepted + 1]])
        for i, t in enumerate(emitted):
            yield int(t), logprobs[i]
            produced += 1
            if produced >= max_tokens:
                return
        token = after


def token_generator(
    model: nn.Module,
    prompt: mx.array,
    max_tokens: int,
    sampling: SamplingSettings,
    logits_processors: Sequence[LogitsProcessor],
    prompt_cache: Sequence[Any],
) -> Generator[tuple[int, mx.array]] | None:
    """The family's decode: speculative when the model has its draft head (and it is not
    turned off with EXO_SPECULATIVE_DECODING=0), else None (mlx-lm's generate_step)."""
    if not isinstance(model, Model) or model.draft is None or not speculative_enabled():
        return None
    caches = [
        c for c in cast(Sequence[object], prompt_cache) if isinstance(c, LayerCache)
    ]
    if not caches or len(caches) != len(prompt_cache):
        return None
    if caches[-1].draft is None:
        return None
    return speculative_generate_step(
        prompt,
        model,
        max_tokens=max_tokens,
        settings=sampling,
        prompt_cache=caches,
        logits_processors=logits_processors,
        draft_tokens=draft_length(),
    )
