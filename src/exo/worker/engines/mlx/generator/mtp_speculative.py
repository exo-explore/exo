"""Speculative decoding with a model's own MTP (multi-token-prediction) head, for one
request at a time, on one node or across pipeline stages.

A model takes part by implementing ``MTPSpeculativeModel`` (qwen4_exp does, with the head
in models/qwen3/qwen4_exp_mtp.py). Each step verifies the last token and up to two drafts
in one forward on every stage, which reads the weights once. The last stage, which has the
head, samples every row: a draft stands if the sample before it equals it, and the sample
after a kept draft follows it. This is exact for any sampler: every kept token is the
sample the model would have drawn from its distribution given the tokens before it. That
stage then drafts the next tokens from the head, and one small all-gather gives every rank
the decision, so the stages drop the same rejected rows (``spec_rollback``).

How many drafts a step sends (0, 1 or 2) is decided on that stage by ``DraftGate``: the
choice with the most expected tokens per millisecond. A verify step costs more than a plain
one, and how much more depends on the context length (attention reads every key once per
row), the hardware and the placement, so the gate times its own steps per (rows, context
band) and keeps the head's acceptance per bin of its probability; both carry over between
requests. ``EXO_MTP_GATES=static`` uses fixed thresholds instead:
- ``EXO_MTP_MIN_P1`` (0.6): below it the step is a plain one-token step;
- ``EXO_MTP_CHAIN_P1`` (0.9): chain a second draft only above it (the chain is a head call);
- ``EXO_MTP_DEPTH2`` (0.4): send the second draft when p1 * p2 >= depth2 * (1 + p1);
  ``off`` for one draft per step.
"""

import bisect
import os
import time
from collections.abc import Callable, Generator, Sequence
from typing import Protocol, cast

import mlx.core as mx
from mlx_lm.generate import GenerationResponse, generation_stream, wired_limit
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.runner.bootstrap import logger

# The last token and up to two drafts
SPEC_MAX_ROWS = 3


def _gate(name: str, default: str) -> float | None:
    value = os.environ.get(name, default).strip().lower()
    return None if value in ("off", "none", "") else float(value)


_MIN_P1 = _gate("EXO_MTP_MIN_P1", "0.6") or 0.0
_CHAIN_P1 = _gate("EXO_MTP_CHAIN_P1", "0.9") or 0.0
_DEPTH2 = _gate("EXO_MTP_DEPTH2", "0.4")
_ADAPTIVE = os.environ.get("EXO_MTP_GATES", "adaptive").strip().lower() != "static"

# Context bands (tokens) for the step costs: <4K, 4-8K, ..., 64-128K, >=128K
_BANDS = (4096, 8192, 16384, 32768, 65536, 131072)


def context_band(tokens: int) -> int:
    return bisect.bisect_right(_BANDS, tokens)


class DraftGate:
    """How many drafts the next step sends, on the stage with the head.

    With the head's probability p1 of its draft and acceptance a(p) estimated per bin of p
    (the head's probability counts as ``PRIOR_WEIGHT`` observations, then what was kept),
    a step with k drafts yields 1 + a1 (+ a1 * a2) tokens on average for the cost of a
    (k + 1)-row step in the current context band. A second draft is chained (one more head
    call) only when it could win with the best acceptance a chained draft has had. Costs
    not measured yet in a band are guessed low from another row count's, so that each
    choice gets tried; then exponential averages of the measured step times take over."""

    PRIOR_RATIO = {1: 1.0, 2: 1.3, 3: 1.6}
    DEFAULT_MS = 30.0
    PRIOR_WEIGHT = 8.0
    BINS = 10
    ALPHA = 0.1

    def __init__(self) -> None:
        self.cost: dict[tuple[int, int], float] = {}  # (band, rows) -> ms
        self.chain_ms = 2.0
        # [first or chained draft][probability bin] -> [kept, offered]
        self.accepted = [[[0, 0] for _ in range(self.BINS)] for _ in range(2)]

    def _bin(self, p: float) -> int:
        return min(max(int(p * self.BINS), 0), self.BINS - 1)

    def acceptance(self, which: int, p: float) -> float:
        kept, offered = self.accepted[which][self._bin(p)]
        return (kept + self.PRIOR_WEIGHT * p) / (offered + self.PRIOR_WEIGHT)

    def step_cost(self, band: int, rows: int) -> float:
        """ms: measured, or guessed from the nearest band's (any row count) or 30 ms."""
        measured = self.cost.get((band, rows))
        if measured is not None:
            return measured
        known = sorted(self.cost.items(), key=lambda item: abs(item[0][0] - band))
        if not known:
            return self.DEFAULT_MS * self.PRIOR_RATIO[rows]
        (_, near_rows), near_ms = known[0]
        return near_ms / self.PRIOR_RATIO[near_rows] * self.PRIOR_RATIO[rows]

    def first(self, band: int, p1: float) -> tuple[bool, bool]:
        """(send the head's draft, chain a second one) for a draft of probability p1."""
        if not _ADAPTIVE:
            send = p1 >= _MIN_P1
            return send, send and _DEPTH2 is not None and p1 >= _CHAIN_P1
        a1 = self.acceptance(0, p1)
        plain = 1 / self.step_cost(band, 1)
        one = (1 + a1) / self.step_cost(band, 2)
        best_two = (1 + a1 + a1 * self.acceptance(1, 0.999)) / (
            self.step_cost(band, 3) + self.chain_ms
        )
        chain = best_two > max(plain, one)
        return chain or one > plain, chain

    def second(self, band: int, p1: float, p2: float) -> bool:
        """Whether a chained draft of probability p2 goes with the first."""
        if not _ADAPTIVE:
            return _DEPTH2 is not None and p1 * p2 >= _DEPTH2 * (1 + p1)
        a1 = self.acceptance(0, p1)
        one = (1 + a1) / self.step_cost(band, 2)
        two = (1 + a1 + a1 * self.acceptance(1, p2)) / self.step_cost(band, 3)
        return two > one

    def record(
        self, band: int, rows: int, ms: float, p: tuple[float, float], kept: int
    ) -> None:
        """A step of ``rows`` rows took ``ms`` and kept ``kept`` of its drafts, whose
        probabilities were ``p``."""
        cost = self.cost.get((band, rows))
        self.cost[(band, rows)] = (
            ms
            if cost is None
            else (1 - self.ALPHA) * cost + self.ALPHA * min(ms, 3 * cost)
        )
        if rows >= 2:
            first = self.accepted[0][self._bin(p[0])]
            first[0] += int(kept >= 1)
            first[1] += 1
        if rows == 3 and kept >= 1:  # the chained draft's acceptance after a kept first
            second = self.accepted[1][self._bin(p[1])]
            second[0] += int(kept == 2)
            second[1] += 1

    def record_chain(self, ms: float) -> None:
        self.chain_ms = (1 - self.ALPHA) * self.chain_ms + self.ALPHA * ms

    def describe(self, band: int) -> str:
        if not _ADAPTIVE:
            return "gates static"
        costs = "/".join(
            f"{self.cost[(band, r)]:.1f}" if (band, r) in self.cost else "-"
            for r in (1, 2, 3)
        )
        return f"gates adaptive, band {band}: ms per 1/2/3-row step {costs}, chain {self.chain_ms:.1f}"


# One per process: what it has measured carries over between requests
_DRAFT_GATE = DraftGate()


class _Detokenizer(Protocol):
    @property
    def last_segment(self) -> str: ...
    def add_token(self, token: int) -> None: ...
    def finalize(self) -> None: ...


class MTPSpeculativeModel(Protocol):
    def spec_ready(self, caches: Sequence[object]) -> bool: ...
    def spec_is_last_stage(self) -> bool: ...
    def spec_forward(
        self, inputs: mx.array, caches: Sequence[object]
    ) -> mx.array | None: ...
    def spec_begin_verify(self, caches: Sequence[object], rows: int) -> None: ...
    def spec_end_verify(self) -> None: ...
    def spec_rollback(self, caches: Sequence[object], n: int) -> None: ...
    def spec_release(self, caches: Sequence[object]) -> None: ...
    def spec_track_head(self, on: bool) -> None: ...
    def spec_draft(
        self, next_tokens: mx.array, caches: Sequence[object], first: int = 0
    ) -> tuple[mx.array, mx.array]: ...
    def spec_chain(
        self, draft: int, row: int, caches: Sequence[object]
    ) -> tuple[mx.array, mx.array]: ...
    def spec_head_trim(self, caches: Sequence[object], n: int) -> None: ...
    def spec_offsets(self, caches: Sequence[object]) -> tuple[int, int]: ...


def mtp_speculative_agreed(
    model: Model,
    task: TextGenerationTaskParams,
    has_media: bool,
    caches: KVCacheType,
    group: mx.distributed.Group | None,
) -> bool:
    """Whether this request decodes speculatively. Every rank of a model that supports it
    asks, and they compare, so a node without the head (or with it turned off) only makes
    the request decode as before. Requests that want logprobs or carry images do."""
    spec_ready = cast(
        Callable[[Sequence[object]], bool] | None, getattr(model, "spec_ready", None)
    )
    if spec_ready is None:
        return False
    ready = not task.logprobs and not has_media and spec_ready(caches)
    if group is None:
        return ready
    gathered = mx.distributed.all_gather(
        mx.array([int(ready)], dtype=mx.int32), group=group
    )
    mx.eval(gathered)
    return bool(int(mx.min(gathered).item()))


def mtp_stream_generate(
    model: Model,
    tokenizer: TokenizerWrapper,
    prompt: list[int],
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]],
    caches: KVCacheType,
    group: mx.distributed.Group | None,
) -> Generator[GenerationResponse]:
    """mlx-lm's stream_generate from the prompt's last tokens (the rest is prefilled),
    speculatively. Yields the same responses, one per token."""
    m = cast(MTPSpeculativeModel, cast(object, model))
    detokenizer = cast(_Detokenizer, tokenizer.detokenizer)
    eos = set(tokenizer.eos_token_ids or ())
    no_logprobs = cast(mx.array, cast(object, None))  # never asked for here
    last_stage = m.spec_is_last_stage()
    history = mx.array(prompt)

    def pick(logits: mx.array, tokens: mx.array) -> mx.array:
        for processor in logits_processors:
            logits = processor(tokens, logits)
        return sampler(logits - mx.logsumexp(logits, keepdims=True))

    def share(values: list[int]) -> list[int]:
        """The last stage's values, on every rank."""
        if group is None:
            return values
        gathered = mx.distributed.all_gather(
            mx.array(values, dtype=mx.int32), group=group
        )
        mx.eval(gathered)
        rows = cast(list[list[int]], gathered.reshape(group.size(), -1).tolist())
        return rows[-1]

    gate = _DRAFT_GATE
    chained = (
        0  # a chained draft's pair in the head's cache, trimmed before its next pairs
    )
    sent_p = (0.0, 0.0)  # the probabilities of the drafts in flight
    chain_ms = 0.0  # this step's chain call, timed apart from the step

    def next_drafts(
        drafts: mx.array, probs: mx.array, row: int, band: int
    ) -> list[int]:
        """On the last stage: the drafts the next step sends, from the head's row ``row``
        (and one chained from it), as the gate decides."""
        nonlocal chained, sent_p, chain_ms
        d1, p1 = int(drafts[row].item()), float(probs[row].item())
        chained, chain_ms, sent_p = 0, 0.0, (p1, 0.0)
        send, chain = gate.first(band, p1)
        if not send:
            return []
        if not chain:
            return [d1]
        t0 = time.perf_counter()
        second, second_p = m.spec_chain(d1, row, caches)
        mx.eval(second, second_p)
        chain_ms = 1000 * (time.perf_counter() - t0)
        gate.record_chain(chain_ms)
        chained = 1
        p2 = float(second_p.item())
        sent_p = (p1, p2)
        return [d1, int(second.item())] if gate.second(band, p1, p2) else [d1]

    def encode(drafts: list[int]) -> list[int]:
        return drafts + [-1] * (SPEC_MAX_ROWS - 1 - len(drafts))

    steps = index = 0
    context = 0  # the trunk's length on the last stage, for the gate's context band
    finish: str | None = None
    token = prompt[-1]
    prompt_tps = 0.0
    by_rows = {r: [0, 0] for r in range(1, SPEC_MAX_ROWS + 1)}  # [steps, drafts kept]
    tic = time.perf_counter()
    with wired_limit(model, [generation_stream]), mx.stream(generation_stream):
        try:
            logits = m.spec_forward(mx.array([prompt]), caches)
            m.spec_track_head(False)
            if last_stage:
                assert logits is not None
                first = pick(logits[:, -1, :], history)
                drafts, probs = m.spec_draft(first, caches, first=len(prompt) - 1)
                mx.eval(first, drafts, probs)
                context = m.spec_offsets(caches)[0]
                band = context_band(context)
                values = [
                    int(first.item()),
                    *encode(next_drafts(drafts, probs, 0, band)),
                ]
            else:
                values = [0] * SPEC_MAX_ROWS
            first_token, *sent = share(values)
            ds = [d for d in sent if d >= 0]
            prompt_tps = len(prompt) / max(time.perf_counter() - tic, 1e-9)
            tic = time.perf_counter()
            pending, drafted = [first_token], [False]
            while True:
                for token, from_draft in zip(pending, drafted, strict=True):
                    if token in eos:
                        finish = "stop"
                        break
                    detokenizer.add_token(token)
                    if index + 1 == max_tokens:
                        finish = "length"
                        break
                    yield GenerationResponse(
                        text=detokenizer.last_segment,
                        token=token,
                        logprobs=no_logprobs,
                        from_draft=from_draft,
                        prompt_tokens=len(prompt),
                        prompt_tps=prompt_tps,
                        generation_tokens=index + 1,
                        generation_tps=(index + 1)
                        / max(time.perf_counter() - tic, 1e-9),
                        peak_memory=mx.get_peak_memory() / 1e9,
                        finish_reason=None,
                    )
                    index += 1
                if finish is not None:
                    break
                if logits_processors:
                    history = mx.concatenate(
                        [history, mx.array(pending, dtype=history.dtype)]
                    )
                inputs = [pending[-1], *ds]
                rows_n = len(inputs)
                step_p, step_band = sent_p, context_band(context + index)
                t_start = time.perf_counter()
                if rows_n > 1:
                    m.spec_begin_verify(caches, rows_n)
                try:
                    logits = m.spec_forward(mx.array([inputs]), caches)
                finally:
                    m.spec_end_verify()
                if last_stage:
                    assert logits is not None
                    # One GPU job: a sample per row (row i follows the drafts before it)
                    # and the head on (h_i, sample_i) for every row. The drafts kept are
                    # the longest run that the samples confirm; the head's pair for the
                    # first row not kept is the right one, so its draft is the next, and
                    # the pairs after it are dropped.
                    samples: list[mx.array] = []
                    for i in range(rows_n):
                        seen = (
                            mx.concatenate(
                                [history, mx.array(ds[:i], dtype=history.dtype)]
                            )
                            if logits_processors and i
                            else history
                        )
                        samples.append(pick(logits[:, i, :], seen))
                    rows = mx.concatenate(samples)
                    m.spec_head_trim(caches, chained)
                    drafts, probs = m.spec_draft(rows, caches)
                    mx.eval(rows, drafts, probs)
                    ys = cast(list[int], rows.tolist())
                    kept = 0
                    while kept < len(ds) and ys[kept] == ds[kept]:
                        kept += 1
                    m.spec_head_trim(caches, rows_n - 1 - kept)
                    new = next_drafts(
                        drafts, probs, kept, context_band(context + index + kept + 1)
                    )
                    ys += [0] * (SPEC_MAX_ROWS - len(ys))
                    values = [kept, *ys, *encode(new)]
                else:
                    values = [0] * (2 * SPEC_MAX_ROWS)
                shared = share(values)
                kept, ys = shared[0], shared[1 : 1 + SPEC_MAX_ROWS]
                if rows_n - 1 - kept:
                    m.spec_rollback(caches, rows_n - 1 - kept)
                if last_stage:
                    step_ms = 1000 * (time.perf_counter() - t_start) - chain_ms
                    gate.record(step_band, rows_n, step_ms, step_p, kept)
                by_rows[rows_n][0] += 1
                by_rows[rows_n][1] += kept
                steps += 1
                pending = ys[: kept + 1]
                drafted = [True] * kept + [False]
                ds = [d for d in shared[1 + SPEC_MAX_ROWS :] if d >= 0]
                if steps % 256 == 0:
                    mx.clear_cache()
        finally:
            m.spec_end_verify()
            m.spec_release(caches)
            if last_stage:
                m.spec_head_trim(caches, chained)  # leave the head in step
            m.spec_track_head(True)
            elapsed = time.perf_counter() - tic
            trunk_at, head_at = m.spec_offsets(caches)
            sizes = "; ".join(
                f"{r} rows: {st} steps, {kp}/{(r - 1) * st} drafts kept"
                for r, (st, kp) in by_rows.items()
                if st
            )
            logger.info(
                f"MTP speculative: {index + 1} tokens in {steps} steps "
                f"({(index + 1) / max(steps, 1):.2f} per step; {sizes}), "
                f"{(index + 1) / max(elapsed, 1e-9):.1f} tok/s; "
                f"cache offsets trunk {trunk_at}, head {head_at}"
                + (f"; {gate.describe(context_band(trunk_at))}" if last_stage else "")
            )
    detokenizer.finalize()
    yield GenerationResponse(
        text=detokenizer.last_segment,
        token=token,
        logprobs=no_logprobs,
        from_draft=False,
        prompt_tokens=len(prompt),
        prompt_tps=prompt_tps,
        generation_tokens=index + 1,
        generation_tps=(index + 1) / max(time.perf_counter() - tic, 1e-9),
        peak_memory=mx.get_peak_memory() / 1e9,
        finish_reason="stop" if token in eos else "length",
    )
