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

Which drafts a step sends depends on the head's probability p1 of its draft (and p2 of a
second draft chained from it), since a verify step costs more than a plain one:
- ``EXO_MTP_MIN_P1`` (0.6): below it the step is a plain one-token step;
- ``EXO_MTP_CHAIN_P1`` (0.9): chain a second draft only above it (the chain is a head call);
- ``EXO_MTP_DEPTH2`` (0.4): send the second draft when p1 * p2 >= depth2 * (1 + p1);
  ``off`` for one draft per step.
The defaults were tuned on replayed agent requests (Qwen3.8-Flash-Next, two Macs).
"""

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

    chained = (
        0  # a chained draft's pair in the head's cache, trimmed before its next pairs
    )

    def next_drafts(drafts: mx.array, probs: mx.array, row: int) -> list[int]:
        """On the last stage: the drafts the next step sends, from the head's row ``row``
        (and one chained from it), as the gates allow."""
        nonlocal chained
        d1, p1 = int(drafts[row].item()), float(probs[row].item())
        chained = 0
        if p1 < _MIN_P1:
            return []
        if _DEPTH2 is None or p1 < _CHAIN_P1:
            return [d1]
        second, second_p = m.spec_chain(d1, row, caches)
        mx.eval(second, second_p)
        chained = 1
        p2 = float(second_p.item())
        return [d1, int(second.item())] if p1 * p2 >= _DEPTH2 * (1 + p1) else [d1]

    def encode(drafts: list[int]) -> list[int]:
        return drafts + [-1] * (SPEC_MAX_ROWS - 1 - len(drafts))

    steps = index = 0
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
                values = [int(first.item()), *encode(next_drafts(drafts, probs, 0))]
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
                    new = next_drafts(drafts, probs, kept)
                    ys += [0] * (SPEC_MAX_ROWS - len(ys))
                    values = [kept, *ys, *encode(new)]
                else:
                    values = [0] * (2 * SPEC_MAX_ROWS)
                shared = share(values)
                kept, ys = shared[0], shared[1 : 1 + SPEC_MAX_ROWS]
                if rows_n - 1 - kept:
                    m.spec_rollback(caches, rows_n - 1 - kept)
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
