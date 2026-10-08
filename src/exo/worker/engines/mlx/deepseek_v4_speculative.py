"""Speculative decoding for DeepSeek V4 with its MTP block (single request).

Every round feeds the current token ``x`` together with ``K`` drafts of the
tokens after it: the MTP block drafts the first, and is chained on its own
output to draft the rest. Each next token is sampled from the main model's
logits exactly as in normal decoding, and a position's sample is used only
while every earlier draft matched the sample before it. The output
distribution is therefore exactly that of normal decoding: drafts only
decide how much of the round's work is kept.

Unmatched drafts are left in the KV caches, so each round snapshots the
cache state first and rolls the extra tokens back afterwards. Matched drafts
are tokens that have been fed but not yet returned; later steps return them
without a forward pass, or they are rolled back if the batch changes first.
"""

from dataclasses import dataclass, field
from typing import Any, cast

import mlx.core as mx
from mlx_lm.generate import GenerationBatch
from mlx_lm.models.cache import RotatingKVCache
from mlx_lm.models.deepseek_v4 import DeepseekV4Cache, Model

from exo.worker.engines.mlx import deepseek_v4_mtp as mtp_module
from exo.worker.engines.mlx.deepseek_v4_mtp import (
    DeepseekV4MTP,
    draft_argmax,
    get_mtp,
)
from exo.worker.runner.bootstrap import logger

DRAFT_TOKENS = 2
# The verification graph is submitted once after this many layers so the GPU
# starts while the CPU builds the rest (one extra commit per round).
_EARLY_SUBMIT_LAYER = 7


@dataclass
class _BranchSnapshot:
    buffer_kv: mx.array | None
    buffer_gate: mx.array | None
    buffer_count: int
    prev_kv: mx.array | None
    prev_gate: mx.array | None
    pool: mx.array | None
    buffer_lengths: list[int] | None
    pool_lengths: list[int] | None


@dataclass
class _CacheSnapshot:
    keys: mx.array | None
    values: mx.array | None
    offset: int
    idx: int
    branches: dict[str, _BranchSnapshot]


@dataclass
class _Recorded:
    """The compressor inputs of every token of a multi-token step."""

    kv: mx.array
    gate: mx.array
    ratio: int


@dataclass
class _Inflight:
    """A verification that has been submitted but not inspected."""

    current: mx.array
    drafts: mx.array
    logprobs: mx.array
    samples: mx.array
    hidden: mx.array
    snapshots: list[_CacheSnapshot]
    recorded: dict[tuple[int, str], _Recorded]


@dataclass
class _Pending:
    """Tokens of the last verification that were fed but not yet returned.

    ``fed`` tokens of that verification stay in the caches; the first
    ``returned`` of them have been handed to the caller.
    """

    tokens: list[int]
    logprobs: list[mx.array]
    snapshots: list[_CacheSnapshot]
    recorded: dict[tuple[int, str], _Recorded]
    fed: int
    returned: int


@dataclass
class _SpeculativeState:
    uids: tuple[int, ...]
    mtp_cache: DeepseekV4Cache
    drafts: mx.array | None = None
    pending: _Pending | None = None
    inflight: _Inflight | None = None
    rounds: int = 0
    tokens: int = 0
    accepted_per_position: list[int] = field(default_factory=lambda: [0] * DRAFT_TOKENS)


_recording: dict[tuple[int, str], _Recorded] | None = None
_original_accumulate = DeepseekV4Cache.accumulate_windows


def _recording_accumulate(
    self: DeepseekV4Cache,
    kv: mx.array,
    gate: mx.array,
    key: str,
    ratio: int,
    start_pos: int | mx.array,
) -> tuple[mx.array, mx.array, int | mx.array]:
    if _recording is not None:
        _recording[(id(self), key)] = _Recorded(kv, gate, ratio)
    return _original_accumulate(self, kv, gate, key, ratio, start_pos)


def _caches(batch: GenerationBatch) -> list[DeepseekV4Cache]:
    return cast(list[DeepseekV4Cache], batch.prompt_cache)


def _snapshot(cache: DeepseekV4Cache) -> _CacheSnapshot:
    branches = {
        key: _BranchSnapshot(
            buffer_kv=branch.buffer_kv,
            buffer_gate=branch.buffer_gate,
            buffer_count=branch.buffer_count,
            prev_kv=branch.prev_kv,
            prev_gate=branch.prev_gate,
            pool=branch.pool,
            buffer_lengths=branch.buffer_lengths,
            pool_lengths=branch.pool_lengths,
        )
        for key, branch in cache._branches.items()
    }
    local = cache.local
    assert isinstance(local, RotatingKVCache)
    return _CacheSnapshot(
        keys=local.keys,
        values=local.values,
        offset=local.offset,
        idx=local._idx,
        branches=branches,
    )


def _restore(cache: DeepseekV4Cache, snapshot: _CacheSnapshot) -> None:
    """Undo a whole multi-token forward. Its updates replace the cache's
    arrays rather than writing into them, so the snapshot's references still
    hold the earlier state."""
    local = cache.local
    assert isinstance(local, RotatingKVCache)
    local.keys = snapshot.keys
    local.values = snapshot.values
    local.offset = snapshot.offset
    local._idx = snapshot.idx
    for key, branch in cache._branches.items():
        before = snapshot.branches[key]
        branch.buffer_kv = before.buffer_kv
        branch.buffer_gate = before.buffer_gate
        branch.buffer_count = before.buffer_count
        branch.prev_kv = before.prev_kv
        branch.prev_gate = before.prev_gate
        branch.pool = before.pool
        branch.buffer_lengths = before.buffer_lengths
        branch.pool_lengths = before.pool_lengths
        branch._new_pool_lengths = None


def _carried_length(snapshot: _BranchSnapshot, ratio: int) -> int:
    if snapshot.buffer_kv is None:
        return 0
    if snapshot.buffer_kv.shape[1] == ratio:
        return snapshot.buffer_count
    return snapshot.buffer_kv.shape[1]


def _keep_tokens(
    cache: DeepseekV4Cache,
    snapshot: _CacheSnapshot,
    recorded: dict[tuple[int, str], _Recorded],
    keep: int,
    total: int,
) -> None:
    """Undo all but the first ``keep`` tokens of a ``total``-token forward.

    Steps are shorter than every compression ratio, so at most one window
    closes inside a step. If it closes on a kept token, the step's emitted
    row and overlap state stay and only the kept tokens after it are carried;
    otherwise the compressor goes back to its snapshot and the kept tokens
    are appended to its carry buffer.
    """
    drop = total - keep
    local = cache.local
    assert isinstance(local, RotatingKVCache)
    assert local.keys is not None and local.values is not None
    local.keys = local.keys[..., :-drop, :]
    local.values = local.values[..., :-drop, :]
    local.offset -= drop
    local._idx = local.keys.shape[2]

    for key, branch in cache._branches.items():
        record = recorded.get((id(cache), key))
        if record is None:
            continue
        before = snapshot.branches[key]
        assert total < record.ratio
        carried = _carried_length(before, record.ratio)
        closing = record.ratio - carried - 1
        branch._new_pool_lengths = None
        if 0 <= closing < keep:
            carry = keep - closing - 1
            branch.buffer_kv = record.kv[:, closing + 1 : keep] if carry else None
            branch.buffer_gate = record.gate[:, closing + 1 : keep] if carry else None
            branch.buffer_count = carry
            continue
        branch.prev_kv = before.prev_kv
        branch.prev_gate = before.prev_gate
        branch.pool = before.pool
        branch.buffer_lengths = before.buffer_lengths
        branch.pool_lengths = before.pool_lengths
        kept_kv = record.kv[:, :keep]
        kept_gate = record.gate[:, :keep]
        if carried and before.buffer_kv is not None and before.buffer_gate is not None:
            branch.buffer_kv = mx.concatenate(
                [before.buffer_kv[:, :carried], kept_kv], axis=1
            )
            branch.buffer_gate = mx.concatenate(
                [before.buffer_gate[:, :carried], kept_gate], axis=1
            )
        else:
            branch.buffer_kv = kept_kv
            branch.buffer_gate = kept_gate
        branch.buffer_count = carried + keep


def _state(batch: GenerationBatch) -> _SpeculativeState | None:
    return cast(_SpeculativeState | None, getattr(batch, "_exo_speculative", None))


def _set_state(batch: GenerationBatch, state: _SpeculativeState | None) -> None:
    batch._exo_speculative = state  # pyright: ignore[reportAttributeAccessIssue]


def _usable(batch: GenerationBatch) -> tuple[Model, DeepseekV4MTP] | None:
    model = batch.model
    mtp = get_mtp(model)
    if mtp is None or not isinstance(model, Model):
        return None
    if len(batch.uids) != 1:
        return None
    if batch.logits_processors is not None and any(batch.logits_processors):
        return None
    caches = cast(list[object], batch.prompt_cache)
    if not all(isinstance(c, DeepseekV4Cache) for c in caches):
        return None
    # A batch that held several requests keeps batch caches (e.g.
    # BatchRotatingKVCache) even once it is back to one.
    locals_: list[object] = [cast(DeepseekV4Cache, c).local for c in caches]
    if not all(isinstance(local, RotatingKVCache) for local in locals_):
        return None
    return model, mtp


def _sample(batch: GenerationBatch, logprobs: mx.array) -> mx.array:
    sampler = (batch.samplers[0] if batch.samplers else None) or batch.fallback_sampler
    return sampler(logprobs)


def _draft(
    model: Model,
    mtp: DeepseekV4MTP,
    state: _SpeculativeState,
    hidden: mx.array,
    next_ids: mx.array,
) -> mx.array:
    """Feed committed positions to the MTP block and draft ``DRAFT_TOKENS``
    tokens after the last one.

    Only committed positions stay in the MTP cache: the chained steps append
    entries built from drafts, which are dropped again afterwards (their
    effect is already captured in the lazy draft graph).
    """
    embed = model.model.embed_tokens
    out, block_state = mtp.forward_with_state(
        hidden, embed(next_ids), next_ids, state.mtp_cache
    )
    drafts = [draft_argmax(model, mtp, out[:, -1, :])]
    if DRAFT_TOKENS > 1:
        local = state.mtp_cache.local
        assert isinstance(local, RotatingKVCache)
        # Copies of the array handles: single-token cache updates write in
        # place, which would otherwise change what we restore below.
        keys = None if local.keys is None else mx.array(local.keys)
        values = None if local.values is None else mx.array(local.values)
        offset, idx = local.offset, local._idx
        chained = block_state[:, -1:]
        for _ in range(DRAFT_TOKENS - 1):
            ids = drafts[-1].reshape(1, 1).astype(mx.uint32)
            out, chained = mtp.forward_with_state(
                chained, embed(ids), ids, state.mtp_cache
            )
            drafts.append(draft_argmax(model, mtp, out[:, -1, :]))
        local.keys, local.values = keys, values
        local.offset, local._idx = offset, idx
    return mx.concatenate([d.reshape(1).astype(mx.uint32) for d in drafts])


def flush(batch: GenerationBatch) -> None:
    """Undo submitted and fed-but-unreturned tokens; leave speculative mode."""
    state = _state(batch)
    if state is None:
        return
    if state.inflight is not None:
        for cache, snapshot in zip(
            _caches(batch), state.inflight.snapshots, strict=True
        ):
            _restore(cache, snapshot)
        state.inflight = None
    pending = state.pending
    if pending is not None and pending.returned < pending.fed:
        for cache, snapshot in zip(_caches(batch), pending.snapshots, strict=True):
            _keep_tokens(
                cache,
                snapshot,
                pending.recorded,
                keep=pending.returned,
                total=pending.fed,
            )
        index = pending.returned - 1
        batch._next_tokens = mx.array([pending.tokens[index]], dtype=mx.uint32)
        batch._next_logprobs = pending.logprobs[index]
    _set_state(batch, None)


def after_normal_step(batch: GenerationBatch, sampled: mx.array) -> mx.array | None:
    """After a regular single-token step, draft the tokens after ``sampled``."""
    usable = _usable(batch)
    if usable is None:
        _set_state(batch, None)
        return None
    model, mtp = usable
    state = _state(batch)
    if state is None or state.uids != tuple(batch.uids):
        state = _SpeculativeState(
            uids=tuple(batch.uids),
            mtp_cache=DeepseekV4Cache(model.args.sliding_window),
        )
        _set_state(batch, state)
    hidden = cast(mx.array, getattr(model.model, "exo_last_hidden"))  # noqa: B009
    # Finish the step before drafting on top of it. Building the draft graph
    # on the step's unevaluated hidden state and submitting the two
    # separately intermittently corrupts the step's logits (NaN), as seen
    # after a flush. This only runs on the first step of a speculative run.
    next_logprobs = batch._next_logprobs
    assert isinstance(next_logprobs, mx.array)
    mx.eval(sampled, next_logprobs, hidden)
    state.drafts = _draft(model, mtp, state, hidden, sampled.reshape(-1, 1))
    return state.drafts


def _build(
    batch: GenerationBatch, model: Model, current: mx.array, drafts: mx.array
) -> _Inflight:
    """Build the verification of ``[current, *drafts]``: the caches advance by
    every token, and the returned snapshots hold their state from before."""
    global _recording
    snapshots = [_snapshot(c) for c in _caches(batch)]
    _recording = {}
    mtp_module.early_submit_layer = _EARLY_SUBMIT_LAYER
    try:
        inputs = mx.concatenate([current.reshape(1).astype(mx.uint32), drafts])
        logits = model(inputs[None], cache=batch.prompt_cache)
    finally:
        mtp_module.early_submit_layer = None
        recorded = _recording
        _recording = None
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    samples = mx.concatenate(
        [_sample(batch, logprobs[:, i, :]).reshape(1) for i in range(logprobs.shape[1])]
    )
    hidden = cast(mx.array, getattr(model.model, "exo_last_hidden"))  # noqa: B009
    return _Inflight(
        current=current,
        drafts=drafts,
        logprobs=logprobs,
        samples=samples,
        hidden=hidden,
        snapshots=snapshots,
        recorded=recorded,
    )


def _submit(state: _SpeculativeState, inflight: _Inflight) -> None:
    state.inflight = inflight
    state.drafts = inflight.drafts
    mx.async_eval(
        inflight.samples,
        inflight.logprobs,
        inflight.hidden,
        inflight.current,
        inflight.drafts,
    )


def speculative_step(
    batch: GenerationBatch,
) -> tuple[list[int], list[mx.array]] | None:
    """One generation step using MTP drafts, or None to run a normal step.

    Each round inspects the verification submitted by the previous round,
    keeps the matched prefix, drafts again and submits the next verification
    before returning, so the caller's token handling overlaps with the GPU.
    """
    state = _state(batch)
    usable = _usable(batch)
    if state is None or usable is None or state.uids != tuple(batch.uids):
        flush(batch)
        return None
    model, mtp = usable

    pending = state.pending
    if pending is not None and pending.returned < pending.fed:
        # A matched draft was already fed; return it without a forward.
        index = pending.returned - 1
        pending.returned += 1
        token = pending.tokens[index]
        batch.tokens[0].append(token)
        return [token], [pending.logprobs[index][0]]

    if state.inflight is None:
        if state.drafts is None:
            return None
        current = cast(mx.array, batch._next_tokens)
        _submit(state, _build(batch, model, current, state.drafts))
    inflight = state.inflight
    assert inflight is not None
    state.inflight = None
    current_logprobs = batch._next_logprobs

    samples = cast(list[int], inflight.samples.tolist())
    drafts = cast(list[int], inflight.drafts.tolist())
    token = cast(int, inflight.current.item())
    matched = 0
    while matched < len(drafts) and samples[matched] == drafts[matched]:
        state.accepted_per_position[matched] += 1
        matched += 1
    kept = matched + 1  # the current token plus the matched drafts

    total = len(drafts) + 1
    state.rounds += 1
    state.tokens += kept

    logprobs = inflight.logprobs
    if kept < total:
        for cache, snapshot in zip(_caches(batch), inflight.snapshots, strict=True):
            _keep_tokens(
                cache,
                snapshot,
                inflight.recorded,
                keep=kept,
                total=total,
            )
    state.pending = _Pending(
        tokens=samples[:matched],
        logprobs=[logprobs[:, i, :] for i in range(matched)],
        snapshots=inflight.snapshots,
        recorded=inflight.recorded,
        fed=kept,
        returned=1,
    )
    next_token = inflight.samples[matched : matched + 1]
    batch._next_tokens = next_token
    batch._next_logprobs = logprobs[:, matched, :]
    next_ids = inflight.samples[:kept].reshape(1, kept)
    drafts_next = _draft(model, mtp, state, inflight.hidden[:, :kept], next_ids)
    # Start the GPU on the drafts while the verification graph is built.
    mx.async_eval(drafts_next)
    _submit(state, _build(batch, model, next_token, drafts_next))

    if state.rounds % 100 == 0:
        logger.info(
            f"MTP speculative: {state.tokens / state.rounds:.2f} tokens/round, "
            f"draft hits {state.accepted_per_position} over {state.rounds} rounds"
        )
    batch.tokens[0].append(token)
    assert isinstance(current_logprobs, mx.array)
    return [token], [current_logprobs[0]]


_original_extend = GenerationBatch.extend
_original_filter = GenerationBatch.filter
_original_extract_cache = GenerationBatch.extract_cache


def _flushing_extend(self: GenerationBatch, batch: GenerationBatch) -> None:
    flush(self)
    flush(batch)
    _original_extend(self, batch)


def _flushing_filter(self: GenerationBatch, keep: list[int]) -> None:
    flush(self)
    _original_filter(self, keep)


def _flushing_extract_cache(self: GenerationBatch, idx: int) -> Any:  # pyright: ignore[reportAny]
    flush(self)
    return _original_extract_cache(self, idx)


def patch_speculative_batch() -> None:
    """Record compressor inputs, and leave speculative mode before the batch
    changes (its caches must then hold exactly the returned tokens)."""
    DeepseekV4Cache.accumulate_windows = _recording_accumulate
    GenerationBatch.extend = _flushing_extend
    GenerationBatch.filter = _flushing_filter
    GenerationBatch.extract_cache = _flushing_extract_cache
