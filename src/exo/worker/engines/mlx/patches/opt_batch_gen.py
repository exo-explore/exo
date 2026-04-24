from dataclasses import dataclass, field
from typing import Protocol, cast

import mlx.core as mx
from mlx_lm.generate import GenerationBatch


class _ExtendableCache(Protocol):
    def extend(self, other: object) -> None: ...


def _extend_cache_inplace(
    cache_a: list[object], cache_b: list[object]
) -> list[object]:
    """Inline of mlx_lm.generate._extend_cache (private API, reimplemented).
    In-place appends entries of cache_b into entries of cache_a when both are
    non-empty; returns whichever list holds the merged result."""
    if not cache_a:
        return cache_b
    if not cache_b:
        return cache_a
    for ca, cb in zip(cache_a, cache_b, strict=True):
        cast(_ExtendableCache, ca).extend(cb)
    return cache_a

_PRECOMPUTE_TOP_K = 20


@dataclass
class BatchTopKLogprobs:
    uids: list[int] = field(default_factory=list)
    indices: mx.array | None = None
    values: mx.array | None = None
    selected: mx.array | None = None
    _uid_to_row: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._uid_to_row = {uid: i for i, uid in enumerate(self.uids)}

    def for_uid(self, uid: int) -> tuple[list[int], list[float], float] | None:
        if self.indices is None or self.values is None or self.selected is None:
            return None
        row = self._uid_to_row.get(uid)
        if row is None:
            return None
        return (
            cast(list[int], self.indices[row].tolist()),
            cast(list[float], self.values[row].tolist()),
            float(self.selected[row].item()),
        )


@dataclass
class _TopKBuffer:
    needs_topk: bool = False
    pending: BatchTopKLogprobs = field(default_factory=BatchTopKLogprobs)
    ready: BatchTopKLogprobs = field(default_factory=BatchTopKLogprobs)


def _get_buffer(batch: GenerationBatch) -> _TopKBuffer:
    buf = getattr(batch, "_topk_buffer", None)
    if buf is None:
        buf = _TopKBuffer()
        batch._topk_buffer = buf  # pyright: ignore[reportAttributeAccessIssue]
    return buf


def set_needs_topk(batch: GenerationBatch, needed: bool) -> None:
    _get_buffer(batch).needs_topk = needed


def take_ready_topk(batch: GenerationBatch) -> BatchTopKLogprobs:
    return _get_buffer(batch).ready


def _patched_step(self: GenerationBatch) -> tuple[list[int], list[mx.array]]:
    self._current_tokens = self._next_tokens
    self._current_logprobs = self._next_logprobs
    inputs = self._current_tokens
    assert inputs is not None, "_step requires initialized _next_tokens"

    buf = _get_buffer(self)
    buf.ready = buf.pending
    buf.pending = BatchTopKLogprobs()

    logits = self.model(inputs[:, None], cache=self.prompt_cache)
    logits = logits[:, -1, :]

    if self.logits_processors is not None and any(self.logits_processors):
        processed_logits: list[mx.array] = []
        for e in range(len(self.uids)):
            sample_logits = logits[e : e + 1]
            for processor in self.logits_processors[e]:
                sample_logits = processor(mx.array(self.tokens[e]), sample_logits)
            processed_logits.append(sample_logits)
        logits = mx.concatenate(processed_logits, axis=0)

    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    if self.samplers is not None and any(self.samplers):
        all_samples: list[mx.array] = []
        for e in range(len(self.uids)):
            sample_sampler = self.samplers[e] or self.fallback_sampler
            all_samples.append(sample_sampler(logprobs[e : e + 1]))
        sampled = mx.concatenate(all_samples, axis=0)
    else:
        sampled = self.fallback_sampler(logprobs)

    self._next_tokens = sampled
    self._next_logprobs = logprobs

    if buf.needs_topk:
        batch_size = len(self.uids)
        k = min(_PRECOMPUTE_TOP_K, logprobs.shape[1])
        pending_indices = mx.argpartition(-logprobs, k, axis=1)[:, :k]
        pending_values = mx.take_along_axis(logprobs, pending_indices, axis=1)
        sort_order = mx.argsort(-pending_values, axis=1)
        pending_indices = mx.take_along_axis(pending_indices, sort_order, axis=1)
        pending_values = mx.take_along_axis(pending_values, sort_order, axis=1)
        pending_selected = logprobs[mx.arange(batch_size), sampled]
        buf.pending = BatchTopKLogprobs(
            uids=list(self.uids),
            indices=pending_indices,
            values=pending_values,
            selected=pending_selected,
        )
        mx.async_eval(
            self._next_tokens,
            self._next_logprobs,
            pending_indices,
            pending_values,
            pending_selected,
        )
    else:
        mx.async_eval(self._next_tokens, self._next_logprobs)

    current_lp = self._current_logprobs
    if isinstance(current_lp, mx.array):
        mx.eval(inputs, current_lp)
    elif current_lp:
        mx.eval(inputs, *current_lp)
    else:
        mx.eval(inputs)

    token_list = cast(list[int], inputs.tolist())
    for sti, ti in zip(self.tokens, token_list, strict=True):
        sti.append(ti)

    if isinstance(current_lp, mx.array):
        current_lp = list(current_lp)
    return token_list, current_lp


def _as_array(lp: object) -> mx.array | None:
    if isinstance(lp, mx.array):
        return lp
    if isinstance(lp, list) and lp:
        return mx.stack(cast(list[mx.array], lp))
    return None


def _patched_extend(self: GenerationBatch, batch: GenerationBatch) -> None:
    """Upstream ``GenerationBatch.extend`` assumes ``_current_logprobs`` is
    always a list once a step has run, but ``_step`` sets it to the mx.array
    from the previous step's ``_next_logprobs``. When a new request's batch
    gets merged into an already-running one, ``list.extend`` is called on an
    mx.array and crashes. Normalize both sides before concatenating."""
    self.uids.extend(batch.uids)
    self.prompt_cache = _extend_cache_inplace(self.prompt_cache, batch.prompt_cache)
    self.tokens.extend(batch.tokens)
    if self.samplers is not None and batch.samplers is not None:
        self.samplers.extend(batch.samplers)
    if self.logits_processors is not None and batch.logits_processors is not None:
        self.logits_processors.extend(batch.logits_processors)
    self.max_tokens.extend(batch.max_tokens)
    self.state_machines.extend(batch.state_machines)

    if self._current_tokens is None:
        self._current_tokens = batch._current_tokens
        self._current_logprobs = batch._current_logprobs
    elif batch._current_tokens is not None:
        self._current_tokens = mx.concatenate(
            [self._current_tokens, batch._current_tokens]
        )
        a = _as_array(self._current_logprobs)
        b = _as_array(batch._current_logprobs)
        if a is None:
            self._current_logprobs = b if b is not None else []
        elif b is None:
            self._current_logprobs = a
        else:
            self._current_logprobs = mx.concatenate([a, b], axis=0)

    if self._next_tokens is None:
        self._next_tokens = batch._next_tokens
        self._next_logprobs = batch._next_logprobs
    elif batch._next_tokens is not None:
        self._next_tokens = mx.concatenate([self._next_tokens, batch._next_tokens])
        a = _as_array(self._next_logprobs)
        b = _as_array(batch._next_logprobs)
        if a is None:
            self._next_logprobs = b if b is not None else []
        elif b is None:
            self._next_logprobs = a
        else:
            self._next_logprobs = mx.concatenate([a, b], axis=0)

    self._token_context.extend(batch._token_context)
    self._num_tokens.extend(batch._num_tokens)
    self._matcher_states.extend(batch._matcher_states)


def apply_batch_gen_patch() -> None:
    GenerationBatch._step = _patched_step
    GenerationBatch.extend = _patched_extend
