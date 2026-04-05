import gc
import os
import time
from typing import Any, cast

import mlx.core as mx
from mlx_lm.generate import BatchGenerator, generation_stream

_PRECOMPUTE_TOP_K = 20

_TRACE = os.environ.get("EXO_TRACING_ENABLED", "false").lower() in ("true", "1")

_original_public_next = BatchGenerator.next

_pending_topk_idx: mx.array | None = None
_pending_topk_val: mx.array | None = None
_pending_selected_lps: mx.array | None = None


# ── profiling state (only active when EXO_TRACING_ENABLED=1) ──
class _Prof:
    interval = 64
    step_count = 0
    forward = 0.0
    sample = 0.0
    async_eval = 0.0
    tolist = 0.0
    topk_tolist = 0.0
    response_build = 0.0
    clear_cache = 0.0
    gc_collections = 0
    last_gc0 = -1
    # per-step max for spotting outliers
    tolist_max = 0.0
    forward_max = 0.0

    @classmethod
    def reset(cls) -> None:
        cls.forward = 0.0
        cls.sample = 0.0
        cls.async_eval = 0.0
        cls.tolist = 0.0
        cls.topk_tolist = 0.0
        cls.response_build = 0.0
        cls.clear_cache = 0.0
        cls.gc_collections = 0
        cls.tolist_max = 0.0
        cls.forward_max = 0.0


def _fast_next(self: BatchGenerator) -> list[BatchGenerator.Response]:
    global _pending_topk_idx, _pending_topk_val, _pending_selected_lps

    tic = time.perf_counter()
    batch = self.active_batch
    assert batch is not None
    batch_size = len(batch)

    prev_tokens = batch.y
    prev_logprobs = batch.logprobs

    has_processors = any(p for ps in batch.logits_processors for p in ps)
    if has_processors:
        for i, toks in enumerate(batch.tokens):
            batch.tokens[i] = mx.concatenate([toks, prev_tokens[i : i + 1]])

    # ── forward pass (lazy graph build) ──
    _t0 = time.perf_counter()
    logits = self.model(prev_tokens[:, None], cache=batch.cache)
    logits = logits[:, -1, :]
    _fwd_dt = time.perf_counter() - _t0
    if _TRACE:
        _Prof.forward += _fwd_dt
        _Prof.forward_max = max(_Prof.forward_max, _fwd_dt)

    if has_processors:
        processed_logits: list[mx.array] = []
        for e in range(batch_size):
            sample_logits: mx.array = logits[e : e + 1]
            for processor in batch.logits_processors[e]:
                sample_logits = processor(batch.tokens[e], sample_logits)
            processed_logits.append(sample_logits)
        logits = mx.concatenate(processed_logits, axis=0)

    # ── sampling ──
    _t0 = time.perf_counter()
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    if (
        batch_size == 1
        or any(batch.samplers)
        and all(s is batch.samplers[0] for s in batch.samplers)
    ):
        sampler = batch.samplers[0] or self.sampler
        batch.y = sampler(logprobs)
    elif any(batch.samplers):
        all_samples: list[mx.array] = []
        for e in range(batch_size):
            s = batch.samplers[e] or self.sampler
            all_samples.append(s(logprobs[e : e + 1]))
        batch.y = mx.concatenate(all_samples, axis=0)
    else:
        batch.y = self.sampler(logprobs)
    batch.logprobs = list(logprobs)
    if _TRACE:
        _Prof.sample += time.perf_counter() - _t0

    # ── topk tolist from PREVIOUS step (blocks on prior async_eval) ──
    _t0 = time.perf_counter()
    emit_topk_indices: list[list[int]] = (
        cast(list[list[int]], _pending_topk_idx.tolist())
        if _pending_topk_idx is not None
        else []
    )
    emit_topk_values: list[list[float]] = (
        cast(list[list[float]], _pending_topk_val.tolist())
        if _pending_topk_val is not None
        else []
    )
    emit_selected_lps: list[float] = (
        cast(list[float], _pending_selected_lps.tolist())
        if _pending_selected_lps is not None
        else []
    )
    if _TRACE:
        _Prof.topk_tolist += time.perf_counter() - _t0

    # ── async eval (queue current step for GPU) ──
    _t0 = time.perf_counter()
    needs_topk: bool = getattr(self, "_needs_topk", False)
    if needs_topk:
        k = min(_PRECOMPUTE_TOP_K, logprobs.shape[1])
        _pending_topk_idx = mx.argpartition(-logprobs, k, axis=1)[:, :k]
        _pending_topk_val = mx.take_along_axis(logprobs, _pending_topk_idx, axis=1)
        sort_order = mx.argsort(-_pending_topk_val, axis=1)
        _pending_topk_idx = mx.take_along_axis(_pending_topk_idx, sort_order, axis=1)
        _pending_topk_val = mx.take_along_axis(_pending_topk_val, sort_order, axis=1)
        _pending_selected_lps = logprobs[mx.arange(batch_size), batch.y]
        mx.async_eval(
            batch.y,
            *batch.logprobs,
            *batch.tokens,
            _pending_topk_idx,
            _pending_topk_val,
            _pending_selected_lps,
        )
    else:
        _pending_topk_idx = None
        _pending_topk_val = None
        _pending_selected_lps = None
        mx.async_eval(batch.y, *batch.logprobs, *batch.tokens)
    if _TRACE:
        _Prof.async_eval += time.perf_counter() - _t0

    # ── tolist (blocks until prev token materialized) ──
    _t0 = time.perf_counter()
    prev_token_list: list[int] = cast(list[int], prev_tokens.tolist())
    _tolist_dt = time.perf_counter() - _t0
    if _TRACE:
        _Prof.tolist += _tolist_dt
        _Prof.tolist_max = max(_Prof.tolist_max, _tolist_dt)

    toc = time.perf_counter()
    self._stats.generation_time += toc - tic

    # ─��� response building ──
    _t0 = time.perf_counter()
    keep_idx: list[int] = []
    end_idx: list[int] = []
    responses: list[Any] = []
    stop_tokens = self.stop_tokens

    for e in range(batch_size):
        t = prev_token_list[e]
        uid = batch.uids[e]
        num_tok = batch.num_tokens[e] + 1
        batch.num_tokens[e] = num_tok

        if t in stop_tokens:
            finish_reason = "stop"
            end_idx.append(e)
        elif num_tok >= batch.max_tokens[e]:
            finish_reason = "length"
            end_idx.append(e)
        else:
            finish_reason = None
            keep_idx.append(e)

        cache = None
        if finish_reason is not None:
            cache = batch.extract_cache(e)
        response = self.Response(uid, t, prev_logprobs[e], finish_reason, cache)
        if emit_topk_indices and e < len(emit_topk_indices):
            response._topk_indices = emit_topk_indices[e]  # pyright: ignore[reportAttributeAccessIssue]
            response._topk_values = emit_topk_values[e]  # pyright: ignore[reportAttributeAccessIssue]
            response._selected_logprob = emit_selected_lps[e]  # pyright: ignore[reportAttributeAccessIssue]
        responses.append(response)

    if end_idx:
        if keep_idx:
            batch.filter(keep_idx)
            if (
                _pending_topk_idx is not None
                and _pending_topk_val is not None
                and _pending_selected_lps is not None
            ):
                ki = mx.array(keep_idx)
                _pending_topk_idx = _pending_topk_idx[ki]
                _pending_topk_val = _pending_topk_val[ki]
                _pending_selected_lps = _pending_selected_lps[ki]
        else:
            self.active_batch = None
            _pending_topk_idx = None
            _pending_topk_val = None
            _pending_selected_lps = None
    if _TRACE:
        _Prof.response_build += time.perf_counter() - _t0

    self._next_count += 1
    if self._next_count % 512 == 0:
        _t0 = time.perf_counter()
        mx.clear_cache()
        if _TRACE:
            _Prof.clear_cache += time.perf_counter() - _t0

    # ── GC tracking + periodic log ──
    if _TRACE:
        gc_stats = gc.get_stats()
        gen0 = gc_stats[0]["collections"]
        if _Prof.last_gc0 < 0:
            _Prof.last_gc0 = gen0
        if gen0 != _Prof.last_gc0:
            _Prof.gc_collections += gen0 - _Prof.last_gc0
            _Prof.last_gc0 = gen0

        _Prof.step_count += 1
        if _Prof.step_count % _Prof.interval == 0:
            n = _Prof.interval
            from loguru import logger as _logger
            _logger.info(
                f"[PROF decode x{n}] "
                f"forward={_Prof.forward/n*1000:.2f}ms(max={_Prof.forward_max*1000:.1f}) "
                f"sample={_Prof.sample/n*1000:.2f}ms "
                f"async_eval={_Prof.async_eval/n*1000:.2f}ms "
                f"tolist={_Prof.tolist/n*1000:.2f}ms(max={_Prof.tolist_max*1000:.1f}) "
                f"topk_tolist={_Prof.topk_tolist/n*1000:.2f}ms "
                f"response={_Prof.response_build/n*1000:.2f}ms "
                f"clear_cache={_Prof.clear_cache/n*1000:.2f}ms "
                f"gc_collections={_Prof.gc_collections}"
            )
            _Prof.reset()

    self._stats.generation_tokens += len(responses)
    return responses


def _patched_public_next(self: BatchGenerator) -> list[BatchGenerator.Response]:
    batch = self.active_batch
    # Only do decode with fast_next
    if batch is not None and not self.unprocessed_prompts:
        with mx.stream(generation_stream):
            return _fast_next(self)
    return _original_public_next(self)


def apply_batch_gen_patch() -> None:
    BatchGenerator.next = _patched_public_next
