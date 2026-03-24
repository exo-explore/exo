import os
import time
from typing import Any, cast

import mlx.core as mx
from mlx_lm.generate import BatchGenerator, generation_stream
from mlx_lm.models.cache import BatchRotatingKVCache

EXO_NO_BATCH_OPT = os.environ.get("EXO_NO_BATCH_OPT", "0") == "1"
_PRECOMPUTE_TOP_K = 20

_original_public_next = BatchGenerator.next

_pending_topk_idx: mx.array | None = None
_pending_topk_val: mx.array | None = None
_pending_selected_lps: mx.array | None = None


def _fast_brc_update_in_place(
    self: BatchRotatingKVCache, keys: mx.array, values: mx.array
) -> tuple[mx.array, mx.array]:
    if self._lengths is not None:
        raise RuntimeError(
            "finalize() should be called before decoding with BatchRotatingKVCache"
        )

    batch_size, n_kv_heads, seq_len, k_head_dim = keys.shape
    prev = self._offset
    if self.keys is None or (
        prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
    ):
        v_head_dim = values.shape[3]
        new_size = min(self.step, self.max_size - prev)
        k_shape = (batch_size, n_kv_heads, new_size, k_head_dim)
        v_shape = (batch_size, n_kv_heads, new_size, v_head_dim)
        new_k = mx.zeros(k_shape, keys.dtype)
        new_v = mx.zeros(v_shape, values.dtype)
        if self.keys is not None and self.values is not None:
            self.keys = mx.concatenate([self.keys, new_k], axis=2)
            self.values = mx.concatenate([self.values, new_v], axis=2)
        else:
            self.keys, self.values = new_k, new_v
        self._idx = prev

    assert self.keys is not None and self.values is not None
    trim_size = self.keys.shape[2] - self.max_size
    if trim_size > 0:
        self.keys = self._trim(trim_size, self.keys)
        self.values = self._trim(trim_size, self.values)
        self._idx = self.max_size
        self.left_padding -= trim_size

    if self._idx == self.max_size:
        self.rotated = True
        self._idx = 0
    if self.rotated:
        self.left_padding -= seq_len

    self.keys[..., self._idx : self._idx + seq_len, :] = keys
    self.values[..., self._idx : self._idx + seq_len, :] = values
    self._offset += seq_len
    self.offset += seq_len
    self._idx += seq_len

    self.keys = mx.depends(self.keys, (self.left_padding, self.offset))

    if self._offset < self.max_size:
        return self.keys[..., : self._offset, :], self.values[..., : self._offset, :]
    return self.keys, self.values


def _fast_next(self: BatchGenerator) -> list[BatchGenerator.Response]:
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

    logits = self.model(prev_tokens[:, None], cache=batch.cache)
    logits = logits[:, -1, :]

    if has_processors:
        processed_logits: list[mx.array] = []
        for e in range(batch_size):
            sample_logits: mx.array = logits[e : e + 1]
            for processor in batch.logits_processors[e]:
                sample_logits = processor(batch.tokens[e], sample_logits)
            processed_logits.append(sample_logits)
        logits = mx.concatenate(processed_logits, axis=0)

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

    global _pending_topk_idx, _pending_topk_val, _pending_selected_lps

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

    prev_token_list: list[int] = cast(list[int], prev_tokens.tolist())

    toc = time.perf_counter()
    self._stats.generation_time += toc - tic

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
        if emit_topk_indices:
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

    self._next_count += 1
    if self._next_count % 512 == 0:
        mx.clear_cache()
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
    if EXO_NO_BATCH_OPT:
        return
    BatchGenerator.next = _patched_public_next
    BatchRotatingKVCache._update_in_place = _fast_brc_update_in_place
