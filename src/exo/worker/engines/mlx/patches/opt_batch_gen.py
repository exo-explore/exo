import os
import time
from typing import Any

import mlx.core as mx
from mlx_lm.generate import BatchGenerator, generation_stream
from mlx_lm.models.cache import ArraysCache, BatchKVCache, BatchRotatingKVCache, KVCache

EXO_NO_BATCH_OPT = os.environ.get("EXO_NO_BATCH_OPT", "0") == "1"

_original_public_next = BatchGenerator.next
_orig_brc_update_in_place = BatchRotatingKVCache._update_in_place


def _convert_cache(
    batch_cache: BatchKVCache | BatchRotatingKVCache | ArraysCache,
) -> KVCache | BatchRotatingKVCache | ArraysCache:
    if isinstance(batch_cache, BatchKVCache):
        c = KVCache()
        c.keys = batch_cache.keys
        c.values = batch_cache.values
        c.offset = batch_cache._idx
        c.make_mask = batch_cache.make_mask  # pyright: ignore[reportAttributeAccessIssue]
        return c
    return batch_cache


def _sync_back(fast_cache: KVCache, batch_cache: BatchKVCache) -> None:
    batch_cache.keys = fast_cache.keys
    batch_cache.values = fast_cache.values
    n_new = fast_cache.offset - batch_cache._idx
    batch_cache._idx = fast_cache.offset
    batch_cache.offset += n_new


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

    if self._offset < self.max_size:
        return self.keys[..., : self._offset, :], self.values[..., : self._offset, :]
    return self.keys, self.values


def _fast_next(self: BatchGenerator) -> list[BatchGenerator.Response]:
    tic = time.perf_counter()
    batch = self.active_batch
    assert batch is not None
    batch_size = len(batch)

    y = batch.y
    prev_logprobs = batch.logprobs

    has_processors = any(p for ps in batch.logits_processors for p in ps)
    if has_processors:
        for i, toks in enumerate(batch.tokens):
            batch.tokens[i] = mx.concatenate([toks, y[i : i + 1]])

    batch_cache: list[BatchKVCache | BatchRotatingKVCache | ArraysCache] = batch.cache
    fast_cache = [_convert_cache(c) for c in batch_cache]

    logits = self.model(y[:, None], cache=fast_cache)
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

    for fast_c, orig_c in zip(fast_cache, batch_cache, strict=True):
        if fast_c is not orig_c:
            _sync_back(fast_c, orig_c)  # pyright: ignore[reportArgumentType]

    if has_processors:
        mx.async_eval(batch.y, *batch.logprobs, *batch.tokens)
    else:
        mx.async_eval(batch.y, *batch.logprobs)

    y_list: list[int] = [int(y.item())] if batch_size == 1 else [int(v) for v in y]

    toc = time.perf_counter()
    self._stats.generation_time += toc - tic

    keep_idx: list[int] = []
    end_idx: list[int] = []
    responses: list[Any] = []
    stop_tokens = self.stop_tokens

    for e in range(batch_size):
        t = y_list[e]
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
        responses.append(self.Response(uid, t, prev_logprobs[e], finish_reason, cache))

    if end_idx:
        if keep_idx:
            batch.filter(keep_idx)
        else:
            self.active_batch = None

    self._next_count += 1
    if self._next_count % 512 == 0:
        mx.clear_cache()
    self._stats.generation_tokens += len(responses)
    return responses


def _has_arrays_cache(batch: Any) -> bool:
    return any(isinstance(c, ArraysCache) for c in batch.cache)


def _patched_public_next(self: BatchGenerator) -> list[BatchGenerator.Response]:
    batch = self.active_batch
    if batch is not None and not self.unprocessed_prompts:
        with mx.stream(generation_stream):
            return _fast_next(self)
    return _original_public_next(self)


def apply_batch_gen_patch() -> None:
    if EXO_NO_BATCH_OPT:
        return
    BatchGenerator.next = _patched_public_next
    BatchRotatingKVCache._update_in_place = _fast_brc_update_in_place
