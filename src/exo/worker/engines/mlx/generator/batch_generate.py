import time
from dataclasses import dataclass, field
from typing import Any, Callable, cast

import mlx.core as mx
from mlx_lm.generate import (
    BatchGenerator as MlxBatchGenerator,
)
from mlx_lm.models.cache import (
    BatchKVCache,
    RotatingKVCache,
    create_causal_mask,
    dynamic_roll,
)
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.api import (
    CompletionTokensDetails,
    FinishReason,
    GenerationStats,
    PromptTokensDetails,
    TopLogprobItem,
    Usage,
)
from exo.shared.types.memory import Memory
from exo.shared.types.mlx import KVCacheType, Model
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    encode_prompt,
    make_kv_cache,
)
from exo.worker.engines.mlx.constants import DEFAULT_TOP_LOGPROBS, MAX_TOKENS
from exo.worker.engines.mlx.generator.generate import (
    ban_token_ids,
    eos_ids_from_tokenizer,
    extract_top_logprobs,
    prefill,
)
from exo.worker.engines.mlx.utils_mlx import fix_unmatched_think_end_tokens
from exo.worker.runner.bootstrap import logger


# ---------------------------------------------------------------------------
# Right-padded BatchKVCache patch
#
# mlx_lm's BatchKVCache left-pads shorter sequences, shifting data to different
# positions.  This changes Metal's SDPA kernel tiling boundaries and produces
# different results vs B=1 (97.7% token mismatch at 4000 tokens on real models).
#
# Right-padding keeps data at position 0 for every sequence so the SDPA kernel
# sees identical data layout regardless of batch composition → bit-exact output.
# ---------------------------------------------------------------------------
def _rpad_init(self: Any, left_padding: list[int]) -> None:
    self.keys = None
    self.values = None
    # Keep original left-padding behavior for __init__ (used for left-padded prefill).
    # Right-padding is only set up by merge() when combining B=1 caches for generation.
    self.left_padding = mx.array(left_padding)
    self._rpad = mx.zeros(len(left_padding), dtype=mx.int32)
    self._has_rpad = False
    self.offset = mx.array([-lp for lp in left_padding])
    self._idx = 0
    self._right_padding = None
    self.step = 256
    self._py_offsets = None


@classmethod
def _rpad_merge(cls: type, caches: list[Any]) -> Any:
    lengths = [c.size() for c in caches]
    max_length = max(lengths)
    padding = [max_length - length for length in lengths]
    batch_size = len(caches)
    n_heads = max(c.keys.shape[1] for c in caches if c.keys is not None)
    dk = max(c.keys.shape[3] for c in caches if c.keys is not None)
    dv = max(c.values.shape[3] for c in caches if c.values is not None)
    dt = next(iter(c.keys.dtype for c in caches if c.keys is not None))

    keys = mx.zeros((batch_size, n_heads, max_length, dk), dtype=dt)
    values = mx.zeros((batch_size, n_heads, max_length, dv), dtype=dt)
    for i, c in enumerate(caches):
        if c.keys is None:
            continue
        keys[i : i + 1, :, 0 : c.offset] = c.keys[..., : c.offset, :]
        values[i : i + 1, :, 0 : c.offset] = c.values[..., : c.offset, :]

    cache = BatchKVCache.__new__(BatchKVCache)
    cache.keys = keys
    cache.values = values
    cache.left_padding = mx.zeros(batch_size, dtype=mx.int32)
    cache._rpad = mx.array(padding)
    cache._has_rpad = max(padding) > 0
    cache.offset = mx.array(lengths)
    cache._idx = max_length
    cache._right_padding = None
    cache.step = 256
    cache._py_offsets = list(lengths)
    return cache


def _rpad_update_and_fetch(self: Any, keys: mx.array, values: mx.array) -> Any:
    n_new = keys.shape[2]
    batch_size = keys.shape[0]
    py_offsets: list[int] | None = self._py_offsets

    # Fast path: no right padding (B=1 or equal-length sequences)
    if py_offsets is None or not self._has_rpad or len(py_offsets) != batch_size:
        prev = self._idx
        if self.keys is None or (prev + n_new) > self.keys.shape[2]:
            n_kv_heads = keys.shape[1]
            k_head_dim = keys.shape[3]
            v_head_dim = values.shape[3]
            n_steps = (self.step + prev + n_new - 1) // self.step
            k_shape = (batch_size, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (batch_size, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
        self.offset += n_new
        self._idx += n_new
        self.keys[..., prev : self._idx, :] = keys
        self.values[..., prev : self._idx, :] = values
        return self.keys[..., : self._idx, :], self.values[..., : self._idx, :]

    # Right-padded path: per-element write at each sequence's offset
    max_needed = max(py_offsets) + n_new
    if self.keys is None or max_needed > self.keys.shape[2]:
        n_kv_heads = keys.shape[1]
        k_head_dim = keys.shape[3]
        v_head_dim = values.shape[3]
        n_steps = (self.step + max_needed - 1) // self.step
        k_shape = (batch_size, n_kv_heads, n_steps * self.step, k_head_dim)
        v_shape = (batch_size, n_kv_heads, n_steps * self.step, v_head_dim)
        new_k = mx.zeros(k_shape, keys.dtype)
        new_v = mx.zeros(v_shape, values.dtype)
        if self.keys is not None:
            cur = self.keys.shape[2]
            new_k[..., :cur, :] = self.keys
            new_v[..., :cur, :] = self.values
        self.keys = new_k
        self.values = new_v

    for b in range(batch_size):
        pos = py_offsets[b]
        self.keys[b : b + 1, :, pos : pos + n_new, :] = keys[b : b + 1]
        self.values[b : b + 1, :, pos : pos + n_new, :] = values[b : b + 1]
        py_offsets[b] += n_new

    self.offset = self.offset + n_new
    self._idx = max(py_offsets)
    return self.keys[..., : self._idx, :], self.values[..., : self._idx, :]


def _rpad_make_mask(
    self: Any, n: int, return_array: bool = False, **kwargs: Any
) -> Any:
    if self._has_rpad:
        return create_causal_mask(
            n, offset=self._idx, right_padding=self._rpad, **kwargs
        )
    return create_causal_mask(
        n, offset=self._idx, left_padding=self.left_padding, **kwargs
    )


def _rpad_finalize(self: Any) -> None:
    self._right_padding = None
    # If cache was left-padded (from __init__ / initial prefill), convert to right-padded
    # by rolling real data to position 0.  Merged caches already have _has_rpad=True.
    if not self._has_rpad and self.keys is not None:
        lp = self.left_padding
        if bool(mx.any(lp > 0).item()):
            self.keys = self.keys[..., : self._idx, :]
            self.values = self.values[..., : self._idx, :]
            self.keys = dynamic_roll(self.keys, -lp[:, None], axis=2)
            self.values = dynamic_roll(self.values, -lp[:, None], axis=2)
            self._rpad = lp
            self.left_padding = mx.zeros_like(lp)
            self._has_rpad = True
            # Set per-element offsets so update_and_fetch writes at the correct
            # position for each sequence (not at the shared _idx).
            self._py_offsets = self.offset.tolist()


def _rpad_prepare(
    self: Any,
    *,
    left_padding: Any = None,
    lengths: Any = None,
    right_padding: Any = None,
) -> None:
    if left_padding is not None:
        if self.keys is not None:
            raise ValueError("Left padding can only be added to an empty BatchKVCache")
        lp = mx.array(left_padding)
        rpad = getattr(self, "_rpad", mx.zeros_like(lp))
        self._rpad = rpad + lp
        self._has_rpad = True
    if right_padding is not None and max(right_padding) > 0:
        self._right_padding = mx.array(right_padding)


def _rpad_filter(self: Any, batch_indices: Any) -> None:
    if self.keys is not None:
        self.keys = self.keys[batch_indices]
        self.values = self.values[batch_indices]
    self.left_padding = self.left_padding[batch_indices]
    self.offset = self.offset[batch_indices]
    if self._py_offsets is not None:
        # Rebuild _py_offsets from the filtered offset array to stay in sync
        self._py_offsets = self.offset.tolist()
        self._idx = max(self._py_offsets)
        # Recalculate _rpad relative to new _idx: each element's padding is
        # the gap between its offset and the max offset in the batch.
        self._rpad = mx.array([self._idx - o for o in self._py_offsets])
    else:
        self._rpad = self._rpad[batch_indices]
    self._has_rpad = bool(mx.any(self._rpad > 0).item())


def _rpad_size(self: Any) -> int:
    if self._py_offsets:
        return max(self._py_offsets)
    return int(mx.max(self.offset).item())


def _rpad_trim(self: Any, n: int) -> int:
    n = min(self._idx, n)
    self._idx -= n
    self.offset = self.offset - n
    if self._py_offsets:
        self._py_offsets = [o - n for o in self._py_offsets]
    return n


def _to_left_padded(cache: Any) -> None:
    """Convert a right-padded BatchKVCache to left-padded (right-justified) layout."""
    if not getattr(cache, "_has_rpad", False) or cache.keys is None:
        return
    rpad = cache._rpad
    if bool(mx.any(rpad > 0).item()):
        cache.keys = cache.keys[..., : cache._idx, :]
        cache.values = cache.values[..., : cache._idx, :]
        cache.keys = dynamic_roll(cache.keys, rpad[:, None], axis=2)
        cache.values = dynamic_roll(cache.values, rpad[:, None], axis=2)
        cache.left_padding = rpad
    cache._rpad = mx.zeros(cache.keys.shape[0], dtype=mx.int32)
    cache._has_rpad = False
    cache._py_offsets = None


_stock_extend = BatchKVCache.extend
_stock_extract = BatchKVCache.extract


def _rpad_extract(self: Any, idx: int) -> Any:
    """Extract a single sequence's cache, respecting right-padded layout."""
    if self._has_rpad and self._py_offsets is not None:
        from mlx_lm.models.cache import KVCache

        end = self._py_offsets[idx]
        cache = KVCache()
        cache.keys = mx.contiguous(self.keys[idx : idx + 1, :, :end, :])
        cache.values = mx.contiguous(self.values[idx : idx + 1, :, :end, :])
        cache.offset = end
        return cache
    return _stock_extract(self, idx)


def _rpad_extend(self: Any, other: Any) -> None:
    """Extend batch cache, converting right-padded caches to left-padded first."""
    _to_left_padded(self)
    _to_left_padded(other)
    _stock_extend(self, other)
    batch_size = self.keys.shape[0]
    self._rpad = mx.zeros(batch_size, dtype=mx.int32)
    self._has_rpad = False
    self._py_offsets = None


BatchKVCache.__init__ = _rpad_init
BatchKVCache.merge = _rpad_merge
BatchKVCache.update_and_fetch = _rpad_update_and_fetch
BatchKVCache.make_mask = _rpad_make_mask
BatchKVCache.finalize = _rpad_finalize
BatchKVCache.prepare = _rpad_prepare
BatchKVCache.filter = _rpad_filter
BatchKVCache.extend = _rpad_extend
BatchKVCache.extract = _rpad_extract
BatchKVCache.size = _rpad_size
BatchKVCache.trim = _rpad_trim
BatchKVCache.is_trimmable = lambda self: True
BatchKVCache.empty = lambda self: self.keys is None
_MIN_PREFIX_HIT_RATIO_TO_UPDATE = 0.5


def _stop_sequences(task_params: TextGenerationTaskParams) -> list[str]:
    if task_params.stop is None:
        return []
    if isinstance(task_params.stop, str):
        return [task_params.stop]
    return task_params.stop


@dataclass
class _EngineTask:
    uid: int
    task_params: TextGenerationTaskParams
    all_prompt_tokens: mx.array
    prefix_hit_length: int
    matched_index: int | None
    cache_snapshots: list[CacheSnapshot] | None
    on_generation_token: Callable[[], None] | None = None
    generated_text_parts: list[str] = field(default_factory=list)
    accumulated_text: str = ""
    completion_tokens: int = 0
    generation_start_time: float = 0.0
    in_thinking: bool = False
    reasoning_tokens: int = 0


@dataclass(eq=False)
class ExoBatchGenerator:
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None

    _mlx_gen: MlxBatchGenerator = field(init=False)
    _active_tasks: dict[int, _EngineTask] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._mlx_gen = MlxBatchGenerator(
            model=self.model,
            stop_tokens=set(eos_ids_from_tokenizer(self.tokenizer)),
            prefill_step_size=4096,
        )

    @property
    def has_work(self) -> bool:
        return (
            bool(self._active_tasks)
            or bool(self._mlx_gen.unprocessed_prompts)
            or self._mlx_gen.active_batch is not None
        )

    def submit(
        self,
        task_params: TextGenerationTaskParams,
        prompt: str,
        on_prefill_progress: Callable[[int, int], None] | None = None,
        distributed_prompt_progress_callback: Callable[[], None] | None = None,
        on_generation_token: Callable[[], None] | None = None,
    ) -> int:
        all_prompt_tokens = encode_prompt(self.tokenizer, prompt)
        all_prompt_tokens = fix_unmatched_think_end_tokens(
            all_prompt_tokens, self.tokenizer
        )

        is_bench = task_params.bench

        prefix_hit_length = 0
        matched_index: int | None = None
        prompt_tokens = all_prompt_tokens

        if self.kv_prefix_cache is not None and not is_bench:
            cache, remaining_tokens, matched_index = self.kv_prefix_cache.get_kv_cache(
                self.model, all_prompt_tokens
            )
            prefix_hit_length = len(all_prompt_tokens) - len(remaining_tokens)
            if prefix_hit_length > 0:
                logger.info(
                    f"KV cache hit: {prefix_hit_length}/{len(all_prompt_tokens)} tokens "
                    f"cached ({100 * prefix_hit_length / len(all_prompt_tokens):.1f}%)"
                )
                prompt_tokens = remaining_tokens
            else:
                cache = make_kv_cache(self.model)
        else:
            cache = make_kv_cache(self.model)

        seed = task_params.seed or 42
        mx.random.seed(seed)

        sampler = make_sampler(
            temp=task_params.temperature
            if task_params.temperature is not None
            else 0.7,
            top_p=task_params.top_p if task_params.top_p is not None else 1.0,
            top_k=task_params.top_k if task_params.top_k is not None else 0,
        )

        _prefill_tps, _prefill_tokens, cache_snapshots = prefill(
            self.model,
            self.tokenizer,
            sampler,
            prompt_tokens[:-1],
            cache,
            self.group,
            on_prefill_progress,
            distributed_prompt_progress_callback,
        )

        # We need to clamp rotating kv caches to max size so that mlx lm's _merge_caches behaves
        for c in cache:
            if (
                isinstance(c, RotatingKVCache)
                and c.keys is not None
                and c.values is not None
                and c.keys.shape[2] > c.max_size
            ):
                trim_size = c.keys.shape[2] - c.max_size
                c.keys = c._trim(trim_size, c.keys)
                c.values = c._trim(trim_size, c.values)
                c._idx = c.max_size

        last_tokens = prompt_tokens[-2:]

        logits_processors: list[Any] = []
        if is_bench:
            eos_ids = eos_ids_from_tokenizer(self.tokenizer)
            logits_processors = [ban_token_ids(eos_ids)]

        max_tokens = task_params.max_output_tokens or MAX_TOKENS

        uids = self._mlx_gen.insert(
            prompts=[last_tokens.tolist()],
            max_tokens=[max_tokens],
            caches=[list(cache)],
            samplers=[sampler],
            logits_processors=[logits_processors],
        )

        self._active_tasks[uids[0]] = _EngineTask(
            uid=uids[0],
            task_params=task_params,
            all_prompt_tokens=all_prompt_tokens,
            prefix_hit_length=prefix_hit_length,
            matched_index=matched_index,
            cache_snapshots=cache_snapshots or None,
            on_generation_token=on_generation_token,
            generation_start_time=time.perf_counter(),
        )

        return uids[0]

    def step(self) -> list[tuple[int, GenerationResponse]]:
        if not self.has_work:
            return []

        responses = self._mlx_gen.next()
        results: list[tuple[int, GenerationResponse]] = []

        for response in responses:
            if response.uid not in self._active_tasks:
                continue

            state = self._active_tasks[response.uid]
            if state.on_generation_token is not None:
                state.on_generation_token()
            text = (
                ""
                if response.finish_reason == "stop"
                else self.tokenizer.decode([response.token])
            )
            state.completion_tokens += 1
            state.generated_text_parts.append(text)
            state.accumulated_text += text

            think_start = self.tokenizer.think_start
            think_end = self.tokenizer.think_end
            if think_start is not None and text == think_start:
                state.in_thinking = True
            elif think_end is not None and text == think_end:
                state.in_thinking = False
            if state.in_thinking:
                state.reasoning_tokens += 1

            finish_reason: FinishReason | None = cast(
                FinishReason | None, response.finish_reason
            )
            task_params = state.task_params
            stop_sequences = _stop_sequences(task_params)
            max_stop_len = max((len(s) for s in stop_sequences), default=0)

            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in state.accumulated_text:
                        stop_index = state.accumulated_text.find(stop_seq)
                        text_before_stop = state.accumulated_text[:stop_index]
                        chunk_start = len(state.accumulated_text) - len(text)
                        text = text_before_stop[chunk_start:]
                        finish_reason = "stop"
                        break

            is_done = finish_reason is not None

            logprob: float | None = None
            top_logprobs: list[TopLogprobItem] | None = None
            if task_params.logprobs:
                logprob, top_logprobs = extract_top_logprobs(
                    logprobs=response.logprobs,
                    tokenizer=self.tokenizer,
                    top_logprobs=task_params.top_logprobs or DEFAULT_TOP_LOGPROBS,
                    selected_token=response.token,
                )

            stats: GenerationStats | None = None
            usage: Usage | None = None
            if is_done:
                generation_elapsed = time.perf_counter() - state.generation_start_time
                generation_tps = (
                    state.completion_tokens / generation_elapsed
                    if generation_elapsed > 0
                    else 0.0
                )
                mlx_stats = self._mlx_gen.stats()
                stats = GenerationStats(
                    prompt_tps=float(mlx_stats.prompt_tps)
                    if mlx_stats.prompt_time > 0
                    else 0.0,
                    generation_tps=float(generation_tps),
                    prompt_tokens=len(state.all_prompt_tokens),
                    generation_tokens=state.completion_tokens,
                    peak_memory_usage=Memory.from_gb(mx.get_peak_memory() / 1e9),
                )
                total_prompt_tokens = len(state.all_prompt_tokens)
                usage = Usage(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=state.completion_tokens,
                    total_tokens=total_prompt_tokens + state.completion_tokens,
                    prompt_tokens_details=PromptTokensDetails(
                        cached_tokens=state.prefix_hit_length
                    ),
                    completion_tokens_details=CompletionTokensDetails(
                        reasoning_tokens=state.reasoning_tokens
                    ),
                )

            results.append(
                (
                    response.uid,
                    GenerationResponse(
                        text=text,
                        token=response.token,
                        logprob=logprob,
                        top_logprobs=top_logprobs,
                        finish_reason=finish_reason,
                        stats=stats,
                        usage=usage,
                    ),
                )
            )

            if is_done:
                self._save_prefix_cache(state, cast(KVCacheType, response.prompt_cache))
                del self._active_tasks[response.uid]
            elif max_stop_len > 0 and len(state.accumulated_text) > max_stop_len:
                state.accumulated_text = state.accumulated_text[-max_stop_len:]

        return results

    def cancel(self, uids: list[int]) -> None:
        self._mlx_gen.remove(uids)
        for uid in uids:
            self._active_tasks.pop(uid, None)

    def close(self) -> None:
        self._mlx_gen.close()

    def _save_prefix_cache(self, state: _EngineTask, prompt_cache: KVCacheType) -> None:
        if self.kv_prefix_cache is None or state.task_params.bench:
            return

        try:
            generated_tokens_array = mx.array(
                self.tokenizer.encode(
                    "".join(state.generated_text_parts), add_special_tokens=False
                )
            )
            full_prompt_tokens = mx.concatenate(
                [state.all_prompt_tokens, generated_tokens_array]
            )
            hit_ratio = (
                state.prefix_hit_length / len(state.all_prompt_tokens)
                if len(state.all_prompt_tokens) > 0
                else 0.0
            )
            if (
                state.matched_index is not None
                and hit_ratio >= _MIN_PREFIX_HIT_RATIO_TO_UPDATE
            ):
                self.kv_prefix_cache.update_kv_cache(
                    state.matched_index,
                    full_prompt_tokens,
                    prompt_cache,
                    state.cache_snapshots,
                    restore_pos=state.prefix_hit_length,
                )
            else:
                self.kv_prefix_cache.add_kv_cache(
                    full_prompt_tokens, prompt_cache, state.cache_snapshots
                )
        except Exception:
            logger.warning("Failed to save prefix cache", exc_info=True)
