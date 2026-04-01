import contextlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, cast

import mlx.core as mx
from mlx_lm.generate import (
    BatchGenerator as MlxBatchGenerator,
)
from mlx_lm.generate import (
    generation_stream,
    stream_generate,
)
from mlx_lm.models.cache import RotatingKVCache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import StreamingDetokenizer, TokenizerWrapper

from exo.api.types import (
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
    has_non_kv_caches,
    make_kv_cache,
)
from exo.worker.engines.mlx.constants import DEFAULT_TOP_LOGPROBS, KV_BITS, KV_GROUP_SIZE, MAX_TOKENS
from exo.worker.engines.mlx.generator.generate import (
    ban_token_ids,
    eos_ids_from_tokenizer,
    extract_top_logprobs,
    patch_embed_tokens,
    prefill,
)
from exo.worker.engines.mlx.utils_mlx import (
    fix_unmatched_think_end_tokens,
    system_prompt_token_count,
)
from exo.worker.engines.mlx.vision import (
    MediaRegion,
    VisionProcessor,
    VisionResult,
    prepare_vision,
)
from exo.worker.runner.bootstrap import logger

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
    detokenizer: StreamingDetokenizer
    on_generation_token: Callable[[], None] | None = None
    generated_text_parts: list[str] = field(default_factory=list)
    potential_stop_sequence_text: str = ""
    completion_tokens: int = 0
    generation_start_time: float = 0.0
    generation_time_at_start: float = 0.0
    in_thinking: bool = False
    reasoning_tokens: int = 0
    prefill_tps: float = 0.0
    media_regions: list[MediaRegion] = field(default_factory=list)


@dataclass(eq=False)
class ExoBatchGenerator:
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    vision_processor: VisionProcessor | None = None

    _mlx_gen: MlxBatchGenerator = field(init=False)
    _active_tasks: dict[int, _EngineTask] = field(default_factory=dict, init=False)
    _pp_spec_active: bool = field(init=False, default=False)
    _pp_spec_gen: Generator[tuple[int, mx.array], None, None] | None = field(init=False, default=None)
    _pp_spec_uid: int | None = field(init=False, default=None)
    _pp_spec_eos: set[int] = field(init=False, default_factory=set)
    _uid_counter: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._mlx_gen = MlxBatchGenerator(
            model=self.model,
            stop_tokens=set(eos_ids_from_tokenizer(self.tokenizer)),
            prefill_step_size=4096,
        )
        self._mlx_gen._needs_topk = False  # pyright: ignore[reportAttributeAccessIssue]
        self._pp_spec_eos = set(eos_ids_from_tokenizer(self.tokenizer))

        # Enable PP speculation if draft model is configured and we're in PP mode
        draft_path = os.environ.get("EXO_PP_DRAFT_MODEL", "")
        if draft_path and self.group is not None and self.group.size() > 1:
            try:
                from ..pp_speculation import get_pipeline_info
                if get_pipeline_info(self.model) is not None:
                    self._pp_spec_active = True
                    logger.info("PP speculation enabled in BatchGenerator")
            except Exception:
                pass

    @property
    def has_work(self) -> bool:
        return (
            bool(self._active_tasks)
            or bool(self._mlx_gen.unprocessed_prompts)
            or self._mlx_gen.active_batch is not None
            or self._pp_spec_gen is not None
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

        vision: VisionResult | None = None
        media_regions: list[MediaRegion] = []

        if self.vision_processor is not None:
            try:
                vision = prepare_vision(
                    images=task_params.images,
                    chat_template_messages=task_params.chat_template_messages,
                    vision_processor=self.vision_processor,
                    tokenizer=self.tokenizer,
                    model=self.model,
                    model_id=task_params.model,
                    task_params=task_params,
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Vision processing failed, falling back to text-only"
                )

        if vision is not None:
            all_prompt_tokens = vision.prompt_tokens
            media_regions = vision.media_regions

        is_bench = task_params.bench

        prefix_hit_length = 0
        matched_index: int | None = None
        prompt_tokens = all_prompt_tokens

        if self.kv_prefix_cache is not None and not is_bench:
            cache, remaining_tokens, matched_index = self.kv_prefix_cache.get_kv_cache(
                self.model, all_prompt_tokens, media_regions=media_regions
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

        seed = task_params.seed if task_params.seed is not None else 42
        mx.random.seed(seed)

        sampler = make_sampler(
            temp=task_params.temperature
            if task_params.temperature is not None
            else 0.7,
            top_p=task_params.top_p if task_params.top_p is not None else 1.0,
            min_p=task_params.min_p if task_params.min_p is not None else 0.05,
            top_k=task_params.top_k if task_params.top_k is not None else 0,
        )

        vision_ctx = (
            patch_embed_tokens(
                self.model, vision.embeddings, prefix_hit_length, len(prompt_tokens) - 1
            )
            if vision is not None
            else contextlib.nullcontext()
        )
        with vision_ctx:
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

        if not is_bench:
            min_prefix_hit_length = max(
                1000, system_prompt_token_count(task_params, self.tokenizer)
            )
            self._save_prefix_cache(
                all_prompt_tokens,
                list(cache),
                cache_snapshots,
                prefix_hit_length,
                matched_index,
                min_prefix_hit_length,
                media_regions,
            )

        last_tokens = prompt_tokens[-2:]

        logits_processors: list[Callable[[mx.array, mx.array], mx.array]] = (
            make_logits_processors(
                repetition_penalty=task_params.repetition_penalty,
                repetition_context_size=task_params.repetition_context_size,
            )
        )
        if is_bench:
            # Only sample length eos tokens
            eos_ids = eos_ids_from_tokenizer(self.tokenizer)
            logits_processors = [ban_token_ids(eos_ids)] + logits_processors

        max_tokens = task_params.max_output_tokens or MAX_TOKENS

        if self._pp_spec_active:
            return self._submit_pp_spec(
                task_params, all_prompt_tokens, prefix_hit_length, matched_index,
                cache_snapshots, cache, last_tokens, sampler, logits_processors,
                max_tokens, on_generation_token, _prefill_tps,
            )

        uids = self._mlx_gen.insert(
            prompts=[last_tokens.tolist()],
            max_tokens=[max_tokens],
            caches=[list(cache)],
            samplers=[sampler],
            logits_processors=[logits_processors],
        )

        assert len(uids) == 1

        uid = uids[0]

        self._active_tasks[uid] = _EngineTask(
            uid=uid,
            task_params=task_params,
            all_prompt_tokens=all_prompt_tokens,
            prefix_hit_length=prefix_hit_length,
            matched_index=matched_index,
            cache_snapshots=cache_snapshots or None,
            detokenizer=self.tokenizer.detokenizer,
            on_generation_token=on_generation_token,
            generation_start_time=time.perf_counter(),
            prefill_tps=_prefill_tps,
            generation_time_at_start=self._mlx_gen._stats.generation_time,
            media_regions=media_regions,
        )

        return uid

    def _submit_pp_spec(
        self,
        task_params: TextGenerationTaskParams,
        all_prompt_tokens: mx.array,
        prefix_hit_length: int,
        matched_index: int | None,
        cache_snapshots: list[CacheSnapshot] | None,
        cache: list[Any],
        last_tokens: mx.array,
        sampler: Callable,
        logits_processors: list[Callable],
        max_tokens: int,
        on_generation_token: Callable[[], None] | None,
        prefill_tps: float,
    ) -> int:
        """Set up PP speculative decode for this task."""
        from ..pp_speculation import (
            get_pipeline_info,
            pp_speculative_decode_loop,
            _install_spec_layers,
        )

        pp_info = get_pipeline_info(self.model)
        assert pp_info is not None
        pp_rank, pp_world_size, pp_group = pp_info

        inner = getattr(self.model, "language_model", self.model)
        _install_spec_layers(inner)

        _pp_draft = getattr(self.model, "_pp_draft_model", None)
        _pp_draft_cache = getattr(self.model, "_pp_draft_cache", None)

        # Prefill draft cache with tail of prompt (rank 0 only)
        # The draft model uses a RotatingKVCache, so only recent tokens matter.
        if pp_rank == 0 and _pp_draft is not None:
            _draft_kv_window = int(os.environ.get("EXO_DRAFT_KV_WINDOW", "4096"))
            _draft_tokens = all_prompt_tokens[-_draft_kv_window:]
            _draft_chunk = 512
            for i in range(0, len(_draft_tokens), _draft_chunk):
                _pp_draft(_draft_tokens[i:i + _draft_chunk][None], cache=_pp_draft_cache)
                mx.eval([c.state if hasattr(c, 'state') else c for c in _pp_draft_cache])
            mx.clear_cache()
            logger.info(f"Draft model prefilled with {len(_draft_tokens)} tokens (of {len(all_prompt_tokens)} total)")

        # First token via standard PP
        _first_gen = stream_generate(
            model=self.model, tokenizer=self.tokenizer, prompt=last_tokens,
            max_tokens=1, sampler=sampler, logits_processors=logits_processors,
            prompt_cache=cache, prefill_step_size=1,
            kv_group_size=KV_GROUP_SIZE, kv_bits=KV_BITS,
        )
        _first_out = next(_first_gen)
        first_y = mx.array([_first_out.token])
        mx.eval(first_y)

        logger.info(f"PP speculation active: rank={pp_rank}")

        # Create the spec decode generator
        self._pp_spec_gen = pp_speculative_decode_loop(
            model=self.model, draft_model=_pp_draft,
            prompt_cache=cache, draft_cache=_pp_draft_cache,
            sampler=sampler, logits_processors=logits_processors,
            first_y=first_y, first_logprobs=mx.zeros(1),
            max_tokens=max_tokens - 1,
            pp_rank=pp_rank, pp_world_size=pp_world_size,
            pp_group=pp_group,
        )

        self._uid_counter += 1
        uid = self._uid_counter
        self._pp_spec_uid = uid

        # Store first token to yield on first step()
        self._pp_first_token = _first_out.token

        self._active_tasks[uid] = _EngineTask(
            uid=uid,
            task_params=task_params,
            all_prompt_tokens=all_prompt_tokens,
            prefix_hit_length=prefix_hit_length,
            matched_index=matched_index,
            cache_snapshots=cache_snapshots or None,
            detokenizer=self.tokenizer.detokenizer,
            on_generation_token=on_generation_token,
            generation_start_time=time.perf_counter(),
            prefill_tps=prefill_tps,
        )

        return uid

    def _step_pp_spec(self) -> list[MlxBatchGenerator.Response]:
        """Get next token from PP speculative decode loop."""
        uid = self._pp_spec_uid
        assert uid is not None

        # Yield the first token if we haven't yet
        if hasattr(self, '_pp_first_token'):
            tok = self._pp_first_token
            del self._pp_first_token
            finish = "stop" if tok in self._pp_spec_eos else None
            return [MlxBatchGenerator.Response(
                uid=uid, token=tok, logprobs=mx.zeros(1),
                finish_reason=finish, prompt_cache=lambda: [],
            )]

        assert self._pp_spec_gen is not None
        try:
            tok_id, lp = next(self._pp_spec_gen)
            finish = "stop" if tok_id in self._pp_spec_eos else None
            return [MlxBatchGenerator.Response(
                uid=uid, token=tok_id, logprobs=lp,
                finish_reason=finish, prompt_cache=lambda: [],
            )]
        except StopIteration:
            # max_tokens reached
            self._pp_spec_gen = None
            self._pp_spec_uid = None
            return [MlxBatchGenerator.Response(
                uid=uid, token=0, logprobs=mx.zeros(1),
                finish_reason="length", prompt_cache=lambda: [],
            )]

    def step(self) -> list[tuple[int, GenerationResponse]]:
        if not self.has_work:
            return []

        # Use PP speculation decode if active
        if self._pp_spec_gen is not None:
            _step_tic = time.perf_counter()
            responses = self._step_pp_spec()
            _next_elapsed = time.perf_counter() - _step_tic
        else:
            self._mlx_gen._needs_topk = any(  # pyright: ignore[reportAttributeAccessIssue]
                t.task_params.logprobs for t in self._active_tasks.values()
            )
            _step_tic = time.perf_counter()
            responses = self._mlx_gen.next()
            _next_elapsed = time.perf_counter() - _step_tic

        results: list[tuple[int, GenerationResponse]] = []

        for response in responses:
            if response.uid not in self._active_tasks:
                logger.warning(
                    f"response uid {response.uid} was not found - should be active"
                )
                continue

            state = self._active_tasks[response.uid]
            if state.on_generation_token is not None:
                state.on_generation_token()
            if response.finish_reason != "stop":
                state.detokenizer.add_token(response.token)
            if response.finish_reason is not None:
                state.detokenizer.finalize()
            text = state.detokenizer.last_segment
            state.completion_tokens += 1
            state.generated_text_parts.append(text)
            state.potential_stop_sequence_text += text

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
                    if stop_seq in state.potential_stop_sequence_text:
                        stop_index = state.potential_stop_sequence_text.find(stop_seq)
                        text_before_stop = state.potential_stop_sequence_text[
                            :stop_index
                        ]
                        chunk_start = len(state.potential_stop_sequence_text) - len(
                            text
                        )
                        text = text_before_stop[chunk_start:]
                        finish_reason = "stop"
                        break

            is_done = finish_reason is not None

            logprob: float | None = None
            top_logprobs: list[TopLogprobItem] | None = None
            if task_params.logprobs:
                with mx.stream(generation_stream):
                    logprob, top_logprobs = extract_top_logprobs(
                        logprobs=response.logprobs,
                        tokenizer=self.tokenizer,
                        top_logprobs=task_params.top_logprobs or DEFAULT_TOP_LOGPROBS,
                        selected_token=response.token,
                        precomputed_indices=getattr(response, "_topk_indices", None),
                        precomputed_values=getattr(response, "_topk_values", None),
                        precomputed_selected=getattr(
                            response, "_selected_logprob", None
                        ),
                    )

            stats: GenerationStats | None = None
            usage: Usage | None = None
            if is_done:
                if self._pp_spec_gen is not None or self._pp_spec_uid is not None:
                    gen_elapsed = time.perf_counter() - state.generation_start_time
                    generation_tps = (
                        state.completion_tokens / gen_elapsed
                        if gen_elapsed > 0
                        else 0.0
                    )
                    # Clean up spec state
                    self._pp_spec_gen = None
                    self._pp_spec_uid = None
                else:
                    gen_time_delta = (
                        self._mlx_gen._stats.generation_time
                        - state.generation_time_at_start
                    )
                    generation_tps = (
                        state.completion_tokens / gen_time_delta
                        if gen_time_delta > 0
                        else 0.0
                    )

                stats = GenerationStats(
                    prompt_tps=state.prefill_tps,
                    generation_tps=generation_tps,
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
                del self._active_tasks[response.uid]
            elif (
                max_stop_len > 0
                and len(state.potential_stop_sequence_text) > max_stop_len
            ):
                state.potential_stop_sequence_text = state.potential_stop_sequence_text[
                    -max_stop_len:
                ]

        _step_elapsed = time.perf_counter() - _step_tic
        _overhead = _step_elapsed - _next_elapsed
        if self._mlx_gen._next_count % 64 == 0 and responses:
            logger.debug(
                f"step overhead: {_overhead * 1000:.2f}ms (next={_next_elapsed * 1000:.2f}ms total={_step_elapsed * 1000:.2f}ms)"
            )

        return results

    def cancel(self, uids: list[int]) -> None:
        self._mlx_gen.remove(uids)
        for uid in uids:
            self._active_tasks.pop(uid, None)

    def close(self) -> None:
        self._mlx_gen.close()

    def _save_prefix_cache(
        self,
        all_prompt_tokens: mx.array,
        cache: KVCacheType,
        cache_snapshots: list[CacheSnapshot] | None,
        prefix_hit_length: int,
        matched_index: int | None,
        min_prefix_hit_length: int = 1000,
        media_regions: list[MediaRegion] | None = None,
    ) -> None:
        if self.kv_prefix_cache is None:
            return

        try:
            hit_ratio = (
                prefix_hit_length / len(all_prompt_tokens)
                if len(all_prompt_tokens) > 0
                else 0.0
            )
            if matched_index is not None and (
                prefix_hit_length >= min_prefix_hit_length
                and hit_ratio >= _MIN_PREFIX_HIT_RATIO_TO_UPDATE
            ):
                self.kv_prefix_cache.update_kv_cache(
                    matched_index,
                    all_prompt_tokens,
                    cache,
                    cache_snapshots,
                    restore_pos=prefix_hit_length,
                    media_regions=media_regions,
                )
            else:
                self.kv_prefix_cache.add_kv_cache(
                    all_prompt_tokens,
                    cache,
                    cache_snapshots,
                    media_regions=media_regions,
                )
        except Exception:
            logger.warning("Failed to save prefix cache", exc_info=True)
