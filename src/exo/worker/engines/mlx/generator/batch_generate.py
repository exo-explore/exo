import contextlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Literal, cast, final

import mlx.core as mx
from mlx_lm.generate import (
    BatchGenerator as MlxBatchGenerator,
)
from mlx_lm.generate import (
    PromptProcessingBatch,
    generation_stream,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache
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
from exo.shared.constants import EXO_NONBLOCKING_PREFILL, EXO_PREFILL_STEP_TOKENS
from exo.shared.types.memory import Memory
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    cache_length,
    encode_prompt,
    get_prefix_length,
    make_kv_cache,
)
from exo.worker.engines.mlx.constants import DEFAULT_TOP_LOGPROBS, MAX_TOKENS
from exo.worker.engines.mlx.generator.generate import (
    PrefillCancelled,
    ban_token_ids,
    eos_ids_from_tokenizer,
    extract_top_logprobs,
    patch_embed_tokens,
    prefill,
)
from exo.worker.engines.mlx.generator.remote_prefill import remote_prefill
from exo.worker.engines.mlx.models.registry import keeps_think_tags, model_type_of
from exo.worker.engines.mlx.patches.opt_batch_gen import (
    set_needs_topk,
    take_ready_topk,
)
from exo.worker.engines.mlx.types import KVCacheType, Model
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
from exo.worker.runner.context_length import check_context_length

_MIN_PREFIX_HIT_RATIO_TO_UPDATE = 0.5
REMOTE_PREFILL_MIN_TOKENS = 1000
# The step's prompt-token budget is split between the prompts in it; each still gets at
# least this many, so a full prompt batch keeps making progress.
MIN_PREFILL_TOKENS_PER_PROMPT = 32


def _stop_sequences(task_params: TextGenerationTaskParams) -> list[str]:
    if task_params.stop is None:
        return []
    if isinstance(task_params.stop, str):
        return [task_params.stop]
    return task_params.stop


def can_prefill_in_batch(
    cache: KVCacheType,
    uncached_tokens: int,
    uses_remote_prefill: bool,
    has_vision: bool,
    group_size: int,
) -> bool:
    """Whether a local prefill can be left to the running batch, a chunk per step.

    - Remote prefill (and its local fallback) and vision prompts keep the synchronous path.
    - Multi-node instances do too: their ranks must stay in lock step.
    - Only a plain KVCache qualifies. Rotating, SSM, CacheList, DeepSeek V4 and quantised
      caches need the snapshot and trim handling that prefill() and the prefix cache do,
      which the batch does not.
    - The batch prefills prompt[:-2] as its own segment, so it must not be empty.
    """
    return (
        not uses_remote_prefill
        and not has_vision
        and group_size == 1
        and all(type(entry) is KVCache for entry in cache)
        and uncached_tokens > 2
    )


def without_prompt_context(
    processor: Callable[[mx.array, mx.array], mx.array], skipped_tokens: int
) -> Callable[[mx.array, mx.array], mx.array]:
    """A logits processor that does not see the first skipped_tokens of the context.

    mlx_lm passes a processor every token the batch has processed for the sequence. On
    the synchronous path that is the prompt's last two tokens and the reply; when the
    batch prefills the prompt it is the whole uncached prompt as well, so a presence or
    repetition penalty would also count the prompt. Skipping it keeps the penalties as
    the synchronous path applies them.
    """

    def processor_without_prompt(tokens: mx.array, logits: mx.array) -> mx.array:
        return processor(tokens[skipped_tokens:], logits)

    return processor_without_prompt


def prefill_tokens_per_prompt(step_tokens: int, prompts_in_step: int) -> int:
    """How many tokens of each prompt the next step prefills.

    The batch right-pads its prompts to the longest chunk, so a step costs about
    prompts x chunk. Sharing one budget keeps two long prompts from doubling the stall.
    """
    return max(MIN_PREFILL_TOKENS_PER_PROMPT, step_tokens // max(1, prompts_in_step))


def matched_index_if_unchanged(
    cached_prompts: list[mx.array],
    all_prompt_tokens: mx.array,
    prefix_hit_length: int,
    matched_index: int | None,
) -> int | None:
    """The prefix-cache entry matched at submit, if it still holds the matched prefix.

    A prompt prefilled in the batch is saved to the prefix cache steps after submit()
    looked it up. Other requests can save in between, and eviction removes entries by
    position, so the index may now name another prompt; updating through it would
    overwrite that prompt's entry. If the entry has gone or changed, save a new one.
    """
    if matched_index is None or matched_index >= len(cached_prompts):
        return None
    if (
        get_prefix_length(all_prompt_tokens, cached_prompts[matched_index])
        < prefix_hit_length
    ):
        return None
    return matched_index


@dataclass
class _EngineTask:
    uid: int
    task_params: TextGenerationTaskParams
    all_prompt_tokens: mx.array
    prefix_hit_length: int
    matched_index: int | None
    detokenizer: StreamingDetokenizer
    on_generation_token: Callable[[], None] | None = None
    generated_text_parts: list[str] = field(default_factory=list)
    potential_stop_sequence_text: str = ""
    completion_tokens: int = 0
    generation_start_time: float = 0.0
    prefill_tps: float = 0.0
    prefix_cache_hit: Literal["none", "partial", "exact"] = "none"
    media_regions: list[MediaRegion] = field(default_factory=list)
    first_gen_token_time: float | None = None
    last_gen_token_time: float | None = None


@final
@dataclass(frozen=True)
class _PrefillInBatch:
    """A local prompt that the running batch prefills a chunk per step.

    It goes into the batch as two segments: prompt[:-2], then the last two tokens, which
    are what the synchronous path inserts after prefill(). When the first segment ends,
    the cache holds what prefill() would have left in it, and is saved to the prefix cache.
    """

    all_prompt_tokens: mx.array
    first_segment_tokens: int
    # Uncached tokens - 1: the total prefill() reports progress against
    progress_total: int
    prefix_hit_length: int
    matched_index: int | None
    media_regions: list[MediaRegion]
    # None when this request is not saved to the prefix cache (a bench run without it)
    min_prefix_hit_length: int | None
    submitted_at: float
    on_prefill_progress: Callable[[int, int], None] | None
    distributed_prompt_progress_callback: Callable[[], None] | None


@dataclass(eq=False)
class ExoBatchGenerator:
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    vision_processor: VisionProcessor | None = None
    # The longest context the model takes; 0 if unknown
    context_length: int = 0

    _mlx_gen: MlxBatchGenerator = field(init=False)
    _active_tasks: dict[int, _EngineTask] = field(default_factory=dict, init=False)
    # Prompts still being prefilled by the batch (EXO_NONBLOCKING_PREFILL), by uid
    _prefills_in_batch: dict[int, _PrefillInBatch] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        self._mlx_gen = MlxBatchGenerator(
            model=self.model,
            stop_tokens=[[t] for t in eos_ids_from_tokenizer(self.tokenizer)],
            prefill_step_size=4096,
        )
        self._step_count = 0
        # Cancelled sequences left to finish at their next step
        self._finishing: set[int] = set()

    @property
    def has_work(self) -> bool:
        return (
            bool(self._active_tasks)
            or bool(self._mlx_gen._unprocessed_sequences)
            or len(self._mlx_gen._prompt_batch) > 0
            or len(self._mlx_gen._generation_batch) > 0
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
        if not keeps_think_tags(task_params, model_type_of(self.model)):
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
        # Before any work on a request the model can't take (and any collective)
        check_context_length(
            len(all_prompt_tokens), task_params.max_output_tokens, self.context_length
        )

        is_bench = task_params.bench

        prefix_hit_length = 0
        matched_index: int | None = None
        is_exact_hit = False
        prompt_tokens = all_prompt_tokens

        if self.kv_prefix_cache is not None and (
            not is_bench or task_params.use_prefix_cache
        ):
            cache, remaining_tokens, matched_index, is_exact_hit = (
                self.kv_prefix_cache.get_kv_cache(
                    self.model, all_prompt_tokens, media_regions=media_regions
                )
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
                self.model,
                vision.embeddings,
                prefix_hit_length,
                len(prompt_tokens) - 1,
                image_token_id=vision.image_token_id,
            )
            if vision is not None
            else contextlib.nullcontext()
        )
        uncached_count = len(prompt_tokens)
        use_remote = (
            uncached_count > REMOTE_PREFILL_MIN_TOKENS
            and task_params.prefill_endpoint is not None
        )
        # A synchronous prefill stalls every stream on this instance until it finishes.
        # Where it can, leave it to the batch, which interleaves it with decode steps.
        prefill_in_batch = EXO_NONBLOCKING_PREFILL and can_prefill_in_batch(
            cache,
            uncached_tokens=uncached_count,
            uses_remote_prefill=use_remote,
            has_vision=vision is not None,
            group_size=self.group.size() if self.group is not None else 1,
        )
        min_prefix_hit_length: int | None = None

        _prefill_tps: float = 0.0
        _prefill_tokens: int = 0
        cache_snapshots: list[CacheSnapshot] = []
        remote_prefilled = False
        with vision_ctx:
            if use_remote and task_params.prefill_endpoint is not None:
                try:
                    _prefill_tps, _prefill_tokens, cache_snapshots = remote_prefill(
                        prompt_tokens[:-1],
                        cache,
                        on_prefill_progress,
                        endpoint=task_params.prefill_endpoint,
                        request_id=str(uuid.uuid4()),
                        model_id=str(task_params.model),
                        start_pos=prefix_hit_length,
                    )
                    remote_prefilled = True
                except Exception:
                    logger.opt(exception=True).warning(
                        "Remote prefill failed, falling back to local prefill"
                    )

            if not remote_prefilled and not prefill_in_batch:
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

        prefix_cache_hit: Literal["none", "partial", "exact"] = "none"
        if matched_index is not None and prefix_hit_length > 0:
            assert self.kv_prefix_cache is not None
            if is_exact_hit:
                prefix_cache_hit = "exact"
                _prefill_tps = self.kv_prefix_cache.prefill_tps[matched_index]
            else:
                prefix_cache_hit = "partial"

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

        if not is_bench or task_params.use_prefix_cache:
            min_prefix_hit_length = max(
                1000, system_prompt_token_count(task_params, self.tokenizer)
            )
            # A prompt prefilled in the batch is saved when step() sees it filled
            if not prefill_in_batch:
                self._save_prefix_cache(
                    all_prompt_tokens,
                    list(cache),
                    cache_snapshots,
                    prefix_hit_length,
                    matched_index,
                    min_prefix_hit_length,
                    media_regions,
                    prefill_tps=_prefill_tps,
                )

        last_tokens = prompt_tokens[-2:]

        logits_processors: list[Callable[[mx.array, mx.array], mx.array]] = (
            make_logits_processors(
                repetition_penalty=task_params.repetition_penalty,
                repetition_context_size=task_params.repetition_context_size
                if task_params.repetition_context_size is not None
                else 20,
                presence_penalty=task_params.presence_penalty,
                frequency_penalty=task_params.frequency_penalty,
            )
        )
        if is_bench:
            # Only sample length eos tokens
            eos_ids = eos_ids_from_tokenizer(self.tokenizer)
            logits_processors = [ban_token_ids(eos_ids)] + logits_processors
        if prefill_in_batch:
            # The batch's token context ends with the same tokens as the synchronous
            # path's, after len(prompt_tokens) - 2 more (see without_prompt_context)
            logits_processors = [
                without_prompt_context(processor, len(prompt_tokens) - 2)
                for processor in logits_processors
            ]

        max_tokens = task_params.max_output_tokens or MAX_TOKENS

        if prefill_in_batch:
            # The batch prefills prompt[:-2] itself. The last two tokens are a segment of
            # their own, so the cache is in the synchronous path's state once it ends.
            uids = self._mlx_gen.insert_segments(
                [
                    [
                        cast(list[int], prompt_tokens[:-2].tolist()),
                        cast(list[int], last_tokens.tolist()),
                    ]
                ],
                max_tokens=[max_tokens],
                caches=[list(cache)],
                samplers=[sampler],
                logits_processors=[logits_processors],
            )
        else:
            uids = self._mlx_gen.insert(
                prompts=[cast(list[int], last_tokens.tolist())],
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
            detokenizer=self.tokenizer.detokenizer,
            on_generation_token=on_generation_token,
            generation_start_time=time.perf_counter(),
            prefill_tps=_prefill_tps,
            prefix_cache_hit=prefix_cache_hit,
            media_regions=media_regions,
        )

        if prefill_in_batch:
            self._prefills_in_batch[uid] = _PrefillInBatch(
                all_prompt_tokens=all_prompt_tokens,
                first_segment_tokens=len(prompt_tokens) - 2,
                progress_total=len(prompt_tokens) - 1,
                prefix_hit_length=prefix_hit_length,
                matched_index=matched_index,
                media_regions=media_regions,
                min_prefix_hit_length=min_prefix_hit_length,
                submitted_at=time.perf_counter(),
                on_prefill_progress=on_prefill_progress,
                distributed_prompt_progress_callback=distributed_prompt_progress_callback,
            )
            if on_prefill_progress is not None:
                on_prefill_progress(0, len(prompt_tokens) - 1)

        return uid

    def step(self) -> list[tuple[int, GenerationResponse]]:
        if not self.has_work:
            return []

        if EXO_NONBLOCKING_PREFILL:
            self._share_prefill_budget()

        gb = self._mlx_gen._generation_batch
        set_needs_topk(
            gb,
            any(t.task_params.logprobs for t in self._active_tasks.values()),
        )
        _step_tic = time.perf_counter()
        prompt_responses, responses = self._mlx_gen.next()
        _next_elapsed = time.perf_counter() - _step_tic

        topk = take_ready_topk(gb)
        if self._prefills_in_batch:
            self._advance_prefills_in_batch(prompt_responses)

        results: list[tuple[int, GenerationResponse]] = []

        for response in responses:
            if response.uid in self._finishing:
                if response.finish_reason is not None:
                    self._finishing.discard(response.uid)
                continue
            if response.uid not in self._active_tasks:
                logger.warning(
                    f"response uid {response.uid} was not found - should be active"
                )
                continue

            state = self._active_tasks[response.uid]
            now = time.perf_counter()
            if state.first_gen_token_time is None:
                state.first_gen_token_time = now
            state.last_gen_token_time = now
            if state.on_generation_token is not None:
                state.on_generation_token()
            if response.finish_reason != "stop":
                state.detokenizer.add_token(response.token)
            if response.finish_reason is not None:
                state.detokenizer.finalize()
            text = state.detokenizer.last_segment
            state.completion_tokens += 1
            if state.task_params.bench:
                delta = now - state.first_gen_token_time
                logger.debug(
                    f"[bench] uid={response.uid} tok#{state.completion_tokens} {text!r} t={delta:.4f}s"
                )
            state.generated_text_parts.append(text)
            state.potential_stop_sequence_text += text

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
                precomputed = topk.for_uid(response.uid)
                precomputed_indices, precomputed_values, precomputed_selected = (
                    precomputed if precomputed is not None else (None, None, None)
                )
                with mx.stream(generation_stream):
                    logprob, top_logprobs = extract_top_logprobs(
                        logprobs=response.logprobs,
                        tokenizer=self.tokenizer,
                        top_logprobs=task_params.top_logprobs or DEFAULT_TOP_LOGPROBS,
                        selected_token=response.token,
                        precomputed_indices=precomputed_indices,
                        precomputed_values=precomputed_values,
                        precomputed_selected=precomputed_selected,
                    )

            stats: GenerationStats | None = None
            usage: Usage | None = None
            if is_done:
                if state.completion_tokens > 1:
                    gen_span = state.last_gen_token_time - state.first_gen_token_time
                    generation_tps = (
                        (state.completion_tokens - 1) / gen_span
                        if gen_span > 0
                        else 0.0
                    )
                else:
                    generation_tps = 0.0

                stats = GenerationStats(
                    prompt_tps=state.prefill_tps,
                    generation_tps=generation_tps,
                    prompt_tokens=len(state.all_prompt_tokens),
                    generation_tokens=state.completion_tokens,
                    peak_memory_usage=Memory.from_gb(mx.get_peak_memory() / 1e9),
                    prefix_cache_hit=state.prefix_cache_hit,
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
                        reasoning_tokens=0
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
        self._step_count += 1
        if self._step_count % 64 == 0 and responses:
            logger.debug(
                f"step overhead: {_overhead * 1000:.2f}ms (next={_next_elapsed * 1000:.2f}ms total={_step_elapsed * 1000:.2f}ms)"
            )

        return results

    def cancel(self, uids: list[int]) -> None:
        # A prompt the batch is still prefilling is waiting, not generating, so mlx_lm's
        # remove() below drops it; forget its pending prefix-cache save with it.
        for uid in uids:
            self._prefills_in_batch.pop(uid, None)
        # A sequence that is generating is finished by mlx_lm at its next step, as if it had
        # reached its token limit, instead of being removed from the batch here. Removing it
        # between steps deadlocked pipeline-parallel instances; finishing is the path every
        # completed request takes. Its last token is discarded.
        generation_batch = self._mlx_gen._generation_batch
        generating = {uid: i for i, uid in enumerate(generation_batch.uids)}
        for uid in uids:
            if uid in generating:
                generation_batch.max_tokens[generating[uid]] = 0
                self._finishing.add(uid)
        waiting = [uid for uid in uids if uid not in generating]
        if waiting:
            self._mlx_gen.remove(waiting)
        for uid in uids:
            self._active_tasks.pop(uid, None)

    def _share_prefill_budget(self) -> None:
        """Size the next step's prompt chunks so the step prefills about
        EXO_PREFILL_STEP_TOKENS in all, however many prompts it takes."""
        mlx_gen = self._mlx_gen
        prefilling = len(mlx_gen._prompt_batch)
        admitted = min(
            mlx_gen.prefill_batch_size - prefilling,
            len(mlx_gen._unprocessed_sequences),
        )
        mlx_gen.prefill_step_size = prefill_tokens_per_prompt(
            EXO_PREFILL_STEP_TOKENS, prefilling + admitted
        )

    def _advance_prefills_in_batch(
        self, prompt_responses: list[PromptProcessingBatch.Response]
    ) -> None:
        """Follow the prompts the batch prefilled this step: save each one's cache when
        its first segment ends, report progress, and run the runner's cancel hook.

        The runner's cancel hook (distributed_prompt_progress_callback) raises
        PrefillCancelled once it has recorded a cancellation, and cancels the uid itself at
        the end of its step. Raised out of here it would abort the step for every other
        sequence in the batch, so it is caught and nothing more is needed.
        """
        for response in prompt_responses:
            pending = self._prefills_in_batch.get(response.uid)
            if pending is None:
                continue
            processed = cast(tuple[int, int], response.progress)[0]
            # Only the first segment's end comes at exactly its length: the tail segments
            # end at one and two tokens more.
            if (
                response.end_of_segment
                and not response.end_of_prompt
                and processed == pending.first_segment_tokens
            ):
                self._finish_prefill_in_batch(response.uid, pending)
            if response.end_of_prompt:
                del self._prefills_in_batch[response.uid]
            if pending.on_prefill_progress is not None:
                pending.on_prefill_progress(
                    min(processed, pending.progress_total), pending.progress_total
                )
            if (
                pending.distributed_prompt_progress_callback is not None
                and not response.end_of_prompt
            ):
                with contextlib.suppress(PrefillCancelled):
                    pending.distributed_prompt_progress_callback()

    def _finish_prefill_in_batch(self, uid: int, pending: _PrefillInBatch) -> None:
        """Record the prefill speed and save the filled cache to the prefix cache.

        The speed is wall time from submit, so it includes the decode steps in between.
        """
        elapsed = time.perf_counter() - pending.submitted_at
        prefill_tps = pending.first_segment_tokens / elapsed if elapsed > 0 else 0.0
        task = self._active_tasks.get(uid)
        if task is not None:
            task.prefill_tps = prefill_tps
        logger.info(
            f"Prefilled {pending.first_segment_tokens} tokens in the batch in "
            f"{elapsed:.2f}s ({prefill_tps:.0f} tok/s, decode steps included)"
        )
        if pending.min_prefix_hit_length is None or self.kv_prefix_cache is None:
            return
        self._save_prefill_in_batch(
            uid, pending, pending.min_prefix_hit_length, prefill_tps
        )

    def _save_prefill_in_batch(
        self,
        uid: int,
        pending: _PrefillInBatch,
        min_prefix_hit_length: int,
        prefill_tps: float,
    ) -> None:
        """Copy a prompt's cache out of the batch and save it to the prefix cache.

        Saving is best effort, as in _save_prefix_cache: if the copy fails, the request
        carries on and only a later prefix hit is lost, so the error is logged.
        """
        kv_prefix_cache = self.kv_prefix_cache
        assert kv_prefix_cache is not None
        expected_length = len(pending.all_prompt_tokens) - 2
        try:
            cache, _ = cast(
                tuple[list[KVCache], list[int]],
                self._mlx_gen.extract_cache([uid])[uid],
            )
            if cache_length(cache) != expected_length:
                logger.warning(
                    f"Prefill in batch left {cache_length(cache)} tokens in the cache, "
                    f"expected {expected_length}; not saved to the prefix cache"
                )
                return
            # Left lazy, the copy would keep the batch's whole cache buffer alive
            mx.eval(
                [array for entry in cache for array in entry.state if array is not None]
            )
        except Exception:
            logger.warning("Failed to copy a prefilled cache", exc_info=True)
            return
        self._save_prefix_cache(
            pending.all_prompt_tokens,
            cache,
            [],
            pending.prefix_hit_length,
            matched_index_if_unchanged(
                kv_prefix_cache.prompts,
                pending.all_prompt_tokens,
                pending.prefix_hit_length,
                pending.matched_index,
            ),
            min_prefix_hit_length,
            pending.media_regions,
            prefill_tps=prefill_tps,
        )

    def close(self) -> None:
        self._mlx_gen.close()
        mx.clear_cache()

    def _save_prefix_cache(
        self,
        all_prompt_tokens: mx.array,
        cache: KVCacheType,
        cache_snapshots: list[CacheSnapshot] | None,
        prefix_hit_length: int,
        matched_index: int | None,
        min_prefix_hit_length: int = 1000,
        media_regions: list[MediaRegion] | None = None,
        prefill_tps: float = 0.0,
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
                    prefill_tps=prefill_tps,
                )
            else:
                self.kv_prefix_cache.add_kv_cache(
                    all_prompt_tokens,
                    cache,
                    cache_snapshots,
                    media_regions=media_regions,
                    prefill_tps=prefill_tps,
                )
        except Exception:
            logger.warning("Failed to save prefix cache", exc_info=True)
