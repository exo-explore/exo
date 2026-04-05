import contextlib
import functools
import math
import os
import sys
import time
from copy import deepcopy
from typing import Callable, Generator, cast, get_args

import mlx.core as mx
from mlx_lm.generate import (
    maybe_quantize_kv_cache,
    stream_generate,
)
from mlx_lm.models.cache import ArraysCache, RotatingKVCache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.api.types import (
    CompletionTokensDetails,
    FinishReason,
    GenerationStats,
    PromptTokensDetails,
    TopLogprobItem,
    Usage,
)
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.mlx import KVCacheType, Model
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
)
from exo.worker.engines.mlx.auto_parallel import (
    PipelineFirstLayer,
    PipelineLastLayer,
    clear_prefill_sends,
    flush_prefill_sends,
    set_pipeline_prefill,
    set_pipeline_queue_sends,
)
from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    encode_prompt,
    has_non_kv_caches,
    make_kv_cache,
    snapshot_ssm_states,
)
from exo.worker.engines.mlx.constants import (
    DEFAULT_TOP_LOGPROBS,
    KV_BITS,
    KV_GROUP_SIZE,
    MAX_TOKENS,
)
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    fix_unmatched_think_end_tokens,
    mx_barrier,
    system_prompt_token_count,
)
from exo.worker.engines.mlx.vision import (
    MediaRegion,
    VisionProcessor,
    VisionResult,
    get_inner_model,
    prepare_vision,
)
from exo.worker.runner.bootstrap import logger

generation_stream = mx.new_stream(mx.default_device())

_MIN_PREFIX_HIT_RATIO_TO_UPDATE = 0.5


@contextlib.contextmanager
def patch_embed_tokens(
    model: Model, embeddings: mx.array, start_offset: int = 0, token_count: int = 0
) -> Generator[None]:
    inner = get_inner_model(model)  # type: ignore
    original_embed = inner.embed_tokens  # type: ignore
    end_offset = start_offset + token_count
    offset = [start_offset]

    def _inject(input_ids: mx.array) -> mx.array:
        start = offset[0]
        if start >= end_offset:
            return original_embed(input_ids)  # type: ignore
        chunk_len = input_ids.shape[-1]
        end = min(start + chunk_len, end_offset)
        offset[0] = end
        if end - start < chunk_len:
            return original_embed(input_ids)  # type: ignore
        return embeddings[:, start:end, :]

    for attr in dir(original_embed):  # type: ignore
        if not attr.startswith("_") and not hasattr(_inject, attr):
            with contextlib.suppress(AttributeError, TypeError):
                setattr(_inject, attr, getattr(original_embed, attr))  # type: ignore

    inner.embed_tokens = _inject
    try:
        yield
    finally:
        inner.embed_tokens = original_embed


class PrefillCancelled(BaseException):
    """Raised when prefill is cancelled via the progress callback."""


def _has_pipeline_communication_layer(model: Model):
    for layer in model.layers:
        if isinstance(layer, (PipelineFirstLayer, PipelineLastLayer)):
            return True
    return False


def pipeline_parallel_prefill(
    model: Model,
    prompt: mx.array,
    prompt_cache: KVCacheType,
    prefill_step_size: int,
    kv_group_size: int | None,
    kv_bits: int | None,
    prompt_progress_callback: Callable[[int, int], None],
    distributed_prompt_progress_callback: Callable[[], None] | None,
    group: mx.distributed.Group,
) -> None:
    """Prefill the KV cache for pipeline parallel with overlapping stages.

    Each rank processes the full prompt through its real cache, offset by leading
    and trailing dummy iterations.

    Total iterations per rank = N_real_chunks + world_size - 1:
      - rank r leading dummies  (skip_pipeline_io, throwaway cache)
      - N_real_chunks real      (pipeline IO active, real cache)
      - (world_size-1-r) trailing dummies (skip_pipeline_io, throwaway cache)

    e.g.
    Timeline (2 ranks, 3 chunks of 10240 tokens @ step=4096):
        iter 0: R0 real[0:4096]     R1 dummy
        iter 1: R0 real[4096:8192]  R1 real[0:4096]
        iter 2: R0 real[8192:10240] R1 real[4096:8192]
        iter 3: R0 dummy            R1 real[8192:10240]

    This function is designed to match mlx_lm's stream_generate exactly in terms of
    side effects (given the same prefill step size)
    """
    prefill_step_size = prefill_step_size // min(4, group.size())

    quantize_cache_fn: Callable[..., None] = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=0,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    _prompt_cache: KVCacheType = prompt_cache
    rank = group.rank()
    world_size = group.size()

    # Build list of real prompt chunk sizes
    total = len(prompt)
    real_chunk_sizes: list[int] = []
    remaining = total - 1
    while remaining:
        n = min(prefill_step_size, remaining)
        real_chunk_sizes.append(n)
        remaining -= n
    n_real = len(real_chunk_sizes)

    # Each rank does: [rank leading dummies] [N real chunks] [world_size-1-rank trailing dummies]
    n_leading = rank
    n_trailing = world_size - 1 - rank
    n_total = n_leading + n_real + n_trailing

    t_start = time.perf_counter()
    processed = 0
    logger.info(
        f"[R{rank}] Pipeline prefill: {n_real} real + {n_leading} leading + {n_trailing} trailing = {n_total} iterations"
    )
    clear_prefill_sends()

    # Initial callback matching generate_step
    prompt_progress_callback(0, total)

    from exo.worker.engines.mlx.trace import request_trace

    try:
        with mx.stream(generation_stream):
            for _ in range(n_leading):
                if distributed_prompt_progress_callback is not None:
                    distributed_prompt_progress_callback()

            for i in range(n_real):
                chunk_size = real_chunk_sizes[i]
                _t_fwd = time.perf_counter()
                model(
                    prompt[processed : processed + chunk_size][None],
                    cache=_prompt_cache,
                )
                quantize_cache_fn(_prompt_cache)
                request_trace.record(f"prefill.chunk{i}.forward({chunk_size}tok)", _t_fwd)
                processed += chunk_size

                if distributed_prompt_progress_callback is not None:
                    _t_cb = time.perf_counter()
                    distributed_prompt_progress_callback()
                    request_trace.record(f"prefill.chunk{i}.distributed_cb", _t_cb)

                _t_flush = time.perf_counter()
                flush_prefill_sends()
                request_trace.record(f"prefill.chunk{i}.flush_sends", _t_flush)

                _t_eval = time.perf_counter()
                mx.eval([c.state for c in _prompt_cache])  # type: ignore
                request_trace.record(f"prefill.chunk{i}.eval_cache", _t_eval)

                # Break shared-buffer references in DeltaNet (ArraysCache) entries.
                # Use async_eval so the CPU can proceed to the next chunk's forward
                # while the GPU finishes contiguous on the same command queue.
                _t_contig = time.perf_counter()
                for _c in _prompt_cache:
                    if isinstance(_c, ArraysCache):
                        _c.cache = [mx.contiguous(x) if x is not None else x for x in _c.cache]
                        mx.async_eval(*[x for x in _c.cache if x is not None])
                request_trace.record(f"prefill.chunk{i}.contiguous", _t_contig)

                # Log memory every 5 chunks for profiling
                if i % 5 == 0 or i == n_real - 1:
                    active_gb = mx.metal.get_active_memory() / 1024**3
                    peak_gb = mx.metal.get_peak_memory() / 1024**3
                    logger.info(f"[MEM] prefill chunk {i+1}/{n_real} ({processed} tokens): active={active_gb:.2f} GB, peak={peak_gb:.2f} GB")

                prompt_progress_callback(processed, total)

            for _ in range(n_trailing):
                if distributed_prompt_progress_callback is not None:
                    distributed_prompt_progress_callback()

    finally:
        clear_prefill_sends()

    # Post-loop: process the remaining 1 token not covered by the chunk loop.
    # The chunk loop processes total-1 tokens; this handles the last one.
    # (Previously did 2 forward passes to match stream_generate's extra generated
    # token, but that's unnecessary — prefill() trims conditionally now.)
    _t_post = time.perf_counter()
    with mx.stream(generation_stream):
        model(prompt[-1:][None], cache=_prompt_cache)
        quantize_cache_fn(_prompt_cache)
    flush_prefill_sends()
    request_trace.record("prefill.post_loop_token", _t_post)

    assert _prompt_cache is not None
    with mx.stream(generation_stream):
        mx.eval([c.state for c in _prompt_cache])  # type: ignore

    # Final callback matching generate_step
    prompt_progress_callback(total, total)

    logger.info(
        f"[R{rank}] Prefill: {n_real} real + {n_leading}+{n_trailing} dummy iterations, "
        f"Processed {processed} tokens in {(time.perf_counter() - t_start) * 1000:.1f}ms"
    )


def prefill(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    prompt_tokens: mx.array,
    cache: KVCacheType,
    group: mx.distributed.Group | None,
    on_prefill_progress: Callable[[int, int], None] | None,
    distributed_prompt_progress_callback: Callable[[], None] | None,
) -> tuple[float, int, list[CacheSnapshot]]:
    """Prefill the KV cache with prompt tokens.

    This runs the model over the prompt tokens to populate the cache,
    then trims off the extra generated token.

    Returns:
        (tokens_per_sec, num_tokens, snapshots)
    """
    num_tokens = len(prompt_tokens)
    if num_tokens == 0:
        return 0.0, 0, []

    logger.debug(f"Prefilling {num_tokens} tokens...")
    start_time = time.perf_counter()
    has_ssm = has_non_kv_caches(cache)
    snapshots: list[CacheSnapshot] = []

    # TODO(evan): kill the callbacks/runner refactor
    def progress_callback(processed: int, total: int) -> None:
        elapsed = time.perf_counter() - start_time
        tok_per_sec = processed / elapsed if elapsed > 0 else 0
        logger.debug(
            f"Prefill progress: {processed}/{total} tokens ({tok_per_sec:.1f} tok/s)"
        )
        if has_ssm:
            # Only keep the last 2 snapshots — the rollback at the end uses
            # snapshots[-2] exclusively.  Earlier snapshots are never read, so
            # deepcopy-ing the entire cache on every chunk is wasted work.
            if len(snapshots) >= 2:
                snapshots.pop(0)
            snapshots.append(snapshot_ssm_states(cache))

        if on_prefill_progress is not None:
            on_prefill_progress(processed, total)

    def combined_progress_callback(processed: int, total: int) -> None:
        if distributed_prompt_progress_callback is not None:
            distributed_prompt_progress_callback()
        progress_callback(processed, total)

    from exo.worker.engines.mlx.trace import request_trace, T

    set_pipeline_prefill(model, is_prefill=True)

    # Release any cached Metal buffers before prefill to maximize headroom
    # for the forward pass intermediates during long context prefills.
    with T("prefill.clear_cache"):
        mx.clear_cache()

    with T("prefill.barrier"):
        mx_barrier(group)

    # Memory checkpoint before prefill
    with T("prefill.mem_checkpoint"):
        mx.eval(mx.zeros(1))
        active_gb = mx.metal.get_active_memory() / 1024**3
        peak_gb = mx.metal.get_peak_memory() / 1024**3
        cache_gb = mx.metal.get_cache_memory() / 1024**3
    logger.info(f"[MEM] before prefill ({num_tokens} tokens): active={active_gb:.2f} GB, peak={peak_gb:.2f} GB, cache={cache_gb:.2f} GB")
    logger.info("Starting prefill")

    is_pipeline = _has_pipeline_communication_layer(model)

    prefill_step_size = int(os.environ.get("EXO_PREFILL_STEP_SIZE", "4096"))

    try:
        if is_pipeline and num_tokens >= prefill_step_size:
            set_pipeline_queue_sends(model, queue_sends=True)
            assert group is not None, "Pipeline prefill requires a distributed group"
            with T("prefill.pipeline_parallel"):
                pipeline_parallel_prefill(
                    model=model,
                    prompt=prompt_tokens,
                    prompt_cache=cache,
                    prefill_step_size=prefill_step_size,
                    kv_group_size=KV_GROUP_SIZE,
                    kv_bits=KV_BITS,
                    prompt_progress_callback=progress_callback,
                    distributed_prompt_progress_callback=distributed_prompt_progress_callback,
                    group=group,
                )
        else:
            with T("prefill.stream_generate"):
                for _ in stream_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_tokens,
                    max_tokens=1,
                    sampler=sampler,
                    prompt_cache=cache,
                    prefill_step_size=prefill_step_size,
                    kv_group_size=KV_GROUP_SIZE,
                    kv_bits=KV_BITS,
                    prompt_progress_callback=combined_progress_callback,
                ):
                    break
    except PrefillCancelled:
        set_pipeline_queue_sends(model, queue_sends=False)
        set_pipeline_prefill(model, is_prefill=False)
        raise

    set_pipeline_queue_sends(model, queue_sends=False)
    set_pipeline_prefill(model, is_prefill=False)

    # Trim extra entries from the cache so the decode path can reprocess last_tokens[-2:].
    # - stream_generate path: generated 1 extra token → trim(2) (generated + last prompt token)
    # - pipeline_parallel path: 1 post-loop token, no generation → trim(1)
    # SSM/ArraysCache layers are rolled back to snapshots[-2] (state after last real chunk).
    _trim_n = 1 if (is_pipeline and num_tokens >= prefill_step_size) else 2
    with T("prefill.cache_trim_and_rollback"):
        pre_gen = deepcopy(snapshots[-2]) if has_ssm else None
        for i, c in enumerate(cache):
            if has_ssm and isinstance(c, (ArraysCache, RotatingKVCache)):
                assert pre_gen is not None
                if pre_gen.states[i] is not None:
                    cache[i] = deepcopy(pre_gen.states[i])  # type: ignore
            else:
                assert not isinstance(c, (ArraysCache, RotatingKVCache))
                c.trim(_trim_n)

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0.0
    logger.debug(
        f"Prefill complete: {num_tokens} tokens in {elapsed:.2f}s "
        f"({tokens_per_sec:.1f} tok/s)"
    )
    # Exclude the last snapshot
    return tokens_per_sec, num_tokens, snapshots[:-1] if snapshots else []


def warmup_inference(
    model: Model,
    tokenizer: TokenizerWrapper,
    group: mx.distributed.Group | None,
    model_id: ModelId,
) -> int:
    logger.info(f"warming up inference for instance: {model_id}")

    content = "Prompt to warm up the inference engine. Repeat this."

    warmup_task_params = TextGenerationTaskParams(
        model=model_id,
        input=[InputMessage(role="user", content=content)],
        max_output_tokens=50,
        temperature=0.0,
    )

    warmup_prompt = apply_chat_template(
        tokenizer=tokenizer,
        task_params=warmup_task_params,
    )

    tokens_generated = 0

    mx_barrier(group)

    logger.info("Generating warmup tokens")

    t = time.monotonic()

    for _r in mlx_generate(
        model=model,
        tokenizer=tokenizer,
        task=warmup_task_params,
        prompt=warmup_prompt,
        kv_prefix_cache=None,
        group=group,
        is_warmup=True,
    ):
        tokens_generated += 1

    check_for_cancel_every = min(
        math.ceil(tokens_generated / min(time.monotonic() - t, 0.001)), 100
    )

    mx_barrier(group)

    logger.info(f"warmed up by generating {tokens_generated} tokens")
    if group is not None:
        check_for_cancel_every = int(
            mx.max(
                mx.distributed.all_gather(
                    mx.array([check_for_cancel_every]),
                    group=group,
                )
            ).item()
        )

    logger.info(
        f"runner checking for cancellation every {check_for_cancel_every} tokens"
    )

    return check_for_cancel_every


def ban_token_ids(token_ids: list[int]) -> Callable[[mx.array, mx.array], mx.array]:
    token_ids = [int(t) for t in token_ids]

    def proc(_history: mx.array, logits: mx.array) -> mx.array:
        for tid in token_ids:
            logits[..., tid] = -1e9
        return logits

    return proc


def eos_ids_from_tokenizer(tokenizer: TokenizerWrapper) -> list[int]:
    eos: list[int] | None = getattr(tokenizer, "eos_token_ids", None)
    if eos is None:
        return []
    return eos


def extract_top_logprobs(
    logprobs: mx.array,
    tokenizer: TokenizerWrapper,
    top_logprobs: int,
    selected_token: int,
    precomputed_indices: list[int] | None = None,
    precomputed_values: list[float] | None = None,
    precomputed_selected: float | None = None,
) -> tuple[float, list[TopLogprobItem]]:
    if (
        precomputed_indices is not None
        and precomputed_values is not None
        and precomputed_selected is not None
    ):
        top_indices_list: list[int] = precomputed_indices[:top_logprobs]
        top_values_list: list[float] = precomputed_values[:top_logprobs]
        selected_logprob = precomputed_selected
    else:
        selected_logprob_arr = logprobs[selected_token]
        top_logprobs = min(top_logprobs, logprobs.shape[0] - 1)
        top_indices = mx.argpartition(-logprobs, top_logprobs)[:top_logprobs]
        top_values = logprobs[top_indices]
        sort_order = mx.argsort(-top_values)
        top_indices = top_indices[sort_order]
        top_values = top_values[sort_order]
        mx.eval(selected_logprob_arr, top_indices, top_values)
        selected_logprob = float(selected_logprob_arr.item())
        top_indices_list = top_indices.tolist()  # type: ignore
        top_values_list = top_values.tolist()  # type: ignore

    # Convert to list of TopLogprobItem
    top_logprob_items: list[TopLogprobItem] = []
    for token_id, token_logprob in zip(top_indices_list, top_values_list, strict=True):
        if math.isnan(token_logprob):
            continue

        # Decode token ID to string
        token_str = tokenizer.decode([token_id])
        top_logprob_items.append(
            TopLogprobItem(
                token=token_str,
                logprob=token_logprob,
                bytes=list(token_str.encode("utf-8")),
            )
        )

    return selected_logprob, top_logprob_items


def mlx_generate(
    model: Model,
    tokenizer: TokenizerWrapper,
    task: TextGenerationTaskParams,
    prompt: str,
    kv_prefix_cache: KVPrefixCache | None,
    group: mx.distributed.Group | None,
    on_prefill_progress: Callable[[int, int], None] | None = None,
    distributed_prompt_progress_callback: Callable[[], None] | None = None,
    on_generation_token: Callable[[], None] | None = None,
    vision_processor: VisionProcessor | None = None,
    is_warmup: bool = False,
) -> Generator[GenerationResponse]:
    # Ensure that generation stats only contains peak memory for this generation
    mx.reset_peak_memory()
    # TODO: Randomise task seed and set in taskparams, instead of hard coding as 42.
    seed = task.seed or 42
    mx.random.seed(seed)

    # Encode prompt once at the top and fix unmatched think tags
    all_prompt_tokens = encode_prompt(tokenizer, prompt)
    all_prompt_tokens = fix_unmatched_think_end_tokens(all_prompt_tokens, tokenizer)
    min_prefix_hit_length = max(1000, system_prompt_token_count(task, tokenizer))

    vision: VisionResult | None = None
    if vision_processor is not None:
        try:
            vision = prepare_vision(
                images=task.images,
                chat_template_messages=task.chat_template_messages,
                vision_processor=vision_processor,
                tokenizer=tokenizer,
                model=model,
                model_id=task.model,
                task_params=task,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Vision processing failed, falling back to text-only"
            )
    if vision is not None:
        all_prompt_tokens = vision.prompt_tokens
    media_regions: list[MediaRegion] = vision.media_regions if vision else []

    # Do not use the prefix cache if we are trying to do benchmarks.
    is_bench = task.bench
    if is_bench:
        kv_prefix_cache = None

    # Use prefix cache if available, otherwise create fresh cache
    prefix_hit_length = 0
    matched_index: int | None = None
    if kv_prefix_cache is None:
        caches = make_kv_cache(model=model)
        prompt_tokens = all_prompt_tokens
    else:
        caches, prompt_tokens, matched_index = kv_prefix_cache.get_kv_cache(
            model, all_prompt_tokens, media_regions=media_regions
        )
        prefix_hit_length = len(all_prompt_tokens) - len(prompt_tokens)
        if prefix_hit_length > 0:
            logger.info(
                f"KV cache hit: {prefix_hit_length}/{len(all_prompt_tokens)} tokens cached ({100 * prefix_hit_length / len(all_prompt_tokens):.1f}%)"
            )

    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] = (
        make_logits_processors(
            repetition_penalty=task.repetition_penalty,
            repetition_context_size=task.repetition_context_size,
        )
    )
    if is_bench:
        # Only sample length eos tokens
        eos_ids = eos_ids_from_tokenizer(tokenizer)
        logits_processors = [ban_token_ids(eos_ids)] + logits_processors

    sampler = make_sampler(
        temp=task.temperature if task.temperature is not None else 0.7,
        top_p=task.top_p if task.top_p is not None else 1.0,
        min_p=task.min_p if task.min_p is not None else 0.05,
        top_k=task.top_k if task.top_k is not None else 0,
    )

    # Normalize stop sequences to a list
    stop_sequences: list[str] = (
        ([task.stop] if isinstance(task.stop, str) else task.stop)
        if task.stop is not None
        else []
    )
    max_stop_len = max((len(s) for s in stop_sequences), default=0)

    maybe_vision_ctx = (
        patch_embed_tokens(
            model, vision.embeddings, prefix_hit_length, len(prompt_tokens) - 1
        )
        if vision is not None
        else contextlib.nullcontext()
    )
    with maybe_vision_ctx:
        prefill_tps, prefill_tokens, ssm_snapshots_list = prefill(
            model,
            tokenizer,
            sampler,
            prompt_tokens[:-1],
            caches,
            group,
            on_prefill_progress,
            distributed_prompt_progress_callback,
        )
    cache_snapshots: list[CacheSnapshot] | None = ssm_snapshots_list or None

    # stream_generate starts from the last token
    last_token = prompt_tokens[-2:]

    max_tokens = task.max_output_tokens or MAX_TOKENS
    accumulated_text = ""
    generated_text_parts: list[str] = []
    generation_start_time = time.perf_counter()
    usage: Usage | None = None
    in_thinking = False
    reasoning_tokens = 0
    think_start = tokenizer.think_start
    think_end = tokenizer.think_end

    # Memory checkpoint after prefill, before decode
    mx.eval(mx.zeros(1))
    active_gb = mx.metal.get_active_memory() / 1024**3
    peak_gb = mx.metal.get_peak_memory() / 1024**3
    cache_gb = mx.metal.get_cache_memory() / 1024**3
    logger.info(f"[MEM] after prefill, before decode: active={active_gb:.2f} GB, peak={peak_gb:.2f} GB, cache={cache_gb:.2f} GB")
    logger.info("Starting decode")
    mx_barrier(group)

    # --- PP idle-time speculation (skipped during warmup) ---
    _pp_spec_gen = None
    _pp_draft = getattr(model, "_pp_draft_model", None)
    _pp_draft_cache = getattr(model, "_pp_draft_cache", None)
    # Both ranks must enter the speculation loop — check env var, not model attribute
    # (draft model is only on rank 0, but rank 1 must participate in the protocol)
    _has_pp_draft = bool(os.environ.get("EXO_PP_DRAFT_MODEL", ""))
    logger.info(f"PP spec check: is_warmup={is_warmup}, has_draft_env={_has_pp_draft}, "
                f"draft_model={'yes' if _pp_draft else 'no'}, "
                f"group={'size=' + str(group.size()) if group else 'None'}")
    if (not is_warmup
        and _has_pp_draft
        and group is not None
        and group.size() > 1):
        try:
            from ..pp_speculation import (
                get_pipeline_info,
                pp_speculative_decode_loop,
                _install_spec_layers,
                _configure_layers,
            )
            pp_info = get_pipeline_info(model)
            logger.info(f"PP spec: get_pipeline_info returned {pp_info}")
            if pp_info is not None:
                pp_rank, pp_world_size, pp_group = pp_info
                inner = getattr(model, "language_model", model)
                _install_spec_layers(inner)

                # Prefill draft cache with tail of prompt (rank 0 only, instant — no PP needed)
                # The draft model uses a RotatingKVCache, so only recent tokens matter.
                if pp_rank == 0 and _pp_draft is not None:
                    _draft_kv_window = int(os.environ.get("EXO_DRAFT_KV_WINDOW", "4096"))
                    _draft_prompt = all_prompt_tokens[-_draft_kv_window:]
                    _draft_chunk = 512
                    for i in range(0, len(_draft_prompt), _draft_chunk):
                        _pp_draft(_draft_prompt[i:i + _draft_chunk][None], cache=_pp_draft_cache)
                        mx.eval([c.state if hasattr(c, 'state') else c for c in _pp_draft_cache])
                    logger.info(f"Draft model prefilled with {len(_draft_prompt)} tokens (of {len(all_prompt_tokens)} total)")

                # First token via standard PP (both ranks, synchronized)
                _first_gen = stream_generate(
                    model=model, tokenizer=tokenizer, prompt=last_token,
                    max_tokens=1, sampler=sampler, logits_processors=logits_processors,
                    prompt_cache=caches, prefill_step_size=1,
                    kv_group_size=KV_GROUP_SIZE, kv_bits=KV_BITS,
                )
                _first_out = next(_first_gen)
                first_y = mx.array([_first_out.token])
                mx.eval(first_y)

                logger.info(f"PP speculation active: rank={pp_rank}")

                def _spec_token_gen():
                    from mlx_lm.generate import GenerationResponse
                    _detok = tokenizer.detokenizer
                    gen_start = time.perf_counter()
                    # Clear finish_reason from max_tokens=1 — this is just the first token
                    _first_fixed = GenerationResponse(
                        text=_first_out.text, token=_first_out.token,
                        logprobs=_first_out.logprobs, from_draft=False,
                        prompt_tokens=_first_out.prompt_tokens,
                        prompt_tps=_first_out.prompt_tps,
                        generation_tokens=_first_out.generation_tokens,
                        generation_tps=_first_out.generation_tps,
                        peak_memory=_first_out.peak_memory,
                        finish_reason=None,
                    )
                    yield _first_fixed

                    for tok_id, lp in pp_speculative_decode_loop(
                        model=model, draft_model=_pp_draft,
                        prompt_cache=caches, draft_cache=_pp_draft_cache,
                        sampler=sampler, logits_processors=logits_processors,
                        first_y=first_y, first_logprobs=mx.zeros(1),
                        max_tokens=max_tokens - 1,
                        pp_rank=pp_rank, pp_world_size=pp_world_size,
                        pp_group=pp_group,
                    ):
                        if tok_id in tokenizer.eos_token_ids:
                            elapsed = time.perf_counter() - gen_start
                            yield GenerationResponse(
                                text="", token=tok_id, logprobs=lp, from_draft=False,
                                prompt_tokens=len(last_token), prompt_tps=prefill_tps or 0.0,
                                generation_tokens=1, generation_tps=1.0/elapsed if elapsed > 0 else 0,
                                peak_memory=mx.get_peak_memory()/1e9, finish_reason="stop",
                            )
                            return
                        _detok.add_token(tok_id)
                        elapsed = time.perf_counter() - gen_start
                        yield GenerationResponse(
                            text=_detok.last_segment, token=tok_id, logprobs=lp, from_draft=False,
                            prompt_tokens=len(last_token), prompt_tps=prefill_tps or 0.0,
                            generation_tokens=1, generation_tps=1.0/elapsed if elapsed > 0 else 0,
                            peak_memory=mx.get_peak_memory()/1e9,
                        )

                _pp_spec_gen = _spec_token_gen()
        except Exception as e:
            sys.stderr.write(f"[PP speculation] setup failed: {e}\n")
            sys.stderr.flush()
            _pp_spec_gen = None

    _decode_gen = _pp_spec_gen if _pp_spec_gen is not None else stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=last_token,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        prompt_cache=caches,
        prefill_step_size=1,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
    )

    for completion_tokens, out in enumerate(
        _decode_gen,
        start=1,
    ):
        generated_text_parts.append(out.text)
        accumulated_text += out.text

        if think_start is not None and out.text == think_start:
            in_thinking = True
        elif think_end is not None and out.text == think_end:
            in_thinking = False
        if in_thinking:
            reasoning_tokens += 1

        # Check for stop sequences
        text = out.text
        finish_reason: FinishReason | None = cast(
            FinishReason | None, out.finish_reason
        )
        stop_matched = False

        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in accumulated_text:
                    # Trim text to just before the stop sequence
                    stop_index = accumulated_text.find(stop_seq)
                    text_before_stop = accumulated_text[:stop_index]
                    chunk_start = len(accumulated_text) - len(out.text)
                    text = text_before_stop[chunk_start:]
                    finish_reason = "stop"
                    stop_matched = True
                    break

        is_done = finish_reason is not None

        stats: GenerationStats | None = None
        if is_done:
            stats = GenerationStats(
                prompt_tps=float(prefill_tps or out.prompt_tps),
                generation_tps=float(out.generation_tps),
                prompt_tokens=int(prefill_tokens + out.prompt_tokens),
                generation_tokens=int(out.generation_tokens),
                peak_memory_usage=Memory.from_gb(out.peak_memory),
            )
            if not stop_matched and out.finish_reason not in get_args(FinishReason):
                logger.warning(
                    f"Model generated unexpected finish_reason: {out.finish_reason}"
                )

            total_prompt_tokens = len(all_prompt_tokens)
            usage = Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_prompt_tokens + completion_tokens,
                prompt_tokens_details=PromptTokensDetails(
                    cached_tokens=prefix_hit_length
                ),
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=reasoning_tokens
                ),
            )

        # Extract logprobs from the full vocabulary logprobs array
        logprob: float | None = None
        top_logprobs: list[TopLogprobItem] | None = None
        if task.logprobs:
            with mx.stream(generation_stream):
                logprob, top_logprobs = extract_top_logprobs(
                    logprobs=out.logprobs,
                    tokenizer=tokenizer,
                    top_logprobs=task.top_logprobs or DEFAULT_TOP_LOGPROBS,
                    selected_token=out.token,
                )

        if is_done:
            # Log generation stats
            generation_elapsed = time.perf_counter() - generation_start_time
            generated_tokens = len(generated_text_parts)
            generation_tps = (
                generated_tokens / generation_elapsed if generation_elapsed > 0 else 0.0
            )
            logger.debug(
                f"Generation complete: prefill {prompt_tokens} tokens @ "
                f"{prefill_tps:.1f} tok/s, generated {generated_tokens} tokens @ "
                f"{generation_tps:.1f} tok/s"
            )
            if kv_prefix_cache is not None:
                generated_tokens_array = mx.array(
                    tokenizer.encode(
                        "".join(generated_text_parts), add_special_tokens=False
                    )
                )
                full_prompt_tokens = mx.concatenate(
                    [all_prompt_tokens, generated_tokens_array]
                )
                hit_ratio = (
                    prefix_hit_length / len(all_prompt_tokens)
                    if len(all_prompt_tokens) > 0
                    else 0.0
                )
                if matched_index is not None and (
                    prefix_hit_length >= min_prefix_hit_length
                    and hit_ratio >= _MIN_PREFIX_HIT_RATIO_TO_UPDATE
                ):
                    kv_prefix_cache.update_kv_cache(
                        matched_index,
                        full_prompt_tokens,
                        caches,
                        cache_snapshots,
                        restore_pos=prefix_hit_length,
                        media_regions=media_regions,
                    )
                else:
                    kv_prefix_cache.add_kv_cache(
                        full_prompt_tokens,
                        caches,
                        cache_snapshots,
                        media_regions=media_regions,
                    )

        if on_generation_token is not None:
            on_generation_token()

        yield GenerationResponse(
            text=text,
            token=out.token,
            logprob=logprob,
            top_logprobs=top_logprobs,
            finish_reason=finish_reason,
            stats=stats,
            usage=usage,
        )

        if is_done:
            mx_barrier(group)
            break

        # Limit accumulated_text to what's needed for stop sequence detection
        if max_stop_len > 0 and len(accumulated_text) > max_stop_len:
            accumulated_text = accumulated_text[-max_stop_len:]
