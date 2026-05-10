import contextlib
import functools
import math
import time
import uuid
from typing import Any, Callable, Generator, Literal, cast, get_args

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import (
    PromptProcessingBatch,
    maybe_quantize_kv_cache,
    stream_generate,
)
from mlx_lm.models.cache import trim_prompt_cache as mlx_trim_prompt_cache
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
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
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
    copy_snapshot_entry,
    encode_prompt,
    has_non_kv_caches,
    is_non_trimmable_cache_entry,
    make_kv_cache,
    snapshot_ssm_states,
)
from exo.worker.engines.mlx.constants import (
    DEFAULT_TOP_LOGPROBS,
    KV_BITS,
    KV_GROUP_SIZE,
    MAX_TOKENS,
)
from exo.worker.engines.mlx.generator.drafter import (
    Drafter,
    DraftMode,
    make_drafter,
    resolve_asymmetric_draft_mode,
    resolve_draft_mode,
)

# Reuse the same env-gated diagnostic helper as ``pipelined_drafter`` so
# spec-diag INFO logs are off by default (Codex P2, PR #21 round 3).
# Kept private (underscore) to advertise that this is a runner-internal
# debug surface, not a public API; the cross-module import is the
# pragmatic shape because the helper lives at the same engine layer.
from exo.worker.engines.mlx.generator.pipelined_drafter import (
    _spec_diag,  # pyright: ignore[reportPrivateUsage]
)
from exo.worker.engines.mlx.generator.remote_prefill import remote_prefill
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.utils_mlx import (
    CoupledDrafter,
    apply_chat_template,
    fix_unmatched_think_end_tokens,
    mx_barrier,
    mx_broadcast_int_list,
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


def _broadcast_clamped_num_draft_tokens(
    *,
    effective_num_draft_tokens: int,
    group: mx.distributed.Group,
) -> int:
    """Make every target rank in a multi-target asymmetric placement
    agree on ``num_draft_tokens`` by broadcasting rank 0's
    (potentially-clamped) value over ``group``.

    Only rank 0 of the target subgroup holds the
    ``DrafterTransport`` (the socket to the drafter rank), so only
    rank 0 can call ``clamp_num_draft_tokens_to_transport``. Pre-fix
    non-root target ranks used the unclamped
    ``effective_num_draft_tokens`` to size their
    ``PipelinedModelDrafter`` buffers, so a per-request override
    above the transport budget would have desynchronized the
    spec-decode collectives in ``_broadcast_drafts`` /
    ``_broadcast_target_tokens`` (Codex P1 on PR #20 round 3).

    The broadcast is a one-shot length-1 int collective. It rides
    the same ``mx.distributed.Group`` the spec-decode rounds use,
    not the ``target_peer_fanout`` socket, so it does not interact
    with the JACCL int/float wire-conflation issue the fanout
    works around -- correctness only depends on every target rank
    agreeing on a positive int slot count, which
    ``mx_broadcast_int_list`` already enforces.

    Returns the K every rank should use for the rest of this
    request. Rank 0's return value is identical to the input; non-
    root ranks' return value is the rank-0-clamped value.
    """
    is_root_in_target_group = group.rank() == 0
    broadcast = mx_broadcast_int_list(
        [int(effective_num_draft_tokens)] if is_root_in_target_group else None,
        length=1,
        group=group,
        is_root=is_root_in_target_group,
    )
    consensus_k = broadcast[0]
    if consensus_k != effective_num_draft_tokens:
        # On rank 0 this is unreachable (we just sent the value);
        # on non-root ranks this is the expected path whenever
        # the per-request override was clamped at the root.
        logger.info(
            f"non-root target rank adopting clamped "
            f"num_draft_tokens={consensus_k} from rank 0 "
            f"(local pre-broadcast={effective_num_draft_tokens})"
        )
    return consensus_k


REMOTE_PREFILL_MIN_TOKENS = 1000

generation_stream = mx.new_stream(mx.default_device())

_MIN_PREFIX_HIT_RATIO_TO_UPDATE = 0.5


@contextlib.contextmanager
def patch_embed_tokens(
    model: Model,
    embeddings: mx.array,
    start_offset: int = 0,
    token_count: int = 0,
    image_token_id: int | None = None,
) -> Generator[None]:
    inner = get_inner_model(model)  # type: ignore
    original_embed = inner.embed_tokens  # type: ignore
    end_offset = start_offset + token_count
    offset = [start_offset]

    def _inject(input_ids: mx.array) -> mx.array:
        chunk_start = offset[0]
        chunk_len = input_ids.shape[-1]
        chunk_end = chunk_start + chunk_len
        offset[0] = chunk_end

        # The injection window is [start_offset, end_offset).
        if chunk_end <= start_offset or chunk_start >= end_offset:
            return original_embed(input_ids)  # type: ignore

        # Mixed chunk: splice the pre-computed embeddings for the overlap
        # into `original_embed(input_ids)` for any text-only fringes.
        overlap_start = max(chunk_start, start_offset)
        overlap_end = min(chunk_end, end_offset)
        dst_start = overlap_start - chunk_start
        dst_end = overlap_end - chunk_start
        text_embeds: mx.array = original_embed(input_ids)  # type: ignore
        return mx.concatenate(
            [
                text_embeds[:, :dst_start, :],
                embeddings[:, overlap_start:overlap_end, :],
                text_embeds[:, dst_end:, :],
            ],
            axis=1,
        )

    for attr in dir(original_embed):  # type: ignore
        if not attr.startswith("_") and not hasattr(_inject, attr):
            with contextlib.suppress(AttributeError, TypeError):
                setattr(_inject, attr, getattr(original_embed, attr))  # type: ignore

    inner.embed_tokens = _inject

    # Gemma 4 (e2b/e4b) has a second, independent embedding table that produces
    # per-layer conditioning signals via self.embed_tokens_per_layer(input_ids).
    # The injected vision embeddings live in the main residual stream only, so
    # if image_token_id positions are passed through as-is the per-layer table
    # produces garbage signals at those positions (the `<image>` token was never
    # trained to have meaningful per-layer inputs).
    original_per_layer = getattr(inner, "embed_tokens_per_layer", None)  # type: ignore
    if original_per_layer is not None and image_token_id is not None:

        def _clean_per_layer(input_ids: mx.array) -> mx.array:
            clean_ids = mx.where(
                input_ids == image_token_id, mx.zeros_like(input_ids), input_ids
            )
            return original_per_layer(clean_ids)  # type: ignore

        inner.embed_tokens_per_layer = _clean_per_layer

    try:
        yield
    finally:
        inner.embed_tokens = original_embed
        if original_per_layer is not None and image_token_id is not None:
            inner.embed_tokens_per_layer = original_per_layer


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

    try:
        with mx.stream(generation_stream):
            for _ in range(n_leading):
                if distributed_prompt_progress_callback is not None:
                    distributed_prompt_progress_callback()

            for i in range(n_real):
                chunk_size = real_chunk_sizes[i]
                model(
                    prompt[processed : processed + chunk_size][None],
                    cache=_prompt_cache,
                )
                quantize_cache_fn(_prompt_cache)
                processed += chunk_size

                if distributed_prompt_progress_callback is not None:
                    distributed_prompt_progress_callback()

                flush_prefill_sends()

                prompt_progress_callback(processed, total)

            for _ in range(n_trailing):
                if distributed_prompt_progress_callback is not None:
                    distributed_prompt_progress_callback()

    finally:
        clear_prefill_sends()

    # Post-loop: process remaining 1 token + add +1 entry to match stream_generate.
    for _ in range(2):
        with mx.stream(generation_stream):
            model(prompt[-1:][None], cache=_prompt_cache)
            quantize_cache_fn(_prompt_cache)
        flush_prefill_sends()

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
            snapshots.append(snapshot_ssm_states(cache))

        if on_prefill_progress is not None:
            on_prefill_progress(processed, total)

    def combined_progress_callback(processed: int, total: int) -> None:
        if distributed_prompt_progress_callback is not None:
            distributed_prompt_progress_callback()
        progress_callback(processed, total)

    set_pipeline_prefill(model, is_prefill=True)

    mx_barrier(group)
    logger.info("Starting prefill")

    is_pipeline = _has_pipeline_communication_layer(model)

    prefill_step_size = 4096

    try:
        if is_pipeline and num_tokens >= prefill_step_size:
            set_pipeline_queue_sends(model, queue_sends=True)
            assert group is not None, "Pipeline prefill requires a distributed group"
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
            # Use max_tokens=1 because max_tokens=0 does not work.
            # We just throw away the generated token - we only care about filling the cache
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
                break  # Stop after first iteration - cache is now filled
    except PrefillCancelled:
        set_pipeline_queue_sends(model, queue_sends=False)
        set_pipeline_prefill(model, is_prefill=False)
        raise

    set_pipeline_queue_sends(model, queue_sends=False)
    set_pipeline_prefill(model, is_prefill=False)

    # stream_generate added 1 extra generated token to the cache, so we should trim it.
    # Because of needing to roll back arrays cache, we will generate on 2 tokens so trim 1 more.
    pre_gen = snapshots[-2] if has_ssm else None
    for i, c in enumerate(cache):
        non_trimmable = is_non_trimmable_cache_entry(c)
        if has_ssm and non_trimmable:
            assert pre_gen is not None
            restored = copy_snapshot_entry(pre_gen.states[i])
            if restored is not None:
                cache[i] = restored  # type: ignore
        else:
            assert not non_trimmable
            c.trim(2)

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0.0
    logger.debug(
        f"Prefill complete: {num_tokens} tokens in {elapsed:.2f}s "
        f"({tokens_per_sec:.1f} tok/s)"
    )
    # Exclude the last snapshot
    return tokens_per_sec, num_tokens, snapshots[:-1] if snapshots else []


class BatchedPrefillUnsupportedError(Exception):
    """Raised when ``batched_prefill`` cannot run for the requested batch.

    The caller is expected to recover by falling back to per-slot
    :func:`prefill`. Reasons include cache types that do not implement
    ``merge``/``extract`` (e.g. ``DeepseekV4Cache``), pipeline-parallel
    targets where collective semantics differ, or any prompt being too
    short to leave a decode-seed token after slicing.
    """


def batched_prefill(
    *,
    model: Model,
    prompt_tokens_list: list[mx.array],
    caches_list: list[KVCacheType],
    on_progress: Callable[[int, int], None] | None = None,
    prefill_step_size: int = 4096,
) -> tuple[float, int]:
    """Run K prefills in a single batched forward pass.

    Wraps :class:`mlx_lm.generate.PromptProcessingBatch`. After return, each cache in
    ``caches_list`` is filled in-place to offset ``len(prompt_tokens_list[i]) - 1``
    so the decode loop can seed from the last prompt token (matching the
    exact-prefix-hit shape ``mlx_generate`` already handles via
    ``kv_prefix_cache.get_kv_cache``).

    The K prompts are right-padded to the longest length; per-cache
    ``prepare(lengths=, right_padding=)`` + ``finalize()`` remove the
    padding from the cache state. Total wall-clock cost is roughly the
    cost of one prefill at the longest prompt's length, amortising weight
    loads across the batch — which is the whole point on a single GPU
    where matmul throughput is otherwise weight-bandwidth-bound for the
    sequential per-slot path.

    Args:
        model: target model. Must produce caches whose layers support
            ``merge``/``extract`` (e.g. ``KVCache`` + ``RotatingKVCache`` for
            Gemma-4; ``DeepseekV4Cache`` is not supported and raises
            :class:`BatchedPrefillUnsupportedError`).
        prompt_tokens_list: per-slot full prompt tokens. Each prompt is
            sliced to ``prompt[:-1]`` internally so the decode seed
            (``prompt[-1]``) is left out of the cache.
        caches_list: per-slot fresh caches (offset 0). Mutated in place;
            on return each cache's layers point at the extracted
            per-sequence state from the batched forward.
        on_progress: aggregate ``(processed_max_seq, total_max_seq)``
            callback fired once per ``prefill_step_size`` chunk. The
            ``processed`` count is the per-slot maximum (longest prompt's
            chunk count) so progress monotonically increases even when
            slots have unequal lengths.
        prefill_step_size: chunk size for the prefill loop.

    Returns:
        ``(aggregate_tps, total_tokens)``: sum of per-slot tokens divided
        by batched wall-clock time, useful for telemetry / bench output.

    Raises:
        BatchedPrefillUnsupportedError: cache layers do not implement
            ``merge``/``extract`` (caller should fall back to per-slot
            :func:`prefill`).
        ValueError: ``len(prompt_tokens_list) != len(caches_list)`` or any
            prompt has fewer than 2 tokens (need at least 1 prefill +
            1 seed token).
    """
    if len(prompt_tokens_list) != len(caches_list):
        raise ValueError(
            f"prompt_tokens_list ({len(prompt_tokens_list)}) and caches_list "
            f"({len(caches_list)}) must have the same length"
        )
    if not prompt_tokens_list:
        return 0.0, 0
    if any(int(p.size) < 2 for p in prompt_tokens_list):
        raise ValueError(
            "batched_prefill requires every prompt to have length >= 2 "
            "(1 token to prefill + 1 token for the decode seed)"
        )

    # Slice off the decode seed so the post-prefill cache offset lands at
    # ``len(prompt) - 1`` per slot — same invariant ``mlx_generate``'s
    # exact-prefix-hit branch produces.
    prefill_tokens: list[list[int]] = [
        [int(t) for t in cast(list[int], p[:-1].tolist())] for p in prompt_tokens_list
    ]
    total_tokens = sum(len(p) for p in prefill_tokens)
    if total_tokens == 0:
        return 0.0, 0

    batch_size = len(prefill_tokens)
    uids = list(range(batch_size))

    start_time = time.perf_counter()

    try:
        batch: object = PromptProcessingBatch(
            model=model,
            uids=uids,
            caches=[list(c) for c in caches_list],
            prefill_step_size=prefill_step_size,
        )
    except ValueError as e:
        # ``_merge_caches`` raises ``ValueError`` for cache types without
        # a ``merge`` method. Surface as a typed unsupported error so the
        # caller can fall back cleanly.
        raise BatchedPrefillUnsupportedError(
            f"cache layer does not support batching: {e}"
        ) from e

    logger.debug(
        f"Batched prefill: {batch_size} slots, "
        f"lengths={[len(p) for p in prefill_tokens]}, total={total_tokens}"
    )
    try:
        # ``PromptProcessingBatch.prompt`` does the right-padding +
        # chunked forward internally; one call processes all K
        # sequences in lock-step.
        batch.prompt(prefill_tokens)  # type: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    except Exception as e:
        # Convert mlx-internal failures (e.g. shape mismatches between
        # ``prepare(right_padding=...)`` and the model's attention
        # implementation) into the typed unsupported error so the
        # caller falls back to per-slot prefill instead of taking the
        # whole runner down.
        raise BatchedPrefillUnsupportedError(
            f"PromptProcessingBatch.prompt() raised during batched prefill: {e!r}"
        ) from e

    if on_progress is not None:
        max_len = max(len(p) for p in prefill_tokens)
        on_progress(max_len, max_len)

    # Re-extract per-sequence caches and update the original cache lists
    # in place. Each ``extract_cache(idx)`` produces fresh per-layer
    # cache objects of the original (non-batched) type with the
    # post-prefill state for sequence ``idx``; we overwrite the
    # caller-supplied list contents so any references the caller still
    # holds (e.g. the SequentialGenerator's per-slot ``caches`` ref)
    # see the new state.
    for idx, original_cache in enumerate(caches_list):
        extracted = cast(list[object], batch.extract_cache(idx))
        if len(extracted) != len(original_cache):
            raise BatchedPrefillUnsupportedError(
                f"extract_cache({idx}) returned {len(extracted)} layers, "
                f"original cache has {len(original_cache)}"
            )
        for i, layer in enumerate(extracted):
            original_cache[i] = layer  # type: ignore[index]

    elapsed = time.perf_counter() - start_time
    aggregate_tps = total_tokens / elapsed if elapsed > 0 else 0.0
    logger.debug(
        f"Batched prefill complete: {batch_size} slots, "
        f"{total_tokens} tokens in {elapsed:.2f}s "
        f"({aggregate_tps:.1f} tok/s aggregate)"
    )
    return aggregate_tps, total_tokens


def resolve_speculative_decoding(
    draft_model: Model | None,
    group: mx.distributed.Group | None,
    max_tokens: int,
    num_draft_tokens: int | None,
    drafter_min_output_tokens: int | None,
) -> tuple[Model | None, dict[str, object]]:
    """Decide whether to actually use speculative decoding for this request.

    Pure helper so we can unit-test the policy without spinning up MLX. Returns
    ``(effective_draft_model, spec_kwargs)`` for forwarding to
    ``stream_generate``.

    Policy:
    - Distributed runs: drafter is dropped (mlx_lm does not pipe the drafter
      through the multi-device path yet).
    - Single-device + drafter + ``max_tokens <= drafter_min_output_tokens``:
      drafter is dropped (item 8 -- short outputs don't amortise the prefill
      cost).
    - Single-device + drafter active: forward ``num_draft_tokens`` (item 1)
      via kwargs so ``speculative_generate_step`` honors it.
    """
    if group is not None or draft_model is None:
        return None, {}

    if (
        drafter_min_output_tokens is not None
        and max_tokens <= drafter_min_output_tokens
    ):
        logger.debug(
            f"Short generation (max_tokens={max_tokens} <= "
            f"{drafter_min_output_tokens}); skipping drafter for this request."
        )
        return None, {}

    spec_kwargs: dict[str, object] = {}
    if num_draft_tokens is not None:
        spec_kwargs["num_draft_tokens"] = num_draft_tokens
    return draft_model, spec_kwargs


def _spec_drafter_prefill(
    drafter: Model,
    drafter_cache: KVCacheType,
    tokens: mx.array,
    step: int = 4096,
) -> None:
    """Advance ``drafter_cache`` by running ``drafter`` on ``tokens``.

    Used on the speculative-decoding path to bring the drafter cache to the
    same offset as the target cache before stream_generate's
    ``speculative_generate_step._prefill`` ingests the final two prompt
    tokens. Without this, the drafter cache would be empty (or stuck at a
    prefix-cache hit boundary) while the target cache is at ``prompt - 2``,
    desyncing mlx_lm's spec bookkeeping.
    """
    if tokens.size == 0:
        return
    y = tokens
    while y.size > 0:
        n = min(step, y.size)
        drafter(y[:n][None], cache=drafter_cache)
        mx.eval([c.state for c in drafter_cache])  # type: ignore[reportArgumentType]
        y = y[n:]


def warmup_inference(
    model: Model,
    tokenizer: TokenizerWrapper,
    group: mx.distributed.Group | None,
    model_id: ModelId,
    draft_model: Model | None = None,
    num_draft_tokens: int | None = None,
    drafter_min_output_tokens: int | None = None,
) -> int:
    """Run a throwaway generation to JIT-compile kernels and prime caches.

    When ``draft_model`` is supplied (single-device only), the drafter
    participates in the warmup so the *first real request* doesn't pay a
    cold-cache penalty on the drafter's first speculative step. This is
    item 3 from the drafter tuning plan.

    ``num_draft_tokens`` is the runner's effective K and ``drafter_min_output_tokens``
    is the runner's short-skip threshold. Codex flagged (P2, PR #19 round-(N+10),
    generate.py:525) that omitting these forwards meant the warmup `mlx_generate`
    call dropped to the implicit fallback K=1 and never exercised the
    short-skip gate. Because mlx_lm's speculative_generate_step builds
    a verify-graph keyed on K, the first real request at K>1 paid the
    JIT/graph setup cost the warmup was supposed to absorb. Threading
    K through here ensures the warmed verify-graph matches production
    decoding shape; warmup max_tokens is also boosted above the
    short-skip threshold so the drafter actually fires (the threshold
    demotes the warmup itself to draft_mode='none' otherwise).
    """
    logger.info(f"warming up inference for instance: {model_id}")

    content = InputMessageContent(
        "Prompt to warm up the inference engine. Repeat this."
    )

    # Codex P2 (PR #19 round-(N+10)): the warmup must request at least
    # one token *more* than the short-skip threshold so the drafter
    # actually engages during warmup. ``mlx_generate`` demotes
    # draft_mode -> 'none' for any request whose ``max_output_tokens``
    # is at or below ``drafter_min_output_tokens``, so a too-low
    # warmup token budget would silently bypass the very kernels we
    # just plumbed K into. Use a generous floor either way (we only
    # care about JIT compile cost; the actual decoded text is thrown
    # away by the caller).
    warmup_max_output_tokens = 50
    if drafter_min_output_tokens is not None:
        warmup_max_output_tokens = max(
            warmup_max_output_tokens, drafter_min_output_tokens + 1
        )

    warmup_task_params = TextGenerationTaskParams(
        model=model_id,
        input=[InputMessage(role="user", content=content)],
        max_output_tokens=warmup_max_output_tokens,
        temperature=0.0,
    )

    warmup_prompt = apply_chat_template(
        tokenizer=tokenizer,
        task_params=warmup_task_params,
    )

    tokens_generated = 0

    mx_barrier(group)

    logger.info(
        "Generating warmup tokens"
        + (" (with drafter)" if draft_model is not None else "")
    )

    t = time.monotonic()

    for _r in mlx_generate(
        model=model,
        tokenizer=tokenizer,
        task=warmup_task_params,
        prompt=warmup_prompt,
        kv_prefix_cache=None,
        group=group,
        draft_model=draft_model,
        num_draft_tokens=num_draft_tokens,
        drafter_min_output_tokens=drafter_min_output_tokens,
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

    # Marks the processor as not dependent on the running token history,
    # so the speculative-decoding verify loop can apply it once to a
    # batched ``(K+1, vocab)`` logits tensor and sample all positions
    # in a single host-device sync. Stateful processors (e.g. repetition
    # penalty) leave this attribute unset and force the per-position
    # path.
    proc.position_independent = True  # type: ignore[reportAttributeAccessIssue]
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


def _resolve_coupled_drafter_telemetry(
    *,
    coupled_dispatch_fired: bool,
    coupled_drafter: CoupledDrafter | None,
    effective_num_draft_tokens: int,
) -> tuple[str | None, Literal["mtp", "dflash"] | None, int | None]:
    """Compute coupled-drafter telemetry fields for ``GenerationStats``.

    Codex P2 (PR #25 round-(N+1), generate.py:1710): the dispatch in
    :func:`mlx_generate` builds a :class:`CoupledModelDrafter` only
    when the loaded coupled drafter's ``kind`` is one we know how to
    drive (``"mtp"`` today; ``"dflash"`` lands in a follow-up). When
    the kind is not yet wired, dispatch falls back to
    ``make_drafter(mode="none")`` and runs no speculation. The
    telemetry stamper here gates on the DISPATCH signal
    (``coupled_dispatch_fired``) rather than the RESOURCE signal
    (``coupled_drafter`` present + active mode), so a loaded-but-
    not-dispatched coupled drafter doesn't leak ``drafter_model_id``
    / ``drafter_kind`` / ``num_draft_tokens`` onto a request that
    actually ran with ``draft_mode="none"``.

    Returns a 3-tuple ``(drafter_id, drafter_kind, num_draft_tokens)``
    where any element is ``None`` iff coupled dispatch did not fire
    for this request. Callers fall through to the standard /
    pipelined / ngram telemetry branches when this returns
    ``(None, None, None)``.
    """
    if coupled_dispatch_fired and coupled_drafter is not None:
        return (
            str(coupled_drafter.model_id),
            coupled_drafter.kind,
            effective_num_draft_tokens,
        )
    return (None, None, None)


def _request_is_greedy_sampling(task: TextGenerationTaskParams) -> bool:
    """Return ``True`` iff the request samples deterministically.

    Codex P1 (PR #19 round-(N+4)): the n-gram speculative loop's
    ``target == draft`` accept rule is only distribution-correct
    under greedy decoding (``argmax`` sampling). The MLX
    ``make_sampler`` returns ``argmax`` whenever ``temp == 0.0``;
    other params (``top_p``, ``top_k``, ``min_p``) only affect the
    distribution when temperature > 0, so the temperature gate is
    sufficient to identify "deterministic / argmax-equivalent"
    sampling.

    A request that omits temperature (``task.temperature is None``)
    inherits the runner default of 0.7 (see ``make_sampler`` call
    site), which is non-greedy.
    """
    return task.temperature == 0.0


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
    draft_model: Model | None = None,
    drafter_kv_prefix_cache: KVPrefixCache | None = None,
    drafter_model_id: ModelId | None = None,
    num_draft_tokens: int | None = None,
    drafter_min_output_tokens: int | None = None,
    asymmetric_drafter_rank: int | None = None,
    asymmetric_drafter_transport: object | None = None,
    target_peer_fanout: object | None = None,
    precomputed_target_cache: KVCacheType | None = None,
    coupled_drafter: CoupledDrafter | None = None,
) -> Generator[GenerationResponse]:
    """Generate tokens for ``task``.

    The ``precomputed_target_cache`` argument is the seam used by
    :class:`SequentialGenerator._start_batch` to inject a target-side cache
    that has already been prefilled (typically via :func:`batched_prefill`
    across multiple in-flight requests on a single GPU). When supplied:

    * the prefix-cache lookup is bypassed entirely (we don't pollute the
      shared ``KVPrefixCache`` with per-request entries — V1 trade-off);
    * the local :func:`prefill` call is a no-op (its prompt slice is
      length 0);
    * cache offset is assumed to be ``len(all_prompt_tokens) - 1`` so the
      decode loop seeds from the last prompt token (identical shape to
      the existing ``is_exact_hit`` path of ``KVPrefixCache.get_kv_cache``).

    Eligibility is enforced by the caller — see
    :meth:`SequentialGenerator._batch_eligible_for_prefill`.
    """
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
    if is_bench and not task.use_prefix_cache:
        kv_prefix_cache = None

    # Resolve drafting strategy up-front so cache setup below can branch on
    # the *effective* mode rather than the unfiltered ``draft_model``.
    # Precedence: per-request ``draft_mode`` > per-request ``use_drafter`` >
    # ``EXO_DRAFT_MODE`` env var > implicit default (``model`` if a drafter
    # is loaded, else ``none``). Distributed runs always degrade to
    # ``none`` because mlx_lm does not yet route either model-drafter or
    # n-gram drafting through the pipeline-parallel path.
    request_use_drafter = task.use_drafter
    request_num_draft_tokens = task.num_draft_tokens
    request_draft_mode = task.draft_mode
    effective_num_draft_tokens = (
        request_num_draft_tokens
        if request_num_draft_tokens is not None
        else num_draft_tokens
    ) or 0
    max_tokens = task.max_output_tokens or MAX_TOKENS
    # ``asymmetric_drafter_rank`` is set on every target rank in an
    # asymmetric placement (it's a property of the placement, not of
    # any one rank). ``asymmetric_drafter_transport`` is set only on
    # the target root rank (rank 0 of the target subgroup), which owns
    # the socket to the drafter. Both ranks must enter the pipelined
    # branch because they need to make matching TP collectives every
    # round; the non-root rank consumes drafts via a rank-0 broadcast
    # on the target subgroup (see :class:`PipelinedModelDrafter`).
    # Codex P1 (PR #20 round-(N+10), generate.py:949): per-request
    # ``draft_mode`` overrides MUST flow through to the asymmetric
    # path. Pre-fix the asymmetric branch hard-coded ``"pipelined"``,
    # so a client asking for ``draft_mode="none"`` (warmup, A/B
    # opt-out) silently got pipelined spec decoding anyway.
    # ``resolve_asymmetric_draft_mode`` honors the same precedence as
    # ``resolve_draft_mode`` but uses ``"pipelined"`` as the implicit
    # default for an asymmetric placement.
    has_asymmetric_drafter = asymmetric_drafter_rank is not None
    # Coupled drafters are valid on every collocated placement -- the
    # drafter consumes the target's hidden state in-process, which is
    # cheap to produce on single-device and on symmetric tensor-
    # parallel (the post-all-reduce hidden state is identical on every
    # rank, so each rank can drive its replicated drafter
    # independently and the per-rank ``mx.random`` lockstep produces
    # the same bonus tokens). The asymmetric drafter placement, by
    # contrast, would have to ship full hidden tensors / KV cache
    # entries cross-node every round and lose the speedup; the loader
    # never produces a coupled drafter on that path so the
    # ``has_asymmetric_drafter`` exclusion is belt-and-braces.
    coupled_drafter_eligible: bool = (
        coupled_drafter is not None and not has_asymmetric_drafter
    )
    if has_asymmetric_drafter:
        draft_mode: DraftMode = resolve_asymmetric_draft_mode(
            has_asymmetric_drafter=True,
            request_use_drafter=request_use_drafter,
            request_draft_mode=request_draft_mode,
        )
    elif group is not None and not coupled_drafter_eligible:
        # Multi-device standard drafters still can't ride the TP /
        # pipeline-parallel path -- ``stream_generate`` doesn't thread
        # the secondary ``draft_model`` through ``group``, and n-gram
        # drafting hasn't been wired through the broadcast either. We
        # narrow on ``coupled_drafter_eligible`` because the coupled
        # path is self-contained on each rank: every TP rank loads a
        # replicated drafter and consumes the post-all-reduce hidden
        # state in-process, so no extra group routing is needed.
        draft_mode = "none"
    else:
        draft_mode = resolve_draft_mode(
            has_drafter_model=draft_model is not None,
            request_use_drafter=request_use_drafter,
            request_draft_mode=request_draft_mode,
            has_coupled_drafter=coupled_drafter_eligible,
        )
    # Provisional gate -- re-narrowed after the short-output and
    # ngram-greedy demotions below so a per-request opt-out (or any
    # demotion that flips ``draft_mode`` to ``"none"``) reliably
    # disables the wire. The final ``asymmetric_drafter_active`` is
    # computed once both demotions have run.
    asymmetric_drafter_requested = has_asymmetric_drafter and draft_mode == "pipelined"
    # Item 8: short-output skip applies to drafter-model paths
    # (``"model"`` and ``"pipelined"``) where the drafter prefill cost
    # dominates. N-gram drafting has no prefill (microsecond suffix-
    # match per round) so the threshold is irrelevant; baseline non-
    # spec wouldn't be cheaper anyway.
    if (
        draft_mode in ("model", "pipelined")
        and drafter_min_output_tokens is not None
        and max_tokens <= drafter_min_output_tokens
    ):
        logger.info(
            f"draft_mode demoted to 'none' for short request "
            f"(max_tokens={max_tokens} <= {drafter_min_output_tokens})"
        )
        draft_mode = "none"
    # Codex P1 (PR #19 round-(N+4), drafter.py:692): the n-gram
    # speculative loop accepts drafts via exact token equality and
    # emits the target token at first mismatch. That rule is only
    # distribution-correct under greedy decoding (temperature == 0):
    # for stochastic sampling, accepting/rejecting on equality changes
    # the model's output distribution because we never apply the
    # rejection-sampling correction (``min(1, p_target / p_draft)``)
    # that proper speculative sampling requires. Demote ``"ngram"`` to
    # ``"none"`` whenever the request would sample non-greedily so
    # users don't silently lose reproducibility/quality. Model-mode
    # drafting goes through ``mlx_lm.speculative_generate_step`` which
    # already implements rejection sampling, so it stays unchanged.
    if draft_mode == "ngram" and not _request_is_greedy_sampling(task):
        logger.info(
            f"draft_mode demoted from 'ngram' to 'none' for non-greedy "
            f"sampling (temperature={task.temperature!r}); "
            f"n-gram drafting only preserves the output distribution "
            f"under greedy decoding"
        )
        draft_mode = "none"

    # The remote-transport setup paths below (session opening,
    # PipelinedModelDrafter wiring, drafter-rank lifecycle tasks) only
    # make sense when the resolved mode actually uses the remote
    # drafter. Codex P2 (PR #20 round-(N+1)): pre-fix, the setup ran
    # whenever an asymmetric drafter was *configured*, even after
    # ``draft_mode`` was demoted to ``"none"`` (short-output skip,
    # n-gram-greedy-only demotion, or per-request override). That
    # defeated the demotion path by adding a socket round-trip and a
    # drafter ``reset_and_prefill`` call to every "skip the drafter"
    # request. Compute ``asymmetric_drafter_active`` AFTER the
    # short-output skip and the ngram-greedy demotion so both
    # demotions flow through correctly; we AND with the earlier
    # rank/use-drafter gate (``asymmetric_drafter_requested``) so a
    # per-request opt-out still disables the wire.
    asymmetric_drafter_active = (
        asymmetric_drafter_requested and draft_mode == "pipelined"
    )
    asymmetric_drafter_is_root = (
        asymmetric_drafter_active and asymmetric_drafter_transport is not None
    )
    effective_draft_model = (
        draft_model if draft_mode in ("model", "pipelined") else None
    )
    # Coupled (mtp/dflash) drafter dispatch: active only on single-node
    # placements where the loader produced a coupled drafter, the
    # resolved draft_mode would have used a sibling LM (i.e. ``"model"``),
    # and the request didn't opt out via ``draft_mode="none"`` /
    # ``use_drafter=False`` (both paths flip ``draft_mode`` to a non-
    # ``"model"`` value above). Coupled drafters do NOT use ``draft_model``
    # / ``drafter_caches`` -- they consume the target's hidden + shared KV
    # in-place via :class:`CoupledModelDrafter`'s captured-prefill path --
    # so ``spec_active`` stays ``False`` here on the coupled path and the
    # standard drafter-cache bookkeeping below is correctly skipped.
    coupled_drafter_active = coupled_drafter_eligible and draft_mode == "model"
    # Reused below: drafter-model paths need paired drafter caches; the
    # ngram and none paths don't. The variable name is preserved for
    # readability with the existing cache bookkeeping code below.
    spec_active = (
        draft_mode in ("model", "pipelined") and effective_draft_model is not None
    )
    if effective_num_draft_tokens < 1:
        # Defaulted to 0 above when the runner didn't pre-resolve K and the
        # request didn't override either. Clamp to 1 so n-gram and model
        # drafters don't crash on zero-K proposals.
        effective_num_draft_tokens = 1

    if asymmetric_drafter_is_root and asymmetric_drafter_transport is not None:
        # Only the root has access to the transport's clamp; we then
        # propagate the clamped K to every non-root target rank below
        # so all ranks agree on the wire-format slot count
        # (``k + 1``) used by ``_broadcast_drafts`` /
        # ``_broadcast_target_tokens``. Pre-fix non-root target ranks
        # used the unclamped ``effective_num_draft_tokens`` to size
        # their PipelinedModelDrafter, so a per-request override
        # above the transport budget would have desynchronized the
        # spec-decode collectives (Codex P1, PR #20 round 3).
        #
        # The clamp helper accepts the ``HasNumDraftTokens`` Protocol,
        # so it works on both :class:`DrafterTransport` (in-process
        # path: the transport itself is the session) and
        # :class:`RemoteTransport` (production asymmetric path: the
        # session factory). Pre-fix the call site only invoked the
        # clamp for ``DrafterTransport``, so production
        # ``RemoteTransport`` placements silently skipped the clamp
        # and oversized per-request K landed in ``forward(...)``,
        # raising ``ValueError`` and killing the request (Codex P1,
        # PR #20 round 5 / PR #21 round-(N+10)). Both transport
        # kinds share the same ``num_draft_tokens`` property so the
        # Protocol-based helper works for either.
        from exo.worker.engines.mlx.generator.drafter_transport import (
            HasNumDraftTokens,
            clamp_num_draft_tokens_to_transport,
        )

        if isinstance(asymmetric_drafter_transport, HasNumDraftTokens):
            clamped_k, was_clamped = clamp_num_draft_tokens_to_transport(
                effective_num_draft_tokens, asymmetric_drafter_transport
            )
            if was_clamped:
                logger.warning(
                    f"clamping num_draft_tokens={effective_num_draft_tokens} "
                    f"to transport max={clamped_k} "
                    f"(request_num_draft_tokens={request_num_draft_tokens}); "
                    f"raise EXO_NUM_DRAFT_TOKENS at runner startup to widen "
                    f"the wire-protocol budget"
                )
            effective_num_draft_tokens = clamped_k

    # Codex P1 (PR #20 round 3, generate.py): only the root target rank
    # holds the transport (and therefore the only rank that can clamp);
    # non-root ranks must adopt the (potentially-clamped) value via a
    # broadcast so all target ranks agree on the wire-format slot count
    # used by ``_broadcast_drafts`` / ``_broadcast_target_tokens``.
    if asymmetric_drafter_active and group is not None and group.size() > 1:
        effective_num_draft_tokens = _broadcast_clamped_num_draft_tokens(
            effective_num_draft_tokens=effective_num_draft_tokens,
            group=group,
        )

    prefix_hit_length = 0
    matched_index: int | None = None
    is_exact_hit = False
    if precomputed_target_cache is not None:
        # External batched-prefill path: caller supplies a cache already
        # filled to ``len(all_prompt_tokens) - 1`` and we leave a single
        # decode-seed token in ``prompt_tokens``. ``prefill()`` below
        # short-circuits because the slice ``prompt_tokens[:-1]`` is
        # empty; the prefix-cache update path is also skipped because
        # ``matched_index`` stays None and ``is_exact_hit`` stays False.
        caches = precomputed_target_cache
        prompt_tokens = all_prompt_tokens[-1:]
        prefix_hit_length = int(all_prompt_tokens.size) - 1
    elif kv_prefix_cache is None:
        caches = make_kv_cache(model=model)
        prompt_tokens = all_prompt_tokens
    else:
        caches, prompt_tokens, matched_index, is_exact_hit = (
            kv_prefix_cache.get_kv_cache(
                model,
                all_prompt_tokens,
                media_regions=media_regions,
            )
        )
        prefix_hit_length = len(all_prompt_tokens) - len(prompt_tokens)
        if prefix_hit_length > 0:
            logger.info(
                f"KV cache hit: {prefix_hit_length}/{len(all_prompt_tokens)} tokens cached ({100 * prefix_hit_length / len(all_prompt_tokens):.1f}%)"
            )

    # Drafter cache lookup. We mirror the target's prefix-cache contract on
    # the drafter so multi-turn workloads don't pay the drafter's prefill
    # cost on every request (item 6). The aligned_hit logic below ensures
    # both caches start the spec loop at the same offset; mismatched
    # offsets would corrupt mlx_lm's spec_step bookkeeping.
    drafter_caches: KVCacheType = []
    drafter_matched_index: int | None = None
    if spec_active and effective_draft_model is not None:
        if drafter_kv_prefix_cache is None:
            drafter_caches = make_kv_cache(model=effective_draft_model)
            drafter_remaining = all_prompt_tokens
        else:
            (
                drafter_caches,
                drafter_remaining,
                drafter_matched_index,
                _,
            ) = drafter_kv_prefix_cache.get_kv_cache(
                effective_draft_model,
                all_prompt_tokens,
                media_regions=media_regions,
            )
        target_hit = prefix_hit_length
        drafter_hit = len(all_prompt_tokens) - len(drafter_remaining)
        aligned_hit = min(target_hit, drafter_hit)
        # Trim whichever cache overshoots so both start at ``aligned_hit``.
        if target_hit > aligned_hit:
            mlx_trim_prompt_cache(cast(list[object], caches), target_hit - aligned_hit)  # type: ignore[reportArgumentType]
            prompt_tokens = all_prompt_tokens[aligned_hit:]
            prefix_hit_length = aligned_hit
            if matched_index is not None and aligned_hit < target_hit:
                # Trimming below the prior match invalidates the
                # update-in-place path; treat as a fresh add.
                matched_index = None
                is_exact_hit = False
        if drafter_hit > aligned_hit:
            mlx_trim_prompt_cache(
                cast("list[Any]", drafter_caches),
                drafter_hit - aligned_hit,
            )
            if drafter_matched_index is not None and aligned_hit < drafter_hit:
                drafter_matched_index = None

    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] = (
        make_logits_processors(
            repetition_penalty=task.repetition_penalty,
            repetition_context_size=task.repetition_context_size
            if task.repetition_context_size is not None
            else 20,
            presence_penalty=task.presence_penalty,
            frequency_penalty=task.frequency_penalty,
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
            model,
            vision.embeddings,
            prefix_hit_length,
            len(prompt_tokens) - 1,
            image_token_id=vision.image_token_id,
        )
        if vision is not None
        else contextlib.nullcontext()
    )
    use_remote = (
        len(prompt_tokens) > REMOTE_PREFILL_MIN_TOKENS
        and task.prefill_endpoint is not None
        and not spec_active
    )
    remote_prefilled = False
    prefill_tps = 0.0
    prefill_tokens = 0
    ssm_snapshots_list: list[CacheSnapshot] = []
    with maybe_vision_ctx:
        if use_remote and task.prefill_endpoint is not None:
            try:
                prefill_tps, prefill_tokens, ssm_snapshots_list = remote_prefill(
                    prompt_tokens[:-1],
                    caches,
                    on_prefill_progress,
                    endpoint=task.prefill_endpoint,
                    request_id=str(uuid.uuid4()),
                    model_id=str(task.model),
                    start_pos=prefix_hit_length,
                )
                remote_prefilled = True
            except Exception:
                logger.opt(exception=True).warning(
                    "Remote prefill failed, falling back to local prefill"
                )
        if not remote_prefilled:
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
        # On the spec path we mirror exo's prefill on the drafter so its
        # cache reaches the same offset as the target's (prompt - 2 after
        # the trim(2) inside exo.prefill). mlx_lm's
        # speculative_generate_step._prefill then advances both caches by
        # 1 (decode_prompt size = 2 -> processes 1 token), arriving at
        # prompt - 1 with ``y = decode_prompt[-1:]`` -- the canonical
        # entry state for the spec loop.
        if spec_active and effective_draft_model is not None:
            drafter_prefill_tokens = prompt_tokens[:-2]
            if drafter_prefill_tokens.size > 0:
                _spec_drafter_prefill(
                    effective_draft_model,
                    drafter_caches,
                    drafter_prefill_tokens,
                )
    cache_snapshots: list[CacheSnapshot] | None = ssm_snapshots_list or None

    if kv_prefix_cache is not None and matched_index is not None and is_exact_hit:
        prefill_tps = kv_prefix_cache.prefill_tps[matched_index]

    if kv_prefix_cache is not None:
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
                all_prompt_tokens,
                caches,
                cache_snapshots,
                restore_pos=prefix_hit_length,
                media_regions=media_regions,
                prefill_tps=prefill_tps,
            )
        else:
            kv_prefix_cache.add_kv_cache(
                all_prompt_tokens,
                caches,
                cache_snapshots,
                media_regions=media_regions,
                prefill_tps=prefill_tps,
            )

    # Drafter prefix cache update (item 6). Snapshot the drafter cache
    # *before* stream_generate starts mutating it so subsequent requests
    # can resume from this prompt boundary instead of replaying the
    # drafter prefill.
    if (
        spec_active
        and drafter_kv_prefix_cache is not None
        and effective_draft_model is not None
    ):
        if (
            drafter_matched_index is not None
            and prefix_hit_length >= min_prefix_hit_length
        ):
            drafter_kv_prefix_cache.update_kv_cache(
                drafter_matched_index,
                all_prompt_tokens,
                drafter_caches,
                None,
                restore_pos=prefix_hit_length,
                media_regions=media_regions,
                prefill_tps=prefill_tps,
            )
        else:
            drafter_kv_prefix_cache.add_kv_cache(
                all_prompt_tokens,
                drafter_caches,
                None,
                media_regions=media_regions,
                prefill_tps=prefill_tps,
            )

    # stream_generate starts from the last 2 tokens; caches already cover
    # prompt[:-2] via exo's prefill + c.trim(2). The non-spec and spec paths
    # share the same entry state -- spec just additionally has the drafter
    # cache pre-aligned to the same offset (see drafter prefill above).
    decode_prompt = prompt_tokens[-2:]

    max_tokens = task.max_output_tokens or MAX_TOKENS
    accumulated_text = ""
    generated_text_parts: list[str] = []
    generation_start_time = time.perf_counter()
    usage: Usage | None = None
    # Speculative decoding telemetry (item 4). `from_draft_count` is the
    # number of tokens stream_generate flagged as drafter-accepted; we report
    # it on the final GenerationStats so dashboards / clients can A/B
    # configurations on real traffic.
    from_draft_count = 0
    logger.info("Starting decode")
    mx_barrier(group)

    # Dispatch to the selected drafting strategy via ``make_drafter``.
    # The factory routes:
    #   * ``"model"``    -> mlx_lm.speculative_generate_step (well-tested upstream)
    #   * ``"pipelined"`` -> custom spec loop with cross-round speculation
    #                       behind a ``DrafterTransport`` (in-process or remote)
    #   * ``"ngram"``    -> in-house n-gram suffix-match spec loop
    #   * ``"none"``     -> plain ``mlx_lm.stream_generate``
    # Per-task session for the asymmetric remote drafter (if active).
    # Opened in the ``if`` branch below; closed in the ``finally`` at
    # the end of the function so a fault, cancellation, or normal
    # completion all funnel through ``session.shutdown()`` and free
    # the drafter rank's per-session KV cache. Without this, every
    # completed request would leak ~50-100 MB of KV cache on the
    # drafter rank until the runner shuts down.
    drafter: Drafter
    asymmetric_session: object | None = None
    # Codex P2.3 (PR #20): explicitly require ``draft_mode != "none"``
    # in addition to ``asymmetric_drafter_active``. Today the active
    # flag is derived as ``... and draft_mode == "pipelined"`` so the
    # check is a tautology, but spelling it out at the session-open
    # site keeps the invariant obvious and prevents a future re-shape
    # of the active-flag formula from silently opening drafter
    # sockets for demoted-to-``"none"`` requests.
    # Codex P2 (PR #25 round-(N+1), generate.py:1710): track whether
    # the coupled-drafter dispatch actually fired (i.e. we constructed
    # a :class:`CoupledModelDrafter` instance for this request) so the
    # telemetry block below can gate on the DISPATCH signal rather
    # than the RESOURCE signal (``coupled_drafter_active``). The
    # resource signal is true whenever the loader produced a coupled
    # drafter and ``draft_mode == "model"``, but a request can still
    # land on the dispatch's "kind not yet wired" fallback (e.g.
    # DFlash today), which routes through ``make_drafter(mode="none")``
    # and runs no speculation. Without an explicit dispatch signal,
    # ``GenerationStats`` would surface ``draft_mode="none"`` together
    # with non-null coupled-drafter telemetry, misattributing the run
    # on draft-performance dashboards.
    coupled_dispatch_fired = False
    if asymmetric_drafter_active and draft_mode != "none":
        assert asymmetric_drafter_rank is not None
        target_subgroup_size = group.size() if group is not None else 1
        from exo.worker.engines.mlx.generator.drafter_transport import (
            DrafterTransport as _DrafterTransport,
        )
        from exo.worker.engines.mlx.generator.remote_drafter import (
            RemoteTransport as _RemoteTransport,
        )

        if asymmetric_drafter_is_root:
            # Target root rank: open a per-request session on the
            # ``RemoteTransport`` wire so concurrent target requests
            # don't interleave OP_FORWARD frames on the same socket.
            # Test fakes pass a bare ``DrafterTransport``; in that
            # singular-task path we use it directly.
            if isinstance(asymmetric_drafter_transport, _RemoteTransport):
                asymmetric_session = asymmetric_drafter_transport.open_session()
                session_transport: object = asymmetric_session
            elif isinstance(asymmetric_drafter_transport, _DrafterTransport):
                session_transport = asymmetric_drafter_transport
            else:
                raise TypeError(
                    "asymmetric_drafter_transport must be a RemoteTransport "
                    "(production asymmetric placement) or a DrafterTransport "
                    "(test fakes); "
                    f"got {type(asymmetric_drafter_transport).__name__}"
                )
            # Sync this request's drafter cache against the prompt before
            # constructing the drafter wrapper. The session sends OP_PREFILL
            # with prompt[:-2] (matching ``_spec_drafter_prefill``'s
            # invariant: align the drafter's offset to ``len(prompt) - 2``
            # so the spec loop's first OP_FORWARD seeds from prompt[-2]).
            _diag_t0 = time.perf_counter()
            _spec_diag(
                f"rank 0: about to materialize prefill_prompt "
                f"via tolist() ({all_prompt_tokens.size} prompt tokens total)"
            )
            prefill_prompt: list[int] = [
                int(t) for t in cast(list[int], all_prompt_tokens[:-2].tolist())
            ]
            _spec_diag(
                f"rank 0: prefill_prompt materialized in "
                f"{(time.perf_counter() - _diag_t0) * 1000:.1f}ms "
                f"(len={len(prefill_prompt)}); about to send OP_PREFILL"
            )
            try:
                _diag_t1 = time.perf_counter()
                cast(_DrafterTransport, session_transport).reset_and_prefill(
                    prefill_prompt
                )
                _spec_diag(
                    f"rank 0: OP_PREFILL ACK received in "
                    f"{(time.perf_counter() - _diag_t1) * 1000:.1f}ms"
                )
                drafter = make_drafter(
                    mode=draft_mode,
                    num_draft_tokens=effective_num_draft_tokens,
                    draft_model=None,
                    draft_cache=None,
                    target_subgroup_size=target_subgroup_size,
                    pipelined_transport=session_transport,
                    target_group=group,
                    target_peer_fanout=target_peer_fanout,
                    is_target_root=True,
                )
            except BaseException:
                # ``make_drafter`` or ``reset_and_prefill`` raised;
                # release the freshly-allocated session so the drafter
                # rank doesn't hold its KV cache forever.
                try:
                    if asymmetric_session is not None:
                        cast(_DrafterTransport, asymmetric_session).shutdown()
                except Exception:
                    logger.opt(exception=True).warning(
                        "asymmetric drafter session shutdown raised "
                        "during error recovery; ignoring"
                    )
                asymmetric_session = None
                raise
        else:
            # Non-root target rank in a multi-target placement: no
            # socket, no session, no drafter prefill (the drafter rank
            # only knows about the root's session). The consumer
            # drafter receives drafts each round via a rank-0
            # broadcast on ``group``; the broadcast is the only
            # cross-rank wire this rank needs.
            assert group is not None and target_subgroup_size > 1, (
                "asymmetric_drafter non-root rank requires a target "
                "subgroup of size > 1 (V1 single-target placements "
                "only have rank 0; this branch should not be reached)"
            )
            drafter = make_drafter(
                mode=draft_mode,
                num_draft_tokens=effective_num_draft_tokens,
                draft_model=None,
                draft_cache=None,
                target_subgroup_size=target_subgroup_size,
                pipelined_transport=None,
                target_group=group,
                target_peer_fanout=target_peer_fanout,
                is_target_root=False,
            )
    elif coupled_drafter_active and coupled_drafter is not None:
        # Coupled-drafter dispatch: build the adapter +
        # ``CoupledModelDrafter`` here rather than threading them through
        # ``make_drafter``. The factory's per-mode wiring is built around
        # standard drafters (``draft_model`` / ``draft_cache`` paired with
        # mlx-lm's spec loop); coupled drafters carry no drafter cache and
        # require the target adapter to be constructed against the live
        # model instance, so a dedicated branch is clearer than overloading
        # ``make_drafter`` with an entirely orthogonal third path.
        #
        # This branch runs on both single-device and tensor-parallel
        # placements. The TP target's per-rank ``__call__`` reduces its
        # output to the full hidden state (via the in-layer
        # ``ShardedToAllLinear`` / ``ShardedMoE`` all-sum), so each
        # rank's replicated drafter consumes an identical hidden state
        # and produces identical draft tokens / bonus samples under the
        # shared ``mx.random.seed(seed)`` set at the top of this
        # generation step.
        from exo.worker.engines.mlx.generator.coupled_drafter import (
            CoupledModelDrafter,
            Gemma4MTPTargetAdapter,
            Qwen3_5DFlashTargetAdapter,
        )
        from exo.worker.engines.mlx.vendor.gemma4_mtp_hooks import (
            resolve_gemma4_text_model,
        )
        from exo.worker.engines.mlx.vendor.qwen3_5_dflash_hooks import (
            resolve_qwen3_5_text_model,
        )

        target_adapter: Gemma4MTPTargetAdapter | Qwen3_5DFlashTargetAdapter
        if coupled_drafter.kind == "mtp":
            # The loader's ``attach_mtp_hooks`` already enforced that
            # ``model`` is a Gemma 4 ``Model`` when a coupled drafter
            # was paired with it. The generator-side mirror walks the
            # multimodal wrapper too -- vision-capable Gemma 4 (e.g.
            # ``gemma-4-26b-a4b-it-4bit``) loads as
            # ``mlx_lm.models.gemma4.Model`` whose ``language_model``
            # is the gemma4_text Model the adapter targets. The ``cast``
            # to ``nn.Module`` for the loaded drafter narrows the
            # ``object``-typed ``CoupledDrafter.model`` slot to the
            # ``nn.Module``-Protocol API ``CoupledModelDrafter``
            # consumes (``bind`` / ``set_shared_kv`` / ``draft_block`` /
            # ``accept_lens``).
            if resolve_gemma4_text_model(model) is None:
                raise TypeError(
                    f"coupled_drafter.kind='mtp' requires a Gemma 4 target; "
                    f"got {type(model).__name__!r}. The loader's "
                    "attach_mtp_hooks gate should have caught this."
                )
            target_adapter = Gemma4MTPTargetAdapter(model)
        else:
            # DFlash branch -- mirrors the MTP branch but resolves a
            # Qwen 3.5 target instead of Gemma 4. The loader's
            # ``attach_dflash_hooks`` is the upstream gate; this
            # check is a defence-in-depth guard against a card
            # declaring ``coupled_drafter.kind='dflash'`` against a
            # non-Qwen 3.5 target slipping through.
            if resolve_qwen3_5_text_model(model) is None:
                raise TypeError(
                    f"coupled_drafter.kind='dflash' requires a Qwen 3.5 target; "
                    f"got {type(model).__name__!r}. The loader's "
                    "attach_dflash_hooks gate should have caught this."
                )
            target_adapter = Qwen3_5DFlashTargetAdapter(model)
        drafter = CoupledModelDrafter(
            target_adapter=target_adapter,
            drafter=cast("nn.Module", coupled_drafter.model),
            kind=coupled_drafter.kind,
            num_draft_tokens=effective_num_draft_tokens,
        )
        coupled_dispatch_fired = True
    else:
        drafter = make_drafter(
            mode=draft_mode,
            num_draft_tokens=effective_num_draft_tokens,
            draft_model=effective_draft_model if spec_active else None,
            draft_cache=drafter_caches if spec_active else None,
        )

    # ``decode_prompt`` is the prefill-tail (last two tokens of the
    # prompt). The cache is already aligned to ``all_prompt_tokens[:-2]``
    # via ``exo.prefill`` + ``trim(2)``; mlx_lm's internal ``_prefill``
    # advances by one more token, then the spec loop seeds from the
    # last. ``full_context_tokens`` is the full prompt so the n-gram
    # drafter can match against the entire history (including
    # prefix-cached portions); other drafters ignore it.
    #
    # Codex P2 (PR #19 round-(N+7), generate.py): only the n-gram
    # drafter consumes ``context_tokens``; ``ModelDrafter`` and
    # ``NoSpecDrafter`` ``del context_tokens`` immediately. Pre-fix
    # we paid the O(N) Python-side ``mx.array.tolist()`` +
    # ``int(...)`` conversion on every request, including
    # non-spec runs and prefix-cache-hit requests where decode work
    # is otherwise small. Build the list only when n-gram drafting
    # is actually selected; pass an empty sequence otherwise.
    if draft_mode == "ngram":
        full_context_tokens: list[int] = [
            int(t) for t in cast(list[int], all_prompt_tokens.tolist())
        ]
    else:
        full_context_tokens = []
    _spec_diag_rank = group.rank() if group is not None else 0
    _spec_diag(
        f"rank {_spec_diag_rank}: about to enter drafter.stream() "
        f"(decode_prompt size={int(decode_prompt.size)}, "
        f"max_tokens={max_tokens}, mode={draft_mode})"
    )

    try:
        for completion_tokens, out in enumerate(
            drafter.stream(
                model=model,
                tokenizer=tokenizer,
                prompt=decode_prompt,
                context_tokens=full_context_tokens,
                prompt_cache=caches,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                prefill_step_size=1,
            ),
            start=1,
        ):
            generated_text_parts.append(out.text)
            accumulated_text += out.text
            if getattr(out, "from_draft", False):
                from_draft_count += 1

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
                # Drafter telemetry: stamp the id whenever speculation
                # actually ran for this request. The asymmetric
                # ``"pipelined"`` path has no in-process draft model
                # (the weights live on the drafter rank), so guarding
                # solely on ``effective_draft_model is not None`` would
                # spuriously zero out telemetry for the very topology
                # the drafter buys us the most. We instead trust
                # ``drafter.mode`` together with the asymmetric flag,
                # which is set iff the placement actually wired a
                # drafter rank into this instance.
                telemetry_drafter_id: str | None = None
                telemetry_k: int | None = None
                # Coupled-drafter telemetry: the loader stamps the
                # drafter's ``model_id`` on ``coupled_drafter`` (Phase 2a)
                # and the dispatch above pinned ``drafter.mode == "model"``
                # so the existing acceptance-fraction code path works
                # unchanged. We surface the architecture separately via
                # ``drafter_kind`` so dashboards can disambiguate
                # standard / mtp / dflash without re-shaping ``draft_mode``.
                telemetry_drafter_kind: Literal["standard", "mtp", "dflash"] | None = (
                    None
                )
                # Coupled-drafter branch (gated on dispatch signal --
                # see :func:`_resolve_coupled_drafter_telemetry`).
                coupled_id, coupled_kind, coupled_k = (
                    _resolve_coupled_drafter_telemetry(
                        coupled_dispatch_fired=coupled_dispatch_fired,
                        coupled_drafter=coupled_drafter,
                        effective_num_draft_tokens=effective_num_draft_tokens,
                    )
                )
                if coupled_id is not None:
                    telemetry_drafter_id = coupled_id
                    telemetry_drafter_kind = coupled_kind
                    telemetry_k = coupled_k
                elif (
                    drafter.mode == "model" and effective_draft_model is not None
                ) or drafter.mode == "pipelined":
                    telemetry_k = effective_num_draft_tokens
                    if drafter_model_id is not None:
                        telemetry_drafter_id = str(drafter_model_id)
                    telemetry_drafter_kind = "standard"
                elif drafter.mode == "ngram":
                    telemetry_k = effective_num_draft_tokens

                # Pull per-round counters from the drafter when it
                # surfaces them. Only the pipelined and coupled drafters
                # do today; ``getattr(..., None)`` keeps this future-
                # proof for drafter implementations that grow a
                # ``metrics()`` method later. ``mlx_lm``'s built-in spec
                # loop doesn't expose proposal counts, so the standard
                # ``"model"`` mode surfaces only ``accepted_draft_tokens``
                # (from the ``from_draft`` flag on each yielded token).
                #
                # Codex P2 (PR #25 round-(N+2), coupled_drafter.py:569):
                # the coupled path emits both accepted drafts AND a
                # verifier bonus token per round but yields a flat token
                # stream with no per-token provenance, so flagging every
                # round-loop emission as ``from_draft=True`` produced
                # ``accepted > proposed`` on high-acceptance runs (full-
                # acceptance round of K drafts emits K+1 tokens, all
                # marked accepted, while proposed counts only K).
                # :class:`CoupledModelDrafter` now reports
                # ``accepted_draft_tokens`` directly (sum of
                # ``drafter.accept_lens``) and emits ``from_draft=False``
                # on every round-loop response, so we prefer the
                # metric over the per-emit tally when the drafter
                # surfaces it.
                drafter_metrics_fn = cast(
                    "Callable[[], dict[str, int]] | None",
                    getattr(drafter, "metrics", None),
                )
                drafter_metrics: dict[str, int] = (
                    drafter_metrics_fn() if drafter_metrics_fn is not None else {}
                )
                proposed = int(drafter_metrics.get("proposed_draft_tokens", 0))
                spec_rounds = int(drafter_metrics.get("spec_decode_rounds", 0))
                accepted_from_metrics: int | None = (
                    int(drafter_metrics["accepted_draft_tokens"])
                    if "accepted_draft_tokens" in drafter_metrics
                    else None
                )
                accepted = (
                    accepted_from_metrics
                    if accepted_from_metrics is not None
                    else from_draft_count
                )

                stats = GenerationStats(
                    prompt_tps=float(prefill_tps or out.prompt_tps),
                    generation_tps=float(out.generation_tps),
                    prompt_tokens=int(prefill_tokens + out.prompt_tokens),
                    generation_tokens=int(out.generation_tokens),
                    peak_memory_usage=Memory.from_gb(out.peak_memory),
                    drafter_model_id=telemetry_drafter_id,
                    accepted_draft_tokens=accepted,
                    proposed_draft_tokens=proposed,
                    spec_decode_rounds=spec_rounds,
                    num_draft_tokens=telemetry_k,
                    draft_mode=drafter.mode,
                    drafter_kind=telemetry_drafter_kind,
                )
                if not stop_matched and out.finish_reason not in get_args(FinishReason):
                    logger.warning(
                        f"Model generated unexpected finish_reason: {out.finish_reason}"
                    )

                # OpenAI-compatible surface for spec-decode telemetry.
                # ``accepted_prediction_tokens`` is OpenAI's term for
                # tokens supplied by a Predicted Output that ended up in
                # the completion -- semantically equivalent to our
                # ``accepted_draft_tokens``. ``rejected_prediction_tokens``
                # is the count of predicted tokens that didn't make it,
                # i.e. drafts that the verifier rejected. We can only
                # populate this when the drafter surfaces a proposal
                # count; otherwise leave it at 0 rather than guess.
                rejected_prediction_tokens = (
                    max(0, proposed - accepted) if proposed > 0 else 0
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
                        reasoning_tokens=0,
                        accepted_prediction_tokens=accepted,
                        rejected_prediction_tokens=rejected_prediction_tokens,
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
                # Per-request generation summary. INFO level because it's
                # one line per completed request -- bounded volume, and
                # the operator absolutely needs visibility into drafter
                # effectiveness without flipping ``-vv``. When the
                # drafter ran, surface acceptance fraction + per-position
                # acceptance rate (when proposal count is available) +
                # rounds + K.
                generation_elapsed = time.perf_counter() - generation_start_time
                generated_tokens = len(generated_text_parts)
                generation_tps = (
                    generated_tokens / generation_elapsed
                    if generation_elapsed > 0
                    else 0.0
                )
                base_msg = (
                    f"Generation complete: prefill {prompt_tokens} tokens @ "
                    f"{prefill_tps:.1f} tok/s, generated {generated_tokens} "
                    f"tokens @ {generation_tps:.1f} tok/s"
                )
                if stats is not None and stats.drafter_model_id is not None:
                    fraction = stats.drafter_acceptance_fraction
                    rate = stats.drafter_acceptance_rate
                    fraction_str = f"{fraction:.1%}" if fraction is not None else "n/a"
                    rate_str = f"{rate:.1%}" if rate is not None else "n/a"
                    drafter_msg = (
                        f", drafter={stats.draft_mode}/"
                        f"{stats.drafter_model_id} "
                        f"K={stats.num_draft_tokens} "
                        f"rounds={stats.spec_decode_rounds} "
                        f"accepted={stats.accepted_draft_tokens}/"
                        f"{stats.proposed_draft_tokens or 'n/a'} "
                        f"(rate={rate_str}, "
                        f"fraction_of_emitted={fraction_str})"
                    )
                else:
                    drafter_msg = ""
                logger.info(base_msg + drafter_msg)
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
    finally:
        # Free the per-request drafter-rank KV cache. ``shutdown`` is
        # idempotent on ``_SessionHandle``; the ``try / except`` is
        # belt-and-suspenders for the rare case where the wire is
        # already torn down (e.g. runner shutdown raced this call).
        if asymmetric_session is not None:
            try:
                from exo.worker.engines.mlx.generator.drafter_transport import (
                    DrafterTransport as _DrafterTransport,
                )

                cast(_DrafterTransport, asymmetric_session).shutdown()
            except Exception:
                logger.opt(exception=True).warning(
                    "asymmetric drafter session shutdown raised; the "
                    "drafter rank will free its session cache on its "
                    "next OP_SHUTDOWN"
                )
