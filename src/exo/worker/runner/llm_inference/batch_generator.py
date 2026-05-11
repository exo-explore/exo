import contextlib
import itertools
import os
import time
from collections import OrderedDict, deque
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from typing import BinaryIO

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.constants import EXO_MAX_CONCURRENT_REQUESTS
from exo.shared.types.chunks import ErrorChunk, GenerationChunk, PrefillProgressChunk
from exo.shared.types.common import ModelId
from exo.shared.types.events import ChunkGenerated, Event
from exo.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    GenerationTask,
    TaskId,
    TextGeneration,
)
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    GenerationResponse,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Engine
from exo.worker.engines.mlx.cache import KVPrefixCache, encode_prompt, make_kv_cache
from exo.worker.engines.mlx.disaggregated.adapter import write_cache_to_wire
from exo.worker.engines.mlx.disaggregated.serve import run_prefill_for_request
from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator
from exo.worker.engines.mlx.generator.generate import (
    BatchedPrefillUnsupportedError,
    PrefillCancelled,
    batched_prefill,
    mlx_generate,
    warmup_inference,
)
from exo.worker.engines.mlx.generator.remote_drafter import RemoteTransport
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.utils_mlx import (
    CoupledDrafter,
    apply_chat_template,
    fix_unmatched_think_end_tokens,
    mx_all_gather_tasks,
    mx_any,
)
from exo.worker.engines.mlx.vision import VisionProcessor
from exo.worker.runner.bootstrap import logger

from .model_output_parsers import apply_all_parsers, map_responses_to_chunks
from .tool_parsers import ToolParser


class GeneratorQueue[T]:
    def __init__(self):
        self._q = deque[T]()

    def push(self, t: T):
        self._q.append(t)

    def gen(self) -> Generator[T | None]:
        while True:
            if len(self._q) == 0:
                yield None
            else:
                yield self._q.popleft()


EXO_RUNNER_MUST_FAIL = "EXO RUNNER MUST FAIL"
EXO_RUNNER_MUST_OOM = "EXO RUNNER MUST OOM"
EXO_RUNNER_MUST_TIMEOUT = "EXO RUNNER MUST TIMEOUT"


def _acceptance_fraction_for_adaptive_k(
    response: GenerationResponse,
) -> float | None:
    """Compute the drafter-acceptance fraction to feed adaptive K, or
    return ``None`` when the response shouldn't update the rolling
    window.

    The rolling window steers the next request's ``num_draft_tokens``
    via :func:`adaptive_num_draft_tokens`, so a misgated sample either
    poisons the controller (a non-spec request contributing 0/N) or
    starves it (a real spec round being silently dropped).

    Eligibility:
      * ``stats.draft_mode in {"model", "ngram", "pipelined"}`` -- the
        request actually ran a speculative loop. The previous gate
        keyed off ``drafter_model_id is not None``, but n-gram
        speculation does NOT load a drafter model (it speculates from
        the in-context suffix), so its responses set
        ``drafter_model_id=None`` and were silently dropped under
        ``EXO_DRAFT_MODE=ngram`` + ``EXO_ADAPTIVE_DRAFT_TOKENS=1``,
        pinning K at the fallback value forever. ``pipelined`` mode
        (asymmetric placement, drafter on a peer rank) emits the same
        ``accepted_draft_tokens`` telemetry as ``model`` and must
        feed the rolling window too -- pre-fix the gate excluded
        ``pipelined`` so V3 socket-transport runs left
        ``adaptive_num_draft_tokens`` permanently pinned to the
        fallback (Codex P2, PR #20 round 5,
        batch_generator.py:111-112).
      * ``stats.generation_tokens > 0`` -- guard the division. Empty
        generations (e.g. immediate stop sequence hit on prefill)
        carry no acceptance signal.

    Returns:
      ``stats.accepted_draft_tokens / stats.generation_tokens`` when
      both gates pass; ``None`` otherwise. ``accepted_draft_tokens``
      is populated identically across ``model``, ``ngram``, and
      ``pipelined`` modes, so the formula is unchanged across
      strategies.
    """
    stats = response.stats
    if stats is None:
        return None
    if stats.draft_mode not in ("model", "ngram", "pipelined"):
        return None
    if stats.generation_tokens <= 0:
        return None
    return stats.accepted_draft_tokens / stats.generation_tokens


# Drafter-tuning env vars. Read once per process at SequentialGenerator
# construction time so every request in this runner sees the same K and
# short-skip threshold (avoids surprises mid-stream).
EXO_NUM_DRAFT_TOKENS = "EXO_NUM_DRAFT_TOKENS"
EXO_DRAFTER_MIN_OUTPUT_TOKENS = "EXO_DRAFTER_MIN_OUTPUT_TOKENS"
EXO_ADAPTIVE_DRAFT_TOKENS = "EXO_ADAPTIVE_DRAFT_TOKENS"  # "1" to enable
DEFAULT_NUM_DRAFT_TOKENS = 5  # purpose-built family pairs hit ~80% acceptance
DEFAULT_DRAFTER_MIN_OUTPUT_TOKENS = 16

# Batched prefill (B>=2 prompts processed in one forward) is the
# remaining lever for slot-1 TTFT on long-prompt mixed traffic. The
# round-robin landed in PR #15 cut slot-1 TTFT 5.2x by interleaving
# decode ticks; the residual 11s outliers in the 6K-token
# long_context_summary bench are entirely sequential per-slot
# prefills. Setting ``EXO_BATCH_PREFILL=0`` disables the optimisation
# (escape hatch for shared-prefix workloads where the per-slot
# prefix-cache hit rate exceeds the batched-forward speedup; see
# ``mlx_generate``'s ``precomputed_target_cache`` docstring for the
# trade-off rationale).
EXO_BATCH_PREFILL = "EXO_BATCH_PREFILL"
# Rolling-window size used by adaptive K. Keep small so the controller is
# responsive to traffic shifts (code completion vs reasoning) without
# oscillating on per-request noise.
ADAPTIVE_K_WINDOW = 8


def adaptive_num_draft_tokens(rolling_fractions: list[float], fallback: int) -> int:
    """Pick K (num_draft_tokens) from a rolling window of acceptance fractions.

    The bands are based on the geometric expectation
    ``(1 - p^(K+1)) / (1 - p)`` from the speculative-decoding literature:
    K=2 is the right call when the drafter is missing, K=4 around 50-75%
    acceptance, K=6 above 75%. Below the warmup threshold (need at least 2
    observations) we fall back to the configured default rather than
    twitching at K=2 on first request.
    """
    if len(rolling_fractions) < 2:
        return fallback
    average = sum(rolling_fractions) / len(rolling_fractions)
    if average < 0.5:
        return 2
    if average < 0.75:
        return 4
    return 6


def parse_env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(f"{name}={raw!r} is not a valid int; falling back to {default}")
        return default
    if value < minimum:
        logger.warning(f"{name}={value} below minimum {minimum}; clamping to {minimum}")
        return minimum
    return value


def _check_for_debug_prompts(task_params: TextGenerationTaskParams) -> None:
    """Check for debug prompt triggers in the input."""
    from exo.worker.engines.mlx.utils_mlx import mlx_force_oom

    if len(task_params.input) == 0:
        return
    prompt = task_params.input[0].content
    if not prompt:
        return
    if EXO_RUNNER_MUST_FAIL in prompt:
        raise Exception("Artificial runner exception - for testing purposes only.")
    if EXO_RUNNER_MUST_OOM in prompt:
        mlx_force_oom()
    if EXO_RUNNER_MUST_TIMEOUT in prompt:
        time.sleep(100)


@dataclass(eq=False)
class SequentialGenerator(Engine):
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    tool_parser: ToolParser | None
    model_id: ModelId
    device_rank: int
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]
    vision_processor: VisionProcessor | None = None
    # Optional draft model for speculative decoding (single-device only).
    # `mlx_generate` itself enforces ``draft_model=None`` whenever ``group is
    # not None``; this field is only ever populated for single-device runners.
    draft_model: Model | None = None
    # Parallel KVPrefixCache for the drafter so multi-turn conversations
    # don't pay drafter prefill on every request. None disables drafter
    # prefix caching (single-shot drafter prefill on every call).
    drafter_kv_prefix_cache: KVPrefixCache | None = None
    # The chosen drafter's ModelId. Used for telemetry (GenerationStats) so
    # dashboards can attribute speedup to a specific drafter.
    draft_model_id: ModelId | None = None
    # Coupled (mtp/dflash) drafter loaded via mlx-vlm. When set,
    # ``draft_model`` is None (the loader picks one or the other).
    # Single-device only -- the coupled wire would have to ship target
    # hidden states / KV cache cross-node, which negates the speedup.
    #
    # Phase 2a invariant: the field is plumbed through the loader and
    # stored here, but the generator does NOT yet dispatch through the
    # coupled-drafter round loop. The follow-up that adds
    # ``rollback_speculative_cache`` + extended forward kwargs to the
    # mlx-lm fork's gemma4_text.py also wires this field into
    # ``mlx_generate`` and only then does it actually drive speculative
    # decoding. Until that lands, this field is read by ``close()``
    # for cleanup ordering and by ``__post_init__``-style validation
    # in tests.
    coupled_drafter: CoupledDrafter | None = None
    # K (num_draft_tokens) for speculative_generate_step. None falls back to
    # the env var EXO_NUM_DRAFT_TOKENS, then DEFAULT_NUM_DRAFTER_TOKENS.
    num_draft_tokens: int | None = None
    # max_output_tokens threshold below which the drafter is skipped per
    # request. None falls back to the env var EXO_DRAFTER_MIN_OUTPUT_TOKENS.
    drafter_min_output_tokens: int | None = None
    # Item 7: when True, K is recomputed each request from a rolling window
    # of observed acceptance fractions. Disabled by default so K stays
    # predictable for benchmarking.
    adaptive_draft_tokens: bool = False
    # Asymmetric placement telemetry: ``drafter_rank_in_parent`` mirrors
    # :attr:`DrafterPlacement.drafter_rank` (advisory only; the drafter
    # is NOT a member of any ``mx.distributed.Group`` under the v3+
    # wire). ``None`` for symmetric/single-device builds. When set
    # together with ``remote_drafter_transport``, every request runs
    # the pipelined+remote drafter path: the spec loop talks to the
    # drafter via the dedicated drafter TCP socket owned by
    # ``RemoteTransport`` rather than ``mx.distributed`` collectives.
    drafter_rank_in_parent: int | None = None
    # Long-lived transport bound to the drafter rank. Allocated once at
    # builder.build() time; reused across requests so the executor
    # thread + drafter cache lifecycle isn't paid per-request. Each
    # in-flight request opens its own session via
    # :meth:`RemoteTransport.open_session`; the per-session handle is
    # the actual ``DrafterTransport`` consumed by the spec loop. Closed
    # in :meth:`close` (sends ``OP_SHUTDOWN`` to the drafter rank).
    remote_drafter_transport: RemoteTransport | None = None
    # Inter-target-rank TCP fanout for spec-decode int broadcasts.
    # Allocated alongside the drafter wire on multi-target asymmetric
    # placements (see :class:`TargetPeerFanout`); ``None`` for
    # single-target / symmetric instances. The runner stores it so the
    # spec-decode loop can sidestep ``mx.distributed.send`` / ``recv``
    # for inter-target int broadcasts -- those collide with the
    # model's TP ``all_sum`` collectives on the JACCL backend and
    # silently corrupt the int wire.
    target_peer_fanout: object | None = None
    check_for_cancel_every: int = 50

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _maybe_queue: list[TextGeneration] = field(default_factory=list, init=False)
    _maybe_cancel: list[TextGeneration] = field(default_factory=list, init=False)
    _all_tasks: dict[TaskId, TextGeneration] = field(default_factory=dict, init=False)
    _queue: deque[TextGeneration] = field(default_factory=deque, init=False)
    # Rolling window of recently-observed drafter-acceptance fractions for
    # adaptive K. Only populated when adaptive_draft_tokens is True.
    _recent_acceptance: deque[float] = field(
        default_factory=lambda: deque(maxlen=ADAPTIVE_K_WINDOW),
        init=False,
    )
    # Maximum number of in-flight tasks the runner will round-robin through
    # in :meth:`step`. Set to 1 by ``builder.build`` whenever the runner
    # owns a long-lived ``RemoteTransport`` (asymmetric pipelined drafter):
    # the wire protocol assumes one in-flight prefill/forward session, so
    # interleaving two target requests on the same socket would corrupt
    # the drafter's per-request state. For all other configurations
    # (no drafter, n-gram drafter, in-process model drafter where every
    # ``mlx_generate`` call allocates its own draft KVCache) this defaults
    # to ``EXO_MAX_CONCURRENT_REQUESTS`` and gives concurrent requests the
    # cooperative-scheduling semantics the dispatcher always claimed but
    # never delivered: prior to this field every spec-config runner pinned
    # ``_active`` to a singular slot and slot 1's TTFT equalled slot 0's
    # full completion time (measured 14s on a K=3 single-host n-gram bench
    # in the PR #15 concurrency leg).
    max_concurrent_tasks: int = 1
    # Currently in-flight tasks, keyed by ``TaskId`` for O(1) cancel/finish.
    # Insertion order is the round-robin order; ``OrderedDict`` makes that
    # preservation explicit (CPython dicts already preserve it but we want
    # the contract to be load-bearing). Capped by ``max_concurrent_tasks``;
    # ``step`` round-robins one ``next(gen)`` call per active task per
    # tick. Each tuple is (task, mlx generator, response queue, parsed-
    # output generator) -- the same shape the previous singular ``_active``
    # slot held, just multiplexed.
    _active_tasks: OrderedDict[
        TaskId,
        tuple[
            TextGeneration,
            # mlx generator that does work
            Generator[GenerationResponse],
            # queue that the 1st generator should push to and 3rd generator should pull from
            GeneratorQueue[GenerationResponse],
            # generator to get parsed outputs
            Iterator[GenerationChunk | None],
        ],
    ] = field(default_factory=OrderedDict, init=False)
    # Tasks that failed during ``_build_generator`` or mid-stream. Drained
    # by ``step`` so per-task failures surface as ``FinishedResponse`` to
    # the caller without taking down the runner subprocess. We accept the
    # rank-desync risk: ``_build_generator`` failures are deterministic
    # in practice (config / per-request K mismatch) so all ranks fail
    # together; any non-deterministic failure was already a desync hazard.
    _pending_failed: list[TaskId] = field(default_factory=list, init=False)

    def warmup(self):
        # Codex P2 (PR #19 round-(N+10), generate.py:525): forward the
        # runner's effective K and short-skip threshold so the warmup
        # path JIT-compiles the same speculative_generate_step shape
        # that production traffic will use. Without this the warmup
        # ran at the implicit K=1 fallback and the first real request
        # at K>1 paid the verify-graph setup cost we meant to absorb.
        self.check_for_cancel_every = warmup_inference(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            model_id=self.model_id,
            draft_model=self.draft_model,
            num_draft_tokens=self.num_draft_tokens,
            drafter_min_output_tokens=self.drafter_min_output_tokens,
        )

    def submit(
        self,
        task: GenerationTask,
    ) -> None:
        assert isinstance(task, TextGeneration)
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._all_tasks[task.task_id] = task
        self._maybe_queue.append(task)

    def agree_on_tasks(self) -> None:
        """Agree between all ranks about the task ordering (some may have received in different order or not at all)."""
        agreed, different = mx_all_gather_tasks(self._maybe_queue, self.group)
        # Extend from `agreed` (sorted by task_id on all ranks) to guarantee every
        # rank enqueues tasks in the same order, preventing TP collective deadlocks.
        self._queue.extend(agreed)
        self._maybe_queue = list(different)

    def agree_on_cancellations(self) -> None:
        """Agree between all ranks about which tasks to cancel."""
        has_cancel_all = False
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                has_cancel_all = True
                continue
            if task_id in self._all_tasks:
                self._maybe_cancel.append(self._all_tasks[task_id])

        if mx_any(has_cancel_all, self.group):
            self._cancelled_tasks.add(CANCEL_ALL_TASKS)

        agreed, different = mx_all_gather_tasks(self._maybe_cancel, self.group)
        self._cancelled_tasks.update(task.task_id for task in agreed)
        self._maybe_cancel = list(different)

    def step(
        self,
    ) -> Iterator[
        tuple[TaskId, GenerationChunk | FinishedResponse | CancelledResponse]
    ]:
        output: list[
            tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]
        ] = []

        # Top up the active set from the queue. ``agree_on_tasks`` is a
        # collective op across the MLX group; we only call it when there
        # might be new work to admit (active set has slack and queue is
        # potentially non-empty after ``agree_on_tasks`` runs). Calling
        # it on every tick is safe but wastes a collective when the
        # active set is already full.
        if len(self._active_tasks) < self.max_concurrent_tasks:
            self.agree_on_tasks()
            self._admit_queued_tasks()

        # Drain failures recorded by ``_start_next`` (this tick or any
        # prior tick that left them queued) so the runner loop marks
        # them complete and proceeds with the next task instead of
        # tearing down the subprocess (regression: K=8 ValueError took
        # the target rank with it on 14:35:05).
        while self._pending_failed:
            output.append((self._pending_failed.pop(0), FinishedResponse()))

        if not self._active_tasks:
            return itertools.chain(
                iter(output),
                map(
                    lambda task: (task, CancelledResponse()),
                    self._cancelled_tasks,
                ),
            )

        # Round-robin one ``next(gen)`` per active task. Each generator
        # owns its own KV cache (``mlx_generate`` allocates fresh caches
        # per request), so interleaving generators per-tick is safe -- the
        # only shared state is the model weights themselves, which are
        # read-only during forward. Snapshot the items so per-task
        # exceptions can ``del self._active_tasks[task_id]`` mid-iteration
        # without invalidating the loop.
        for task_id, (task, gen, queue, output_generator) in list(
            self._active_tasks.items()
        ):
            try:
                response = next(gen)
                queue.push(response)
                # Observe drafter acceptance once the final stats arrive. We
                # do this here (and not in mlx_generate) because the rolling
                # buffer is owned by the generator and must persist across
                # requests for adaptive K to converge.
                if self.adaptive_draft_tokens:
                    fraction = _acceptance_fraction_for_adaptive_k(response)
                    if fraction is not None:
                        self._recent_acceptance.append(fraction)
                # drain potentially many responses every time
                while (parsed := next(output_generator, None)) is not None:
                    output.append((task_id, parsed))

            except (StopIteration, PrefillCancelled):
                output.append((task_id, FinishedResponse()))
                del self._active_tasks[task_id]

            except Exception as e:
                # ALWAYS log first. Without this, an exception silently
                # swallowed on a non-root target rank presents to the
                # operator as "rank 1 returned ready in 0.4 s with no
                # tokens"; the actual error -- which may be a master
                # divergence, an MLX collective desync, or a bad model
                # weights load -- is invisible. Logging is unconditional
                # because the multi-rank re-raise path below also relies
                # on it (the supervisor records the message but not the
                # traceback).
                logger.opt(exception=True).error(
                    "generator.step raised; "
                    f"task_id={task_id} "
                    f"command_id={task.command_id} "
                    f"device_rank={self.device_rank} "
                    f"group_size={self.group.size() if self.group is not None else 1} "
                    f"exc={type(e).__name__}: {e}"
                )

                # Multi-rank targets MUST re-raise. Any exception here
                # (whether a request-level bug or a system-level MLX
                # error) means this rank exited the generator without
                # participating in the verify-forward TP collective the
                # peer rank is now waiting on. Swallowing leaves the
                # peer hung indefinitely; raising hands control to
                # ``handle_generation_tasks`` -> supervisor ->
                # ``RunnerFailed``. The peer's ``_kill_runner`` rule
                # then tears down its own runner via the
                # ``RunnerFailed``-on-peer trigger (see
                # ``worker/plan.py``), the master rebuilds the instance
                # via ``CreateRunner``, and the next request sees a
                # fresh group. Total recovery is bounded by the
                # supervisor escalation chain (~25 s), not "manual
                # operator restart".
                #
                # Single-rank runners keep the legacy swallow path: a
                # malformed request shouldn't crash the (only) runner
                # and break unrelated concurrent tasks sharing the
                # process. With ``max_concurrent_tasks > 1`` a
                # malformed request also must not affect the *other*
                # in-flight tasks sharing this generator.
                if self.group is not None and self.group.size() > 1:
                    self._send_error(task, e)
                    del self._active_tasks[task_id]
                    raise

                self._send_error(task, e)
                del self._active_tasks[task_id]
                output.append((task_id, FinishedResponse()))

        # Top up again if we just retired any task -- keeps slot 1's
        # TTFT independent of slot 0's completion length, which is the
        # whole point of ``max_concurrent_tasks > 1``.
        if self._queue and len(self._active_tasks) < self.max_concurrent_tasks:
            self._admit_queued_tasks()

        return filter(
            lambda chunk: (
                not isinstance(chunk[1], GenerationChunk) or self.device_rank == 0
            ),
            itertools.chain(
                output,
                map(lambda task: (task, CancelledResponse()), self._cancelled_tasks),
            ),
        )

    def _admit_queued_tasks(self) -> None:
        """Top up ``_active_tasks`` from ``_queue``, batching prefill when possible.

        Cooperatively schedules eligible tasks through a single
        :func:`batched_prefill` forward when ``EXO_BATCH_PREFILL`` is on
        (default) and at least 2 tasks pass the eligibility filter
        (``_batch_eligible_for_prefill``). Ineligible tasks (vision,
        remote prefill, in-process model drafter, etc.) and any task
        in a single-eligible-task admit cycle fall back to the
        per-slot :meth:`_start_one` path. Eligibility is read at admit
        time so a request that becomes ineligible mid-tick (e.g.
        because ``EXO_BATCH_PREFILL`` was toggled) cleanly degrades.

        The function never raises; per-task setup failures are routed
        through :meth:`_send_error` + ``_pending_failed`` (same
        liveness contract as :meth:`_start_one`).
        """
        if not self._queue:
            return

        # Drain the queue up to the active-set slack, then partition by
        # batch eligibility. We can't peek-without-pop because
        # ``self._queue`` is a deque drained by the caller, so collect
        # candidates first and re-route into ``_start_one`` if the
        # batch path bails.
        slack = self.max_concurrent_tasks - len(self._active_tasks)
        candidates: list[TextGeneration] = []
        while self._queue and len(candidates) < slack:
            candidates.append(self._queue.popleft())

        if not candidates:
            return

        batch_enabled = os.environ.get(EXO_BATCH_PREFILL, "1") != "0"
        if not batch_enabled:
            for task in candidates:
                self._start_one(task)
            return

        eligible: list[tuple[TextGeneration, mx.array, KVCacheType]] = []
        leftover: list[TextGeneration] = []
        for task in candidates:
            prep = self._prepare_for_batch_prefill(task)
            if prep is None:
                leftover.append(task)
            else:
                eligible.append(prep)

        logger.debug(
            f"_admit_queued_tasks candidates={len(candidates)} "
            f"eligible={len(eligible)} leftover={len(leftover)} "
            f"slack={slack} batch_enabled={batch_enabled}"
        )

        # Single-eligible: a batched forward of size 1 has no parallelism
        # win and adds the PromptBatch + merge_caches overhead, so just
        # take the per-slot path.
        if len(eligible) < 2:
            for task in candidates:
                self._start_one(task)
            return

        prompts = [tup[1] for tup in eligible]
        caches = [tup[2] for tup in eligible]

        try:
            tps, total = batched_prefill(
                model=self.model,
                prompt_tokens_list=prompts,
                caches_list=caches,
            )
            logger.info(
                f"batched_prefill: {len(eligible)} slots, {total} tokens "
                f"({tps:.1f} tok/s aggregate)"
            )
            for task, prompt_tokens, cache in eligible:
                self._emit_prefill_complete(task, prompt_tokens)
                self._start_one(task, precomputed_target_cache=cache)
            for task in leftover:
                self._start_one(task)
            return
        except BatchedPrefillUnsupportedError:
            logger.info(
                "batched_prefill unsupported for this model/cache; "
                "falling back to per-slot prefill"
            )
            for task in candidates:
                self._start_one(task)
            return
        except Exception as e:
            # Untyped failure: charge the error to every batched task so
            # one bad request doesn't take the runner down. ``leftover``
            # tasks were not part of the failed batch and proceed
            # normally on the per-slot path.
            for task, _, _ in eligible:
                self._send_error(task, e)
                self._pending_failed.append(task.task_id)
            for task in leftover:
                self._start_one(task)
            return

    def _start_one(
        self,
        task: TextGeneration,
        *,
        precomputed_target_cache: KVCacheType | None = None,
    ) -> None:
        """Build one slot's generator and add it to ``_active_tasks``.

        ``precomputed_target_cache`` is forwarded to ``mlx_generate`` to
        skip its prefix-cache lookup + local prefill. Set by
        :meth:`_admit_queued_tasks` after a batched prefill; ``None``
        otherwise.
        """
        # Only forward ``precomputed_target_cache`` when it was set so
        # existing test seams that monkeypatch ``_build_generator`` with
        # the legacy ``(self, task)`` signature still work; the per-slot
        # admit path (``precomputed_target_cache is None``) is the
        # default and predates the batched-prefill seam.
        try:
            if precomputed_target_cache is None:
                gen = self._build_generator(task)
            else:
                gen = self._build_generator(
                    task, precomputed_target_cache=precomputed_target_cache
                )
        except Exception as e:
            # Preserve runner liveness: surface the error to the client
            # via ``_send_error`` and queue a ``FinishedResponse`` for
            # ``step`` to drain on the next tick. The active set is
            # unchanged so the next ``step`` either picks up the next
            # queued task or returns idle (instead of asserting and
            # crashing the subprocess).
            self._send_error(task, e)
            self._pending_failed.append(task.task_id)
            return
        queue = GeneratorQueue[GenerationResponse]()

        if task.task_params.bench:
            output_generator: Iterator[GenerationChunk | None] = map(
                lambda r: map_responses_to_chunks(r, self.model_id), queue.gen()
            )
        else:
            output_generator = apply_all_parsers(
                queue.gen(),
                apply_chat_template(self.tokenizer, task.task_params),
                self.tool_parser,
                self.tokenizer,
                type(self.model),
                self.model_id,
                task.task_params.tools,
            )
        self._active_tasks[task.task_id] = (task, gen, queue, output_generator)

    def _batch_eligible_for_prefill(self, task: TextGeneration) -> bool:
        """Return ``True`` when ``task`` can be co-prefilled with peers.

        V1 eligibility is narrow on purpose: only single-rank text-only
        generation without remote prefill or an in-process model
        drafter. The asymmetric pipelined drafter still qualifies
        because ``draft_model`` is ``None`` on the target rank — the
        drafter cache lives on the remote rank and is prefilled per-
        session over the wire, independent of target prefill batching.

        Multi-rank target paths (TP/PP) are excluded because
        :func:`pipeline_parallel_prefill`'s collective semantics need
        per-slot driver loops; a follow-up can lift this once the
        batched forward is folded into the pipeline driver.
        """
        params = task.task_params
        if self.group is not None and self.group.size() > 1:
            return False
        if params.images:
            return False
        if params.prefill_endpoint is not None:
            return False
        # In-process model drafter ("model" mode) needs a paired
        # drafter prefill aligned to the target's offset; batching
        # only the target without batching the drafter would desync
        # them. The asymmetric drafter (``self.draft_model is None``
        # but ``remote_drafter_transport is not None``) is fine
        # because its drafter prefill goes over the wire per-session.
        return self.draft_model is None

    def _prepare_for_batch_prefill(
        self, task: TextGeneration
    ) -> tuple[TextGeneration, mx.array, KVCacheType] | None:
        """Encode the prompt and allocate a fresh cache for batched prefill.

        Returns ``None`` when ``task`` is ineligible or when the
        encoded prompt is too short to leave a decode-seed token
        (length < 2). The encoding mirrors :func:`mlx_generate`'s
        ``encode_prompt`` + ``fix_unmatched_think_end_tokens`` so the
        cache offset agreed by ``batched_prefill`` matches what
        ``mlx_generate`` later sees on the inner side of
        ``precomputed_target_cache``.
        """
        if not self._batch_eligible_for_prefill(task):
            return None
        try:
            prompt_str = apply_chat_template(self.tokenizer, task.task_params)
            prompt_tokens = encode_prompt(self.tokenizer, prompt_str)
            prompt_tokens = fix_unmatched_think_end_tokens(
                prompt_tokens, self.tokenizer
            )
        except Exception:
            # Encoding failure surfaces through the per-slot path so
            # the existing ``_send_error`` plumbing reports it; we
            # don't swallow it here.
            logger.opt(exception=True).warning(
                "Prompt encoding failed during batch-prefill prep; "
                "falling back to per-slot path"
            )
            return None
        if int(prompt_tokens.size) < 2:
            return None
        try:
            cache = make_kv_cache(self.model)
        except Exception:
            logger.opt(exception=True).warning(
                "make_kv_cache failed during batch-prefill prep; "
                "falling back to per-slot path"
            )
            return None
        return (task, prompt_tokens, cache)

    def _emit_prefill_complete(
        self, task: TextGeneration, prompt_tokens: mx.array
    ) -> None:
        """Fire a single ``processed=total`` ``PrefillProgressChunk``.

        ``batched_prefill`` runs as one forward so per-chunk progress
        events would mix slots. We elide intermediate progress and
        emit a single completion event per slot at the end of the
        batched forward so dashboards stop showing 0% prefill.
        """
        if self.device_rank != 0:
            return
        total = int(prompt_tokens.size)
        self.event_sender.send(
            ChunkGenerated(
                command_id=task.command_id,
                chunk=PrefillProgressChunk(
                    model=self.model_id,
                    processed_tokens=total,
                    total_tokens=total,
                ),
            )
        )

    def _send_error(self, task: TextGeneration, e: Exception) -> None:
        if self.device_rank == 0:
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=self.model_id,
                        finish_reason="error",
                        error_message=str(e),
                    ),
                )
            )

    def _build_generator(
        self,
        task: TextGeneration,
        *,
        precomputed_target_cache: KVCacheType | None = None,
    ) -> Generator[GenerationResponse]:
        _check_for_debug_prompts(task.task_params)
        prompt = apply_chat_template(self.tokenizer, task.task_params)

        def on_prefill_progress(processed: int, total: int) -> None:
            if self.device_rank == 0:
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=task.command_id,
                        chunk=PrefillProgressChunk(
                            model=self.model_id,
                            processed_tokens=processed,
                            total_tokens=total,
                        ),
                    )
                )

        def distributed_prompt_progress_callback() -> None:
            self.agree_on_cancellations()
            if self.should_cancel(task.task_id):
                raise PrefillCancelled()

            self.agree_on_tasks()

        tokens_since_cancel_check = self.check_for_cancel_every

        def on_generation_token() -> None:
            nonlocal tokens_since_cancel_check
            tokens_since_cancel_check += 1
            if tokens_since_cancel_check >= self.check_for_cancel_every:
                tokens_since_cancel_check = 0
                self.agree_on_cancellations()
                if self.should_cancel(task.task_id):
                    raise PrefillCancelled()

                self.agree_on_tasks()

        # Adaptive K (item 7): when enabled, recompute K from the rolling
        # window of observed acceptance fractions. The configured value
        # (`self.num_draft_tokens`) is the warmup fallback used until the
        # window has enough data.
        if self.adaptive_draft_tokens and self.num_draft_tokens is not None:
            effective_num_draft_tokens: int | None = adaptive_num_draft_tokens(
                list(self._recent_acceptance), fallback=self.num_draft_tokens
            )
        else:
            effective_num_draft_tokens = self.num_draft_tokens

        # Phase 2c lands the coupled-drafter dispatch: ``mlx_generate``
        # now accepts the loader's ``CoupledDrafter`` and routes through
        # :class:`CoupledModelDrafter` whenever the placement is single-
        # node and the resolved ``draft_mode`` would have used a sibling
        # drafter (i.e. ``"model"``). On asymmetric / multi-rank
        # placements ``mlx_generate`` ignores ``coupled_drafter`` -- the
        # builder gate already steered those topologies to the standard
        # path, but we forward the field unconditionally so the dispatch
        # narrows in one place.
        return mlx_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            task=task.task_params,
            prompt=prompt,
            kv_prefix_cache=self.kv_prefix_cache,
            on_prefill_progress=on_prefill_progress,
            distributed_prompt_progress_callback=distributed_prompt_progress_callback,
            on_generation_token=on_generation_token,
            group=self.group,
            vision_processor=self.vision_processor,
            draft_model=self.draft_model,
            drafter_kv_prefix_cache=self.drafter_kv_prefix_cache,
            drafter_model_id=self.draft_model_id,
            num_draft_tokens=effective_num_draft_tokens,
            drafter_min_output_tokens=self.drafter_min_output_tokens,
            asymmetric_drafter_rank=self.drafter_rank_in_parent,
            asymmetric_drafter_transport=self.remote_drafter_transport,
            target_peer_fanout=self.target_peer_fanout,
            precomputed_target_cache=precomputed_target_cache,
            coupled_drafter=self.coupled_drafter,
        )

    def close(self) -> None:
        if self.remote_drafter_transport is not None:
            try:
                self.remote_drafter_transport.shutdown()
            except Exception:
                # Drafter rank may already be gone (e.g. due to a
                # parallel shutdown of the cluster); log and continue
                # so target-side cleanup isn't blocked on a peer that
                # can't ack. The shutdown call is idempotent so a
                # later retry is harmless.
                logger.opt(exception=True).warning(
                    "Drafter rank shutdown failed; continuing close"
                )
            self.remote_drafter_transport = None
        # Codex P2 (PR #20): drop the drafter model BEFORE the target
        # model so the drafter's KV cache / weights are released while
        # the target group is still alive. Reordering this after
        # ``del self.model, self.tokenizer, self.group`` triggered an
        # ``AttributeError`` chain on multi-rank teardown when the
        # drafter held a weak reference into the target group.
        # Coupled drafters bind to the target's input embeddings via
        # ``bind`` so they hold a stronger reference than the standard
        # drafter; release them first.
        if self.coupled_drafter is not None:
            del self.coupled_drafter
            self.coupled_drafter = None
        if self.draft_model is not None:
            del self.draft_model
            self.draft_model = None
        # Close every TCP socket the target-peer fanout owns (one per
        # peer on rank 0, single rank-zero socket on peers). Inline
        # the socket import + isinstance check to keep this module's
        # top-level imports thin. ``OSError`` here is benign -- the
        # peer may already have closed (e.g. supervisor SIGKILL chain)
        # and we just want to free the local FDs before the runner
        # exits.
        if self.target_peer_fanout is not None:
            from exo.worker.engines.mlx.utils_mlx import TargetPeerFanout as _Fanout

            if isinstance(self.target_peer_fanout, _Fanout):
                import socket as _socket

                for sock in self.target_peer_fanout.peer_sockets.values():
                    if isinstance(sock, _socket.socket):
                        with contextlib.suppress(OSError):
                            sock.close()
                if isinstance(self.target_peer_fanout.rank_zero_socket, _socket.socket):
                    with contextlib.suppress(OSError):
                        self.target_peer_fanout.rank_zero_socket.close()
            self.target_peer_fanout = None
        del self.model, self.tokenizer, self.group

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        cache = run_prefill_for_request(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            kv_prefix_cache=self.kv_prefix_cache,
            request=request,
        )
        write_cache_to_wire(
            wfile,
            cache,
            request_id=request.request_id,
            model_id=request.model_id,
            start_pos=request.start_pos,
        )


@dataclass(eq=False)
class BatchGenerator(Engine):
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    tool_parser: ToolParser | None
    model_id: ModelId
    device_rank: int
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]
    check_for_cancel_every: int = 50
    vision_processor: VisionProcessor | None = None

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _maybe_queue: list[TextGeneration] = field(default_factory=list, init=False)
    _maybe_cancel: list[TextGeneration] = field(default_factory=list, init=False)
    _all_tasks: dict[TaskId, TextGeneration] = field(default_factory=dict, init=False)
    _queue: deque[TextGeneration] = field(default_factory=deque, init=False)
    _gen: ExoBatchGenerator = field(init=False)
    _active_tasks: dict[
        int,
        tuple[
            TextGeneration,
            GeneratorQueue[GenerationResponse],
            Iterator[GenerationChunk | None],
        ],
    ] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._gen = ExoBatchGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            kv_prefix_cache=self.kv_prefix_cache,
            vision_processor=self.vision_processor,
        )

    def warmup(self):
        self.check_for_cancel_every = warmup_inference(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            model_id=self.model_id,
        )

    def submit(
        self,
        task: GenerationTask,
    ) -> None:
        assert isinstance(task, TextGeneration)
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._all_tasks[task.task_id] = task
        self._maybe_queue.append(task)

    def agree_on_tasks(self) -> None:
        """Agree between all ranks about the task ordering (some may have received in different order or not at all)."""
        agreed, different = mx_all_gather_tasks(self._maybe_queue, self.group)
        # Extend from `agreed` (sorted by task_id on all ranks) to guarantee every
        # rank enqueues tasks in the same order, preventing TP collective deadlocks.
        self._queue.extend(agreed)
        self._maybe_queue = list(different)

    def agree_on_cancellations(self) -> None:
        """Agree between all ranks about which tasks to cancel."""
        has_cancel_all = False
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                has_cancel_all = True
                continue
            if task_id in self._all_tasks:
                self._maybe_cancel.append(self._all_tasks[task_id])

        if mx_any(has_cancel_all, self.group):
            self._cancelled_tasks.add(CANCEL_ALL_TASKS)

        agreed, different = mx_all_gather_tasks(self._maybe_cancel, self.group)
        self._cancelled_tasks.update(task.task_id for task in agreed)
        self._maybe_cancel = list(different)

    def step(
        self,
    ) -> Iterator[
        tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]
    ]:
        if not self._queue:
            self.agree_on_tasks()

        output: list[
            tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]
        ] = []

        # Submit any queued tasks to the engine
        while self._queue and len(self._active_tasks) < EXO_MAX_CONCURRENT_REQUESTS:
            task = self._queue.popleft()
            try:
                prompt = apply_chat_template(self.tokenizer, task.task_params)
                uid = self._start_task(task, prompt)
            except PrefillCancelled:
                continue
            except Exception as e:
                self._send_error(task, e)
                output.append((task.task_id, FinishedResponse()))
                continue

            queue = GeneratorQueue[GenerationResponse]()
            if task.task_params.bench:
                output_generator: Iterator[GenerationChunk | None] = map(
                    lambda r: map_responses_to_chunks(r, self.model_id), queue.gen()
                )
            else:
                output_generator = apply_all_parsers(
                    queue.gen(),
                    prompt,
                    self.tool_parser,
                    self.tokenizer,
                    type(self.model),
                    self.model_id,
                    task.task_params.tools,
                )
            self._active_tasks[uid] = (task, queue, output_generator)

        if not self._gen.has_work:
            return itertools.chain(output, self._apply_cancellations())

        results = self._gen.step()

        for uid, response in results:
            if uid not in self._active_tasks:
                # should we error here?
                logger.warning(f"{uid=} not found in active tasks")
                continue

            task, queue, output_generator = self._active_tasks[uid]
            queue.push(response)
            # If a generator fails to parse for some reason and returns early, we should not crash
            while (parsed := next(output_generator, None)) is not None:
                output.append((task.task_id, parsed))

            # check if original response was terminal and append a Finished()
            if response.finish_reason is not None:
                output.append((task.task_id, FinishedResponse()))
                del self._active_tasks[uid]

        return filter(
            lambda chunk: (
                not isinstance(chunk[1], GenerationChunk) or self.device_rank == 0
            ),
            itertools.chain(output, self._apply_cancellations()),
        )

    def _apply_cancellations(
        self,
    ) -> Iterator[tuple[TaskId, CancelledResponse]]:
        if not self._cancelled_tasks:
            return iter([])

        cancel_all = CANCEL_ALL_TASKS in self._cancelled_tasks

        uids_to_cancel: list[int] = []
        results: list[tuple[TaskId, CancelledResponse]] = []

        for uid, (task, _, _) in list(self._active_tasks.items()):
            if task.task_id in self._cancelled_tasks or cancel_all:
                uids_to_cancel.append(uid)
                results.append((task.task_id, CancelledResponse()))
                del self._active_tasks[uid]

        if uids_to_cancel:
            self._gen.cancel(uids_to_cancel)

        already_cancelled = {tid for tid, _ in results}
        for tid in self._cancelled_tasks:
            if tid != CANCEL_ALL_TASKS and tid not in already_cancelled:
                results.append((tid, CancelledResponse()))

        self._cancelled_tasks.clear()
        return iter(results)

    def _send_error(self, task: TextGeneration, e: Exception) -> None:
        if self.device_rank == 0:
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=self.model_id,
                        finish_reason="error",
                        error_message=str(e),
                    ),
                )
            )

    def _start_task(self, task: TextGeneration, prompt: str) -> int:
        _check_for_debug_prompts(task.task_params)

        def on_prefill_progress(processed: int, total: int) -> None:
            if self.device_rank == 0:
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=task.command_id,
                        chunk=PrefillProgressChunk(
                            model=self.model_id,
                            processed_tokens=processed,
                            total_tokens=total,
                        ),
                    )
                )

        def distributed_prompt_progress_callback() -> None:
            self.agree_on_cancellations()
            if self.should_cancel(task.task_id):
                raise PrefillCancelled()

            self.agree_on_tasks()

        tokens_since_cancel_check = self.check_for_cancel_every

        def on_generation_token() -> None:
            nonlocal tokens_since_cancel_check
            tokens_since_cancel_check += 1
            if tokens_since_cancel_check >= self.check_for_cancel_every:
                tokens_since_cancel_check = 0
                self.agree_on_cancellations()
                if self.should_cancel(task.task_id):
                    self._cancelled_tasks.add(task.task_id)

                self.agree_on_tasks()

        return self._gen.submit(
            task_params=task.task_params,
            prompt=prompt,
            on_prefill_progress=on_prefill_progress,
            distributed_prompt_progress_callback=distributed_prompt_progress_callback,
            on_generation_token=on_generation_token,
        )

    def close(self) -> None:
        self._gen.close()
        del self.model, self.tokenizer, self.group

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        cache = run_prefill_for_request(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            kv_prefix_cache=self.kv_prefix_cache,
            request=request,
        )
        write_cache_to_wire(
            wfile,
            cache,
            request_id=request.request_id,
            model_id=request.model_id,
            start_pos=request.start_pos,
        )
