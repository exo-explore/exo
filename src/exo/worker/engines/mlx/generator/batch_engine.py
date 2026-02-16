"""Batch generation engine using mlx_lm's BatchGenerator for continuous batching."""

import time
from dataclasses import dataclass, field
from typing import get_args

import mlx.core as mx
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import StreamingDetokenizer, TokenizerWrapper

from exo.shared.types.api import FinishReason, GenerationStats
from exo.shared.types.common import CommandId
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import TaskId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.constants import MAX_TOKENS
from exo.worker.engines.mlx.generator.distributed_sync import share_object
from exo.worker.engines.mlx.utils_mlx import apply_chat_template
from exo.worker.runner.bootstrap import logger


@dataclass
class PendingInsert:
    """Pre-tokenized request ready for batch insertion."""

    command_id: CommandId
    task_id: TaskId
    tokens: list[int]
    max_tokens: int
    prompt_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None


@dataclass
class ActiveRequest:
    """Tracks an active request in the batch."""

    command_id: CommandId
    task_id: TaskId
    uid: int  # BatchGenerator's internal ID
    detokenizer: StreamingDetokenizer
    tokens_generated: int = 0
    prompt_tokens: int = 0
    start_time: float = field(default_factory=time.perf_counter)


@dataclass
class BatchedGenerationResponse:
    """Response from batch engine, tagged with command_id and task_id."""

    command_id: CommandId
    task_id: TaskId
    response: GenerationResponse


class BatchGenerationEngine:
    """Manages continuous batching using mlx_lm's BatchGenerator."""

    def __init__(
        self,
        model: Model,
        tokenizer: TokenizerWrapper,
        group: mx.distributed.Group | None = None,
        max_tokens: int = MAX_TOKENS,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 2048,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.active_requests: dict[int, ActiveRequest] = {}
        self._pending_inserts: list[PendingInsert] = []
        self._pending_completions: list[
            int
        ] = []  # UIDs completed but not yet synced/removed

        self.group = group
        self.rank = group.rank() if group else 0
        self.is_distributed = group is not None and group.size() > 1

        sampler = make_sampler(temp=0.7, top_p=1.0)

        eos_tokens: set[int] = set(tokenizer.eos_token_ids or [])

        self.batch_gen: BatchGenerator = BatchGenerator(
            model=model,
            max_tokens=max_tokens,
            stop_tokens=eos_tokens,
            sampler=sampler,
            completion_batch_size=completion_batch_size,
            prefill_batch_size=prefill_batch_size,
            prefill_step_size=prefill_step_size,
        )

        logger.info(
            f"BatchGenerationEngine initialized with completion_batch_size={completion_batch_size}, "
            f"prefill_batch_size={prefill_batch_size}, distributed={self.is_distributed}"
        )

    def queue_request(
        self,
        command_id: CommandId,
        task_id: TaskId,
        task_params: TextGenerationTaskParams,
    ) -> str:
        """Queue a pre-tokenized request for insertion. Only rank 0 should call this.

        Tokenization happens here (eagerly) so that sync_and_insert_pending()
        only does the lightweight batch_gen.insert() call, keeping the decode
        thread unblocked for as long as possible.

        Returns the prompt string for caller use (e.g. thinking-mode detection).
        """
        assert self.rank == 0, "Only rank 0 should queue requests"
        prompt_str = apply_chat_template(self.tokenizer, task_params)
        tokens: list[int] = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        max_tokens = task_params.max_output_tokens or self.max_tokens
        self._pending_inserts.append(
            PendingInsert(
                command_id=command_id,
                task_id=task_id,
                tokens=tokens,
                max_tokens=max_tokens,
                prompt_tokens=len(tokens),
                temperature=task_params.temperature,
                top_p=task_params.top_p,
                top_k=task_params.top_k,
            )
        )
        logger.info(
            f"Queued request {command_id} for insertion (pending={len(self._pending_inserts)}, prompt_tokens={len(tokens)})"
        )
        return prompt_str

    def sync_and_insert_pending(self) -> list[int]:
        """Sync pre-tokenized pending inserts across ranks and insert them. Returns UIDs.

        Tokens are already prepared by queue_request(), so this method only does
        the lightweight batch_gen.insert() call plus distributed sync if needed.
        """
        inserts_to_process: list[PendingInsert]

        if not self.is_distributed:
            # Non-distributed: just insert directly from pending
            inserts_to_process = list(self._pending_inserts)
        else:
            # Distributed: broadcast pre-tokenized inserts from rank 0 to all ranks
            assert self.group is not None
            inserts_to_process = share_object(
                self._pending_inserts if self.rank == 0 else None,
                self.rank,
                self.group,
            )

        if not inserts_to_process:
            self._pending_inserts.clear()
            return []

        # Update sampler from per-request parameters (last request wins for batch)
        last = inserts_to_process[-1]
        self.batch_gen.sampler = make_sampler(  # pyright: ignore[reportAttributeAccessIssue]
            temp=last.temperature if last.temperature is not None else 0.7,
            top_p=last.top_p if last.top_p is not None else 1.0,
            top_k=last.top_k if last.top_k is not None else 0,
        )

        # Single batched insert for efficient prefill — tokens already prepared
        all_tokens = [p.tokens for p in inserts_to_process]
        all_max_tokens = [p.max_tokens for p in inserts_to_process]
        uids = self.batch_gen.insert(all_tokens, max_tokens=all_max_tokens)

        # Track all inserted requests
        for i, uid in enumerate(uids):
            p = inserts_to_process[i]
            self.active_requests[uid] = ActiveRequest(
                command_id=p.command_id,
                task_id=p.task_id,
                uid=uid,
                detokenizer=self.tokenizer.detokenizer,
                prompt_tokens=p.prompt_tokens,
            )
            logger.info(
                f"Inserted request {p.command_id} with uid={uid}, prompt_tokens={p.prompt_tokens}, max_tokens={p.max_tokens}"
            )

        self._pending_inserts.clear()
        return uids

    def step(self) -> list[BatchedGenerationResponse]:
        """Run one decode step. Tracks completions but does not sync - call sync_completions() at budget boundaries."""
        responses = self.batch_gen.next()
        if not responses:
            return []

        results: list[BatchedGenerationResponse] = []

        for r in responses:
            uid: int = r.uid
            req = self.active_requests.get(uid)
            if req is None:
                logger.warning(f"Received response for unknown uid={uid}")
                continue

            req.tokens_generated += 1

            # Decode the token
            token: int = r.token
            req.detokenizer.add_token(token)
            text: str = req.detokenizer.last_segment

            stats: GenerationStats | None = None
            finish_reason: FinishReason | None = None

            raw_finish_reason: str | None = r.finish_reason
            if raw_finish_reason is not None:
                # Finalize to get remaining text
                req.detokenizer.finalize()
                text = req.detokenizer.last_segment

                elapsed = time.perf_counter() - req.start_time
                generation_tps = req.tokens_generated / elapsed if elapsed > 0 else 0.0

                stats = GenerationStats(
                    prompt_tps=0.0,  # Not tracked per-request in batch mode
                    generation_tps=generation_tps,
                    prompt_tokens=req.prompt_tokens,
                    generation_tokens=req.tokens_generated,
                    peak_memory_usage=Memory.from_gb(mx.get_peak_memory() / 1e9),
                )

                if raw_finish_reason in get_args(FinishReason):
                    finish_reason = raw_finish_reason  # pyright: ignore[reportAssignmentType]
                else:
                    logger.warning(f"Unknown finish_reason: {raw_finish_reason}")
                    finish_reason = "stop"

                # Track completion but don't remove yet - wait for sync_completions()
                self._pending_completions.append(uid)
                logger.info(
                    f"Request {req.command_id} completed: {req.tokens_generated} tokens, {generation_tps:.2f} tps, reason={finish_reason}"
                )

            results.append(
                BatchedGenerationResponse(
                    command_id=req.command_id,
                    task_id=req.task_id,
                    response=GenerationResponse(
                        text=text,
                        token=token,
                        finish_reason=finish_reason,
                        stats=stats,
                        usage=None,
                    ),
                )
            )

        # In non-distributed mode, clean up completions immediately
        if not self.is_distributed:
            self._remove_completed()

        return results

    def sync_completions(self) -> None:
        """Sync and remove completed requests. Call at time budget boundaries in distributed mode."""
        if not self.is_distributed:
            # Non-distributed: early return if nothing to do
            if not self._pending_completions:
                return
            self._remove_completed()
            return

        # Distributed mode: ALWAYS sync to ensure all ranks participate in collective op
        # This prevents deadlock if one rank has completions and another doesn't
        assert self.group is not None
        self._pending_completions = share_object(
            self._pending_completions if self.rank == 0 else None,
            self.rank,
            self.group,
        )
        self._remove_completed()

    def _remove_completed(self) -> None:
        """Remove completed requests from tracking."""
        for uid in self._pending_completions:
            if uid in self.active_requests:
                del self.active_requests[uid]
        self._pending_completions.clear()

    @property
    def has_active_requests(self) -> bool:
        return bool(self.active_requests or self.batch_gen.unprocessed_prompts)

    @property
    def has_pending_inserts(self) -> bool:
        return bool(self._pending_inserts)

    @property
    def active_count(self) -> int:
        return len(self.active_requests)

    @property
    def pending_count(self) -> int:
        return len(self.batch_gen.unprocessed_prompts)

    @property
    def pending_insert_count(self) -> int:
        return len(self._pending_inserts)

    @property
    def has_pending_completions(self) -> bool:
        return bool(self._pending_completions)
