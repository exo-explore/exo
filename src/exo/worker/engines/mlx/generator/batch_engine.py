"""Batch generation engine using mlx_lm's BatchGenerator for continuous batching."""

import time
from dataclasses import dataclass, field

import mlx.core as mx
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import StreamingDetokenizer, TokenizerWrapper

from exo.shared.types.api import FinishReason, GenerationStats
from exo.shared.types.common import CommandId
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import ChatCompletionTaskParams, TaskId
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.constants import MAX_TOKENS
from exo.worker.engines.mlx.generator.distributed_sync import share_object
from exo.worker.engines.mlx.utils_mlx import apply_chat_template
from exo.worker.runner.bootstrap import logger


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
        self._pending_inserts: list[
            tuple[CommandId, TaskId, ChatCompletionTaskParams]
        ] = []
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
        task_params: ChatCompletionTaskParams,
    ) -> None:
        """Queue a request for insertion. Only rank 0 should call this.

        In distributed mode, rank 0 receives tasks from the control plane and
        queues them here. The actual insertion happens in sync_and_insert_pending()
        which ensures all ranks insert the same requests together.
        """
        assert self.rank == 0, "Only rank 0 should queue requests"
        self._pending_inserts.append((command_id, task_id, task_params))
        logger.info(
            f"Queued request {command_id} for insertion (pending={len(self._pending_inserts)})"
        )

    def sync_and_insert_pending(self) -> list[int]:
        """Sync pending inserts across ranks and insert them. Returns UIDs.

        This method ensures all ranks insert the same requests in the same order.
        In non-distributed mode, it simply inserts all pending requests.
        In distributed mode, it broadcasts pending requests from rank 0 to all ranks.

        Batches all pending inserts into a single batch_gen.insert() call for
        efficient prefill batching.
        """
        inserts_to_process: list[tuple[CommandId, TaskId, ChatCompletionTaskParams]]

        if not self.is_distributed:
            # Non-distributed: just insert directly from pending
            inserts_to_process = list(self._pending_inserts)
        else:
            # Distributed: broadcast pending inserts from rank 0 to all ranks
            assert self.group is not None
            pending_data = self._pending_inserts if self.rank == 0 else None
            synced_data = share_object(pending_data, self.rank, self.group)

            if synced_data is None:
                self._pending_inserts.clear()
                return []

            inserts_to_process = synced_data

        if not inserts_to_process:
            self._pending_inserts.clear()
            return []

        # Prepare all requests for batched insertion
        all_tokens: list[list[int]] = []
        all_max_tokens: list[int] = []
        all_prompt_tokens: list[int] = []
        request_info: list[tuple[CommandId, TaskId]] = []

        for cmd_id, task_id, params in inserts_to_process:
            prompt_str = apply_chat_template(self.tokenizer, params)
            tokens: list[int] = self.tokenizer.encode(
                prompt_str, add_special_tokens=False
            )
            max_tokens = params.max_tokens or self.max_tokens

            all_tokens.append(tokens)
            all_max_tokens.append(max_tokens)
            all_prompt_tokens.append(len(tokens))
            request_info.append((cmd_id, task_id))

        # Single batched insert for efficient prefill
        uids = self.batch_gen.insert(all_tokens, max_tokens=all_max_tokens)

        # Track all inserted requests
        for i, uid in enumerate(uids):
            cmd_id, task_id = request_info[i]
            self.active_requests[uid] = ActiveRequest(
                command_id=cmd_id,
                task_id=task_id,
                uid=uid,
                detokenizer=self.tokenizer.detokenizer,
                prompt_tokens=all_prompt_tokens[i],
            )
            logger.info(
                f"Inserted request {cmd_id} with uid={uid}, prompt_tokens={all_prompt_tokens[i]}, max_tokens={all_max_tokens[i]}"
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

                if raw_finish_reason == "stop":
                    finish_reason = "stop"
                elif raw_finish_reason == "length":
                    finish_reason = "length"
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
                        text=text, token=token, finish_reason=finish_reason, stats=stats
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
        synced_uids = share_object(
            self._pending_completions if self.rank == 0 else None,
            self.rank,
            self.group,
        )
        if synced_uids:
            self._pending_completions = synced_uids

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
