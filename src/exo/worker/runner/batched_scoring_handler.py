"""Batched scoring handler for processing multiple Completion requests concurrently."""

import time
from dataclasses import dataclass, field

from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.models.model_cards import ModelId
from exo.shared.types.api import TopLogprobItem
from exo.shared.types.chunks import CompletionChunk, ErrorChunk
from exo.shared.types.events import ChunkGenerated, Event
from exo.shared.types.tasks import Completion
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.generator.generate import score_tokens_batched
from exo.worker.runner.bootstrap import logger


@dataclass
class PendingScoringRequest:
    """A scoring request waiting to be batched."""

    task: Completion
    tokens: list[int]
    prompt_text: str
    top_k: int | None
    echo: bool


@dataclass
class BatchedScoringHandler:
    """
    Handles batched scoring for multiple Completion requests.

    Collects multiple scoring requests and processes them in a single
    batched forward pass for improved throughput.
    """

    model: Model
    tokenizer: TokenizerWrapper
    model_id: ModelId
    device_rank: int
    max_batch_size: int = 32
    batch_timeout_ms: int = 10

    pending: list[PendingScoringRequest] = field(default_factory=list)
    pending_start_time: float | None = None

    @property
    def has_pending(self) -> bool:
        """Check if there are pending requests."""
        return len(self.pending) > 0

    def add_request(
        self,
        task: Completion,
        tokens: list[int],
        prompt_text: str,
    ) -> None:
        """Add a Completion request to the pending batch."""
        task_params = task.task_params
        top_k = task_params.logprobs

        self.pending.append(
            PendingScoringRequest(
                task=task,
                tokens=tokens,
                prompt_text=prompt_text,
                top_k=top_k,
                echo=task_params.echo,
            )
        )

        if self.pending_start_time is None:
            self.pending_start_time = time.perf_counter()

        logger.debug(f"Added scoring request to batch (pending={len(self.pending)})")

    def should_flush(self) -> bool:
        """Check if the batch should be flushed."""
        if not self.has_pending:
            return False

        # Flush if batch is full
        if len(self.pending) >= self.max_batch_size:
            return True

        # Flush if timeout reached
        if self.pending_start_time is not None:
            elapsed_ms = (time.perf_counter() - self.pending_start_time) * 1000
            if elapsed_ms >= self.batch_timeout_ms:
                return True

        return False

    def flush(self) -> list[Event]:
        """Process all pending requests and return events."""
        if not self.has_pending:
            return []

        requests = self.pending
        self.pending = []
        self.pending_start_time = None

        logger.info(f"Processing batch of {len(requests)} scoring requests")

        # Collect all token sequences
        token_sequences = [req.tokens for req in requests]

        # Get common top_k (use first request's top_k, they should all be the same)
        top_k = requests[0].top_k if requests else None

        try:
            # Run batched scoring
            all_results = score_tokens_batched(
                model=self.model,
                tokenizer=self.tokenizer,
                token_sequences=token_sequences,
                top_k=top_k,
            )

            # Generate events for each request
            events: list[Event] = []
            for req, logprob_results in zip(requests, all_results, strict=True):
                if self.device_rank != 0:
                    continue

                event = self._build_completion_event(req, logprob_results)
                events.append(event)

            logger.info(f"Batch scoring complete ({len(events)} events)")
            return events

        except Exception as e:
            # Return error events for all requests
            logger.error(f"Batch scoring failed: {e}")
            events = []
            for req in requests:
                if self.device_rank == 0:
                    events.append(
                        ChunkGenerated(
                            command_id=req.task.command_id,
                            chunk=ErrorChunk(
                                model=self.model_id,
                                finish_reason="error",
                                error_message=str(e),
                            ),
                        )
                    )
            return events

    def _build_completion_event(
        self,
        req: PendingScoringRequest,
        logprob_results: list[tuple[float, list[TopLogprobItem]]],
    ) -> Event:
        """Build a ChunkGenerated event for a completed scoring request."""
        tokens = req.tokens
        tokenizer = self.tokenizer

        # Build response in completions format
        token_strings: list[str] = []
        token_logprobs: list[float | None] = []
        top_logprobs: list[dict[str, float]] = []
        text_offset: list[int] = []

        offset = 0
        for i, token_id in enumerate(tokens):
            token_str = tokenizer.decode([token_id])
            token_strings.append(token_str)

            if i < len(logprob_results):
                logprob, top_items = logprob_results[i]
                # First token has no logprob (None in OpenAI format)
                token_logprobs.append(logprob if i > 0 else None)
                top_lp_dict = {item.token: item.logprob for item in top_items}
                top_logprobs.append(top_lp_dict)
            else:
                token_logprobs.append(None)
                top_logprobs.append({})

            text_offset.append(offset)
            offset += len(token_str)

        return ChunkGenerated(
            command_id=req.task.command_id,
            chunk=CompletionChunk(
                model=self.model_id,
                text=req.prompt_text if req.echo else "",
                tokens=token_strings,
                token_logprobs=token_logprobs,
                top_logprobs=top_logprobs,
                text_offset=text_offset,
                finish_reason="stop",
            ),
        )

    def close(self) -> None:
        """Clean up resources."""
        self.pending.clear()
        self.pending_start_time = None
