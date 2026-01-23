"""Batched inference handler for processing multiple ChatCompletion requests concurrently."""

import time
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import mlx.core as mx
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
    HarmonyEncodingName,
    Role,
    StreamableParser,
    load_harmony_encoding,
)

from exo.shared.models.model_cards import ModelId
from exo.shared.types.api import (
    GenerationStats,
    TopLogprobItem,
)
from exo.shared.types.chunks import ErrorChunk, TokenChunk
from exo.shared.types.common import CommandId
from exo.shared.types.events import ChunkGenerated, Event
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import ChatCompletion
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.constants import MAX_TOKENS
from exo.worker.engines.mlx.generator.generate import extract_top_logprobs
from exo.worker.engines.mlx.utils_mlx import apply_chat_template
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.pipelined_generator import PipelinedGenerator, PipelinedResponse

# Type alias for the finish_reason values TokenChunk accepts
TokenFinishReason = Literal["stop", "length", "content_filter"]


@dataclass
class PendingRequest:
    """A request waiting to be added to the batch."""

    task: ChatCompletion
    prompt: str
    max_tokens: int
    sampler: Callable[[mx.array], mx.array]
    should_extract_logprobs: bool
    top_k: int


@dataclass
class ActiveRequest:
    """A request currently being processed in the batch."""

    command_id: CommandId
    should_extract_logprobs: bool
    top_k: int
    gpt_oss_parser: Any | None = None  # StreamableParser for GPT-OSS models
    gpt_oss_thinking: bool = False
    tokens_generated: int = 0
    reasoning_tokens: int = 0
    prompt_tokens: int = 0
    start_time: float = field(default_factory=time.perf_counter)


class BatchedInferenceHandler:
    """
    Handles batched inference for multiple ChatCompletion requests.

    Uses MLX-LM's BatchGenerator to process multiple requests concurrently,
    improving throughput for scenarios with multiple concurrent requests.
    """

    def __init__(
        self,
        model: Model,
        tokenizer: TokenizerWrapper,
        model_id: ModelId,
        device_rank: int,
        world_size: int = 1,
        max_batch_size: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.device_rank = device_rank
        self.world_size = world_size
        self.max_batch_size = max_batch_size

        # GPT-OSS model detection
        self.is_gpt_oss = isinstance(model, GptOssModel)
        self._gpt_oss_encoding: Any | None = None
        if self.is_gpt_oss:
            self._gpt_oss_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            logger.info("GPT-OSS model detected, enabling per-request stream parsing")

        # Pending requests waiting to be batched
        self.pending: list[PendingRequest] = []

        # Active batch generator and request tracking
        self.batch_generator: BatchGenerator | None = None
        self.pipelined_generator: PipelinedGenerator | None = None
        self.uid_to_request: dict[int, ActiveRequest] = {}

        # Use pipelined generator for multi-device pipeline parallelism
        self.use_pipelined = world_size > 1
        if self.use_pipelined:
            logger.info(f"Using PipelinedGenerator with {world_size} streams for pipeline overlap")

        # EOS tokens for the model
        self.stop_tokens: set[int] = set()
        eos_ids: list[int] | None = getattr(tokenizer, "eos_token_ids", None)
        if eos_ids:
            self.stop_tokens = set(eos_ids)

    @property
    def is_active(self) -> bool:
        """Check if there's an active batch being processed."""
        if self.use_pipelined:
            return self.pipelined_generator is not None and self.pipelined_generator.has_active
        return self.batch_generator is not None and len(self.uid_to_request) > 0

    @property
    def has_pending(self) -> bool:
        """Check if there are pending requests waiting to be batched."""
        return len(self.pending) > 0

    @property
    def current_batch_size(self) -> int:
        """Current number of active requests in the batch."""
        return len(self.uid_to_request)

    def add_request(self, task: ChatCompletion) -> None:
        """Add a ChatCompletion request to the pending batch."""
        task_params = task.task_params

        # Build prompt
        prompt = apply_chat_template(self.tokenizer, task_params)

        # Determine max tokens
        max_tokens = task_params.max_tokens or MAX_TOKENS

        # Create sampler for this request
        sampler = make_sampler(
            temp=task_params.temperature if task_params.temperature is not None else 0.7,
            top_p=task_params.top_p if task_params.top_p is not None else 1.0,
        )

        # Logprobs configuration
        should_extract_logprobs = task_params.logprobs is True
        top_k = task_params.top_logprobs if task_params.top_logprobs is not None else 0

        pending_request = PendingRequest(
            task=task,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            should_extract_logprobs=should_extract_logprobs,
            top_k=top_k,
        )

        self.pending.append(pending_request)

        logger.info(
            f"Added request to batch queue (pending={len(self.pending)}, active={self.current_batch_size})"
        )

    def flush(self) -> None:
        """Start processing pending requests by adding them to the batch/pipelined generator."""
        if not self.has_pending:
            return

        # Determine how many requests to flush (up to available slots)
        available_slots = self.max_batch_size - self.current_batch_size
        requests_to_flush = self.pending[:available_slots]
        self.pending = self.pending[available_slots:]

        # Prepare batch data - tokenize prompts
        tokenized_prompts: list[list[int]] = []
        max_tokens_list: list[int] = []
        samplers: list[Callable[[mx.array], mx.array]] = []
        prompt_token_counts: list[int] = []

        for req in requests_to_flush:
            tokens = self.tokenizer.encode(req.prompt)
            tokenized_prompts.append(tokens)
            max_tokens_list.append(req.max_tokens)
            samplers.append(req.sampler)
            prompt_token_counts.append(len(tokens))

        if self.use_pipelined:
            self._flush_pipelined(requests_to_flush, tokenized_prompts, max_tokens_list, samplers, prompt_token_counts)
        else:
            self._flush_batch(requests_to_flush, tokenized_prompts, max_tokens_list, samplers, prompt_token_counts)

    def _flush_pipelined(
        self,
        requests_to_flush: list[PendingRequest],
        tokenized_prompts: list[list[int]],
        max_tokens_list: list[int],
        samplers: list[Callable[[mx.array], mx.array]],
        prompt_token_counts: list[int],
    ) -> None:
        """Flush using PipelinedGenerator (multi-stream pipeline overlap)."""
        if self.pipelined_generator is None:
            logger.info(f"Creating PipelinedGenerator for {len(requests_to_flush)} requests ({self.world_size} streams)")
            mx.reset_peak_memory()
            self.pipelined_generator = PipelinedGenerator(
                model=self.model,
                world_size=self.world_size,
                stop_tokens=self.stop_tokens if self.stop_tokens else None,
                max_tokens=MAX_TOKENS,
            )
        else:
            logger.info(f"Adding {len(requests_to_flush)} requests to PipelinedGenerator")

        uids = self.pipelined_generator.insert(
            prompts=tokenized_prompts,
            max_tokens=max_tokens_list,
            samplers=samplers,
        )

        for uid, req, prompt_tokens in zip(uids, requests_to_flush, prompt_token_counts, strict=True):
            parser = None
            if self.is_gpt_oss and self._gpt_oss_encoding is not None:
                parser = StreamableParser(self._gpt_oss_encoding, role=Role.ASSISTANT)  # pyright: ignore[reportAny]
            self.uid_to_request[uid] = ActiveRequest(
                command_id=req.task.command_id,
                should_extract_logprobs=req.should_extract_logprobs,
                top_k=req.top_k,
                prompt_tokens=prompt_tokens,
                gpt_oss_parser=parser,
            )

        logger.info(f"Flushed {len(requests_to_flush)} requests into pipelined generator (active={self.pipelined_generator.active_count}, uids={list(self.uid_to_request.keys())})")

    def _flush_batch(
        self,
        requests_to_flush: list[PendingRequest],
        tokenized_prompts: list[list[int]],
        max_tokens_list: list[int],
        samplers: list[Callable[[mx.array], mx.array]],
        prompt_token_counts: list[int],
    ) -> None:
        """Flush using BatchGenerator (single-stream, for non-pipeline instances)."""
        if self.batch_generator is None:
            logger.info(f"Creating new BatchGenerator for {len(requests_to_flush)} requests")
            mx.reset_peak_memory()
            self.batch_generator = BatchGenerator(
                model=self.model,
                max_tokens=MAX_TOKENS,
                stop_tokens=self.stop_tokens if self.stop_tokens else None,
                prefill_batch_size=1,
            )
        else:
            logger.info(f"Adding {len(requests_to_flush)} requests to existing BatchGenerator")

        # Insert into batch generator
        uids: list[int] = self.batch_generator.insert(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            prompts=tokenized_prompts,
            max_tokens=max_tokens_list,
            samplers=samplers,  # pyright: ignore[reportCallIssue]
        )

        for uid, req, prompt_tokens in zip(uids, requests_to_flush, prompt_token_counts, strict=True):  # pyright: ignore[reportUnknownArgumentType]
            parser = None
            if self.is_gpt_oss and self._gpt_oss_encoding is not None:
                parser = StreamableParser(self._gpt_oss_encoding, role=Role.ASSISTANT)  # pyright: ignore[reportAny]
            self.uid_to_request[uid] = ActiveRequest(
                command_id=req.task.command_id,
                should_extract_logprobs=req.should_extract_logprobs,
                top_k=req.top_k,
                prompt_tokens=prompt_tokens,
                gpt_oss_parser=parser,
            )

        logger.info(f"Flushed {len(requests_to_flush)} requests into batch (active={self.current_batch_size}, uids={list(self.uid_to_request.keys())})")

    def step(self) -> Generator[Event, None, None]:
        """
        Process one generation step and yield ChunkGenerated events.

        Returns a generator of events for completed tokens across all active requests.
        """
        if self.use_pipelined:
            yield from self._step_pipelined()
            return

        if self.batch_generator is None or not self.uid_to_request:
            return

        # Get next tokens for all active requests
        # BatchGenerator.next() returns list of Response objects
        logger.debug(f"BatchGenerator.next() called (active_uids={list(self.uid_to_request.keys())})")
        responses: list[Any] = self.batch_generator.next()  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        logger.debug(f"BatchGenerator.next() returned {len(responses)} responses")  # pyright: ignore[reportUnknownArgumentType]

        completed_uids: list[int] = []

        for response in responses:  # pyright: ignore[reportUnknownVariableType]
            uid: int = response.uid  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            if uid not in self.uid_to_request:
                logger.warning(f"Received response for unknown uid: {uid}")
                continue

            active_request = self.uid_to_request[uid]
            active_request.tokens_generated += 1

            # Extract response fields with explicit typing
            resp_token: int = response.token  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            resp_finish_reason: str | None = response.finish_reason  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            resp_logprobs: mx.array = response.logprobs  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

            # Only emit events from device_rank 0
            if self.device_rank != 0:
                if resp_finish_reason is not None:
                    completed_uids.append(uid)  # pyright: ignore[reportUnknownArgumentType]
                continue

            # Decode token to text, applying GPT-OSS parsing if needed
            token_text = self.tokenizer.decode([resp_token])
            if active_request.gpt_oss_parser is not None:
                parser = active_request.gpt_oss_parser  # pyright: ignore[reportAny]
                parser.process(resp_token)  # pyright: ignore[reportAny]
                delta: str | None = parser.last_content_delta  # pyright: ignore[reportAny]
                channel: str = parser.current_channel  # pyright: ignore[reportAny]

                # Track reasoning tokens (analysis channel = thinking)
                if channel == "analysis":
                    active_request.reasoning_tokens += 1

                # Handle thinking tag transitions
                prefix = ""
                if channel == "analysis" and not active_request.gpt_oss_thinking:
                    active_request.gpt_oss_thinking = True
                    prefix = "<think>"
                elif channel != "analysis" and active_request.gpt_oss_thinking:
                    active_request.gpt_oss_thinking = False
                    prefix = "</think>"

                if resp_finish_reason is not None and active_request.gpt_oss_thinking:
                    # Close thinking tag on finish
                    prefix = "</think>"
                    active_request.gpt_oss_thinking = False

                effective_delta = delta or ""
                token_text = prefix + effective_delta if (prefix or effective_delta) else ""
                # Skip empty tokens (channel markers with no content delta)
                if not token_text and resp_finish_reason is None:
                    continue

            # Extract logprobs if requested
            logprob: float | None = None
            top_logprobs: list[TopLogprobItem] | None = None
            if active_request.should_extract_logprobs:
                logprob, top_logprobs = extract_top_logprobs(
                    logprobs_array=resp_logprobs,  # pyright: ignore[reportUnknownArgumentType]
                    selected_token=resp_token,  # pyright: ignore[reportUnknownArgumentType]
                    tokenizer=self.tokenizer,
                    top_k=active_request.top_k,
                )

            # Build stats for final token
            stats: GenerationStats | None = None
            finish_reason: TokenFinishReason | None = None
            if resp_finish_reason is not None:
                elapsed_time = time.perf_counter() - active_request.start_time
                prompt_tps = active_request.prompt_tokens / max(elapsed_time, 0.001)
                generation_tps = active_request.tokens_generated / max(elapsed_time, 0.001)

                # Get peak memory
                peak_memory_bytes = 0
                if mx.metal.is_available():
                    peak_memory_bytes = mx.metal.get_peak_memory()

                stats = GenerationStats(
                    prompt_tps=prompt_tps,
                    generation_tps=generation_tps,
                    prompt_tokens=active_request.prompt_tokens,
                    generation_tokens=active_request.tokens_generated,
                    reasoning_tokens=active_request.reasoning_tokens,
                    peak_memory_usage=Memory.from_bytes(peak_memory_bytes),
                )

                # Map finish reason to the narrower type TokenChunk expects
                if resp_finish_reason == "stop":
                    finish_reason = "stop"
                elif resp_finish_reason == "length":
                    finish_reason = "length"
                elif resp_finish_reason == "content_filter":
                    finish_reason = "content_filter"
                else:
                    # Unknown finish reasons default to "stop"
                    logger.warning(f"Unknown finish_reason: {resp_finish_reason}, mapping to 'stop'")
                    finish_reason = "stop"

                completed_uids.append(uid)  # pyright: ignore[reportUnknownArgumentType]

            yield ChunkGenerated(
                command_id=active_request.command_id,
                chunk=TokenChunk(
                    model=self.model_id,
                    text=token_text,
                    token_id=resp_token,  # pyright: ignore[reportUnknownArgumentType]
                    logprob=logprob,
                    top_logprobs=top_logprobs,
                    finish_reason=finish_reason,
                    stats=stats,
                ),
            )

        # Clean up completed requests
        for uid in completed_uids:
            del self.uid_to_request[uid]

    def _step_pipelined(self) -> Generator[Event, None, None]:
        """Process one generation step using the multi-stream PipelinedGenerator."""
        if self.pipelined_generator is None or not self.uid_to_request:
            return

        logger.debug(f"PipelinedGenerator.next() called (active={self.pipelined_generator.active_count})")
        responses: list[PipelinedResponse] = self.pipelined_generator.next()
        logger.debug(f"PipelinedGenerator.next() returned {len(responses)} responses")

        completed_uids: list[int] = []

        for response in responses:
            uid = response.uid
            if uid not in self.uid_to_request:
                logger.warning(f"Received response for unknown uid: {uid}")
                continue

            active_request = self.uid_to_request[uid]
            active_request.tokens_generated += 1

            resp_token: int = response.token
            resp_finish_reason: str | None = response.finish_reason
            resp_logprobs: mx.array = response.logprobs

            # Only emit events from device_rank 0
            if self.device_rank != 0:
                if resp_finish_reason is not None:
                    completed_uids.append(uid)
                continue

            # Decode token to text
            token_text = self.tokenizer.decode([resp_token])
            if active_request.gpt_oss_parser is not None:
                parser = active_request.gpt_oss_parser  # pyright: ignore[reportAny]
                parser.process(resp_token)  # pyright: ignore[reportAny]
                delta: str | None = parser.last_content_delta  # pyright: ignore[reportAny]
                channel: str = parser.current_channel  # pyright: ignore[reportAny]

                if channel == "analysis":
                    active_request.reasoning_tokens += 1

                prefix = ""
                if channel == "analysis" and not active_request.gpt_oss_thinking:
                    active_request.gpt_oss_thinking = True
                    prefix = "<think>"
                elif channel != "analysis" and active_request.gpt_oss_thinking:
                    active_request.gpt_oss_thinking = False
                    prefix = "</think>"

                if resp_finish_reason is not None and active_request.gpt_oss_thinking:
                    prefix = "</think>"
                    active_request.gpt_oss_thinking = False

                effective_delta = delta or ""
                token_text = prefix + effective_delta if (prefix or effective_delta) else ""
                if not token_text and resp_finish_reason is None:
                    continue

            # Extract logprobs if requested
            logprob: float | None = None
            top_logprobs: list[TopLogprobItem] | None = None
            if active_request.should_extract_logprobs:
                logprob, top_logprobs = extract_top_logprobs(
                    logprobs_array=resp_logprobs,
                    selected_token=resp_token,
                    tokenizer=self.tokenizer,
                    top_k=active_request.top_k,
                )

            # Build stats for final token
            stats: GenerationStats | None = None
            finish_reason: TokenFinishReason | None = None
            if resp_finish_reason is not None:
                elapsed_time = time.perf_counter() - active_request.start_time
                prompt_tps = active_request.prompt_tokens / max(elapsed_time, 0.001)
                generation_tps = active_request.tokens_generated / max(elapsed_time, 0.001)

                peak_memory_bytes = 0
                if mx.metal.is_available():
                    peak_memory_bytes = mx.metal.get_peak_memory()

                stats = GenerationStats(
                    prompt_tps=prompt_tps,
                    generation_tps=generation_tps,
                    prompt_tokens=active_request.prompt_tokens,
                    generation_tokens=active_request.tokens_generated,
                    reasoning_tokens=active_request.reasoning_tokens,
                    peak_memory_usage=Memory.from_bytes(peak_memory_bytes),
                )

                if resp_finish_reason == "stop":
                    finish_reason = "stop"
                elif resp_finish_reason == "length":
                    finish_reason = "length"
                else:
                    finish_reason = "stop"

                completed_uids.append(uid)

            yield ChunkGenerated(
                command_id=active_request.command_id,
                chunk=TokenChunk(
                    model=self.model_id,
                    text=token_text,
                    token_id=resp_token,
                    logprob=logprob,
                    top_logprobs=top_logprobs,
                    finish_reason=finish_reason,
                    stats=stats,
                ),
            )

        for uid in completed_uids:
            del self.uid_to_request[uid]

    def emit_error(self, command_id: CommandId, error_message: str) -> Event:
        """Create an error event for a failed request."""
        return ChunkGenerated(
            command_id=command_id,
            chunk=ErrorChunk(
                model=self.model_id,
                finish_reason="error",
                error_message=error_message,
            ),
        )

    def _close_generator(self) -> None:
        """Close and clean up the batch/pipelined generator."""
        if self.batch_generator is not None:
            self.batch_generator.close()  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
            self.batch_generator = None
        if self.pipelined_generator is not None:
            self.pipelined_generator.close()
            self.pipelined_generator = None
        self.uid_to_request.clear()
        logger.info("Generator closed")

    def close(self) -> None:
        """Close the handler and clean up resources."""
        self._close_generator()
        self.pending.clear()
