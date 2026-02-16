import base64
import json
import resource
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Callable, Literal

import mlx.core as mx
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.tokenizer_utils import TokenizerWrapper
from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
    HarmonyEncodingName,
    Role,
    StreamableParser,
    load_harmony_encoding,
)
from pydantic import ValidationError

from exo.shared.constants import EXO_MAX_CHUNK_SIZE, EXO_TRACING_ENABLED
from exo.shared.models.model_cards import ModelId, ModelTask
from exo.shared.tracing import clear_trace_buffer, get_trace_buffer
from exo.shared.types.api import ImageGenerationStats
from exo.shared.types.chunks import ErrorChunk, ImageChunk, TokenChunk, ToolCallChunk
from exo.shared.types.common import CommandId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
    TraceEventData,
    TracesCollected,
)
from exo.shared.types.tasks import (
    ConnectToGroup,
    ImageEdits,
    ImageGeneration,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ImageGenerationResponse,
    PartialImageResponse,
    ToolCallItem,
    ToolCallResponse,
)
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerFailed,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    ShardMetadata,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.image import (
    DistributedImageModel,
    generate_image,
    initialize_image_model,
    warmup_image_generator,
)
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.generator.batch_engine import (
    BatchedGenerationResponse,
    BatchGenerationEngine,
)
from exo.worker.engines.mlx.generator.generate import warmup_inference
from exo.worker.engines.mlx.generator.time_budget import TimeBudget
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    detect_thinking_prompt_suffix,
    initialize_mlx,
    load_mlx_items,
    mlx_force_oom,
)
from exo.worker.runner.bootstrap import logger


def _is_primary_output_node(shard_metadata: ShardMetadata) -> bool:
    """Check if this node is the primary output node for image generation.

    For CFG models: the last pipeline stage in CFG group 0 (positive prompt).
    For non-CFG models: the last pipeline stage.
    """
    if isinstance(shard_metadata, CfgShardMetadata):
        is_pipeline_last = (
            shard_metadata.pipeline_rank == shard_metadata.pipeline_world_size - 1
        )
        return is_pipeline_last and shard_metadata.cfg_rank == 0
    elif isinstance(shard_metadata, PipelineShardMetadata):
        return shard_metadata.device_rank == shard_metadata.world_size - 1
    return False


def _narrow_finish_reason(
    raw: str | None,
) -> Literal["stop", "length", "content_filter"] | None:
    """Narrow a finish_reason to the subset accepted by TokenChunk."""
    if raw is None or raw in ("stop", "length", "content_filter"):
        return raw
    return "stop"


@dataclass
class ToolCallTracker:
    """Per-request tool call state machine for the batch generation path.

    Mirrors the logic of parse_tool_calls() but operates on individual tokens
    from the batch engine rather than a streaming Generator.
    """

    tool_call_start: str
    tool_call_end: str
    tool_parser: Callable[[str], dict[str, Any] | list[dict[str, Any]]]
    in_tool_call: bool = False
    tool_call_text_parts: list[str] = field(default_factory=list)

    def process_token(
        self, response: GenerationResponse, model_id: ModelId
    ) -> TokenChunk | ToolCallChunk | None:
        text = response.text

        if text == self.tool_call_start:
            self.in_tool_call = True
            return None

        if self.in_tool_call and text == self.tool_call_end:
            result: TokenChunk | ToolCallChunk
            try:
                parsed = self.tool_parser("".join(self.tool_call_text_parts).strip())
                logger.info(
                    f"parsed tool_call_text_parts={self.tool_call_text_parts!r} into {parsed=}"
                )
                if isinstance(parsed, list):
                    tools = [_validate_single_tool(tool) for tool in parsed]
                else:
                    tools = [_validate_single_tool(parsed)]
                result = ToolCallChunk(
                    model=model_id,
                    tool_calls=tools,
                    usage=response.usage,
                    stats=response.stats,
                )
            except (
                json.JSONDecodeError,
                ValidationError,
                ValueError,
                AttributeError,
            ) as e:
                logger.opt(exception=e).warning("tool call parsing failed")
                result = TokenChunk(
                    model=model_id,
                    text=self.tool_call_start
                    + "".join(self.tool_call_text_parts)
                    + self.tool_call_end,
                    token_id=response.token,
                    finish_reason=_narrow_finish_reason(response.finish_reason),
                    usage=response.usage,
                    stats=response.stats,
                    logprob=response.logprob,
                    top_logprobs=response.top_logprobs,
                )
            self.in_tool_call = False
            self.tool_call_text_parts = []
            return result

        if self.in_tool_call:
            self.tool_call_text_parts.append(text)
            if response.finish_reason is not None:
                logger.info(
                    "tool call parsing interrupted, yield partial tool call as text"
                )
                partial = TokenChunk(
                    model=model_id,
                    text=self.tool_call_start + "".join(self.tool_call_text_parts),
                    token_id=0,
                    finish_reason=_narrow_finish_reason(response.finish_reason),
                    usage=None,
                )
                self.in_tool_call = False
                self.tool_call_text_parts = []
                return partial
            return None

        return TokenChunk(
            model=model_id,
            text=text,
            token_id=response.token,
            finish_reason=_narrow_finish_reason(response.finish_reason),
            usage=response.usage,
            stats=response.stats,
            logprob=response.logprob,
            top_logprobs=response.top_logprobs,
        )


_KIMI_SECTION_TOKENS = frozenset(
    {"<|tool_calls_section_begin|>", "<|tool_calls_section_end|>"}
)


@dataclass
class GptOssTracker:
    """Per-request GPT-OSS parsing state for the batch generation path.

    Wraps StreamableParser to handle thinking channels and tool call routing
    on a per-token basis, mirroring parse_gpt_oss() for the batch path.
    """

    stream: StreamableParser
    thinking: bool = False
    current_tool_name: str | None = None
    tool_arg_parts: list[str] = field(default_factory=list)

    def process_token(
        self, response: GenerationResponse, model_id: ModelId
    ) -> list[TokenChunk | ToolCallChunk]:
        """Process a single token and return chunks to emit.

        May return 0, 1, or multiple chunks for a single input token
        (e.g. thinking state transition + text delta).
        """
        self.stream.process(response.token)
        delta: str = self.stream.last_content_delta or ""
        ch: str | None = self.stream.current_channel
        recipient: str | None = self.stream.current_recipient

        chunks: list[TokenChunk | ToolCallChunk] = []

        # Handle tool call transitions
        if recipient != self.current_tool_name:
            if self.current_tool_name is not None:
                name = self.current_tool_name
                prefix = "functions."
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                chunks.append(
                    ToolCallChunk(
                        model=model_id,
                        tool_calls=[
                            ToolCallItem(
                                name=name,
                                arguments="".join(self.tool_arg_parts).strip(),
                            )
                        ],
                        usage=response.usage,
                        stats=response.stats,
                    )
                )
                self.tool_arg_parts = []
            self.current_tool_name = recipient

        # Inside tool call: accumulate args
        if self.current_tool_name is not None:
            if delta:
                self.tool_arg_parts.append(delta)
            return chunks

        # Handle thinking state transitions
        if ch == "analysis" and not self.thinking:
            self.thinking = True
            chunks.append(
                TokenChunk(
                    model=model_id,
                    text="<think>",
                    token_id=response.token,
                    usage=None,
                )
            )

        if ch != "analysis" and self.thinking:
            self.thinking = False
            chunks.append(
                TokenChunk(
                    model=model_id,
                    text="</think>",
                    token_id=response.token,
                    usage=None,
                )
            )

        # Emit text delta
        if delta:
            chunks.append(
                TokenChunk(
                    model=model_id,
                    text=delta,
                    token_id=response.token,
                    finish_reason=_narrow_finish_reason(response.finish_reason),
                    usage=response.usage,
                    stats=response.stats,
                    logprob=response.logprob,
                    top_logprobs=response.top_logprobs,
                )
            )

        # Handle finish
        if response.finish_reason is not None:
            if self.thinking:
                self.thinking = False
                chunks.append(
                    TokenChunk(
                        model=model_id,
                        text="</think>",
                        token_id=response.token,
                        usage=None,
                    )
                )
            # Emit final chunk with finish_reason if no delta was emitted above
            if not delta:
                chunks.append(
                    TokenChunk(
                        model=model_id,
                        text=response.text,
                        token_id=response.token,
                        finish_reason=_narrow_finish_reason(response.finish_reason),
                        usage=response.usage,
                        stats=response.stats,
                        logprob=response.logprob,
                        top_logprobs=response.top_logprobs,
                    )
                )

        return chunks


def _make_chunk(
    response: GenerationResponse,
    model_id: ModelId,
    tracker: ToolCallTracker | None,
) -> TokenChunk | ToolCallChunk | None:
    """Create a chunk from a generation response, routing through tool call tracker if present."""
    if tracker is not None:
        return tracker.process_token(response, model_id)
    return TokenChunk(
        model=model_id,
        text=response.text,
        token_id=response.token,
        finish_reason=_narrow_finish_reason(response.finish_reason),
        usage=response.usage,
        stats=response.stats,
        logprob=response.logprob,
        top_logprobs=response.top_logprobs,
    )


def _process_generation_results(
    results: list[BatchedGenerationResponse],
    event_sender: MpSender[Event],
    device_rank: int,
    shard_metadata: ShardMetadata,
    in_flight_tasks: dict[CommandId, TaskId],
    tool_call_trackers: dict[CommandId, ToolCallTracker],
    is_kimi: bool,
    gpt_oss_trackers: dict[CommandId, GptOssTracker],
    thinking_first_token: dict[CommandId, bool],
    tokenizer: TokenizerWrapper | None,
) -> None:
    """Process batch generation results, emitting chunks and completing finished tasks."""
    model_id = shard_metadata.model_card.model_id
    for r in results:
        if device_rank == 0:
            # Kimi: skip section boundary tokens
            if (
                is_kimi
                and r.response.text in _KIMI_SECTION_TOKENS
                and r.response.finish_reason is None
            ):
                continue

            # GPT-OSS: use dedicated tracker (handles thinking + tool calls)
            if r.command_id in gpt_oss_trackers:
                chunks = gpt_oss_trackers[r.command_id].process_token(
                    r.response, model_id
                )
                for chunk in chunks:
                    event_sender.send(
                        ChunkGenerated(command_id=r.command_id, chunk=chunk)
                    )
            else:
                # Thinking models: prepend think_start on first token
                if (
                    r.command_id in thinking_first_token
                    and not thinking_first_token[r.command_id]
                    and tokenizer is not None
                    and tokenizer.think_start is not None
                    and tokenizer.think_start_id is not None
                ):
                    thinking_first_token[r.command_id] = True
                    event_sender.send(
                        ChunkGenerated(
                            command_id=r.command_id,
                            chunk=TokenChunk(
                                model=model_id,
                                text=tokenizer.think_start,
                                token_id=tokenizer.think_start_id,
                                usage=None,
                            ),
                        )
                    )

                chunk = _make_chunk(
                    r.response, model_id, tool_call_trackers.get(r.command_id)
                )
                if chunk is not None:
                    event_sender.send(
                        ChunkGenerated(command_id=r.command_id, chunk=chunk)
                    )

        if r.response.finish_reason is not None:
            task_id = in_flight_tasks.pop(r.command_id, None)
            if task_id is not None:
                _send_traces_if_enabled(
                    event_sender, task_id, shard_metadata.device_rank
                )
                event_sender.send(
                    TaskStatusUpdated(task_id=task_id, task_status=TaskStatus.Complete)
                )
            tool_call_trackers.pop(r.command_id, None)
            gpt_oss_trackers.pop(r.command_id, None)
            thinking_first_token.pop(r.command_id, None)


def _run_generation_steps(
    batch_engine: BatchGenerationEngine,
    event_sender: MpSender[Event],
    device_rank: int,
    shard_metadata: ShardMetadata,
    group: object | None,
    in_flight_tasks: dict[CommandId, TaskId],
    tool_call_trackers: dict[CommandId, ToolCallTracker],
    is_kimi: bool,
    gpt_oss_trackers: dict[CommandId, GptOssTracker],
    thinking_first_token: dict[CommandId, bool],
    tokenizer: TokenizerWrapper | None,
) -> None:
    """Run one TimeBudget cycle of batch generation."""
    if batch_engine.has_pending_inserts:
        batch_engine.sync_and_insert_pending()
    for _ in TimeBudget(budget=0.5, group=group):  # pyright: ignore[reportArgumentType]
        if not batch_engine.has_active_requests:
            break
        try:
            results = batch_engine.step()
        except Exception as e:
            logger.opt(exception=e).error("batch_engine.step() failed")
            model_id = shard_metadata.model_card.model_id
            if device_rank == 0:
                for cmd_id in list(in_flight_tasks.keys()):
                    event_sender.send(
                        ChunkGenerated(
                            command_id=cmd_id,
                            chunk=ErrorChunk(
                                model=model_id,
                                finish_reason="error",
                                error_message=str(e),
                            ),
                        )
                    )
            for _cmd_id, task_id in in_flight_tasks.items():
                event_sender.send(
                    TaskStatusUpdated(task_id=task_id, task_status=TaskStatus.Complete)
                )
            in_flight_tasks.clear()
            tool_call_trackers.clear()
            gpt_oss_trackers.clear()
            thinking_first_token.clear()
            raise
        _process_generation_results(
            results,
            event_sender,
            device_rank,
            shard_metadata,
            in_flight_tasks,
            tool_call_trackers,
            is_kimi,
            gpt_oss_trackers,
            thinking_first_token,
            tokenizer,
        )
    batch_engine.sync_completions()


def _drain_batch_engine(
    batch_engine: BatchGenerationEngine,
    event_sender: MpSender[Event],
    device_rank: int,
    shard_metadata: ShardMetadata,
    in_flight_tasks: dict[CommandId, TaskId],
    tool_call_trackers: dict[CommandId, ToolCallTracker],
    is_kimi: bool,
    gpt_oss_trackers: dict[CommandId, GptOssTracker],
    thinking_first_token: dict[CommandId, bool],
    tokenizer: TokenizerWrapper | None,
) -> None:
    """Drain all active requests from the batch engine, emitting chunks for each token."""
    if batch_engine.has_pending_inserts:
        batch_engine.sync_and_insert_pending()
    while batch_engine.has_active_requests:
        try:
            results = batch_engine.step()
        except Exception as e:
            logger.opt(exception=e).error("batch_engine.step() failed during drain")
            model_id = shard_metadata.model_card.model_id
            if device_rank == 0:
                for cmd_id in list(in_flight_tasks.keys()):
                    event_sender.send(
                        ChunkGenerated(
                            command_id=cmd_id,
                            chunk=ErrorChunk(
                                model=model_id,
                                finish_reason="error",
                                error_message=str(e),
                            ),
                        )
                    )
            for _cmd_id, task_id in in_flight_tasks.items():
                event_sender.send(
                    TaskStatusUpdated(task_id=task_id, task_status=TaskStatus.Complete)
                )
            in_flight_tasks.clear()
            tool_call_trackers.clear()
            gpt_oss_trackers.clear()
            thinking_first_token.clear()
            raise
        _process_generation_results(
            results,
            event_sender,
            device_rank,
            shard_metadata,
            in_flight_tasks,
            tool_call_trackers,
            is_kimi,
            gpt_oss_trackers,
            thinking_first_token,
            tokenizer,
        )
        batch_engine.sync_completions()


def main(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 2048), hard), hard))

    instance, runner_id, shard_metadata = (
        bound_instance.instance,
        bound_instance.bound_runner_id,
        bound_instance.bound_shard,
    )
    device_rank = shard_metadata.device_rank
    logger.info("hello from the runner")
    if getattr(shard_metadata, "immediate_exception", False):
        raise Exception("Fake exception - runner failed to spin up.")
    if timeout := getattr(shard_metadata, "should_timeout", 0):
        time.sleep(timeout)

    setup_start_time = time.time()

    model: Model | DistributedImageModel | None = None
    tokenizer = None
    group = None
    batch_engine: BatchGenerationEngine | None = None

    current_status: RunnerStatus = RunnerIdle()
    logger.info("runner created")
    event_sender.send(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
    )
    seen = set[TaskId]()
    in_flight_tasks: dict[CommandId, TaskId] = {}
    tool_call_trackers: dict[CommandId, ToolCallTracker] = {}
    gpt_oss_trackers: dict[CommandId, GptOssTracker] = {}
    thinking_first_token: dict[CommandId, bool] = {}
    is_kimi = False
    is_gpt_oss = False

    with task_receiver:
        while True:
            # Phase 1: Run generation if batch engine has active work
            if batch_engine is not None and (
                batch_engine.has_active_requests or batch_engine.has_pending_inserts
            ):
                _run_generation_steps(
                    batch_engine,
                    event_sender,
                    device_rank,
                    shard_metadata,
                    group,
                    in_flight_tasks,
                    tool_call_trackers,
                    is_kimi,
                    gpt_oss_trackers,
                    thinking_first_token,
                    tokenizer,
                )
                # Update runner status based on remaining work
                if batch_engine.has_active_requests:
                    new_status: RunnerStatus = RunnerRunning(
                        active_requests=batch_engine.active_count
                    )
                else:
                    new_status = RunnerReady()
                if new_status != current_status:
                    current_status = new_status
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                # Non-blocking poll for new tasks
                new_tasks = task_receiver.collect()
                if not new_tasks and batch_engine.has_active_requests:
                    continue
            else:
                # No active work — blocking wait for next task
                try:
                    new_tasks = [next(task_receiver)]
                except StopIteration:
                    break

            # Phase 2: Process new tasks
            for task in new_tasks:
                if task.task_id in seen:
                    logger.warning("repeat task - potential error")
                seen.add(task.task_id)
                event_sender.send(
                    TaskStatusUpdated(
                        task_id=task.task_id, task_status=TaskStatus.Running
                    )
                )

                emit_task_completion = True

                match task:
                    case ConnectToGroup() if isinstance(
                        current_status, (RunnerIdle, RunnerFailed)
                    ):
                        logger.info("runner connecting")
                        current_status = RunnerConnecting()
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                        event_sender.send(TaskAcknowledged(task_id=task.task_id))
                        group = initialize_mlx(bound_instance)

                        logger.info("runner connected")
                        current_status = RunnerConnected()

                    # we load the model if it's connected with a group, or idle without a group. we should never tell a model to connect if it doesn't need to
                    case LoadModel() if (
                        isinstance(current_status, RunnerConnected)
                        and group is not None
                    ) or (isinstance(current_status, RunnerIdle) and group is None):
                        current_status = RunnerLoading()
                        logger.info("runner loading")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                        event_sender.send(TaskAcknowledged(task_id=task.task_id))

                        def on_model_load_timeout() -> None:
                            event_sender.send(
                                RunnerStatusUpdated(
                                    runner_id=runner_id,
                                    runner_status=RunnerFailed(
                                        error_message="Model loading timed out"
                                    ),
                                )
                            )
                            time.sleep(0.5)

                        if ModelTask.TextGeneration in shard_metadata.model_card.tasks:
                            model, tokenizer = load_mlx_items(
                                bound_instance,
                                group,
                                on_timeout=on_model_load_timeout,
                            )
                            logger.info(
                                f"model has_tool_calling={tokenizer.has_tool_calling}"
                            )

                            # Apply model-specific tokenizer patches
                            model_id_lower = shard_metadata.model_card.model_id.lower()
                            if "kimi" in model_id_lower:
                                patch_kimi_tokenizer(tokenizer)
                                is_kimi = True
                                logger.info("applied kimi tokenizer patch")
                            elif "glm" in model_id_lower:
                                patch_glm_tokenizer(tokenizer)
                                logger.info("applied glm tokenizer patch")

                            is_gpt_oss = isinstance(model, GptOssModel)

                        elif (
                            ModelTask.TextToImage in shard_metadata.model_card.tasks
                            or ModelTask.ImageToImage in shard_metadata.model_card.tasks
                        ):
                            model = initialize_image_model(bound_instance)
                        else:
                            raise ValueError(
                                f"Unknown model task(s): {shard_metadata.model_card.tasks}"
                            )
                        current_status = RunnerLoaded()
                        logger.info("runner loaded")
                    case StartWarmup() if isinstance(current_status, RunnerLoaded):
                        assert model

                        current_status = RunnerWarmingUp()
                        logger.info("runner warming up")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                        event_sender.send(TaskAcknowledged(task_id=task.task_id))

                        logger.info(f"warming up inference for instance: {instance}")
                        if ModelTask.TextGeneration in shard_metadata.model_card.tasks:
                            assert not isinstance(model, DistributedImageModel)
                            assert tokenizer

                            toks = warmup_inference(
                                model=model,
                                tokenizer=tokenizer,
                                group=group,
                                # kv_prefix_cache=kv_prefix_cache,  # supply for warmup-time prefix caching
                            )
                            logger.info(f"warmed up by generating {toks} tokens")
                            logger.info(
                                f"runner initialized in {time.time() - setup_start_time} seconds"
                            )
                            batch_engine = BatchGenerationEngine(
                                model=model, tokenizer=tokenizer, group=group
                            )
                        elif (
                            ModelTask.TextToImage in shard_metadata.model_card.tasks
                            or ModelTask.ImageToImage in shard_metadata.model_card.tasks
                        ):
                            assert isinstance(model, DistributedImageModel)
                            image = warmup_image_generator(model=model)
                            if image is not None:
                                logger.info(
                                    f"warmed up by generating {image.size} image"
                                )
                            else:
                                logger.info("warmup completed (non-primary node)")

                        current_status = RunnerReady()
                        logger.info("runner ready")
                    case TextGeneration(
                        task_params=task_params, command_id=command_id
                    ) if (
                        isinstance(current_status, (RunnerReady, RunnerRunning))
                        and batch_engine is not None
                    ):
                        logger.info(f"received chat request: {task}")
                        _check_for_debug_prompts(task_params)
                        event_sender.send(TaskAcknowledged(task_id=task.task_id))
                        batch_engine.queue_request(
                            command_id, task.task_id, task_params
                        )
                        batch_engine.sync_and_insert_pending()
                        in_flight_tasks[command_id] = task.task_id

                        # GPT-OSS: use dedicated tracker (handles thinking + tool calls)
                        if is_gpt_oss and tokenizer is not None:
                            gpt_oss_trackers[command_id] = GptOssTracker(
                                stream=StreamableParser(
                                    get_gpt_oss_encoding(), role=Role.ASSISTANT
                                ),
                            )
                        elif (
                            tokenizer is not None
                            and tokenizer.has_tool_calling
                            and tokenizer.tool_call_start is not None
                            and tokenizer.tool_call_end is not None
                        ):
                            tool_call_trackers[command_id] = ToolCallTracker(
                                tool_call_start=tokenizer.tool_call_start,
                                tool_call_end=tokenizer.tool_call_end,
                                tool_parser=tokenizer._tool_parser,  # pyright: ignore[reportAny]
                            )

                        # Thinking models: detect if chat template consumed a think tag
                        if (
                            tokenizer is not None
                            and not is_gpt_oss
                            and tokenizer.has_thinking
                        ):
                            prompt = apply_chat_template(tokenizer, task_params)
                            if detect_thinking_prompt_suffix(prompt, tokenizer):
                                thinking_first_token[command_id] = False
                        current_status = RunnerRunning(
                            active_requests=batch_engine.active_count
                        )
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id,
                                runner_status=current_status,
                            )
                        )
                        emit_task_completion = False
                        logger.info(
                            f"runner running with {batch_engine.active_count} active requests"
                        )
                    case ImageGeneration(
                        task_params=task_params, command_id=command_id
                    ) if isinstance(current_status, RunnerReady):
                        assert isinstance(model, DistributedImageModel)
                        logger.info(
                            f"received image generation request: {str(task)[:500]}"
                        )
                        current_status = RunnerRunning()
                        logger.info("runner running")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                        event_sender.send(TaskAcknowledged(task_id=task.task_id))

                        try:
                            image_index = 0
                            for response in generate_image(
                                model=model, task=task_params
                            ):
                                is_primary_output = _is_primary_output_node(
                                    shard_metadata
                                )

                                if is_primary_output:
                                    match response:
                                        case PartialImageResponse():
                                            logger.info(
                                                f"sending partial ImageChunk {response.partial_index}/{response.total_partials}"
                                            )
                                            _process_image_response(
                                                response,
                                                command_id,
                                                shard_metadata,
                                                event_sender,
                                                image_index,
                                            )
                                        case ImageGenerationResponse():
                                            logger.info("sending final ImageChunk")
                                            _process_image_response(
                                                response,
                                                command_id,
                                                shard_metadata,
                                                event_sender,
                                                image_index,
                                            )
                                            image_index += 1
                        # can we make this more explicit?
                        except Exception as e:
                            if _is_primary_output_node(shard_metadata):
                                event_sender.send(
                                    ChunkGenerated(
                                        command_id=command_id,
                                        chunk=ErrorChunk(
                                            model=shard_metadata.model_card.model_id,
                                            finish_reason="error",
                                            error_message=str(e),
                                        ),
                                    )
                                )
                            raise
                        finally:
                            _send_traces_if_enabled(
                                event_sender,
                                task.task_id,
                                shard_metadata.device_rank,
                            )

                        current_status = RunnerReady()
                        logger.info("runner ready")
                    case ImageEdits(task_params=task_params, command_id=command_id) if (
                        isinstance(current_status, RunnerReady)
                    ):
                        assert isinstance(model, DistributedImageModel)
                        logger.info(f"received image edits request: {str(task)[:500]}")
                        current_status = RunnerRunning()
                        logger.info("runner running")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                        event_sender.send(TaskAcknowledged(task_id=task.task_id))

                        try:
                            image_index = 0
                            for response in generate_image(
                                model=model, task=task_params
                            ):
                                if _is_primary_output_node(shard_metadata):
                                    match response:
                                        case PartialImageResponse():
                                            logger.info(
                                                f"sending partial ImageChunk {response.partial_index}/{response.total_partials}"
                                            )
                                            _process_image_response(
                                                response,
                                                command_id,
                                                shard_metadata,
                                                event_sender,
                                                image_index,
                                            )
                                        case ImageGenerationResponse():
                                            logger.info("sending final ImageChunk")
                                            _process_image_response(
                                                response,
                                                command_id,
                                                shard_metadata,
                                                event_sender,
                                                image_index,
                                            )
                                            image_index += 1
                        except Exception as e:
                            if _is_primary_output_node(shard_metadata):
                                event_sender.send(
                                    ChunkGenerated(
                                        command_id=command_id,
                                        chunk=ErrorChunk(
                                            model=shard_metadata.model_card.model_id,
                                            finish_reason="error",
                                            error_message=str(e),
                                        ),
                                    )
                                )
                            raise
                        finally:
                            _send_traces_if_enabled(
                                event_sender,
                                task.task_id,
                                shard_metadata.device_rank,
                            )

                        current_status = RunnerReady()
                        logger.info("runner ready")
                    case Shutdown():
                        # Drain batch engine before shutting down
                        if batch_engine is not None and (
                            batch_engine.has_active_requests
                            or batch_engine.has_pending_inserts
                        ):
                            _drain_batch_engine(
                                batch_engine,
                                event_sender,
                                device_rank,
                                shard_metadata,
                                in_flight_tasks,
                                tool_call_trackers,
                                is_kimi,
                                gpt_oss_trackers,
                                thinking_first_token,
                                tokenizer,
                            )
                        current_status = RunnerShuttingDown()
                        logger.info("runner shutting down")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                        event_sender.send(TaskAcknowledged(task_id=task.task_id))

                        current_status = RunnerShutdown()
                    case _:
                        raise ValueError(
                            f"Received {task.__class__.__name__} outside of state machine in {current_status=}"
                        )

                if emit_task_completion:
                    event_sender.send(
                        TaskStatusUpdated(
                            task_id=task.task_id,
                            task_status=TaskStatus.Complete,
                        )
                    )
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                if isinstance(current_status, RunnerShutdown):
                    del model, tokenizer, group
                    mx.clear_cache()
                    import gc

                    gc.collect()
                    break
            else:
                continue
            break


@cache
def get_gpt_oss_encoding():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoding


def filter_kimi_tokens(
    responses: Generator[GenerationResponse | ToolCallResponse],
) -> Generator[GenerationResponse]:
    for resp in responses:
        assert isinstance(resp, GenerationResponse)
        if (
            resp.text == "<|tool_calls_section_begin|>"
            or resp.text == "<|tool_calls_section_end|>"
        ):
            continue
        yield resp


def parse_gpt_oss(
    responses: Generator[GenerationResponse | ToolCallResponse],
) -> Generator[GenerationResponse | ToolCallResponse]:
    encoding = get_gpt_oss_encoding()
    stream = StreamableParser(encoding, role=Role.ASSISTANT)
    thinking = False
    current_tool_name: str | None = None
    tool_arg_parts: list[str] = []

    for response in responses:
        assert isinstance(response, GenerationResponse)
        stream.process(response.token)

        delta = stream.last_content_delta
        ch = stream.current_channel
        recipient = stream.current_recipient

        if recipient != current_tool_name:
            if current_tool_name is not None:
                prefix = "functions."
                if current_tool_name.startswith(prefix):
                    current_tool_name = current_tool_name[len(prefix) :]
                yield ToolCallResponse(
                    tool_calls=[
                        ToolCallItem(
                            name=current_tool_name,
                            arguments="".join(tool_arg_parts).strip(),
                        )
                    ],
                    usage=response.usage,
                )
                tool_arg_parts = []
            current_tool_name = recipient

        # If inside a tool call, accumulate arguments
        if current_tool_name is not None:
            if delta:
                tool_arg_parts.append(delta)
            continue

        if ch == "analysis" and not thinking:
            thinking = True
            yield response.model_copy(update={"text": "<think>"})

        if ch != "analysis" and thinking:
            thinking = False
            yield response.model_copy(update={"text": "</think>"})

        if delta:
            yield response.model_copy(update={"text": delta})

        if response.finish_reason is not None:
            if thinking:
                yield response.model_copy(update={"text": "</think>"})
            yield response


def parse_thinking_models(
    responses: Generator[GenerationResponse | ToolCallResponse],
    tokenizer: TokenizerWrapper,
) -> Generator[GenerationResponse | ToolCallResponse]:
    """
    For models that inject thinking tags in the prompt (like GLM-4.7),
    prepend the thinking tag to the output stream so the frontend
    can properly parse thinking content.
    """
    first = True
    for response in responses:
        if isinstance(response, ToolCallResponse):
            yield response
            continue
        if first:
            first = False
            yield response.model_copy(
                update={
                    "text": tokenizer.think_start,
                    "token": tokenizer.think_start_id,
                }
            )
        yield response


def _send_image_chunk(
    encoded_data: str,
    command_id: CommandId,
    model_id: ModelId,
    event_sender: MpSender[Event],
    image_index: int,
    is_partial: bool,
    partial_index: int | None = None,
    total_partials: int | None = None,
    stats: ImageGenerationStats | None = None,
    image_format: Literal["png", "jpeg", "webp"] | None = None,
) -> None:
    """Send base64-encoded image data as chunks via events."""
    data_chunks = [
        encoded_data[i : i + EXO_MAX_CHUNK_SIZE]
        for i in range(0, len(encoded_data), EXO_MAX_CHUNK_SIZE)
    ]
    total_chunks = len(data_chunks)
    for chunk_index, chunk_data in enumerate(data_chunks):
        # Only include stats on the last chunk of the final image
        chunk_stats = (
            stats if chunk_index == total_chunks - 1 and not is_partial else None
        )
        event_sender.send(
            ChunkGenerated(
                command_id=command_id,
                chunk=ImageChunk(
                    model=model_id,
                    data=chunk_data,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    image_index=image_index,
                    is_partial=is_partial,
                    partial_index=partial_index,
                    total_partials=total_partials,
                    stats=chunk_stats,
                    format=image_format,
                ),
            )
        )


def _send_traces_if_enabled(
    event_sender: MpSender[Event],
    task_id: TaskId,
    rank: int,
) -> None:
    if not EXO_TRACING_ENABLED:
        return

    traces = get_trace_buffer()
    if traces:
        trace_data = [
            TraceEventData(
                name=t.name,
                start_us=t.start_us,
                duration_us=t.duration_us,
                rank=t.rank,
                category=t.category,
            )
            for t in traces
        ]
        event_sender.send(
            TracesCollected(
                task_id=task_id,
                rank=rank,
                traces=trace_data,
            )
        )
    clear_trace_buffer()


def _process_image_response(
    response: ImageGenerationResponse | PartialImageResponse,
    command_id: CommandId,
    shard_metadata: ShardMetadata,
    event_sender: MpSender[Event],
    image_index: int,
) -> None:
    """Process a single image response and send chunks."""
    encoded_data = base64.b64encode(response.image_data).decode("utf-8")
    is_partial = isinstance(response, PartialImageResponse)
    # Extract stats from final ImageGenerationResponse if available
    stats = response.stats if isinstance(response, ImageGenerationResponse) else None
    _send_image_chunk(
        encoded_data=encoded_data,
        command_id=command_id,
        model_id=shard_metadata.model_card.model_id,
        event_sender=event_sender,
        image_index=response.image_index,
        is_partial=is_partial,
        partial_index=response.partial_index if is_partial else None,
        total_partials=response.total_partials if is_partial else None,
        stats=stats,
        image_format=response.format,
    )


def parse_tool_calls(
    responses: Generator[GenerationResponse | ToolCallResponse],
    tool_call_start: str,
    tool_call_end: str,
    tool_parser: Callable[[str], dict[str, Any] | list[dict[str, Any]]],
) -> Generator[GenerationResponse | ToolCallResponse]:
    in_tool_call = False
    tool_call_text_parts: list[str] = []
    for response in responses:
        assert isinstance(response, GenerationResponse)
        # assumption: the tool call start is one token
        if response.text == tool_call_start:
            in_tool_call = True
            continue
        # assumption: the tool call end is one token
        if in_tool_call and response.text == tool_call_end:
            try:
                # tool_parser returns an arbitrarily nested python dictionary
                # we actually don't want the python dictionary, we just want to
                # parse the top level { function: ..., arguments: ... } structure
                # as we're just gonna hand it back to the api anyway
                parsed = tool_parser("".join(tool_call_text_parts).strip())
                logger.info(f"parsed {tool_call_text_parts=} into {parsed=}")
                if isinstance(parsed, list):
                    tools = [_validate_single_tool(tool) for tool in parsed]
                else:
                    tools = [_validate_single_tool(parsed)]
                yield ToolCallResponse(tool_calls=tools, usage=response.usage)

            except (
                json.JSONDecodeError,
                ValidationError,
                ValueError,
                AttributeError,
            ) as e:
                # ValueError: our parsers raise this for malformed tool calls
                # AttributeError: upstream parsers (e.g. glm47) may raise this when regex doesn't match
                logger.opt(exception=e).warning("tool call parsing failed")
                # assumption: talking about tool calls, not making a tool call
                response.text = (
                    tool_call_start + "".join(tool_call_text_parts) + tool_call_end
                )
                yield response

            in_tool_call = False
            tool_call_text_parts = []
            continue

        if in_tool_call:
            tool_call_text_parts.append(response.text)
            if response.finish_reason is not None:
                logger.info(
                    "toll call parsing interrupted, yield partial tool call as text"
                )
                yield GenerationResponse(
                    text=tool_call_start + "".join(tool_call_text_parts),
                    token=0,
                    finish_reason=response.finish_reason,
                    usage=None,
                )
            continue
        # fallthrough
        yield response


def patch_kimi_tokenizer(tokenizer: TokenizerWrapper):
    """
    Version of to-be-upstreamed kimi-k2 tool parser
    """
    import ast
    import json
    from typing import Any

    import regex as re

    # kimi has a fixed function naming scheme, with a json formatted arg
    #   functions.multiply:0 <|tool_call_argument_begin|> {"a": 2, "b": 3}
    #   Also needs to handle tools like call_0<|tool_call_argument_begin|>{"filePath": "..."}
    _func_name_regex = re.compile(
        r"^\s*(.+)[:](\d+)\s*<\|tool_call_argument_begin\|>", re.DOTALL
    )
    _func_arg_regex = re.compile(r"<\|tool_call_argument_begin\|>\s*(.*)\s*", re.DOTALL)

    # kimi has a tool_calls_section - we're leaving this up to the caller to handle
    tool_call_start = "<|tool_call_begin|>"
    tool_call_end = "<|tool_call_end|>"

    def _deserialize(value: str) -> Any:  # pyright: ignore[reportAny]
        try:
            return json.loads(value)  # pyright: ignore[reportAny]
        except Exception:
            pass

        try:
            return ast.literal_eval(value)  # pyright: ignore[reportAny]
        except Exception:
            pass
        return value

    def parse_tool_call(text: str, tools: Any | None = None):
        func_name_match = _func_name_regex.search(text)
        if func_name_match is None:
            raise ValueError(f"Could not parse function name from tool call: {text!r}")
        original_func_name = func_name_match.group(1)
        tool_id = func_name_match.group(2)
        # strip off the `functions.` prefix, if it exists.
        func_name = original_func_name[original_func_name.find(".") + 1 :]

        func_args_match = _func_arg_regex.search(text)
        if func_args_match is None:
            raise ValueError(f"Could not parse function args from tool call: {text!r}")
        func_args = func_args_match.group(1)
        # the args should be valid json - no need to check against our tools to deserialize
        arg_dct = _deserialize(func_args)  # pyright: ignore[reportAny]

        return dict(
            id=f"{original_func_name}:{tool_id}",
            name=func_name,
            arguments=arg_dct,  # pyright: ignore[reportAny]
        )

    tokenizer._tool_call_start = tool_call_start
    tokenizer._tool_call_end = tool_call_end
    tokenizer._tool_parser = parse_tool_call


def patch_glm_tokenizer(tokenizer: TokenizerWrapper):
    """
    Fixed version of mlx_lm's glm47 tool parser that handles regex match failures.
    """
    import ast
    import json
    from typing import Any

    import regex as re

    _func_name_regex = re.compile(r"^(.*?)<arg_key>", re.DOTALL)
    _func_arg_regex = re.compile(
        r"<arg_key>(.*?)</arg_key>(?:\n|\s)*<arg_value>(.*?)(?:</arg_value>|(?=<arg_key>)|$)",
        re.DOTALL,
    )

    tool_call_start = "<tool_call>"
    tool_call_end = "</tool_call>"

    def _is_string_type(
        tool_name: str,
        arg_name: str,
        tools: list[Any] | None,
    ) -> bool:
        if tools is None:
            return False
        for tool in tools:  # pyright: ignore[reportAny]
            func = tool["function"]  # pyright: ignore[reportAny]
            if func["name"] == tool_name:
                params = func["parameters"]  # pyright: ignore[reportAny]
                if params is None:
                    return False
                props = params.get("properties", {})  # pyright: ignore[reportAny]
                arg_props = props.get(arg_name, {})  # pyright: ignore[reportAny]
                arg_type = arg_props.get("type", None)  # pyright: ignore[reportAny]
                return arg_type == "string"  # pyright: ignore[reportAny]
        return False

    def _deserialize(value: str) -> Any:  # pyright: ignore[reportAny]
        try:
            return json.loads(value)  # pyright: ignore[reportAny]
        except Exception:
            pass
        try:
            return ast.literal_eval(value)  # pyright: ignore[reportAny]
        except Exception:
            pass
        return value

    def parse_tool_call(text: str, tools: list[Any] | None = None):
        func_name_match = _func_name_regex.search(text)
        if func_name_match is None:
            raise ValueError(f"Could not parse function name from tool call: {text!r}")
        func_name = func_name_match.group(1)

        pairs = _func_arg_regex.findall(text)
        arg_dct: dict[str, Any] = {}
        for key, value in pairs:  # pyright: ignore[reportAny]
            arg_key = key.strip()  # pyright: ignore[reportAny]
            arg_val = value.strip()  # pyright: ignore[reportAny]
            if not _is_string_type(func_name, arg_key, tools):  # pyright: ignore[reportAny]
                arg_val = _deserialize(arg_val)  # pyright: ignore[reportAny]
            arg_dct[arg_key] = arg_val
        return dict(name=func_name, arguments=arg_dct)

    tokenizer._tool_call_start = tool_call_start
    tokenizer._tool_call_end = tool_call_end
    tokenizer._tool_parser = parse_tool_call


def _validate_single_tool(obj: dict[str, Any]) -> ToolCallItem:
    if (
        ((name := obj.get("name")) is not None)
        and ((args := obj.get("arguments")) is not None)
        and isinstance(name, str)
    ):
        raw_id: object = obj.get("id")
        extra = {"id": str(raw_id)} if raw_id is not None else {}
        return ToolCallItem(
            **extra,
            name=name,
            arguments=json.dumps(args),
        )
    else:
        raise ValidationError


EXO_RUNNER_MUST_FAIL = "EXO RUNNER MUST FAIL"
EXO_RUNNER_MUST_OOM = "EXO RUNNER MUST OOM"
EXO_RUNNER_MUST_TIMEOUT = "EXO RUNNER MUST TIMEOUT"


def _check_for_debug_prompts(task_params: TextGenerationTaskParams) -> None:
    """Check for debug prompt triggers in the input.

    Extracts the first user input text and checks for debug triggers.
    """
    if len(task_params.input) == 0:
        logger.debug("Empty message list in debug prompt check")
        return
    prompt = task_params.input[0].content

    if not prompt:
        return

    if EXO_RUNNER_MUST_FAIL in prompt:
        logger.info("raising exception")
        raise Exception("Artificial runner exception - for testing purposes only.")
    if EXO_RUNNER_MUST_OOM in prompt:
        mlx_force_oom()
    if EXO_RUNNER_MUST_TIMEOUT in prompt:
        time.sleep(100)
