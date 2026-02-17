import base64
import math
import resource
import time
from collections.abc import Generator
from functools import cache
from typing import Literal

import mlx.core as mx
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.tokenizer_utils import TokenizerWrapper
from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
    HarmonyEncodingName,
    Role,
    StreamableParser,
    load_harmony_encoding,
)

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
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.generator.generate import mlx_generate, warmup_inference
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    detect_thinking_prompt_suffix,
    initialize_mlx,
    load_mlx_items,
    mlx_force_oom,
    mx_any,
)
from exo.worker.runner.bootstrap import logger

from .tool_parsers import ToolParser, make_mlx_parser


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


def main(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
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
    cancelled_tasks = set[TaskId]()

    # type checker was unhappy with me - splitting these fixed it
    inference_model: Model | None = None
    image_model: DistributedImageModel | None = None
    tokenizer = None
    tool_parser: ToolParser | None = None
    group = None
    kv_prefix_cache: KVPrefixCache | None = None
    check_for_cancel_every: int | None = None

    current_status: RunnerStatus = RunnerIdle()
    logger.info("runner created")
    event_sender.send(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
    )
    seen = set[TaskId]()
    with task_receiver as tasks:
        for task in tasks:
            if task.task_id in seen:
                logger.warning("repeat task - potential error")
            seen.add(task.task_id)
            cancelled_tasks.discard(TaskId("CANCEL_CURRENT_TASK"))
            event_sender.send(
                TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Running)
            )
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
                    isinstance(current_status, RunnerConnected) and group is not None
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
                        inference_model, tokenizer = load_mlx_items(
                            bound_instance, group, on_timeout=on_model_load_timeout
                        )
                        logger.info(
                            f"model has_tool_calling={tokenizer.has_tool_calling} using tokens {tokenizer.tool_call_start}, {tokenizer.tool_call_end}"
                        )
                        if tokenizer.has_tool_calling:
                            assert tokenizer.tool_call_start
                            assert tokenizer.tool_call_end
                            assert tokenizer.tool_parser  # pyright: ignore[reportAny]
                            tool_parser = make_mlx_parser(
                                tokenizer.tool_call_start,
                                tokenizer.tool_call_end,
                                tokenizer.tool_parser,  # pyright: ignore[reportAny]
                            )
                        kv_prefix_cache = KVPrefixCache(group)

                    elif (
                        ModelTask.TextToImage in shard_metadata.model_card.tasks
                        or ModelTask.ImageToImage in shard_metadata.model_card.tasks
                    ):
                        image_model = initialize_image_model(bound_instance)
                    else:
                        raise ValueError(
                            f"Unknown model task(s): {shard_metadata.model_card.tasks}"
                        )
                    current_status = RunnerLoaded()
                    logger.info("runner loaded")
                case StartWarmup() if isinstance(current_status, RunnerLoaded):
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
                        assert inference_model
                        assert tokenizer

                        t = time.perf_counter()
                        toks = warmup_inference(
                            model=inference_model,
                            tokenizer=tokenizer,
                            group=group,
                        )
                        logger.info(f"warmed up by generating {toks} tokens")
                        check_for_cancel_every = min(
                            math.ceil(toks / max(time.perf_counter() - t, 0.001)), 100
                        )
                        if group is not None:
                            check_for_cancel_every = int(
                                mx.max(
                                    mx.distributed.all_gather(
                                        mx.array([check_for_cancel_every]), group=group
                                    )
                                ).item()
                            )

                        logger.info(
                            f"runner checking for cancellation every {check_for_cancel_every} tokens"
                        )
                        logger.info(
                            f"runner initialized in {time.time() - setup_start_time} seconds"
                        )
                    elif (
                        ModelTask.TextToImage in shard_metadata.model_card.tasks
                        or ModelTask.ImageToImage in shard_metadata.model_card.tasks
                    ):
                        assert image_model
                        image = warmup_image_generator(model=image_model)
                        if image is not None:
                            logger.info(f"warmed up by generating {image.size} image")
                        else:
                            logger.info("warmup completed (non-primary node)")

                    current_status = RunnerReady()
                    logger.info("runner ready")
                case TextGeneration(task_params=task_params, command_id=command_id) if (
                    isinstance(current_status, RunnerReady)
                ):
                    logger.info(f"received chat request: {task}")
                    current_status = RunnerRunning()
                    logger.info("runner running")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    event_sender.send(TaskAcknowledged(task_id=task.task_id))
                    assert inference_model
                    assert tokenizer
                    assert check_for_cancel_every

                    try:
                        _check_for_debug_prompts(task_params)

                        # Build prompt once - used for both generation and thinking detection
                        prompt = apply_chat_template(tokenizer, task_params)

                        # Generate responses using the actual MLX generation
                        mlx_generator = mlx_generate(
                            model=inference_model,
                            tokenizer=tokenizer,
                            task=task_params,
                            prompt=prompt,
                            kv_prefix_cache=kv_prefix_cache,
                            group=group,
                        )

                        # For other thinking models (GLM, etc.), check if we need to
                        # prepend the thinking tag that was consumed by the chat template
                        if detect_thinking_prompt_suffix(prompt, tokenizer):
                            mlx_generator = parse_thinking_models(
                                mlx_generator, tokenizer
                            )

                        # GPT-OSS specific parsing to match other model formats.
                        if isinstance(inference_model, GptOssModel):
                            mlx_generator = parse_gpt_oss(mlx_generator)
                        elif tool_parser:
                            mlx_generator = parse_tool_calls(mlx_generator, tool_parser)

                        completion_tokens = 0
                        tokens_since_last_cancel_check = 0
                        for response in mlx_generator:
                            tokens_since_last_cancel_check += 1
                            if tokens_since_last_cancel_check >= check_for_cancel_every:
                                tokens_since_last_cancel_check = 0
                                cancelled_tasks.update(cancel_receiver.collect())
                                want_to_cancel = (task.task_id in cancelled_tasks) or (
                                    TaskId("CANCEL_CURRENT_TASK") in cancelled_tasks
                                )
                                if mx_any(want_to_cancel, group):
                                    break

                            match response:
                                case GenerationResponse():
                                    completion_tokens += 1
                                    if (
                                        device_rank == 0
                                        and response.finish_reason == "error"
                                    ):
                                        event_sender.send(
                                            ChunkGenerated(
                                                command_id=command_id,
                                                chunk=ErrorChunk(
                                                    error_message=response.text,
                                                    model=shard_metadata.model_card.model_id,
                                                ),
                                            )
                                        )

                                    elif device_rank == 0:
                                        assert response.finish_reason not in (
                                            "error",
                                            "tool_calls",
                                            "function_call",
                                        )
                                        event_sender.send(
                                            ChunkGenerated(
                                                command_id=command_id,
                                                chunk=TokenChunk(
                                                    model=shard_metadata.model_card.model_id,
                                                    text=response.text,
                                                    token_id=response.token,
                                                    usage=response.usage,
                                                    finish_reason=response.finish_reason,
                                                    stats=response.stats,
                                                    logprob=response.logprob,
                                                    top_logprobs=response.top_logprobs,
                                                ),
                                            )
                                        )
                                case ToolCallResponse():
                                    if device_rank == 0:
                                        event_sender.send(
                                            ChunkGenerated(
                                                command_id=command_id,
                                                chunk=ToolCallChunk(
                                                    tool_calls=response.tool_calls,
                                                    model=shard_metadata.model_card.model_id,
                                                    usage=response.usage,
                                                    stats=response.stats,
                                                ),
                                            )
                                        )

                    # can we make this more explicit?
                    except Exception as e:
                        if device_rank == 0:
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

                    current_status = RunnerReady()
                    logger.info("runner ready")
                case ImageGeneration(
                    task_params=task_params, command_id=command_id
                ) if isinstance(current_status, RunnerReady):
                    assert image_model
                    logger.info(f"received image generation request: {str(task)[:500]}")
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
                            model=image_model, task=task_params
                        ):
                            is_primary_output = _is_primary_output_node(shard_metadata)

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
                            event_sender, task.task_id, shard_metadata.device_rank
                        )

                    current_status = RunnerReady()
                    logger.info("runner ready")
                case ImageEdits(task_params=task_params, command_id=command_id) if (
                    isinstance(current_status, RunnerReady)
                ):
                    assert image_model
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
                            model=image_model, task=task_params
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
                            event_sender, task.task_id, shard_metadata.device_rank
                        )

                    current_status = RunnerReady()
                    logger.info("runner ready")
                case Shutdown():
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
            was_cancelled = (task.task_id in cancelled_tasks) or (
                TaskId("CANCEL_CURRENT_TASK") in cancelled_tasks
            )
            if not was_cancelled:
                event_sender.send(
                    TaskStatusUpdated(
                        task_id=task.task_id, task_status=TaskStatus.Complete
                    )
                )
            event_sender.send(
                RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
            )
            if isinstance(current_status, RunnerShutdown):
                del inference_model, image_model, tokenizer, group
                mx.clear_cache()
                import gc

                gc.collect()
                break


@cache
def get_gpt_oss_encoding():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoding


def parse_gpt_oss(
    responses: Generator[GenerationResponse],
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
    responses: Generator[GenerationResponse],
    tokenizer: TokenizerWrapper,
) -> Generator[GenerationResponse]:
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
    responses: Generator[GenerationResponse], tool_parser: ToolParser
) -> Generator[GenerationResponse | ToolCallResponse]:
    in_tool_call = False
    tool_call_text_parts: list[str] = []
    for response in responses:
        if response.text.startswith(tool_parser.start_parsing):
            in_tool_call = True

        if in_tool_call:
            tool_call_text_parts.append(response.text)
            if response.text.endswith(tool_parser.end_parsing):
                # parse the actual tool calls from the tool call text
                parsed = tool_parser.parse_tool_calls(
                    "".join(tool_call_text_parts).strip()
                )
                logger.info(f"parsed {tool_call_text_parts=} into {parsed=}")
                if parsed is not None:
                    yield ToolCallResponse(
                        tool_calls=parsed, usage=response.usage, stats=response.stats
                    )
                else:
                    logger.warning(
                        f"tool call parsing failed for text {''.join(tool_call_text_parts)}"
                    )
                    response.text = "".join(tool_call_text_parts)
                    yield response

                in_tool_call = False
                tool_call_text_parts = []
                continue

            if response.finish_reason is not None:
                logger.info(
                    "tool call parsing interrupted, yield partial tool call as text"
                )
                response = response.model_copy(
                    update={
                        "text": "".join(tool_call_text_parts),
                        "token": 0,
                    }
                )
                yield response

            continue

        # fallthrough
        yield response


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
