import base64
import json
import time
from collections.abc import Generator
from functools import cache
from typing import Any, Callable, Literal, cast

import mlx.core as mx
from anyio import EndOfStream, WouldBlock
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.tokenizer_utils import TokenizerWrapper
from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
    HarmonyEncodingName,
    Role,
    StreamableParser,
    load_harmony_encoding,
)
from pydantic import ValidationError

from exo.shared.constants import EXO_MAX_CHUNK_SIZE
from exo.shared.models.model_cards import ModelId, ModelTask
from exo.shared.types.api import ChatCompletionMessageText, ImageGenerationStats
from exo.shared.types.chunks import ErrorChunk, ImageChunk, TokenChunk, ToolCallChunk
from exo.shared.types.common import CommandId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    ChatCompletion,
    Completion,
    ConnectToGroup,
    ImageEdits,
    ImageGeneration,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskStatus,
)
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
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.image import (
    DistributedImageModel,
    generate_image,
    initialize_image_model,
    warmup_image_generator,
)
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.generator.generate import (
    mlx_generate,
    score_tokens,
    warmup_inference,
)
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    detect_thinking_prompt_suffix,
    initialize_mlx,
    load_mlx_items,
    mlx_force_oom,
)
from exo.worker.runner.batched_handler import BatchedInferenceHandler
from exo.worker.runner.bootstrap import logger

# Batching configuration
BATCH_ENABLED = True
BATCH_MAX_SIZE = 128
BATCH_TIMEOUT_MS = 20  # Short timeout - flush quickly to avoid request timeouts


def _should_use_serial_processing(
    task: ChatCompletion,
    tokenizer: TokenizerWrapper,
    model: Model,
    model_id: ModelId,
) -> bool:
    """
    Determine if a ChatCompletion task requires serial processing.

    Currently always returns False - batch mode handles all cases.
    Post-processing (GPT-OSS, thinking models, tool calls) can be applied
    per-request to the individual streams from the batch generator.
    """
    # All tasks can use batch mode - post-processing is per-request
    return False


def _process_serial_chat_completion(
    task: ChatCompletion,
    model: Model,
    tokenizer: TokenizerWrapper,
    shard_metadata: ShardMetadata,
    event_sender: MpSender[Event],
) -> None:
    """Process a ChatCompletion task serially (original implementation)."""
    task_params = task.task_params
    command_id = task.command_id
    device_rank = shard_metadata.device_rank

    if task_params.messages[0].content is not None:
        _check_for_debug_prompts(task_params.messages[0].content)

    # Build prompt once - used for both generation and thinking detection
    prompt = apply_chat_template(tokenizer, task_params)

    # Generate responses using the actual MLX generation
    mlx_generator = mlx_generate(
        model=model,
        tokenizer=tokenizer,
        task=task_params,
        prompt=prompt,
    )

    # GPT-OSS specific parsing to match other model formats.
    if isinstance(model, GptOssModel):
        mlx_generator = parse_gpt_oss(mlx_generator)

    # For other thinking models (GLM, etc.), check if we need to
    # prepend the thinking tag that was consumed by the chat template
    if detect_thinking_prompt_suffix(prompt, tokenizer):
        mlx_generator = parse_thinking_models(mlx_generator, tokenizer)

    # Kimi-K2 has tool call sections - we don't care about them
    if "kimi" in shard_metadata.model_card.model_id.lower():
        mlx_generator = filter_kimi_tokens(mlx_generator)
        patch_kimi_tokenizer(tokenizer)

    if tokenizer.has_tool_calling:
        assert tokenizer.tool_call_start
        assert tokenizer.tool_call_end
        assert tokenizer.tool_parser  # pyright: ignore[reportAny]
        mlx_generator = parse_tool_calls(
            mlx_generator,
            tokenizer.tool_call_start,
            tokenizer.tool_call_end,
            tokenizer.tool_parser,  # pyright: ignore[reportAny]
        )

    for response in mlx_generator:
        match response:
            case GenerationResponse():
                if device_rank == 0 and response.finish_reason == "error":
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
                                logprob=response.logprob,
                                top_logprobs=response.top_logprobs,
                                finish_reason=response.finish_reason,
                                stats=response.stats,
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
                            ),
                        )
                    )


def main(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
):
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
    tokenizer: TokenizerWrapper | None = None
    group = None
    batch_handler: BatchedInferenceHandler | None = None

    current_status: RunnerStatus = RunnerIdle()
    logger.info("runner created")
    event_sender.send(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
    )

    def process_task(task: Task) -> bool:
        """
        Process a single task. Returns True if the runner should continue,
        False if it should shut down.
        """
        nonlocal current_status, model, tokenizer, group, batch_handler
        event_sender.send(
            TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Running)
        )
        # NOTE: TaskAcknowledged is sent per-case below, AFTER the initial status
        # update, to avoid a race where the supervisor sees the ack before the
        # status change and re-dispatches the same lifecycle command.
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
                    model, tokenizer = load_mlx_items(
                        bound_instance, group, on_timeout=on_model_load_timeout
                    )
                    logger.info(f"model has_tool_calling={tokenizer.has_tool_calling}")

                    # Initialize batch handler for text generation models
                    if BATCH_ENABLED:
                        batch_handler = BatchedInferenceHandler(
                            model=model,
                            tokenizer=tokenizer,
                            model_id=shard_metadata.model_card.model_id,
                            device_rank=device_rank,
                            max_batch_size=BATCH_MAX_SIZE,
                            batch_timeout_ms=BATCH_TIMEOUT_MS,
                        )
                        logger.info(
                            f"Batch handler initialized (max_batch_size={BATCH_MAX_SIZE})"
                        )
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
                        # kv_prefix_cache=kv_prefix_cache,  # supply for warmup-time prefix caching
                    )
                    logger.info(f"warmed up by generating {toks} tokens")
                    logger.info(
                        f"runner initialized in {time.time() - setup_start_time} seconds"
                    )
                elif (
                    ModelTask.TextToImage in shard_metadata.model_card.tasks
                    or ModelTask.ImageToImage in shard_metadata.model_card.tasks
                ):
                    assert isinstance(model, DistributedImageModel)
                    image = warmup_image_generator(model=model)
                    if image is not None:
                        logger.info(f"warmed up by generating {image.size} image")
                    else:
                        logger.info("warmup completed (non-primary node)")

                current_status = RunnerReady()
                logger.info("runner ready")
            case ChatCompletion(task_params=task_params, command_id=command_id) if (
                isinstance(current_status, (RunnerReady, RunnerRunning))
            ):
                logger.info(f"received chat request: {task}")
                assert model and not isinstance(model, DistributedImageModel)
                assert tokenizer
                assert task_params.messages[0].content is not None

                # Check if we should use serial processing for this task
                if not BATCH_ENABLED:
                    logger.debug("Serial mode: BATCH_ENABLED is False")
                    use_serial = True
                elif batch_handler is None:
                    logger.debug("Serial mode: batch_handler is None")
                    use_serial = True
                else:
                    use_serial = _should_use_serial_processing(
                        task, tokenizer, model, shard_metadata.model_card.model_id
                    )

                if use_serial:
                    # Serial processing for complex tasks
                    current_status = RunnerRunning()
                    logger.info("runner running (serial mode)")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    event_sender.send(TaskAcknowledged(task_id=task.task_id))

                    try:
                        _process_serial_chat_completion(
                            task, model, tokenizer, shard_metadata, event_sender
                        )
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
                else:
                    # Batch processing for simple tasks
                    assert batch_handler is not None
                    try:
                        _check_for_debug_prompts(task_params.messages[0].content)
                        batch_handler.add_request(task)

                        # Update status to running if not already
                        if not isinstance(current_status, RunnerRunning):
                            current_status = RunnerRunning()
                            logger.info("runner running (batch mode)")
                            event_sender.send(
                                RunnerStatusUpdated(
                                    runner_id=runner_id, runner_status=current_status
                                )
                            )
                        event_sender.send(TaskAcknowledged(task_id=task.task_id))

                        # Return True to indicate task was added to batch
                        # (completion will be sent when batch processes)
                        return True
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
            case Completion(task_params=task_params, command_id=command_id) if (
                isinstance(current_status, RunnerReady)
            ):
                logger.info(f"received completion request: {task}")
                current_status = RunnerRunning()
                logger.info("runner running")
                event_sender.send(
                    RunnerStatusUpdated(
                        runner_id=runner_id, runner_status=current_status
                    )
                )
                event_sender.send(TaskAcknowledged(task_id=task.task_id))
                assert model and not isinstance(model, DistributedImageModel)
                assert tokenizer

                try:
                    # Get the prompt - can be string, list of strings, or token IDs
                    prompt = task_params.prompt
                    prompt_text: str
                    tokens: list[int]

                    if isinstance(prompt, str):
                        # String prompt - tokenize it
                        prompt_text = prompt
                        tokens = tokenizer.encode(prompt)
                    elif len(prompt) == 0:
                        prompt_text = ""
                        tokens = []
                    else:
                        # prompt is list[str] | list[int] | list[list[int]]
                        first_elem = prompt[0]
                        if isinstance(first_elem, int):
                            # List of token IDs - use cast for type checker
                            tokens = cast(list[int], prompt)
                            prompt_text = tokenizer.decode(tokens)
                        elif isinstance(first_elem, str):
                            # List of strings - use first one
                            prompt_text = first_elem
                            tokens = tokenizer.encode(prompt_text)
                        else:
                            # List of token ID lists - use first one
                            tokens = first_elem
                            prompt_text = tokenizer.decode(tokens)

                    # Score all tokens to get logprobs
                    logprob_results = score_tokens(
                        model=model,
                        tokenizer=tokenizer,
                        tokens=tokens,
                        top_k=task_params.logprobs,
                    )

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
                            top_lp_dict = {
                                item.token: item.logprob for item in top_items
                            }
                            top_logprobs.append(top_lp_dict)
                        else:
                            token_logprobs.append(None)
                            top_logprobs.append({})

                        text_offset.append(offset)
                        offset += len(token_str)

                    # Import CompletionChunk here to avoid circular imports
                    from exo.shared.types.chunks import CompletionChunk

                    if device_rank == 0:
                        event_sender.send(
                            ChunkGenerated(
                                command_id=command_id,
                                chunk=CompletionChunk(
                                    model=shard_metadata.model_card.model_id,
                                    text=prompt_text if task_params.echo else "",
                                    tokens=token_strings,
                                    token_logprobs=token_logprobs,
                                    top_logprobs=top_logprobs,
                                    text_offset=text_offset,
                                    finish_reason="stop",
                                ),
                            )
                        )

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
                assert isinstance(model, DistributedImageModel)
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
                    # Generate images using the image generation backend
                    # Track image_index for final images only
                    image_index = 0
                    for response in generate_image(model=model, task=task_params):
                        if (
                            shard_metadata.device_rank
                            == shard_metadata.world_size - 1
                        ):
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
                    if shard_metadata.device_rank == shard_metadata.world_size - 1:
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
                    for response in generate_image(model=model, task=task_params):
                        if (
                            shard_metadata.device_rank
                            == shard_metadata.world_size - 1
                        ):
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
                    if shard_metadata.device_rank == shard_metadata.world_size - 1:
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
            case Shutdown():
                if batch_handler is not None:
                    batch_handler.close()
                    batch_handler = None
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
        event_sender.send(
            TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Complete)
        )
        event_sender.send(
            RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
        )
        return not isinstance(current_status, RunnerShutdown)

    # Track tasks that were added to batch (need completion after batch processes)
    batched_task_ids: list[tuple[Task, bool]] = []  # (task, completed)

    with task_receiver as tasks:
        while True:
            # Check if batch handler is active and needs processing
            if batch_handler is not None and (
                batch_handler.is_active or batch_handler.has_pending
            ):
                # Non-blocking check for new tasks
                try:
                    task = tasks.receive_nowait()
                    # Process the task
                    if isinstance(task, ChatCompletion) and isinstance(
                        current_status, (RunnerReady, RunnerRunning)
                    ):
                        # For ChatCompletion, process_task returns True if added to batch
                        was_batched = process_task(task)
                        if was_batched:
                            batched_task_ids.append((task, False))
                    else:
                        # Non-ChatCompletion tasks are processed synchronously
                        should_continue = process_task(task)
                        if not should_continue:
                            break
                except WouldBlock:
                    pass  # No new task available
                except EndOfStream:
                    break

                # Flush batch if ready
                if batch_handler.should_flush():
                    logger.info(f"Flushing batch (pending={len(batch_handler.pending)}, active={batch_handler.current_batch_size})")
                    batch_handler.flush()

                # Step generation and emit events
                if batch_handler.is_active:
                    event_count = 0
                    for event in batch_handler.step():
                        event_sender.send(event)
                        event_count += 1
                    if event_count > 0:
                        logger.debug(f"Emitted {event_count} events from batch")

                # Check for completed batched tasks
                if not batch_handler.is_active and not batch_handler.has_pending:
                    # All batched tasks completed
                    for task, completed in batched_task_ids:
                        if not completed:
                            event_sender.send(
                                TaskStatusUpdated(
                                    task_id=task.task_id,
                                    task_status=TaskStatus.Complete,
                                )
                            )
                    batched_task_ids.clear()

                    # Return to ready state
                    if isinstance(current_status, RunnerRunning):
                        current_status = RunnerReady()
                        logger.info("runner ready (batch completed)")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
            else:
                # No active batch - use blocking receive
                try:
                    task = tasks.receive()
                    should_continue = process_task(task)
                    if not should_continue:
                        break
                except EndOfStream:
                    break

        # Cleanup
        if batch_handler is not None:
            batch_handler.close()
        del model, tokenizer, group
        mx.clear_cache()
        import gc

        gc.collect()


@cache
def get_gpt_oss_encoding():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoding


def filter_kimi_tokens(
    responses: Generator[GenerationResponse],
) -> Generator[GenerationResponse]:
    for resp in responses:
        if (
            resp.text == "<|tool_calls_section_begin|>"
            or resp.text == "<|tool_calls_section_end|>"
        ):
            continue
        yield resp


def parse_gpt_oss(
    responses: Generator[GenerationResponse],
) -> Generator[GenerationResponse]:
    encoding = get_gpt_oss_encoding()
    stream = StreamableParser(encoding, role=Role.ASSISTANT)
    thinking = False

    for response in responses:
        stream.process(response.token)

        delta = stream.last_content_delta
        ch = stream.current_channel

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
            break


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
        if first:
            first = False
            yield response.model_copy(
                update={
                    "text": tokenizer.think_start,
                    "token": tokenizer.think_start_id,  # type: ignore
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
        image_index=response.partial_index if is_partial else image_index,
        is_partial=is_partial,
        partial_index=response.partial_index if is_partial else None,
        total_partials=response.total_partials if is_partial else None,
        stats=stats,
        image_format=response.format,
    )


def parse_tool_calls(
    responses: Generator[GenerationResponse],
    tool_call_start: str,
    tool_call_end: str,
    tool_parser: Callable[[str], dict[str, Any] | list[dict[str, Any]]],
) -> Generator[GenerationResponse | ToolCallResponse]:
    in_tool_call = False
    tool_call_text_parts: list[str] = []
    for response in responses:
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
                yield ToolCallResponse(tool_calls=tools)

            except (json.JSONDecodeError, ValidationError) as e:
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
    _func_name_regex = re.compile(
        r"^\s*(.+):\d+\s*<\|tool_call_argument_begin\|>", re.DOTALL
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
        func_name = _func_name_regex.search(text).group(1)  # pyright: ignore[reportOptionalMemberAccess]
        # strip off the `functions.` prefix, if it exists.
        func_name = func_name[func_name.find(".") + 1 :]

        func_args = _func_arg_regex.search(text).group(1)  # pyright: ignore[reportOptionalMemberAccess]
        # the args should be valid json - no need to check against our tools to deserialize
        arg_dct = _deserialize(func_args)  # pyright: ignore[reportAny]

        return dict(name=func_name, arguments=arg_dct)  # pyright: ignore[reportAny]

    tokenizer._tool_call_start = tool_call_start
    tokenizer._tool_call_end = tool_call_end
    tokenizer._tool_parser = parse_tool_call


def _validate_single_tool(obj: dict[str, Any]) -> ToolCallItem:
    if (
        ((name := obj.get("name")) is not None)
        and ((args := obj.get("arguments")) is not None)
        and isinstance(name, str)
    ):
        return ToolCallItem(name=name, arguments=json.dumps(args))
    else:
        raise ValidationError


EXO_RUNNER_MUST_FAIL = "EXO RUNNER MUST FAIL"
EXO_RUNNER_MUST_OOM = "EXO RUNNER MUST OOM"
EXO_RUNNER_MUST_TIMEOUT = "EXO RUNNER MUST TIMEOUT"


def _check_for_debug_prompts(
    prompt: str | ChatCompletionMessageText | list[ChatCompletionMessageText],
):
    if isinstance(prompt, list):
        if len(prompt) == 0:
            logger.debug("Empty message prompt received in debug prompt")
            return
        prompt = prompt[0]

    if isinstance(prompt, ChatCompletionMessageText):
        prompt = prompt.text

    if EXO_RUNNER_MUST_FAIL in prompt:
        logger.info("raising exception")
        raise Exception("Artificial runner exception - for testing purposes only.")
    if EXO_RUNNER_MUST_OOM in prompt:
        mlx_force_oom()
    if EXO_RUNNER_MUST_TIMEOUT in prompt:
        time.sleep(100)
