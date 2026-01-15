import base64
import gc
import time
from typing import Literal

import mlx.core as mx
from anyio import WouldBlock

from exo.shared.constants import EXO_MAX_CHUNK_SIZE
from exo.shared.models.model_cards import ModelId, ModelTask
from exo.shared.types.api import ChatCompletionMessageText, ImageGenerationStats
from exo.shared.types.chunks import ImageChunk, TokenChunk
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
    ImageGenerationResponse,
    PartialImageResponse,
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
from exo.worker.engines.mlx.generator.batch_engine import BatchGenerationEngine
from exo.worker.engines.mlx.generator.generate import warmup_inference
from exo.worker.engines.mlx.generator.time_budget import TimeBudget
from exo.worker.engines.mlx.utils_mlx import (
    initialize_mlx,
    load_mlx_items,
    mlx_force_oom,
)
from exo.worker.runner.bootstrap import logger


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
    pending_shutdown: Shutdown | None = None

    current_status: RunnerStatus = RunnerIdle()

    def send_status(status: RunnerStatus) -> None:
        event_sender.send(
            RunnerStatusUpdated(runner_id=runner_id, runner_status=status)
        )

    logger.info("runner created")
    send_status(current_status)

    def handle_task(task: Task, is_deferred: bool = False) -> bool:
        nonlocal current_status, model, tokenizer, group, batch_engine, pending_shutdown

        # For Shutdown, check if we need to defer BEFORE sending Running/Acknowledged
        if (
            isinstance(task, Shutdown)
            and not is_deferred
            and batch_engine is not None
            and (batch_engine.has_active_requests or batch_engine.has_pending_inserts)
        ):
            logger.info("deferring shutdown until active requests complete")
            pending_shutdown = task
            return True

        event_sender.send(
            TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Running)
        )
        event_sender.send(TaskAcknowledged(task_id=task.task_id))

        match task:
            case ConnectToGroup() if isinstance(
                current_status, (RunnerIdle, RunnerFailed)
            ):
                logger.info("runner connecting")
                current_status = RunnerConnecting()
                send_status(current_status)
                group = initialize_mlx(bound_instance)

                logger.info("runner connected")
                current_status = RunnerConnected()
                event_sender.send(
                    TaskStatusUpdated(
                        task_id=task.task_id, task_status=TaskStatus.Complete
                    )
                )
                send_status(current_status)

            case LoadModel() if (
                isinstance(current_status, RunnerConnected) and group is not None
            ) or (isinstance(current_status, RunnerIdle) and group is None):
                current_status = RunnerLoading()
                logger.info("runner loading")
                send_status(current_status)

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
                event_sender.send(
                    TaskStatusUpdated(
                        task_id=task.task_id, task_status=TaskStatus.Complete
                    )
                )
                send_status(current_status)

            case StartWarmup() if isinstance(current_status, RunnerLoaded):
                assert model is not None
                current_status = RunnerWarmingUp()
                logger.info("runner warming up")
                send_status(current_status)

                logger.info(f"warming up inference for instance: {instance}")
                if ModelTask.TextGeneration in shard_metadata.model_card.tasks:
                    assert not isinstance(model, DistributedImageModel)
                    assert tokenizer is not None
                    toks = warmup_inference(model=model, tokenizer=tokenizer)
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
                        logger.info(f"warmed up by generating {image.size} image")
                    else:
                        logger.info("warmup completed (non-primary node)")

                current_status = RunnerReady()
                logger.info("runner ready")
                event_sender.send(
                    TaskStatusUpdated(
                        task_id=task.task_id, task_status=TaskStatus.Complete
                    )
                )
                send_status(current_status)

            case ChatCompletion(task_params=task_params, command_id=command_id) if (
                isinstance(current_status, (RunnerReady, RunnerRunning))
            ):
                assert batch_engine is not None

                # In distributed mode, only rank 0 should queue requests
                # Other ranks should skip - they'll participate in sync_and_insert_pending()
                is_distributed_mode = group is not None and group.size() > 1
                if is_distributed_mode and shard_metadata.device_rank != 0:
                    logger.debug(
                        f"Rank {shard_metadata.device_rank} skipping ChatCompletionTask (only rank 0 queues)"
                    )
                    return True

                if task_params.messages and task_params.messages[0].content is not None:
                    _check_for_debug_prompts(task_params.messages[0].content)

                # Queue the request - actual insertion happens in sync_and_insert_pending()
                batch_engine.queue_request(
                    command_id=command_id, task_id=task.task_id, task_params=task_params
                )

                # Status will be updated after actual insertion in the main loop
                # For now, set to RunnerRunning to indicate we're processing
                current_status = RunnerRunning(
                    active_requests=batch_engine.active_count
                    + batch_engine.pending_insert_count
                )
                send_status(current_status)

            case ImageGeneration(
                task_params=task_params, command_id=command_id
            ) if isinstance(current_status, RunnerReady):
                assert isinstance(model, DistributedImageModel)
                logger.info(f"received image generation request: {str(task)[:500]}")
                current_status = RunnerRunning()
                logger.info("runner running")
                send_status(current_status)

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
                except Exception as e:
                    if shard_metadata.device_rank == shard_metadata.world_size - 1:
                        event_sender.send(
                            ChunkGenerated(
                                command_id=command_id,
                                chunk=ImageChunk(
                                    idx=0,
                                    model=shard_metadata.model_card.model_id,
                                    data="",
                                    chunk_index=0,
                                    total_chunks=1,
                                    image_index=0,
                                    finish_reason="error",
                                    error_message=str(e),
                                ),
                            )
                        )
                    raise

                current_status = RunnerReady()
                logger.info("runner ready")
                event_sender.send(
                    TaskStatusUpdated(
                        task_id=task.task_id, task_status=TaskStatus.Complete
                    )
                )
                send_status(current_status)

            case ImageEdits(task_params=task_params, command_id=command_id) if (
                isinstance(current_status, RunnerReady)
            ):
                assert isinstance(model, DistributedImageModel)
                logger.info(f"received image edits request: {str(task)[:500]}")
                current_status = RunnerRunning()
                logger.info("runner running")
                send_status(current_status)

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
                                chunk=ImageChunk(
                                    idx=0,
                                    model=shard_metadata.model_card.model_id,
                                    data="",
                                    chunk_index=0,
                                    total_chunks=1,
                                    image_index=0,
                                    finish_reason="error",
                                    error_message=str(e),
                                ),
                            )
                        )
                    raise

                current_status = RunnerReady()
                logger.info("runner ready")
                event_sender.send(
                    TaskStatusUpdated(
                        task_id=task.task_id, task_status=TaskStatus.Complete
                    )
                )
                send_status(current_status)

            case Shutdown():
                current_status = RunnerShuttingDown()
                logger.info("runner shutting down")
                send_status(current_status)
                event_sender.send(
                    TaskStatusUpdated(
                        task_id=task.task_id, task_status=TaskStatus.Complete
                    )
                )
                current_status = RunnerShutdown()
                send_status(current_status)
                return False

            case _:
                raise ValueError(
                    f"Received {task.__class__.__name__} outside of state machine in {current_status=}"
                )

        return True

    with task_receiver as tasks:
        running = True
        is_rank_0 = shard_metadata.device_rank == 0

        while running:
            # Use batch_engine.is_distributed since it's set correctly after group initialization
            # (the group variable is None at loop start, but set by ConnectToGroup task)
            if batch_engine is not None and batch_engine.is_distributed:
                assert group is not None
                assert batch_engine is not None

                # Distributed mode: tasks wake up all ranks, then we sync and generate

                # Check deferred shutdown FIRST - all ranks must check and process together
                # This must run before any collective operations to prevent deadlock
                if (
                    pending_shutdown is not None
                    and not batch_engine.has_active_requests
                    and not batch_engine.has_pending_inserts
                ):
                    handle_task(pending_shutdown, is_deferred=True)
                    running = False
                    continue

                # When idle, block waiting for task (exo sends tasks to all ranks)
                # When active, poll non-blocking to batch incoming requests
                if (
                    not batch_engine.has_active_requests
                    and not batch_engine.has_pending_inserts
                ):
                    # IDLE: Block until task arrives (all ranks receive the same task)
                    task = tasks.receive()
                    task_result = handle_task(task)
                    if not task_result:
                        running = False
                        continue
                else:
                    # ACTIVE: Poll for new tasks without blocking
                    while True:
                        try:
                            task = tasks.receive_nowait()
                            task_result = handle_task(task)
                            if not task_result:
                                running = False
                                break
                        except WouldBlock:
                            break
                    if not running:
                        continue

                # Sync and insert pending requests (collective operation)
                # Rank 0 broadcasts its pending to all ranks
                inserted = batch_engine.sync_and_insert_pending()
                if is_rank_0 and inserted:
                    current_status = RunnerRunning(
                        active_requests=batch_engine.active_count
                    )
                    send_status(current_status)

                # Run generation for time budget
                if batch_engine.has_active_requests:
                    time_budget = TimeBudget(budget=0.5, group=group)
                    for _ in time_budget:
                        if not batch_engine.has_active_requests:
                            break
                        for resp in batch_engine.step():
                            # Send token IMMEDIATELY for smooth streaming (only rank 0)
                            if is_rank_0:
                                event_sender.send(
                                    ChunkGenerated(
                                        command_id=resp.command_id,
                                        chunk=TokenChunk(
                                            idx=resp.response.token,
                                            model=shard_metadata.model_card.model_id,
                                            text=resp.response.text,
                                            token_id=resp.response.token,
                                            finish_reason=resp.response.finish_reason,
                                            stats=resp.response.stats,
                                        ),
                                    )
                                )
                                if resp.response.finish_reason is not None:
                                    event_sender.send(
                                        TaskStatusUpdated(
                                            task_id=resp.task_id,
                                            task_status=TaskStatus.Complete,
                                        )
                                    )

                # Sync completions at budget boundary (always call - it's a collective operation)
                batch_engine.sync_completions()

                # Update status after budget
                if is_rank_0:
                    current_status = (
                        RunnerRunning(active_requests=batch_engine.active_count)
                        if batch_engine.has_active_requests
                        else RunnerReady()
                    )
                    send_status(current_status)

            elif batch_engine is not None:
                # Non-distributed mode with batch engine: original logic with queue + insert
                while True:
                    try:
                        task = tasks.receive_nowait()
                        running = handle_task(task)
                        if not running:
                            break
                    except WouldBlock:
                        break

                if not running:
                    break

                # Insert any queued requests (non-distributed just inserts directly)
                # Status was already sent in handle_task when queueing
                if batch_engine.has_pending_inserts:
                    batch_engine.sync_and_insert_pending()

                if batch_engine.has_active_requests:
                    for resp in batch_engine.step():
                        if shard_metadata.device_rank == 0:
                            event_sender.send(
                                ChunkGenerated(
                                    command_id=resp.command_id,
                                    chunk=TokenChunk(
                                        idx=resp.response.token,
                                        model=shard_metadata.model_card.model_id,
                                        text=resp.response.text,
                                        token_id=resp.response.token,
                                        finish_reason=resp.response.finish_reason,
                                        stats=resp.response.stats,
                                    ),
                                )
                            )
                        if resp.response.finish_reason is not None:
                            event_sender.send(
                                TaskStatusUpdated(
                                    task_id=resp.task_id,
                                    task_status=TaskStatus.Complete,
                                )
                            )

                    if batch_engine.has_active_requests:
                        current_status = RunnerRunning(
                            active_requests=batch_engine.active_count
                        )
                    else:
                        current_status = RunnerReady()
                    send_status(current_status)

                    # Process deferred shutdown after all requests complete
                    if (
                        pending_shutdown is not None
                        and not batch_engine.has_active_requests
                        and not batch_engine.has_pending_inserts
                    ):
                        running = handle_task(pending_shutdown, is_deferred=True)
                else:
                    task = tasks.receive()
                    running = handle_task(task)
            else:
                # No batch engine (image generation mode): simple synchronous handling
                task = tasks.receive()
                running = handle_task(task)

    # Cleanup
    del model, tokenizer, group, batch_engine
    mx.clear_cache()
    gc.collect()


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
                    idx=chunk_index,
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
