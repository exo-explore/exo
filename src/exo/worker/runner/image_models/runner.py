import base64
import time
from typing import TYPE_CHECKING, Literal

import mlx.core as mx
from exo_core.constants import EXO_MAX_CHUNK_SIZE, EXO_TRACING_ENABLED
from exo_core.model_cards import ModelTask
from exo_core.types.chunks import ErrorChunk, ImageChunk
from exo_core.types.common import CommandId, ModelId
from exo_core.types.instances import BoundInstance
from exo_core.types.runner_response import (
    ImageGenerationResponse,
    PartialImageResponse,
)
from exo_core.types.runners import (
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
from exo_core.types.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    ShardMetadata,
)
from exo_core.types.tasks import (
    CANCEL_ALL_TASKS,
    ConnectToGroup,
    ImageEdits,
    ImageGeneration,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
)

from exo_core.types.runner_response import ImageGenerationStats
from exo.shared.tracing import clear_trace_buffer, get_trace_buffer
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
    TraceEventData,
    TracesCollected,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.image import (
    DistributedImageModel,
    generate_image,
    initialize_image_model,
    warmup_image_generator,
)
from mlx_engine.utils_mlx import (
    initialize_mlx,
)
from loguru import logger


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


class Runner:
    def __init__(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
        cancel_receiver: MpReceiver[TaskId],
    ):
        self.event_sender = event_sender
        self.task_receiver = task_receiver
        self.cancel_receiver = cancel_receiver
        self.bound_instance = bound_instance

        self.instance, self.runner_id, self.shard_metadata = (
            bound_instance.instance,
            bound_instance.bound_runner_id,
            bound_instance.bound_shard,
        )
        self.device_rank = self.shard_metadata.device_rank

        logger.info("hello from the runner")
        if getattr(self.shard_metadata, "immediate_exception", False):
            raise Exception("Fake exception - runner failed to spin up.")
        if timeout := getattr(self.shard_metadata, "should_timeout", 0):
            time.sleep(timeout)

        self.setup_start_time = time.time()
        self.cancelled_tasks = set[TaskId]()

        self.image_model: DistributedImageModel | None = None
        self.group = None

        self.current_status: RunnerStatus = RunnerIdle()
        logger.info("runner created")
        self.update_status(RunnerIdle())
        self.seen = set[TaskId]()

    def update_status(self, status: RunnerStatus):
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(
                runner_id=self.runner_id, runner_status=self.current_status
            )
        )

    def send_task_status(self, task: Task, status: TaskStatus):
        self.event_sender.send(
            TaskStatusUpdated(task_id=task.task_id, task_status=status)
        )

    def acknowledge_task(self, task: Task):
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    def main(self):
        with self.task_receiver as tasks:
            for task in tasks:
                if task.task_id in self.seen:
                    logger.warning("repeat task - potential error")
                self.seen.add(task.task_id)
                self.cancelled_tasks.discard(CANCEL_ALL_TASKS)
                self.send_task_status(task, TaskStatus.Running)
                self.handle_task(task)
                was_cancelled = (task.task_id in self.cancelled_tasks) or (
                    CANCEL_ALL_TASKS in self.cancelled_tasks
                )
                if not was_cancelled:
                    self.send_task_status(task, TaskStatus.Complete)
                self.update_status(self.current_status)

                if isinstance(self.current_status, RunnerShutdown):
                    break

    def handle_task(self, task: Task):
        match task:
            case ConnectToGroup() if isinstance(
                self.current_status, (RunnerIdle, RunnerFailed)
            ):
                logger.info("runner connecting")
                self.update_status(RunnerConnecting())
                self.acknowledge_task(task)
                self.group = initialize_mlx(self.bound_instance)

                logger.info("runner connected")
                self.current_status = RunnerConnected()

            # we load the model if it's connected with a group, or idle without a group. we should never tell a model to connect if it doesn't need to
            case LoadModel() if (
                isinstance(self.current_status, RunnerConnected)
                and self.group is not None
            ) or (isinstance(self.current_status, RunnerIdle) and self.group is None):
                logger.info("runner loading")
                self.update_status(RunnerLoading())
                self.acknowledge_task(task)

                assert (
                    ModelTask.TextToImage in self.shard_metadata.model_card.tasks
                    or ModelTask.ImageToImage in self.shard_metadata.model_card.tasks
                ), f"Incorrect model task(s): {self.shard_metadata.model_card.tasks}"

                self.image_model = initialize_image_model(self.bound_instance)
                self.current_status = RunnerLoaded()
                logger.info("runner loaded")

            case StartWarmup() if isinstance(self.current_status, RunnerLoaded):
                logger.info("runner warming up")
                self.update_status(RunnerWarmingUp())
                self.acknowledge_task(task)

                logger.info(f"warming up inference for instance: {self.instance}")

                assert self.image_model
                image = warmup_image_generator(model=self.image_model)
                if image is not None:
                    logger.info(f"warmed up by generating {image.size} image")
                else:
                    logger.info("warmup completed (non-primary node)")

                logger.info(
                    f"runner initialized in {time.time() - self.setup_start_time} seconds"
                )

                self.current_status = RunnerReady()
                logger.info("runner ready")

            case ImageGeneration(task_params=task_params, command_id=command_id) if (
                isinstance(self.current_status, RunnerReady)
            ):
                assert self.image_model
                logger.info(f"received image generation request: {str(task)[:500]}")
                logger.info("runner running")
                self.update_status(RunnerRunning())
                self.acknowledge_task(task)

                try:
                    image_index = 0
                    for response in generate_image(
                        model=self.image_model, task=task_params
                    ):
                        is_primary_output = _is_primary_output_node(self.shard_metadata)

                        if is_primary_output:
                            match response:
                                case PartialImageResponse():
                                    logger.info(
                                        f"sending partial ImageChunk {response.partial_index}/{response.total_partials}"
                                    )
                                    _process_image_response(
                                        response,
                                        command_id,
                                        self.shard_metadata,
                                        self.event_sender,
                                        image_index,
                                    )
                                case ImageGenerationResponse():
                                    logger.info("sending final ImageChunk")
                                    _process_image_response(
                                        response,
                                        command_id,
                                        self.shard_metadata,
                                        self.event_sender,
                                        image_index,
                                    )
                                    image_index += 1
                # can we make this more explicit?
                except Exception as e:
                    if _is_primary_output_node(self.shard_metadata):
                        self.event_sender.send(
                            ChunkGenerated(
                                command_id=command_id,
                                chunk=ErrorChunk(
                                    model=self.shard_metadata.model_card.model_id,
                                    finish_reason="error",
                                    error_message=str(e),
                                ),
                            )
                        )
                    raise
                finally:
                    _send_traces_if_enabled(
                        self.event_sender, task.task_id, self.device_rank
                    )

                self.current_status = RunnerReady()
                logger.info("runner ready")

            case ImageEdits(task_params=task_params, command_id=command_id) if (
                isinstance(self.current_status, RunnerReady)
            ):
                assert self.image_model
                logger.info(f"received image edits request: {str(task)[:500]}")
                logger.info("runner running")
                self.update_status(RunnerRunning())
                self.acknowledge_task(task)

                try:
                    image_index = 0
                    for response in generate_image(
                        model=self.image_model, task=task_params
                    ):
                        if _is_primary_output_node(self.shard_metadata):
                            match response:
                                case PartialImageResponse():
                                    logger.info(
                                        f"sending partial ImageChunk {response.partial_index}/{response.total_partials}"
                                    )
                                    _process_image_response(
                                        response,
                                        command_id,
                                        self.shard_metadata,
                                        self.event_sender,
                                        image_index,
                                    )
                                case ImageGenerationResponse():
                                    logger.info("sending final ImageChunk")
                                    _process_image_response(
                                        response,
                                        command_id,
                                        self.shard_metadata,
                                        self.event_sender,
                                        image_index,
                                    )
                                    image_index += 1
                except Exception as e:
                    if _is_primary_output_node(self.shard_metadata):
                        self.event_sender.send(
                            ChunkGenerated(
                                command_id=command_id,
                                chunk=ErrorChunk(
                                    model=self.shard_metadata.model_card.model_id,
                                    finish_reason="error",
                                    error_message=str(e),
                                ),
                            )
                        )
                    raise
                finally:
                    _send_traces_if_enabled(
                        self.event_sender, task.task_id, self.device_rank
                    )

                self.current_status = RunnerReady()
                logger.info("runner ready")

            case Shutdown():
                logger.info("runner shutting down")
                if not TYPE_CHECKING:
                    del self.image_model, self.group
                    mx.clear_cache()
                    import gc

                    gc.collect()

                self.update_status(RunnerShuttingDown())
                self.acknowledge_task(task)

                self.current_status = RunnerShutdown()
            case _:
                raise ValueError(
                    f"Received {task.__class__.__name__} outside of state machine in {self.current_status=}"
                )
