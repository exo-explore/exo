import contextlib
from collections import deque
from collections.abc import Generator, Iterable
from dataclasses import dataclass, field
from typing import BinaryIO

import mlx.core as mx
from loguru import logger

from exo.api.types import ImageEditsTaskParams, ImageGenerationTaskParams
from exo.shared.constants import EXO_TRACING_ENABLED
from exo.shared.tracing import clear_trace_buffer, get_trace_buffer
from exo.shared.types.chunks import Chunk, ErrorChunk
from exo.shared.types.events import (
    Event,
    TraceEventData,
    TracesCollected,
    TransientEvent,
)
from exo.shared.types.tasks import (
    GenerationTask,
    ImageEdits,
    ImageGeneration,
    ImageTask,
    TaskId,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    ModelLoadingResponse,
)
from exo.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    ShardMetadata,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Builder, Engine
from exo.worker.engines.image.distributed_model import (
    DistributedImageModel,
)
from exo.worker.engines.image.generate import (
    generate_image,
    warmup_image_generator,
)
from exo.worker.engines.mlx.utils_mlx import (
    initialize_mlx,
)


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


def _send_traces_if_enabled(
    event_sender: MpSender[Event | TransientEvent],
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


@dataclass
class MfluxBuilder(Builder):
    event_sender: MpSender[Event | TransientEvent]
    cancel_receiver: MpReceiver[TaskId]
    shard_metadata: ShardMetadata | None = None
    image_model: DistributedImageModel | None = None
    group: mx.distributed.Group | None = None

    def connect(self, bound_instance: BoundInstance) -> None:
        self.group = initialize_mlx(bound_instance)

    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        self.shard_metadata = bound_instance.bound_shard
        self.image_model = DistributedImageModel.from_shard_metadata(
            bound_instance.bound_shard, self.group
        )
        return
        # very important!
        yield

    def close(self) -> None:
        with contextlib.suppress(NameError, AttributeError):
            del self.image_model, self.group

    def build(
        self,
    ) -> Engine:
        assert self.image_model
        assert self.shard_metadata

        return ImageEngine(
            self.image_model,
            self.shard_metadata,
            self.event_sender,
            self.cancel_receiver,
        )


@dataclass
class ImageEngine(Engine):
    image_model: DistributedImageModel
    shard_metadata: ShardMetadata
    event_sender: MpSender[Event | TransientEvent]
    cancel_receiver: MpReceiver[TaskId]
    current_gen: (
        Generator[tuple[TaskId, Chunk | FinishedResponse | CancelledResponse]] | None
    ) = field(init=False, default=None)
    queue: deque[ImageTask] = field(init=False, default_factory=deque)

    def warmup(self) -> None:
        image = warmup_image_generator(model=self.image_model)
        if image is not None:
            logger.info(f"warmed up by generating {image.size} image")
        else:
            logger.info("warmup completed (non-primary node)")

    def submit(
        self,
        task: GenerationTask,
    ) -> None:
        assert isinstance(task, (ImageGeneration, ImageEdits))
        self.queue.append(task)

    def step(
        self,
    ) -> Iterable[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]]:
        resp = None
        if self.current_gen is not None:
            resp = next(self.current_gen, None)
        if resp is None and len(self.queue) > 0:
            task = self.queue.popleft()
            self.current_gen = self._run_image_task(task.task_id, task.task_params)
            resp = next(self.current_gen, None)
        return (resp,) if resp is not None else ()

    def close(self) -> None:
        with contextlib.suppress(NameError, AttributeError):
            del self.image_model

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        raise NotImplementedError() from None

    def _run_image_task(
        self,
        task_id: TaskId,
        task_params: ImageGenerationTaskParams | ImageEditsTaskParams,
    ) -> Generator[tuple[TaskId, Chunk | FinishedResponse | CancelledResponse]]:
        assert self.image_model
        logger.info(f"received image task: {str(task_params)[:500]}")

        def cancel_checker() -> bool:
            for cancel_id in self.cancel_receiver.collect():
                self._cancelled_tasks.add(cancel_id)
            return self.should_cancel(task_id)

        try:
            # todo: yield CancelledResponse properly
            for response in generate_image(
                model=self.image_model,
                task=task_params,
                cancel_checker=cancel_checker,
            ):
                if _is_primary_output_node(self.shard_metadata):
                    yield (task_id, response)
        except Exception as e:
            if _is_primary_output_node(self.shard_metadata):
                yield (
                    task_id,
                    ErrorChunk(
                        model=self.shard_metadata.model_card.model_id,
                        finish_reason="error",
                        error_message=str(e),
                    ),
                )
            raise
        finally:
            _send_traces_if_enabled(
                self.event_sender, task_id, self.shard_metadata.device_rank
            )
            yield (task_id, FinishedResponse())

        return
