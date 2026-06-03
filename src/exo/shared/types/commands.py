from pydantic import Field

from exo.api.types import (
    ImageEditsTaskParams,
    ImageGenerationTaskParams,
)
from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.types.common import CommandId, NodeId, SystemId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import Instance, InstanceMeta
from exo.shared.types.worker.shards import Sharding, ShardMetadata
from exo.utils.pydantic_ext import FrozenModel, TaggedModel


class BaseCommand(TaggedModel):
    command_id: CommandId = Field(default_factory=CommandId)


class TestCommand(BaseCommand):
    __test__ = False


class TextGeneration(BaseCommand):
    task_params: TextGenerationTaskParams


class ImageGeneration(BaseCommand):
    task_params: ImageGenerationTaskParams


class ImageEdits(BaseCommand):
    task_params: ImageEditsTaskParams


class PlaceInstance(BaseCommand):
    model_card: ModelCard
    sharding: Sharding
    instance_meta: InstanceMeta
    min_nodes: int


class CreateInstance(BaseCommand):
    instance: Instance


class DeleteInstance(BaseCommand):
    pass


class TaskCancelled(BaseCommand):
    cancelled_command_id: CommandId


class TaskFinished(BaseCommand):
    finished_command_id: CommandId


class RequestEventLog(BaseCommand):
    since_idx: int


class StartDownload(BaseCommand):
    target_node_id: NodeId
    shard_metadata: ShardMetadata


class DeleteDownload(BaseCommand):
    target_node_id: NodeId
    model_id: ModelId


class CancelDownload(BaseCommand):
    target_node_id: NodeId
    model_id: ModelId


DownloadCommand = StartDownload | DeleteDownload | CancelDownload


Command = (
    TestCommand
    | RequestEventLog
    | TextGeneration
    | ImageGeneration
    | ImageEdits
    | PlaceInstance
    | CreateInstance
    | DeleteInstance
    | TaskCancelled
    | TaskFinished
)


class ForwarderCommand(FrozenModel):
    origin: SystemId
    command: Command


class ForwarderDownloadCommand(FrozenModel):
    origin: SystemId
    command: DownloadCommand

class JoinInstance(TaggedModel):
    # TODO: strip this down to less data
    instance: Instance

Mail = JoinInstance
