from pydantic import Field

from exo.shared.models.model_cards import ModelCard
from exo.shared.types.api import (
    ChatCompletionTaskParams,
    ImageEditsInternalParams,
    ImageGenerationTaskParams,
)
from exo.shared.types.chunks import InputImageChunk
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.shards import Sharding
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel


class BaseCommand(TaggedModel):
    command_id: CommandId = Field(default_factory=CommandId)


class TestCommand(BaseCommand):
    __test__ = False


class ChatCompletion(BaseCommand):
    request_params: ChatCompletionTaskParams


class ImageGeneration(BaseCommand):
    request_params: ImageGenerationTaskParams


class ImageEdits(BaseCommand):
    request_params: ImageEditsInternalParams


class PlaceInstance(BaseCommand):
    model_card: ModelCard
    sharding: Sharding
    instance_meta: InstanceMeta
    min_nodes: int


class CreateInstance(BaseCommand):
    instance: Instance


class DeleteInstance(BaseCommand):
    instance_id: InstanceId


class TaskFinished(BaseCommand):
    finished_command_id: CommandId


class SendInputChunk(BaseCommand):
    """Command to send an input image chunk (converted to event by master)."""

    chunk: InputImageChunk


class RequestEventLog(BaseCommand):
    since_idx: int


Command = (
    TestCommand
    | RequestEventLog
    | ChatCompletion
    | ImageGeneration
    | ImageEdits
    | PlaceInstance
    | CreateInstance
    | DeleteInstance
    | TaskFinished
    | SendInputChunk
)


class ForwarderCommand(CamelCaseModel):
    origin: NodeId
    command: Command
