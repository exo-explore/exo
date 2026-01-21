# ruff: noqa: I001 - Import order intentional to avoid circular imports
from pydantic import Field

from exo.shared.types.api import ChatCompletionTaskParams
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.models import ModelMetadata
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.shards import Sharding
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel

# Import FLASH commands from plugin (for serialization compatibility)
from exo.plugins.implementations.flash.types import LaunchFLASH, StopFLASH  # noqa: E402, I001


class BaseCommand(TaggedModel):
    command_id: CommandId = Field(default_factory=CommandId)


class TestCommand(BaseCommand):
    __test__ = False


class ChatCompletion(BaseCommand):
    request_params: ChatCompletionTaskParams


class PlaceInstance(BaseCommand):
    model_meta: ModelMetadata
    sharding: Sharding
    instance_meta: InstanceMeta
    min_nodes: int


class CreateInstance(BaseCommand):
    instance: Instance


class DeleteInstance(BaseCommand):
    instance_id: InstanceId


class TaskFinished(BaseCommand):
    finished_command_id: CommandId


class RequestEventLog(BaseCommand):
    since_idx: int


Command = (
    TestCommand
    | RequestEventLog
    | ChatCompletion
    | PlaceInstance
    | CreateInstance
    | DeleteInstance
    | LaunchFLASH
    | StopFLASH
    | TaskFinished
)


class ForwarderCommand(CamelCaseModel):
    origin: NodeId
    command: Command
