from enum import Enum
from typing import Union

from pydantic import Field

from exo.shared.types.api import ChatCompletionTaskParams
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.models import ModelMetadata
from exo.shared.types.worker.common import InstanceId
from exo.utils.pydantic_ext import CamelCaseModel
from exo.utils.pydantic_tagged import Tagged, tagged_union


# TODO: We need to have a distinction between create instance and spin up instance.
class CommandType(str, Enum):
    ChatCompletion = "ChatCompletion"
    CreateInstance = "CreateInstance"
    SpinUpInstance = "SpinUpInstance"
    DeleteInstance = "DeleteInstance"
    TaskFinished = "TaskFinished"
    RequestEventLog = "RequestEventLog"


class BaseCommand(CamelCaseModel):
    command_id: CommandId = Field(default_factory=CommandId)


class ChatCompletion(BaseCommand):
    request_params: ChatCompletionTaskParams


class CreateInstance(BaseCommand):
    model_meta: ModelMetadata


class SpinUpInstance(BaseCommand):
    instance_id: InstanceId


class DeleteInstance(BaseCommand):
    instance_id: InstanceId


class TaskFinished(BaseCommand):
    finished_command_id: CommandId


class RequestEventLog(BaseCommand):
    since_idx: int


Command = Union[
    RequestEventLog,
    ChatCompletion,
    CreateInstance,
    SpinUpInstance,
    DeleteInstance,
    TaskFinished,
]


@tagged_union(
    {
        CommandType.ChatCompletion: ChatCompletion,
        CommandType.CreateInstance: CreateInstance,
        CommandType.SpinUpInstance: SpinUpInstance,
        CommandType.DeleteInstance: DeleteInstance,
        CommandType.TaskFinished: TaskFinished,
        CommandType.RequestEventLog: RequestEventLog,
    }
)
class TaggedCommand(Tagged[Command]):
    pass


class ForwarderCommand(CamelCaseModel):
    origin: NodeId
    tagged_command: TaggedCommand
