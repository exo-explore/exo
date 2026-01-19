from pydantic import Field

from exo.shared.types.api import ChatCompletionTaskParams
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.shards import Sharding
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel


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
    draft_model: ModelId | None = None  # For speculative decoding
    num_draft_tokens: int = 4  # Tokens to draft per iteration


class CreateInstance(BaseCommand):
    instance: Instance


class DeleteInstance(BaseCommand):
    instance_id: InstanceId


class SetInstanceDraftModel(BaseCommand):
    """Set or update the draft model for an existing instance."""

    instance_id: InstanceId
    draft_model: ModelId | None  # None to disable speculative decoding
    num_draft_tokens: int = 4


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
    | SetInstanceDraftModel
    | TaskFinished
)


class ForwarderCommand(CamelCaseModel):
    origin: NodeId
    command: Command
