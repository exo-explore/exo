"""Command types for exo.

Commands are registered dynamically via the command_registry, allowing plugins
to add their own command types without modifying this file.
"""

from typing import Any, cast

from pydantic import Field, field_validator

from exo.plugins.type_registry import command_registry
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
    """Base class for all commands."""

    command_id: CommandId = Field(default_factory=CommandId)


@command_registry.register
class TestCommand(BaseCommand):
    __test__ = False


@command_registry.register
class ChatCompletion(BaseCommand):
    request_params: ChatCompletionTaskParams


@command_registry.register
class ImageGeneration(BaseCommand):
    request_params: ImageGenerationTaskParams


@command_registry.register
class ImageEdits(BaseCommand):
    request_params: ImageEditsInternalParams


@command_registry.register
class PlaceInstance(BaseCommand):
    model_card: ModelCard
    sharding: Sharding
    instance_meta: InstanceMeta
    min_nodes: int


@command_registry.register
class CreateInstance(BaseCommand):
    instance: Instance


@command_registry.register
class DeleteInstance(BaseCommand):
    instance_id: InstanceId


@command_registry.register
class TaskFinished(BaseCommand):
    finished_command_id: CommandId


@command_registry.register
class SendInputChunk(BaseCommand):
    """Command to send an input image chunk (converted to event by master)."""

    chunk: InputImageChunk


@command_registry.register
class RequestEventLog(BaseCommand):
    since_idx: int


# Union type for core commands - used by ForwarderCommand for network deserialization
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
    """Wrapper for commands that includes origin node."""

    origin: NodeId
    command: BaseCommand

    @field_validator("command", mode="before")
    @classmethod
    def validate_command(cls, v: Any) -> BaseCommand:  # noqa: ANN401  # pyright: ignore[reportAny]
        """Validate command, using registry for plugin commands not in Command union."""
        # First try the registry (handles both core and plugin commands)
        return cast(BaseCommand, command_registry.deserialize(v))  # pyright: ignore[reportAny]
