from enum import Enum
from typing import (
    TYPE_CHECKING,
    Callable,
    Sequence,
)

if TYPE_CHECKING:
    pass

from pydantic import BaseModel

from shared.types.common import NewUUID
from shared.types.state import State

from . import Event


class CommandId(NewUUID):
    pass


class CommandTypes(str, Enum):
    Create = "Create"
    Update = "Update"
    Delete = "Delete"


class Command[
    CommandType: CommandTypes,
](BaseModel):
    command_type: CommandType
    command_id: CommandId


type Decide[CommandTypeT: CommandTypes] = Callable[
    [State, Command[CommandTypeT]],
    Sequence[Event],
]
