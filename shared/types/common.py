from typing import Annotated
from uuid import UUID

from pydantic import TypeAdapter
from pydantic.types import UuidVersion

_NodeId = Annotated[UUID, UuidVersion(4)]
NodeId = type("NodeId", (UUID,), {})
NodeIdParser: TypeAdapter[NodeId] = TypeAdapter(_NodeId)
