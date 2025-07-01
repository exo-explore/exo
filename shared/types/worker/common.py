from typing import Annotated
from uuid import UUID

from pydantic import TypeAdapter
from pydantic.types import UuidVersion

_InstanceId = Annotated[UUID, UuidVersion(4)]
InstanceId = type("InstanceId", (UUID,), {})
InstanceIdParser: TypeAdapter[InstanceId] = TypeAdapter(_InstanceId)

_RunnerId = Annotated[UUID, UuidVersion(4)]
RunnerId = type("RunnerId", (UUID,), {})
RunnerIdParser: TypeAdapter[RunnerId] = TypeAdapter(_RunnerId)
