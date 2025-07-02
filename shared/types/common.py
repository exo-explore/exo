from uuid import uuid4

from pydantic import UUID4, Field
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class NewUUID:
    uuid: UUID4 = Field(default_factory=lambda: uuid4())

    def __hash__(self) -> int:
        return hash(self.uuid)


class NodeId(NewUUID):
    pass
