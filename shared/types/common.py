from typing import Any, Self
from uuid import uuid4

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class ID(str):
    def __new__(cls, value: str | None = None) -> Self:
        return super().__new__(cls, value or str(uuid4()))

    @classmethod
    def __get_pydantic_core_schema__(
            cls,
            _source: type[Any],
            handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        # Re‑use the already‑defined schema for `str`
        return handler.generate_schema(str)


class NodeId(ID):
    pass


class CommandId(ID):
    pass
