from typing import Any, Self
from uuid import uuid4

from pydantic import GetCoreSchemaHandler, field_validator
from pydantic_core import CoreSchema, core_schema

from exo.utils.pydantic_ext import CamelCaseModel


class Id(str):
    def __new__(cls, value: str | None = None) -> Self:
        return super().__new__(cls, value or str(uuid4()))

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source: type, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        # Just use a plain string schema
        return core_schema.no_info_after_validator_function(
            cls, core_schema.str_schema()
        )


class NodeId(Id):
    pass


class SystemId(Id):
    pass


class ModelId(Id):
    def normalize(self) -> str:
        return self.replace("/", "--")

    def short(self) -> str:
        return self.split("/")[-1]


class CommandId(Id):
    pass


class TruncatingString(str):
    truncate_length: int = -1

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,  # pyright: ignore[reportAny]
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(str))

    def __repr__(self):
        tl = type(self).truncate_length
        return (
            f"<{type(self).__name__}: {self[:tl] + '...' if len(self) > tl else self}>"
        )


class SessionId(CamelCaseModel):
    master_node_id: NodeId
    election_clock: int


class Host(CamelCaseModel):
    ip: str
    port: int

    def __str__(self) -> str:
        return f"{self.ip}:{self.port}"

    @field_validator("port")
    @classmethod
    def check_port(cls, v: int) -> int:
        if not (0 <= v <= 65535):
            raise ValueError("Port must be between 0 and 65535")
        return v
