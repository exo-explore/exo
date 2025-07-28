from ipaddress import IPv4Address
from typing import Any, Self
from uuid import uuid4

from pydantic import BaseModel, GetCoreSchemaHandler, field_validator
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


class Host(BaseModel):
    ip: IPv4Address
    port: int

    def __str__(self) -> str:
        return f"{self.ip}:{self.port}"

    @field_validator("port")
    @classmethod
    def check_port(cls, v: int) -> int:
        if not (0 <= v <= 65535):
            raise ValueError("Port must be between 0 and 65535")
        return v
