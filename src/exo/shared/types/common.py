from typing import Any, Literal, Self
from uuid import uuid4

from pydantic import GetCoreSchemaHandler, field_validator
from pydantic_core import CoreSchema, core_schema

from exo.utils.pydantic_ext import FrozenModel

ModelSourceKind = Literal[
    "exo",
    "huggingface",
    "lmstudio",
    "ollama",
    "llamacpp",
]
"""Identifier for where a locally-available model came from.

- ``exo``: model lives in one of ``EXO_MODELS_DIRS`` and is managed by exo's own downloader.
- ``huggingface``: standard HF cache (``~/.cache/huggingface/hub/``), shared with mlx-lm and modern llama.cpp ``-hf``.
- ``lmstudio``: LM Studio's local library (``~/.lmstudio/models/{publisher}/{model}/``).
- ``ollama``: Ollama's content-addressed store (``~/.ollama/models/manifests/`` + ``blobs/``).
- ``llamacpp``: llama.cpp's standalone GGUF cache.
"""

ModelFileFormat = Literal["safetensors", "mlx", "gguf"]
"""On-disk weight format for a discovered model."""


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


class SessionId(FrozenModel):
    master_node_id: NodeId
    election_clock: int


class Host(FrozenModel):
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
