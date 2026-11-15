from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class NativeModel(BaseModel):
    model_config = ConfigDict(
        extra="ignore", strict=True, frozen=True, validate_by_name=True
    )


class ServeCapabilities(NativeModel):
    flags: dict[str, dict[str, JsonValue]]


class NativeCapabilities(NativeModel):
    schema_version: Literal[1] = Field(alias="schema")
    engine: Literal["zig"]
    version: str = Field(min_length=1)
    chip: str | None = None
    backends: list[str]
    families: dict[str, list[str]]
    token_stream_versions: list[int]
    serve: ServeCapabilities

    def require_streaming(self) -> None:
        if 1 not in self.token_stream_versions:
            raise ValueError("TensorFold must support token stream protocol version 1")
        missing = {"--unix-socket", "--name"} - self.serve.flags.keys()
        if missing:
            raise ValueError(
                f"TensorFold does not support required flags: {sorted(missing)}"
            )
        if not {"metal", "cuda"}.intersection(self.backends):
            raise ValueError("TensorFold must provide a Metal or CUDA backend")
