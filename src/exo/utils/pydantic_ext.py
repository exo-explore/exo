from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_serializer, model_validator
from pydantic.alias_generators import to_camel
from pydantic_core.core_schema import (
    SerializerFunctionWrapHandler,
    ValidatorFunctionWrapHandler,
)


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        validate_by_name=True,
        extra="forbid",
        strict=True,
        frozen=True,
    )


class TaggedModel(FrozenModel):
    @classmethod
    def tag(cls) -> str:
        return cls.__name__

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler):
        inner = handler(self)  # pyright: ignore[reportAny]
        return {self.tag(): inner}

    @model_validator(mode="wrap")
    @classmethod
    def _validate(cls, v: Any, handler: ValidatorFunctionWrapHandler) -> Self:  # pyright: ignore[reportAny]
        if isinstance(v, dict) and len(v) == 1 and cls.tag() in v:  # pyright: ignore[reportUnknownArgumentType]
            return handler(v[cls.tag()])  # pyright: ignore[reportAny]

        return handler(v)  # pyright: ignore[reportAny]

    def __str__(self) -> str:
        return f"{self.tag()}({super().__str__()})"
