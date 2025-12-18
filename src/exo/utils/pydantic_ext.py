# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_serializer, model_validator
from pydantic.alias_generators import to_camel
from pydantic_core.core_schema import (
    SerializerFunctionWrapHandler,
    ValidatorFunctionWrapHandler,
)


class CamelCaseModel(BaseModel):
    """
    A model whose fields are aliased to camel-case from snake-case.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        validate_by_name=True,
        extra="forbid",
        # I want to reenable this ASAP, but it's causing an issue with TaskStatus
        strict=True,
    )


class TaggedModel(CamelCaseModel):
    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler):
        inner = handler(self)
        return {self.__class__.__name__: inner}

    @model_validator(mode="wrap")
    @classmethod
    def _validate(cls, v: Any, handler: ValidatorFunctionWrapHandler) -> Self:
        if isinstance(v, dict) and len(v) == 1 and cls.__name__ in v:
            return handler(v[cls.__name__])

        return handler(v)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({super().__str__()})"
