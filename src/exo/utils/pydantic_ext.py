from typing import Any, Self, cast, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    GetCoreSchemaHandler,
    model_serializer,
    model_validator,
)
from pydantic.alias_generators import to_camel
from pydantic_core import core_schema
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
    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler):
        inner = handler(self)  # pyright: ignore[reportAny]
        return {self.__class__.__name__: inner}

    @model_validator(mode="wrap")
    @classmethod
    def _validate(cls, v: Any, handler: ValidatorFunctionWrapHandler) -> Self:  # pyright: ignore[reportAny]
        if isinstance(v, dict) and len(v) == 1 and cls.__name__ in v:  # pyright: ignore[reportUnknownArgumentType]
            return handler(v[cls.__name__])  # pyright: ignore[reportAny]

        return handler(v)  # pyright: ignore[reportAny]

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({super().__str__()})"


class ByTag:
    """Marks a union of TaggedModels, `Annotated[A | B | C, ByTag()]`, to be validated as the
    member its tag names, without trying the others.

    The tag is what a TaggedModel is serialized under, `{"A": {...}}`, or an object's own class.
    Without it pydantic tries the members in turn, and every member that isn't the one costs
    a failed validation: parsing an event through the 22 members of Event took about ten
    times as long as parsing it as its own type. The serialized form doesn't change.
    """

    def __get_pydantic_core_schema__(
        self, source: object, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        members: tuple[type[TaggedModel], ...] = get_args(source)
        return core_schema.tagged_union_schema(
            {member.__name__: handler.generate_schema(member) for member in members},
            discriminator=_tag,
        )


def _tag(value: object) -> str | None:
    if isinstance(value, dict):
        # A serialized TaggedModel has its class name as its only key
        keys = list(cast(dict[object, object], value))
        if len(keys) == 1 and isinstance(keys[0], str):
            return keys[0]
        return None
    return type(value).__name__
