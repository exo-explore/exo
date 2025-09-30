# pyright: reportAny=false, reportPrivateUsage=false, reportUnusedParameter=false, reportUnknownMemberType=false

from collections.abc import Callable
from types import get_original_bases
from typing import (
    Any,
    ClassVar,
    Self,
    Union,
    cast,
    get_args,
    get_origin,
)

import pydantic
from bidict import bidict
from pydantic import (
    BaseModel,
    Field,
    TypeAdapter,
    model_serializer,
    model_validator,
)
from pydantic_core import (
    PydanticCustomError,
)


def tagged_union[T: Tagged[Any]](
    type_map: dict[str, type],
) -> Callable[[type[T]], type[T]]:
    def _decorator(cls: type[T]):
        # validate and process the types
        tagged_union_cls = _ensure_single_tagged_union_base(cls)
        adapter_dict = _ensure_tagged_union_generic_is_union(tagged_union_cls)
        type_bidict = _ensure_bijection_between_union_members_and_type_map(
            set(adapter_dict.keys()), type_map
        )

        # inject the adapter and type class variables
        cast(type[_TaggedImpl[Any]], cls)._type_bidict = type_bidict
        cast(type[_TaggedImpl[Any]], cls)._adapter_dict = adapter_dict

        return cls

    return _decorator


class Tagged[C](BaseModel):
    """
    Utility for helping with serializing unions as adjacently tagged with Pydantic.

    By default, Pydantic uses internally tagged union ser/de BUT to play nicely with
    other cross-language ser/de tools, you need adjacently tagged unions, and Pydantic
    doesn't support those out of the box.
    SEE: https://serde.rs/enum-representations.html#adjacently-tagged

    This type is a Pydantic model in its own right and can be used on fields of other
    Pydantic models. It must be used in combination with `tagged_union` decorator to work.

    Example usage:
    ```python
    FoobarUnion = Union[Foo, Bar, Baz]

    @tagged_union({
        "Foo": Foo,
        "Bar": Bar,
        "Baz": Baz,
    })
    class TaggedFoobarUnion(Tagged[FoobarUnion]): ...
    ```
    """

    t: str = Field(frozen=True)
    """
    The tag corresponding to the type of the object in the union.
    """

    c: C = Field(frozen=True)
    """
    The actual content of the object of that type.
    """

    @classmethod
    def from_(cls, c: C) -> Self:
        t = cast(type[_TaggedImpl[C]], cls)._type_bidict.inv[type(c)]
        return cls(t=t, c=c)

    @model_serializer
    def _model_dump(self) -> dict[str, Any]:
        cls = type(cast(_TaggedImpl[C], self))
        adapter = cls._adapter_dict[cls._type_bidict[self.t]]
        return {
            "t": self.t,
            "c": adapter.dump_python(self.c),
        }

    @model_validator(mode="before")
    @classmethod
    def _model_validate_before(cls, data: Any) -> Any:
        cls = cast(type[_TaggedImpl[C]], cls)

        # check object shape & check "t" type is `str`
        if not isinstance(data, dict):
            raise PydanticCustomError(
                "dict_type", "Wrong object type: expected a dictionary type"
            )
        if "t" not in data or "c" not in data or len(data) != 2:  # pyright: ignore[reportUnknownArgumentType]
            raise ValueError(
                "Wrong object shape: expected exactly {t: <tag>, c: <content>}"
            )
        if not isinstance(data["t"], str):
            raise PydanticCustomError(
                "string_type", 'Wrong field type: expected "t" to be `str`'
            )

        # grab tag & content keys + look up the type based on the tag
        t = data["t"]
        c = cast(Any, data["c"])
        ccls = cls._type_bidict.get(t)
        if ccls is None:
            raise PydanticCustomError(
                "union_tag_not_found",
                'Wrong "t"-value: could not find tag within this discriminated union',
            )
        cadapter = cls._adapter_dict[ccls]

        return {
            "t": t,
            "c": cadapter.validate_python(c),
        }

    @model_validator(mode="after")
    def _model_validate_after(self) -> Self:
        cls = type(cast(_TaggedImpl[C], self))
        ccls = type(self.c)

        # sanity check for consistency
        t = cls._type_bidict.inv.get(ccls)
        if t is None:
            raise ValueError(
                'Wrong "c"-value: could not find a tag corresponding to the type of this value'
            )
        if t != self.t:
            raise ValueError(
                'Wrong "t"-value: the provided tag for this content\'s type mismatches the configured tag'
            )

        return self


class _TaggedImpl[C](Tagged[C]):
    _type_bidict: ClassVar[bidict[str, type]]
    _adapter_dict: ClassVar[dict[type, TypeAdapter[Any]]]


def _ensure_single_tagged_union_base(cls: type[Any]) -> type[Any]:
    bases = get_original_bases(cls)

    # count up all the bases (generic removed) and store last found one
    cnt = 0
    last = None
    for b in bases:
        if pydantic._internal._generics.get_origin(b) == Tagged:  # pyright: ignore[reportAttributeAccessIssue]
            last = cast(type[Tagged[Any]], b)
            cnt += 1

    # sanity-check the bases
    if last is None:
        raise TypeError(f"Expected {Tagged!r} to be a base-class of {cls!r}")
    if cnt > 1:
        raise TypeError(
            f"Expected only one {Tagged!r} base-class of {cls!r}, but got {cnt}"
        )

    return last


def _ensure_tagged_union_generic_is_union(
    cls: type[Any],
) -> dict[type, TypeAdapter[Any]]:
    # extract type of the generic argument
    base_generics = cast(Any, pydantic._internal._generics.get_args(cls))  # pyright: ignore[reportAttributeAccessIssue]
    assert len(base_generics) == 1
    union_cls = base_generics[0]

    # ensure the generic is a union => extract the members
    union_origin = get_origin(union_cls)
    if union_origin != Union:
        raise TypeError(
            f"Expected {Tagged!r} base-class to have its generic be a {Union!r}, but got {union_cls!r}"
        )
    union_members = get_args(union_cls)

    # typecheck each of the members, creating a type<->adapter mapping
    adapter_dict: dict[type, TypeAdapter[Any]] = {}
    for m in union_members:
        if not isinstance(m, type):
            raise TypeError(f"Expected union member {m!r} to be a type")
        adapter_dict[m] = TypeAdapter(m)

    return adapter_dict


def _ensure_bijection_between_union_members_and_type_map(
    members: set[type], type_map: dict[str, type]
) -> bidict[str, type]:
    mapped_members = set(type_map.values())

    illegal_members = mapped_members - members
    for m in illegal_members:
        raise TypeError(
            f"Expected type-map member {m!r} to be member of the union, but is not"
        )
    missing_members = members - mapped_members
    for m in missing_members:
        raise TypeError(
            f"Expected type-map to include a tag for member {m!r}, but is missing"
        )
    assert mapped_members == members

    tag_sets = {m: {t for t in type_map if type_map[t] == m} for m in mapped_members}
    for m, ts in tag_sets.items():
        if len(ts) > 1:
            raise TypeError(
                f"Expected a single tag per member of the union, but found {ts} for member {m!r}"
            )

    return bidict(type_map)
