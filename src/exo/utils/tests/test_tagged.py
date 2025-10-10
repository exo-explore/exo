import anyio
import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from exo.utils.pydantic_ext import TaggedModel


def test_plain_union_prefers_first_member_when_shapes_are_identical():
    class Foo1(BaseModel):
        x: int

    class Foo2(BaseModel):
        x: int

    # Base Pydantic behavior: ambiguous dict goes to the first union member
    ta = TypeAdapter[Foo1 | Foo2](Foo1 | Foo2)
    out = ta.validate_python({"x": 1})
    assert isinstance(out, Foo1), (
        "Base Pydantic should pick the first union member for identical shapes"
    )


def test_tagged_union_serializes_and_deserializes_two_identical_shapes_correctly():
    class Foo1(TaggedModel):
        x: int

    class Foo2(TaggedModel):
        x: int

    t1 = Foo1(x=1)
    assert t1.model_dump() == {"Foo1": {"x": 1}}

    t2 = Foo2(x=2)
    assert t2.model_dump() == {"Foo2": {"x": 2}}

    # ---- deserialize (TypeAdapter -> model_validator(before)) ----
    ta = TypeAdapter[Foo1 | Foo2](Foo1 | Foo2)

    out1 = ta.validate_python({"Foo1": {"x": 10}})
    assert isinstance(out1, Foo1) and out1.x == 10

    out2 = ta.validate_python({"Foo2": {"x": 20}})
    assert isinstance(out2, Foo2) and out2.x == 20


def test_tagged_union_rejects_unknown_tag():
    class Foo1(TaggedModel):
        x: int

    class Foo2(TaggedModel):
        x: int

    ta = TypeAdapter[Foo1 | Foo2](Foo1 | Foo2)
    with pytest.raises(ValidationError):
        ta.validate_python({"NotARealTag": {"x": 0}})


def test_two_tagged_classes_with_different_shapes_are_independent_and_not_cross_deserializable():
    class A1(TaggedModel):
        x: int

    class A2(TaggedModel):
        name: str

    class B1(TaggedModel):
        name: str

    class B2(TaggedModel):
        active: bool

    a_payload = A1(x=123).model_dump()
    b_payload = B1(name="neo").model_dump()

    assert a_payload == {"A1": {"x": 123}}
    assert b_payload == {"B1": {"name": "neo"}}

    ta_a = TypeAdapter[A1 | A2](A1 | A2)
    ta_b = TypeAdapter[B1 | B2](B1 | B2)

    with pytest.raises(ValidationError):
        ta_a.validate_python(b_payload)

    with pytest.raises(ValidationError):
        ta_b.validate_python(a_payload)


class Inner(TaggedModel):
    x: int


class Outer(TaggedModel):
    inner: Inner


class Wrapper(TaggedModel):
    outer: Outer
    label: str


class Container(TaggedModel):
    items: list[Inner]
    nested: Wrapper


def test_single_level_tagging():
    inner = Inner(x=10)
    dumped = inner.model_dump()
    assert dumped == {"Inner": {"x": 10}}

    restored = Inner.model_validate(dumped)
    assert isinstance(restored, Inner)
    assert restored.x == 10


def test_nested_externally_tagged_union_serializes_recursively():
    outer = Outer(inner=Inner(x=42))
    dumped = outer.model_dump()

    assert dumped == {"Outer": {"inner": {"Inner": {"x": 42}}}}

    restored = Outer.model_validate(dumped)
    assert isinstance(restored.inner, Inner)
    assert restored.inner.x == 42


def test_two_level_nested_tagging():
    outer = Outer(inner=Inner(x=123))
    dumped = outer.model_dump()
    assert dumped == {"Outer": {"inner": {"Inner": {"x": 123}}}}

    restored = Outer.model_validate(dumped)
    assert isinstance(restored.inner, Inner)
    assert restored.inner.x == 123


def test_three_level_nested_tagging():
    wrapper = Wrapper(label="deep", outer=Outer(inner=Inner(x=7)))
    dumped = wrapper.model_dump()
    # 3-level structure, each with exactly one tag
    assert dumped == {
        "Wrapper": {
            "label": "deep",
            "outer": {"Outer": {"inner": {"Inner": {"x": 7}}}},
        }
    }

    restored = Wrapper.model_validate(dumped)
    assert isinstance(restored.outer.inner, Inner)
    assert restored.outer.inner.x == 7
    assert restored.label == "deep"


def test_lists_and_mixed_nested_structures():
    container = Container(
        items=[Inner(x=1), Inner(x=2)],
        nested=Wrapper(label="mix", outer=Outer(inner=Inner(x=9))),
    )
    dumped = container.model_dump()

    assert dumped == {
        "Container": {
            "items": [
                {"Inner": {"x": 1}},
                {"Inner": {"x": 2}},
            ],
            "nested": {
                "Wrapper": {
                    "label": "mix",
                    "outer": {"Outer": {"inner": {"Inner": {"x": 9}}}},
                }
            },
        }
    }

    restored = Container.model_validate(dumped)
    assert isinstance(restored.nested.outer.inner, Inner)
    assert [i.x for i in restored.items] == [1, 2]


def test_no_double_tagging_on_repeated_calls():
    """Ensure multiple model_dump calls don't stack tags."""
    inner = Inner(x=11)
    dumped1 = inner.model_dump()
    dumped2 = inner.model_dump()
    assert dumped1 == dumped2 == {"Inner": {"x": 11}}

    outer = Outer(inner=inner)
    d1 = outer.model_dump()
    d2 = outer.model_dump()
    assert d1 == d2 == {"Outer": {"inner": {"Inner": {"x": 11}}}}


class L3A(TaggedModel):
    x: int


class L3B(TaggedModel):
    x: int


class L3C(TaggedModel):
    x: int


L3 = L3A | L3B | L3C


class L2A(TaggedModel):
    child: L3


class L2B(TaggedModel):
    child: L3


class L2C(TaggedModel):
    child: L3


L2 = L2A | L2B | L2C


class L1A(TaggedModel):
    child: L2


class L1B(TaggedModel):
    child: L2


class L1C(TaggedModel):
    child: L2


L1 = L1A | L1B | L1C


@pytest.mark.anyio
async def test_tagged_union_is_fast():
    # payload along the "C" path (worst case for DFS if branches are tried A->B->C)
    payload = {"L1C": {"child": {"L2C": {"child": {"L3C": {"x": 123}}}}}}

    with anyio.fail_after(0.1):
        out = TypeAdapter(L1).validate_python(payload)  # type: ignore

    # Sanity check the result
    assert out.__class__.__name__ == "L1C"  # type: ignore
    assert out.child.__class__.__name__ == "L2C"  # type: ignore
    assert out.child.child.__class__.__name__ == "L3C"  # type: ignore
    assert out.child.child.x == 123  # type: ignore
