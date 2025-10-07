from typing import Union

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from exo.utils.pydantic_tagged import Tagged, tagged_union  # ← CHANGE ME


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
    class Foo1(BaseModel):
        x: int

    class Foo2(BaseModel):
        x: int

    foos = Union[Foo1, Foo2]

    @tagged_union({"Foo1": Foo1, "Foo2": Foo2})
    class TaggedFoos(Tagged[foos]):
        pass

    # ---- serialize (via custom model_serializer) ----
    t1 = TaggedFoos.from_(Foo1(x=1))
    assert t1.model_dump() == {"t": "Foo1", "c": {"x": 1}}

    t2 = TaggedFoos.from_(Foo2(x=2))
    assert t2.model_dump() == {"t": "Foo2", "c": {"x": 2}}

    # ---- deserialize (TypeAdapter -> model_validator(before)) ----
    ta = TypeAdapter(TaggedFoos)

    out1 = ta.validate_python({"t": "Foo1", "c": {"x": 10}})
    assert isinstance(out1.c, Foo1) and out1.c.x == 10

    out2 = ta.validate_python({"t": "Foo2", "c": {"x": 20}})
    assert isinstance(out2.c, Foo2) and out2.c.x == 20


def test_tagged_union_rejects_unknown_tag():
    class Foo1(BaseModel):
        x: int

    class Foo2(BaseModel):
        x: int

    foos = Union[Foo1, Foo2]

    @tagged_union({"Foo1": Foo1, "Foo2": Foo2})
    class TaggedFoos(Tagged[foos]):
        pass

    ta = TypeAdapter(TaggedFoos)
    with pytest.raises(ValidationError):
        ta.validate_python({"t": "NotARealTag", "c": {"x": 0}})


def test_multiple_tagged_classes_do_not_override_each_others_mappings():
    """
    Creating a *new* Tagged[T] class must not mutate the previously defined one.
    This checks both the tag mapping and the per-class adapter dicts.
    """

    class Foo1(BaseModel):
        x: int

    class Foo2(BaseModel):
        x: int

    foos = Union[Foo1, Foo2]

    @tagged_union({"One": Foo1, "Two": Foo2})
    class TaggedEN(Tagged[foos]):
        pass

    # Sanity: initial mapping/behavior
    obj_en_1 = TaggedEN.from_(Foo1(x=5))
    assert obj_en_1.t == "One"
    obj_en_2 = TaggedEN.from_(Foo2(x=6))
    assert obj_en_2.t == "Two"

    # Define a second, different mapping
    @tagged_union({"Uno": Foo1, "Dos": Foo2})
    class TaggedES(Tagged[foos]):
        pass

    # The two classes should have *independent* mappings
    # (not the same object, and not equal content)
    assert TaggedEN._type_bidict is not TaggedES._type_bidict  # pyright: ignore
    assert TaggedEN._type_bidict != TaggedES._type_bidict  # pyright: ignore

    # Their adapters dicts should also be distinct objects
    assert TaggedEN._adapter_dict is not TaggedES._adapter_dict  # pyright: ignore
    # And both should cover the same set of member types
    assert set(TaggedEN._adapter_dict.keys()) == {Foo1, Foo2}  # pyright: ignore
    assert set(TaggedES._adapter_dict.keys()) == {Foo1, Foo2}  # pyright: ignore

    # Re-check that EN behavior has NOT changed after ES was created
    obj_en_1_again = TaggedEN.from_(Foo1(x=7))
    obj_en_2_again = TaggedEN.from_(Foo2(x=8))
    assert obj_en_1_again.t == "One"
    assert obj_en_2_again.t == "Two"

    # ES behavior is per its *own* mapping
    obj_es_1 = TaggedES.from_(Foo1(x=9))
    obj_es_2 = TaggedES.from_(Foo2(x=10))
    assert obj_es_1.t == "Uno"
    assert obj_es_2.t == "Dos"

    # And deserialization respects each class's mapping independently
    ta_en = TypeAdapter(TaggedEN)
    ta_es = TypeAdapter(TaggedES)

    out_en = ta_en.validate_python({"t": "Two", "c": {"x": 123}})
    assert isinstance(out_en.c, Foo2) and out_en.c.x == 123

    out_es = ta_es.validate_python({"t": "Dos", "c": {"x": 456}})
    assert isinstance(out_es.c, Foo2) and out_es.c.x == 456


def test_two_tagged_classes_with_different_shapes_are_independent_and_not_cross_deserializable():
    class A1(BaseModel):
        x: int

    class A2(BaseModel):
        name: str

    union_a = Union[A1, A2]

    @tagged_union({"One": A1, "Two": A2})
    class TaggedA(Tagged[union_a]):
        pass

    class B1(BaseModel):
        name: str

    class B2(BaseModel):
        active: bool

    union_b = Union[B1, B2]

    # Note: using the SAME tag strings intentionally to ensure mappings are per-class
    @tagged_union({"One": B1, "Two": B2})
    class TaggedB(Tagged[union_b]):
        pass

    # --- Per-class state must be independent ---
    assert TaggedA._type_bidict is not TaggedB._type_bidict  # pyright: ignore
    assert TaggedA._adapter_dict is not TaggedB._adapter_dict  # pyright: ignore
    assert set(TaggedA._adapter_dict.keys()) == {A1, A2}  # pyright: ignore
    assert set(TaggedB._adapter_dict.keys()) == {B1, B2}  # pyright: ignore

    # --- Round-trip for each class with overlapping tag strings ---
    a_payload = TaggedA.from_(A1(x=123)).model_dump()
    b_payload = TaggedB.from_(B1(name="neo")).model_dump()

    assert a_payload == {"t": "One", "c": {"x": 123}}
    assert b_payload == {"t": "One", "c": {"name": "neo"}}

    # --- Cross-deserialization must fail despite overlapping "t" values ---
    ta_a = TypeAdapter(TaggedA)
    ta_b = TypeAdapter(TaggedB)

    with pytest.raises(ValidationError):
        ta_a.validate_python(b_payload)  # TaggedA expects {"x": ...} for tag "One"

    with pytest.raises(ValidationError):
        ta_b.validate_python(a_payload)  # TaggedB expects {"name": ...} for tag "One"
