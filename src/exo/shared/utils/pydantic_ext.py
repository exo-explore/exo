from pydantic import BaseModel
from pydantic.alias_generators import to_camel


class CamelCaseModel(BaseModel):
    """
    A model whose fields are aliased to camel-case from snake-case.
    """

    class Config:
        alias_generator = to_camel
        allow_population_by_field_name = True


class Tagged[Tag: str, Content](
    CamelCaseModel
):  # TODO: figure out how to make pydantic work with LiteralString
    """
    Utility for helping with serializing unions as adjacently tagged with Pydantic.

    By default, Pydantic uses internally tagged union ser/de BUT to play nicely with
    other cross-language ser/de tools, you need adjacently tagged unions, and Pydantic
    doesn't support those out of the box.

    SEE: https://serde.rs/enum-representations.html#adjacently-tagged

    Example usage:
    ```python
    TaggedUnion = Annotated[Union[
        Tagged[Literal["Foo"], Foo],
        Tagged[Literal["Bar"], Bar]
    ], Field(discriminator="t")]

    Parser: TypeAdapter[TaggedUnion] = TypeAdapter(TaggedUnion)

    def validate_python(v: any) -> Foo | Bar:
        v = Parser.validate_python(v)
        match v.t:
            case "Foo": return v.c
            case "Bar": return v.c
    ```
    """

    t: Tag
    """
    The tag corresponding to the type of the object in the union.
    """

    c: Content
    """
    The actual content of the object of that type.
    """
