import uuid
from typing import Callable, TypeVar

from pydantic import UUID4, BaseModel

NT = TypeVar("NT", bound=str)
type NewTypeGenerator[NT] = Callable[[str], NT]


class _UuidValidator(BaseModel):
    id: UUID4


def _generate_uuid() -> str:
    """Return a freshly generated RFC-4122 UUID version 4 in canonical string form."""
    return str(uuid.uuid4())


def generate_uuid(type_wrapper: NewTypeGenerator[NT]) -> NT:
    return type_wrapper(_generate_uuid())


def validate_uuid(data: str, type_wrapper: NewTypeGenerator[NT]) -> NT:
    validated_model = _UuidValidator.model_validate({"id": data})
    return type_wrapper(str(validated_model.id))
