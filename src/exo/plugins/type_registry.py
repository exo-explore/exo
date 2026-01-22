"""Dynamic type registry for plugin types.

This module provides a registry system that allows plugins to register their
command and instance types dynamically, eliminating the need for static union
types and avoiding circular imports.
"""

from typing import TypeVar

from loguru import logger

from exo.utils.pydantic_ext import CamelCaseModel

# TypeVar for preserving exact types through the register decorator
_TCls = TypeVar("_TCls", bound=type[CamelCaseModel])


class TypeRegistry[T: CamelCaseModel]:
    """Registry for dynamically registered Pydantic types.

    Enables plugins to register their types at import time. Deserialization
    uses the class name from the tagged JSON format to look up the correct type.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._types: dict[str, type[T]] = {}

    def register(self, cls: _TCls) -> _TCls:
        """Decorator to register a type with this registry.

        Preserves the exact type through the decorator for proper type checking.
        """
        self._types[cls.__name__] = cls  # type: ignore[assignment]
        logger.debug(f"{self._name}: registered {cls.__name__}")
        return cls

    def get(self, name: str) -> type[T] | None:
        """Look up a type by class name."""
        return self._types.get(name)

    def all_types(self) -> dict[str, type[T]]:
        """Return all registered types."""
        return dict(self._types)

    def deserialize(self, data: dict[str, dict[str, object]] | CamelCaseModel) -> T:
        """Deserialize dict to the appropriate registered type.

        Supports two formats:
        1. Tagged format: {"ClassName": {...fields...}} - used for network serialization
        2. Flat format: {...fields...} - used for API requests, tries each type
        """
        # If already deserialized (e.g., from Pydantic), return as-is
        if isinstance(data, CamelCaseModel):
            return data  # type: ignore[return-value]

        # Check for tagged format: single key that matches a registered type
        if len(data) == 1:
            class_name: str = next(iter(data.keys()))
            cls = self._types.get(class_name)
            if cls is not None:
                return cls.model_validate(data[class_name], strict=False)

        # Flat format: try each registered type, use first that validates
        errors: list[str] = []
        for type_name, cls in self._types.items():
            try:
                return cls.model_validate(data, strict=False)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type_name}: {e}")

        # None matched - provide helpful error
        available = ", ".join(self._types.keys())
        raise ValueError(
            f"{self._name}: could not deserialize data. "
            f"Available types: {available}. Errors: {'; '.join(errors[:3])}"
        )


# Global registries for commands, instances, events, and tasks
command_registry: TypeRegistry[CamelCaseModel] = TypeRegistry("CommandRegistry")
instance_registry: TypeRegistry[CamelCaseModel] = TypeRegistry("InstanceRegistry")
event_registry: TypeRegistry[CamelCaseModel] = TypeRegistry("EventRegistry")
task_registry: TypeRegistry[CamelCaseModel] = TypeRegistry("TaskRegistry")
