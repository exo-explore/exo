"""FLASH Plugin - MPI-based simulation support for Exo."""

from typing import TYPE_CHECKING, Any

# Import types directly (these don't cause circular imports)
from exo.plugins.implementations.flash.types import (
    FLASHInstance,
    LaunchFLASH,
    StopFLASH,
)

if TYPE_CHECKING:
    from exo.plugins.implementations.flash.plugin import FLASHPlugin

__all__ = ["FLASHPlugin", "FLASHInstance", "LaunchFLASH", "StopFLASH", "register"]


def register() -> "FLASHPlugin":
    """Entry point for plugin discovery."""
    # Lazy import to avoid circular imports during module loading
    from exo.plugins.implementations.flash.plugin import FLASHPlugin

    return FLASHPlugin()


# For backwards compatibility, allow importing FLASHPlugin from this module
def __getattr__(name: str) -> Any:  # pyright: ignore[reportAny]
    if name == "FLASHPlugin":
        from exo.plugins.implementations.flash.plugin import FLASHPlugin

        return FLASHPlugin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
