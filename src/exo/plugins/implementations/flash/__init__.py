"""FLASH Plugin - MPI-based simulation support for Exo."""

from exo.plugins.implementations.flash.plugin import FLASHPlugin
from exo.plugins.implementations.flash.types import (
    FLASHInstance,
    LaunchFLASH,
    StopFLASH,
)

__all__ = ["FLASHPlugin", "FLASHInstance", "LaunchFLASH", "StopFLASH", "register"]


def register() -> FLASHPlugin:
    """Entry point for plugin discovery."""
    return FLASHPlugin()
