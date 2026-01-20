"""FLASH Plugin - MPI-based simulation support for Exo."""

from exo.plugins.implementations.flash.plugin import FLASHPlugin


def register() -> FLASHPlugin:
    """Entry point for plugin discovery."""
    return FLASHPlugin()
