"""Exo Plugin System.

This module provides the plugin architecture for extending exo with custom
workload types (simulations, ML frameworks, etc.) without modifying core code.
"""

from exo.plugins.base import EXOPlugin, PluginCommand, PluginInstance
from exo.plugins.registry import PluginRegistry, discover_plugins
from exo.plugins.type_registry import (
    command_registry,
    event_registry,
    instance_registry,
)

__all__ = [
    "EXOPlugin",
    "PluginCommand",
    "PluginInstance",
    "PluginRegistry",
    "discover_plugins",
    "command_registry",
    "event_registry",
    "instance_registry",
]
