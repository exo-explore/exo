"""Exo Plugin System.

This module provides the plugin architecture for extending exo with custom
workload types (simulations, ML frameworks, etc.) without modifying core code.
"""

from exo.plugins.base import ExoPlugin, PluginCommand, PluginInstance
from exo.plugins.registry import PluginRegistry, discover_plugins

__all__ = [
    "ExoPlugin",
    "PluginCommand",
    "PluginInstance",
    "PluginRegistry",
    "discover_plugins",
]
