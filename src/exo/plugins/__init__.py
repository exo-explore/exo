"""Exo Plugin System.

This module provides the plugin architecture for extending exo with custom
workload types (simulations, ML frameworks, etc.) without modifying core code.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from exo.plugins.base import EXOPlugin, PluginCommand, PluginInstance
    from exo.plugins.registry import PluginRegistry, discover_plugins

__all__ = [
    "EXOPlugin",
    "PluginCommand",
    "PluginInstance",
    "PluginRegistry",
    "discover_plugins",
]


def __getattr__(name: str) -> Any:  # pyright: ignore[reportAny]
    """Lazy import to avoid circular dependencies."""
    if name in ("EXOPlugin", "PluginCommand", "PluginInstance"):
        from exo.plugins.base import EXOPlugin, PluginCommand, PluginInstance

        return {
            "EXOPlugin": EXOPlugin,
            "PluginCommand": PluginCommand,
            "PluginInstance": PluginInstance,
        }[name]
    if name in ("PluginRegistry", "discover_plugins"):
        from exo.plugins.registry import PluginRegistry, discover_plugins

        return {"PluginRegistry": PluginRegistry, "discover_plugins": discover_plugins}[
            name
        ]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
