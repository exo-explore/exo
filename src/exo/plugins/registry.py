"""Plugin registry for discovering and managing plugins."""

from collections.abc import Callable, Sequence
from typing import Any

from loguru import logger

from exo.plugins.base import EXOPlugin


class PluginRegistry:
    """Central registry for all plugins."""

    _instance: "PluginRegistry | None" = None

    def __init__(self) -> None:
        self._plugins: dict[str, EXOPlugin] = {}
        self._command_handlers: dict[type, EXOPlugin] = {}
        self._instance_handlers: dict[type, EXOPlugin] = {}

    @classmethod
    def get(cls) -> "PluginRegistry":
        """Get the singleton registry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None

    def register(self, plugin: EXOPlugin) -> None:
        """Register a plugin and its types."""
        if plugin.name in self._plugins:
            raise ValueError(f"Plugin '{plugin.name}' already registered")

        logger.info(f"Registering plugin: {plugin.name} v{plugin.version}")

        self._plugins[plugin.name] = plugin

        # Register command handlers
        for cmd_type in plugin.get_command_types():
            self._command_handlers[cmd_type] = plugin
            logger.debug(f"  Registered command: {cmd_type.__name__}")

        # Register instance handler
        instance_type = plugin.get_instance_type()
        self._instance_handlers[instance_type] = plugin
        logger.debug(f"  Registered instance: {instance_type.__name__}")

    def get_plugin(self, name: str) -> EXOPlugin | None:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def get_plugin_for_command(self, command: object) -> EXOPlugin | None:
        """Get the plugin that handles a command."""
        for plugin in self._plugins.values():
            if plugin.handles_command(command):
                return plugin
        return None

    def get_plugin_for_instance(self, instance: object) -> EXOPlugin | None:
        """Get the plugin that manages an instance."""
        for plugin in self._plugins.values():
            if plugin.handles_instance(instance):
                return plugin
        return None

    def all_plugins(self) -> Sequence[EXOPlugin]:
        """Get all registered plugins."""
        return list(self._plugins.values())

    def get_all_api_routes(
        self,
    ) -> Sequence[tuple[str, str, Callable[..., Any], EXOPlugin]]:
        """Get all API routes from all plugins."""
        routes: list[tuple[str, str, Callable[..., Any], EXOPlugin]] = []
        for plugin in self._plugins.values():
            for method, path, handler in plugin.get_api_routes():
                routes.append((method, path, handler, plugin))
        return routes


def discover_plugins() -> None:
    """Auto-discover and register plugins from the implementations directory.

    Plugins should have a register() function that returns an EXOPlugin instance.
    """
    import importlib
    import pkgutil

    registry = PluginRegistry.get()

    try:
        import exo.plugins.implementations as impl_package

        for _, module_name, _ in pkgutil.iter_modules(impl_package.__path__):
            try:
                module = importlib.import_module(
                    f"exo.plugins.implementations.{module_name}"
                )
                if hasattr(module, "register"):
                    plugin = module.register()  # pyright: ignore[reportAny]
                    if plugin is not None:
                        registry.register(plugin)  # pyright: ignore[reportAny]
            except Exception as e:
                logger.warning(f"Failed to load plugin {module_name}: {e}")
    except ImportError:
        logger.debug("No plugin implementations package found")
