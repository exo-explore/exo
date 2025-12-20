"""
Adapter registry for model-specific operations.

This module provides a registry pattern for managing model adapters,
allowing new model families to be added without modifying core code.
"""

from typing import Callable

from exo.worker.engines.mflux.config.model_config import ImageModelConfig
from exo.worker.engines.mflux.pipefusion.adapter import ModelAdapter
from exo.worker.engines.mflux.pipefusion.flux_adapter import FluxModelAdapter

# Type alias for adapter factory functions
AdapterFactory = Callable[[ImageModelConfig], ModelAdapter]

# Registry maps model_family string to adapter factory
_ADAPTER_REGISTRY: dict[str, AdapterFactory] = {
    "flux": FluxModelAdapter,
}


def get_adapter_for_model(config: ImageModelConfig) -> ModelAdapter:
    """Get the appropriate adapter for a model configuration.

    Args:
        config: The model configuration

    Returns:
        A ModelAdapter instance for the model family

    Raises:
        ValueError: If no adapter is registered for the model family
    """
    factory = _ADAPTER_REGISTRY.get(config.model_family)
    if factory is None:
        raise ValueError(f"No adapter found for model family: {config.model_family}")
    return factory(config)


def register_adapter(model_family: str, factory: AdapterFactory) -> None:
    """Register a new adapter factory for a model family.

    Args:
        model_family: The model family identifier (e.g., "flux", "fibo", "qwen")
        factory: A callable that takes an ImageModelConfig and returns a ModelAdapter
    """
    _ADAPTER_REGISTRY[model_family] = factory
