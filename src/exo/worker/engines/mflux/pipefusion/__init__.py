"""
Adapter registry for model-specific operations.

This module provides a registry pattern for managing model adapters,
allowing new model families to be added without modifying core code.

Each adapter owns its model instance and is responsible for all model-specific
distributed inference operations.
"""

from pathlib import Path
from typing import Callable

from exo.worker.engines.mflux.config.model_config import ImageModelConfig
from exo.worker.engines.mflux.pipefusion.adapter import ModelAdapter
from exo.worker.engines.mflux.pipefusion.flux_adapter import FluxModelAdapter

# Type alias for adapter factory functions
# Factory takes (config, model_id, local_path, quantize) and returns a ModelAdapter
AdapterFactory = Callable[[ImageModelConfig, str, Path, int | None], ModelAdapter]

# Registry maps model_family string to adapter factory
_ADAPTER_REGISTRY: dict[str, AdapterFactory] = {
    "flux": FluxModelAdapter,
}


def create_adapter_for_model(
    config: ImageModelConfig,
    model_id: str,
    local_path: Path,
    quantize: int | None = None,
) -> ModelAdapter:
    """Create an adapter with its model for a given configuration.

    The adapter creates and owns the model instance.

    Args:
        config: The model configuration
        model_id: The model identifier (e.g., "black-forest-labs/FLUX.1-schnell")
        local_path: Path to the local model weights
        quantize: Optional quantization bit width

    Returns:
        A ModelAdapter instance that owns the model

    Raises:
        ValueError: If no adapter is registered for the model family
    """
    factory = _ADAPTER_REGISTRY.get(config.model_family)
    if factory is None:
        raise ValueError(f"No adapter found for model family: {config.model_family}")
    return factory(config, model_id, local_path, quantize)


def register_adapter(model_family: str, factory: AdapterFactory) -> None:
    """Register a new adapter factory for a model family.

    Args:
        model_family: The model family identifier (e.g., "flux", "fibo", "qwen")
        factory: A callable that takes (config, model_id, local_path, quantize)
                 and returns a ModelAdapter that owns the model
    """
    _ADAPTER_REGISTRY[model_family] = factory
