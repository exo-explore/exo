"""
Adapter and model factory registries.

This module provides registry patterns for managing model adapters and
model factories, allowing new model families to be added without modifying
core code.
"""

from pathlib import Path
from typing import Any, Callable

from mflux.config.model_config import ModelConfig
from mflux.models.flux.variants.txt2img.flux import Flux1

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


# =============================================================================
# Model Factory Registry
# =============================================================================

# Type alias for model factory functions
# Takes (model_id, local_path, quantize) and returns a model instance
ModelFactory = Callable[[str, Path, int | None], Any]


def _create_flux_model(model_id: str, local_path: Path, quantize: int | None) -> Flux1:
    """Create a Flux1 model instance."""
    return Flux1(
        model_config=ModelConfig.from_name(model_name=model_id, base_model=None),
        local_path=str(local_path),
        quantize=quantize,
    )


# Registry maps model_family string to model factory
_MODEL_REGISTRY: dict[str, ModelFactory] = {
    "flux": _create_flux_model,
}


def create_model(
    config: ImageModelConfig,
    model_id: str,
    local_path: Path,
    quantize: int | None = None,
) -> Any:
    """Create a model instance for a model configuration.

    Args:
        config: The model configuration
        model_id: The model identifier (e.g., "black-forest-labs/FLUX.1-schnell")
        local_path: Path to the local model weights
        quantize: Optional quantization bit width

    Returns:
        A model instance for the model family

    Raises:
        ValueError: If no factory is registered for the model family
    """
    factory = _MODEL_REGISTRY.get(config.model_family)
    if factory is None:
        raise ValueError(f"No model factory found for model family: {config.model_family}")
    return factory(model_id, local_path, quantize)


def register_model_factory(model_family: str, factory: ModelFactory) -> None:
    """Register a new model factory for a model family.

    Args:
        model_family: The model family identifier (e.g., "flux", "fibo", "qwen")
        factory: A callable that takes (model_id, local_path, quantize) and returns a model
    """
    _MODEL_REGISTRY[model_family] = factory
