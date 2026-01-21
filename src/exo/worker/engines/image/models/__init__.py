from pathlib import Path
from typing import Any, Callable

from exo.worker.engines.image.config import ImageModelConfig
from exo.worker.engines.image.models.base import ModelAdapter
from exo.worker.engines.image.models.flux import (
    FLUX_DEV_CONFIG,
    FLUX_SCHNELL_CONFIG,
    FluxModelAdapter,
)
from exo.worker.engines.image.models.qwen import (
    QWEN_IMAGE_CONFIG,
    QWEN_IMAGE_EDIT_CONFIG,
    QwenEditModelAdapter,
    QwenModelAdapter,
)

__all__: list[str] = []

# Type alias for adapter factory functions
# Factory takes (config, model_id, local_path, quantize) and returns a ModelAdapter
AdapterFactory = Callable[
    [ImageModelConfig, str, Path, int | None], ModelAdapter[Any, Any]
]

# Registry maps model_family string to adapter factory
_ADAPTER_REGISTRY: dict[str, AdapterFactory] = {
    "flux": FluxModelAdapter,
    "qwen-edit": QwenEditModelAdapter,
    "qwen": QwenModelAdapter,
}

# Config registry: maps model ID patterns to configs
_CONFIG_REGISTRY: dict[str, ImageModelConfig] = {
    "flux.1-schnell": FLUX_SCHNELL_CONFIG,
    "flux.1-dev": FLUX_DEV_CONFIG,
    "qwen-image-edit": QWEN_IMAGE_EDIT_CONFIG,  # Must come before "qwen-image" for pattern matching
    "qwen-image": QWEN_IMAGE_CONFIG,
}


def get_config_for_model(model_id: str) -> ImageModelConfig:
    """Get configuration for a model ID.

    Args:
        model_id: The model identifier (e.g., "black-forest-labs/FLUX.1-schnell")

    Returns:
        The model configuration

    Raises:
        ValueError: If no configuration found for model ID
    """
    model_id_lower = model_id.lower()

    for pattern, config in _CONFIG_REGISTRY.items():
        if pattern in model_id_lower:
            return config

    raise ValueError(f"No configuration found for model: {model_id}")


def create_adapter_for_model(
    config: ImageModelConfig,
    model_id: str,
    local_path: Path,
    quantize: int | None = None,
) -> ModelAdapter[Any, Any]:
    """Create a model adapter for the given configuration.

    Args:
        config: The model configuration
        model_id: The model identifier
        local_path: Path to the model weights
        quantize: Optional quantization bits

    Returns:
        A ModelAdapter instance

    Raises:
        ValueError: If no adapter found for model family
    """
    factory = _ADAPTER_REGISTRY.get(config.model_family)
    if factory is None:
        raise ValueError(f"No adapter found for model family: {config.model_family}")
    return factory(config, model_id, local_path, quantize)
