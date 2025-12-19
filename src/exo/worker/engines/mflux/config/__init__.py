from exo.worker.engines.mflux.config.flux import FLUX_DEV_CONFIG, FLUX_SCHNELL_CONFIG
from exo.worker.engines.mflux.config.model_config import (
    BlockType,
    ImageModelConfig,
    TransformerBlockConfig,
)

__all__ = [
    "BlockType",
    "ImageModelConfig",
    "TransformerBlockConfig",
    "FLUX_SCHNELL_CONFIG",
    "FLUX_DEV_CONFIG",
    "get_config_for_model",
]


# Config registry: maps model ID patterns to configs
_CONFIG_REGISTRY: dict[str, ImageModelConfig] = {
    "flux.1-schnell": FLUX_SCHNELL_CONFIG,
    "flux1-schnell": FLUX_SCHNELL_CONFIG,
    "schnell": FLUX_SCHNELL_CONFIG,
    "flux.1-dev": FLUX_DEV_CONFIG,
    "flux1-dev": FLUX_DEV_CONFIG,
    "dev": FLUX_DEV_CONFIG,
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
