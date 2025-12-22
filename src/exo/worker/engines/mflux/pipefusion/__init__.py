from pathlib import Path
from typing import Callable

from exo.worker.engines.mflux.config.model_config import ImageModelConfig
from exo.worker.engines.mflux.pipefusion.adapter import ModelAdapter
from exo.worker.engines.mflux.pipefusion.diffusion_runner import DiffusionRunner
from exo.worker.engines.mflux.pipefusion.flux_adapter import FluxModelAdapter

__all__ = [
    "create_adapter_for_model",
    "DiffusionRunner",
    "ModelAdapter",
    "FluxModelAdapter",
]

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
    factory = _ADAPTER_REGISTRY.get(config.model_family)
    if factory is None:
        raise ValueError(f"No adapter found for model family: {config.model_family}")
    return factory(config, model_id, local_path, quantize)
