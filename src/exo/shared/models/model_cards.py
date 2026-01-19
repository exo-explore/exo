"""Model cards for exo.

This module provides the MODEL_CARDS dictionary for backward compatibility.
Model configurations are now loaded from JSON files via ModelRegistry.
"""

from exo.shared.models.model_registry import get_registry
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelConfig, ModelId, ModelMetadata
from exo.utils.pydantic_ext import CamelCaseModel


class ModelCard(CamelCaseModel):
    """Model card with metadata for display and inference."""

    model_id: ModelId
    name: str
    description: str
    tags: list[str]
    metadata: ModelMetadata
    # Grouped model fields (new)
    base_model_id: str = ""
    base_model_name: str = ""
    quantization: str = ""
    architecture: str = ""
    # UI display fields
    tagline: str = ""
    capabilities: list[str] = []
    family: str = ""


def _config_to_card(config: ModelConfig) -> ModelCard:
    """Convert a ModelConfig to a ModelCard for backward compatibility."""
    return ModelCard(
        model_id=ModelId(config.model_id),
        name=config.name,
        description=config.description,
        tags=config.tags,
        metadata=ModelMetadata(
            model_id=ModelId(config.model_id),
            pretty_name=config.name,
            storage_size=Memory.from_bytes(config.storage_size_bytes),
            n_layers=config.n_layers,
            hidden_size=config.hidden_size,
            supports_tensor=config.supports_tensor,
        ),
        base_model_id=config.base_model_id,
        base_model_name=config.base_model_name,
        quantization=config.quantization,
        architecture=config.architecture,
        tagline=config.tagline,
        capabilities=config.capabilities,
        family=config.family,
    )


class _ModelCardsProxy:
    """Proxy class that provides dict-like access to model cards via the registry.

    This ensures MODEL_CARDS always reflects the current state of the registry,
    including any user-added models. Keys are model_id (HuggingFace repo path).
    """

    def __getitem__(self, key: str) -> ModelCard:
        config = get_registry().get(key)
        if config is None:
            raise KeyError(key)
        return _config_to_card(config)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return get_registry().get(key) is not None

    def __iter__(self):
        return iter(get_registry().list_all())

    def __len__(self) -> int:
        return len(get_registry().list_all())

    def get(self, key: str, default: ModelCard | None = None) -> ModelCard | None:
        config = get_registry().get(key)
        if config is None:
            return default
        return _config_to_card(config)

    def keys(self):
        return get_registry().list_all().keys()

    def values(self):
        registry = get_registry()
        for config in registry.list_all().values():
            yield _config_to_card(config)

    def items(self):
        registry = get_registry()
        for model_id, config in registry.list_all().items():
            yield model_id, _config_to_card(config)


# Backward-compatible MODEL_CARDS dict
# This is a proxy that reads from the registry
MODEL_CARDS: dict[str, ModelCard] = _ModelCardsProxy()  # type: ignore[assignment]
