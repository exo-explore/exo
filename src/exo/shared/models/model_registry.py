"""Model registry for loading and managing model configurations.

This module provides a registry that loads model configurations from:
1. New registry structure: base_models.json + variants.json (grouped models)
2. Legacy JSON files in the cards/ directory (shipped with exo)
3. User-added JSON files in ~/.exo/models/ (created via dashboard)

The registry automatically combines base model metadata with variant data
to produce complete ModelConfig objects with grouping information.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, cast

from loguru import logger

from exo.shared.constants import EXO_USER_MODELS_DIR
from exo.shared.models.architecture_support import (
    supports_tensor_parallel,
    supports_vision,
)
from exo.shared.models.registry import (
    BASE_MODELS_BY_ID,
    VARIANTS,
)
from exo.shared.types.models import ModelConfig

# Directory containing built-in model config JSON files (legacy format)
BUILTIN_CARDS_DIR: Final[Path] = Path(__file__).parent / "cards"


def derive_capabilities(
    architecture: str,
    manual_capabilities: list[str] | None = None,
) -> list[str]:
    """Derive model capabilities from architecture and manual overrides.

    Priority:
    1. Manual capabilities from base_models.json (if provided)
    2. Architecture-based detection (for vision)
    3. Default: ["text"]

    Args:
        architecture: The HuggingFace model_type value.
        manual_capabilities: Explicit capabilities from base_models.json.

    Returns:
        List of capability strings (e.g., ["text", "vision", "code"]).
    """
    # If manual capabilities are provided, use them as primary source
    if manual_capabilities:
        return manual_capabilities

    capabilities = ["text"]  # All models support text by default

    # Architecture-based detection for vision
    if supports_vision(architecture):
        capabilities.append("vision")

    return capabilities


def _model_id_to_filename(model_id: str) -> str:
    """Convert model_id to a valid filename by replacing / with --.

    Examples:
        "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit" -> "mlx-community--Meta-Llama-3.1-8B-Instruct-4bit"
    """
    return model_id.replace("/", "--")


def _quantization_display_name(quantization: str) -> str:
    """Format quantization for display in model name."""
    quant_lower = quantization.lower()
    if quant_lower in ("4bit", "4-bit"):
        return "4-bit"
    if quant_lower in ("8bit", "8-bit"):
        return "8-bit"
    if quant_lower in ("bf16", "bfloat16"):
        return "BF16"
    if quant_lower in ("fp16", "float16"):
        return "FP16"
    if quant_lower in ("3bit", "3-bit"):
        return "3-bit"
    if quant_lower in ("6bit", "6-bit"):
        return "6-bit"
    # Return as-is for other quantizations
    return quantization.upper()


def _load_config_from_file(path: Path) -> ModelConfig | None:
    """Load a single model config from a JSON file."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = cast(dict[str, Any], json.load(f))
        return ModelConfig.model_validate(data)
    except Exception as e:
        logger.warning(f"Failed to load model config from {path}: {e}")
        return None


def _variant_to_config(variant: dict[str, object]) -> ModelConfig | None:
    """Convert a variant + base model data to a ModelConfig."""
    base_model_id = str(variant.get("base_model", ""))
    base_model = BASE_MODELS_BY_ID.get(base_model_id)

    if base_model is None:
        logger.warning(f"Base model {base_model_id} not found for variant {variant}")
        return None

    model_id = str(variant["model_id"])
    quantization = str(variant.get("quantization", ""))
    storage_size_raw = variant.get("storage_size_bytes", 0)
    storage_size_bytes = int(str(storage_size_raw)) if storage_size_raw else 0

    base_name = str(base_model["name"])
    architecture = str(base_model.get("architecture", ""))
    n_layers_raw = base_model.get("n_layers", 0)
    n_layers = int(str(n_layers_raw)) if n_layers_raw else 1
    hidden_size_raw = base_model.get("hidden_size", 0)
    hidden_size = int(str(hidden_size_raw)) if hidden_size_raw else 1
    description = str(base_model.get("description", ""))

    # Extract new UI display fields from base model
    tagline = str(base_model.get("tagline", ""))
    family = str(base_model.get("family", ""))
    manual_capabilities_raw = base_model.get("capabilities")
    manual_capabilities = (
        cast(list[str], manual_capabilities_raw)
        if isinstance(manual_capabilities_raw, list)
        else None
    )
    capabilities = derive_capabilities(architecture, manual_capabilities)

    # Generate display name: "Base Name (Quantization)"
    quant_display = _quantization_display_name(quantization)
    display_name = f"{base_name} ({quant_display})"

    # Derive supports_tensor from architecture
    tensor_support = supports_tensor_parallel(architecture)

    return ModelConfig(
        model_id=model_id,
        name=display_name,
        description=description,
        tags=[],
        supports_tensor=tensor_support,
        storage_size_bytes=storage_size_bytes,
        n_layers=n_layers,
        hidden_size=hidden_size,
        is_user_added=False,
        architecture=architecture,
        base_model_id=base_model_id,
        base_model_name=base_name,
        quantization=quantization,
        tagline=tagline,
        capabilities=capabilities,
        family=family,
    )


class ModelRegistry:
    """Registry for model configurations.

    Loads configurations from:
    1. Registry structure (base_models.json + variants.json)
    2. Built-in cards/ directory (legacy format)
    3. User ~/.exo/models/
    """

    def __init__(self) -> None:
        self._configs: dict[str, ModelConfig] = {}
        self._reload()

    def _migrate_user_model_files(self) -> None:
        """Migrate user model files from old naming format to new format.

        Old format: {short_id}.json (e.g., "meta-llama-3.1-8b-instruct-4bit.json")
        New format: {model_id.replace("/", "--")}.json (e.g., "mlx-community--Meta-Llama-3.1-8B-Instruct-4bit.json")
        """
        if not EXO_USER_MODELS_DIR.exists():
            return

        for path in EXO_USER_MODELS_DIR.glob("*.json"):
            # New format files contain "--" (org--repo)
            if "--" in path.stem:
                continue

            # This is an old format file, load and migrate
            config = _load_config_from_file(path)
            if config is None:
                continue

            # Create new filename from model_id
            new_filename = _model_id_to_filename(config.model_id) + ".json"
            new_path = EXO_USER_MODELS_DIR / new_filename

            if new_path.exists():
                # New file already exists, just remove the old one
                logger.info(f"Removing old user model file (already migrated): {path}")
                path.unlink()
            else:
                # Rename to new format
                logger.info(f"Migrating user model file: {path} -> {new_path}")
                path.rename(new_path)

    def _reload(self) -> None:
        """Reload all model configurations from disk."""
        self._configs.clear()
        registry_count = 0
        legacy_count = 0

        # First, load from new registry structure (base_models + variants)
        for variant in VARIANTS:
            config = _variant_to_config(variant)
            if config is not None:
                # Use model_id as the key
                self._configs[config.model_id] = config
                registry_count += 1

        # Then load legacy built-in models (these can override registry if same model_id)
        if BUILTIN_CARDS_DIR.exists():
            for path in BUILTIN_CARDS_DIR.glob("*.json"):
                config = _load_config_from_file(path)
                if config is None:
                    continue
                # Skip if already loaded from registry
                if config.model_id in self._configs:
                    continue
                self._configs[config.model_id] = config
                legacy_count += 1

        # Migrate old user model files before loading
        self._migrate_user_model_files()

        # Load user models (these can override built-in if same model_id)
        user_count = 0
        if EXO_USER_MODELS_DIR.exists():
            for path in EXO_USER_MODELS_DIR.glob("*.json"):
                config = _load_config_from_file(path)
                if config is not None:
                    self._configs[config.model_id] = config
                    user_count += 1

        logger.info(
            f"Loaded {len(self._configs)} model configs "
            f"(registry: {registry_count}, legacy: {legacy_count}, user: {user_count})"
        )

    def get(self, model_id: str) -> ModelConfig | None:
        """Get a model config by model_id (HuggingFace repo path)."""
        return self._configs.get(model_id)

    def list_all(self) -> dict[str, ModelConfig]:
        """Return all model configurations."""
        return dict(self._configs)

    def list_builtin(self) -> dict[str, ModelConfig]:
        """Return only built-in model configurations."""
        return {k: v for k, v in self._configs.items() if not v.is_user_added}

    def list_user_added(self) -> dict[str, ModelConfig]:
        """Return only user-added model configurations."""
        return {k: v for k, v in self._configs.items() if v.is_user_added}

    def list_grouped(self) -> dict[str, list[ModelConfig]]:
        """Return models grouped by base_model_id.

        Returns:
            Dict mapping base_model_id to list of variant configs.
            Models without a base_model_id are grouped under their model_id.
        """
        grouped: dict[str, list[ModelConfig]] = {}
        for model_id, config in self._configs.items():
            # Use base_model_id if available, otherwise use model_id
            group_key = config.base_model_id if config.base_model_id else model_id
            if group_key not in grouped:
                grouped[group_key] = []
            grouped[group_key].append(config)

        # Sort variants within each group by storage size (smallest first)
        for variants in grouped.values():
            variants.sort(key=lambda c: c.storage_size_bytes)

        return grouped

    def get_base_model_variants(self, base_model_id: str) -> list[ModelConfig]:
        """Get all variants for a given base model.

        Args:
            base_model_id: The base model identifier (e.g., "llama-3.1-8b")

        Returns:
            List of ModelConfig objects for all variants, sorted by storage size.
        """
        variants = [
            config
            for config in self._configs.values()
            if config.base_model_id == base_model_id
        ]
        variants.sort(key=lambda c: c.storage_size_bytes)
        return variants

    def add_user_model(self, config: ModelConfig) -> str:
        """Add a user model configuration and persist to disk.

        Args:
            config: The model configuration to add

        Returns:
            The model_id of the added model.
        """
        model_id = config.model_id

        # Ensure is_user_added is True
        config = config.model_copy(update={"is_user_added": True})

        # Create user models directory if needed
        EXO_USER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Save to JSON file using model_id as filename
        filename = _model_id_to_filename(model_id) + ".json"
        path = EXO_USER_MODELS_DIR / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(config.model_dump(), f, indent=2)

        # Add to registry
        self._configs[model_id] = config
        logger.info(f"Added user model: {model_id}")

        return model_id

    def remove_user_model(self, model_id: str) -> bool:
        """Remove a user-added model.

        Args:
            model_id: The model_id of the model to remove.

        Returns:
            True if the model was removed, False if not found or not user-added.
        """
        config = self._configs.get(model_id)
        if config is None or not config.is_user_added:
            return False

        # Remove JSON file
        filename = _model_id_to_filename(model_id) + ".json"
        path = EXO_USER_MODELS_DIR / filename
        if path.exists():
            path.unlink()

        # Remove from registry
        del self._configs[model_id]
        logger.info(f"Removed user model: {model_id}")

        return True

    def update_user_model(self, model_id: str, **updates: object) -> ModelConfig | None:
        """Update a user-added model configuration.

        Args:
            model_id: The model_id of the model to update.
            **updates: Field updates to apply.

        Returns:
            The updated config, or None if not found or not user-added.
        """
        config = self._configs.get(model_id)
        if config is None or not config.is_user_added:
            return None

        # Apply updates
        updated_config = config.model_copy(update=updates)

        # Persist
        filename = _model_id_to_filename(model_id) + ".json"
        path = EXO_USER_MODELS_DIR / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(updated_config.model_dump(), f, indent=2)

        self._configs[model_id] = updated_config
        logger.info(f"Updated user model: {model_id}")

        return updated_config


# Global registry instance (lazy-loaded)
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
