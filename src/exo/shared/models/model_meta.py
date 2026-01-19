from typing import Annotated

import aiofiles
import aiofiles.os as aios
from huggingface_hub import model_info
from loguru import logger
from pydantic import BaseModel, Field

from exo.shared.models.architecture_support import supports_tensor_parallel
from exo.shared.models.model_registry import get_registry
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.worker.download.download_utils import (
    ModelSafetensorsIndex,
    download_file_with_retry,
    ensure_models_dir,
)


class ConfigData(BaseModel):
    model_config = {"extra": "ignore"}  # Allow unknown fields

    # Architecture identification fields
    model_type: str | None = (
        None  # Primary architecture identifier (e.g., "llama", "qwen2")
    )
    architectures: list[str] | None = None  # Fallback list of architecture class names

    # Common field names for number of layers across different architectures
    num_hidden_layers: Annotated[int, Field(ge=0)] | None = None
    num_layers: Annotated[int, Field(ge=0)] | None = None
    n_layer: Annotated[int, Field(ge=0)] | None = None
    n_layers: Annotated[int, Field(ge=0)] | None = None  # Sometimes used
    num_decoder_layers: Annotated[int, Field(ge=0)] | None = None  # Transformer models
    decoder_layers: Annotated[int, Field(ge=0)] | None = None  # Some architectures

    # Common field names for hidden size across different architectures
    hidden_size: Annotated[int, Field(ge=0)] | None = None  # LLaMA, Mistral, Qwen, etc.
    n_embd: Annotated[int, Field(ge=0)] | None = None  # GPT-2
    n_embed: Annotated[int, Field(ge=0)] | None = None  # BLOOM
    d_model: Annotated[int, Field(ge=0)] | None = None  # OLMo, T5, BART

    # Nested config for multimodal models (text model config)
    text_config: dict[str, object] | None = None  # Present in multimodal models

    # Capability detection fields
    vision_config: dict[str, object] | None = None  # Present in vision models
    image_processor_class: str | None = None  # Indicates vision capability
    max_thinking_length: Annotated[int, Field(ge=0)] | None = (
        None  # Indicates reasoning/thinking capability
    )

    @property
    def architecture(self) -> str:
        """Get the normalized architecture identifier.

        Returns model_type if available, otherwise infers from architectures list.
        """
        if self.model_type:
            return self.model_type.lower()

        # Fallback: try to infer from architectures list
        # e.g., ["LlamaForCausalLM"] -> "llama"
        if self.architectures and len(self.architectures) > 0:
            arch_class = self.architectures[0].lower()
            # Extract base architecture from class name
            for suffix in ["forcausallm", "model", "lmheadmodel"]:
                if arch_class.endswith(suffix):
                    return arch_class[: -len(suffix)]
            return arch_class

        return ""

    @property
    def layer_count(self) -> int:
        # Check common field names for layer count at top level
        layer_fields = [
            self.num_hidden_layers,
            self.num_layers,
            self.n_layer,
            self.n_layers,
            self.num_decoder_layers,
            self.decoder_layers,
        ]

        for layer_count in layer_fields:
            if layer_count is not None:
                return layer_count

        # For multimodal models, check text_config for layer count
        if self.text_config is not None:
            text_layer_keys = [
                "num_hidden_layers",
                "num_layers",
                "n_layer",
                "n_layers",
                "num_decoder_layers",
                "decoder_layers",
            ]
            for key in text_layer_keys:
                if key in self.text_config:
                    value = self.text_config[key]
                    if isinstance(value, int) and value > 0:
                        return value

        raise ValueError(
            f"No layer count found in config.json: {self.model_dump_json()}"
        )

    @property
    def hidden_dim(self) -> int:
        """Get hidden size from various possible field names."""
        hidden_fields = [
            self.hidden_size,
            self.n_embd,
            self.n_embed,
            self.d_model,
        ]

        for dim in hidden_fields:
            if dim is not None:
                return dim

        # For multimodal models, check text_config for hidden size
        if self.text_config is not None:
            text_hidden_keys = [
                "hidden_size",
                "n_embd",
                "n_embed",
                "d_model",
            ]
            for key in text_hidden_keys:
                if key in self.text_config:
                    value = self.text_config[key]
                    if isinstance(value, int) and value > 0:
                        return value

        raise ValueError(
            f"No hidden size found in config.json: {self.model_dump_json()}"
        )


async def get_config_data(model_id: str) -> ConfigData:
    """Downloads and parses config.json for a model."""
    target_dir = (await ensure_models_dir()) / str(model_id).replace("/", "--")
    await aios.makedirs(target_dir, exist_ok=True)
    config_path = await download_file_with_retry(
        model_id,
        "main",
        "config.json",
        target_dir,
        lambda curr_bytes, total_bytes, is_renamed: logger.info(
            f"Downloading config.json for {model_id}: {curr_bytes}/{total_bytes} ({is_renamed=})"
        ),
    )
    async with aiofiles.open(config_path, "r") as f:
        return ConfigData.model_validate_json(await f.read())


async def get_safetensors_size(model_id: str) -> Memory:
    """Gets model size from safetensors index or falls back to HF API."""
    target_dir = (await ensure_models_dir()) / str(model_id).replace("/", "--")
    await aios.makedirs(target_dir, exist_ok=True)

    try:
        index_path = await download_file_with_retry(
            model_id,
            "main",
            "model.safetensors.index.json",
            target_dir,
            lambda curr_bytes, total_bytes, is_renamed: logger.info(
                f"Downloading model.safetensors.index.json for {model_id}: {curr_bytes}/{total_bytes} ({is_renamed=})"
            ),
        )
        async with aiofiles.open(index_path, "r") as f:
            index_data = ModelSafetensorsIndex.model_validate_json(await f.read())

        metadata = index_data.metadata
        if metadata is not None:
            return Memory.from_bytes(metadata.total_size)
    except FileNotFoundError:
        # Model doesn't have an index file (single safetensors file)
        # Fall through to HF API
        pass

    # Fallback to HuggingFace API for model size
    info = model_info(model_id)
    if info.safetensors is None:
        raise ValueError(f"No safetensors info found for {model_id}")
    return Memory.from_bytes(info.safetensors.total)


_model_meta_cache: dict[str, ModelMetadata] = {}


async def get_model_meta(model_id: str) -> ModelMetadata:
    if model_id in _model_meta_cache:
        return _model_meta_cache[model_id]
    model_meta = await _get_model_meta(model_id)
    _model_meta_cache[model_id] = model_meta
    return model_meta


async def _get_model_meta(model_id: str) -> ModelMetadata:
    """Fetches storage size and number of layers for a Hugging Face model, returns Pydantic ModelMeta."""
    config_data = await get_config_data(model_id)
    num_layers = config_data.layer_count
    mem_size_bytes = await get_safetensors_size(model_id)

    # Check if we have this model in the registry
    registry = get_registry()
    model_config = registry.get(model_id)

    # Derive supports_tensor from architecture (code is source of truth)
    # Registry config can override if explicitly set, otherwise derive from architecture
    if model_config is not None and model_config.architecture:
        # Use architecture from registry config
        tensor_support = supports_tensor_parallel(model_config.architecture)
    else:
        # Derive from config.json architecture
        tensor_support = supports_tensor_parallel(config_data.architecture)

    return ModelMetadata(
        model_id=ModelId(model_id),
        pretty_name=model_config.name if model_config is not None else model_id,
        storage_size=mem_size_bytes,
        n_layers=num_layers,
        hidden_size=config_data.hidden_dim,
        supports_tensor=tensor_support,
    )
