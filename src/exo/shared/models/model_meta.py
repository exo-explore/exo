from typing import Annotated

import aiofiles
import aiofiles.os as aios
from huggingface_hub import model_info
from loguru import logger
from pydantic import BaseModel, Field

from exo.shared.models.model_cards import MODEL_CARDS
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.worker.download.download_utils import (
    ModelSafetensorsIndex,
    download_file_with_retry,
    ensure_models_dir,
)


class ConfigData(BaseModel):
    model_config = {"extra": "ignore"}  # Allow unknown fields

    # Common field names for number of layers across different architectures
    num_hidden_layers: Annotated[int, Field(ge=0)] | None = None
    num_layers: Annotated[int, Field(ge=0)] | None = None
    n_layer: Annotated[int, Field(ge=0)] | None = None
    n_layers: Annotated[int, Field(ge=0)] | None = None  # Sometimes used
    num_decoder_layers: Annotated[int, Field(ge=0)] | None = None  # Transformer models
    decoder_layers: Annotated[int, Field(ge=0)] | None = None  # Some architectures
    hidden_size: Annotated[int, Field(ge=0)] | None = None

    @property
    def layer_count(self) -> int:
        # Check common field names for layer count
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

        raise ValueError(
            f"No layer count found in config.json: {self.model_dump_json()}"
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
    model_card = next(
        (card for card in MODEL_CARDS.values() if card.model_id == ModelId(model_id)),
        None,
    )

    return ModelMetadata(
        model_id=ModelId(model_id),
        pretty_name=model_card.name if model_card is not None else model_id,
        storage_size=mem_size_bytes,
        n_layers=num_layers,
        hidden_size=config_data.hidden_size or 0,
        # TODO: all custom models currently do not support tensor. We could add a dynamic test for this?
        supports_tensor=model_card.metadata.supports_tensor
        if model_card is not None
        else False,
    )
