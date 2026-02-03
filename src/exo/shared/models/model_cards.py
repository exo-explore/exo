from enum import Enum
from typing import Annotated, Any

import aiofiles
import aiofiles.os as aios
import tomlkit
from anyio import Path, open_file
from huggingface_hub import model_info
from loguru import logger
from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    PositiveInt,
    ValidationError,
    field_validator,
    model_validator,
)
from tomlkit.exceptions import TOMLKitError

from exo.shared.constants import EXO_ENABLE_IMAGE_MODELS, RESOURCES_DIR
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.utils.pydantic_ext import CamelCaseModel

# kinda ugly...
# TODO: load search path from config.toml
_csp = [Path(RESOURCES_DIR) / "inference_model_cards"]
if EXO_ENABLE_IMAGE_MODELS:
    _csp.append(Path(RESOURCES_DIR) / "image_model_cards")

CARD_SEARCH_PATH = _csp

_card_cache: dict[ModelId, "ModelCard"] = {}


async def _refresh_card_cache():
    for path in CARD_SEARCH_PATH:
        async for toml_file in path.rglob("*.toml"):
            try:
                card = await ModelCard.load_from_path(toml_file)
                _card_cache[card.model_id] = card
            except (ValidationError, TOMLKitError):
                pass


async def get_model_cards() -> list["ModelCard"]:
    if len(_card_cache) == 0:
        await _refresh_card_cache()
    return list(_card_cache.values())


class ModelTask(str, Enum):
    TextGeneration = "TextGeneration"
    TextToImage = "TextToImage"
    ImageToImage = "ImageToImage"


class ComponentInfo(CamelCaseModel):
    component_name: str
    component_path: str
    storage_size: Memory
    n_layers: PositiveInt | None
    can_shard: bool
    safetensors_index_filename: str | None


class ModelCard(CamelCaseModel):
    model_id: ModelId
    storage_size: Memory
    n_layers: PositiveInt
    hidden_size: PositiveInt
    supports_tensor: bool
    tasks: list[ModelTask]
    components: list[ComponentInfo] | None = None

    @field_validator("tasks", mode="before")
    @classmethod
    def _validate_tasks(cls, v: list[str | ModelTask]) -> list[ModelTask]:
        return [item if isinstance(item, ModelTask) else ModelTask(item) for item in v]

    async def save(self, path: Path) -> None:
        async with await open_file(path, "w") as f:
            py = self.model_dump(exclude_none=True)
            data = tomlkit.dumps(py)  # pyright: ignore[reportUnknownMemberType]
            await f.write(data)

    async def save_to_default_path(self):
        await self.save(Path(RESOURCES_DIR) / (self.model_id.normalize() + ".toml"))

    @staticmethod
    async def load_from_path(path: Path) -> "ModelCard":
        async with await open_file(path, "r") as f:
            py = tomlkit.loads(await f.read())
            return ModelCard.model_validate(py)

    # Is it okay that model card.load defaults to network access if the card doesn't exist? do we want to be more explicit here?
    @staticmethod
    async def load(model_id: ModelId) -> "ModelCard":
        if model_id not in _card_cache:
            await _refresh_card_cache()
        if (mc := _card_cache.get(model_id)) is not None:
            return mc

        return await ModelCard.fetch_from_hf(model_id)

    @staticmethod
    async def fetch_from_hf(model_id: ModelId) -> "ModelCard":
        """Fetches storage size and number of layers for a Hugging Face model, returns Pydantic ModelMeta."""
        # TODO: failure if files do not exist
        config_data = await fetch_config_data(model_id)
        num_layers = config_data.layer_count
        mem_size_bytes = await fetch_safetensors_size(model_id)

        mc = ModelCard(
            model_id=ModelId(model_id),
            storage_size=mem_size_bytes,
            n_layers=num_layers,
            hidden_size=config_data.hidden_size or 0,
            supports_tensor=config_data.supports_tensor,
            tasks=[ModelTask.TextGeneration],
        )
        await mc.save_to_default_path()
        _card_cache[model_id] = mc
        return mc


# TODO: quantizing and dynamically creating model cards
def _generate_image_model_quant_variants(  # pyright: ignore[reportUnusedFunction]
    base_name: str,
    base_card: ModelCard,
) -> dict[str, ModelCard]:
    """Create quantized variants of an image model card.

    Only the transformer component is quantized; text encoders stay at bf16.
    Sizes are calculated exactly from the base card's component sizes.
    """
    if base_card.components is None:
        raise ValueError(f"Image model {base_name} must have components defined")

    # quantizations = [8, 6, 5, 4, 3]
    quantizations = [8, 4]

    num_transformer_bytes = next(
        c.storage_size.in_bytes
        for c in base_card.components
        if c.component_name == "transformer"
    )

    transformer_bytes = Memory.from_bytes(num_transformer_bytes)

    remaining_bytes = Memory.from_bytes(
        sum(
            c.storage_size.in_bytes
            for c in base_card.components
            if c.component_name != "transformer"
        )
    )

    def with_transformer_size(new_size: Memory) -> list[ComponentInfo]:
        assert base_card.components is not None
        return [
            ComponentInfo(
                component_name=c.component_name,
                component_path=c.component_path,
                storage_size=new_size
                if c.component_name == "transformer"
                else c.storage_size,
                n_layers=c.n_layers,
                can_shard=c.can_shard,
                safetensors_index_filename=c.safetensors_index_filename,
            )
            for c in base_card.components
        ]

    variants = {
        base_name: ModelCard(
            model_id=base_card.model_id,
            storage_size=transformer_bytes + remaining_bytes,
            n_layers=base_card.n_layers,
            hidden_size=base_card.hidden_size,
            supports_tensor=base_card.supports_tensor,
            tasks=base_card.tasks,
            components=with_transformer_size(transformer_bytes),
        )
    }

    for quant in quantizations:
        quant_transformer_bytes = Memory.from_bytes(
            (num_transformer_bytes * quant) // 16
        )
        total_bytes = remaining_bytes + quant_transformer_bytes

        model_id = ModelId(base_card.model_id + f"-{quant}bit")

        variants[f"{base_name}-{quant}bit"] = ModelCard(
            model_id=model_id,
            storage_size=total_bytes,
            n_layers=base_card.n_layers,
            hidden_size=base_card.hidden_size,
            supports_tensor=base_card.supports_tensor,
            tasks=base_card.tasks,
            components=with_transformer_size(quant_transformer_bytes),
        )

    return variants


class ConfigData(BaseModel):
    model_config = {"extra": "ignore"}  # Allow unknown fields

    architectures: list[str] | None = None
    hidden_size: Annotated[int, Field(ge=0)] | None = None
    layer_count: int = Field(
        validation_alias=AliasChoices(
            "num_hidden_layers",
            "num_layers",
            "n_layer",
            "n_layers",
            "num_decoder_layers",
            "decoder_layers",
        )
    )

    @property
    def supports_tensor(self) -> bool:
        return self.architectures in [
            ["Glm4MoeLiteForCausalLM"],
            ["DeepseekV32ForCausalLM"],
            ["DeepseekV3ForCausalLM"],
            ["Qwen3NextForCausalLM"],
            ["Qwen3MoeForCausalLM"],
            ["MiniMaxM2ForCausalLM"],
            ["LlamaForCausalLM"],
            ["GptOssForCausalLM"],
        ]

    @model_validator(mode="before")
    @classmethod
    def defer_to_text_config(cls, data: dict[str, Any]):
        text_config = data.get("text_config")
        if text_config is None:
            return data

        for field in [
            "architectures",
            "hidden_size",
            "num_hidden_layers",
            "num_layers",
            "n_layer",
            "n_layers",
            "num_decoder_layers",
            "decoder_layers",
        ]:
            if (val := text_config.get(field)) is not None:  # pyright: ignore[reportAny]
                data[field] = val

        return data


async def fetch_config_data(model_id: ModelId) -> ConfigData:
    """Downloads and parses config.json for a model."""
    from exo.download.download_utils import (
        download_file_with_retry,
        ensure_models_dir,
    )

    target_dir = (await ensure_models_dir()) / model_id.normalize()
    await aios.makedirs(target_dir, exist_ok=True)
    config_path = await download_file_with_retry(
        model_id,
        "main",
        "config.json",
        target_dir,
        lambda curr_bytes, total_bytes, is_renamed: logger.debug(
            f"Downloading config.json for {model_id}: {curr_bytes}/{total_bytes} ({is_renamed=})"
        ),
    )
    async with aiofiles.open(config_path, "r") as f:
        return ConfigData.model_validate_json(await f.read())


async def fetch_safetensors_size(model_id: ModelId) -> Memory:
    """Gets model size from safetensors index or falls back to HF API."""
    from exo.download.download_utils import (
        download_file_with_retry,
        ensure_models_dir,
    )
    from exo.shared.types.worker.downloads import ModelSafetensorsIndex

    target_dir = (await ensure_models_dir()) / model_id.normalize()
    await aios.makedirs(target_dir, exist_ok=True)
    index_path = await download_file_with_retry(
        model_id,
        "main",
        "model.safetensors.index.json",
        target_dir,
        lambda curr_bytes, total_bytes, is_renamed: logger.debug(
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
