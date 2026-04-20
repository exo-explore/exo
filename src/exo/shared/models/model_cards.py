import json
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
    ValidationInfo,
    field_validator,
    model_validator,
)
from tomlkit.exceptions import TOMLKitError

from exo.shared.constants import (
    EXO_CUSTOM_MODEL_CARDS_DIR,
    EXO_ENABLE_IMAGE_MODELS,
    EXO_MODELS_DIRS,
    RESOURCES_DIR,
)
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.utils.pydantic_ext import CamelCaseModel

# kinda ugly...
# TODO: load search path from config.toml
_custom_cards_dir = Path(str(EXO_CUSTOM_MODEL_CARDS_DIR))
_BUILTIN_CARD_DIRS = [
    Path(RESOURCES_DIR) / "inference_model_cards",
    Path(RESOURCES_DIR) / "image_model_cards",
]

_card_cache: dict[ModelId, "ModelCard"] = {}


def detect_vision_from_config(model_id: ModelId) -> "VisionCardConfig | None":
    normalized = model_id.normalize()
    for model_dir in [d / normalized for d in EXO_MODELS_DIRS]:
        config_path = model_dir / "config.json"
        if not config_path.exists():
            continue
        try:
            with open(config_path) as f:
                raw = json.load(f)  # type: ignore
            return ConfigData.model_validate(
                raw, context={"model_id": str(model_id)}
            ).vision
        except Exception:
            continue
    return None


async def _load_cards_from_dir(directory: Path, *, is_custom: bool) -> None:
    """Load all TOML model cards from a directory into the cache."""
    async for toml_file in directory.rglob("*.toml"):
        try:
            card = await ModelCard.load_from_path(toml_file)
            if is_custom:
                card = card.model_copy(update={"is_custom": True})
            if card.model_id not in _card_cache:
                _card_cache[card.model_id] = card
        except (ValidationError, TOMLKitError):
            pass


async def _refresh_card_cache() -> None:
    for path in _BUILTIN_CARD_DIRS:
        await _load_cards_from_dir(path, is_custom=False)
    await _load_cards_from_dir(_custom_cards_dir, is_custom=True)


def _is_image_card(card: "ModelCard") -> bool:
    return any(t in (ModelTask.TextToImage, ModelTask.ImageToImage) for t in card.tasks)


def get_card(model_id: ModelId) -> "ModelCard | None":
    """Look up a single model card from the cache by ID."""
    return _card_cache.get(model_id)


async def get_model_cards() -> list["ModelCard"]:
    if len(_card_cache) == 0:
        await _refresh_card_cache()
    if EXO_ENABLE_IMAGE_MODELS:
        return list(_card_cache.values())
    return [c for c in _card_cache.values() if not _is_image_card(c)]


class ModelTask(str, Enum):
    TextGeneration = "TextGeneration"
    TextToImage = "TextToImage"
    ImageToImage = "ImageToImage"


class ComponentInfo(CamelCaseModel):
    component_name: str
    component_path: str
    storage_size: Memory
    n_layers: PositiveInt | None = None
    can_shard: bool
    safetensors_index_filename: str | None = None


class VisionCardConfig(CamelCaseModel):
    image_token_id: int
    model_type: str
    weights_repo: str = ""
    image_token: str | None = None
    processor_repo: str | None = None


class SamplingValues(CamelCaseModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None


class SamplingDefaults(SamplingValues):
    thinking: SamplingValues | None = None
    non_thinking: SamplingValues | None = None


class ModelCard(CamelCaseModel):
    model_id: ModelId
    storage_size: Memory
    n_layers: PositiveInt
    hidden_size: PositiveInt
    supports_tensor: bool
    num_key_value_heads: PositiveInt | None = None
    tasks: list[ModelTask]
    components: list[ComponentInfo] | None = None
    family: str = ""
    quantization: str = ""
    base_model: str = ""
    capabilities: list[str] = []
    context_length: int = 0
    uses_cfg: bool = False
    trust_remote_code: bool = True
    is_custom: bool = False
    vision: VisionCardConfig | None = None
    sampling_defaults: SamplingDefaults = Field(default_factory=SamplingDefaults)

    @model_validator(mode="after")
    def _autodetect_vision(self) -> "ModelCard":
        if self.vision is None:
            detected = detect_vision_from_config(self.model_id)
            if detected is not None:
                object.__setattr__(self, "vision", detected)
        return self

    @model_validator(mode="after")
    def _fill_vision_weights_repo(self) -> "ModelCard":
        if self.vision is not None and not self.vision.weights_repo:
            object.__setattr__(
                self,
                "vision",
                self.vision.model_copy(update={"weights_repo": str(self.model_id)}),
            )
        return self

    @field_validator("tasks", mode="before")
    @classmethod
    def _validate_tasks(cls, v: list[str | ModelTask]) -> list[ModelTask]:
        return [item if isinstance(item, ModelTask) else ModelTask(item) for item in v]

    async def save(self, path: Path) -> None:
        async with await open_file(path, "w") as f:
            py = self.model_dump(exclude_none=True, exclude={"is_custom"})
            data = tomlkit.dumps(py)  # pyright: ignore[reportUnknownMemberType]
            await f.write(data)

    async def save_to_custom_dir(self) -> None:
        await aios.makedirs(str(_custom_cards_dir), exist_ok=True)
        await self.save(_custom_cards_dir / (self.model_id.normalize() + ".toml"))

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

        mc = await ModelCard.fetch_from_hf(model_id)
        await mc.save_to_custom_dir()
        _card_cache[model_id] = mc
        return mc

    @staticmethod
    async def fetch_from_hf(model_id: ModelId) -> "ModelCard":
        """Fetches storage size and number of layers for a Hugging Face model, returns Pydantic ModelMeta.

        This is a pure fetch — it does NOT save to disk or update the cache.
        Persistence is handled by the event-sourcing layer (worker event handler).
        """
        # TODO: failure if files do not exist
        config_data = await fetch_config_data(model_id)
        num_layers = config_data.layer_count
        mem_size_bytes = await fetch_safetensors_size(model_id)

        return ModelCard(
            model_id=ModelId(model_id),
            storage_size=mem_size_bytes,
            n_layers=num_layers,
            hidden_size=config_data.hidden_size or 0,
            supports_tensor=config_data.supports_tensor,
            num_key_value_heads=config_data.num_key_value_heads,
            context_length=config_data.max_position_embeddings,
            tasks=[ModelTask.TextGeneration],
            trust_remote_code=False,
            is_custom=True,
            vision=config_data.vision,
        )


def add_to_card_cache(card: "ModelCard") -> None:
    """Add or update a model card in the in-memory cache."""
    _card_cache[card.model_id] = card


async def delete_custom_card(model_id: ModelId) -> bool:
    """Delete a user-added custom model card. Returns True if deleted."""
    card_path = _custom_cards_dir / (ModelId(model_id).normalize() + ".toml")
    if await card_path.exists():
        await card_path.unlink()
        _card_cache.pop(model_id, None)
        return True
    return False


class ConfigData(BaseModel):
    model_config = {"extra": "ignore"}  # Allow unknown fields

    architectures: list[str] | None = None
    hidden_size: Annotated[int, Field(ge=0)] | None = None
    num_key_value_heads: PositiveInt | None = None
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
    max_position_embeddings: int = 0
    vision: VisionCardConfig | None = None

    @property
    def supports_tensor(self) -> bool:
        return self.architectures in [
            ["Glm4MoeLiteForCausalLM"],
            ["GlmMoeDsaForCausalLM"],
            ["DeepseekV32ForCausalLM"],
            ["DeepseekV3ForCausalLM"],
            ["Qwen3NextForCausalLM"],
            ["Qwen3MoeForCausalLM"],
            ["Qwen3_5MoeForConditionalGeneration"],
            ["Qwen3_5ForConditionalGeneration"],
            ["Qwen3VLForConditionalGeneration"],
            ["MiniMaxM2ForCausalLM"],
            ["LlamaForCausalLM"],
            ["GptOssForCausalLM"],
            ["Step3p5ForCausalLM"],
            ["NemotronHForCausalLM"],
            ["Gemma4ForConditionalGeneration"],
        ]

    @model_validator(mode="before")
    @classmethod
    def defer_to_text_config(cls, data: dict[str, Any], info: ValidationInfo):
        text_config = data.get("text_config")
        if text_config is not None:
            for field in [
                "architectures",
                "hidden_size",
                "num_key_value_heads",
                "max_position_embeddings",
                "num_hidden_layers",
                "num_layers",
                "n_layer",
                "n_layers",
                "num_decoder_layers",
                "decoder_layers",
            ]:
                if (val := text_config.get(field)) is not None:  # pyright: ignore[reportAny]
                    data[field] = val

        vision_config = data.get("vision_config")
        image_token_id = data.get("image_token_id")
        if vision_config is not None and image_token_id is not None:
            model_type = str(
                data.get("model_type", vision_config.get("model_type", ""))  # pyright: ignore[reportAny]
            )
            assert info.context is not None

            data["vision"] = VisionCardConfig(
                image_token_id=int(image_token_id),  # pyright: ignore[reportAny]
                model_type=model_type,
                weights_repo=info.context["model_id"],  # type: ignore
            )

        return data


async def fetch_config_data(model_id: ModelId) -> ConfigData:
    """Downloads and parses config.json for a model."""
    from exo.download.download_utils import (
        download_file_with_retry,
        resolve_model_dir,
    )

    target_dir = await resolve_model_dir(model_id)
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
        return ConfigData.model_validate_json(
            await f.read(), context={"model_id": str(model_id)}
        )


async def fetch_safetensors_size(model_id: ModelId) -> Memory:
    """Gets model size from safetensors index or falls back to HF API."""
    from exo.download.download_utils import (
        download_file_with_retry,
        resolve_model_dir,
    )
    from exo.shared.types.worker.downloads import ModelSafetensorsIndex

    target_dir = await resolve_model_dir(model_id)
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
