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
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.text_generation import ReasoningDialect
from exo.utils.pydantic_ext import FrozenModel

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
            if is_custom or card.model_id not in _card_cache:
                _card_cache[card.model_id] = card
        except (ValidationError, TOMLKitError):
            pass


async def _refresh_card_cache() -> None:
    for path in _BUILTIN_CARD_DIRS:
        await _load_cards_from_dir(path, is_custom=False)
    await _load_cards_from_dir(_custom_cards_dir, is_custom=True)


async def _refresh_custom_card_cache() -> None:
    await _load_cards_from_dir(_custom_cards_dir, is_custom=True)


def _is_image_card(card: "ModelCard") -> bool:
    return any(t in (ModelTask.TextToImage, ModelTask.ImageToImage) for t in card.tasks)


def get_card(model_id: ModelId) -> "ModelCard | None":
    """Look up a single model card from the cache by ID."""
    return _card_cache.get(model_id)


async def get_model_cards() -> list["ModelCard"]:
    if len(_card_cache) == 0:
        await _refresh_card_cache()
    else:
        await _refresh_custom_card_cache()
    if EXO_ENABLE_IMAGE_MODELS:
        return list(_card_cache.values())
    return [c for c in _card_cache.values() if not _is_image_card(c)]


class ModelTask(str, Enum):
    TextGeneration = "TextGeneration"
    TextToImage = "TextToImage"
    ImageToImage = "ImageToImage"


class ComponentInfo(FrozenModel):
    component_name: str
    component_path: str
    storage_size: Memory
    n_layers: PositiveInt | None = None
    can_shard: bool
    safetensors_index_filename: str | None = None


class VisionCardConfig(FrozenModel):
    image_token_id: int
    model_type: str
    weights_repo: str = ""
    image_token: str | None = None
    processor_repo: str | None = None


class SamplingValues(FrozenModel):
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


class ModelCard(FrozenModel):
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
    reasoning_dialect: ReasoningDialect = "none"
    context_length: int = 0
    uses_cfg: bool = False
    trust_remote_code: bool = True
    is_custom: bool = False
    vision: VisionCardConfig | None = None
    sampling_defaults: SamplingDefaults = Field(default_factory=SamplingDefaults)
    # Optional speculative-decoding draft models. Listed in *preference order*:
    # the first entry is treated as the default ("fastest") choice. Runners pick
    # one based on `EXO_DRAFTER_PREFERENCE` (`fastest` / `highest_acceptance` /
    # `auto`), falling back to whichever weights are already on disk. All
    # listed drafters MUST share a tokenizer with the target. Conventionally
    # the list is quant-aligned with the target (e.g. `gemma-4-31b-it-4bit`
    # declares `[gemma-4-e2b-it-4bit, gemma-4-e4b-it-4bit]`), but cross-quant
    # drafters are allowed for advanced tuning. These are *standard external*
    # drafters: independent small LMs that decode autoregressively from their
    # own KV cache and ship only token ids over the wire. They compose with
    # asymmetric placement (``drafter_eligible_nodes``) because token-only
    # transport is bandwidth-cheap.
    drafter_model_ids: list[ModelId] = Field(default_factory=list)
    # Optional MTP-style "coupled" drafter for this target. Coupled drafters
    # (e.g. Google's Gemma 4 assistant ``gemma4_assistant`` model_type, or
    # Z-Lab's Qwen3 ``dflash`` drafters) attach to the target architecturally:
    # they consume the target's hidden state every draft step and -- for the
    # MTP variant -- read the target's KV cache directly instead of building
    # their own. This couples them tightly to the target but yields the
    # ~2x speedup Apple/Google reported for MLX-native MTP.
    #
    # The kind (``"mtp"`` for Gemma 4 assistant drafters, ``"dflash"`` for
    # Qwen3 DFlash) is auto-detected from the drafter's HF ``model_type`` at
    # load time via ``mlx_vlm.speculative.drafters.resolve_drafter_kind``. The
    # drafter is loaded through ``mlx_vlm`` (not ``mlx_lm``) because the
    # speculative-drafter loader and architecture live there.
    #
    # Composition with ``drafter_model_ids`` and ``drafter_eligible_nodes``:
    # - When ``drafter_eligible_nodes`` is non-empty AND ``drafter_model_ids``
    #   is non-empty, asymmetric placement (PR #20's pipeline) wins because
    #   the coupled drafter's wire protocol would have to ship full hidden
    #   states / KV cache entries cross-node, which negates its speedup over
    #   any practical link.
    # - Otherwise (single-node placement), if ``coupled_drafter`` is set the
    #   runner loads it via ``mlx_vlm`` and the generator routes through
    #   ``draft_block`` instead of the standard external-drafter loop.
    # - If neither asymmetric nor coupled applies, the legacy single-device
    #   standard-drafter path runs as before.
    #
    # Empty / ``None`` (the default) preserves legacy behaviour. This field
    # is purely additive: cards that don't declare a coupled drafter are
    # functionally unchanged.
    coupled_drafter: ModelId | None = None
    # Nodes the operator has designated as eligible drafter hosts. When this
    # list is non-empty AND the model has at least one declared drafter, the
    # placement layer attempts asymmetric placement: target ranks land on the
    # selected target cycle, the drafter is loaded on the first eligible node
    # reachable from target rank 0 (RDMA for `MlxJaccl`, socket for `MlxRing`),
    # and the parent `mx.distributed` group spans both. Eligibility is
    # *operator-controlled*, not auto-discovered: the operator opts a node in
    # by listing its `NodeId` here (typically in a custom card under
    # `~/.exo/custom_model_cards/`). If no listed node is reachable, placement
    # emits a `DrafterPlacementDegraded` event and falls back -- the user's
    # request still completes, the operator just doesn't get the asymmetric
    # speedup until they fix the eligibility list. Empty (the default) preserves
    # legacy single-device drafter behaviour.
    drafter_eligible_nodes: list[NodeId] = Field(default_factory=list)
    # Nodes the operator has designated as eligible *prefill-only* hosts for
    # this model. When non-empty, placement auto-creates a single-rank
    # prefill-only sibling instance on each viable node (sufficient RAM,
    # alive in topology, not already a target/drafter rank) and emits an
    # ``InstanceLinkCreated`` linking them to the decode instance. The
    # master then routes incoming requests' prefill traffic across the
    # linked prefill instances by in-flight task count, giving the
    # decode instance multi-GPU prefill parallelism for free.
    #
    # This is the right lever for "I have spare nodes in my cluster --
    # use them for prefill so slot N's TTFT doesn't queue behind slot 0's
    # prefill on a single GPU." It composes orthogonally with
    # ``drafter_eligible_nodes``: the chosen drafter node is excluded
    # from prefill candidates automatically.
    #
    # Failure modes are loud-but-graceful: if a candidate fails RAM
    # feasibility or is unreachable the placement layer skips it and
    # logs; the decode instance still comes up. If *no* candidate
    # succeeds, no link is emitted and the user's traffic prefills
    # locally on the target rank as before.
    prefill_eligible_nodes: list[NodeId] = Field(default_factory=list)

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
    async def load_cached_only(model_id: ModelId) -> "ModelCard | None":
        """Local-only variant of :meth:`load`.

        Returns the cached :class:`ModelCard` for ``model_id`` if one
        is present in the in-memory cache or any of the on-disk built-
        in / custom card directories. Returns ``None`` when no cached
        copy exists; never falls back to :meth:`fetch_from_hf`.

        Codex P1 (PR #18, coordinator.py:723 + 908). The full
        :meth:`load` path is unsafe on the master's command-processing
        coroutine because ``fetch_from_hf`` issues blocking HTTP
        requests to Hugging Face when the card is not already on disk.
        That path is reached during ``StartDownload`` /
        ``DeleteDownload`` cascade rebuilds for any drafter or
        previously-installed target whose card was not saved to a
        custom dir; in offline / disconnected environments it stalls
        the entire command queue. The delete-cascade and
        drafter-chain code paths only need cards that are actually on
        the local disk (otherwise the parent target could not have
        been downloaded in the first place), so they should call this
        cache-only variant and treat ``None`` as "no rediscovered
        links". The full :meth:`load` is reserved for paths that
        legitimately need to pull a previously-unseen card from HF
        (initial ``StartDownload`` of a third-party model id).
        """
        if model_id not in _card_cache:
            await _refresh_card_cache()
        return _card_cache.get(model_id)

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
            ["DeepseekV4ForCausalLM"],
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
    if metadata is not None and metadata.total_size is not None:
        return Memory.from_bytes(metadata.total_size)

    info = model_info(model_id)
    if info.safetensors is None:
        raise ValueError(f"No safetensors info found for {model_id}")
    return Memory.from_bytes(info.safetensors.total)
