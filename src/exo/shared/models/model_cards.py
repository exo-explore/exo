from enum import Enum
from typing import Annotated
from pathlib import Path as FsPath
import struct
import os

import aiofiles
import aiofiles.os as aios
import tomlkit
from anyio import Path, open_file
from huggingface_hub import model_info
from loguru import logger
from pydantic import BaseModel, Field, PositiveInt, field_validator
from enum import Enum

from exo.shared.constants import EXO_ENABLE_IMAGE_MODELS
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.utils.pydantic_ext import CamelCaseModel

_card_cache: dict[str, "ModelCard"] = {}


def _parse_env_gguf_models() -> list[ModelId]:
    raw = os.getenv("EXO_GGUF_MODELS", "")
    if not raw:
        return []
    parts = [item.strip() for item in raw.replace("\n", ",").split(",")]
    return [ModelId(part) for part in parts if part]


def get_env_gguf_model_ids() -> list[ModelId]:
    return _parse_env_gguf_models()


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
            py = self.model_dump()
            data = tomlkit.dumps(py)  # pyright: ignore[reportUnknownMemberType]
            await f.write(data)

    @staticmethod
    async def load_from_path(path: Path) -> "ModelCard":
        async with await open_file(path, "r") as f:
            py = tomlkit.loads(await f.read())
            return ModelCard.model_validate(py)

    @staticmethod
    async def load(model_id: ModelId) -> "ModelCard":
        for card in MODEL_CARDS.values():
            if card.model_id == model_id:
                return card
        return await ModelCard.from_hf(model_id)

    @staticmethod
    async def from_hf(model_id: ModelId) -> "ModelCard":
        """Fetches storage size and number of layers for a Hugging Face model, returns Pydantic ModelMeta."""
        if (mc := _card_cache.get(model_id)) is not None:
            return mc
        config_data = await get_config_data(model_id)
        num_layers = config_data.layer_count
        mem_size_bytes = await get_safetensors_size(model_id)

        mc = ModelCard(
            model_id=ModelId(model_id),
            storage_size=mem_size_bytes,
            n_layers=num_layers,
            hidden_size=config_data.hidden_size or 0,
            supports_tensor=config_data.supports_tensor,
            tasks=[ModelTask.TextGeneration],
        )
        _card_cache[model_id] = mc
        return mc

    @staticmethod
    async def from_gguf(model_id: ModelId, gguf_path: FsPath) -> "ModelCard":
        if (mc := _card_cache.get(str(model_id))) is not None:
            return mc
        n_layers, hidden_size = _parse_gguf_header(gguf_path)
        storage_size = Memory.from_bytes(gguf_path.stat().st_size)
        mc = ModelCard(
            model_id=ModelId(model_id),
            storage_size=storage_size,
            n_layers=n_layers,
            hidden_size=hidden_size,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        )
        _card_cache[str(model_id)] = mc
        return mc


async def resolve_model_card(model_id: ModelId) -> "ModelCard":
    if (mc := _card_cache.get(str(model_id))) is not None:
        return mc
    raw_model_id = str(model_id)
    from exo.download.download_utils import (
        ensure_remote_gguf,
        parse_remote_gguf_spec,
        resolve_local_gguf_path,
    )

    if raw_model_id.lower().endswith(".gguf"):
        candidate = FsPath(raw_model_id)
        if candidate.is_absolute():
            if not candidate.exists():
                raise ValueError(f"GGUF file not found: {candidate}")
            try:
                return await ModelCard.from_gguf(model_id, candidate)
            except Exception as exc:
                logger.warning(
                    f"Failed to parse GGUF metadata for {model_id}: {exc}; using fallback."
                )
                size_bytes = candidate.stat().st_size
                return _build_gguf_fallback_card(model_id, size_bytes)

    local_path = resolve_local_gguf_path(model_id)
    if local_path is not None and local_path.exists():
        try:
            return await ModelCard.from_gguf(model_id, local_path)
        except Exception as exc:
            logger.warning(
                f"Failed to parse GGUF metadata for {model_id}: {exc}; using fallback."
            )
            size_bytes = local_path.stat().st_size
            return _build_gguf_fallback_card(model_id, size_bytes)

    if parse_remote_gguf_spec(model_id):
        gguf_path = await ensure_remote_gguf(model_id)
        if gguf_path is None:
            raise ValueError(f"Invalid GGUF model id: {model_id}")
        try:
            return await ModelCard.from_gguf(model_id, gguf_path)
        except Exception as exc:
            logger.warning(
                f"Failed to parse GGUF metadata for {model_id}: {exc}; using fallback."
            )
            size_bytes = gguf_path.stat().st_size
            return _build_gguf_fallback_card(model_id, size_bytes)
    if raw_model_id.lower().endswith(".gguf"):
        raise ValueError(f"GGUF file not found: {raw_model_id}")
    return await ModelCard.from_hf(model_id)


async def get_env_gguf_model_cards() -> list["ModelCard"]:
    cards: list[ModelCard] = []
    for model_id in get_env_gguf_model_ids():
        try:
            cards.append(await resolve_model_card(model_id))
        except Exception as exc:
            logger.warning(f"Failed to load GGUF model {model_id}: {exc}")
    return cards


async def resolve_model_card_for_download(model_id: ModelId) -> "ModelCard":
    from exo.download.download_utils import (
        file_meta,
        parse_remote_gguf_spec,
        resolve_local_gguf_path,
    )

    if spec := parse_remote_gguf_spec(model_id):
        local_path = resolve_local_gguf_path(model_id)
        if local_path is not None and local_path.exists():
            try:
                return await ModelCard.from_gguf(model_id, local_path)
            except Exception as exc:
                logger.warning(
                    f"Failed to parse local GGUF metadata for {model_id}: {exc}; using fallback."
                )
                size_bytes = local_path.stat().st_size
                return _build_gguf_fallback_card(model_id, size_bytes)
        repo_id, revision, file_path = spec
        size_bytes, _etag = await file_meta(repo_id, revision, file_path)
        return _build_gguf_fallback_card(model_id, size_bytes)

    return await resolve_model_card(model_id)


async def get_local_gguf_model_cards() -> list["ModelCard"]:
    from exo.shared.constants import EXO_MODELS_DIR

    if not EXO_MODELS_DIR.exists():
        return []

    cards: list[ModelCard] = []
    for gguf_path in EXO_MODELS_DIR.rglob("*.gguf"):
        if not gguf_path.is_file():
            continue
        model_id = ModelId(str(gguf_path))
        try:
            cards.append(await ModelCard.from_gguf(model_id, gguf_path))
        except Exception as exc:
            logger.warning(
                f"Failed to parse GGUF metadata for {gguf_path}: {exc}; using fallback."
            )
            cards.append(_build_gguf_fallback_card(model_id, gguf_path.stat().st_size))
    return cards


def _build_gguf_fallback_card(model_id: ModelId, size_bytes: int) -> "ModelCard":
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_bytes(size_bytes),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )


def _parse_gguf_header(path: FsPath) -> tuple[int, int]:
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Invalid GGUF header in {path}")

        _version = _read_u32(f)
        _tensor_count = _read_u64(f)
        kv_count = _read_u64(f)

        n_layers: int | None = None
        hidden_size: int | None = None

        for _ in range(kv_count):
            key = _read_string(f)
            value_type = _read_u32(f)
            value = _read_gguf_value(f, value_type)
            if key in ("llama.block_count", "llama.n_layer", "llama.n_layers") or key.endswith(
                (".block_count", ".n_layer", ".n_layers")
            ):
                if isinstance(value, int):
                    n_layers = value
            elif key == "llama.embedding_length" or key.endswith(".embedding_length"):
                if isinstance(value, int):
                    hidden_size = value

        if n_layers is None or hidden_size is None:
            raise ValueError(
                f"Missing GGUF metadata in {path} (n_layers={n_layers}, hidden_size={hidden_size})"
            )
        return n_layers, hidden_size


def _read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("Unexpected end of GGUF file")
    return data


def _read_u32(handle) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _read_u64(handle) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _read_i32(handle) -> int:
    return struct.unpack("<i", _read_exact(handle, 4))[0]


def _read_i64(handle) -> int:
    return struct.unpack("<q", _read_exact(handle, 8))[0]


def _read_string(handle) -> str:
    raw = _read_exact(handle, 8)
    length_u64 = struct.unpack("<Q", raw)[0]
    pos = handle.tell()
    end = handle.seek(0, 2)
    handle.seek(pos)
    remaining = end - pos
    if length_u64 > remaining:
        length_u32 = struct.unpack("<I", raw[:4])[0]
        handle.seek(-4, 1)
        length = length_u32
    else:
        length = length_u64
    return _read_exact(handle, length).decode("utf-8", errors="replace")


def _read_gguf_value(handle, value_type: int):
    if value_type == 0:
        return _read_exact(handle, 1)[0]
    if value_type == 1:
        return struct.unpack("<b", _read_exact(handle, 1))[0]
    if value_type == 2:
        return struct.unpack("<H", _read_exact(handle, 2))[0]
    if value_type == 3:
        return struct.unpack("<h", _read_exact(handle, 2))[0]
    if value_type == 4:
        return _read_u32(handle)
    if value_type == 5:
        return _read_i32(handle)
    if value_type == 6:
        return struct.unpack("<f", _read_exact(handle, 4))[0]
    if value_type == 7:
        return bool(_read_exact(handle, 1)[0])
    if value_type == 8:
        return _read_string(handle)
    if value_type == 9:
        elem_type = _read_u32(handle)
        count = _read_u64(handle)
        _skip_gguf_array(handle, elem_type, count)
        return None
    if value_type == 10:
        return _read_u64(handle)
    if value_type == 11:
        return _read_i64(handle)
    if value_type == 12:
        return struct.unpack("<d", _read_exact(handle, 8))[0]
    raise ValueError(f"Unsupported GGUF value type: {value_type}")


def _skip_gguf_array(handle, elem_type: int, count: int) -> None:
    if count == 0:
        return
    if elem_type in (0, 1, 7):
        handle.seek(count, 1)
        return
    if elem_type in (2, 3):
        handle.seek(count * 2, 1)
        return
    if elem_type in (4, 5, 6):
        handle.seek(count * 4, 1)
        return
    if elem_type in (10, 11, 12):
        handle.seek(count * 8, 1)
        return
    if elem_type == 8:
        for _ in range(count):
            _ = _read_string(handle)
        return
    raise ValueError(f"Unsupported GGUF array element type: {elem_type}")


MODEL_CARDS: dict[str, ModelCard] = {
    # deepseek v3
    "deepseek-v3.1-4bit": ModelCard(
        model_id=ModelId("mlx-community/DeepSeek-V3.1-4bit"),
        storage_size=Memory.from_gb(378),
        n_layers=61,
        hidden_size=7168,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "deepseek-v3.1-8bit": ModelCard(
        model_id=ModelId("mlx-community/DeepSeek-V3.1-8bit"),
        storage_size=Memory.from_gb(713),
        n_layers=61,
        hidden_size=7168,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    # kimi k2
    "kimi-k2-instruct-4bit": ModelCard(
        model_id=ModelId("mlx-community/Kimi-K2-Instruct-4bit"),
        storage_size=Memory.from_gb(578),
        n_layers=61,
        hidden_size=7168,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "kimi-k2-thinking": ModelCard(
        model_id=ModelId("mlx-community/Kimi-K2-Thinking"),
        storage_size=Memory.from_gb(658),
        n_layers=61,
        hidden_size=7168,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    # llama-3.1
    "llama-3.1-8b": ModelCard(
        model_id=ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"),
        storage_size=Memory.from_mb(4423),
        n_layers=32,
        hidden_size=4096,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "llama-3.1-8b-8bit": ModelCard(
        model_id=ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"),
        storage_size=Memory.from_mb(8540),
        n_layers=32,
        hidden_size=4096,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "llama-3.1-8b-bf16": ModelCard(
        model_id=ModelId("mlx-community/Meta-Llama-3.1-8B-Instruct-bf16"),
        storage_size=Memory.from_mb(16100),
        n_layers=32,
        hidden_size=4096,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "llama-3.1-70b": ModelCard(
        model_id=ModelId("mlx-community/Meta-Llama-3.1-70B-Instruct-4bit"),
        storage_size=Memory.from_mb(38769),
        n_layers=80,
        hidden_size=8192,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    # llama-3.2
    "llama-3.2-1b": ModelCard(
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        storage_size=Memory.from_mb(696),
        n_layers=16,
        hidden_size=2048,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "llama-3.2-3b": ModelCard(
        model_id=ModelId("mlx-community/Llama-3.2-3B-Instruct-4bit"),
        storage_size=Memory.from_mb(1777),
        n_layers=28,
        hidden_size=3072,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "llama-3.2-3b-8bit": ModelCard(
        model_id=ModelId("mlx-community/Llama-3.2-3B-Instruct-8bit"),
        storage_size=Memory.from_mb(3339),
        n_layers=28,
        hidden_size=3072,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    # llama-3.3
    "llama-3.3-70b": ModelCard(
        model_id=ModelId("mlx-community/Llama-3.3-70B-Instruct-4bit"),
        storage_size=Memory.from_mb(38769),
        n_layers=80,
        hidden_size=8192,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "llama-3.3-70b-8bit": ModelCard(
        model_id=ModelId("mlx-community/Llama-3.3-70B-Instruct-8bit"),
        storage_size=Memory.from_mb(73242),
        n_layers=80,
        hidden_size=8192,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "llama-3.3-70b-fp16": ModelCard(
        model_id=ModelId("mlx-community/llama-3.3-70b-instruct-fp16"),
        storage_size=Memory.from_mb(137695),
        n_layers=80,
        hidden_size=8192,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    # qwen3
    "qwen3-0.6b": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-0.6B-4bit"),
        storage_size=Memory.from_mb(327),
        n_layers=28,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    ),
    "qwen3-0.6b-8bit": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-0.6B-8bit"),
        storage_size=Memory.from_mb(666),
        n_layers=28,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    ),
    "qwen3-30b": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-30B-A3B-4bit"),
        storage_size=Memory.from_mb(16797),
        n_layers=48,
        hidden_size=2048,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "qwen3-30b-8bit": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-30B-A3B-8bit"),
        storage_size=Memory.from_mb(31738),
        n_layers=48,
        hidden_size=2048,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "qwen3-80b-a3B-4bit": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit"),
        storage_size=Memory.from_mb(44800),
        n_layers=48,
        hidden_size=2048,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "qwen3-80b-a3B-8bit": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit"),
        storage_size=Memory.from_mb(84700),
        n_layers=48,
        hidden_size=2048,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "qwen3-80b-a3B-thinking-4bit": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit"),
        storage_size=Memory.from_mb(84700),
        n_layers=48,
        hidden_size=2048,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "qwen3-80b-a3B-thinking-8bit": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-Next-80B-A3B-Thinking-8bit"),
        storage_size=Memory.from_mb(84700),
        n_layers=48,
        hidden_size=2048,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "qwen3-235b-a22b-4bit": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit"),
        storage_size=Memory.from_gb(132),
        n_layers=94,
        hidden_size=4096,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "qwen3-235b-a22b-8bit": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-235B-A22B-Instruct-2507-8bit"),
        storage_size=Memory.from_gb(250),
        n_layers=94,
        hidden_size=4096,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "qwen3-coder-480b-a35b-4bit": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-Coder-480B-A35B-Instruct-4bit"),
        storage_size=Memory.from_gb(270),
        n_layers=62,
        hidden_size=6144,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "qwen3-coder-480b-a35b-8bit": ModelCard(
        model_id=ModelId("mlx-community/Qwen3-Coder-480B-A35B-Instruct-8bit"),
        storage_size=Memory.from_gb(540),
        n_layers=62,
        hidden_size=6144,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    # gpt-oss
    "gpt-oss-120b-MXFP4-Q8": ModelCard(
        model_id=ModelId("mlx-community/gpt-oss-120b-MXFP4-Q8"),
        storage_size=Memory.from_kb(68_996_301),
        n_layers=36,
        hidden_size=2880,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "gpt-oss-20b-MXFP4-Q8": ModelCard(
        model_id=ModelId("mlx-community/gpt-oss-20b-MXFP4-Q8"),
        storage_size=Memory.from_kb(11_744_051),
        n_layers=24,
        hidden_size=2880,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    # glm 4.5
    "glm-4.5-air-8bit": ModelCard(
        # Needs to be quantized g32 or g16 to work with tensor parallel
        model_id=ModelId("mlx-community/GLM-4.5-Air-8bit"),
        storage_size=Memory.from_gb(114),
        n_layers=46,
        hidden_size=4096,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    ),
    "glm-4.5-air-bf16": ModelCard(
        model_id=ModelId("mlx-community/GLM-4.5-Air-bf16"),
        storage_size=Memory.from_gb(214),
        n_layers=46,
        hidden_size=4096,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    # glm 4.7
    "glm-4.7-4bit": ModelCard(
        model_id=ModelId("mlx-community/GLM-4.7-4bit"),
        storage_size=Memory.from_bytes(198556925568),
        n_layers=91,
        hidden_size=5120,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "glm-4.7-6bit": ModelCard(
        model_id=ModelId("mlx-community/GLM-4.7-6bit"),
        storage_size=Memory.from_bytes(286737579648),
        n_layers=91,
        hidden_size=5120,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "glm-4.7-8bit-gs32": ModelCard(
        model_id=ModelId("mlx-community/GLM-4.7-8bit-gs32"),
        storage_size=Memory.from_bytes(396963397248),
        n_layers=91,
        hidden_size=5120,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    # glm 4.7 flash
    "glm-4.7-flash-4bit": ModelCard(
        model_id=ModelId("mlx-community/GLM-4.7-Flash-4bit"),
        storage_size=Memory.from_gb(18),
        n_layers=47,
        hidden_size=2048,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "glm-4.7-flash-5bit": ModelCard(
        model_id=ModelId("mlx-community/GLM-4.7-Flash-5bit"),
        storage_size=Memory.from_gb(21),
        n_layers=47,
        hidden_size=2048,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "glm-4.7-flash-6bit": ModelCard(
        model_id=ModelId("mlx-community/GLM-4.7-Flash-6bit"),
        storage_size=Memory.from_gb(25),
        n_layers=47,
        hidden_size=2048,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "glm-4.7-flash-8bit": ModelCard(
        model_id=ModelId("mlx-community/GLM-4.7-Flash-8bit"),
        storage_size=Memory.from_gb(32),
        n_layers=47,
        hidden_size=2048,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    # minimax-m2
    "minimax-m2.1-8bit": ModelCard(
        model_id=ModelId("mlx-community/MiniMax-M2.1-8bit"),
        storage_size=Memory.from_bytes(242986745856),
        n_layers=61,
        hidden_size=3072,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
    "minimax-m2.1-3bit": ModelCard(
        model_id=ModelId("mlx-community/MiniMax-M2.1-3bit"),
        storage_size=Memory.from_bytes(100086644736),
        n_layers=61,
        hidden_size=3072,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    ),
}

_IMAGE_MODEL_CARDS: dict[str, ModelCard] = {
    "flux1-schnell": ModelCard(
        model_id=ModelId("black-forest-labs/FLUX.1-schnell"),
        storage_size=Memory.from_bytes(23782357120 + 9524621312),
        n_layers=57,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextToImage],
        components=[
            ComponentInfo(
                component_name="text_encoder",
                component_path="text_encoder/",
                storage_size=Memory.from_kb(0),
                n_layers=12,
                can_shard=False,
                safetensors_index_filename=None,  # Single file
            ),
            ComponentInfo(
                component_name="text_encoder_2",
                component_path="text_encoder_2/",
                storage_size=Memory.from_bytes(9524621312),
                n_layers=24,
                can_shard=False,
                safetensors_index_filename="model.safetensors.index.json",
            ),
            ComponentInfo(
                component_name="transformer",
                component_path="transformer/",
                storage_size=Memory.from_bytes(23782357120),
                n_layers=57,  # 19 transformer_blocks + 38 single_transformer_blocks
                can_shard=True,
                safetensors_index_filename="diffusion_pytorch_model.safetensors.index.json",
            ),
            ComponentInfo(
                component_name="vae",
                component_path="vae/",
                storage_size=Memory.from_kb(0),
                n_layers=None,
                can_shard=False,
                safetensors_index_filename=None,
            ),
        ],
    ),
    "flux1-dev": ModelCard(
        model_id=ModelId("black-forest-labs/FLUX.1-dev"),
        storage_size=Memory.from_bytes(23782357120 + 9524621312),
        n_layers=57,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextToImage],
        components=[
            ComponentInfo(
                component_name="text_encoder",
                component_path="text_encoder/",
                storage_size=Memory.from_kb(0),
                n_layers=12,
                can_shard=False,
                safetensors_index_filename=None,  # Single file
            ),
            ComponentInfo(
                component_name="text_encoder_2",
                component_path="text_encoder_2/",
                storage_size=Memory.from_bytes(9524621312),
                n_layers=24,
                can_shard=False,
                safetensors_index_filename="model.safetensors.index.json",
            ),
            ComponentInfo(
                component_name="transformer",
                component_path="transformer/",
                storage_size=Memory.from_bytes(23802816640),
                n_layers=57,  # 19 transformer_blocks + 38 single_transformer_blocks
                can_shard=True,
                safetensors_index_filename="diffusion_pytorch_model.safetensors.index.json",
            ),
            ComponentInfo(
                component_name="vae",
                component_path="vae/",
                storage_size=Memory.from_kb(0),
                n_layers=None,
                can_shard=False,
                safetensors_index_filename=None,
            ),
        ],
    ),
    "flux1-krea-dev": ModelCard(
        model_id=ModelId("black-forest-labs/FLUX.1-Krea-dev"),
        storage_size=Memory.from_bytes(23802816640 + 9524621312),  # Same as dev
        n_layers=57,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextToImage],
        components=[
            ComponentInfo(
                component_name="text_encoder",
                component_path="text_encoder/",
                storage_size=Memory.from_kb(0),
                n_layers=12,
                can_shard=False,
                safetensors_index_filename=None,
            ),
            ComponentInfo(
                component_name="text_encoder_2",
                component_path="text_encoder_2/",
                storage_size=Memory.from_bytes(9524621312),
                n_layers=24,
                can_shard=False,
                safetensors_index_filename="model.safetensors.index.json",
            ),
            ComponentInfo(
                component_name="transformer",
                component_path="transformer/",
                storage_size=Memory.from_bytes(23802816640),
                n_layers=57,
                can_shard=True,
                safetensors_index_filename="diffusion_pytorch_model.safetensors.index.json",
            ),
            ComponentInfo(
                component_name="vae",
                component_path="vae/",
                storage_size=Memory.from_kb(0),
                n_layers=None,
                can_shard=False,
                safetensors_index_filename=None,
            ),
        ],
    ),
    "qwen-image": ModelCard(
        model_id=ModelId("Qwen/Qwen-Image"),
        storage_size=Memory.from_bytes(16584333312 + 40860802176),
        n_layers=60,  # Qwen has 60 transformer blocks (all joint-style)
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextToImage],
        components=[
            ComponentInfo(
                component_name="text_encoder",
                component_path="text_encoder/",
                storage_size=Memory.from_kb(16584333312),
                n_layers=12,
                can_shard=False,
                safetensors_index_filename=None,  # Single file
            ),
            ComponentInfo(
                component_name="transformer",
                component_path="transformer/",
                storage_size=Memory.from_bytes(40860802176),
                n_layers=60,
                can_shard=True,
                safetensors_index_filename="diffusion_pytorch_model.safetensors.index.json",
            ),
            ComponentInfo(
                component_name="vae",
                component_path="vae/",
                storage_size=Memory.from_kb(0),
                n_layers=None,
                can_shard=False,
                safetensors_index_filename=None,
            ),
        ],
    ),
    "qwen-image-edit-2509": ModelCard(
        model_id=ModelId("Qwen/Qwen-Image-Edit-2509"),
        storage_size=Memory.from_bytes(16584333312 + 40860802176),
        n_layers=60,  # Qwen has 60 transformer blocks (all joint-style)
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.ImageToImage],
        components=[
            ComponentInfo(
                component_name="text_encoder",
                component_path="text_encoder/",
                storage_size=Memory.from_kb(16584333312),
                n_layers=12,
                can_shard=False,
                safetensors_index_filename=None,  # Single file
            ),
            ComponentInfo(
                component_name="transformer",
                component_path="transformer/",
                storage_size=Memory.from_bytes(40860802176),
                n_layers=60,
                can_shard=True,
                safetensors_index_filename="diffusion_pytorch_model.safetensors.index.json",
            ),
            ComponentInfo(
                component_name="vae",
                component_path="vae/",
                storage_size=Memory.from_kb(0),
                n_layers=None,
                can_shard=False,
                safetensors_index_filename=None,
            ),
        ],
    ),
}

if EXO_ENABLE_IMAGE_MODELS:
    MODEL_CARDS.update(_IMAGE_MODEL_CARDS)


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
    architectures: list[str] | None = None

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


async def get_config_data(model_id: ModelId) -> ConfigData:
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


async def get_safetensors_size(model_id: ModelId) -> Memory:
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
