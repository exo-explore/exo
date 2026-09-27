"""Select and realize one pipeline shard of a Hugging Face or GGUF checkpoint.

Key filtering and byte accounting are pure. Reading safetensors or GGUF
headers and realizing the assigned tensors are the file effects, and they run
only from ``TinygradEngine.allocate_weights`` and ``TinygradEngine.load_model``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, cast, final

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

from exo.download.download_utils import build_model_path
from exo.download.huggingface_utils import extract_layer_num
from exo.shared.types.worker.shards import PipelineShardMetadata

if TYPE_CHECKING:
    from tinygrad.tensor import Tensor

_STRICT_CONFIG = ConfigDict(
    extra="ignore", frozen=True, strict=True, validate_by_name=True
)

type SupportedModelType = Literal["llama", "qwen2", "qwen3"]
SUPPORTED_MODEL_TYPES: frozenset[str] = frozenset(("llama", "qwen2", "qwen3"))

_SAFETENSOR_ITEMSIZE: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}

_EMBEDDING_SEGMENT = "embed_tokens"
_FINAL_NORM_SUFFIX = "model.norm.weight"
_LANGUAGE_MODEL_HEAD_SEGMENT = "lm_head"


@final
class TinygradModelSupportError(Exception):
    """Raised when the checkpoint is not a supported Llama or Qwen model.

    ``exo.worker.runner.bootstrap.entrypoint`` handles this by publishing
    ``RunnerTerminationError`` and exiting the runner.
    """


@final
class TinygradWeightError(Exception):
    """Raised when a checkpoint cannot be read or a shard is loaded out of order.

    ``exo.worker.runner.bootstrap.entrypoint`` handles this by publishing
    ``RunnerTerminationError`` and exiting the runner.
    """


@final
class TinygradShardRoleError(Exception):
    """Raised when a rank calls an operation owned by a different rank.

    ``exo.worker.runner.bootstrap.entrypoint`` handles this by publishing
    ``RunnerTerminationError`` and exiting the runner.
    """


def _number_to_float(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return float(value)
    return value


type JsonFloat = Annotated[float, BeforeValidator(_number_to_float)]


@final
class RopeScalingConfig(BaseModel):
    """Hugging Face ``rope_scaling`` object.

    ``legacy_type`` accepts the older ``type`` field. ``default`` means the
    base rotary frequencies are used unchanged.
    """

    model_config = _STRICT_CONFIG

    rope_type: str | None = None
    legacy_type: str | None = Field(default=None, alias="type")
    factor: JsonFloat = 1.0
    low_freq_factor: JsonFloat | None = None
    high_freq_factor: JsonFloat | None = None
    original_max_position_embeddings: int | None = None

    def scaling_kind(self) -> Literal["none", "llama3", "linear"]:
        """Return the rotary scaling this block knows how to apply.

        Raises:
            TinygradModelSupportError: The runner entrypoint handles this by
                publishing ``RunnerTerminationError`` when the checkpoint asks
                for a rotary variant this adapter does not implement.
        """
        raw_kind = self.rope_type if self.rope_type is not None else self.legacy_type
        if raw_kind is None or raw_kind == "default":
            return "none"
        if raw_kind == "llama3" or raw_kind == "linear":
            return raw_kind
        raise TinygradModelSupportError(
            f"Unsupported rope_scaling type {raw_kind!r}. Expected llama3 or linear."
        )


@final
class TransformerArchitecture(BaseModel):
    """Fields read from ``config.json`` for Llama, Qwen2, and Qwen3."""

    model_config = _STRICT_CONFIG

    model_type: SupportedModelType
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int | None = None
    intermediate_size: int
    rms_norm_eps: JsonFloat
    rope_theta: JsonFloat = 10000.0
    vocab_size: int
    num_hidden_layers: int
    head_dim: int | None = None
    rope_scaling: RopeScalingConfig | None = None
    tie_word_embeddings: bool = False
    attention_bias: bool = False

    def resolved_head_dimension(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        return self.hidden_size // self.num_attention_heads

    def resolved_key_value_heads(self) -> int:
        if self.num_key_value_heads is not None:
            return self.num_key_value_heads
        return self.num_attention_heads


@final
class SafetensorTensorRecord(BaseModel):
    """One tensor entry inside a safetensors header."""

    model_config = _STRICT_CONFIG

    dtype: str
    shape: list[int]
    data_offsets: list[int]


@final
@dataclass(frozen=True)
class SafetensorEntry:
    """Where one named tensor lives, plus its serialized size in bytes."""

    relative_file: str
    byte_count: int


def _path_contains_segment(tensor_name: str, segment: str) -> bool:
    return segment in tensor_name.split(".")


def select_shard_tensor_names(
    tensor_names: IterableNames,
    shard: PipelineShardMetadata,
    *,
    embeddings_are_tied: bool,
) -> frozenset[str]:
    """Return the checkpoint names this pipeline rank must realize.

    A name is kept when its layer index is inside ``[start_layer, end_layer)``,
    when it is ``embed_tokens`` on the first rank, when it is ``model.norm`` or
    ``lm_head`` on the last rank, or when tied embeddings mean the last rank
    must load ``embed_tokens`` because ``lm_head`` is absent from the whole
    checkpoint. ``tensor_names`` must be the global name set, not one file.
    """
    names = frozenset(tensor_names)
    language_model_head_present = any(
        _path_contains_segment(name, _LANGUAGE_MODEL_HEAD_SEGMENT) for name in names
    )
    load_tied_embeddings_on_last_rank = (
        embeddings_are_tied and shard.is_last_layer and not language_model_head_present
    )
    selected: set[str] = set()
    for name in names:
        if _path_contains_segment(name, _EMBEDDING_SEGMENT):
            if shard.is_first_layer or load_tied_embeddings_on_last_rank:
                selected.add(name)
            continue
        if name.endswith(_FINAL_NORM_SUFFIX) or _path_contains_segment(
            name, _LANGUAGE_MODEL_HEAD_SEGMENT
        ):
            if shard.is_last_layer:
                selected.add(name)
            continue
        layer_index = extract_layer_num(name)
        if (
            layer_index is not None
            and shard.start_layer <= layer_index < shard.end_layer
        ):
            selected.add(name)
    return frozenset(selected)


type IterableNames = list[str] | tuple[str, ...] | set[str] | frozenset[str]


def record_byte_count(record: SafetensorTensorRecord) -> int:
    """Return the serialized size of one safetensors tensor."""
    if len(record.data_offsets) != 2:
        raise TinygradWeightError("safetensors data_offsets must contain two integers")
    itemsize = _SAFETENSOR_ITEMSIZE.get(record.dtype)
    if itemsize is None:
        return record.data_offsets[1] - record.data_offsets[0]
    element_count = 1
    for dimension in record.shape:
        element_count *= dimension
    return element_count * itemsize


def selected_parameter_byte_count(
    entries: Mapping[str, SafetensorEntry],
    shard: PipelineShardMetadata,
    *,
    embeddings_are_tied: bool,
) -> int:
    """Sum the serialized bytes of the names this shard will realize."""
    selected = select_shard_tensor_names(
        tuple(entries),
        shard,
        embeddings_are_tied=embeddings_are_tied,
    )
    return sum(entries[name].byte_count for name in selected)


def decode_json_value(raw: bytes) -> object:
    """Parse JSON without leaking ``Any`` into callers.

    ``json.loads`` is typed as ``Any``. Callers narrow the object.
    """
    return cast(object, json.loads(raw.decode("utf-8")))


def object_mapping(value: object, label: str) -> dict[str, object]:
    """Return a string-keyed object, or raise ``TinygradWeightError``.

    The runner entrypoint handles ``TinygradWeightError`` by publishing
    ``RunnerTerminationError``.
    """
    if not isinstance(value, dict):
        raise TinygradWeightError(f"{label} must be a JSON object")
    raw_mapping = cast(dict[object, object], value)
    mapping: dict[str, object] = {}
    for key, item in raw_mapping.items():
        if not isinstance(key, str):
            raise TinygradWeightError(f"{label} contains a non-string key")
        mapping[key] = item
    return mapping


def model_directory_for_shard(shard: PipelineShardMetadata) -> Path:
    """Return the on-disk checkpoint directory for ``shard``.

    Raises:
        TinygradWeightError: The runner entrypoint handles this by publishing
            ``RunnerTerminationError`` when the download directory is missing.
    """
    directory = build_model_path(shard.model_card.model_id)
    if not directory.is_dir():
        raise TinygradWeightError(
            f"Model directory for {shard.model_card.model_id} is not available at {directory}"
        )
    return directory


def load_architecture(model_directory: Path) -> TransformerArchitecture:
    """Read ``config.json`` and reject unsupported model types.

    Raises:
        TinygradModelSupportError: The runner entrypoint handles this when
            ``model_type`` is not ``llama``, ``qwen2``, or ``qwen3``, or when
            ``rope_scaling`` names an unimplemented variant.
        TinygradWeightError: The runner entrypoint handles this when the file
            is missing or does not match the supported config shape.
    """
    config_path = model_directory / "config.json"
    if not config_path.is_file():
        raise TinygradWeightError(f"{config_path} is missing")
    payload = object_mapping(decode_json_value(config_path.read_bytes()), "config.json")
    model_type = payload.get("model_type")
    if not isinstance(model_type, str) or model_type not in SUPPORTED_MODEL_TYPES:
        raise TinygradModelSupportError(
            f"Unsupported model_type {model_type!r}. Expected llama, qwen2, or qwen3."
        )
    try:
        architecture = TransformerArchitecture.model_validate(payload)
    except ValidationError as error:
        raise TinygradWeightError(
            f"config.json is not a supported transformer config: {error}"
        ) from error
    if architecture.rope_scaling is not None:
        _ = architecture.rope_scaling.scaling_kind()
    return architecture


def read_safetensors_header(path: Path) -> dict[str, SafetensorTensorRecord]:
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise TinygradWeightError(f"{path} is not a safetensors file")
        header_length = int.from_bytes(length_bytes, "little")
        header_bytes = handle.read(header_length)
    if len(header_bytes) != header_length:
        raise TinygradWeightError(f"{path} has a truncated safetensors header")
    payload = object_mapping(decode_json_value(header_bytes), path.name)
    records: dict[str, SafetensorTensorRecord] = {}
    for name, value in payload.items():
        if name == "__metadata__":
            continue
        try:
            records[name] = SafetensorTensorRecord.model_validate(value)
        except ValidationError as error:
            raise TinygradWeightError(
                f"{path} tensor {name} has an invalid safetensors header"
            ) from error
    return records


def _read_weight_map(index_path: Path) -> dict[str, str]:
    payload = object_mapping(
        decode_json_value(index_path.read_bytes()), index_path.name
    )
    weight_map_value = payload.get("weight_map")
    weight_map = object_mapping(weight_map_value, "weight_map")
    filenames: dict[str, str] = {}
    for name, filename in weight_map.items():
        if not isinstance(filename, str):
            raise TinygradWeightError(f"weight_map entry {name} is not a file name")
        filenames[name] = filename
    return filenames


def checkpoint_tensor_entries(model_directory: Path) -> dict[str, SafetensorEntry]:
    """Index tensor names from headers without copying parameter bytes.

    Prefers ``model.safetensors.index.json``. Otherwise uses
    ``model.safetensors``, then every ``*.safetensors`` file in the directory.

    Raises:
        TinygradWeightError: The runner entrypoint handles this when no
            checkpoint file is present or a header is malformed.
    """
    index_path = model_directory / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = _read_weight_map(index_path)
        headers: dict[str, dict[str, SafetensorTensorRecord]] = {}
        entries: dict[str, SafetensorEntry] = {}
        for tensor_name, relative_file in weight_map.items():
            header = headers.get(relative_file)
            if header is None:
                tensor_path = model_directory / relative_file
                if not tensor_path.is_file():
                    raise TinygradWeightError(
                        f"Indexed checkpoint file {tensor_path} is missing"
                    )
                header = read_safetensors_header(tensor_path)
                headers[relative_file] = header
            record = header.get(tensor_name)
            if record is None:
                raise TinygradWeightError(
                    f"{relative_file} is missing indexed tensor {tensor_name}"
                )
            entries[tensor_name] = SafetensorEntry(
                relative_file=relative_file,
                byte_count=record_byte_count(record),
            )
        return entries

    single_file = model_directory / "model.safetensors"
    if single_file.is_file():
        tensor_paths = [single_file]
    else:
        tensor_paths = sorted(model_directory.glob("*.safetensors"))
    if not tensor_paths:
        raise TinygradWeightError(
            f"{model_directory} does not contain safetensors weights"
        )
    collected: dict[str, SafetensorEntry] = {}
    for tensor_path in tensor_paths:
        for tensor_name, record in read_safetensors_header(tensor_path).items():
            collected[tensor_name] = SafetensorEntry(
                relative_file=tensor_path.name,
                byte_count=record_byte_count(record),
            )
    return collected


def shard_parameter_byte_count(shard: PipelineShardMetadata) -> int:
    """Sum the bytes this shard will place on the device, from headers only.

    Safetensors contribute their serialized size. GGUF contributes the float16
    size of the selected tensors, which is what dequantization realizes.
    ``TinygradEngine.allocate_weights`` is the caller. ``TinygradWeightError``
    and ``TinygradModelSupportError`` are handled by the runner entrypoint as
    a runner termination.
    """
    from exo.backends.tinygrad_checkpoint import selected_checkpoint_byte_count

    return selected_checkpoint_byte_count(shard)


@final
@dataclass(frozen=True)
class RealizedParameterGroup:
    """Parameters realized together.

    ``layer_index`` is ``None`` for embeddings, the final norm, and ``lm_head``.
    """

    layer_index: int | None
    parameters: tuple[tuple[str, Tensor], ...]


def iter_realized_parameter_groups(
    shard: PipelineShardMetadata,
) -> Iterator[RealizedParameterGroup]:
    """Lazily load the checkpoint and realize one decoder layer at a time.

    Unselected keys are dropped before ``realize``, so device memory holds the
    shard rather than the whole file. GGUF payloads are read only for the
    assigned names. ``Device.DEFAULT`` is already selected by the engine.

    Yields:
        The non-layer parameters first, then one group per decoder layer.

    Raises:
        TinygradWeightError: The runner entrypoint handles a missing tensor
            or a checkpoint that cannot be read.
        TinygradModelSupportError: The runner entrypoint handles an
            unsupported architecture.
    """
    from exo.backends.tinygrad_checkpoint import iter_checkpoint_parameter_groups

    yield from iter_checkpoint_parameter_groups(shard)
