"""Translate checkpoint names and realize only the assigned layer interval.

Safetensors and GGUF headers are read before any weight bytes are copied.
Unassigned tensors never become ``Tensor`` values. GGUF dequantization uses
``tinygrad.llm.gguf.ggml_data_to_tensor`` on the selected byte range only.
``gguf_load`` is not used, because it copies the whole file onto the device.
Tokenizer token, merge, and token-type arrays are kept from the header so a
full shard can encode prompts without a separate tokenizer package.
"""

from __future__ import annotations

import ctypes
import gc
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, final

from exo.backends.tinygrad_weights import (
    RealizedParameterGroup,
    SafetensorEntry,
    SafetensorTensorRecord,
    SupportedModelType,
    TinygradModelSupportError,
    TinygradWeightError,
    TransformerArchitecture,
    checkpoint_tensor_entries,
    load_architecture,
    model_directory_for_shard,
    read_safetensors_header,
    select_shard_tensor_names,
    selected_parameter_byte_count,
)
from exo.download.huggingface_utils import extract_layer_num
from exo.shared.types.worker.shards import PipelineShardMetadata

if TYPE_CHECKING:
    from tinygrad.dtype import DType
    from tinygrad.tensor import Tensor

_GGUF_MAGIC = b"GGUF"
_GGUF_DEFAULT_ALIGNMENT = 32
_MAX_GGUF_STRING_BYTES = 16 * 1024 * 1024
_MAX_GGUF_ARRAY_COUNT = 5_000_000
_FLOAT16_BYTES = 2

_GGUF_LAYER = re.compile(r"^blk\.(\d+)\.([A-Za-z0-9_]+)\.(weight|bias)$")

_ROOT_NAMES: dict[str, str] = {
    "model.embed_tokens.weight": "model.embed_tokens.weight",
    "embed_tokens.weight": "model.embed_tokens.weight",
    "tok_embeddings.weight": "model.embed_tokens.weight",
    "token_embd.weight": "model.embed_tokens.weight",
    "model.norm.weight": "model.norm.weight",
    "norm.weight": "model.norm.weight",
    "output_norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
    "output.weight": "lm_head.weight",
}

_GGUF_LAYER_SUFFIX: dict[str, str] = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_output.bias": "self_attn.o_proj.bias",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

_MLX_ATTENTION: dict[str, str] = {
    "wq": "self_attn.q_proj.weight",
    "wk": "self_attn.k_proj.weight",
    "wv": "self_attn.v_proj.weight",
    "wo": "self_attn.o_proj.weight",
}
_MLX_FEED_FORWARD: dict[str, str] = {
    "w1": "mlp.gate_proj.weight",
    "w3": "mlp.up_proj.weight",
    "w2": "mlp.down_proj.weight",
}

_HF_SUFFIXES: frozenset[str] = frozenset(
    {
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.bias",
        "self_attn.o_proj.bias",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }
)

_GGML_NATIVE_ITEMSIZE: dict[int, int] = {
    0: 4,
    1: 2,
    24: 1,
    25: 2,
    26: 4,
    27: 8,
    28: 8,
    30: 2,
}
_GGML_QUANT_BLOCK: dict[int, tuple[int, int]] = {
    2: (32, 18),
    3: (32, 20),
    6: (32, 22),
    7: (32, 24),
    8: (32, 34),
    12: (256, 144),
    13: (256, 176),
    14: (256, 210),
    18: (256, 98),
    21: (256, 110),
    22: (256, 82),
    23: (256, 136),
    39: (32, 17),
    41: (128, 18),
}
_GGML_TYPE_NAMES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    18: "IQ3_XXS",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    30: "BF16",
    39: "MXFP4",
    41: "Q1_0",
}
_GGUF_FILE_TYPE_NAMES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
}

type GgufScalar = int | float | str | bool


@final
class _GgufCursor:
    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle

    def read(self, count: int) -> bytes:
        chunk = self._handle.read(count)
        if len(chunk) != count:
            raise TinygradWeightError("GGUF header is truncated")
        return chunk

    def tell(self) -> int:
        return self._handle.tell()


@final
@dataclass(frozen=True)
class GgufTensorRecord:
    """One GGUF tensor located from the header, before its payload is read."""

    path: Path
    source_name: str
    canonical_name: str | None
    ggml_type: int
    dimensions: tuple[int, ...]
    absolute_start: int
    element_count: int

    def ggml_type_name(self) -> str:
        return _GGML_TYPE_NAMES.get(self.ggml_type, f"ggml-type-{self.ggml_type}")

    def payload_byte_count(self) -> int:
        """Return the on-disk size of this tensor.

        Raises:
            TinygradWeightError: The runner entrypoint handles this when a
                selected tensor uses a GGML type this loader cannot size.
        """
        return ggml_payload_byte_count(self.element_count, self.ggml_type)


_TOKENIZER_STRING_ARRAYS = frozenset({"tokenizer.ggml.tokens", "tokenizer.ggml.merges"})
_TOKENIZER_INTEGER_ARRAY = "tokenizer.ggml.token_type"


@final
@dataclass(frozen=True)
class _RetainedGgufArray:
    key: str
    strings: tuple[str, ...] | None
    integers: tuple[int, ...] | None
    item_count: int


@final
@dataclass(frozen=True)
class GgufCheckpoint:
    """Header fields from one GGUF model, including every split's tensor list."""

    file_type: int | None
    file_type_name: str | None
    architecture_name: str
    scalars: tuple[tuple[str, GgufScalar], ...]
    tokenizer_token_count: int | None
    tokenizer_tokens: tuple[str, ...]
    tokenizer_merges: tuple[str, ...]
    tokenizer_token_types: tuple[int, ...]
    tensors: tuple[GgufTensorRecord, ...]

    def scalar(self, key: str) -> GgufScalar | None:
        for name, value in self.scalars:
            if name == key:
                return value
        return None


def canonical_tensor_name(source_name: str) -> str | None:
    """Map a Hugging Face, MLX, or GGUF name onto the loader's tensor name.

    Unrecognized names return ``None`` and are ignored. Hugging Face names
    that already match the loader are returned unchanged.
    """
    root_name = _ROOT_NAMES.get(source_name)
    if root_name is not None:
        return root_name
    gguf_match = _GGUF_LAYER.fullmatch(source_name)
    if gguf_match is not None:
        suffix = _GGUF_LAYER_SUFFIX.get(f"{gguf_match.group(2)}.{gguf_match.group(3)}")
        if suffix is None:
            return None
        return f"model.layers.{gguf_match.group(1)}.{suffix}"
    return _canonical_mlx_or_hugging_face_name(source_name)


def ggml_payload_byte_count(element_count: int, ggml_type: int) -> int:
    """Return the serialized size of one GGML tensor.

    Raises:
        TinygradWeightError: The runner entrypoint handles this when the type
            is not one ``ggml_data_to_tensor`` can convert, or the element
            count is not a whole number of quant blocks.
    """
    itemsize = _GGML_NATIVE_ITEMSIZE.get(ggml_type)
    if itemsize is not None:
        return element_count * itemsize
    block = _GGML_QUANT_BLOCK.get(ggml_type)
    if block is None:
        type_name = _GGML_TYPE_NAMES.get(ggml_type, str(ggml_type))
        raise TinygradWeightError(f"Unsupported GGML type {type_name}")
    elements_per_block, bytes_per_block = block
    if element_count % elements_per_block != 0:
        raise TinygradWeightError(
            f"GGML type {_GGML_TYPE_NAMES.get(ggml_type, str(ggml_type))} "
            f"requires a multiple of {elements_per_block} elements"
        )
    return (element_count // elements_per_block) * bytes_per_block


def directory_has_safetensors(directory: Path) -> bool:
    if (directory / "model.safetensors.index.json").is_file():
        return True
    if (directory / "model.safetensors").is_file():
        return True
    return any(directory.glob("*.safetensors"))


def directory_has_gguf(directory: Path) -> bool:
    return any(directory.glob("*.gguf"))


def load_checkpoint_architecture(directory: Path) -> TransformerArchitecture:
    """Read ``config.json`` when it exists, otherwise the GGUF metadata.

    Raises:
        TinygradModelSupportError: The runner entrypoint handles an
            architecture other than llama, qwen2, or qwen3.
        TinygradWeightError: The runner entrypoint handles a missing
            checkpoint or incomplete metadata.
    """
    if directory_has_safetensors(directory) or (directory / "config.json").is_file():
        return load_architecture(directory)
    if directory_has_gguf(directory):
        return architecture_from_gguf(read_gguf_checkpoint(directory))
    raise TinygradWeightError(
        f"{directory} does not contain safetensors or GGUF weights"
    )


def read_gguf_checkpoint(directory: Path) -> GgufCheckpoint:
    """Read every GGUF header in ``directory`` without copying weight bytes.

    Later splits contribute tensor locations. Metadata comes from the first
    split. A split's weight payload is not opened here.

    Raises:
        TinygradWeightError: The runner entrypoint handles a missing file, a
            header that is not GGUF version 2 or 3, or a file without
            ``general.architecture``.
    """
    paths = _gguf_paths(directory)
    headers = tuple(_read_gguf_file(path) for path in paths)
    primary = headers[0]
    architecture_name = primary.architecture_name
    if architecture_name is None:
        raise TinygradWeightError(f"{primary.path} is missing general.architecture")
    tensors: list[GgufTensorRecord] = []
    for header in headers:
        tensors.extend(header.tensors)
    tokenizer_tokens, tokenizer_merges, tokenizer_token_types = _tokenizer_arrays(
        primary.retained_arrays
    )
    return GgufCheckpoint(
        file_type=primary.file_type,
        file_type_name=_GGUF_FILE_TYPE_NAMES.get(primary.file_type)
        if primary.file_type is not None
        else None,
        architecture_name=architecture_name,
        scalars=primary.scalars,
        tokenizer_token_count=primary.tokenizer_token_count,
        tokenizer_tokens=tokenizer_tokens,
        tokenizer_merges=tokenizer_merges,
        tokenizer_token_types=tokenizer_token_types,
        tensors=tuple(tensors),
    )


def architecture_from_gguf(checkpoint: GgufCheckpoint) -> TransformerArchitecture:
    """Build the loader architecture from GGUF key-value metadata.

    Raises:
        TinygradModelSupportError: The runner entrypoint handles an
            architecture other than llama, qwen2, or qwen3.
        TinygradWeightError: The runner entrypoint handles a missing field.
    """
    model_type = _supported_architecture(checkpoint.architecture_name)
    prefix = checkpoint.architecture_name
    hidden_size = _required_int(checkpoint, f"{prefix}.embedding_length")
    num_attention_heads = _required_int(checkpoint, f"{prefix}.attention.head_count")
    num_key_value_heads = _required_int(checkpoint, f"{prefix}.attention.head_count_kv")
    intermediate_size = _required_int(checkpoint, f"{prefix}.feed_forward_length")
    num_hidden_layers = _required_int(checkpoint, f"{prefix}.block_count")
    rms_norm_eps = _required_float(
        checkpoint, f"{prefix}.attention.layer_norm_rms_epsilon"
    )
    rope_theta = _required_float(checkpoint, f"{prefix}.rope.freq_base")
    head_dimension = _optional_int(checkpoint, f"{prefix}.attention.key_length")
    if head_dimension is None:
        if num_attention_heads == 0 or hidden_size % num_attention_heads != 0:
            raise TinygradWeightError(
                "GGUF metadata cannot resolve attention head dimension"
            )
        head_dimension = hidden_size // num_attention_heads
    vocab_size = _optional_int(checkpoint, f"{prefix}.vocab_size")
    if vocab_size is None:
        vocab_size = checkpoint.tokenizer_token_count
    if vocab_size is None:
        raise TinygradWeightError("GGUF metadata is missing vocab size")
    source_names = {record.source_name for record in checkpoint.tensors}
    return TransformerArchitecture(
        model_type=model_type,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        vocab_size=vocab_size,
        num_hidden_layers=num_hidden_layers,
        head_dim=head_dimension,
        tie_word_embeddings="output.weight" not in source_names,
        attention_bias=any(name.endswith("attn_q.bias") for name in source_names),
    )


def assigned_gguf_tensors(
    checkpoint: GgufCheckpoint,
    shard: PipelineShardMetadata,
    *,
    embeddings_are_tied: bool,
) -> tuple[GgufTensorRecord, ...]:
    """Return the GGUF tensors whose canonical names belong to ``shard``.

    Raises:
        TinygradWeightError: The runner entrypoint handles two source names
            that translate to the same tensor.
    """
    by_canonical: dict[str, GgufTensorRecord] = {}
    for record in checkpoint.tensors:
        canonical_name = record.canonical_name
        if canonical_name is None:
            continue
        if canonical_name in by_canonical:
            raise TinygradWeightError(
                f"GGUF tensor {canonical_name} is stored more than once"
            )
        by_canonical[canonical_name] = record
    selected = select_shard_tensor_names(
        tuple(by_canonical),
        shard,
        embeddings_are_tied=embeddings_are_tied,
    )
    return tuple(by_canonical[name] for name in sorted(selected))


def selected_checkpoint_byte_count(shard: PipelineShardMetadata) -> int:
    """Return device bytes for the assigned tensors, from headers only.

    Raises:
        TinygradWeightError: The runner entrypoint handles a missing
            checkpoint.
        TinygradModelSupportError: The runner entrypoint handles an
            unsupported architecture.
    """
    directory = model_directory_for_shard(shard)
    architecture = load_checkpoint_architecture(directory)
    if directory_has_safetensors(directory):
        entries = _canonical_safetensor_entries(checkpoint_tensor_entries(directory))
        return selected_parameter_byte_count(
            entries,
            shard,
            embeddings_are_tied=architecture.tie_word_embeddings,
        )
    checkpoint = read_gguf_checkpoint(directory)
    assigned = assigned_gguf_tensors(
        checkpoint,
        shard,
        embeddings_are_tied=architecture.tie_word_embeddings,
    )
    return sum(record.element_count * _FLOAT16_BYTES for record in assigned)


def iter_checkpoint_parameter_groups(
    shard: PipelineShardMetadata,
) -> Iterator[RealizedParameterGroup]:
    """Realize the assigned safetensors or GGUF tensors one layer at a time.

    Raises:
        TinygradWeightError: The runner entrypoint handles a missing tensor,
            a bad shape, or a checkpoint that cannot be read.
        TinygradModelSupportError: The runner entrypoint handles an
            unsupported architecture.
    """
    directory = model_directory_for_shard(shard)
    architecture = load_checkpoint_architecture(directory)
    if directory_has_safetensors(directory):
        lazy = _lazy_safetensor_views(directory, shard, architecture)
        yield from _yield_realized_groups(lazy, shard, _realize_on_default_device)
        return
    if directory_has_gguf(directory):
        lazy = _realized_gguf_tensors(directory, shard, architecture)
        yield from _yield_realized_groups(lazy, shard, lambda tensor: tensor)
        return
    raise TinygradWeightError(
        f"{directory} does not contain safetensors or GGUF weights"
    )


def assert_assigned_tensor_shapes(
    parameters: Mapping[str, Tensor],
    architecture: TransformerArchitecture,
    shard: PipelineShardMetadata,
) -> None:
    """Require every assigned tensor to exist with the architecture's shape.

    Raises:
        TinygradWeightError: The runner entrypoint handles a missing key or
            a shape that does not match the architecture, so the runner does
            not report the shard as ready.
    """
    expected = expected_tensor_shapes(architecture, shard)
    for name, shape in expected.items():
        tensor = parameters.get(name)
        if tensor is None:
            raise TinygradWeightError(f"Assigned checkpoint is missing {name}")
        actual = _tensor_shape(tensor)
        if actual != shape:
            raise TinygradWeightError(
                f"Tensor {name} has shape {actual}, expected {shape}"
            )


def expected_tensor_shapes(
    architecture: TransformerArchitecture,
    shard: PipelineShardMetadata,
) -> dict[str, tuple[int, ...]]:
    """Return the canonical name and shape required for this shard."""
    hidden = architecture.hidden_size
    query_size = (
        architecture.num_attention_heads * architecture.resolved_head_dimension()
    )
    key_size = (
        architecture.resolved_key_value_heads() * architecture.resolved_head_dimension()
    )
    intermediate = architecture.intermediate_size
    head_dimension = architecture.resolved_head_dimension()
    required: dict[str, tuple[int, ...]] = {}
    for layer_index in range(shard.start_layer, shard.end_layer):
        prefix = f"model.layers.{layer_index}"
        required[f"{prefix}.input_layernorm.weight"] = (hidden,)
        required[f"{prefix}.self_attn.q_proj.weight"] = (query_size, hidden)
        required[f"{prefix}.self_attn.k_proj.weight"] = (key_size, hidden)
        required[f"{prefix}.self_attn.v_proj.weight"] = (key_size, hidden)
        required[f"{prefix}.self_attn.o_proj.weight"] = (hidden, query_size)
        required[f"{prefix}.post_attention_layernorm.weight"] = (hidden,)
        required[f"{prefix}.mlp.gate_proj.weight"] = (intermediate, hidden)
        required[f"{prefix}.mlp.up_proj.weight"] = (intermediate, hidden)
        required[f"{prefix}.mlp.down_proj.weight"] = (hidden, intermediate)
        if architecture.model_type == "qwen3":
            required[f"{prefix}.self_attn.q_norm.weight"] = (head_dimension,)
            required[f"{prefix}.self_attn.k_norm.weight"] = (head_dimension,)
        if architecture.attention_bias:
            required[f"{prefix}.self_attn.q_proj.bias"] = (query_size,)
            required[f"{prefix}.self_attn.k_proj.bias"] = (key_size,)
            required[f"{prefix}.self_attn.v_proj.bias"] = (key_size,)
    load_embeddings = shard.is_first_layer or (
        architecture.tie_word_embeddings and shard.is_last_layer
    )
    if load_embeddings:
        required["model.embed_tokens.weight"] = (architecture.vocab_size, hidden)
    if shard.is_last_layer:
        required["model.norm.weight"] = (hidden,)
        if not architecture.tie_word_embeddings:
            required["lm_head.weight"] = (architecture.vocab_size, hidden)
    return required


def permute_gguf_rotary_weight(weight: Tensor, head_count: int) -> Tensor:
    """Reorder llama.cpp interleaved Q or K rows into half-split rotary order.

    GGUF stores consecutive pairs. ``_rotate_half`` consumes the first half
    of the head, then the second half.
    """
    if head_count <= 0:
        raise TinygradWeightError("GGUF rotary permutation requires at least one head")
    if len(weight.shape) == 1:
        length = int(weight.shape[0])
        if length % head_count != 0:
            raise TinygradWeightError(
                f"GGUF rotary bias length {length} is not divisible by {head_count} heads"
            )
        head_dimension = length // head_count
        if head_dimension % 2 != 0:
            raise TinygradWeightError(
                f"GGUF rotary head dimension {head_dimension} must be even"
            )
        grouped = weight.reshape(head_count, head_dimension // 2, 2)
        return grouped.permute(0, 2, 1).reshape(length)
    rows = int(weight.shape[0])
    columns = int(weight.shape[1])
    if rows % head_count != 0:
        raise TinygradWeightError(
            f"GGUF rotary rows {rows} are not divisible by {head_count} heads"
        )
    head_dimension = rows // head_count
    if head_dimension % 2 != 0:
        raise TinygradWeightError(
            f"GGUF rotary head dimension {head_dimension} must be even"
        )
    grouped = weight.reshape(head_count, head_dimension // 2, 2, columns)
    return grouped.permute(0, 2, 1, 3).reshape(rows, columns)


def ggml_tensor_from_file(
    path: Path,
    start: int,
    length: int,
    element_count: int,
    ggml_type: int,
) -> Tensor:
    """Dequantize one GGUF byte range.

    The disk tensor is sliced to ``length`` bytes before conversion, so the
    rest of the file is not realized.

    Raises:
        TinygradWeightError: The runner entrypoint handles a GGML type that
            ``ggml_data_to_tensor`` rejects.
    """
    from tinygrad import Tensor
    from tinygrad.llm.gguf import ggml_data_to_tensor

    window = Tensor(path)[start : start + length]
    try:
        return ggml_data_to_tensor(window, element_count, ggml_type)
    except ValueError as error:
        type_name = _GGML_TYPE_NAMES.get(ggml_type, str(ggml_type))
        raise TinygradWeightError(
            f"GGML type {type_name} in {path.name} is not supported"
        ) from error


def _canonical_mlx_or_hugging_face_name(source_name: str) -> str | None:
    parts = source_name.split(".")
    if parts and parts[0] == "model":
        parts = parts[1:]
    if len(parts) < 3 or parts[0] != "layers" or not parts[1].isdigit():
        return None
    layer = parts[1]
    rest = parts[2:]
    suffix: str | None = None
    if len(rest) == 3 and rest[0] == "attention" and rest[2] == "weight":
        suffix = _MLX_ATTENTION.get(rest[1])
    elif len(rest) == 3 and rest[0] == "feed_forward" and rest[2] == "weight":
        suffix = _MLX_FEED_FORWARD.get(rest[1])
    elif rest == ["attention_norm", "weight"]:
        suffix = "input_layernorm.weight"
    elif rest == ["ffn_norm", "weight"]:
        suffix = "post_attention_layernorm.weight"
    else:
        joined = ".".join(rest)
        if joined in _HF_SUFFIXES:
            suffix = joined
    if suffix is None:
        return None
    return f"model.layers.{layer}.{suffix}"


def _supported_architecture(name: str) -> SupportedModelType:
    if name == "llama" or name == "qwen2" or name == "qwen3":
        return name
    raise TinygradModelSupportError(
        f"Unsupported GGUF architecture {name!r}. Expected llama, qwen2, or qwen3."
    )


def _required_int(checkpoint: GgufCheckpoint, key: str) -> int:
    value = checkpoint.scalar(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TinygradWeightError(f"GGUF metadata is missing {key}")
    return value


def _optional_int(checkpoint: GgufCheckpoint, key: str) -> int | None:
    value = checkpoint.scalar(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TinygradWeightError(f"GGUF metadata {key} is not an integer")
    return value


def _required_float(checkpoint: GgufCheckpoint, key: str) -> float:
    value = checkpoint.scalar(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TinygradWeightError(f"GGUF metadata is missing {key}")
    return float(value)


def _gguf_paths(directory: Path) -> tuple[Path, ...]:
    paths = list(directory.glob("*.gguf"))
    if not paths:
        raise TinygradWeightError(f"{directory} does not contain GGUF weights")

    def sort_key(path: Path) -> tuple[int, str]:
        match = re.search(r"-(\d+)-of-\d+\.gguf$", path.name)
        if match is None:
            return (0, path.name)
        return (int(match.group(1)), path.name)

    return tuple(sorted(paths, key=sort_key))


def _round_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise TinygradWeightError("GGUF alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


@final
@dataclass(frozen=True)
class _GgufFile:
    path: Path
    file_type: int | None
    architecture_name: str | None
    scalars: tuple[tuple[str, GgufScalar], ...]
    tokenizer_token_count: int | None
    retained_arrays: tuple[_RetainedGgufArray, ...]
    tensors: tuple[GgufTensorRecord, ...]


def _read_gguf_file(path: Path) -> _GgufFile:
    with path.open("rb") as handle:
        cursor = _GgufCursor(handle)
        magic = cursor.read(4)
        if magic != _GGUF_MAGIC:
            raise TinygradWeightError(f"{path} is not a GGUF file")
        version = _read_int32(cursor)
        if version not in (2, 3):
            raise TinygradWeightError(f"{path} has unsupported GGUF version {version}")
        tensor_count = _read_int64(cursor)
        metadata_count = _read_int64(cursor)
        if tensor_count < 0 or metadata_count < 0:
            raise TinygradWeightError(f"{path} has a negative GGUF count")
        scalars: list[tuple[str, GgufScalar]] = []
        retained_arrays: list[_RetainedGgufArray] = []
        tokenizer_token_count: int | None = None
        alignment = _GGUF_DEFAULT_ALIGNMENT
        for _ in range(metadata_count):
            key = _read_string(cursor)
            value_type = _read_int32(cursor)
            if value_type == 9:
                retained = _read_metadata_array(cursor, key)
                if retained is None:
                    continue
                retained_arrays.append(retained)
                if key == "tokenizer.ggml.tokens":
                    tokenizer_token_count = retained.item_count
                continue
            value = _read_scalar(cursor, value_type)
            scalars.append((key, value))
            if key == "general.alignment" and isinstance(value, int) and value > 0:
                alignment = value
        tensors: list[GgufTensorRecord] = []
        for _ in range(tensor_count):
            tensors.append(_read_tensor_info(cursor, path))
        data_start = _round_up(cursor.tell(), alignment)
    located = tuple(
        GgufTensorRecord(
            path=record.path,
            source_name=record.source_name,
            canonical_name=record.canonical_name,
            ggml_type=record.ggml_type,
            dimensions=record.dimensions,
            absolute_start=data_start + record.absolute_start,
            element_count=record.element_count,
        )
        for record in tensors
    )
    architecture_name: str | None = None
    file_type: int | None = None
    for key, value in scalars:
        if key == "general.architecture" and isinstance(value, str):
            architecture_name = value
        if key == "general.file_type" and isinstance(value, int):
            file_type = value
    return _GgufFile(
        path=path,
        file_type=file_type,
        architecture_name=architecture_name,
        scalars=tuple(scalars),
        tokenizer_token_count=tokenizer_token_count,
        retained_arrays=tuple(retained_arrays),
        tensors=located,
    )


def _read_tensor_info(cursor: _GgufCursor, path: Path) -> GgufTensorRecord:
    source_name = _read_string(cursor)
    dimension_count = _read_uint32(cursor)
    dimensions = tuple(_read_uint64(cursor) for _ in range(dimension_count))
    ggml_type = _read_uint32(cursor)
    offset = _read_uint64(cursor)
    return GgufTensorRecord(
        path=path,
        source_name=source_name,
        canonical_name=canonical_tensor_name(source_name),
        ggml_type=ggml_type,
        dimensions=dimensions,
        absolute_start=offset,
        element_count=_element_count(dimensions),
    )


def _element_count(dimensions: tuple[int, ...]) -> int:
    count = 1
    for dimension in dimensions:
        count *= dimension
    return count


def _read_uint32(cursor: _GgufCursor) -> int:
    return int.from_bytes(cursor.read(4), "little", signed=False)


def _read_int32(cursor: _GgufCursor) -> int:
    return int.from_bytes(cursor.read(4), "little", signed=True)


def _read_uint64(cursor: _GgufCursor) -> int:
    return int.from_bytes(cursor.read(8), "little", signed=False)


def _read_int64(cursor: _GgufCursor) -> int:
    return int.from_bytes(cursor.read(8), "little", signed=True)


def _read_float32(cursor: _GgufCursor) -> float:
    return float(ctypes.c_float.from_buffer_copy(cursor.read(4)).value)


def _read_float64(cursor: _GgufCursor) -> float:
    return float(ctypes.c_double.from_buffer_copy(cursor.read(8)).value)


def _read_string(cursor: _GgufCursor) -> str:
    length = _read_uint64(cursor)
    if length > _MAX_GGUF_STRING_BYTES:
        raise TinygradWeightError("GGUF string exceeds the header limit")
    return cursor.read(length).decode("utf-8")


def _read_scalar(cursor: _GgufCursor, value_type: int) -> GgufScalar:
    if value_type == 0:
        return cursor.read(1)[0]
    if value_type == 1:
        return int.from_bytes(cursor.read(1), "little", signed=True)
    if value_type == 2:
        return int.from_bytes(cursor.read(2), "little", signed=False)
    if value_type == 3:
        return int.from_bytes(cursor.read(2), "little", signed=True)
    if value_type == 4:
        return _read_uint32(cursor)
    if value_type == 5:
        return _read_int32(cursor)
    if value_type == 6:
        return _read_float32(cursor)
    if value_type == 7:
        return cursor.read(1) != b"\x00"
    if value_type == 8:
        return _read_string(cursor)
    if value_type == 10:
        return _read_uint64(cursor)
    if value_type == 11:
        return _read_int64(cursor)
    if value_type == 12:
        return _read_float64(cursor)
    raise TinygradWeightError(f"Unsupported GGUF value type {value_type}")


def _tokenizer_arrays(
    retained_arrays: tuple[_RetainedGgufArray, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    tokens: tuple[str, ...] = ()
    merges: tuple[str, ...] = ()
    token_types: tuple[int, ...] = ()
    for retained in retained_arrays:
        if retained.key == "tokenizer.ggml.tokens" and retained.strings is not None:
            tokens = retained.strings
        elif retained.key == "tokenizer.ggml.merges" and retained.strings is not None:
            merges = retained.strings
        elif (
            retained.key == "tokenizer.ggml.token_type"
            and retained.integers is not None
        ):
            token_types = retained.integers
    return tokens, merges, token_types


def _read_metadata_array(cursor: _GgufCursor, key: str) -> _RetainedGgufArray | None:
    item_type = _read_int32(cursor)
    count = _read_uint64(cursor)
    if count > _MAX_GGUF_ARRAY_COUNT:
        raise TinygradWeightError(f"GGUF array {key} exceeds the header limit")
    retain_strings = key in _TOKENIZER_STRING_ARRAYS and item_type == 8
    retain_integers = key == _TOKENIZER_INTEGER_ARRAY and item_type != 9
    strings: list[str] = []
    integers: list[int] = []
    for _ in range(count):
        if item_type == 9:
            _ = _read_metadata_array(cursor, key)
            continue
        value = _read_scalar(cursor, item_type)
        if retain_strings and isinstance(value, str):
            strings.append(value)
            continue
        if retain_integers and isinstance(value, int) and not isinstance(value, bool):
            integers.append(value)
    if retain_strings:
        return _RetainedGgufArray(key, tuple(strings), None, count)
    if retain_integers:
        return _RetainedGgufArray(key, None, tuple(integers), count)
    return None


def _canonical_safetensor_entries(
    entries: Mapping[str, SafetensorEntry],
) -> dict[str, SafetensorEntry]:
    canonical_entries: dict[str, SafetensorEntry] = {}
    for source_name, entry in entries.items():
        canonical_name = canonical_tensor_name(source_name)
        if canonical_name is None:
            continue
        if canonical_name in canonical_entries:
            raise TinygradWeightError(
                f"Checkpoint tensor {canonical_name} is stored more than once"
            )
        canonical_entries[canonical_name] = entry
    return canonical_entries


def _lazy_safetensor_views(
    directory: Path,
    shard: PipelineShardMetadata,
    architecture: TransformerArchitecture,
) -> dict[str, Tensor]:
    from tinygrad.nn.state import safe_load_metadata

    entries = checkpoint_tensor_entries(directory)
    canonical_entries = _canonical_safetensor_entries(entries)
    source_names = _source_names(entries)
    selected = select_shard_tensor_names(
        tuple(canonical_entries),
        shard,
        embeddings_are_tied=architecture.tie_word_embeddings,
    )
    names_by_file: dict[str, list[str]] = {}
    for canonical_name in selected:
        entry = canonical_entries[canonical_name]
        names_by_file.setdefault(entry.relative_file, []).append(canonical_name)

    lazy: dict[str, Tensor] = {}
    for relative_file, canonical_names in names_by_file.items():
        tensor_path = directory / relative_file
        header = read_safetensors_header(tensor_path)
        source_tensor, data_start, _metadata = safe_load_metadata(tensor_path)
        del _metadata
        payload = source_tensor[data_start:]
        for canonical_name in canonical_names:
            source_name = source_names[canonical_name]
            record = header.get(source_name)
            if record is None:
                raise TinygradWeightError(
                    f"{relative_file} is missing selected tensor {source_name}"
                )
            lazy[canonical_name] = _safetensor_view(payload, record, source_name)
        del source_tensor
        gc.collect()
    return lazy


def _source_names(entries: Mapping[str, SafetensorEntry]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for source_name in entries:
        canonical_name = canonical_tensor_name(source_name)
        if canonical_name is None:
            continue
        if canonical_name in mapping:
            raise TinygradWeightError(
                f"Checkpoint tensor {canonical_name} is stored more than once"
            )
        mapping[canonical_name] = source_name
    return mapping


def _safetensor_view(
    payload: Tensor, record: SafetensorTensorRecord, source_name: str
) -> Tensor:
    if len(record.data_offsets) != 2:
        raise TinygradWeightError(f"{source_name} is missing safetensors data offsets")
    start, end = record.data_offsets
    dtype = _safetensor_dtype(record.dtype, source_name)
    return payload[start:end].bitcast(dtype).reshape(*record.shape)


def _safetensor_dtype(dtype_name: str, source_name: str) -> DType:
    from tinygrad import dtypes

    resolved: DType
    if dtype_name == "F16":
        resolved = dtypes.float16
    elif dtype_name == "BF16":
        resolved = dtypes.bfloat16
    elif dtype_name == "F32":
        resolved = dtypes.float32
    elif dtype_name == "I32":
        resolved = dtypes.int32
    elif dtype_name == "U8":
        resolved = dtypes.uint8
    elif dtype_name == "BOOL":
        resolved = dtypes.bool
    else:
        raise TinygradWeightError(
            f"{source_name} has unsupported safetensors dtype {dtype_name}"
        )
    return resolved


def _realized_gguf_tensors(
    directory: Path,
    shard: PipelineShardMetadata,
    architecture: TransformerArchitecture,
) -> dict[str, Tensor]:
    checkpoint = read_gguf_checkpoint(directory)
    assigned = assigned_gguf_tensors(
        checkpoint,
        shard,
        embeddings_are_tied=architecture.tie_word_embeddings,
    )
    realized: dict[str, Tensor] = {}
    for record in assigned:
        canonical_name = record.canonical_name
        if canonical_name is None:
            continue
        realized[canonical_name] = _realize_gguf_record(record, architecture)
        gc.collect()
    return realized


def _realize_gguf_record(
    record: GgufTensorRecord, architecture: TransformerArchitecture
) -> Tensor:
    from tinygrad import Device, dtypes

    canonical_name = record.canonical_name
    if canonical_name is None:
        raise TinygradWeightError(
            f"GGUF tensor {record.source_name} has no canonical name"
        )
    converted = ggml_tensor_from_file(
        record.path,
        record.absolute_start,
        record.payload_byte_count(),
        record.element_count,
        record.ggml_type,
    )
    logical_shape = tuple(reversed(record.dimensions))
    tensor = converted.reshape(*logical_shape) if logical_shape else converted
    # Realize before the rotary permutation. Fusing that permutation into the
    # disk bitcast graph broadcasts the head axes incorrectly.
    tensor = tensor.to(Device.DEFAULT).contiguous().realize()
    expected = _shape_for_name(canonical_name, architecture)
    if expected is not None:
        tensor = _orient_tensor(tensor, expected, canonical_name)
    if _needs_gguf_rotary_permutation(canonical_name):
        head_count = (
            architecture.num_attention_heads
            if "q_proj" in canonical_name
            else architecture.resolved_key_value_heads()
        )
        tensor = permute_gguf_rotary_weight(tensor, head_count)
    return tensor.cast(dtypes.float16).contiguous().realize()


def _shape_for_name(
    canonical_name: str, architecture: TransformerArchitecture
) -> tuple[int, ...] | None:
    hidden = architecture.hidden_size
    query_size = (
        architecture.num_attention_heads * architecture.resolved_head_dimension()
    )
    key_size = (
        architecture.resolved_key_value_heads() * architecture.resolved_head_dimension()
    )
    intermediate = architecture.intermediate_size
    head_dimension = architecture.resolved_head_dimension()
    vocab = architecture.vocab_size
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (vocab, hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (vocab, hidden),
    }
    if canonical_name in shapes:
        return shapes[canonical_name]
    layer_index = extract_layer_num(canonical_name)
    if layer_index is None:
        return None
    suffix = canonical_name.split(f"model.layers.{layer_index}.", maxsplit=1)
    if len(suffix) != 2:
        return None
    by_suffix: dict[str, tuple[int, ...]] = {
        "input_layernorm.weight": (hidden,),
        "self_attn.q_proj.weight": (query_size, hidden),
        "self_attn.k_proj.weight": (key_size, hidden),
        "self_attn.v_proj.weight": (key_size, hidden),
        "self_attn.o_proj.weight": (hidden, query_size),
        "self_attn.q_proj.bias": (query_size,),
        "self_attn.k_proj.bias": (key_size,),
        "self_attn.v_proj.bias": (key_size,),
        "self_attn.o_proj.bias": (hidden,),
        "self_attn.q_norm.weight": (head_dimension,),
        "self_attn.k_norm.weight": (head_dimension,),
        "post_attention_layernorm.weight": (hidden,),
        "mlp.gate_proj.weight": (intermediate, hidden),
        "mlp.up_proj.weight": (intermediate, hidden),
        "mlp.down_proj.weight": (hidden, intermediate),
    }
    return by_suffix.get(suffix[1])


def _needs_gguf_rotary_permutation(canonical_name: str) -> bool:
    return canonical_name.endswith(
        (
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
        )
    )


def _orient_tensor(
    tensor: Tensor, expected: tuple[int, ...], canonical_name: str
) -> Tensor:
    actual = _tensor_shape(tensor)
    if actual == expected:
        return tensor
    if len(actual) == 2 and len(expected) == 2 and actual == (expected[1], expected[0]):
        return tensor.transpose(0, 1)
    raise TinygradWeightError(
        f"Tensor {canonical_name} has shape {actual}, expected {expected}"
    )


def _tensor_shape(tensor: Tensor) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in tensor.shape)


def _realize_on_default_device(tensor: Tensor) -> Tensor:
    from tinygrad import Device

    return tensor.to(Device.DEFAULT).contiguous().realize()


def _yield_realized_groups(
    lazy_parameters: dict[str, Tensor],
    shard: PipelineShardMetadata,
    realize: Callable[[Tensor], Tensor],
) -> Iterator[RealizedParameterGroup]:
    non_layer_names = [
        tensor_name
        for tensor_name in tuple(lazy_parameters)
        if extract_layer_num(tensor_name) is None
    ]
    non_layer_parameters = tuple(
        (tensor_name, realize(lazy_parameters.pop(tensor_name)))
        for tensor_name in non_layer_names
    )
    yield RealizedParameterGroup(layer_index=None, parameters=non_layer_parameters)
    for layer_index in range(shard.start_layer, shard.end_layer):
        layer_names = [
            tensor_name
            for tensor_name in tuple(lazy_parameters)
            if extract_layer_num(tensor_name) == layer_index
        ]
        layer_parameters = tuple(
            (tensor_name, realize(lazy_parameters.pop(tensor_name)))
            for tensor_name in layer_names
        )
        gc.collect()
        yield RealizedParameterGroup(
            layer_index=layer_index, parameters=layer_parameters
        )
    if lazy_parameters:
        leftover = ", ".join(sorted(lazy_parameters))
        raise TinygradWeightError(f"Selected tensors were not realized: {leftover}")
