import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from fastapi.responses import JSONResponse
from typing import Any
from vllm.logger import init_logger as init_logger
from vllm.outputs import PoolingRequestOutput as PoolingRequestOutput
from vllm.utils.serial_utils import (
    EMBED_DTYPES as EMBED_DTYPES,
    EmbedDType as EmbedDType,
    Endianness as Endianness,
    binary2tensor as binary2tensor,
    tensor2binary as tensor2binary,
)

logger: Incomplete

@dataclass
class MetadataItem:
    index: int
    embed_dtype: EmbedDType
    endianness: Endianness
    start: int
    end: int
    shape: tuple[int, ...]

def build_metadata_items(
    embed_dtype: EmbedDType,
    endianness: Endianness,
    shape: tuple[int, ...],
    n_request: int,
) -> list[MetadataItem]: ...
def encode_pooling_output_float(output: PoolingRequestOutput) -> list[float]: ...
def encode_pooling_output_binary(
    output: PoolingRequestOutput, embed_dtype: EmbedDType, endianness: Endianness
) -> bytes: ...
def encode_pooling_output_base64(
    output: PoolingRequestOutput, embed_dtype: EmbedDType, endianness: Endianness
) -> str: ...
def encode_pooling_bytes(
    pooling_outputs: list[PoolingRequestOutput],
    embed_dtype: EmbedDType,
    endianness: Endianness,
) -> tuple[list[bytes], list[dict[str, Any]], dict[str, Any]]: ...
def decode_pooling_output(
    items: list[MetadataItem], body: bytes
) -> list[torch.Tensor]: ...
def get_json_response_cls() -> type[JSONResponse]: ...
