import numpy.typing as npt
import torch
from _typeshed import Incomplete
from collections.abc import Mapping
from dataclasses import dataclass

sys_byteorder: Incomplete

@dataclass(frozen=True)
class DTypeInfo:
    torch_dtype: torch.dtype
    torch_view_dtype: torch.dtype
    numpy_view_dtype: npt.DTypeLike
    @property
    def nbytes(self) -> int: ...

EmbedDType: Incomplete
Endianness: Incomplete
EncodingFormat: Incomplete
EMBED_DTYPES: Mapping[EmbedDType, DTypeInfo]
ENDIANNESS: tuple[Endianness, ...]

def tensor2base64(x: torch.Tensor) -> str: ...
def tensor2binary(
    tensor: torch.Tensor, embed_dtype: EmbedDType, endianness: Endianness
) -> bytes: ...
def binary2tensor(
    binary: bytes,
    shape: tuple[int, ...],
    embed_dtype: EmbedDType,
    endianness: Endianness,
) -> torch.Tensor: ...
