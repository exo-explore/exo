from math import ceil
from typing import final

from pydantic import BaseModel, ConfigDict
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor


@final
class PackedTensor(BaseModel):
    # Immutable container for bit-packed weight data with unpacking metadata

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    tensor: Tensor
    original_shape: tuple[int, ...]
    pack_factor: int
    bits: int

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"'{type(self).__name__}' is frozen")

def calculate_pack_factor(bits: int) -> int:
    """
        Pack factor is the number of values packed into a bits. The general
        formula is pack_factor =  (packed_type_bits) // bits.
        In this case, we are dealing with a uint32 value.
    """

    return 32 // bits

def pack_bits(unpacked: Tensor, bits: int) -> PackedTensor:
    """
        This functions converts a float16 tensor into a packed uint32 tensor, as
        used by MLX community for their quantized models.
    """

    pack_factor = calculate_pack_factor(bits)
    original_shape = tuple(int(d) for d in unpacked.shape)
    last_dim = original_shape[-1]
    padded_last_dim = ceil(last_dim / pack_factor) * pack_factor
    packed_last_dim = padded_last_dim // pack_factor
    packed_shape = (*original_shape[:-1], packed_last_dim)

    if last_dim != padded_last_dim:
        pad_shape = (*original_shape[:-1], padded_last_dim - last_dim)
        padding = Tensor.zeros(*pad_shape, dtype = unpacked.dtype)  # pyright: ignore[reportUnknownMemberType]
        unpacked = unpacked.cat(padding, dim = -1)

    packed = Tensor.zeros(*packed_shape, dtype=dtypes.uint32)  # pyright: ignore[reportUnknownMemberType]
    for slot in range(pack_factor):
        values = unpacked[..., slot::pack_factor].cast(dtypes.uint32)
        packed = packed | (values << (slot * bits))

    return PackedTensor(
        tensor = packed,
        original_shape = original_shape,
        pack_factor = pack_factor,
        bits = bits,
    )

def unpack_bits(packed_tensor: PackedTensor) -> Tensor:
    """
        Unpack bit-packed uint32 tensor backed to individual float16 values
    """
    packed = packed_tensor.tensor
    original_shape = packed_tensor.original_shape
    pack_factor = packed_tensor.pack_factor
    bits = packed_tensor.bits
    mask = (1 << bits) - 1

    slices: list[Tensor] = []
    for slot in range(pack_factor):
        slices.append((packed >> (slot * bits)) & mask)

    stacked = Tensor.stack(*slices, dim=0)
    padded_last_dim = packed.shape[-1] * pack_factor
    padded_shape = (*original_shape[:-1], padded_last_dim)
    result = stacked.permute(*range(1, len(stacked.shape)), 0).reshape(padded_shape)  # pyright: ignore[reportUnknownMemberType]

    if padded_last_dim != original_shape[-1]:
        # Trim padding for padded, packed tensors
        result = result[..., :original_shape[-1]]

    return result.cast(dtypes.float16)
