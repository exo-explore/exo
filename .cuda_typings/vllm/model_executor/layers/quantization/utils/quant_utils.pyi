import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar, NamedTuple
from vllm.model_executor.layers.linear import LinearBase as LinearBase
from vllm.platforms import current_platform as current_platform
from vllm.scalar_type import ScalarType as ScalarType, scalar_types as scalar_types

FP8_DTYPE: Incomplete
FP4_DTYPE: Incomplete
MXFP_SCALE_DTYPE: Incomplete

def get_fp8_min_max() -> tuple[float, float]: ...

class _GroupShape(NamedTuple):
    row: int
    col: int

class GroupShape(_GroupShape):
    PER_TENSOR: ClassVar["GroupShape"]
    PER_TOKEN: ClassVar["GroupShape"]
    PER_CHANNEL: ClassVar["GroupShape"]
    def is_per_tensor(self) -> bool: ...
    def is_per_token(self) -> bool: ...
    def is_per_channel(self) -> bool: ...
    def is_per_group(self) -> bool: ...

@dataclass(frozen=True)
class ScaleDesc:
    dtype: torch.dtype
    static: bool
    group_shape: GroupShape

@dataclass(frozen=True)
class QuantKey:
    dtype: torch.dtype
    scale: ScaleDesc
    scale2: ScaleDesc | None = ...
    symmetric: bool = ...

kStaticTensorScale: Incomplete
kFp8StaticTensorSym: Incomplete
kDynamicTensorScale: Incomplete
kFp8DynamicTensorSym: Incomplete
kStaticTokenScale: Incomplete
kFp8StaticTokenSym: Incomplete
kStaticChannelScale: Incomplete
kFp8StaticChannelSym: Incomplete
kDynamicTokenScale: Incomplete
kFp8DynamicTokenSym: Incomplete
kNvfp4DynamicGroupScale: Incomplete
kNvfp4Dynamic: Incomplete
kNvfp4StaticGroupScale: Incomplete
kNvfp4Static: Incomplete
kDynamic128Scale: Incomplete
kFp8Dynamic128Sym: Incomplete
kStatic128BlockScale: Incomplete
kFp8Static128BlockSym: Incomplete
kDynamic64Scale: Incomplete
kFp8Dynamic64Sym: Incomplete
kMxfp4DynamicGroupScale: Incomplete
kMxfp4Dynamic: Incomplete
kMxfp8DynamicGroupScale: Incomplete
kMxfp8Dynamic: Incomplete
kMxfp4StaticGroupScale: Incomplete
kMxfp4Static: Incomplete

def group_broadcast(t, shape): ...
def prep_scale_for_group_broadcast(
    scale: torch.Tensor, x: torch.Tensor, group_shape: GroupShape | None
) -> torch.Tensor: ...
def scaled_quantize(
    x: torch.Tensor,
    group_shape: GroupShape,
    quant_dtype: torch.dtype,
    compute_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def scaled_dequantize(
    x_q: torch.Tensor,
    x_s: torch.Tensor,
    group_shape: GroupShape | None = None,
    out_dtype: torch.dtype = ...,
) -> torch.Tensor: ...
def get_attribute_fallback(obj, attributes: list[str]): ...
def get_and_maybe_dequant_weights(layer: LinearBase, out_dtype: torch.dtype = ...): ...
def pack_quantized_values_into_int32(
    w_q: torch.Tensor, wtype: ScalarType, packed_dim: int = 0
): ...
def unpack_quantized_values_into_int32(
    w_q: torch.Tensor, wtype: ScalarType, packed_dim: int = 0
): ...
def is_layer_skipped(
    prefix: str,
    ignored_layers: list[str],
    fused_mapping: Mapping[str, list[str]] = ...,
    *,
    skip_with_substr: bool = False,
) -> bool: ...
def get_pack_factor(num_bits): ...
def permute_rows(
    q_w: torch.Tensor,
    w_ref: torch.Tensor,
    group_size: int,
    test_perm: torch.Tensor | None = None,
): ...
def quantize_weights(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: int | None,
    zero_points: bool = False,
    ref_zero_points_after_scales: bool = False,
): ...

SUPPORTED_GPTQ_QUANT_TYPES: Incomplete
SUPPORTED_GROUP_SIZES: Incomplete

def gptq_quantize_weights(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    act_order: bool,
    test_perm: torch.Tensor | None = None,
): ...
def sort_weights(q_w: torch.Tensor, g_idx: torch.Tensor): ...
def pack_rows(q_w: torch.Tensor, num_bits: int, size_k: int, size_n: int): ...
def pack_cols(q_w: torch.Tensor, num_bits: int, size_k: int, size_n: int): ...
def unpack_cols(packed_q_w: torch.Tensor, num_bits: int, size_k: int, size_n: int): ...
def gptq_pack(q_w: torch.Tensor, num_bits: int, size_k: int, size_n: int): ...
def awq_pack(q_w: torch.Tensor, num_bits: int, size_k: int, size_n: int): ...
def convert_bf16_scales_to_fp8(
    quant_fp8: Callable, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...
def convert_packed_uint4b8_to_signed_int4_inplace(t: torch.Tensor) -> torch.Tensor: ...
