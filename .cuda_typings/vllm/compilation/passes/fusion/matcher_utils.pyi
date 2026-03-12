import abc
import torch
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from torch._ops import OpOverload as OpOverload
from typing import Any
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.config import get_current_vllm_config as get_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8 as QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape as GroupShape,
    QuantKey as QuantKey,
    kFp8Dynamic128Sym as kFp8Dynamic128Sym,
    kFp8Dynamic64Sym as kFp8Dynamic64Sym,
    kFp8DynamicTensorSym as kFp8DynamicTensorSym,
    kFp8DynamicTokenSym as kFp8DynamicTokenSym,
    kFp8StaticTensorSym as kFp8StaticTensorSym,
    kNvfp4Dynamic as kNvfp4Dynamic,
)
from vllm.model_executor.layers.rotary_embedding import (
    RotaryEmbedding as RotaryEmbedding,
)
from vllm.platforms import current_platform as current_platform

RMS_OP: Incomplete
RMS_ADD_OP: Incomplete
ROTARY_OP: Incomplete
FLASHINFER_ROTARY_OP: Incomplete
QUANT_OPS: dict[QuantKey, OpOverload]
SILU_MUL_OP: Incomplete

class MatcherCustomOp(ABC, metaclass=abc.ABCMeta):
    model_dtype: Incomplete
    device: Incomplete
    enabled: Incomplete
    forward: Incomplete
    def __init__(self, enabled: bool) -> None: ...
    @abstractmethod
    def forward_custom(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def forward_native(self, *args: Any, **kwargs: Any) -> Any: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def empty(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def empty_int64(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def empty_f32(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def inputs(self) -> list[torch.Tensor]: ...

class MatcherRotaryEmbedding(MatcherCustomOp):
    is_neox: Incomplete
    head_size: Incomplete
    num_heads: Incomplete
    num_kv_heads: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    rotary_dim: Incomplete
    rotary_op: Incomplete
    def __init__(
        self,
        is_neox: bool,
        head_size: int,
        num_heads: int,
        num_kv_heads: int,
        use_flashinfer: bool = False,
        match_rocm_aiter: bool | None = None,
        enabled: bool | None = None,
    ) -> None: ...
    def inputs(self) -> list[torch.Tensor]: ...
    def forward_custom(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        cos_sin_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        cos_sin_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class MatcherRMSNorm(MatcherCustomOp):
    epsilon: Incomplete
    match_rocm_aiter: Incomplete
    def __init__(
        self,
        epsilon: float,
        enabled: bool | None = None,
        match_rocm_aiter: bool = False,
    ) -> None: ...
    def inputs(self) -> list[torch.Tensor]: ...
    def forward_rocm_aiter(
        self, input: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor: ...
    def forward_custom(
        self, input: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor: ...
    def forward_native(
        self, input: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor: ...

class MatcherFusedAddRMSNorm(MatcherCustomOp):
    epsilon: Incomplete
    match_rocm_aiter: Incomplete
    def __init__(
        self,
        epsilon: float,
        enabled: bool | None = None,
        match_rocm_aiter: bool = False,
    ) -> None: ...
    def inputs(self) -> list[torch.Tensor]: ...
    def forward_rocm_aiter(
        self, input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def forward_custom(
        self, input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def forward_native(
        self, input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class MatcherQuantFP8(MatcherCustomOp):
    quant_key: Incomplete
    has_col_major_scales: Incomplete
    is_e8m0: Incomplete
    match_rocm_aiter: Incomplete
    is_tma_aligned: Incomplete
    QUANT_OP: Incomplete
    quant_fp8: Incomplete
    def __init__(
        self,
        quant_key: QuantKey,
        enabled: bool | None = None,
        has_col_major_scales: bool = False,
        is_e8m0: bool = False,
        match_rocm_aiter: bool = False,
        is_tma_aligned: bool = False,
    ) -> None: ...
    def forward_rocm_aiter(
        self, input: torch.Tensor, scale: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def forward_custom(
        self, input: torch.Tensor, scale: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def forward_native(
        self, input: torch.Tensor, scale: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def make_scale(
        self, input: torch.Tensor, transposed: bool = False
    ) -> torch.Tensor: ...
    def inputs(self) -> list[torch.Tensor]: ...

class MatcherSiluAndMul(MatcherCustomOp):
    def __init__(self, enabled: bool | None = None) -> None: ...
    def inputs(self) -> list[torch.Tensor]: ...
    def forward_custom(self, x: torch.Tensor) -> torch.Tensor: ...
    def forward_native(self, x: torch.Tensor) -> torch.Tensor: ...
