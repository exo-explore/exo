import torch
from ..inductor_pass import enable_fake_mode as enable_fake_mode
from ..vllm_inductor_pass import (
    VllmInductorPass as VllmInductorPass,
    VllmPatternMatcherPass as VllmPatternMatcherPass,
)
from .matcher_utils import (
    MatcherFusedAddRMSNorm as MatcherFusedAddRMSNorm,
    MatcherQuantFP8 as MatcherQuantFP8,
    MatcherRMSNorm as MatcherRMSNorm,
)
from _typeshed import Incomplete
from torch import fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._ops import OpOverload as OpOverload
from typing import Any, NamedTuple
from vllm.config import (
    VllmConfig as VllmConfig,
    get_current_vllm_config as get_current_vllm_config,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape as GroupShape,
    QuantKey as QuantKey,
    ScaleDesc as ScaleDesc,
    kFp8Dynamic128Sym as kFp8Dynamic128Sym,
    kFp8Dynamic64Sym as kFp8Dynamic64Sym,
    kFp8DynamicTensorSym as kFp8DynamicTensorSym,
    kFp8DynamicTokenSym as kFp8DynamicTokenSym,
    kFp8StaticTensorSym as kFp8StaticTensorSym,
    kNvfp4Dynamic as kNvfp4Dynamic,
    kStaticTensorScale as kStaticTensorScale,
)
from vllm.platforms import current_platform as current_platform

logger: Incomplete
FP8_DTYPE: Incomplete
FP4_DTYPE: Incomplete

def empty_bf16(*args: Any, **kwargs: Any) -> torch.Tensor: ...
def empty_fp32(*args: Any, **kwargs: Any) -> torch.Tensor: ...
def empty_i32(*args: Any, **kwargs: Any) -> torch.Tensor: ...
def empty_i64(*args: Any, **kwargs: Any) -> torch.Tensor: ...

RMS_OP: Incomplete
RMS_ADD_OP: Incomplete
QUANT_OPS: dict[QuantKey, OpOverload]

class FusedRMSQuantKey(NamedTuple):
    quant: QuantKey
    fused_add: bool

FUSED_OPS: dict[FusedRMSQuantKey, OpOverload]

class RMSNormQuantPattern:
    epsilon: Incomplete
    quant_dtype: Incomplete
    model_dtype: Incomplete
    FUSED_OP: Incomplete
    rmsnorm_matcher: Incomplete
    quant_matcher: Incomplete
    def __init__(
        self,
        epsilon: float,
        key: FusedRMSQuantKey,
        has_col_major_scales: bool = False,
        is_e8m0: bool = False,
        is_tma_aligned: bool = False,
    ) -> None: ...

class RMSNormStaticQuantPattern(RMSNormQuantPattern):
    def __init__(
        self, epsilon: float, quant_dtype: torch.dtype, symmetric: bool = True
    ) -> None: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class FusedAddRMSNormStaticQuantPattern(RMSNormQuantPattern):
    def __init__(
        self, epsilon: float, quant_dtype: torch.dtype, symmetric: bool = True
    ) -> None: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class FusedAddRMSNormGroupQuantPattern(RMSNormQuantPattern):
    group_shape: Incomplete
    is_e8m0: Incomplete
    has_col_major_scales: Incomplete
    is_tma_aligned: Incomplete
    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        symmetric: bool = True,
        is_e8m0: bool = False,
        has_col_major_scales: bool = True,
        is_tma_aligned: bool = True,
    ) -> None: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class RMSNormGroupQuantPattern(RMSNormQuantPattern):
    group_shape: Incomplete
    has_col_major_scales: Incomplete
    is_tma_aligned: Incomplete
    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        symmetric: bool = True,
        is_e8m0: bool = False,
        has_col_major_scales: bool = True,
        is_tma_aligned: bool = True,
    ) -> None: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class RMSNormDynamicQuantPattern(RMSNormQuantPattern):
    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape = ...,
        symmetric: bool = True,
    ) -> None: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class FusedAddRMSNormDynamicQuantPattern(RMSNormQuantPattern):
    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape = ...,
        symmetric: bool = True,
    ) -> None: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class RMSNormQuantFusionPass(VllmPatternMatcherPass):
    patterns: PatternMatcherPass
    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None: ...
    matched_count: Incomplete
    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None: ...
    def uuid(self) -> str: ...
