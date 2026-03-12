import torch
from ..inductor_pass import enable_fake_mode as enable_fake_mode
from ..vllm_inductor_pass import (
    VllmInductorPass as VllmInductorPass,
    VllmPatternMatcherPass as VllmPatternMatcherPass,
)
from .act_quant_fusion import ActivationQuantPattern as ActivationQuantPattern
from .matcher_utils import (
    MatcherFusedAddRMSNorm as MatcherFusedAddRMSNorm,
    MatcherQuantFP8 as MatcherQuantFP8,
    MatcherRMSNorm as MatcherRMSNorm,
    MatcherSiluAndMul as MatcherSiluAndMul,
)
from .rms_quant_fusion import FusedRMSQuantKey as FusedRMSQuantKey
from _typeshed import Incomplete
from torch import fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape as GroupShape,
    QuantKey as QuantKey,
    ScaleDesc as ScaleDesc,
    kFp8Dynamic128Sym as kFp8Dynamic128Sym,
)
from vllm.platforms import current_platform as current_platform

logger: Incomplete
FP8_DTYPE: Incomplete

class AiterRMSNormQuantPattern:
    epsilon: Incomplete
    quant_dtype: Incomplete
    rmsnorm_matcher: Incomplete
    quant_matcher: Incomplete
    def __init__(
        self, epsilon: float, key: FusedRMSQuantKey, match_aiter_quant: bool = True
    ) -> None: ...

class AiterRMSNormDynamicQuantPattern(AiterRMSNormQuantPattern):
    FUSED_OP: Incomplete
    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        match_aiter_quant: bool = True,
        group_shape: GroupShape = ...,
        symmetric: bool = True,
    ) -> None: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class AiterFusedAddRMSNormDynamicQuantPattern(AiterRMSNormQuantPattern):
    FUSED_OP: Incomplete
    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        match_aiter_quant: bool = True,
        group_shape: GroupShape = ...,
        symmetric: bool = True,
    ) -> None: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class AiterRMSFp8GroupQuantPattern(AiterRMSNormQuantPattern):
    FUSED_OP: Incomplete
    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        match_aiter_quant: bool = True,
        symmetric: bool = True,
    ) -> None: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class AiterFusedAddRMSFp8GroupQuantPattern(AiterRMSNormQuantPattern):
    FUSED_OP: Incomplete
    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        match_aiter_quant: bool = True,
        symmetric: bool = True,
    ) -> None: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class RocmAiterRMSNormQuantFusionPass(VllmPatternMatcherPass):
    patterns: PatternMatcherPass
    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None: ...
    matched_count: Incomplete
    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None: ...
    def uuid(self) -> str: ...

class AiterSiluMulFp8GroupQuantPattern(ActivationQuantPattern):
    FUSED_SILU_MUL_QUANT_OP: Incomplete
    silu_and_mul_matcher: Incomplete
    quant_matcher: Incomplete
    def __init__(self) -> None: ...
    def get_inputs(self) -> list[torch.Tensor]: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class RocmAiterSiluMulFp8GroupQuantFusionPass(VllmPatternMatcherPass):
    patterns: PatternMatcherPass
    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None: ...
    matched_count: Incomplete
    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph) -> None: ...
    def uuid(self) -> str: ...

class AddAiterRMSNormPadPattern:
    AITER_TRITON_ADD_RMSNORM_PAD_OP: Incomplete
    epsilon: Incomplete
    hidden_size: Incomplete
    x_pad_to_multiple: Incomplete
    rmsnorm_matcher: Incomplete
    def __init__(
        self, epsilon: float, hidden_size: int, x_pad_to_multiple: int
    ) -> None: ...
    def get_inputs(self) -> list[torch.Tensor]: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class RocmAiterTritonAddRMSNormPadFusionPass(VllmPatternMatcherPass):
    patterns: PatternMatcherPass
    def __init__(self, config: VllmConfig) -> None: ...
    matched_count: Incomplete
    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph) -> None: ...
    def uuid(self) -> str: ...
