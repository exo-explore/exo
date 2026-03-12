import abc
import torch
from ..inductor_pass import enable_fake_mode as enable_fake_mode
from ..vllm_inductor_pass import (
    VllmInductorPass as VllmInductorPass,
    VllmPatternMatcherPass as VllmPatternMatcherPass,
)
from .matcher_utils import (
    MatcherQuantFP8 as MatcherQuantFP8,
    MatcherSiluAndMul as MatcherSiluAndMul,
)
from .rms_quant_fusion import (
    QUANT_OPS as QUANT_OPS,
    empty_bf16 as empty_bf16,
    empty_fp32 as empty_fp32,
    empty_i32 as empty_i32,
)
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._ops import OpOverload as OpOverload
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey as QuantKey,
    kFp8StaticTensorSym as kFp8StaticTensorSym,
    kNvfp4Dynamic as kNvfp4Dynamic,
)
from vllm.platforms import current_platform as current_platform

logger: Incomplete
FP8_DTYPE: Incomplete
FP4_DTYPE: Incomplete
SILU_MUL_OP: Incomplete
FUSED_OPS: dict[QuantKey, OpOverload]
silu_and_mul_nvfp4_quant_supported: Incomplete

class ActivationQuantPattern(ABC, metaclass=abc.ABCMeta):
    quant_key: Incomplete
    quant_dtype: Incomplete
    QUANT_OP: Incomplete
    FUSED_OP: Incomplete
    silu_and_mul_matcher: Incomplete
    def __init__(self, quant_key: QuantKey) -> None: ...
    def empty_quant(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    @abstractmethod
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class SiluMulFp8StaticQuantPattern(ActivationQuantPattern):
    quant_matcher: Incomplete
    def __init__(self) -> None: ...
    def get_inputs(self) -> list[torch.Tensor]: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class SiluMulNvfp4QuantPattern(ActivationQuantPattern):
    def __init__(self) -> None: ...
    def get_inputs(self) -> list[torch.Tensor]: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class ActivationQuantFusionPass(VllmPatternMatcherPass):
    patterns: PatternMatcherPass
    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None: ...
    matched_count: Incomplete
    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph) -> None: ...
    def uuid(self) -> str: ...
