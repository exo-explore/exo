import abc
import torch
from ..fx_utils import is_func as is_func
from ..inductor_pass import enable_fake_mode as enable_fake_mode
from ..vllm_inductor_pass import (
    VllmInductorPass as VllmInductorPass,
    VllmPatternMatcherPass as VllmPatternMatcherPass,
)
from .matcher_utils import MatcherQuantFP8 as MatcherQuantFP8
from .rms_quant_fusion import (
    QUANT_OPS as QUANT_OPS,
    empty_bf16 as empty_bf16,
    empty_fp32 as empty_fp32,
    empty_i32 as empty_i32,
)
from _typeshed import Incomplete
from abc import ABC
from collections.abc import Callable as Callable
from torch import fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass
from typing import Any, ParamSpec
from vllm.config import (
    VllmConfig as VllmConfig,
    get_layers_from_vllm_config as get_layers_from_vllm_config,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey as QuantKey,
    kNvfp4Dynamic as kNvfp4Dynamic,
    kStaticTensorScale as kStaticTensorScale,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.math_utils import round_up as round_up

logger: Incomplete
P = ParamSpec("P")
FP8_DTYPE: Incomplete
FP4_DTYPE: Incomplete
ATTN_OP: Incomplete
RESHAPE_OP: Incomplete

class AttentionQuantPattern(ABC, metaclass=abc.ABCMeta):
    layer: Incomplete
    layer_name: Incomplete
    num_heads: Incomplete
    head_size: Incomplete
    quant_key: Incomplete
    quant_dtype: Incomplete
    dtype: Incomplete
    QUANT_OP: Incomplete
    def __init__(
        self, layer: Attention, quant_key: QuantKey, dtype: torch.dtype
    ) -> None: ...
    def empty(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    def empty_quant(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...
    @staticmethod
    def wrap_trace_fn(
        trace_fn: Callable[P, fx.GraphModule],
        *process_fx_fns: Callable[[fx.GraphModule], None],
    ) -> Callable[P, fx.GraphModule]: ...
    @staticmethod
    def fx_view_to_reshape(gm: torch.fx.GraphModule) -> None: ...
    @staticmethod
    def remove_noop_permutes(gm: torch.fx.GraphModule) -> None: ...
    def register_if_supported(self, pm_pass: PatternMatcherPass) -> None: ...

class AttentionFp8StaticQuantPattern(AttentionQuantPattern):
    quant_matcher: Incomplete
    def __init__(
        self, layer: Attention, dtype: torch.dtype, symmetric: bool = True
    ) -> None: ...

class AttentionNvfp4QuantPattern(AttentionQuantPattern):
    def __init__(self, layer: Attention, dtype: torch.dtype) -> None: ...

class AttnFusionPass(VllmPatternMatcherPass):
    patterns: Incomplete
    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None: ...
    matched_count: Incomplete
    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.graph.Graph) -> None: ...
    def uuid(self) -> str: ...
