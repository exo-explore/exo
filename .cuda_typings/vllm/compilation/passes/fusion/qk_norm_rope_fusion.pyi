import torch
from ..inductor_pass import enable_fake_mode as enable_fake_mode
from ..vllm_inductor_pass import (
    VllmInductorPass as VllmInductorPass,
    VllmPatternMatcherPass as VllmPatternMatcherPass,
)
from .matcher_utils import (
    MatcherRMSNorm as MatcherRMSNorm,
    MatcherRotaryEmbedding as MatcherRotaryEmbedding,
)
from .rms_quant_fusion import (
    empty_bf16 as empty_bf16,
    empty_fp32 as empty_fp32,
    empty_i64 as empty_i64,
)
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from torch import fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass
from typing import ParamSpec
from vllm.config import (
    VllmConfig as VllmConfig,
    get_layers_from_vllm_config as get_layers_from_vllm_config,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.rotary_embedding import (
    RotaryEmbedding as RotaryEmbedding,
)

logger: Incomplete
FUSED_QK_ROPE_OP: Incomplete
P = ParamSpec("P")

class QkNormRopePattern:
    num_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    eps: Incomplete
    rmsnorm_matcher: Incomplete
    is_neox: Incomplete
    rope_flashinfer: Incomplete
    rope_matcher: Incomplete
    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        eps: float,
        is_neox: bool,
        rope_flashinfer: bool = False,
    ) -> None: ...
    def get_inputs(self) -> list[torch.Tensor]: ...
    @staticmethod
    def wrap_trace_fn(
        trace_fn: Callable[P, fx.GraphModule],
        *process_fx_fns: Callable[[fx.GraphModule], None],
    ) -> Callable[P, fx.GraphModule]: ...
    @staticmethod
    def fx_view_to_reshape(gm: torch.fx.GraphModule) -> None: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class QKNormRoPEFusionPass(VllmPatternMatcherPass):
    patterns: PatternMatcherPass
    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None: ...
    matched_count: Incomplete
    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None: ...
    def uuid(self) -> str: ...
