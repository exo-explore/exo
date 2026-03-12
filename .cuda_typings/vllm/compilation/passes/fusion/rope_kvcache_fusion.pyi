import torch
from ..inductor_pass import enable_fake_mode as enable_fake_mode
from ..vllm_inductor_pass import (
    VllmInductorPass as VllmInductorPass,
    VllmPatternMatcherPass as VllmPatternMatcherPass,
)
from .matcher_utils import MatcherRotaryEmbedding as MatcherRotaryEmbedding
from .rms_quant_fusion import empty_bf16 as empty_bf16, empty_i64 as empty_i64
from _typeshed import Incomplete
from torch import fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.config import (
    VllmConfig as VllmConfig,
    get_layers_from_vllm_config as get_layers_from_vllm_config,
)
from vllm.config.utils import Range as Range
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention.attention import (
    Attention as Attention,
    get_attention_context as get_attention_context,
)
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)

logger: Incomplete

def fused_rope_and_unified_kv_cache_update_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: str = "",
) -> torch.Tensor: ...
def fused_rope_and_unified_kv_cache_update_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: str = "",
) -> torch.Tensor: ...

class RopeReshapeKVCachePattern:
    FUSED_OP: Incomplete
    layer_name: Incomplete
    num_heads: Incomplete
    num_kv_heads: Incomplete
    head_size: Incomplete
    head_size_v: Incomplete
    is_neox: Incomplete
    q_size: Incomplete
    k_size: Incomplete
    v_size: Incomplete
    rope_matcher: Incomplete
    def __init__(self, layer: Attention, is_neox: bool) -> None: ...
    def get_inputs(self) -> list[torch.Tensor]: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class RopeKVCacheFusionPass(VllmPatternMatcherPass):
    patterns: PatternMatcherPass
    max_token_num: Incomplete
    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None: ...
    matched_count: Incomplete
    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None: ...
    def is_applicable_for_range(self, compile_range: Range) -> bool: ...
    def uuid(self) -> str: ...
