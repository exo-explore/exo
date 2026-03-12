from .fusion.act_quant_fusion import (
    ActivationQuantFusionPass as ActivationQuantFusionPass,
)
from .fusion.allreduce_rms_fusion import AllReduceFusionPass as AllReduceFusionPass
from .fusion.attn_quant_fusion import AttnFusionPass as AttnFusionPass
from .fusion.collective_fusion import AsyncTPPass as AsyncTPPass
from .fusion.qk_norm_rope_fusion import QKNormRoPEFusionPass as QKNormRoPEFusionPass
from .fusion.rms_quant_fusion import RMSNormQuantFusionPass as RMSNormQuantFusionPass
from .fusion.rocm_aiter_fusion import (
    RocmAiterRMSNormQuantFusionPass as RocmAiterRMSNormQuantFusionPass,
    RocmAiterSiluMulFp8GroupQuantFusionPass as RocmAiterSiluMulFp8GroupQuantFusionPass,
    RocmAiterTritonAddRMSNormPadFusionPass as RocmAiterTritonAddRMSNormPadFusionPass,
)
from .fusion.rope_kvcache_fusion import RopeKVCacheFusionPass as RopeKVCacheFusionPass
from .fusion.sequence_parallelism import (
    SequenceParallelismPass as SequenceParallelismPass,
)
from .inductor_pass import (
    CustomGraphPass as CustomGraphPass,
    InductorPass as InductorPass,
    get_pass_context as get_pass_context,
)
from .utility.fix_functionalization import (
    FixFunctionalizationPass as FixFunctionalizationPass,
)
from .utility.noop_elimination import NoOpEliminationPass as NoOpEliminationPass
from .utility.scatter_split_replace import (
    ScatterSplitReplacementPass as ScatterSplitReplacementPass,
)
from .utility.split_coalescing import SplitCoalescingPass as SplitCoalescingPass
from .vllm_inductor_pass import VllmInductorPass as VllmInductorPass
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from torch import fx as fx
from typing import ParamSpec, TypeVar
from vllm import envs as envs
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.compilation.passes.utility.post_cleanup import (
    PostCleanupPass as PostCleanupPass,
)
from vllm.config import (
    VllmConfig as VllmConfig,
    set_current_vllm_config as set_current_vllm_config,
)
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform
from vllm.utils.system_utils import set_env_var as set_env_var

logger: Incomplete
P = ParamSpec("P")
R = TypeVar("R")

def with_pattern_match_debug(fn: Callable[P, R]) -> Callable[P, R]: ...

class PostGradPassManager(CustomGraphPass):
    passes: list[InductorPass]
    def __init__(self) -> None: ...
    @with_pattern_match_debug
    def __call__(self, graph: fx.Graph) -> None: ...
    pass_config: Incomplete
    post_cleanup: Incomplete
    fix_functionalization: Incomplete
    def configure(self, config: VllmConfig) -> None: ...
    def add(self, pass_: InductorPass) -> None: ...
    def uuid(self) -> str: ...
