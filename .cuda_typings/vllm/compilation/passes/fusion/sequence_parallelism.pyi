import torch
import torch.fx as fx
from ..inductor_pass import enable_fake_mode as enable_fake_mode
from ..utility.noop_elimination import NoOpEliminationPass as NoOpEliminationPass
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
from collections.abc import Callable as Callable, Sequence
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.config import VllmConfig as VllmConfig
from vllm.config.utils import Range as Range
from vllm.distributed import (
    get_tp_group as get_tp_group,
    tensor_model_parallel_all_reduce as tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym as kFp8StaticTensorSym,
)

logger: Incomplete
SP_MIN_HIDDEN_SIZE: dict[int, int]
SP_MIN_PER_GPU_SIZE_MB: dict[int, float]

def get_sequence_parallelism_threshold(
    hidden_size: int, tp_size: int, element_size: int
) -> int | None: ...
def get_first_out_wrapper(
    fn: Callable[..., Sequence[torch.Tensor]],
) -> Callable[..., torch.Tensor]: ...

class _SequenceParallelPatternHelper:
    epsilon: Incomplete
    dtype: Incomplete
    device: Incomplete
    tp_group: Incomplete
    tp_size: Incomplete
    def __init__(
        self, epsilon: float, dtype: torch.dtype, device: str | None
    ) -> None: ...

class FirstAllReduceRMSNormPattern(_SequenceParallelPatternHelper):
    rmsnorm_matcher: Incomplete
    def __init__(
        self, epsilon: float, dtype: torch.dtype, device: str | None
    ) -> None: ...
    def get_inputs(self) -> list[torch.Tensor]: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class MiddleAllReduceRMSNormPattern(_SequenceParallelPatternHelper):
    rmsnorm_matcher: Incomplete
    def __init__(
        self, epsilon: float, dtype: torch.dtype, device: str | None
    ) -> None: ...
    def get_inputs(self) -> list[torch.Tensor]: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class FirstAllReduceRMSNormStaticFP8Pattern(_SequenceParallelPatternHelper):
    rmsnorm_matcher: Incomplete
    quant_matcher: Incomplete
    def __init__(
        self, epsilon: float, dtype: torch.dtype, device: str | None
    ) -> None: ...
    def get_inputs(self) -> list[torch.Tensor]: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class MiddleAllReduceRMSNormStaticFP8Pattern(_SequenceParallelPatternHelper):
    rmsnorm_matcher: Incomplete
    quant_matcher: Incomplete
    def __init__(
        self, epsilon: float, dtype: torch.dtype, device: str | None
    ) -> None: ...
    def get_inputs(self) -> list[torch.Tensor]: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class SequenceParallelismPass(VllmPatternMatcherPass):
    min_token_num: Incomplete
    noop_cleanup: Incomplete
    patterns: PatternMatcherPass
    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None: ...
    def is_applicable_for_range(self, compile_range: Range) -> bool: ...
    matched_count: Incomplete
    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None: ...
