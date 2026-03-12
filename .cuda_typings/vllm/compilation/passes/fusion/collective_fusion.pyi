import torch
import torch.fx as fx
from ..inductor_pass import enable_fake_mode as enable_fake_mode
from ..vllm_inductor_pass import (
    VllmInductorPass as VllmInductorPass,
    VllmPatternMatcherPass as VllmPatternMatcherPass,
)
from _typeshed import Incomplete
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.config import VllmConfig as VllmConfig
from vllm.config.utils import Range as Range
from vllm.distributed import get_tp_group as get_tp_group
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform

FP8_DTYPE: Incomplete
logger: Incomplete

class BasePattern:
    dtype: Incomplete
    device: Incomplete
    tp: Incomplete
    tp_size: Incomplete
    def __init__(self, dtype: torch.dtype, device: str | None) -> None: ...

class GEMMReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class AllGatherGEMMPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class ScaledMMReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class AllGatherScaledMMPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class CutlassScaledMMReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class AllGatherCutlassScaledMMPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class AsyncTPPass(VllmPatternMatcherPass):
    patterns: PatternMatcherPass
    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None: ...
    def is_applicable_for_range(self, compile_range: Range) -> bool: ...
    matched_count: Incomplete
    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None: ...
