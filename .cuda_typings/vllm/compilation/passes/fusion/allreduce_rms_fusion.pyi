import torch
import torch.fx as fx
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
from torch._inductor.pattern_matcher import PatternMatcherPass
from types import ModuleType
from vllm.config import VllmConfig as VllmConfig
from vllm.config.utils import Range as Range
from vllm.distributed import (
    get_tp_group as get_tp_group,
    tensor_model_parallel_all_reduce as tensor_model_parallel_all_reduce,
)
from vllm.distributed.device_communicators.flashinfer_all_reduce import (
    destroy_fi_ar_workspace as destroy_fi_ar_workspace,
    get_fi_ar_quant_workspace as get_fi_ar_quant_workspace,
    get_fi_ar_workspace as get_fi_ar_workspace,
    initialize_fi_ar_quant_workspace as initialize_fi_ar_quant_workspace,
    initialize_fi_ar_workspace as initialize_fi_ar_workspace,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym as kFp8StaticTensorSym,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)

FP8_DTYPE: Incomplete
logger: Incomplete
flashinfer_comm: ModuleType | None
STATIC_FP4_QUANT_OP: Incomplete
FI_ALLREDUCE_FUSION_MAX_SIZE_MB: dict[int, dict[int, float]]
ar_fusion_patterns: Incomplete
MiB: Incomplete

def call_trtllm_fused_allreduce_norm(
    allreduce_in: torch.Tensor,
    residual: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    world_size: int,
    launch_with_pdl: bool,
    fp32_acc: bool,
    max_token_num: int,
    pattern_code: int,
    norm_out: torch.Tensor | None = None,
    quant_out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
    scale_factor: torch.Tensor | None = None,
) -> None: ...
def call_trtllm_fused_allreduce_norm_fake(
    allreduce_in: torch.Tensor,
    residual: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    world_size: int,
    launch_with_pdl: bool,
    fp32_acc: bool,
    max_token_num: int,
    pattern_code: int,
    norm_out: torch.Tensor | None = None,
    quant_out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
    scale_factor: torch.Tensor | None = None,
) -> None: ...

flashinfer_trtllm_fused_allreduce_norm: Incomplete

class FlashInferFusedAllReduceParams:
    world_size: Incomplete
    launch_with_pdl: bool
    fp32_acc: bool
    max_token_num: Incomplete
    def __init__(self, world_size: int, max_token_num: int = 1024) -> None: ...
    def get_trtllm_fused_allreduce_kwargs(self) -> dict[str, bool | int]: ...

class BasePattern:
    dtype: Incomplete
    device: Incomplete
    tp: Incomplete
    tp_size: Incomplete
    def __init__(self, dtype: torch.dtype, device: str | None) -> None: ...

class AllReduceRMSNormPattern(BasePattern):
    epsilon: Incomplete
    allreduce_params: Incomplete
    rmsnorm_matcher: Incomplete
    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
        allreduce_params: FlashInferFusedAllReduceParams,
    ) -> None: ...
    def get_inputs(self) -> list[torch.Tensor]: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class AllReduceFusedAddRMSNormPattern(BasePattern):
    epsilon: Incomplete
    allreduce_params: Incomplete
    rmsnorm_matcher: Incomplete
    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
        allreduce_params: FlashInferFusedAllReduceParams,
    ) -> None: ...
    def get_inputs(self) -> list[torch.Tensor]: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class AllReduceFusedRMSNormStaticQuantFP8Pattern(BasePattern):
    epsilon: Incomplete
    allreduce_params: Incomplete
    quant_dtype: Incomplete
    rmsnorm_matcher: Incomplete
    quant_matcher: Incomplete
    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
        allreduce_params: FlashInferFusedAllReduceParams,
    ) -> None: ...
    def get_inputs(self) -> list[torch.Tensor]: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class AllReduceFusedAddRMSNormStaticQuantFP8Pattern(BasePattern):
    epsilon: Incomplete
    allreduce_params: Incomplete
    quant_dtype: Incomplete
    rmsnorm_matcher: Incomplete
    quant_matcher: Incomplete
    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
        allreduce_params: FlashInferFusedAllReduceParams,
    ) -> None: ...
    def get_inputs(self) -> list[torch.Tensor]: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class AllReduceFusedRMSNormStaticQuantNVFP4Pattern(BasePattern):
    epsilon: Incomplete
    allreduce_params: Incomplete
    rmsnorm_matcher: Incomplete
    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
        allreduce_params: FlashInferFusedAllReduceParams,
    ) -> None: ...
    def get_inputs(self) -> list[torch.Tensor]: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class AllReduceFusedAddRMSNormStaticQuantNVFP4Pattern(BasePattern):
    epsilon: Incomplete
    allreduce_params: Incomplete
    rmsnorm_matcher: Incomplete
    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
        allreduce_params: FlashInferFusedAllReduceParams,
    ) -> None: ...
    def get_inputs(self) -> list[torch.Tensor]: ...
    def register(self, pm_pass: PatternMatcherPass) -> None: ...

class AllReduceFusionPass(VllmPatternMatcherPass):
    disabled: bool
    tp_size: Incomplete
    patterns: PatternMatcherPass
    hidden_dim: Incomplete
    group: Incomplete
    max_token_num: Incomplete
    allreduce_params: Incomplete
    def __init__(self, config: VllmConfig) -> None: ...
    @enable_fake_mode
    def register_patterns(self) -> None: ...
    def is_applicable_for_range(self, compile_range: Range) -> bool: ...
    matched_count: Incomplete
    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None: ...
    def __del__(self) -> None: ...
