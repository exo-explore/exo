import torch
from .deepep_ht_prepare_finalize import (
    DeepEPHTPrepareAndFinalize as DeepEPHTPrepareAndFinalize,
)
from .deepep_ll_prepare_finalize import (
    DEEPEP_QUANT_BLOCK_SHAPE as DEEPEP_QUANT_BLOCK_SHAPE,
    DeepEPLLPrepareAndFinalize as DeepEPLLPrepareAndFinalize,
)
from .mori_prepare_finalize import MoriPrepareAndFinalize as MoriPrepareAndFinalize
from _typeshed import Incomplete
from vllm.distributed import get_ep_group as get_ep_group
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEParallelConfig as FusedMoEParallelConfig,
    FusedMoEQuantConfig as FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.flashinfer_a2a_prepare_finalize import (
    FlashInferA2APrepareAndFinalize as FlashInferA2APrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPrepareAndFinalize as FusedMoEPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    make_moe_prepare_and_finalize_naive_dp_ep as make_moe_prepare_and_finalize_naive_dp_ep,
    make_moe_prepare_and_finalize_no_dp_ep as make_moe_prepare_and_finalize_no_dp_ep,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.import_utils import has_deep_ep as has_deep_ep, has_mori as has_mori

logger: Incomplete

def maybe_roundup_layer_hidden_size(
    hidden_size: int,
    act_dtype: torch.dtype,
    moe_parallel_config: FusedMoEParallelConfig,
) -> int: ...
def maybe_make_prepare_finalize(
    moe: FusedMoEConfig,
    quant_config: FusedMoEQuantConfig | None,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    allow_new_interface: bool = False,
    use_monolithic: bool = False,
) -> FusedMoEPrepareAndFinalize | None: ...
