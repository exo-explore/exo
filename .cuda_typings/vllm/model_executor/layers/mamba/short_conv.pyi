import torch
from _typeshed import Incomplete
from vllm.config import (
    CacheConfig as CacheConfig,
    ModelConfig as ModelConfig,
    get_current_vllm_config as get_current_vllm_config,
)
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.forward_context import (
    ForwardContext as ForwardContext,
    get_forward_context as get_forward_context,
)
from vllm.model_executor.custom_op import CustomOp as CustomOp
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    MergedColumnParallelLinear as MergedColumnParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase as MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator as MambaStateDtypeCalculator,
    MambaStateShapeCalculator as MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn as causal_conv1d_fn,
    causal_conv1d_update as causal_conv1d_update,
)
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)
from vllm.v1.attention.backend import AttentionMetadata as AttentionMetadata
from vllm.v1.attention.backends.short_conv_attn import (
    ShortConvAttentionMetadata as ShortConvAttentionMetadata,
)

class ShortConv(MambaBase, CustomOp):
    config: Incomplete
    layer_idx: Incomplete
    conv_dim: Incomplete
    L_cache: Incomplete
    bias: Incomplete
    conv: Incomplete
    in_proj: Incomplete
    out_proj: Incomplete
    kv_cache: Incomplete
    model_config: Incomplete
    cache_config: Incomplete
    prefix: Incomplete
    def __init__(
        self,
        config,
        dim: int,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward_native(self, hidden_states: torch.Tensor, output: torch.Tensor): ...
    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor): ...
    def forward_cuda(self, hidden_states: torch.Tensor, output: torch.Tensor): ...
    def get_state_dtype(self) -> tuple[torch.dtype, ...]: ...
    def get_state_shape(self) -> tuple[tuple[int, ...]]: ...
    @property
    def mamba_type(self) -> str: ...

def short_conv(
    hidden_states: torch.Tensor, output: torch.Tensor, layer_name: str
) -> None: ...
def short_conv_fake(
    hidden_states: torch.Tensor, output: torch.Tensor, layer_name: str
) -> None: ...
