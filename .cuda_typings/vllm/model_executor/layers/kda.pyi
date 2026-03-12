import torch
from .fla.ops.kda import (
    FusedRMSNormGated as FusedRMSNormGated,
    chunk_kda as chunk_kda,
    fused_kda_gate as fused_kda_gate,
    fused_recurrent_kda as fused_recurrent_kda,
)
from .linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    ReplicatedLinear as ReplicatedLinear,
    RowParallelLinear as RowParallelLinear,
)
from .mamba.abstract import MambaBase as MambaBase
from .mamba.mamba_utils import (
    MambaStateDtypeCalculator as MambaStateDtypeCalculator,
    MambaStateShapeCalculator as MambaStateShapeCalculator,
)
from .mamba.ops.causal_conv1d import (
    causal_conv1d_fn as causal_conv1d_fn,
    causal_conv1d_update as causal_conv1d_update,
)
from .quantization.base_config import QuantizationConfig as QuantizationConfig
from _typeshed import Incomplete
from torch import nn
from vllm.config import (
    CacheConfig as CacheConfig,
    ModelConfig as ModelConfig,
    get_current_vllm_config as get_current_vllm_config,
)
from vllm.distributed import (
    divide as divide,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.forward_context import (
    ForwardContext as ForwardContext,
    get_forward_context as get_forward_context,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.model_loader.weight_utils import (
    sharded_weight_loader as sharded_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs as set_weight_attrs
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)
from vllm.v1.attention.backend import AttentionMetadata as AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata as GDNAttentionMetadata,
)

logger: Incomplete

def kda_attention(
    q_proj_states: torch.Tensor,
    k_proj_states: torch.Tensor,
    v_proj_states: torch.Tensor,
    g1: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None: ...
def kda_attention_fake(
    q_proj_states: torch.Tensor,
    k_proj_states: torch.Tensor,
    v_proj_states: torch.Tensor,
    g1: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None: ...

class KimiDeltaAttention(nn.Module, MambaBase):
    @property
    def mamba_type(self) -> str: ...
    def get_state_dtype(
        self,
    ) -> tuple[torch.dtype, torch.dtype, torch.dtype, torch.dtype]: ...
    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]: ...
    tp_size: Incomplete
    tp_rank: Incomplete
    hidden_size: Incomplete
    model_config: Incomplete
    cache_config: Incomplete
    head_dim: Incomplete
    num_heads: Incomplete
    layer_idx: Incomplete
    prefix: Incomplete
    local_num_heads: Incomplete
    conv_size: Incomplete
    q_proj: Incomplete
    k_proj: Incomplete
    v_proj: Incomplete
    f_a_proj: Incomplete
    f_b_proj: Incomplete
    dt_bias: Incomplete
    b_proj: Incomplete
    q_conv1d: Incomplete
    k_conv1d: Incomplete
    v_conv1d: Incomplete
    A_log: Incomplete
    g_a_proj: Incomplete
    g_b_proj: Incomplete
    o_norm: Incomplete
    o_proj: Incomplete
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        model_config: ModelConfig | None = None,
        rms_norm_eps: float = 1e-05,
        prefix: str = "",
        **kwargs,
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, output: torch.Tensor
    ) -> None: ...
