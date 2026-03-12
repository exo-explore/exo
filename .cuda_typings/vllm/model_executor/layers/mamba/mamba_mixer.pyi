import torch
from _typeshed import Incomplete
from typing import NamedTuple
from vllm.config import (
    CacheConfig as CacheConfig,
    ModelConfig as ModelConfig,
    get_current_vllm_config as get_current_vllm_config,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.forward_context import (
    ForwardContext as ForwardContext,
    get_forward_context as get_forward_context,
)
from vllm.model_executor.custom_op import PluggableLayer as PluggableLayer
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
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
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_scan_fn as selective_scan_fn,
    selective_state_update as selective_state_update,
)
from vllm.model_executor.utils import set_weight_attrs as set_weight_attrs
from vllm.platforms import current_platform as current_platform
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)
from vllm.v1.attention.backends.mamba1_attn import (
    Mamba1AttentionMetadata as Mamba1AttentionMetadata,
)

class MambaMixer(MambaBase, PluggableLayer):
    time_step_rank: Incomplete
    ssm_state_size: Incomplete
    use_rms_norm: Incomplete
    activation: Incomplete
    is_lora_enabled: Incomplete
    conv_kernel_size: Incomplete
    intermediate_size: Incomplete
    conv1d: Incomplete
    in_proj: Incomplete
    x_proj: Incomplete
    dt_proj: Incomplete
    A: Incomplete
    D: Incomplete
    out_proj: Incomplete
    dt_layernorm: Incomplete
    b_layernorm: Incomplete
    c_layernorm: Incomplete
    kv_cache: Incomplete
    model_config: Incomplete
    cache_config: Incomplete
    prefix: Incomplete
    def __init__(
        self,
        hidden_size: int,
        ssm_state_size: int,
        conv_kernel_size: int,
        intermediate_size: int,
        time_step_rank: int,
        use_conv_bias: bool,
        use_bias: bool,
        use_rms_norm: bool,
        rms_norm_has_weight: bool = True,
        rms_norm_eps: float = 1e-05,
        activation: str = "silu",
        is_lora_enabled: bool = False,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor): ...
    def forward_impl(self, hidden_states: torch.Tensor, output: torch.Tensor): ...
    def get_state_dtype(self) -> tuple[torch.dtype]: ...
    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]: ...
    @property
    def mamba_type(self) -> str: ...

class PrefillDecodeSplit(NamedTuple):
    hidden_states_BC_p: torch.Tensor
    hidden_states_BC_d: torch.Tensor
    gate_p: torch.Tensor
    gate_d: torch.Tensor

def split_batch_to_prefill_and_decode(
    hidden_states_BC: torch.Tensor,
    gate: torch.Tensor,
    num_prefill_tokens: int,
    num_decode_tokens: int,
) -> PrefillDecodeSplit: ...
def mamba_mixer(
    hidden_states: torch.Tensor, output: torch.Tensor, layer_name: str
) -> None: ...
def mamba_mixer_fake(
    hidden_states: torch.Tensor, output: torch.Tensor, layer_name: str
) -> None: ...
