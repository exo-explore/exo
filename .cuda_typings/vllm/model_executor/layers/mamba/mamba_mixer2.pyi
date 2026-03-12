import torch
from _typeshed import Incomplete
from vllm.config import (
    CacheConfig as CacheConfig,
    ModelConfig as ModelConfig,
    get_current_vllm_config as get_current_vllm_config,
)
from vllm.distributed import (
    divide as divide,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather as tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce as tensor_model_parallel_all_reduce,
)
from vllm.forward_context import (
    ForwardContext as ForwardContext,
    get_forward_context as get_forward_context,
)
from vllm.model_executor.custom_op import (
    CustomOp as CustomOp,
    PluggableLayer as PluggableLayer,
)
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
from vllm.model_executor.layers.mamba.ops.layernorm_gated import (
    rms_norm_gated as rms_norm_gated,
)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_state_update as selective_state_update,
)
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen as mamba_chunk_scan_combined_varlen,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.model_loader.weight_utils import (
    LoaderFunction as LoaderFunction,
    composed_weight_loader as composed_weight_loader,
    sharded_weight_loader as sharded_weight_loader,
)
from vllm.model_executor.parameter import BasevLLMParameter as BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs as set_weight_attrs
from vllm.platforms import current_platform as current_platform
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)
from vllm.v1.attention.backend import AttentionMetadata as AttentionMetadata
from vllm.v1.attention.backends.mamba2_attn import (
    Mamba2AttentionMetadata as Mamba2AttentionMetadata,
)

class Mixer2RMSNormGated(CustomOp):
    tp_size: Incomplete
    tp_rank: Incomplete
    full_hidden_size: Incomplete
    group_size: Incomplete
    per_rank_hidden_size: Incomplete
    n_groups: Incomplete
    variance_epsilon: Incomplete
    use_rms_norm: Incomplete
    weight: Incomplete
    def __init__(
        self,
        full_hidden_size: int,
        full_n_groups: int,
        use_rms_norm: bool = True,
        eps: float = 1e-06,
    ) -> None: ...
    def forward_native(self, x: torch.Tensor, gate: torch.Tensor): ...
    def forward_cuda(
        self, x: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

def mamba_v2_sharded_weight_loader(
    shard_spec: list[tuple[int, int, float]], tp_size: int, tp_rank: int
) -> LoaderFunction: ...

class MambaMixer2(MambaBase, PluggableLayer):
    tp_size: Incomplete
    ssm_state_size: Incomplete
    conv_kernel_size: Incomplete
    activation: Incomplete
    intermediate_size: Incomplete
    head_dim: Incomplete
    num_heads: Incomplete
    n_groups: Incomplete
    groups_ssm_state_size: Incomplete
    conv_dim: Incomplete
    conv1d: Incomplete
    in_proj: Incomplete
    A: Incomplete
    D: Incomplete
    dt_bias: Incomplete
    use_rms_norm: Incomplete
    out_proj: Incomplete
    norm: Incomplete
    split_hidden_states_B_C_fn: Incomplete
    kv_cache: Incomplete
    model_config: Incomplete
    cache_config: Incomplete
    prefix: Incomplete
    num_spec: Incomplete
    tped_intermediate_size: Incomplete
    tped_conv_size: Incomplete
    tped_dt_size: Incomplete
    is_blackwell: Incomplete
    def __init__(
        self,
        hidden_size: int,
        ssm_state_size: int,
        conv_kernel_size: int,
        intermediate_size: int,
        use_conv_bias: bool,
        use_bias: bool,
        n_groups: int = 1,
        num_heads: int = 128,
        head_dim: int = 64,
        rms_norm_eps: float = 1e-05,
        activation: str = "silu",
        use_rms_norm: bool = True,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, mup_vector: torch.Tensor | None = None
    ): ...
    def conv_ssm_forward(
        self, projected_states: torch.Tensor, output: torch.Tensor
    ): ...
    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]: ...
    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]: ...
    @property
    def mamba_type(self) -> str: ...

def mamba_mixer2(
    projected_states: torch.Tensor, output: torch.Tensor, layer_name: str
) -> None: ...
def mamba_mixer2_fake(
    projected_states: torch.Tensor, output: torch.Tensor, layer_name: str
) -> None: ...
