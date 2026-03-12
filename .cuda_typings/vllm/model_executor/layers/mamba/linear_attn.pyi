import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from torch import nn
from vllm.config import (
    CacheConfig as CacheConfig,
    ModelConfig as ModelConfig,
    get_current_vllm_config as get_current_vllm_config,
)
from vllm.distributed.communication_op import (
    tensor_model_parallel_all_reduce as tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.forward_context import (
    ForwardContext as ForwardContext,
    get_forward_context as get_forward_context,
)
from vllm.model_executor.custom_op import CustomOp as CustomOp
from vllm.model_executor.layers.lightning_attn import (
    lightning_attention as lightning_attention,
    linear_decode_forward_triton as linear_decode_forward_triton,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase as MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator as MambaStateDtypeCalculator,
    MambaStateShapeCalculator as MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)
from vllm.v1.attention.backend import AttentionMetadata as AttentionMetadata
from vllm.v1.attention.backends.linear_attn import (
    LinearAttentionMetadata as LinearAttentionMetadata,
)

class MiniMaxText01RMSNormTP(CustomOp):
    name: str
    tp_world: Incomplete
    tp_rank: Incomplete
    weight: Incomplete
    variance_epsilon: Incomplete
    def __init__(self, hidden_size: int, eps: float = 1e-06) -> None: ...
    @staticmethod
    def weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None: ...
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    @staticmethod
    def forward_qk(
        q_norm: MiniMaxText01RMSNormTP,
        k_norm: MiniMaxText01RMSNormTP,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

def clear_linear_attention_cache_for_new_sequences(
    kv_cache: torch.Tensor,
    state_indices_tensor: torch.Tensor,
    attn_metadata: LinearAttentionMetadata,
) -> None: ...
def linear_attention_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    slope_rate: torch.Tensor,
    state_indices_tensor: torch.Tensor,
    q_start: int = 0,
    q_end: int | None = None,
    slot_start: int = 0,
    slot_end: int | None = None,
    block_size: int = 32,
) -> torch.Tensor: ...
def linear_attention_prefill_and_mix(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    state_indices_tensor: torch.Tensor,
    attn_metadata: LinearAttentionMetadata,
    slope_rate: torch.Tensor,
    block_size: int,
    decode_fn: Callable[..., torch.Tensor],
    prefix_fn: Callable[..., torch.Tensor],
    layer_idx: int | None = None,
) -> torch.Tensor: ...

class MiniMaxText01LinearKernel:
    @staticmethod
    def jit_linear_forward_prefix(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_caches: torch.Tensor,
        slope_rate: torch.Tensor,
        block_size: int,
        layer_idx: int | None = None,
        **kwargs,
    ) -> torch.Tensor: ...

class MiniMaxText01LinearAttention(nn.Module, MambaBase):
    @property
    def mamba_type(self) -> str: ...
    def get_state_dtype(self) -> tuple[torch.dtype]: ...
    def get_state_shape(self) -> tuple[tuple[int, int, int], ...]: ...
    layer_idx: Incomplete
    BLOCK: Incomplete
    hidden_size: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    total_num_heads: Incomplete
    hidden_inner_size: Incomplete
    tp_size: Incomplete
    tp_rank: Incomplete
    tp_heads: Incomplete
    qkv_size: Incomplete
    tp_hidden: Incomplete
    model_config: Incomplete
    cache_config: Incomplete
    prefix: Incomplete
    qkv_proj: Incomplete
    output_gate: Incomplete
    out_proj: Incomplete
    norm: Incomplete
    slope_rate: Incomplete
    tp_slope: Incomplete
    def __init__(
        self,
        hidden_size: int,
        hidden_inner_size: int,
        num_heads: int,
        head_dim: int,
        max_position: int,
        block_size: int,
        num_hidden_layer: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        layer_idx: int = 0,
        linear_layer_idx: int = 0,
        prefix: str = "linear_attn",
    ) -> None: ...
    @staticmethod
    def weight_direct_load(
        param: torch.Tensor, loaded_weight: torch.Tensor
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, output: torch.Tensor, positions: torch.Tensor
    ) -> None: ...

def linear_attention(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
) -> None: ...
def linear_attention_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
) -> None: ...
