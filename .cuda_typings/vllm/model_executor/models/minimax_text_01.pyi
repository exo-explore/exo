import torch
from .interfaces import HasInnerState as HasInnerState, IsHybrid as IsHybrid
from .utils import (
    PPMissingLayer as PPMissingLayer,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_layers as make_layers,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers import MiniMaxConfig as MiniMaxConfig
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import (
    CacheConfig as CacheConfig,
    ModelConfig as ModelConfig,
    VllmConfig as VllmConfig,
)
from vllm.distributed.parallel_state import (
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context as get_forward_context
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.fused_moe import FusedMoE as FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear as MergedColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    ReplicatedLinear as ReplicatedLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.mamba.linear_attn import (
    MiniMaxText01LinearAttention as MiniMaxText01LinearAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc as MambaStateCopyFunc,
    MambaStateCopyFuncCalculator as MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator as MambaStateDtypeCalculator,
    MambaStateShapeCalculator as MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.rotary_embedding import get_rope as get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.utils import maybe_prefix as maybe_prefix
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.v1.attention.backend import AttentionMetadata as AttentionMetadata

def replace_weight_name(
    name: str, key: str = None, to: str = None, count: int = None, prefix: str = None
) -> str: ...
def weight_loader_with_alias(alias: str): ...

class MiniMaxText01MLP(nn.Module):
    layer_idx: Incomplete
    gate_up_proj: Incomplete
    down_proj: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        layer_idx: int = None,
        prefix: str = "mlp",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class MiniMaxText01MoE(nn.Module):
    layer_idx: Incomplete
    tp_size: Incomplete
    num_total_experts: Incomplete
    top_k: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    quant_config: Incomplete
    params_dtype: Incomplete
    gate: Incomplete
    experts: Incomplete
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype | None = None,
        layer_idx: int = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "moe",
    ) -> None: ...
    @staticmethod
    def gate_weight_loader(
        param: nn.Parameter, loaded_weight: torch.Tensor
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class MiniMaxText01Attention(nn.Module):
    layer_idx: Incomplete
    hidden_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    total_num_kv_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    scaling: Incomplete
    sliding_window: Incomplete
    prefix: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    attn: Incomplete
    rotary_emb: Incomplete
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        max_position: int = ...,
        rope_parameters: dict | None = None,
        sliding_window: int | None = None,
        quant_config: QuantizationConfig | None = None,
        layer_idx: int = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "mha",
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        positions: torch.Tensor,
        **kwargs,
    ) -> None: ...

class MiniMaxText01DecoderLayer(nn.Module):
    prefix: Incomplete
    hidden_size: Incomplete
    expert_num: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    block_sparse_moe: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    layernorm_attention_alpha: Incomplete
    layernorm_attention_beta: Incomplete
    layernorm_mlp_alpha: Incomplete
    layernorm_mlp_beta: Incomplete
    postnorm: Incomplete
    shared_moe: bool
    shared_mlp: Incomplete
    coefficient: Incomplete
    shared_moe_mode: Incomplete
    def __init__(
        self,
        config: MiniMaxConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        expert_num: int = 1,
        layer_id: int = None,
        linear_layer_id: int | None = None,
        prefix: str = "decoder",
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: torch.Tensor | None,
        is_warmup: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @staticmethod
    def shared_moe_coefficient_loader(
        param: torch.Tensor, loaded_weight: torch.Tensor
    ) -> None: ...

class MiniMaxText01Model(nn.Module):
    vocab_size: Incomplete
    decoder_attention_types: Incomplete
    num_layers: Incomplete
    embed_tokens: Incomplete
    cache_shape: Incomplete
    norm: Incomplete
    embed_scale: float
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors: ...

class MiniMaxText01ForCausalLM(nn.Module, HasInnerState, IsHybrid):
    config: Incomplete
    CONCAT_FFN: bool
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    kv_cache: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs): ...
    def get_seqlen_agnostic_capture_inputs(self, batch_size: int): ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]: ...
    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...], ...]: ...
    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc]: ...
