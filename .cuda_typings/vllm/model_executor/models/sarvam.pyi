import abc
import torch
from .bailing_moe import BailingMoeForCausalLM as BailingMoeForCausalLM
from .interfaces import (
    MixtureOfExperts as MixtureOfExperts,
    SupportsLoRA as SupportsLoRA,
    SupportsPP as SupportsPP,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    PPMissingLayer as PPMissingLayer,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from vllm.config import (
    CacheConfig as CacheConfig,
    ParallelConfig as ParallelConfig,
    VllmConfig as VllmConfig,
)
from vllm.distributed import (
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.fused_moe import SharedFusedMoE as SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    MergedColumnParallelLinear as MergedColumnParallelLinear,
    ReplicatedLinear as ReplicatedLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.mla import (
    MLAModules as MLAModules,
    MultiHeadLatentAttentionWrapper as MultiHeadLatentAttentionWrapper,
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
from vllm.sequence import IntermediateTensors as IntermediateTensors

def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float: ...

class SarvamMLAAttention(nn.Module):
    config: Incomplete
    hidden_size: Incomplete
    qk_nope_head_dim: Incomplete
    qk_rope_head_dim: Incomplete
    qk_head_dim: Incomplete
    v_head_dim: Incomplete
    q_lora_rank: Incomplete
    kv_lora_rank: Incomplete
    total_num_heads: Incomplete
    num_local_heads: Incomplete
    scaling: Incomplete
    max_position_embeddings: Incomplete
    q_a_proj: Incomplete
    q_a_layernorm: Incomplete
    q_b_proj: Incomplete
    q_proj: Incomplete
    kv_a_proj_with_mqa: Incomplete
    kv_a_layernorm: Incomplete
    kv_b_proj: Incomplete
    o_proj: Incomplete
    rotary_emb: Incomplete
    mla_attn: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class SarvamMLAMLP(nn.Module):
    gate_up_proj: Incomplete
    down_proj: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        intermediate_size: int,
        config,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class SarvamMLAMoE(nn.Module):
    config: Incomplete
    tp_size: Incomplete
    tp_rank: Incomplete
    hidden_size: Incomplete
    num_experts: Incomplete
    top_k: Incomplete
    routed_scaling_factor: Incomplete
    n_group: Incomplete
    topk_group: Incomplete
    use_grouped_topk: Incomplete
    norm_expert_prob: Incomplete
    router_dtype: Incomplete
    gate: Incomplete
    score_function: Incomplete
    num_shared_experts: Incomplete
    shared_experts: Incomplete
    experts: Incomplete
    def __init__(
        self,
        config,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def maybe_get_fused_moe(self) -> SharedFusedMoE: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class SarvamMLABlock(nn.Module):
    input_layernorm: Incomplete
    self_attn: Incomplete
    post_attention_layernorm: Incomplete
    mlp: Incomplete
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class SarvamMLAModel(nn.Module):
    config: Incomplete
    vocab_size: Incomplete
    embed_dim: Incomplete
    tie_word_embeddings: Incomplete
    embed_tokens: Incomplete
    embedding_dropout: Incomplete
    make_empty_intermediate_tensors: Incomplete
    norm: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class SarvamMixtureOfExperts(MixtureOfExperts, metaclass=abc.ABCMeta):
    num_logical_experts: Incomplete
    num_routed_experts: Incomplete
    num_shared_experts: Incomplete
    num_physical_experts: Incomplete
    num_local_physical_experts: Incomplete
    num_redundant_experts: int
    def extract_moe_parameters(self, example_moe: SarvamMLAMoE | None) -> None: ...
    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ) -> None: ...
    eplb_state: Incomplete
    def set_eplb_state(self, eplb_state) -> None: ...

class SarvamMLAForCausalLM(
    nn.Module, SupportsPP, SupportsLoRA, SarvamMixtureOfExperts, metaclass=abc.ABCMeta
):
    packed_modules_mapping: Incomplete
    config: Incomplete
    quant_config: Incomplete
    model: Incomplete
    tie_word_embeddings: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    expert_weights: Incomplete
    num_moe_layers: int
    moe_layers: Incomplete
    moe_mlp_layers: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...

class SarvamMoEForCausalLM(BailingMoeForCausalLM, metaclass=abc.ABCMeta):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
