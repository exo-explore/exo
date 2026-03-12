import abc
import torch
from .interfaces import SupportsLoRA as SupportsLoRA, SupportsPP as SupportsPP
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
from transformers.configuration_utils import PretrainedConfig as PretrainedConfig
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.distributed import (
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.fused_moe import SharedFusedMoE as SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear as MergedColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
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

class BailingAttention(nn.Module):
    hidden_size: Incomplete
    total_num_heads: Incomplete
    total_kv_heads: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    q_size_per_rank: Incomplete
    num_kv_heads: Incomplete
    kv_size_per_rank: Incomplete
    scale: Incomplete
    use_qk_norm: Incomplete
    use_rmsnorm: Incomplete
    query_key_value: Incomplete
    query_layernorm: Incomplete
    key_layernorm: Incomplete
    dense: Incomplete
    rotary_emb: Incomplete
    attn: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor: ...

class BailingMLP(nn.Module):
    gate_up_proj: Incomplete
    down_proj: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        intermediate_size: int,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool | None = True,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x): ...

class BailingMoE(nn.Module):
    tp_size: Incomplete
    tp_rank: Incomplete
    num_experts: Incomplete
    top_k: Incomplete
    norm_expert_prob: Incomplete
    hidden_size: Incomplete
    quant_config: Incomplete
    num_shared_experts: Incomplete
    score_function: Incomplete
    n_group: Incomplete
    topk_group: Incomplete
    use_grouped_topk: Incomplete
    routed_scaling_factor: Incomplete
    router_dtype: Incomplete
    gate: Incomplete
    correction_bias: Incomplete
    shared_experts: Incomplete
    experts: Incomplete
    def __init__(
        self,
        intermediate_size: int,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool | None = True,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class BailingMoeBlock(nn.Module):
    config: Incomplete
    input_layernorm: Incomplete
    attention: Incomplete
    post_attention_layernorm: Incomplete
    mlp: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor: ...

class BailingMoeModel(nn.Module):
    config: Incomplete
    vocab_size: Incomplete
    embed_dim: Incomplete
    tie_word_embeddings: Incomplete
    word_embeddings: Incomplete
    embedding_dropout: Incomplete
    make_empty_intermediate_tensors: Incomplete
    norm: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class BailingMoeForCausalLM(nn.Module, SupportsPP, SupportsLoRA, metaclass=abc.ABCMeta):
    packed_modules_mapping: Incomplete
    config: Incomplete
    quant_config: Incomplete
    max_position_embeddings: Incomplete
    model: Incomplete
    tie_word_embeddings: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...

class BailingMoeV2ForCausalLM(BailingMoeForCausalLM, metaclass=abc.ABCMeta): ...
