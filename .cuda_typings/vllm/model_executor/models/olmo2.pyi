import abc
import torch
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed import (
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.distributed.communication_op import (
    tensor_model_parallel_all_gather as tensor_model_parallel_all_gather,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
)
from vllm.distributed.utils import (
    split_tensor_along_last_dim as split_tensor_along_last_dim,
)
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear as MergedColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.rotary_embedding import get_rope as get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.interfaces import (
    SupportsLoRA as SupportsLoRA,
    SupportsPP as SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    extract_layer_index as extract_layer_index,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.configs import Olmo3Config as Olmo3Config

class Olmo2Attention(nn.Module):
    config: Incomplete
    tp_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    total_num_kv_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    max_position_embeddings: Incomplete
    qkv_proj: Incomplete
    tp_rank: Incomplete
    k_norm: Incomplete
    q_norm: Incomplete
    scaling: Incomplete
    attn: Incomplete
    rotary_emb: Incomplete
    o_proj: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class Olmo2MLP(nn.Module):
    gate_up_proj: Incomplete
    act_fn: Incomplete
    down_proj: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Olmo2DecoderLayer(nn.Module):
    self_attn: Incomplete
    mlp: Incomplete
    post_attention_layernorm: Incomplete
    post_feedforward_layernorm: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor: ...

class Olmo2Model(nn.Module):
    config: Incomplete
    embed_tokens: Incomplete
    norm: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Olmo2ForCausalLM(nn.Module, SupportsPP, SupportsLoRA, metaclass=abc.ABCMeta):
    packed_modules_mapping: Incomplete
    config: Incomplete
    model: Incomplete
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
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
