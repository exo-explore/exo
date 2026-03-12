import abc
import torch
from .interfaces import SupportsPP as SupportsPP
from .utils import (
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    make_layers as make_layers,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.distributed import (
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.configs import JAISConfig as JAISConfig

class SwiGLUActivation(nn.Module):
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor: ...

class JAISAttention(nn.Module):
    hidden_size: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    attn_scale_power: Incomplete
    scale: Incomplete
    c_attn: Incomplete
    c_proj: Incomplete
    attn: Incomplete
    def __init__(
        self,
        config: JAISConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class JAISMLP(nn.Module):
    swiglu: Incomplete
    c_fc: Incomplete
    c_fc2: Incomplete
    c_proj: Incomplete
    act: Incomplete
    def __init__(
        self,
        intermediate_size: int,
        config: JAISConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class JAISBlock(nn.Module):
    ln_1: Incomplete
    attn: Incomplete
    ln_2: Incomplete
    mlp: Incomplete
    def __init__(
        self,
        config: JAISConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class JAISModel(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    wte: Incomplete
    wpe: Incomplete
    embeddings_scale: Incomplete
    ln_f: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> IntermediateTensors | torch.Tensor: ...

class JAISLMHeadModel(nn.Module, SupportsPP, metaclass=abc.ABCMeta):
    config: Incomplete
    quant_config: Incomplete
    transformer: Incomplete
    lm_head: Incomplete
    output_logits_scale: Incomplete
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
    ) -> IntermediateTensors | torch.Tensor: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
