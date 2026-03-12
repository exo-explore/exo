import abc
import torch
import torch.nn as nn
from .interfaces import SupportsPP as SupportsPP
from .nemotron_h import (
    NemotronHAttentionDecoderLayer as NemotronHAttentionDecoderLayer,
    NemotronHMoEDecoderLayer as NemotronHMoEDecoderLayer,
)
from _typeshed import Incomplete
from collections.abc import Callable, Iterable
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import (
    CacheConfig as CacheConfig,
    ModelConfig as ModelConfig,
    VllmConfig as VllmConfig,
)
from vllm.config.parallel import ParallelConfig as ParallelConfig
from vllm.model_executor.layers.fused_moe import FusedMoE as FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
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
from vllm.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    maybe_prefix as maybe_prefix,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.configs import NemotronHConfig as NemotronHConfig

class NemotronHMTPAttentionDecoderLayer(NemotronHAttentionDecoderLayer):
    has_start_projections: Incomplete
    has_end_norm: Incomplete
    enorm: Incomplete
    hnorm: Incomplete
    eh_proj: Incomplete
    final_layernorm: Incomplete
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
        has_start_projections: bool = False,
        has_end_norm: bool = False,
    ) -> None: ...
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class NemotronHMTPMoEDecoderLayer(NemotronHMoEDecoderLayer):
    has_start_projections: Incomplete
    has_end_norm: Incomplete
    enorm: Incomplete
    hnorm: Incomplete
    eh_proj: Incomplete
    final_layernorm: Incomplete
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
        has_start_projections: bool = False,
        has_end_norm: bool = False,
    ) -> None: ...
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class NemotronHMultiTokenPredictor(nn.Module):
    config: Incomplete
    vocab_size: Incomplete
    org_vocab_size: Incomplete
    mtp_start_layer_idx: Incomplete
    num_mtp_layers: Incomplete
    pattern_str: Incomplete
    pattern_len: Incomplete
    embed_tokens: Incomplete
    layers: Incomplete
    make_empty_intermediate_tensors: Callable[..., IntermediateTensors]
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...

class NemotronHMTP(nn.Module, SupportsPP, metaclass=abc.ABCMeta):
    packed_modules_mapping: Incomplete
    vllm_config: Incomplete
    config: Incomplete
    quant_config: Incomplete
    mtp_start_layer_idx: Incomplete
    num_redundant_experts: int
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
