import abc
import torch
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    PPMissingLayer as PPMissingLayer,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed.parallel_state import get_pp_group as get_pp_group
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE as FusedMoE
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer as Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm as Qwen3_5RMSNorm,
)
from vllm.model_executor.models.qwen3_next import (
    QwenNextMixtureOfExperts as QwenNextMixtureOfExperts,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.configs.qwen3_5 import (
    Qwen3_5TextConfig as Qwen3_5TextConfig,
)
from vllm.transformers_utils.configs.qwen3_5_moe import (
    Qwen3_5MoeTextConfig as Qwen3_5MoeTextConfig,
)

logger: Incomplete

class Qwen3_5MultiTokenPredictor(nn.Module):
    config: Incomplete
    vocab_size: Incomplete
    mtp_start_layer_idx: Incomplete
    num_mtp_layers: Incomplete
    embed_tokens: Incomplete
    fc: Incomplete
    layers: Incomplete
    make_empty_intermediate_tensors: Incomplete
    norm: Incomplete
    pre_fc_norm_hidden: Incomplete
    pre_fc_norm_embedding: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor: ...
    def load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Qwen3_5MTP(nn.Module, SupportsMultiModal, metaclass=abc.ABCMeta):
    packed_modules_mapping: Incomplete
    vllm_config: Incomplete
    quant_config: Incomplete
    config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ): ...
    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Qwen3_5MoeMTP(Qwen3_5MTP, QwenNextMixtureOfExperts, metaclass=abc.ABCMeta):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
