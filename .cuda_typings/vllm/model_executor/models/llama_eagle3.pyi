import abc
import torch
import torch.nn as nn
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    get_draft_quant_config as get_draft_quant_config,
    maybe_prefix as maybe_prefix,
    process_eagle_weight as process_eagle_weight,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from transformers import LlamaConfig as LlamaConfig
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import (
    VllmConfig as VllmConfig,
    get_current_vllm_config as get_current_vllm_config,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear as QKVParallelLinear,
    ReplicatedLinear as ReplicatedLinear,
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
    maybe_remap_kv_scale_name as maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.llama import (
    LlamaDecoderLayer as LlamaDecoderLayer,
    LlamaForCausalLM as LlamaForCausalLM,
)
from vllm.multimodal.inputs import NestedTensors as NestedTensors

logger: Incomplete

class LlamaDecoderLayer(LlamaDecoderLayer):
    hidden_norm: Incomplete
    layer_idx: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: LlamaConfig | None = None,
        layer_idx: int = 0,
    ) -> None: ...
    def get_quant_config(
        self, vllm_config: VllmConfig
    ) -> QuantizationConfig | None: ...
    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class LlamaModel(nn.Module):
    config: Incomplete
    vocab_size: Incomplete
    quant_config: Incomplete
    use_aux_hidden_state: Incomplete
    embed_tokens: Incomplete
    layers: Incomplete
    fc: Incomplete
    norm: Incomplete
    def __init__(
        self, *, vllm_config: VllmConfig, start_layer_id: int = 0, prefix: str = ""
    ) -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Eagle3LlamaForCausalLM(LlamaForCausalLM, metaclass=abc.ABCMeta):
    config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    draft_id_to_target_id: Incomplete
    use_parallel_drafting: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]): ...
