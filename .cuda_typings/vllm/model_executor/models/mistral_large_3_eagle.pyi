import abc
import torch
from .interfaces import SupportsMultiModal as SupportsMultiModal
from .utils import (
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed.parallel_state import get_pp_group as get_pp_group
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import RowParallelLinear as RowParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2DecoderLayer as DeepseekV2DecoderLayer,
    DeepseekV2Model as DeepseekV2Model,
)
from vllm.model_executor.models.mistral_large_3 import (
    MistralLarge3ForCausalLM as MistralLarge3ForCausalLM,
)

logger: Incomplete

class EagleMistralLarge3Model(DeepseekV2Model):
    config: Incomplete
    vllm_config: Incomplete
    vocab_size: Incomplete
    embed_tokens: Incomplete
    layers: Incomplete
    start_layer: int
    end_layer: Incomplete
    fc: Incomplete
    norm: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = "", start_layer_id: int = 0
    ) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class EagleMistralLarge3ForCausalLM(MistralLarge3ForCausalLM, metaclass=abc.ABCMeta):
    remapping: Incomplete
    quant_config: Incomplete
    model_cls: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def get_language_model(self) -> torch.nn.Module: ...
    embed_input_ids: Incomplete
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
