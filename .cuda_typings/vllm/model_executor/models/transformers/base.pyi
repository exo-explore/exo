import abc
import torch
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from transformers import PreTrainedModel as PreTrainedModel
from vllm.config import VllmConfig as VllmConfig
from vllm.config.utils import getattr_iter as getattr_iter
from vllm.distributed import get_pp_group as get_pp_group, get_tp_group as get_tp_group
from vllm.distributed.utils import get_pp_indices as get_pp_indices
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention import (
    Attention as Attention,
    EncoderOnlyAttention as EncoderOnlyAttention,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    SupportsEagle as SupportsEagle,
    SupportsEagle3 as SupportsEagle3,
    SupportsLoRA as SupportsLoRA,
    SupportsPP as SupportsPP,
    SupportsQuant as SupportsQuant,
)
from vllm.model_executor.models.interfaces_base import VllmModel as VllmModel
from vllm.model_executor.models.transformers.utils import (
    get_feature_request_tip as get_feature_request_tip,
    init_on_device_without_buffers as init_on_device_without_buffers,
    log_replacement as log_replacement,
    replace_conv_class as replace_conv_class,
    replace_linear_class as replace_linear_class,
    replace_rms_norm_class as replace_rms_norm_class,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    PPMissingLayer as PPMissingLayer,
    WeightsMapper as WeightsMapper,
    make_empty_intermediate_tensors_factory as make_empty_intermediate_tensors_factory,
    maybe_prefix as maybe_prefix,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.v1.attention.backend import AttentionType as AttentionType

logger: Incomplete

def vllm_flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float | None = None,
    attention_instances: dict[int, Attention] | None = None,
    **kwargs,
): ...

class Base(
    nn.Module,
    VllmModel,
    SupportsQuant,
    SupportsLoRA,
    SupportsPP,
    SupportsEagle,
    SupportsEagle3,
    metaclass=abc.ABCMeta,
):
    embedding_modules: Incomplete
    hf_to_vllm_mapper: Incomplete
    def __init_subclass__(cls, *args, **kwargs) -> None: ...
    config: Incomplete
    text_config: Incomplete
    cache_config: Incomplete
    device_config: Incomplete
    model_config: Incomplete
    parallel_config: Incomplete
    quant_config: Incomplete
    pp_group: Incomplete
    tp_group: Incomplete
    skip_prefixes: list[str]
    skip_substrs: list[str]
    ignore_unexpected_prefixes: list[str]
    ignore_unexpected_suffixes: list[str]
    model: PreTrainedModel
    attention_instances: Incomplete
    embed_scale: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def pipeline_parallel(self) -> None: ...
    def recursive_replace(self) -> None: ...
    def create_attention_instances(self) -> dict[int, Attention]: ...
    def init_parameters(self, module: nn.Module, dtype: torch.dtype | None = None): ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    @staticmethod
    def check_version(min_version: str, feature: str): ...
    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None: ...
    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]: ...
