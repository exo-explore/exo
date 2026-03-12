import abc
import torch
import torch.nn as nn
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .siglip import SiglipVisionModel as SiglipVisionModel
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    StageMissingLayer as StageMissingLayer,
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from typing import Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems as MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    PromptReplacement as PromptReplacement,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.processors.bagel import BagelProcessor as BagelProcessor
from vllm.utils.tensor_schema import TensorSchema as TensorSchema

logger: Incomplete

class BagelImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor

BagelImageInputs: TypeAlias = BagelImagePixelInputs

class BagelVisionMLP(nn.Module):
    fc1: Incomplete
    act: Incomplete
    fc2: Incomplete
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: str = "gelu_pytorch_tanh",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class PositionEmbedding(nn.Module):
    max_num_patch_per_side: Incomplete
    hidden_size: Incomplete
    def __init__(self, max_num_patch_per_side: int, hidden_size: int) -> None: ...
    def forward(self, position_ids: torch.Tensor) -> torch.Tensor: ...

class BagelProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> BagelProcessor: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]: ...
    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int: ...

class BagelDummyInputsBuilder(BaseDummyInputsBuilder[BagelProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class BagelMultiModalProcessor(BaseMultiModalProcessor[BagelProcessingInfo]): ...

class BagelForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsLoRA, SupportsPP, metaclass=abc.ABCMeta
):
    hf_to_vllm_mapper: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multimodal_config: Incomplete
    language_model: Incomplete
    vit_model: Incomplete
    connector: Incomplete
    vit_pos_embed: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
