import abc
import torch
import torch.nn as nn
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from torch import Tensor as Tensor
from transformers import PretrainedConfig as PretrainedConfig
from typing import Annotated, Literal
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.model_executor.layers.linear import ReplicatedLinear as ReplicatedLinear
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.models.aimv2 import AIMv2Model as AIMv2Model
from vllm.model_executor.models.siglip import SiglipVisionModel as SiglipVisionModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    flatten_bn as flatten_bn,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageSize as ImageSize,
    MultiModalDataItems as MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    PromptReplacement as PromptReplacement,
)
from vllm.renderers import TokenizeParams as TokenizeParams
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.processors.ovis import OvisProcessor as OvisProcessor
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

IMAGE_TOKEN: str
IMAGE_INDICATOR_IDS: Incomplete
IMAGE_PAD_TOKEN_MAP: Incomplete
IMAGE_PAD_TOKEN_ID_MAP: Incomplete

def st_argmax(y_soft: torch.Tensor, dim: int): ...

class VisualTokenizer(torch.nn.Module):
    config: Incomplete
    backbone: Incomplete
    head: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def device(self) -> torch.device: ...
    def tokenize(self, logits: torch.Tensor) -> torch.Tensor: ...
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor: ...
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor: ...

class OvisImagePatchInputs(TensorSchema):
    type: Literal["image_patches"]
    flat_data: Annotated[torch.Tensor, None]
    indicator_tokens: Annotated[torch.Tensor, None]
    patches_per_image: Annotated[list[int], None]

class VisualEmbedding(torch.nn.Embedding):
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(self, visual_tokens: Tensor) -> Tensor: ...
    @property
    def device(self): ...
    @property
    def dtype(self): ...

class OvisProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object): ...
    def get_default_tok_params(self) -> TokenizeParams: ...
    def get_image_segment_len(self) -> int: ...
    def get_image_pad_token(self) -> str: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...

class OvisDummyInputsBuilder(BaseDummyInputsBuilder[OvisProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class OvisMultiModalProcessor(BaseMultiModalProcessor[OvisProcessingInfo]):
    def image_indicators_to_visual_tokens(
        self, image_indicators: list[int]
    ) -> list[int]: ...

class Ovis(nn.Module, SupportsMultiModal, SupportsPP, metaclass=abc.ABCMeta):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: PretrainedConfig
    llm: Incomplete
    visual_tokenizer: Incomplete
    vte: Incomplete
    image_pad_token_id: Incomplete
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
