import abc
import torch
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
    SupportsMultiModal as SupportsMultiModal,
)
from .llama import LlamaModel as LlamaModel
from .utils import AutoWeightsLoader as AutoWeightsLoader, maybe_prefix as maybe_prefix
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from torch import nn
from transformers import (
    Idefics3Config as Idefics3Config,
    Idefics3ImageProcessor as Idefics3ImageProcessor,
    Idefics3Processor,
)
from typing import Annotated, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.model_executor.layers.linear import ReplicatedLinear as ReplicatedLinear
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageProcessorItems as ImageProcessorItems,
    MultiModalDataItems as MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

class Idefics3ImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, None]
    pixel_attention_mask: Annotated[torch.Tensor, None]
    num_patches: Annotated[torch.Tensor, None]

class Idefics3ImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor, None]

ImageInputs: TypeAlias = Idefics3ImagePixelInputs | Idefics3ImageEmbeddingInputs

class Idefics3ProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> Idefics3Processor: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_num_patches(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Idefics3Processor,
        mm_kwargs: Mapping[str, object],
    ) -> int: ...
    def get_image_repl(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Idefics3Processor,
        mm_kwargs: Mapping[str, object],
    ) -> str: ...
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Idefics3Processor,
        mm_kwargs: Mapping[str, object],
    ) -> int: ...

class Idefics3DummyInputsBuilder(BaseDummyInputsBuilder[Idefics3ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class Idefics3MultiModalProcessor(BaseMultiModalProcessor[Idefics3ProcessingInfo]): ...

class Idefics3SimpleMLP(nn.Module):
    proj: Incomplete
    def __init__(
        self,
        config: Idefics3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Idefics3Connector(nn.Module):
    scale_factor: Incomplete
    modality_projection: Incomplete
    def __init__(
        self,
        config: Idefics3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def pixel_shuffle(self, x: torch.Tensor, scale_factor: int = 2) -> torch.Tensor: ...
    def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor: ...

class Idefics3Model(nn.Module):
    config: Incomplete
    vocab_size: Incomplete
    vision_model: Incomplete
    connector: Incomplete
    text_model: Incomplete
    image_seq_len: Incomplete
    image_token_id: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def image_pixels_to_features(
        self, pixel_values: torch.Tensor, pixel_attention_mask: torch.Tensor
    ) -> torch.Tensor: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...

class Idefics3ForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsLoRA, metaclass=abc.ABCMeta
):
    packed_modules_mapping: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multimodal_config: Incomplete
    model: Incomplete
    image_token_id: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
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
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def get_mm_mapping(self) -> MultiModelKeys: ...
    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int: ...
    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int: ...
