import abc
import torch
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLateInteraction as SupportsLateInteraction,
    SupportsMultiModal as SupportsMultiModal,
)
from .interfaces_base import default_pooling_type as default_pooling_type
from .modernbert import (
    ModernBertEmbeddings as ModernBertEmbeddings,
    ModernBertLayer as ModernBertLayer,
)
from .siglip import SiglipVisionModel as SiglipVisionModel
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from torch import nn
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.model_executor.layers.pooler.tokwise import (
    pooler_for_token_embed as pooler_for_token_embed,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
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
    PromptIndexTargets as PromptIndexTargets,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.configs.colmodernvbert import (
    ColModernVBertConfig as ColModernVBertConfig,
)

class ColModernVBertConnector(nn.Module):
    pixel_shuffle_factor: Incomplete
    proj: Incomplete
    def __init__(self, config: ColModernVBertConfig) -> None: ...
    def pixel_shuffle(self, features: torch.Tensor) -> torch.Tensor: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...

class ColModernVBertProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> ColModernVBertConfig: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...
    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int: ...

class ColModernVBertDummyInputsBuilder(
    BaseDummyInputsBuilder[ColModernVBertProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class ColModernVBertMultiModalProcessor(
    BaseMultiModalProcessor[ColModernVBertProcessingInfo]
): ...

class ColModernVBertForRetrieval(
    nn.Module, SupportsMultiModal, SupportsLateInteraction, metaclass=abc.ABCMeta
):
    is_pooling_model: bool
    config: Incomplete
    vision_model: Incomplete
    connector: Incomplete
    text_embeddings: Incomplete
    text_layers: Incomplete
    text_final_norm: Incomplete
    custom_text_proj: Incomplete
    pooler: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    hf_to_vllm_mapper: Incomplete
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
