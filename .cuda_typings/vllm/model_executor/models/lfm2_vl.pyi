import abc
import torch
import torch.nn as nn
from .interfaces import (
    IsHybrid as IsHybrid,
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .lfm2_siglip2 import Siglip2Model as Siglip2Model
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from .vision import is_vit_use_data_parallel as is_vit_use_data_parallel
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers.models.lfm2_vl import Lfm2VlProcessor
from transformers.models.lfm2_vl.configuration_lfm2_vl import Lfm2VlConfig
from transformers.models.lfm2_vl.image_processing_lfm2_vl_fast import (
    Lfm2VlImageProcessorFast as Lfm2VlImageProcessorFast,
)
from typing import Annotated, Literal
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.forward_context import set_forward_context as set_forward_context
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc as MambaStateCopyFunc,
    MambaStateCopyFuncCalculator as MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator as MambaStateDtypeCalculator,
    MambaStateShapeCalculator as MambaStateShapeCalculator,
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
    ImageSize as ImageSize,
    MultiModalDataItems as MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    PromptReplacement as PromptReplacement,
    PromptUpdateDetails as PromptUpdateDetails,
)
from vllm.renderers import TokenizeParams as TokenizeParams
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

class Lfm2VLImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, None]
    spatial_shapes: Annotated[torch.Tensor, None]
    num_patches: Annotated[torch.Tensor, None]

LFM2VLImageInputs = Lfm2VLImagePixelInputs

class Lfm2VLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs): ...
    def get_image_processor(self, **kwargs: object) -> Lfm2VlImageProcessorFast: ...
    def get_default_tok_params(self) -> TokenizeParams: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_image_size_with_most_features(self) -> ImageSize: ...
    def smart_resize(
        self,
        height: int,
        width: int,
        downsample_factor: int,
        min_image_tokens: int,
        max_image_tokens: int,
        encoder_patch_size: int,
    ) -> tuple[int, int]: ...
    def get_num_patches(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Lfm2VlProcessor,
        mm_kwargs: Mapping[str, object],
    ) -> int: ...
    def get_image_repl(
        self,
        image_width: int,
        image_height: int,
        spatial_shapes: torch.Tensor,
        processor: Lfm2VlProcessor,
        mm_kwargs: Mapping[str, object],
    ) -> str: ...
    def get_num_image_tokens(
        self,
        *,
        spatial_shapes: torch.Tensor,
        processor: Lfm2VlProcessor,
        mm_kwargs: Mapping[str, object],
    ) -> tuple[int, int]: ...

class Lfm2VLDummyInputsBuilder(BaseDummyInputsBuilder[Lfm2VLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class Lfm2VLMultiModalProcessor(BaseMultiModalProcessor[Lfm2VLProcessingInfo]): ...

class Lfm2VLMultiModalProjector(nn.Module):
    use_data_parallel: Incomplete
    factor: Incomplete
    projector_use_layernorm: Incomplete
    layer_norm: Incomplete
    linear_1: Incomplete
    act: Incomplete
    linear_2: Incomplete
    def __init__(self, config: Lfm2VlConfig, prefix: str = "") -> None: ...
    def forward(
        self, vision_features_packed: torch.Tensor, spatial_shapes: torch.Tensor
    ) -> torch.Tensor: ...

class Lfm2VLForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    IsHybrid,
    metaclass=abc.ABCMeta,
):
    merge_by_field_config: bool
    hf_to_vllm_mapper: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, ...]: ...
    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int]]: ...
    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc]: ...
    config: Incomplete
    vllm_config: Incomplete
    multimodal_config: Incomplete
    use_data_parallel: Incomplete
    vision_tower: Incomplete
    multi_modal_projector: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None: ...
    def image_pixels_to_features(
        self, pixel_values: torch.FloatTensor, spatial_shapes: torch.Tensor
    ) -> torch.Tensor: ...
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
    def get_mm_mapping(self) -> MultiModelKeys: ...
