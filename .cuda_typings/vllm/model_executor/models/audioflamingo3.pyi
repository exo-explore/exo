import abc
import torch
import torch.nn as nn
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from transformers import PretrainedConfig as PretrainedConfig
from transformers.models.qwen2_audio import Qwen2AudioEncoder
from typing import Annotated, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems as DictEmbeddingItems,
    ModalityData as ModalityData,
    ModalityDataItems as ModalityDataItems,
    MultiModalDataItems as MultiModalDataItems,
    MultiModalDataParser as MultiModalDataParser,
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

MAX_AUDIO_LEN: Incomplete

class AudioFlamingo3FeatureInputs(TensorSchema):
    type: Literal["audio_features"]
    input_features: Annotated[torch.Tensor | list[torch.Tensor], None]
    feature_attention_mask: Annotated[torch.Tensor, None]
    chunk_counts: Annotated[torch.Tensor, None]

class AudioFlamingo3EmbeddingInputs(TensorSchema):
    type: Literal["audio_embeds"]
    audio_embeds: Annotated[list[torch.Tensor], None]

AudioFlamingo3Inputs: TypeAlias = (
    AudioFlamingo3FeatureInputs | AudioFlamingo3EmbeddingInputs
)

class AudioFlamingo3Encoder(Qwen2AudioEncoder):
    avg_pooler: Incomplete
    pos_emb: Incomplete
    def __init__(self, config: PretrainedConfig) -> None: ...
    def forward(
        self,
        input_features: torch.Tensor | list[torch.Tensor],
        attention_mask: torch.Tensor = None,
    ): ...

class AudioFlamingo3MultiModalProjector(nn.Module):
    linear_1: Incomplete
    act: Incomplete
    linear_2: Incomplete
    def __init__(self, config: PretrainedConfig) -> None: ...
    def forward(self, audio_features): ...

class AudioFlamingo3MultiModalDataParser(MultiModalDataParser): ...

class AudioFlamingo3ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object): ...
    def get_feature_extractor(self, **kwargs: object): ...
    def get_data_parser(self): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...

class AudioFlamingo3DummyInputsBuilder(
    BaseDummyInputsBuilder[AudioFlamingo3ProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class AudioFlamingo3MultiModalProcessor(
    BaseMultiModalProcessor[AudioFlamingo3ProcessingInfo]
): ...

class AudioFlamingo3ForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA, metaclass=abc.ABCMeta
):
    packed_modules_mapping: Incomplete
    def get_mm_mapping(self) -> MultiModelKeys: ...
    config: Incomplete
    multimodal_config: Incomplete
    quant_config: Incomplete
    audio_tower: Incomplete
    multi_modal_projector: Incomplete
    language_model: Incomplete
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
