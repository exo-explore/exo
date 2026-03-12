import abc
import torch
from .minicpmv import (
    MiniCPMV2_6 as MiniCPMV2_6,
    MiniCPMV4_5 as MiniCPMV4_5,
    MiniCPMVDummyInputsBuilder as MiniCPMVDummyInputsBuilder,
    MiniCPMVMultiModalDataParser as MiniCPMVMultiModalDataParser,
    MiniCPMVMultiModalProcessor as MiniCPMVMultiModalProcessor,
    MiniCPMVProcessingInfo as MiniCPMVProcessingInfo,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    cast_overflow_tensors as cast_overflow_tensors,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Iterable, Mapping
from torch import nn
from transformers import BatchFeature as BatchFeature
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.whisper.modeling_whisper import (
    WhisperConfig as WhisperConfig,
    WhisperEncoder,
)
from typing import Annotated, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.multimodal import (
    MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    NestedTensors as NestedTensors,
)
from vllm.multimodal.parse import (
    AudioItem as AudioItem,
    AudioProcessorItems as AudioProcessorItems,
    DictEmbeddingItems as DictEmbeddingItems,
    ModalityData as ModalityData,
    ModalityDataItems as ModalityDataItems,
    MultiModalDataItems as MultiModalDataItems,
)
from vllm.multimodal.processing import (
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
)
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

CPU_DEVICE: Incomplete
FLAG_GEMS_CONFIG: Incomplete

class MiniCPMOAudioFeatureInputs(TensorSchema):
    type: Literal["audio_features"]
    audio_features: Annotated[torch.Tensor | list[torch.Tensor], None]
    audio_feature_lens: Annotated[torch.Tensor | list[torch.Tensor], None]

class MiniCPMOAudioEmbeddingInputs(TensorSchema):
    type: Literal["audio_embeds"]
    audio_embeds: Annotated[torch.Tensor | list[torch.Tensor], None]

MiniCPMOAudioInputs: TypeAlias = (
    MiniCPMOAudioFeatureInputs | MiniCPMOAudioEmbeddingInputs
)

class MiniCPMOAudioEmbeddingItems(DictEmbeddingItems):
    def __init__(
        self,
        data: Mapping[str, torch.Tensor],
        fields_factory: Callable[
            [Mapping[str, torch.Tensor]], Mapping[str, MultiModalFieldConfig]
        ],
    ) -> None: ...

class MiniCPMOMultiModalDataParser(MiniCPMVMultiModalDataParser): ...

class MiniCPMOProcessingInfo(MiniCPMVProcessingInfo):
    audio_pattern: str
    def get_data_parser(self): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_audio_placeholder(
        self, audio_lens: int, chunk_input: bool = True, chunk_length: int = 1
    ) -> str: ...
    def get_default_audio_pool_step(self) -> int: ...
    def get_default_audio_sampling_rate(self) -> int: ...
    def get_chunk_length(self) -> int: ...
    def get_max_audio_tokens_per_chunk(self) -> int: ...
    def get_max_audio_chunks_with_most_features(self) -> int: ...
    def get_max_audio_tokens(self) -> int: ...
    def get_audio_len_by_num_chunks(self, num_chunks: int) -> int: ...
    def get_num_frames_with_most_features(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> int: ...

class MiniCPMODummyInputsBuilder(MiniCPMVDummyInputsBuilder[MiniCPMOProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class MiniCPMOMultiModalProcessor(MiniCPMVMultiModalProcessor[MiniCPMOProcessingInfo]):
    def get_audio_prompt_texts(
        self, audio_lens: int, chunk_input: bool = True, chunk_length: int = 1
    ) -> str: ...
    def process_audios(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]: ...
    def process_mm_inputs(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]: ...

class MultiModalProjector(nn.Module):
    linear1: Incomplete
    relu: Incomplete
    linear2: Incomplete
    def __init__(self, in_dim: int, out_dim: int) -> None: ...
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor: ...

class MiniCPMWhisperEncoderLayer(nn.Module):
    embed_dim: Incomplete
    self_attn: Incomplete
    self_attn_layer_norm: Incomplete
    dropout: Incomplete
    activation_fn: Incomplete
    activation_dropout: Incomplete
    fc1: Incomplete
    fc2: Incomplete
    final_layer_norm: Incomplete
    def __init__(self, config: WhisperConfig, layer_idx: int) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor: ...

class MiniCPMWhisperEncoder(WhisperEncoder):
    layers: Incomplete
    def __init__(self, config: WhisperConfig) -> None: ...
    def forward(
        self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> BaseModelOutputWithPast: ...

class MiniCPMOBaseModel:
    packed_modules_mapping: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    apm: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    audio_avg_pooler: Incomplete
    audio_projection_layer: Incomplete
    audio_encoder_layer: int
    def init_audio_module(self, *, vllm_config: VllmConfig, prefix: str = ""): ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def subsequent_chunk_mask(
        self,
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = ...,
        num_lookhead: int = 0,
    ) -> torch.Tensor: ...
    def get_audio_hidden_states(
        self, data: MiniCPMOAudioFeatureInputs
    ) -> list[torch.Tensor]: ...

class MiniCPMO2_6(MiniCPMOBaseModel, MiniCPMV2_6, metaclass=abc.ABCMeta):
    apm: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...

class MiniCPMO4_5(MiniCPMOBaseModel, MiniCPMV4_5, metaclass=abc.ABCMeta):
    apm: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...

class MiniCPMO(MiniCPMOBaseModel, MiniCPMV2_6, metaclass=abc.ABCMeta):
    def __new__(cls, *, vllm_config: VllmConfig, prefix: str = ""): ...
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
