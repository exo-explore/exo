import abc
import torch
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsLoRA as SupportsLoRA,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    flatten_bn as flatten_bn,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Iterable, Mapping
from torch import nn
from transformers import ProcessorMixin as ProcessorMixin
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.whisper import WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from typing import Annotated, Literal, TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.model_executor.layers.activation import (
    MulAndSilu as MulAndSilu,
    get_act_fn as get_act_fn,
)
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.model_loader import DefaultModelLoader as DefaultModelLoader
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
    NestedTensors as NestedTensors,
)
from vllm.multimodal.parse import (
    MultiModalDataItems as MultiModalDataItems,
    MultiModalDataParser as MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
)
from vllm.renderers import TokenizeParams as TokenizeParams
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.configs.ultravox import UltravoxConfig as UltravoxConfig
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

class UltravoxAudioFeatureInputs(TensorSchema):
    type: Literal["audio_features"]
    data: Annotated[torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]], None]
    lens: Annotated[torch.Tensor, None]
    token_len: Annotated[torch.Tensor, None]
    num_chunks: Annotated[torch.Tensor, None]

class UltravoxAudioEmbeddingInputs(TensorSchema):
    type: Literal["audio_embeds"]
    data: Annotated[torch.Tensor | list[torch.Tensor], None]

UltravoxAudioInputs: TypeAlias = (
    UltravoxAudioFeatureInputs | UltravoxAudioEmbeddingInputs
)

class UltravoxProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> ProcessorMixin: ...
    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor: ...
    def get_default_tok_params(self) -> TokenizeParams: ...
    def get_data_parser(self): ...
    def get_target_channels(self) -> int: ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...

class UltravoxDummyInputsBuilder(BaseDummyInputsBuilder[UltravoxProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class UltravoxMultiModalProcessor(BaseMultiModalProcessor[UltravoxProcessingInfo]): ...

class StackAudioFrames(nn.Module):
    stack_factor: Incomplete
    def __init__(self, stack_factor: int = 8) -> None: ...
    def forward(self, audio_embeds: torch.Tensor) -> torch.Tensor: ...

class UltravoxFeedForwardProjector(nn.Module):
    hidden_dim: Incomplete
    ln_pre: Incomplete
    linear_1: Incomplete
    act: Incomplete
    linear_2: Incomplete
    ln_mid: nn.Module
    ln_post: Incomplete
    def __init__(self, config: UltravoxConfig) -> None: ...
    def forward(
        self, audio_features: torch.Tensor, audio_token_len: torch.Tensor
    ) -> torch.Tensor: ...

class UltravoxTransformerProjector(nn.Module, ModuleUtilsMixin):
    config: Incomplete
    ln_pre: Incomplete
    linear_in: Incomplete
    embed_positions: Incomplete
    layers: Incomplete
    ln_post: Incomplete
    linear_out: Incomplete
    def __init__(self, config: UltravoxConfig) -> None: ...
    def forward(
        self, audio_features: torch.Tensor, audio_token_len: torch.Tensor
    ) -> torch.Tensor: ...

class ModifiedWhisperEncoder(WhisperEncoder):
    base_model_prefix: str
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def max_context_length(self): ...
    def get_attention_mask_by_audio_len(
        self, audio_lens: torch.Tensor | None, hidden_states: torch.Tensor
    ): ...
    def forward(
        self, input_features: torch.Tensor, audio_lens: torch.Tensor | None = None
    ): ...

class UltravoxModel(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA, metaclass=abc.ABCMeta
):
    packed_modules_mapping: Incomplete
    hf_to_vllm_mapper: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    multi_modal_config: Incomplete
    secondary_weights: Incomplete
    audio_tower: Incomplete
    multi_modal_projector: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def get_mm_mapping(self) -> MultiModelKeys: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

def pad_and_concat_to_dim3(
    features: torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]],
) -> torch.Tensor: ...
