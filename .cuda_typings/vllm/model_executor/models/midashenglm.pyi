import abc
import torch
import torch.nn as nn
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Iterable, Mapping
from typing import Annotated
from vllm.config import VllmConfig as VllmConfig
from vllm.config.multimodal import BaseDummyOptions as BaseDummyOptions
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.conv import Conv2dLayer as Conv2dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
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
    PromptUpdateDetails as PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.configs.midashenglm import DashengConfig as DashengConfig
from vllm.utils.tensor_schema import (
    TensorSchema as TensorSchema,
    TensorShape as TensorShape,
)

def calculate_mel_frames_dasheng(
    audio_length_samples: int,
    n_fft: int = 512,
    hop_size: int = 160,
    dasheng_subsampling: int = 4,
    center: bool = True,
    model_subsampling: int = 5,
) -> int: ...

class AudioPatchEmbed(nn.Module):
    input_size: Incomplete
    patch_size: Incomplete
    patch_stride: Incomplete
    grid_size: Incomplete
    num_patches: Incomplete
    flatten: Incomplete
    proj: Incomplete
    norm: Incomplete
    def __init__(
        self,
        input_size: _Tuple2 = 64,
        patch_size: _Tuple2 = 16,
        patch_stride: _Tuple2 = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten: bool = False,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class LayerScale(nn.Module):
    inplace: Incomplete
    gamma: Incomplete
    def __init__(
        self, dim, init_values: float = 1e-05, inplace: bool = False
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class DashengMlp(nn.Module):
    fc1: Incomplete
    act: Incomplete
    fc2: Incomplete
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class DashengAttention(nn.Module):
    embed_dim: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    scale: Incomplete
    qkv: Incomplete
    proj: Incomplete
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None): ...

class DashengBlock(nn.Module):
    norm1: Incomplete
    attn: Incomplete
    ls1: Incomplete
    norm2: Incomplete
    mlp: Incomplete
    ls2: Incomplete
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        init_values: float | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...

class DashengFrontend(nn.Module):
    config: Incomplete
    spectrogram_window: torch.Tensor
    melscale_fbanks: torch.Tensor
    def __init__(self, config: DashengConfig) -> None: ...
    def forward(self, waveform: torch.Tensor) -> torch.Tensor: ...

class DashengAudioTransformer(nn.Module):
    target_length: Incomplete
    hop_length: Incomplete
    front_end: Incomplete
    init_bn: Incomplete
    patch_embed: Incomplete
    time_pos_embed: Incomplete
    freq_pos_embed: Incomplete
    blocks: Incomplete
    norm: Incomplete
    def __init__(
        self,
        config: DashengConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward_features(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...
    def forward(
        self, x: torch.Tensor, x_length: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class AudioProjectorSubsample(nn.Module):
    k: Incomplete
    net: Incomplete
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        downsample_rate: int = 5,
        dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x, mask=None): ...

class MiDashengLMAudioInputs(TensorSchema):
    input_values: Annotated[torch.Tensor, None]
    audio_length: Annotated[torch.Tensor, None]

class MiDashengLMProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self): ...
    def get_feature_extractor(self): ...
    def get_data_parser(self): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_min_audio_len(self): ...
    def get_max_audio_len(self): ...

class MiDashengLMDummyInputsBuilder(BaseDummyInputsBuilder[MiDashengLMProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...

class MiDashengLMMultiModalProcessor(
    BaseMultiModalProcessor[MiDashengLMProcessingInfo]
): ...

class MiDashengLMModel(
    nn.Module, SupportsMultiModal, SupportsPP, metaclass=abc.ABCMeta
):
    packed_modules_mapping: Incomplete
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    config: Incomplete
    quant_config: Incomplete
    audio_encoder: Incomplete
    audio_projector: Incomplete
    decoder: Incomplete
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
