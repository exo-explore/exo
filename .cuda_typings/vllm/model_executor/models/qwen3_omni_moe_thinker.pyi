import abc
import numpy as np
import torch
import torch.nn as nn
from .interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
    SupportsMRoPE as SupportsMRoPE,
    SupportsMultiModal as SupportsMultiModal,
    SupportsPP as SupportsPP,
    SupportsTranscription as SupportsTranscription,
)
from .qwen2_5_omni_thinker import (
    Qwen2_5OmniAudioFeatureInputs as Qwen2_5OmniAudioFeatureInputs,
    Qwen2_5OmniConditionalGenerationMixin as Qwen2_5OmniConditionalGenerationMixin,
    Qwen2_5OmniThinkerDummyInputsBuilder as Qwen2_5OmniThinkerDummyInputsBuilder,
    Qwen2_5OmniThinkerMultiModalProcessor as Qwen2_5OmniThinkerMultiModalProcessor,
    check_interleaved_audio_video as check_interleaved_audio_video,
    merge_interleaved_embeddings as merge_interleaved_embeddings,
)
from .qwen2_5_vl import (
    Qwen2_5_VLProcessingInfo as Qwen2_5_VLProcessingInfo,
    Qwen2_5_VisionAttention as Qwen2_5_VisionAttention,
)
from .qwen3_moe import (
    Qwen3MoeForCausalLM as Qwen3MoeForCausalLM,
    Qwen3MoeModel as Qwen3MoeModel,
)
from .utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    maybe_prefix as maybe_prefix,
)
from .vision import get_vit_attn_backend as get_vit_attn_backend
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Iterable, Iterator, Mapping
from transformers import PretrainedConfig as PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature as BatchFeature
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig as Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeThinkerConfig as Qwen3OmniMoeThinkerConfig,
)
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)
from typing import Any, Literal
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import (
    ModelConfig as ModelConfig,
    SpeechToTextConfig as SpeechToTextConfig,
    VllmConfig as VllmConfig,
)
from vllm.distributed import (
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.inputs.data import PromptType as PromptType
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv3dLayer as Conv3dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.rotary_embedding import get_rope as get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.model_executor.models.qwen2_audio import (
    Qwen2AudioProcessingInfo as Qwen2AudioProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec as MultiModalFeatureSpec,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems as AudioProcessorItems,
    MultiModalDataItems as MultiModalDataItems,
)
from vllm.multimodal.processing.processor import (
    MultiModalPromptUpdates as MultiModalPromptUpdates,
    PlaceholderFeaturesInfo as PlaceholderFeaturesInfo,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.transformers_utils.processor import (
    cached_processor_from_config as cached_processor_from_config,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum as AttentionBackendEnum,
)

logger: Incomplete
ISO639_1_SUPPORTED_LANGS: Incomplete

class SinusoidsPositionEmbedding(nn.Module):
    length: Incomplete
    channels: Incomplete
    max_timescale: Incomplete
    def __init__(
        self, length: int, channels: int, max_timescale: int = 10000
    ) -> None: ...
    def forward(self, seqlen: int) -> torch.Tensor: ...

class Qwen3OmniMoeAudioAttention(nn.Module):
    embed_dim: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    num_local_heads: Incomplete
    scaling: Incomplete
    qkv: Incomplete
    out_proj: Incomplete
    attn: Incomplete
    def __init__(
        self, config: Qwen3OmniMoeAudioEncoderConfig, prefix: str = ""
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
    ) -> torch.Tensor: ...

class Qwen3OmniMoeAudioEncoderLayer(nn.Module):
    embed_dim: Incomplete
    self_attn: Incomplete
    self_attn_layer_norm: Incomplete
    activation_fn: Incomplete
    fc1: Incomplete
    fc2: Incomplete
    final_layer_norm: Incomplete
    def __init__(
        self, config: Qwen3OmniMoeAudioEncoderConfig, prefix: str = ""
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
    ) -> torch.Tensor: ...

class Qwen3OmniMoeAudioEncoder(nn.Module):
    num_mel_bins: Incomplete
    max_source_positions: Incomplete
    n_window: Incomplete
    n_window_infer: Incomplete
    conv_chunksize: Incomplete
    positional_embedding: Incomplete
    conv2d1: Incomplete
    conv2d2: Incomplete
    conv2d3: Incomplete
    conv_out: Incomplete
    layers: Incomplete
    ln_post: Incomplete
    proj1: Incomplete
    act: Incomplete
    proj2: Incomplete
    attn_backend: Incomplete
    def __init__(
        self, config: Qwen3OmniMoeAudioEncoderConfig, prefix: str = ""
    ) -> None: ...
    def compute_attn_mask_seqlen(
        self, cu_seqlens: torch.Tensor
    ) -> torch.Tensor | None: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def device(self) -> torch.device: ...
    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
        aftercnn_lens: torch.Tensor,
    ): ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Qwen3_VisionPatchEmbed(nn.Module):
    patch_size: Incomplete
    temporal_patch_size: Incomplete
    hidden_size: Incomplete
    proj: Incomplete
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Qwen3_VisionMLP(nn.Module):
    linear_fc1: Incomplete
    linear_fc2: Incomplete
    act_fn: Incomplete
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = False,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = ...,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor): ...

class Qwen3_VisionBlock(nn.Module):
    norm1: Incomplete
    norm2: Incomplete
    attn: Incomplete
    mlp: Incomplete
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = ...,
        norm_layer: Callable[[int], nn.Module] | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor | None,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor: ...

class Qwen3_VisionPatchMerger(nn.Module):
    hidden_size: Incomplete
    use_postshuffle_norm: Incomplete
    ln_q: Incomplete
    mlp: Incomplete
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable[[int], nn.Module] | None = None,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Qwen3Omni_VisionTransformer(nn.Module):
    hidden_size: Incomplete
    num_heads: Incomplete
    image_size: Incomplete
    patch_size: Incomplete
    spatial_merge_size: Incomplete
    spatial_merge_unit: Incomplete
    temporal_patch_size: Incomplete
    num_grid_per_side: Incomplete
    apply_vit_abs_pos_embed: Incomplete
    deepstack_visual_indexes: Incomplete
    patch_embed: Incomplete
    pos_embed: Incomplete
    rotary_pos_emb: Incomplete
    blocks: Incomplete
    merger: Incomplete
    merger_list: Incomplete
    attn_backend: Incomplete
    def __init__(
        self,
        vision_config,
        norm_eps: float = 1e-06,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def device(self) -> torch.device: ...
    def rot_pos_emb(self, grid_thw): ...
    def fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> torch.Tensor: ...
    def compute_attn_mask_seqlen(self, cu_seqlens: torch.Tensor) -> torch.Tensor: ...
    def forward(self, x: torch.Tensor, grid_thw: list[list[int]]) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Qwen3MoeLLMModel(Qwen3MoeModel):
    deepstack_multiscale_layer_start: int
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...

class Qwen3MoeLLMForCausalLM(Qwen3MoeForCausalLM, metaclass=abc.ABCMeta):
    config: Incomplete
    quant_config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    logits_processor: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...

class Qwen3OmniMoeThinkerProcessingInfo(
    Qwen2AudioProcessingInfo, Qwen2_5_VLProcessingInfo
):
    def get_hf_config(self): ...
    def get_hf_processor(self, **kwargs: object) -> Qwen3OmniMoeProcessor: ...
    def get_feature_extractor(self, **kwargs: object): ...
    def get_supported_mm_limits(self) -> Mapping[str, int | None]: ...
    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int] | None = None
    ) -> Mapping[str, int] | None: ...

Qwen3OmniMoeThinkerDummyInputsBuilder = Qwen2_5OmniThinkerDummyInputsBuilder

class Qwen3OmniMoeThinkerMultiModalProcessor(Qwen2_5OmniThinkerMultiModalProcessor):
    def get_updates_use_audio_in_video(
        self,
        thinker_config: PretrainedConfig,
        audio_len: int,
        video_grid_thw: list[int] | torch.Tensor,
        video_second_per_grid_t: float,
    ) -> list[int]: ...

class Qwen3OmniMoeConditionalGenerationMixin(Qwen2_5OmniConditionalGenerationMixin): ...

class Qwen3OmniMoeThinkerForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsMRoPE,
    Qwen3OmniMoeConditionalGenerationMixin,
    SupportsTranscription,
    metaclass=abc.ABCMeta,
):
    hf_to_vllm_mapper: Incomplete
    packed_modules_mapping: Incomplete
    supported_languages = ISO639_1_SUPPORTED_LANGS
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    vllm_config: Incomplete
    config: Incomplete
    multimodal_config: Incomplete
    quant_config: Incomplete
    audio_tower: Incomplete
    use_deepstack: Incomplete
    deepstack_num_level: Incomplete
    visual_dim: Incomplete
    multiscale_dim: Incomplete
    visual: Incomplete
    deepstack_input_embeds: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None: ...
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
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def iter_mm_features(
        self, mm_features: list[MultiModalFeatureSpec]
    ) -> Iterator[tuple[int, str, dict[str, Any]]]: ...
    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig: ...
    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType: ...
    def get_mrope_input_positions(
        self, input_tokens: list[int], mm_features: list[MultiModalFeatureSpec]
    ) -> tuple[torch.Tensor, int]: ...
    def get_mm_mapping(self) -> MultiModelKeys: ...
