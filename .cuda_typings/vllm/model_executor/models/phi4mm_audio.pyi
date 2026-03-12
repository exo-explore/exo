import abc
import torch
from _typeshed import Incomplete
from torch import Tensor, nn
from transformers import PretrainedConfig as PretrainedConfig
from typing import Any, Literal
from vllm.model_executor.models.phi4mm_utils import (
    AbsolutePositionalEncoding as AbsolutePositionalEncoding,
    ConvModule as ConvModule,
    FeedForward as FeedForward,
    MeanVarianceNormLayer as MeanVarianceNormLayer,
    MultiHeadedAttention as MultiHeadedAttention,
    MultiSequential as MultiSequential,
    NemoConvSubsampling as NemoConvSubsampling,
    T5RelativeAttentionLogitBias as T5RelativeAttentionLogitBias,
    adaptive_enc_mask as adaptive_enc_mask,
    get_offset as get_offset,
    unfold_tensor as unfold_tensor,
)

class ConformerEncoderLayer(nn.Module):
    feed_forward_in: Incomplete
    self_attn: Incomplete
    conv: Incomplete
    feed_forward_out: Incomplete
    layer_norm_att: Incomplete
    layer_norm: Incomplete
    def __init__(
        self,
        d_model: int = 512,
        ext_pw_out_channel: int = 0,
        depthwise_seperable_out_channel: int = 256,
        depthwise_multiplier: int = 1,
        n_head: int = 4,
        d_ffn: int = 2048,
        ext_pw_kernel_size: int = 1,
        kernel_size: int = 3,
        dropout_rate: float = 0.1,
        causal: bool = False,
        batch_norm: bool = False,
        activation: str = "relu",
        chunk_se: int = 0,
        chunk_size: int = 18,
        conv_activation: str = "relu",
        conv_glu_type: str = "sigmoid",
        bias_in_glu: bool = True,
        linear_glu_in_convm: bool = False,
        attention_inner_dim: int = -1,
        attention_glu_type: str = "swish",
        activation_checkpointing: str = "",
        export: bool = False,
        use_pt_scaled_dot_product_attention: bool = False,
        attn_group_sizes: int = 1,
    ) -> None: ...
    def forward(
        self,
        x: torch.Tensor,
        pos_k: torch.Tensor,
        pos_v: torch.Tensor,
        mask: torch.Tensor,
        relative_attention_bias: Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

class TransformerEncoderBase(abc.ABC, nn.Module, metaclass=abc.ABCMeta):
    input_size: Incomplete
    input_layer: Incomplete
    chunk_size: Incomplete
    left_chunk: Incomplete
    attention_dim: Incomplete
    num_heads: Incomplete
    attention_group_size: Incomplete
    time_reduction: Incomplete
    nemo_conv_settings: Incomplete
    encoder_embedding_config: Incomplete
    embed: Incomplete
    pos_emb: Incomplete
    relative_attention_bias_type: Incomplete
    relative_attention_bias_layer: Incomplete
    encoder_embedding: Incomplete
    def __init__(
        self,
        input_size: int,
        chunk_size: int | list[int],
        left_chunk: int | list[int],
        attention_dim: int = 256,
        attention_heads: int = 4,
        input_layer: str = "nemo_conv",
        cnn_out: int = -1,
        cnn_layer_norm: bool = False,
        time_reduction: int = 4,
        dropout_rate: float = 0.0,
        padding_idx: int = -1,
        relative_attention_bias_args: dict[str, Any] | None = None,
        positional_dropout_rate: float = 0.0,
        nemo_conv_settings: dict[str, Any] | None = None,
        conv2d_extra_padding: Literal["feat", "feat_time", "none", True] = "none",
        attention_group_size: int = 1,
        encoder_embedding_config: dict[str, Any] | None = None,
    ) -> None: ...
    def compute_lens_change(
        self, feature_lens: int | torch.Tensor
    ) -> int | torch.Tensor: ...
    @abc.abstractmethod
    def forward(self) -> Any: ...
    def forward_embeddings(
        self,
        xs_pad: torch.Tensor,
        masks: torch.Tensor,
        chunk_size_nc: int | list[int] | None = None,
        left_chunk_nc: int | list[int] | None = None,
    ) -> (
        tuple[
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor,
            torch.Tensor,
        ]
        | tuple[
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ): ...
    def get_offset(self) -> int: ...

class ConformerEncoder(TransformerEncoderBase):
    extra_multi_layer_output_idxs: list[int]
    num_blocks: Incomplete
    num_lang: Incomplete
    kernel_size: Incomplete
    replication_pad_for_subsample_embedding: bool
    num_heads_k: Incomplete
    encoders: Incomplete
    extra_layer_output_idx: Incomplete
    def __init__(
        self,
        input_size: int,
        chunk_size: int | list[int],
        left_chunk: int | list[int],
        num_lang: int | None = None,
        attention_dim: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        input_layer: str = "nemo_conv",
        causal: bool = True,
        batch_norm: bool = False,
        cnn_out: int = -1,
        cnn_layer_norm: bool = False,
        ext_pw_out_channel: int = 0,
        ext_pw_kernel_size: int = 1,
        depthwise_seperable_out_channel: int = 256,
        depthwise_multiplier: int = 1,
        chunk_se: int = 0,
        kernel_size: int = 3,
        activation: str = "relu",
        conv_activation: str = "relu",
        conv_glu_type: str = "sigmoid",
        bias_in_glu: bool = True,
        linear_glu_in_convm: bool = False,
        attention_glu_type: str = "swish",
        export: bool = False,
        extra_layer_output_idx: int = -1,
        extra_multi_layer_output_idxs: list[int] = [],
        activation_checkpointing: str = "",
        relative_attention_bias_args: dict[str, Any] | None = None,
        time_reduction: int = 4,
        use_pt_scaled_dot_product_attention: bool = False,
        nemo_conv_settings: dict[str, Any] | None = None,
        conv2d_extra_padding: Literal["feat", "feat_time", "none", True] = "none",
        replication_pad_for_subsample_embedding: bool = False,
        attention_group_size: int = 1,
        encoder_embedding_config: dict[str, Any] | None = None,
    ) -> None: ...
    def init_relative_attention_bias(
        self, input_tensor: torch.Tensor
    ) -> torch.Tensor | None: ...
    def calculate_hs_mask(
        self, xs_pad: torch.Tensor, device: torch.device, mask: torch.Tensor | None
    ) -> torch.Tensor: ...
    @torch.jit.ignore
    def forward(
        self, xs_pad: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class WindowQformer(nn.Module):
    decoders: Incomplete
    queries: Incomplete
    after_norm: Incomplete
    window_size: Incomplete
    def __init__(
        self,
        window_size: int = 8,
        num_queries: int = 1,
        num_blocks: int = 2,
        attention_dim: int = 512,
        attention_heads: int = 8,
        linear_units: int = 2048,
        dropout_rate: float = 0.0,
        normalize_before: bool = True,
    ) -> None: ...
    def forward(
        self,
        audio_embed: torch.Tensor,
        mask: torch.Tensor | None,
        embed_len: int | None = None,
    ) -> tuple[torch.Tensor, int | None]: ...

class AudioEmbedding(nn.Module):
    config: Incomplete
    layer_idx: int
    encoder: Incomplete
    audio_dim_out: Incomplete
    audio_dim_in: Incomplete
    freeze_audio_processor: Incomplete
    downsample_rate: Incomplete
    qformer: Incomplete
    conv_ds: Incomplete
    audio_projection: Incomplete
    linear_downsample_rate: Incomplete
    audio_projection_for_vision: Incomplete
    vocab_size: Incomplete
    input_embeds: Incomplete
    audio_embed_sizes: Incomplete
    def __init__(self, config: PretrainedConfig, **kwargs: Any) -> None: ...
    def set_audio_embeds(self, input_embeds: torch.Tensor) -> None: ...
    def set_audio_embed_sizes(self, audio_embed_sizes: torch.Tensor) -> None: ...
    def get_audio_features(
        self,
        input_embeds: torch.Tensor,
        audio_attention_mask: torch.Tensor | None = None,
        audio_projection_mode: str = "speech",
    ) -> torch.Tensor: ...
    def forward(
        self,
        audio_features: torch.Tensor,
        audio_attention_mask: torch.Tensor | None = None,
        audio_projection_mode: str = "speech",
    ) -> torch.Tensor: ...
