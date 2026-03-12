import abc
import torch
import torch.nn as nn
from .utils import maybe_prefix as maybe_prefix
from .vision import (
    get_vit_attn_backend as get_vit_attn_backend,
    is_vit_use_data_parallel as is_vit_use_data_parallel,
)
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from transformers.models.glm_ocr.configuration_glm_ocr import (
    GlmOcrVisionConfig as GlmOcrVisionConfig,
)
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    parallel_state as parallel_state,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv2dLayer as Conv2dLayer
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear as QKVParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.rotary_embedding import get_rope as get_rope
from vllm.model_executor.layers.rotary_embedding.common import (
    ApplyRotaryEmb as ApplyRotaryEmb,
)
from vllm.model_executor.models.glm4_1v import (
    Glm4vDummyInputsBuilder as Glm4vDummyInputsBuilder,
    Glm4vForConditionalGeneration as Glm4vForConditionalGeneration,
    Glm4vMultiModalProcessor as Glm4vMultiModalProcessor,
    Glm4vPatchMerger as Glm4vPatchMerger,
    Glm4vProcessingInfo as Glm4vProcessingInfo,
    Glm4vVisionBlock as Glm4vVisionBlock,
    Glm4vVisionMLP as Glm4vVisionMLP,
    Glm4vVisionPatchEmbed as Glm4vVisionPatchEmbed,
    Glm4vVisionTransformer as Glm4vVisionTransformer,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY

logger: Incomplete

class GlmOcrVisionMLP(Glm4vVisionMLP): ...

class GlmOcrVisionAttention(nn.Module):
    tp_size: Incomplete
    tp_rank: Incomplete
    hidden_size_per_attention_head: Incomplete
    num_attention_heads_per_partition: Incomplete
    head_dim: Incomplete
    q_norm: Incomplete
    k_norm: Incomplete
    qkv: Incomplete
    proj: Incomplete
    attn: Incomplete
    apply_rotary_emb: Incomplete
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]: ...
    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class GlmOcrVisionBlock(Glm4vVisionBlock):
    norm1: Incomplete
    norm2: Incomplete
    attn: Incomplete
    mlp: Incomplete
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        norm_layer: Callable[[int], nn.Module] | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...

class GlmOcrVisionPatchEmbed(Glm4vVisionPatchEmbed): ...
class GlmOcrPatchMerger(Glm4vPatchMerger): ...

class GlmOcrVisionTransformer(Glm4vVisionTransformer):
    hidden_size: Incomplete
    num_heads: Incomplete
    patch_size: Incomplete
    spatial_merge_size: Incomplete
    out_hidden_size: Incomplete
    patch_embed: Incomplete
    rotary_pos_emb: Incomplete
    blocks: Incomplete
    merger: Incomplete
    downsample: Incomplete
    post_layernorm: Incomplete
    attn_backend: Incomplete
    def __init__(
        self,
        vision_config: GlmOcrVisionConfig,
        norm_eps: float = 1e-05,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, x: torch.Tensor, grid_thw: torch.Tensor | list[list[int]]
    ) -> torch.Tensor: ...

class GlmOcrForConditionalGeneration(
    Glm4vForConditionalGeneration, metaclass=abc.ABCMeta
):
    visual: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
