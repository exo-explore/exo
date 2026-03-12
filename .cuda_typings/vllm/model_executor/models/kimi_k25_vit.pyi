import torch
import torch.nn as nn
from _typeshed import Incomplete
from typing import Any
from vllm.distributed import (
    divide as divide,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    ReplicatedLinear as ReplicatedLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.models.utils import maybe_prefix as maybe_prefix
from vllm.model_executor.models.vision import (
    is_vit_use_data_parallel as is_vit_use_data_parallel,
    run_dp_sharded_mrope_vision_model as run_dp_sharded_mrope_vision_model,
)
from vllm.transformers_utils.configs.kimi_k25 import (
    KimiK25VisionConfig as KimiK25VisionConfig,
)

logger: Incomplete

def get_rope_shape_decorate(func): ...
@get_rope_shape_decorate
def get_rope_shape(org, interpolation_mode, shape): ...
def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos): ...
def get_1d_sincos_pos_embed(embed_dim, t_size, cls_token: bool = False): ...

class Learnable2DInterpPosEmbDivided_fixed(nn.Module):
    height: Incomplete
    width: Incomplete
    num_frames: Incomplete
    dim: Incomplete
    interpolation_mode: Incomplete
    weight: Incomplete
    def __init__(
        self,
        height: int,
        width: int,
        num_frames: int,
        dim: int,
        interpolation_mode: str = "bicubic",
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor: ...

class MoonVision3dPatchEmbed(nn.Module):
    patch_size: Incomplete
    proj: Incomplete
    pos_emb: Incomplete
    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int | tuple[int, int] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
        pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
    ) -> None: ...
    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor: ...

class Rope2DPosEmbRepeated(nn.Module):
    dim: Incomplete
    max_height: Incomplete
    max_width: Incomplete
    theta_base: Incomplete
    def __init__(
        self, dim: int, max_height: int, max_width: int, theta_base: int = 10000
    ) -> None: ...
    def extra_repr(self): ...
    def get_freqs_cis(
        self, grid_thws: torch.Tensor, device: torch.device
    ) -> torch.Tensor: ...

class MLP2(nn.Module):
    use_data_parallel: Incomplete
    fc0: Incomplete
    fc1: Incomplete
    activation: Incomplete
    def __init__(
        self,
        dims: list[int],
        activation,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class MoonViTEncoderLayer(nn.Module):
    use_data_parallel: Incomplete
    num_heads: Incomplete
    hidden_dim: Incomplete
    hidden_size_per_attention_head: Incomplete
    tp_size: Incomplete
    num_attention_heads_per_partition: Incomplete
    norm0: Incomplete
    norm1: Incomplete
    mlp: Incomplete
    wqkv: Incomplete
    wo: Incomplete
    attn: Incomplete
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        activation=...,
        attn_bias: bool = False,
    ) -> None: ...
    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
    ): ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
    ): ...

class MoonViT3dEncoder(nn.Module):
    video_attn_type: Incomplete
    rope_2d: Incomplete
    blocks: Incomplete
    final_layernorm: Incomplete
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        block_cfg: dict,
        video_attn_type: str = "spatial_temporal",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, grid_thws: torch.Tensor
    ) -> torch.Tensor: ...

def tpool_patch_merger(
    x: torch.Tensor,
    grid_thws: torch.Tensor,
    merge_kernel_size: tuple[int, int] = (2, 2),
) -> list[torch.Tensor]: ...

class MoonViT3dPretrainedModel(nn.Module):
    config: Incomplete
    merge_kernel_size: Incomplete
    patch_size: Incomplete
    merge_type: Incomplete
    patch_embed: Incomplete
    encoder: Incomplete
    def __init__(
        self,
        config: KimiK25VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self, pixel_values: torch.Tensor, grid_thws: torch.Tensor
    ) -> torch.Tensor: ...

def mm_projector_forward(
    mm_projector: torch.nn.Module, vt_output: list[torch.Tensor]
): ...
def vision_tower_forward(
    vision_tower: Any,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    mm_projector: Any,
    use_data_parallel: bool,
) -> list[torch.Tensor]: ...

class KimiK25MultiModalProjector(nn.Module):
    use_data_parallel: Incomplete
    hidden_size: Incomplete
    pre_norm: Incomplete
    linear_1: Incomplete
    linear_2: Incomplete
    act: Incomplete
    def __init__(
        self,
        config: KimiK25VisionConfig,
        use_data_parallel: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, image_features: torch.Tensor) -> torch.Tensor: ...
