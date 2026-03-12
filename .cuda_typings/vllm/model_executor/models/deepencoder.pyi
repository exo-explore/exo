import torch
import torch.nn as nn
from .clip import (
    CLIPEncoder as CLIPEncoder,
    CLIPVisionEmbeddings as CLIPVisionEmbeddings,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from transformers import CLIPVisionConfig as CLIPVisionConfig
from vllm.model_executor.custom_op import PluggableLayer as PluggableLayer
from vllm.model_executor.layers.attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv2dLayer as Conv2dLayer
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)

class MLPBlock(nn.Module):
    lin1: Incomplete
    lin2: Incomplete
    act: Incomplete
    def __init__(
        self, embedding_dim: int, mlp_dim: int, act: type[nn.Module] = ...
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class LayerNorm2d(nn.Module):
    weight: Incomplete
    bias: Incomplete
    eps: Incomplete
    def __init__(self, num_channels: int, eps: float = 1e-06) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class ImageEncoderViT(nn.Module):
    img_size: Incomplete
    patch_embed: Incomplete
    pos_embed: nn.Parameter | None
    blocks: Incomplete
    neck: Incomplete
    net_2: Incomplete
    net_3: Incomplete
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = ...,
        act_layer: type[nn.Module] = ...,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: tuple[int, ...] = (),
        last_conv_output: int = 1024,
    ) -> None: ...
    def get_abs_pos(self, abs_pos: torch.Tensor, tgt_size: int): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Block(nn.Module):
    norm1: Incomplete
    attn: Incomplete
    norm2: Incomplete
    mlp: Incomplete
    window_size: Incomplete
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = ...,
        act_layer: type[nn.Module] = ...,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: tuple[int, int] | None = None,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class RelPosAttention(PluggableLayer):
    num_heads: Incomplete
    scale: Incomplete
    qkv: Incomplete
    proj: Incomplete
    use_rel_pos: Incomplete
    rel_pos_h: Incomplete
    rel_pos_w: Incomplete
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: tuple[int, int] | None = None,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

def window_partition(
    x: torch.Tensor, window_size: int
) -> tuple[torch.Tensor, tuple[int, int]]: ...
def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: tuple[int, int],
    hw: tuple[int, int],
) -> torch.Tensor: ...
def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor: ...
def add_decomposed_rel_pos(
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: tuple[int, int],
    k_size: tuple[int, int],
) -> torch.Tensor: ...

class PatchEmbed(nn.Module):
    proj: Incomplete
    def __init__(
        self,
        kernel_size: tuple[int, int] = (16, 16),
        stride: tuple[int, int] = (16, 16),
        padding: tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

def build_sam_vit_b(): ...

class DeepCLIPVisionEmbeddings(CLIPVisionEmbeddings):
    def get_abs_pos(self, abs_pos: torch.Tensor, tgt_size: int): ...
    def forward(
        self, pixel_values: torch.Tensor, patch_embeds: torch.Tensor | None = None
    ) -> torch.Tensor: ...

class DeepCLIPVisionTransformer(nn.Module):
    config: Incomplete
    embeddings: Incomplete
    pre_layrnorm: Incomplete
    transformer: Incomplete
    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        prefix: str = "",
    ) -> None: ...
    @property
    def dtype(self): ...
    @property
    def device(self): ...
    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_embeds: torch.Tensor | None = None,
        *,
        select_layers: list[int] | None = None,
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
