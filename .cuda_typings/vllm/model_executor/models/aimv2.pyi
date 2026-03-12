import torch
import torch.nn as nn
from _typeshed import Incomplete
from collections.abc import Iterable
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.distributed.utils import divide as divide
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv2dLayer as Conv2dLayer
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear as MergedColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.transformers_utils.configs.ovis import AIMv2Config as AIMv2Config

class AIMv2SwiGLUFFN(nn.Module):
    fc13: Incomplete
    fc2: Incomplete
    act_fn: Incomplete
    def __init__(
        self, config: AIMv2Config, quant_config: QuantizationConfig, prefix: str
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class AIMv2PatchEmbed(nn.Module):
    proj: Incomplete
    norm: Incomplete
    def __init__(self, config: AIMv2Config) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class AIMv2ViTPreprocessor(nn.Module):
    patchifier: Incomplete
    pos_embed: Incomplete
    def __init__(self, config: AIMv2Config) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class AIMv2Attention(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    scale: Incomplete
    qkv: Incomplete
    proj: Incomplete
    tp_size: Incomplete
    num_heads_per_partition: Incomplete
    attn: Incomplete
    def __init__(
        self, config: AIMv2Config, quant_config: QuantizationConfig, prefix: str
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class AIMv2Block(nn.Module):
    attn: Incomplete
    norm_1: Incomplete
    mlp: Incomplete
    norm_2: Incomplete
    def __init__(
        self, config: AIMv2Config, quant_config: QuantizationConfig, prefix: str
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class AIMv2Transformer(nn.Module):
    blocks: Incomplete
    post_trunk_norm: Incomplete
    def __init__(
        self,
        config: AIMv2Config,
        quant_config: QuantizationConfig,
        *,
        require_post_norm: bool | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, tokens: torch.Tensor) -> torch.Tensor: ...

class AIMv2Model(torch.nn.Module):
    preprocessor: Incomplete
    trunk: Incomplete
    def __init__(
        self,
        config: AIMv2Config,
        quant_config: QuantizationConfig,
        *,
        require_post_norm: bool | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
