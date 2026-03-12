import abc
import torch
from .step3_vl import Step3VLForConditionalGeneration as Step3VLForConditionalGeneration
from .utils import (
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from .vision import (
    is_vit_use_data_parallel as is_vit_use_data_parallel,
    run_dp_sharded_vision_model as run_dp_sharded_vision_model,
)
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from torch import nn
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention as MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv2dLayer as Conv2dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)

def rotate_half(x): ...
def apply_rotary_emb(
    freqs, t, start_index: int = 0, scale: float = 1.0, seq_dim: int = -2
): ...

class PerceptionEncoderRope2D(nn.Module):
    dim: Incomplete
    max_grid_height: Incomplete
    max_grid_width: Incomplete
    use_cls_token: Incomplete
    theta: Incomplete
    max_freq: Incomplete
    num_freqs: Incomplete
    def __init__(
        self,
        dim: int,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool = False,
        theta: int = 10000,
        max_freq: int = 10,
        num_freqs: int = 1,
        theta_rescale_factor: float = 1.0,
    ) -> None: ...
    def forward(self, q: torch.Tensor, k: torch.Tensor, grid_hw: tuple[int, int]): ...

class PerceptionEncoderLayerScale(nn.Module):
    inplace: Incomplete
    gamma: Incomplete
    def __init__(
        self, dim, init_values: float = 1e-05, inplace: bool = False
    ) -> None: ...
    def forward(self, x): ...

class PerceptionEncoderMLP(nn.Module):
    fc1: Incomplete
    activation: Incomplete
    fc2: Incomplete
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        act_layer: Callable[[], nn.Module],
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class PerceptionEncoderVisionAttention(nn.Module):
    embed_dim: Incomplete
    total_num_heads: Incomplete
    head_dim: Incomplete
    scale: Incomplete
    num_heads: Incomplete
    qkv_proj: Incomplete
    out_proj: Incomplete
    attn: Incomplete
    rope: Incomplete
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor: ...

class PerceptionEncoderVisionBlock(nn.Module):
    attn: Incomplete
    ls_1: Incomplete
    ls_2: Incomplete
    ln_1: Incomplete
    ln_2: Incomplete
    mlp: Incomplete
    def __init__(
        self,
        d_model: int,
        n_head: int,
        max_grid_height: int,
        max_grid_width: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = ...,
        norm_layer: Callable = ...,
        use_cls_token: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor, grid_hw: tuple[int, int]): ...

class PerceptionEncoderVisionTransformer(nn.Module):
    width: Incomplete
    layers: Incomplete
    resblocks: Incomplete
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        max_grid_height: int,
        max_grid_width: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = ...,
        norm_layer: Callable = ...,
        use_cls_token: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def forward(self, x: torch.Tensor, grid_hw: tuple[int, int]): ...

class PerceptionEncoder(nn.Module):
    patch_size: Incomplete
    output_dim: Incomplete
    heads: Incomplete
    width: Incomplete
    layers: Incomplete
    use_abs_posemb: Incomplete
    use_cls_token: Incomplete
    use_rope2d: Incomplete
    image_size: Incomplete
    conv1: Incomplete
    ln_pre: Incomplete
    ln_post: Incomplete
    transformer: Incomplete
    vit_downsampler1: Incomplete
    vit_downsampler2: Incomplete
    class_embedding: Incomplete
    posemb_grid_size: Incomplete
    positional_embedding: Incomplete
    def __init__(
        self,
        config,
        act_layer: Callable,
        norm_layer: Callable = ...,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def sample_abs_posemb(self, grid_h: int, grid_w: int): ...
    def forward_features(self, x: torch.Tensor): ...
    def forward(self, x: torch.Tensor): ...

class StepVLForConditionalGeneration(
    Step3VLForConditionalGeneration, metaclass=abc.ABCMeta
):
    hf_to_vllm_mapper: Incomplete
    config: Incomplete
    multimodal_config: Incomplete
    use_data_parallel: Incomplete
    vision_model: Incomplete
    vit_large_projector: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
