import torch
from _typeshed import Incomplete
from torch import Tensor, nn

class BlockBase(nn.Module):
    input_size: Incomplete
    output_size: Incomplete
    def __init__(self, input_size: int, output_size: int) -> None: ...

def get_activation(name: str = "relu") -> torch.nn.Module: ...
def adaptive_enc_mask(
    x_len: int, chunk_start_idx: list[int], left_window: int = 0, right_window: int = 0
) -> torch.Tensor: ...

class GLU(nn.Module):
    dim: Incomplete
    act_fn: Incomplete
    def __init__(self, dim: int = -1, act_name: str = "sigmoid") -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class GLUPointWiseConv(nn.Module):
    glu_type: Incomplete
    output_dim: Incomplete
    bias_in_glu: Incomplete
    ext_pw_conv_1d: Incomplete
    glu_act: Incomplete
    b1: Incomplete
    b2: Incomplete
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        glu_type: str = "sigmoid",
        bias_in_glu: bool = True,
        causal: bool = False,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class DepthWiseSeparableConv1d(nn.Module):
    dw_conv: Incomplete
    pw_conv: Incomplete
    depthwise_seperable_out_channel: Incomplete
    def __init__(
        self,
        input_dim: int,
        depthwise_seperable_out_channel: int,
        kernel_size: int,
        depthwise_multiplier: int,
        padding: int = 0,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class ConvModule(nn.Module):
    layer_norm: Incomplete
    input_dim: Incomplete
    ext_pw_out_channel: Incomplete
    ext_pw_kernel_size: Incomplete
    depthwise_seperable_out_channel: Incomplete
    glu_type: Incomplete
    bias_in_glu: Incomplete
    linear_glu_in_convm: Incomplete
    causal: Incomplete
    batch_norm: Incomplete
    kernel_size: Incomplete
    bn_layer: Incomplete
    act: Incomplete
    dropout: Incomplete
    export: Incomplete
    dw_sep_conv_1d: Incomplete
    ln2: Incomplete
    def __init__(
        self,
        input_dim: int,
        ext_pw_out_channel: int,
        depthwise_seperable_out_channel: int,
        ext_pw_kernel_size: int,
        kernel_size: int,
        depthwise_multiplier: int,
        dropout_rate: float,
        causal: bool = False,
        batch_norm: bool = False,
        chunk_se: int = 0,
        chunk_size: int = 18,
        activation: str = "relu",
        glu_type: str = "sigmoid",
        bias_in_glu: bool = True,
        linear_glu_in_convm: bool = False,
        export: bool = False,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class GLULinear(nn.Module):
    linear: Incomplete
    glu_act: Incomplete
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        glu_type: str = "sigmoid",
        bias_in_glu: bool = True,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class FeedForward(nn.Module):
    d_model: Incomplete
    d_inner: Incomplete
    layer_norm: Incomplete
    net: Incomplete
    def __init__(
        self,
        d_model: int,
        d_inner: int,
        dropout_rate: float,
        activation: str = "sigmoid",
        bias_in_glu: bool = True,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class T5RelativeAttentionLogitBias(nn.Module):
    num_heads: Incomplete
    num_buckets: Incomplete
    max_distance: Incomplete
    symmetric: Incomplete
    bias_values: Incomplete
    def __init__(
        self,
        num_heads: int,
        num_buckets: int = -1,
        max_distance: int = 1000,
        symmetric: bool = False,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class AbsolutePositionalEncoding(nn.Module):
    d_model: Incomplete
    xscale: Incomplete
    dropout: Incomplete
    pe: Incomplete
    def __init__(
        self, d_model: int, dropout_rate: float, max_len: int = 5000
    ) -> None: ...
    def extend_pe(self, x: torch.Tensor) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class MeanVarianceNormLayer(nn.Module):
    input_size: Incomplete
    global_mean: Incomplete
    global_invstd: Incomplete
    def __init__(self, input_size: int) -> None: ...
    def forward(self, input_: Tensor) -> Tensor: ...

class CausalConv1D(nn.Conv1d):
    cache_drop_size: Incomplete
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: str | int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None: ...
    def update_cache(
        self, x: Tensor, cache: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]: ...
    def forward(
        self, x: Tensor, cache: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor | None]: ...

class CausalConv2D(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: str | int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class NemoConvSubsampling(torch.nn.Module):
    subsampling_factor: Incomplete
    is_causal: Incomplete
    subsampling_causal_cond: Incomplete
    subsampling_conv_chunking_factor: Incomplete
    out: Incomplete
    conv2d_subsampling: bool
    conv: Incomplete
    def __init__(
        self,
        feat_in: int,
        feat_out: int,
        subsampling_factor: int = 4,
        subsampling: str = "dw_striding",
        conv_channels: int = 256,
        subsampling_conv_chunking_factor: int = 1,
        activation: torch.nn.Module = ...,
        is_causal: bool = False,
    ) -> None: ...
    def get_sampling_frames(self) -> list[int]: ...
    def get_streaming_cache_size(self) -> list[int]: ...
    def forward(
        self, x: Tensor, mask: Tensor | None
    ) -> tuple[Tensor, Tensor | None]: ...
    def reset_parameters(self) -> None: ...
    def conv_split_by_batch(self, x: Tensor) -> tuple[Tensor, bool]: ...
    def conv_split_by_channel(self, x: Tensor) -> Tensor: ...
    def channel_chunked_conv(
        self, conv: torch.nn.Module, chunk_size: int, x: Tensor
    ) -> Tensor: ...
    def change_subsampling_conv_chunking_factor(
        self, subsampling_conv_chunking_factor: int
    ) -> None: ...

def calc_length_int(
    lengths: int,
    all_paddings: int,
    kernel_size: int,
    stride: int,
    ceil_mode: bool,
    repeat_num: int = 1,
) -> int: ...

class AttModule(nn.Module):
    export_mode: bool
    def __init__(self) -> None: ...
    def set_export(self, mode: bool = True) -> None: ...
    def forward(
        self,
        x: Tensor,
        memory: Tensor | None = None,
        pos_emb: Tensor | None = None,
        att_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]: ...

class AttBlock(BlockBase, AttModule):
    def memory_dims(self, max_len: bool = False) -> tuple[int, int]: ...

def masked_softmax(scores: Tensor, mask: Tensor | None) -> Tensor: ...

class MultiHeadedAttention(nn.Module):
    inv_sqrt_d_k: torch.jit.Final[float]
    h: torch.jit.Final[int]
    h_k: torch.jit.Final[int]
    g: torch.jit.Final[int]
    d_k: Incomplete
    linear_q: Incomplete
    linear_k: Incomplete
    linear_v: Incomplete
    linear_out: Incomplete
    attn: Incomplete
    dropout: Incomplete
    dropout_rate: Incomplete
    use_pt_scaled_dot_product_attention: Incomplete
    quant_q: Incomplete
    quant_x: Incomplete
    dequant: Incomplete
    ffunc: Incomplete
    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float,
        attention_inner_dim: int = -1,
        glu_type: str = "swish",
        bias_in_glu: bool = True,
        use_pt_scaled_dot_product_attention: bool = False,
        n_value: int = -1,
        group_size: int = 1,
    ) -> None: ...
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_k: Tensor | None,
        pos_v: Tensor | None,
        mask: Tensor | None,
        relative_attention_bias: Tensor | None = None,
    ) -> Tensor: ...

class MultiSequential(torch.nn.Sequential):
    @torch.jit.ignore
    def forward(self, *args) -> tuple: ...

def get_offset(input_layer: str, time_reduction: int) -> int: ...
def unfold_tensor(xs_pad: Tensor, max_seq_len: int) -> Tensor: ...
