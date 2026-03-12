from _typeshed import Incomplete
from transformers import PretrainedConfig

class DashengConfig(PretrainedConfig):
    model_type: str
    embed_dim: Incomplete
    outputdim: Incomplete
    patch_size: Incomplete
    patch_stride: Incomplete
    input_channels: Incomplete
    target_length: Incomplete
    depth: Incomplete
    num_heads: Incomplete
    mlp_ratio: Incomplete
    qkv_bias: Incomplete
    init_values: Incomplete
    drop_rate: Incomplete
    attn_drop_rate: Incomplete
    f_min: Incomplete
    f_max: Incomplete
    center: Incomplete
    win_length: Incomplete
    hop_length: Incomplete
    sample_rate: Incomplete
    n_fft: Incomplete
    n_mels: Incomplete
    def __init__(
        self,
        embed_dim: int = 768,
        outputdim: int = 527,
        patch_size: int | tuple[int, int] = 16,
        patch_stride: int | tuple[int, int] = 16,
        input_channels: int = 1,
        target_length: int = 1012,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        init_values: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        center: bool = True,
        win_length: int = 512,
        hop_length: int = 160,
        sample_rate: int = 16000,
        n_fft: int = 512,
        n_mels: int = 64,
        **kwargs,
    ) -> None: ...

class MiDashengLMConfig(PretrainedConfig):
    model_type: str
    audio_encoder_config: Incomplete
    subsample_factor: Incomplete
    text_config: Incomplete
    audio_token_id: Incomplete
    def __init__(
        self,
        audio_encoder_config: dict | None = None,
        subsample_factor: int = 5,
        text_config: dict | None = None,
        audio_token_id: int | None = None,
        **kwargs,
    ) -> None: ...
