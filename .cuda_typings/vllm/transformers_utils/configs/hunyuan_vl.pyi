from _typeshed import Incomplete
from transformers import PretrainedConfig

class HunYuanVLVisionConfig(PretrainedConfig):
    model_type: str
    base_config_key: str
    hidden_act: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    interpolate_mode: Incomplete
    learnable_mlp_pooling_size: Incomplete
    num_attention_heads: Incomplete
    num_key_value_heads: Incomplete
    num_channels: Incomplete
    num_hidden_layers: Incomplete
    out_hidden_size: Incomplete
    patch_size: Incomplete
    remove_prenorm: Incomplete
    spatial_merge_size: Incomplete
    temporal_patch_size: Incomplete
    rms_norm_eps: Incomplete
    resize_resolution: Incomplete
    img_max_token_num: Incomplete
    max_image_size: Incomplete
    min_image_size: Incomplete
    video_max_image_size: Incomplete
    video_min_image_size: Incomplete
    anyres_vit_max_image_size: Incomplete
    max_vit_seq_len: Incomplete
    text_hidden_size: Incomplete
    def __init__(
        self,
        hidden_act: str = "gelu",
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        interpolate_mode: str = "bilinear",
        rms_norm_eps: float = 1e-05,
        learnable_mlp_pooling_size: int = 0,
        num_attention_heads: int = 16,
        num_key_value_heads=None,
        num_channels: int = 3,
        num_hidden_layers: int = 27,
        out_hidden_size: int = 4096,
        patch_size: int = 16,
        remove_prenorm: bool = True,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 1,
        resize_resolution: int = 2048,
        img_max_token_num: int = 4096,
        max_image_size: int = 2048,
        video_max_image_size: int = 768,
        video_min_image_size: int = 256,
        min_image_size: int = 512,
        anyres_vit_max_image_size: int = 2048,
        max_vit_seq_len: int = 16384,
        text_hidden_size: int = 3072,
        **kwargs,
    ) -> None: ...

class HunYuanVLTextConfig(PretrainedConfig):
    model_type: str
    keys_to_ignore_at_inference: Incomplete
    vocab_size: Incomplete
    max_position_embeddings: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    num_hidden_layers: Incomplete
    num_attention_heads: Incomplete
    head_dim: Incomplete
    num_key_value_heads: Incomplete
    hidden_act: Incomplete
    initializer_range: Incomplete
    rms_norm_eps: Incomplete
    pretraining_tp: Incomplete
    use_cache: Incomplete
    rope_theta: Incomplete
    rope_scaling: Incomplete
    attention_bias: Incomplete
    attention_dropout: Incomplete
    def __init__(
        self,
        vocab_size: int = 290943,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads=None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-05,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        eod_token_id: int = 3,
        pretraining_tp: int = 1,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling=None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        head_dim=None,
        **kwargs,
    ) -> None: ...

class HunYuanVLConfig(PretrainedConfig):
    model_type: str
    sub_configs: Incomplete
    keys_to_ignore_at_inference: Incomplete
    vision_config: Incomplete
    text_config: Incomplete
    image_token_id: Incomplete
    im_start_id: Incomplete
    im_end_id: Incomplete
    im_newline_id: Incomplete
    video_start_id: Incomplete
    video_end_id: Incomplete
    def __init__(
        self,
        text_config=None,
        vision_config=None,
        im_start_id: int = 120118,
        im_end_id: int = 120119,
        image_token_id: int = 120120,
        im_newline_id: int = 120121,
        video_start_id: int = 120122,
        video_end_id: int = 120123,
        **kwargs,
    ) -> None: ...
    def __setattr__(self, key, value) -> None: ...
    def __getattribute__(self, key): ...
