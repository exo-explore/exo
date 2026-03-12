from _typeshed import Incomplete
from transformers.configuration_utils import PretrainedConfig

__all__ = ["Qwen3ASRConfig", "Qwen3ASRThinkerConfig", "Qwen3ASRAudioEncoderConfig"]

class Qwen3ASRAudioEncoderConfig(PretrainedConfig):
    model_type: str
    num_mel_bins: Incomplete
    d_model: Incomplete
    encoder_layers: Incomplete
    encoder_attention_heads: Incomplete
    encoder_ffn_dim: Incomplete
    dropout: Incomplete
    attention_dropout: Incomplete
    activation_function: Incomplete
    activation_dropout: Incomplete
    num_hidden_layers: Incomplete
    initializer_range: Incomplete
    scale_embedding: Incomplete
    max_source_positions: Incomplete
    n_window: Incomplete
    output_dim: Incomplete
    n_window_infer: Incomplete
    conv_chunksize: Incomplete
    downsample_hidden_size: Incomplete
    def __init__(
        self,
        num_mel_bins: int = 128,
        encoder_layers: int = 32,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        d_model: int = 1280,
        dropout: int = 0,
        attention_dropout: int = 0,
        activation_function: str = "gelu",
        activation_dropout: int = 0,
        scale_embedding: bool = False,
        initializer_range: float = 0.02,
        max_source_positions: int = 1500,
        n_window: int = 100,
        output_dim: int = 3584,
        n_window_infer: int = 400,
        conv_chunksize: int = 500,
        downsample_hidden_size: int = 480,
        **kwargs,
    ) -> None: ...

class Qwen3ASRTextConfig(PretrainedConfig):
    model_type: str
    base_config_key: str
    vocab_size: Incomplete
    max_position_embeddings: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    num_hidden_layers: Incomplete
    num_attention_heads: Incomplete
    num_key_value_heads: Incomplete
    head_dim: Incomplete
    hidden_act: Incomplete
    initializer_range: Incomplete
    rms_norm_eps: Incomplete
    use_cache: Incomplete
    rope_theta: Incomplete
    rope_scaling: Incomplete
    attention_bias: Incomplete
    attention_dropout: Incomplete
    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 128000,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-06,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 5000000.0,
        rope_scaling=None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        **kwargs,
    ) -> None: ...

class Qwen3ASRThinkerConfig(PretrainedConfig):
    model_type: str
    attribute_map: Incomplete
    sub_configs: Incomplete
    user_token_id: Incomplete
    audio_start_token_id: Incomplete
    initializer_range: Incomplete
    audio_config: Incomplete
    text_config: Incomplete
    audio_token_id: Incomplete
    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id: int = 151646,
        audio_start_token_id: int = 151647,
        user_token_id: int = 872,
        initializer_range: float = 0.02,
        **kwargs,
    ) -> None: ...

class Qwen3ASRConfig(PretrainedConfig):
    model_type: str
    sub_configs: Incomplete
    thinker_config: Incomplete
    support_languages: Incomplete
    def __init__(
        self, thinker_config=None, support_languages=None, **kwargs
    ) -> None: ...
    def get_text_config(self, decoder: bool = False) -> PretrainedConfig: ...
