from _typeshed import Incomplete
from transformers import PretrainedConfig

class FunAudioChatAudioEncoderConfig(PretrainedConfig):
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
    bos_token_id: Incomplete
    codebook_size: Incomplete
    continuous_features_mode: Incomplete
    crq_transformer_config: Incomplete
    eos_token_id: Incomplete
    group_size: Incomplete
    enable_audio_invert_tower: Incomplete
    pad_token_id: Incomplete
    def __init__(
        self,
        _attn_implementation: str | None = None,
        num_mel_bins: int = 128,
        encoder_layers: int = 32,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        d_model: int = 1280,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_function: str = "gelu",
        activation_dropout: float = 0.0,
        scale_embedding: bool = False,
        initializer_range: float = 0.02,
        max_source_positions: int = 1500,
        n_window: int = 100,
        output_dim: int = 3584,
        bos_token_id: int | None = None,
        codebook_size: int | None = None,
        continuous_features_mode: str = "replace",
        crq_transformer_config: dict | None = None,
        eos_token_id: int | None = None,
        group_size: int = 5,
        enable_audio_invert_tower: bool = True,
        pad_token_id: int | None = None,
        **kwargs,
    ) -> None: ...

class FunAudioChatConfig(PretrainedConfig):
    model_type: str
    attribute_map: Incomplete
    audio_token_index: Incomplete
    ignore_index: Incomplete
    audio_config: Incomplete
    text_config: Incomplete
    hidden_size: Incomplete
    def __init__(
        self,
        audio_config: PretrainedConfig | dict | None = None,
        text_config: PretrainedConfig | dict | None = None,
        audio_token_index: int = 151646,
        ignore_index: int = -100,
        hidden_size: int | None = None,
        **kwargs,
    ) -> None: ...
