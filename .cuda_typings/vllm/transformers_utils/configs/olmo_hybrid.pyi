from _typeshed import Incomplete
from transformers.configuration_utils import PretrainedConfig

class OlmoHybridConfig(PretrainedConfig):
    model_type: str
    keys_to_ignore_at_inference: Incomplete
    base_model_tp_plan: Incomplete
    base_model_pp_plan: Incomplete
    layer_types: Incomplete
    linear_num_key_heads: Incomplete
    linear_num_value_heads: Incomplete
    linear_key_head_dim: Incomplete
    linear_value_head_dim: Incomplete
    linear_a_log_min: Incomplete
    linear_a_log_max: Incomplete
    linear_dt_min: Incomplete
    linear_dt_max: Incomplete
    linear_dt_init_floor: Incomplete
    linear_conv_kernel_dim: Incomplete
    linear_allow_neg_eigval: Incomplete
    vocab_size: Incomplete
    max_position_embeddings: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    num_hidden_layers: Incomplete
    num_attention_heads: Incomplete
    num_key_value_heads: Incomplete
    hidden_act: Incomplete
    initializer_range: Incomplete
    rms_norm_eps: Incomplete
    use_cache: Incomplete
    attention_bias: Incomplete
    attention_dropout: Incomplete
    rope_parameters: Incomplete
    tie_word_embeddings: Incomplete
    pad_token_id: Incomplete
    bos_token_id: Incomplete
    eos_token_id: Incomplete
    def __init__(
        self,
        vocab_size: int | None = 100352,
        hidden_size: int | None = 3840,
        intermediate_size: int | None = 11008,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 30,
        num_key_value_heads: int | None = None,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 65536,
        initializer_range: float | None = 0.02,
        use_cache: bool | None = True,
        pad_token_id: int | None = 100277,
        bos_token_id: int | None = None,
        eos_token_id: int | None = 100257,
        tie_word_embeddings: bool | None = False,
        rope_parameters=None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        rms_norm_eps: float | None = 1e-06,
        layer_types: list[str] | None = None,
        linear_num_key_heads: int | None = None,
        linear_num_value_heads: int | None = None,
        linear_key_head_dim: int | None = None,
        linear_value_head_dim: int | None = None,
        linear_a_log_min: float = 0.0,
        linear_a_log_max: float = 16.0,
        linear_dt_min: float = 0.001,
        linear_dt_max: float = 0.1,
        linear_dt_init_floor: float = 0.0001,
        linear_conv_kernel_dim: int = 4,
        linear_allow_neg_eigval: bool = True,
        **kwargs,
    ) -> None: ...
