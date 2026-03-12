from _typeshed import Incomplete
from transformers.configuration_utils import PretrainedConfig

class Olmo3Config(PretrainedConfig):
    model_type: str
    keys_to_ignore_at_inference: Incomplete
    vocab_size: Incomplete
    max_position_embeddings: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    num_hidden_layers: Incomplete
    num_attention_heads: Incomplete
    num_key_value_heads: Incomplete
    hidden_act: Incomplete
    initializer_range: Incomplete
    use_cache: Incomplete
    rope_parameters: Incomplete
    attention_bias: Incomplete
    attention_dropout: Incomplete
    rms_norm_eps: Incomplete
    sliding_window: Incomplete
    layer_types: Incomplete
    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads=None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: int = 1,
        bos_token_id=None,
        eos_token_id: int = 50279,
        tie_word_embeddings: bool = False,
        rope_parameters=None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-05,
        sliding_window: int = 4096,
        layer_types=None,
        **kwargs,
    ) -> None: ...
