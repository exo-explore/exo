from _typeshed import Incomplete
from transformers import PretrainedConfig

logger: Incomplete

class NemotronConfig(PretrainedConfig):
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
    norm_eps: Incomplete
    use_cache: Incomplete
    rope_parameters: Incomplete
    attention_bias: Incomplete
    attention_dropout: Incomplete
    mlp_bias: Incomplete
    def __init__(
        self,
        vocab_size: int = 256000,
        hidden_size: int = 6144,
        intermediate_size: int = 24576,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 48,
        head_dim=None,
        num_key_value_heads=None,
        hidden_act: str = "relu2",
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.0134,
        norm_eps: float = 1e-05,
        use_cache: bool = True,
        pad_token_id=None,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        tie_word_embeddings: bool = False,
        rope_parameters=None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        **kwargs,
    ) -> None: ...
