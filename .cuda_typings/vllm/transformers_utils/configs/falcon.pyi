from _typeshed import Incomplete
from transformers.configuration_utils import PretrainedConfig

class RWConfig(PretrainedConfig):
    model_type: str
    keys_to_ignore_at_inference: Incomplete
    attribute_map: Incomplete
    vocab_size: Incomplete
    hidden_size: Incomplete
    n_layer: Incomplete
    n_head: Incomplete
    layer_norm_epsilon: Incomplete
    initializer_range: Incomplete
    use_cache: Incomplete
    hidden_dropout: Incomplete
    attention_dropout: Incomplete
    bos_token_id: Incomplete
    eos_token_id: Incomplete
    multi_query: Incomplete
    n_head_kv: Incomplete
    alibi: Incomplete
    bias: Incomplete
    parallel_attn: Incomplete
    new_decoder_architecture: Incomplete
    def __init__(
        self,
        vocab_size: int = 250880,
        hidden_size: int = 64,
        n_layer: int = 2,
        n_head: int = 8,
        layer_norm_epsilon: float = 1e-05,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        multi_query: bool = True,
        n_head_kv=None,
        alibi: bool = False,
        bias: bool = False,
        parallel_attn: bool = False,
        new_decoder_architecture: bool = False,
        **kwargs,
    ) -> None: ...
    @property
    def head_dim(self): ...
    @property
    def rotary(self): ...
