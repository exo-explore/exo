from _typeshed import Incomplete
from transformers.configuration_utils import PretrainedConfig
from typing import Any

class FlexOlmoConfig(PretrainedConfig):
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
    rms_norm_eps: Incomplete
    use_cache: Incomplete
    rope_parameters: Incomplete
    attention_bias: Incomplete
    attention_dropout: Incomplete
    num_experts_per_tok: Incomplete
    num_experts: Incomplete
    output_router_logits: Incomplete
    router_aux_loss_coef: Incomplete
    norm_topk_prob: Incomplete
    def __init__(
        self,
        vocab_size: int = 100352,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads=None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-06,
        use_cache: bool = True,
        pad_token_id: int = 100277,
        bos_token_id=None,
        eos_token_id: int = 100257,
        tie_word_embeddings: bool = False,
        rope_parameters: dict[str, Any] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        num_experts_per_tok: int = 5,
        num_experts: int = 7,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.01,
        norm_topk_prob: bool = False,
        **kwargs,
    ) -> None: ...
