from _typeshed import Incomplete
from transformers import PretrainedConfig

class ChatGLMConfig(PretrainedConfig):
    model_type: str
    attribute_map: Incomplete
    num_layers: Incomplete
    vocab_size: Incomplete
    padded_vocab_size: Incomplete
    hidden_size: Incomplete
    ffn_hidden_size: Incomplete
    kv_channels: Incomplete
    num_attention_heads: Incomplete
    seq_length: Incomplete
    max_position_embeddings: Incomplete
    hidden_dropout: Incomplete
    attention_dropout: Incomplete
    layernorm_epsilon: Incomplete
    rmsnorm: Incomplete
    apply_residual_connection_post_layernorm: Incomplete
    post_layer_norm: Incomplete
    add_bias_linear: Incomplete
    add_qkv_bias: Incomplete
    bias_dropout_fusion: Incomplete
    multi_query_attention: Incomplete
    multi_query_group_num: Incomplete
    apply_query_key_layer_scaling: Incomplete
    attention_softmax_in_fp32: Incomplete
    fp32_residual_connection: Incomplete
    quantization_bit: Incomplete
    pre_seq_len: Incomplete
    prefix_projection: Incomplete
    interleaved_qkv: Incomplete
    def __init__(
        self,
        num_layers: int = 28,
        padded_vocab_size: int = 65024,
        hidden_size: int = 4096,
        ffn_hidden_size: int = 13696,
        kv_channels: int = 128,
        num_attention_heads: int = 32,
        seq_length: int = 2048,
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        layernorm_epsilon: float = 1e-05,
        rmsnorm: bool = True,
        apply_residual_connection_post_layernorm: bool = False,
        post_layer_norm: bool = True,
        add_bias_linear: bool = False,
        add_qkv_bias: bool = False,
        interleaved_qkv: bool = False,
        bias_dropout_fusion: bool = True,
        multi_query_attention: bool = False,
        multi_query_group_num: int = 1,
        apply_query_key_layer_scaling: bool = True,
        attention_softmax_in_fp32: bool = True,
        fp32_residual_connection: bool = False,
        quantization_bit: int = 0,
        pre_seq_len=None,
        prefix_projection: bool = False,
        **kwargs,
    ) -> None: ...
