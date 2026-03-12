from _typeshed import Incomplete
from transformers.configuration_utils import PretrainedConfig

logger: Incomplete

class JAISConfig(PretrainedConfig):
    model_type: str
    keys_to_ignore_at_inference: Incomplete
    attribute_map: Incomplete
    vocab_size: Incomplete
    n_positions: Incomplete
    n_embd: Incomplete
    n_layer: Incomplete
    n_head: Incomplete
    n_inner: Incomplete
    activation_function: Incomplete
    resid_pdrop: Incomplete
    embd_pdrop: Incomplete
    attn_pdrop: Incomplete
    layer_norm_epsilon: Incomplete
    initializer_range: Incomplete
    scale_attn_weights: Incomplete
    use_cache: Incomplete
    scale_attn_by_inverse_layer_idx: Incomplete
    reorder_and_upcast_attn: Incomplete
    bos_token_id: Incomplete
    eos_token_id: Incomplete
    position_embedding_type: Incomplete
    mup_width_scale: Incomplete
    mup_embeddings_scale: Incomplete
    mup_output_alpha: Incomplete
    mup_scale_qk_dot_by_d: Incomplete
    alibi_scaling: Incomplete
    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner=None,
        activation_function: str = "gelu_new",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-05,
        initializer_range: float = 0.02,
        scale_attn_weights: bool = True,
        use_cache: bool = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        scale_attn_by_inverse_layer_idx: bool = False,
        reorder_and_upcast_attn: bool = False,
        position_embedding_type: str = "learned",
        mup_width_scale: float = 1.0,
        mup_embeddings_scale: float = 1.0,
        mup_output_alpha: float = 1.0,
        mup_scale_qk_dot_by_d: bool = False,
        alibi_scaling=None,
        architectures=None,
        **kwargs,
    ) -> None: ...
