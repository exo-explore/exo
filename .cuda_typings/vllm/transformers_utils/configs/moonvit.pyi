from _typeshed import Incomplete
from transformers.configuration_utils import PretrainedConfig

class MoonViTConfig(PretrainedConfig):
    model_type: str
    patch_size: Incomplete
    init_pos_emb_height: Incomplete
    init_pos_emb_width: Incomplete
    num_hidden_layers: Incomplete
    num_attention_heads: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    merge_kernel_size: Incomplete
    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 27,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        merge_kernel_size: tuple[int, int] = (2, 2),
        **kwargs,
    ) -> None: ...
