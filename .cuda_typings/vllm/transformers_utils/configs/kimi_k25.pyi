from _typeshed import Incomplete
from transformers import DeepseekV3Config
from transformers.configuration_utils import PretrainedConfig

class KimiK25VisionConfig(PretrainedConfig):
    model_type: str
    patch_size: Incomplete
    init_pos_emb_height: Incomplete
    init_pos_emb_width: Incomplete
    init_pos_emb_time: Incomplete
    pos_emb_type: Incomplete
    num_attention_heads: Incomplete
    num_hidden_layers: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    merge_kernel_size: Incomplete
    video_attn_type: Incomplete
    merge_type: Incomplete
    mm_projector_type: Incomplete
    mm_hidden_size: Incomplete
    projector_hidden_act: Incomplete
    projector_ln_eps: Incomplete
    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        num_attention_heads: int = 16,
        num_hidden_layers: int = 27,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        merge_kernel_size: tuple[int, int] = (2, 2),
        video_attn_type: str = "spatial_temporal",
        merge_type: str = "sd2_tpool",
        mm_projector_type: str = "patchmerger",
        mm_hidden_size: int | None = None,
        projector_hidden_act: str = "gelu",
        projector_ln_eps: float = 1e-05,
        **kwargs,
    ) -> None: ...

class KimiK25Config(PretrainedConfig):
    model_type: str
    vision_config: Incomplete
    text_config: Incomplete
    ignore_index: Incomplete
    media_placeholder_token_id: Incomplete
    use_unified_vision_chunk: Incomplete
    video_placeholder: Incomplete
    quantization_config: Incomplete
    def __init__(
        self,
        vision_config: dict | KimiK25VisionConfig | None = None,
        text_config: dict | DeepseekV3Config | None = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        use_unified_vision_chunk: bool = False,
        video_placeholder: str = "<|kimi_k25_video_placeholder|>",
        **kwargs,
    ) -> None: ...
    @property
    def hidden_size(self) -> int: ...
    @property
    def vocab_size(self) -> int: ...
