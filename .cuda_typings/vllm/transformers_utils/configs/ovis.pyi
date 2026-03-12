from _typeshed import Incomplete
from transformers import PretrainedConfig
from typing import Any

class AIMv2Config(PretrainedConfig):
    model_type: str
    hidden_size: Incomplete
    intermediate_size: Incomplete
    num_hidden_layers: Incomplete
    num_attention_heads: Incomplete
    num_channels: Incomplete
    patch_size: Incomplete
    image_size: Incomplete
    attention_dropout: Incomplete
    rms_norm_eps: Incomplete
    projection_dropout: Incomplete
    qkv_bias: Incomplete
    use_bias: Incomplete
    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 2816,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 8,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 14,
        rms_norm_eps: float = 1e-05,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        qkv_bias: bool = False,
        use_bias: bool = False,
        **kwargs: Any,
    ) -> None: ...

class BaseVisualTokenizerConfig(PretrainedConfig):
    vocab_size: Incomplete
    tokenize_function: Incomplete
    tau: Incomplete
    depths: Incomplete
    backbone_kwargs: Incomplete
    drop_cls_token: Incomplete
    backbone_config: Incomplete
    hidden_stride: Incomplete
    def __init__(
        self,
        vocab_size: int = 16384,
        tokenize_function: str = "softmax",
        tau: float = 1.0,
        depths=None,
        drop_cls_token: bool = False,
        backbone_config: PretrainedConfig | dict | None = None,
        hidden_stride: int = 1,
        **kwargs,
    ) -> None: ...

class Aimv2VisualTokenizerConfig(BaseVisualTokenizerConfig):
    model_type: str
    drop_cls_token: bool
    def __init__(self, **kwargs) -> None: ...

class SiglipVisualTokenizerConfig(BaseVisualTokenizerConfig):
    model_type: str
    drop_cls_token: bool
    def __init__(self, **kwargs) -> None: ...

class OvisConfig(PretrainedConfig):
    model_type: str
    text_config: Incomplete
    visual_tokenizer_config: Incomplete
    multimodal_max_length: Incomplete
    hidden_size: Incomplete
    conversation_formatter_class: Incomplete
    llm_attn_implementation: Incomplete
    disable_tie_weight: Incomplete
    def __init__(
        self,
        llm_config: PretrainedConfig | dict | None = None,
        visual_tokenizer_config: PretrainedConfig | dict | None = None,
        multimodal_max_length: int = 8192,
        hidden_size=None,
        conversation_formatter_class=None,
        llm_attn_implementation=None,
        disable_tie_weight: bool = False,
        **kwargs,
    ) -> None: ...
