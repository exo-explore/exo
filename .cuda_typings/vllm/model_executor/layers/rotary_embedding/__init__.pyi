import torch
from .base import RotaryEmbedding as RotaryEmbedding
from .deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding as DeepseekScalingRotaryEmbedding,
)
from .dual_chunk_rope import DualChunkRotaryEmbedding as DualChunkRotaryEmbedding
from .dynamic_ntk_alpha_rope import (
    DynamicNTKAlphaRotaryEmbedding as DynamicNTKAlphaRotaryEmbedding,
)
from .dynamic_ntk_scaling_rope import (
    DynamicNTKScalingRotaryEmbedding as DynamicNTKScalingRotaryEmbedding,
)
from .fope import FourierRotaryEmbedding as FourierRotaryEmbedding
from .linear_scaling_rope import (
    LinearScalingRotaryEmbedding as LinearScalingRotaryEmbedding,
)
from .llama3_rope import Llama3RotaryEmbedding as Llama3RotaryEmbedding
from .llama4_vision_rope import (
    Llama4VisionRotaryEmbedding as Llama4VisionRotaryEmbedding,
)
from .mrope import MRotaryEmbedding as MRotaryEmbedding
from .mrope_interleaved import (
    MRotaryEmbeddingInterleaved as MRotaryEmbeddingInterleaved,
)
from .ntk_scaling_rope import NTKScalingRotaryEmbedding as NTKScalingRotaryEmbedding
from .phi3_long_rope_scaled_rope import (
    Phi3LongRoPEScaledRotaryEmbedding as Phi3LongRoPEScaledRotaryEmbedding,
)
from .xdrope import XDRotaryEmbedding as XDRotaryEmbedding
from .yarn_scaling_rope import YaRNScalingRotaryEmbedding as YaRNScalingRotaryEmbedding
from typing import Any

def get_rope(
    head_size: int,
    max_position: int,
    is_neox_style: bool = True,
    rope_parameters: dict[str, Any] | None = None,
    dtype: torch.dtype | None = None,
    dual_chunk_attention_config: dict[str, Any] | None = None,
) -> RotaryEmbedding: ...
