from typing import Any, Optional

import mlx.nn as nn

class YarnRoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        traditional: bool = ...,
        max_position_embeddings: int = ...,
        base: float = ...,
        scaling_factor: float = ...,
        original_max_position_embeddings: int = ...,
        beta_fast: float = ...,
        beta_slow: float = ...,
        mscale: float = ...,
        mscale_all_dim: float = ...,
    ) -> None: ...

class Llama3RoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        traditional: bool = ...,
        max_position_embeddings: int = ...,
        base: float = ...,
        scaling_factor: float = ...,
        original_max_position_embeddings: int = ...,
        low_freq_factor: float = ...,
        high_freq_factor: float = ...,
    ) -> None: ...

class SuScaledRoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        traditional: bool = ...,
        max_position_embeddings: int = ...,
        base: float = ...,
        short_factor: Any = ...,
        long_factor: Any = ...,
        original_max_position_embeddings: int = ...,
    ) -> None: ...

def initialize_rope(
    dims: int,
    base: float = ...,
    traditional: bool = ...,
    scaling_config: Optional[dict[str, Any]] = ...,
    max_position_embeddings: Optional[int] = ...,
) -> nn.Module: ...
