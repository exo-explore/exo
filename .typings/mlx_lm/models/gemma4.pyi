from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from . import gemma4_text
from .base import BaseModelArgs
from .cache import KVCache, RotatingKVCache

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: Optional[dict[str, Any]]
    vocab_size: int

    def __post_init__(self) -> None: ...

class Model(nn.Module):
    args: ModelArgs
    model_type: str
    language_model: gemma4_text.Model

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> mx.array: ...
    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]: ...
    @property
    def layers(self) -> list[gemma4_text.DecoderLayer]: ...
    @property
    def quant_predicate(self) -> Any: ...
    def make_cache(self) -> list[KVCache | RotatingKVCache]: ...
