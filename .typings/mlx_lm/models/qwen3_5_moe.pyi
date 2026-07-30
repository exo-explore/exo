from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .cache import ArraysCache, KVCache
from .qwen3_5 import DecoderLayer, Model as Qwen3_5Model, TextModel

@dataclass
class ModelArgs:
    model_type: str
    text_config: dict[str, Any]

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs: ...

class Model(Qwen3_5Model):
    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]: ...
