from dataclasses import dataclass
from typing import Any

from .qwen3_5 import Model as Qwen3_5Model

@dataclass
class ModelArgs:
    model_type: str
    text_config: dict[str, Any]

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs: ...

class Model(Qwen3_5Model):
    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]: ...
