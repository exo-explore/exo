from .base import BaseRenderer as BaseRenderer
from _typeshed import Incomplete
from dataclasses import dataclass, field
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.tokenizers.registry import (
    tokenizer_args_from_config as tokenizer_args_from_config,
)
from vllm.utils.import_utils import resolve_obj_by_qualname as resolve_obj_by_qualname

logger: Incomplete

@dataclass
class RendererRegistry:
    renderers: dict[str, tuple[str, str]] = field(default_factory=dict)
    def register(self, renderer_mode: str, module: str, class_name: str) -> None: ...
    def load_renderer_cls(self, renderer_mode: str) -> type[BaseRenderer]: ...
    def load_renderer(
        self, renderer_mode: str, config: VllmConfig, tokenizer_kwargs: dict[str, Any]
    ) -> BaseRenderer: ...

RENDERER_REGISTRY: Incomplete

def renderer_from_config(config: VllmConfig, **kwargs): ...
