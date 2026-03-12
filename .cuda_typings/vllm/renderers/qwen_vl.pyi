import abc
from .base import BaseRenderer as BaseRenderer
from .hf import HfRenderer as HfRenderer
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.tokenizers import cached_get_tokenizer as cached_get_tokenizer
from vllm.tokenizers.qwen_vl import QwenVLTokenizer as QwenVLTokenizer

class QwenVLRenderer(BaseRenderer[QwenVLTokenizer], metaclass=abc.ABCMeta):
    @classmethod
    def from_config(
        cls, config: VllmConfig, tokenizer_kwargs: dict[str, Any]
    ) -> HfRenderer: ...
