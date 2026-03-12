from .hf import HfRenderer as HfRenderer, HfTokenizer as HfTokenizer
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.tokenizers.kimi_audio import KimiAudioTokenizer as KimiAudioTokenizer
from vllm.tokenizers.registry import get_tokenizer as get_tokenizer

class KimiAudioRenderer(HfRenderer):
    @classmethod
    def from_config(
        cls, config: VllmConfig, tokenizer_kwargs: dict[str, Any]
    ) -> HfRenderer: ...
