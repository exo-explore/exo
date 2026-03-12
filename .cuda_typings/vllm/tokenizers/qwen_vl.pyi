import abc
from .hf import HfTokenizer as HfTokenizer, get_cached_tokenizer as get_cached_tokenizer
from .protocol import TokenizerLike as TokenizerLike
from collections.abc import Set as Set

def get_qwen_vl_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer: ...

class QwenVLTokenizer(TokenizerLike, metaclass=abc.ABCMeta):
    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> HfTokenizer: ...
