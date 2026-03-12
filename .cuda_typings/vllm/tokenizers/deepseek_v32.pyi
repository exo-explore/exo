import abc
from .deepseek_v32_encoding import encode_messages as encode_messages
from .hf import HfTokenizer as HfTokenizer, get_cached_tokenizer as get_cached_tokenizer
from .protocol import TokenizerLike as TokenizerLike
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam as ChatCompletionMessageParam,
)

def get_deepseek_v32_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer: ...

class DeepseekV32Tokenizer(TokenizerLike, metaclass=abc.ABCMeta):
    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> HfTokenizer: ...
