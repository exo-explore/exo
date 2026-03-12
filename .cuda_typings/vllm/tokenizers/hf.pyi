import abc
from .protocol import TokenizerLike as TokenizerLike
from pathlib import Path
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import TypeAlias
from vllm.transformers_utils.config import (
    get_sentence_transformer_tokenizer_config as get_sentence_transformer_tokenizer_config,
)

HfTokenizer: TypeAlias = PreTrainedTokenizer | PreTrainedTokenizerFast

def get_cached_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer: ...

class CachedHfTokenizer(TokenizerLike, metaclass=abc.ABCMeta):
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> HfTokenizer: ...
