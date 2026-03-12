import torch
from _typeshed import Incomplete
from collections.abc import Sequence
from typing import TypeAlias
from vllm.tokenizers import TokenizerLike as TokenizerLike

LogitsProcessor: TypeAlias

def get_bad_words_logits_processors(
    bad_words: list[str], tokenizer: TokenizerLike
) -> list[LogitsProcessor]: ...

class NoBadWordsLogitsProcessor:
    bad_words_ids: Incomplete
    word_bias: torch.FloatTensor
    def __init__(self, bad_words_ids: list[list[int]]) -> None: ...
    def __call__(
        self, past_tokens_ids: Sequence[int], logits: torch.FloatTensor
    ) -> torch.Tensor: ...
