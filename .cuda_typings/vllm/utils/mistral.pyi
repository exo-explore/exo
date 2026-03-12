import vllm.tokenizers.mistral as mt
from typing import TypeGuard
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.utils.import_utils import LazyLoader as LazyLoader

def is_mistral_tokenizer(
    obj: TokenizerLike | None,
) -> TypeGuard[mt.MistralTokenizer]: ...
