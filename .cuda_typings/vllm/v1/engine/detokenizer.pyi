import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from tokenizers import Tokenizer as Tokenizer
from transformers import PreTrainedTokenizerFast
from vllm.logger import init_logger as init_logger
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.tokenizers.detokenizer_utils import (
    convert_prompt_ids_to_tokens as convert_prompt_ids_to_tokens,
    detokenize_incrementally as detokenize_incrementally,
)
from vllm.utils import (
    length_from_prompt_token_ids_or_embeds as length_from_prompt_token_ids_or_embeds,
)
from vllm.v1.engine import EngineCoreRequest as EngineCoreRequest

logger: Incomplete
USE_FAST_DETOKENIZER: Incomplete
INVALID_PREFIX_ERR_MSG: str

class IncrementalDetokenizer:
    token_ids: list[int]
    def __init__(self) -> None: ...
    @property
    def output_token_ids(self) -> list[int]: ...
    def num_output_tokens(self) -> int: ...
    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None: ...
    def get_next_output_text(self, finished: bool, delta: bool) -> str: ...
    @classmethod
    def from_new_request(
        cls, tokenizer: TokenizerLike | None, request: EngineCoreRequest
    ) -> IncrementalDetokenizer: ...

class BaseIncrementalDetokenizer(IncrementalDetokenizer, ABC, metaclass=abc.ABCMeta):
    stop: Incomplete
    min_tokens: Incomplete
    include_stop_str_in_output: Incomplete
    stop_buffer_length: Incomplete
    output_text: str
    def __init__(self, request: EngineCoreRequest) -> None: ...
    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None: ...
    @abstractmethod
    def decode_next(self, next_token_id: int) -> str: ...
    def get_next_output_text(self, finished: bool, delta: bool) -> str: ...

class FastIncrementalDetokenizer(BaseIncrementalDetokenizer):
    request_id: Incomplete
    skip_special_tokens: Incomplete
    tokenizer: Tokenizer
    stream: Incomplete
    spaces_between_special_tokens: Incomplete
    last_special: bool
    added_token_ids: Incomplete
    def __init__(
        self, tokenizer: PreTrainedTokenizerFast, request: EngineCoreRequest
    ) -> None: ...
    def decode_next(self, next_token_id: int) -> str: ...

class SlowIncrementalDetokenizer(BaseIncrementalDetokenizer):
    tokenizer: Incomplete
    prompt_len: Incomplete
    tokens: Incomplete
    prefix_offset: int
    read_offset: int
    skip_special_tokens: Incomplete
    spaces_between_special_tokens: Incomplete
    def __init__(
        self, tokenizer: TokenizerLike, request: EngineCoreRequest
    ) -> None: ...
    @property
    def output_token_ids(self) -> list[int]: ...
    def num_output_tokens(self) -> int: ...
    def decode_next(self, next_token_id: int) -> str: ...

def check_stop_strings(
    output_text: str, new_char_count: int, stop: list[str], include_in_output: bool
) -> tuple[str, int] | None: ...
