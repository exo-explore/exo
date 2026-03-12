from .protocol import TokenizerLike as TokenizerLike
from _typeshed import Incomplete
from mistral_common.protocol.instruct.request import (
    ChatCompletionRequest as MistralChatCompletionRequest,
)
from pathlib import Path
from transformers import BatchEncoding as BatchEncoding
from transformers.tokenization_mistral_common import (
    MistralCommonTokenizer as MistralCommonBackend,
)
from typing import Any, overload
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam as ChatCompletionMessageParam,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as ChatCompletionRequest,
)
from vllm.logger import init_logger as init_logger

logger: Incomplete

def maybe_serialize_tool_calls(request: MistralChatCompletionRequest): ...
def truncate_tool_call_ids(request: MistralChatCompletionRequest): ...
def validate_request_params(request: ChatCompletionRequest): ...

class MistralTokenizer(TokenizerLike):
    IS_MISTRAL_TOKENIZER: bool
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> MistralTokenizer: ...
    transformers_tokenizer: Incomplete
    mistral: Incomplete
    instruct: Incomplete
    tokenizer: Incomplete
    version: int
    is_tekken: Incomplete
    is_spm: Incomplete
    def __init__(self, tokenizer: MistralCommonBackend) -> None: ...
    def num_special_tokens_to_add(self) -> int: ...
    @property
    def all_special_tokens(self) -> list[str]: ...
    @property
    def all_special_ids(self) -> list[int]: ...
    @property
    def bos_token_id(self) -> int: ...
    @property
    def eos_token_id(self) -> int: ...
    @property
    def pad_token_id(self) -> int: ...
    @property
    def is_fast(self) -> bool: ...
    @property
    def vocab_size(self) -> int: ...
    @property
    def max_token_id(self) -> int: ...
    @property
    def max_chars_per_token(self) -> int: ...
    @property
    def truncation_side(self) -> str: ...
    def __hash__(self) -> int: ...
    def __len__(self) -> int: ...
    def __call__(
        self,
        text: str | list[str],
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> BatchEncoding: ...
    @property
    def vocab(self) -> list[str]: ...
    def get_vocab(self) -> dict[str, int]: ...
    def get_added_vocab(self) -> dict[str, int]: ...
    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]: ...
    def apply_chat_template(
        self,
        messages: list["ChatCompletionMessageParam"],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> list[int]: ...
    def decode(
        self, ids: list[int] | int, skip_special_tokens: bool = False
    ) -> str: ...
    def batch_decode(
        self, ids: list[list[int]] | list[int], skip_special_tokens: bool = False
    ) -> str: ...
    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int: ...
    @overload
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...
    def convert_tokens_to_string(self, tokens: list[str]) -> str: ...
    def convert_ids_to_tokens(
        self, ids: list[int], skip_special_tokens: bool = False
    ) -> list[str]: ...
