from _typeshed import Incomplete
from pathlib import Path
from transformers import BatchEncoding
from typing import Any, overload
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam as ChatCompletionMessageParam,
)
from vllm.logger import init_logger as init_logger
from vllm.tokenizers.protocol import TokenizerLike as TokenizerLike

logger: Incomplete

class KimiAudioTokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> KimiAudioTokenizer: ...
    name_or_path: Incomplete
    def __init__(
        self, *, vocab_file: Path, name_or_path: str, truncation_side: str
    ) -> None: ...
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
    @property
    def added_tokens_decoder(self) -> dict[int, Any]: ...
    @added_tokens_decoder.setter
    def added_tokens_decoder(self, value: dict[int, Any]) -> None: ...
    def get_vocab(self) -> dict[str, int]: ...
    def __len__(self) -> int: ...
    def get_added_vocab(self) -> dict[str, int]: ...
    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> list[int]: ...
    def decode(
        self, ids: list[int] | int, skip_special_tokens: bool = False
    ) -> str: ...
    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int: ...
    @overload
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...
    def convert_ids_to_tokens(
        self, ids: list[int], skip_special_tokens: bool = False
    ) -> list[str]: ...
    def convert_tokens_to_string(self, tokens: list[str]) -> str: ...
    def __call__(
        self,
        text: str | list[str],
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs,
    ) -> BatchEncoding: ...
    def get_chat_template(
        self, chat_template: str | None, tools: list[dict[str, Any]] | None = None
    ) -> str | None: ...
    def apply_chat_template(
        self,
        messages: list[ChatCompletionMessageParam] | None = None,
        tools: list[dict[str, Any]] | None = None,
        chat_template: str | None = None,
        tokenize: bool = False,
        **kwargs,
    ) -> str | list[int]: ...
