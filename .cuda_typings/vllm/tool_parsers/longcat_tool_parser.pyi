from _typeshed import Incomplete
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.tool_parsers.hermes_tool_parser import (
    Hermes2ProToolParser as Hermes2ProToolParser,
)

class LongcatFlashToolParser(Hermes2ProToolParser):
    tool_call_start_token: str
    tool_call_end_token: str
    tool_call_regex: Incomplete
    tool_call_start_token_ids: Incomplete
    tool_call_end_token_ids: Incomplete
    tool_call_start_token_array: Incomplete
    tool_call_end_token_array: Incomplete
    def __init__(self, tokenizer: TokenizerLike) -> None: ...
