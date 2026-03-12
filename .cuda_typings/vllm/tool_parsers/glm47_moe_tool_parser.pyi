from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.tool_parsers.glm4_moe_tool_parser import (
    Glm4MoeModelToolParser as Glm4MoeModelToolParser,
)

logger: Incomplete

class Glm47MoeModelToolParser(Glm4MoeModelToolParser):
    func_detail_regex: Incomplete
    func_arg_regex: Incomplete
    def __init__(self, tokenizer: TokenizerLike) -> None: ...
