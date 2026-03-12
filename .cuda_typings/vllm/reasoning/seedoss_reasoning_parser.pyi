from vllm.reasoning.basic_parsers import (
    BaseThinkingReasoningParser as BaseThinkingReasoningParser,
)

class SeedOSSReasoningParser(BaseThinkingReasoningParser):
    @property
    def start_token(self) -> str: ...
    @property
    def end_token(self) -> str: ...
