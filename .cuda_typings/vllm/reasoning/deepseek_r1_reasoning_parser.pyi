from collections.abc import Sequence
from vllm.entrypoints.openai.engine.protocol import DeltaMessage as DeltaMessage
from vllm.reasoning.basic_parsers import (
    BaseThinkingReasoningParser as BaseThinkingReasoningParser,
)

class DeepSeekR1ReasoningParser(BaseThinkingReasoningParser):
    @property
    def start_token(self) -> str: ...
    @property
    def end_token(self) -> str: ...
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None: ...
