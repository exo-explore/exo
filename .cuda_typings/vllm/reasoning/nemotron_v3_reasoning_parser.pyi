from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest as ResponsesRequest,
)
from vllm.reasoning.deepseek_r1_reasoning_parser import (
    DeepSeekR1ReasoningParser as DeepSeekR1ReasoningParser,
)

class NemotronV3ReasoningParser(DeepSeekR1ReasoningParser):
    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]: ...
