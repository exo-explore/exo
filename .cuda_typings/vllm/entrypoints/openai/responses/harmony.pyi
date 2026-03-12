from _typeshed import Incomplete
from openai.types.responses import (
    ResponseOutputItem as ResponseOutputItem,
    ResponseReasoningItem,
)
from openai_harmony import Message, StreamableParser as StreamableParser
from vllm.entrypoints.openai.parser.harmony_utils import (
    BUILTIN_TOOL_TO_MCP_SERVER_LABEL as BUILTIN_TOOL_TO_MCP_SERVER_LABEL,
    flatten_chat_text_content as flatten_chat_text_content,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponseInputOutputItem as ResponseInputOutputItem,
    ResponsesRequest as ResponsesRequest,
)
from vllm.logger import init_logger as init_logger
from vllm.utils import random_uuid as random_uuid

logger: Incomplete

def response_input_to_harmony(
    response_msg: ResponseInputOutputItem,
    prev_responses: list[ResponseOutputItem | ResponseReasoningItem],
) -> Message: ...
def response_previous_input_to_harmony(chat_msg) -> list[Message]: ...
def construct_harmony_previous_input_messages(
    request: ResponsesRequest,
) -> list[Message]: ...
def harmony_to_response_output(message: Message) -> list[ResponseOutputItem]: ...
def parser_state_to_response_output(
    parser: StreamableParser,
) -> list[ResponseOutputItem]: ...
