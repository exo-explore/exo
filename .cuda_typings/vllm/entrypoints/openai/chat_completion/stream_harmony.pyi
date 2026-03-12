from openai_harmony import StreamableParser as StreamableParser
from typing import NamedTuple
from vllm.entrypoints.chat_utils import make_tool_call_id as make_tool_call_id
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall as DeltaFunctionCall,
    DeltaMessage as DeltaMessage,
    DeltaToolCall as DeltaToolCall,
)

class TokenState(NamedTuple):
    channel: str | None
    recipient: str | None
    text: str

def extract_harmony_streaming_delta(
    harmony_parser: StreamableParser,
    token_states: list[TokenState],
    prev_recipient: str | None,
    include_reasoning: bool,
) -> tuple[DeltaMessage | None, bool]: ...
