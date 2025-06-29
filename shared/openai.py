from typing import TYPE_CHECKING, Literal, TypeAlias, get_type_hints

if TYPE_CHECKING:
    import openai.types as openai_types
    import openai.types.chat as openai_chat
    types = openai_types
    chat = openai_chat
else:
    types = None
    chat = None

FinishReason: TypeAlias = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
assert get_type_hints(chat.chat_completion_chunk.Choice)["finish_reason"] == FinishReason, (
    "Upstream changed Choice.finish_reason; update FinishReason alias."
)

__all__ = ["types", "chat", "FinishReason"]