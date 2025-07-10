from typing import TYPE_CHECKING, Literal, TypeAlias, get_type_hints

FinishReason: TypeAlias = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call"
]

if TYPE_CHECKING:
    import openai.types as openai_types
    import openai.types.chat as openai_chat

    types = openai_types
    chat = openai_chat

    assert (
        get_type_hints(chat.chat_completion_chunk.Choice)["finish_reason"] == FinishReason
    ), "Upstream changed Choice.finish_reason; update FinishReason alias."
else:
    types = None
    chat = None

__all__ = ["types", "chat", "FinishReason"]
