import asyncio
import functools
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Coroutine,
    ParamSpec,
    TypeVar,
    final,
)

import openai
import pytest
from openai._streaming import AsyncStream
from openai.types.chat import (
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice

from exo.master.main import async_main as master_main

_P = ParamSpec("_P")
_R = TypeVar("_R")

OPENAI_API_KEY: str = "<YOUR_OPENAI_API_KEY>"
OPENAI_API_URL: str = "http://0.0.0.0:8000/v1"

def with_master_main(
    func: Callable[_P, Awaitable[_R]]
) -> Callable[_P, Coroutine[Any, Any, _R]]:
    @pytest.mark.asyncio
    @functools.wraps(func)
    async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        master_task = asyncio.create_task(master_main())
        try:
            return await func(*args, **kwargs)
        finally:
            master_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await master_task
    return wrapper

@final
class ChatMessage:
    """Strictly-typed chat message for OpenAI API."""
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def to_openai(self) -> ChatCompletionMessageParam:
        if self.role == "user":
            return {"role": "user", "content": self.content}  # type: ChatCompletionUserMessageParam
        elif self.role == "assistant":
            return {"role": "assistant", "content": self.content}  # type: ChatCompletionAssistantMessageParam
        elif self.role == "system":
            return {"role": "system", "content": self.content}  # type: ChatCompletionSystemMessageParam
        else:
            raise ValueError(f"Unsupported role: {self.role}")

async def stream_chatgpt_response(
    messages: list[ChatMessage],
    model: str = "gpt-3.5-turbo",
) -> AsyncGenerator[Choice, None]:
    client = openai.AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_URL,
    )
    openai_messages: list[ChatCompletionMessageParam] = [m.to_openai() for m in messages]
    stream: AsyncStream[ChatCompletionChunk] = await client.chat.completions.create(
        model=model,
        messages=openai_messages,
        stream=True,
    )
    async for chunk in stream:
        for choice in chunk.choices:
            yield choice
