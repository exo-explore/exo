# pyright: reportAny=false

from collections.abc import AsyncGenerator

from exo.master.api import API
from exo.shared.types.chunks import TokenChunk
from exo.worker.tests.constants import COMMAND_1_ID, MODEL_A_ID


async def test_collect_chat_completion_populates_thinking_and_tool_calls_and_null_content() -> None:
    api = object.__new__(API)

    async def _fake_stream(_: API, __: str) -> AsyncGenerator[TokenChunk, None]:
        # All text is empty => final `content` should be None.
        yield TokenChunk(
            idx=0,
            model=MODEL_A_ID,
            text="",
            token_id=0,
            finish_reason=None,
            reasoning_content="a",
            tool_calls=None,
        )
        yield TokenChunk(
            idx=1,
            model=MODEL_A_ID,
            text="",
            token_id=1,
            finish_reason="stop",
            reasoning_content="b",
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_test",
                    "type": "function",
                    "function": {"name": "do", "arguments": "{}"},
                }
            ],
        )

    api._chat_chunk_stream = _fake_stream.__get__(api, API)  # type: ignore[method-assign]

    resp = await api._collect_chat_completion(COMMAND_1_ID)

    msg = resp.choices[0].message  # type: ignore[union-attr]
    assert msg.content is None
    assert msg.thinking == "ab"

    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0]["type"] == "function"


async def test_chunk_to_response_includes_thinking_and_tool_calls() -> None:
    from exo.master.api import chunk_to_response

    chunk = TokenChunk(
        idx=0,
        model=MODEL_A_ID,
        text="hi",
        token_id=0,
        finish_reason=None,
        reasoning_content="thought",
        tool_calls=[{"type": "function", "function": {"name": "x", "arguments": "{}"}}],
    )

    resp = chunk_to_response(chunk, COMMAND_1_ID)
    delta = resp.choices[0].delta  # type: ignore[union-attr]
    assert delta.thinking == "thought"
    assert delta.tool_calls is not None
    assert delta.tool_calls[0]["type"] == "function"
