import asyncio

import pytest


@pytest.mark.asyncio
async def test_master_api_multiple_response_sequential() -> None:
    # TODO
    return
    messages = [ChatMessage(role="user", content="Hello, who are you?")]
    token_count = 0
    text: str = ""
    async for choice in stream_chatgpt_response(messages):
        print(choice, flush=True)
        if choice.delta and choice.delta.content:
            text += choice.delta.content
            token_count += 1
        if choice.finish_reason:
            break

    assert token_count >= 3, f"Expected at least 3 tokens, got {token_count}"
    assert len(text) > 0, "Expected non-empty response text"

    await asyncio.sleep(0.1)

    messages = [ChatMessage(role="user", content="What time is it in France?")]
    token_count = 0
    text = ""  # re-initialize, do not redeclare type
    async for choice in stream_chatgpt_response(messages):
        print(choice, flush=True)
        if choice.delta and choice.delta.content:
            text += choice.delta.content
            token_count += 1
        if choice.finish_reason:
            break

    assert token_count >= 3, f"Expected at least 3 tokens, got {token_count}"
    assert len(text) > 0, "Expected non-empty response text"
