"""Tests for Claude Messages API conversion functions and types."""

import pydantic
import pytest

from exo.api.adapters.claude import (
    claude_request_to_text_generation,
    finish_reason_to_claude_stop_reason,
)
from exo.api.types.claude_api import (
    ClaudeImageBlock,
    ClaudeImageSource,
    ClaudeMessage,
    ClaudeMessagesRequest,
    ClaudeTextBlock,
    ClaudeThinkingBlock,
    ClaudeToolResultBlock,
)
from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import Base64Image


class TestFinishReasonToClaudeStopReason:
    """Tests for finish_reason to Claude stop_reason mapping."""

    def test_stop_maps_to_end_turn(self):
        assert finish_reason_to_claude_stop_reason("stop") == "end_turn"

    def test_length_maps_to_max_tokens(self):
        assert finish_reason_to_claude_stop_reason("length") == "max_tokens"

    def test_tool_calls_maps_to_tool_use(self):
        assert finish_reason_to_claude_stop_reason("tool_calls") == "tool_use"

    def test_function_call_maps_to_tool_use(self):
        assert finish_reason_to_claude_stop_reason("function_call") == "tool_use"

    def test_content_filter_maps_to_end_turn(self):
        assert finish_reason_to_claude_stop_reason("content_filter") == "end_turn"

    def test_none_returns_none(self):
        assert finish_reason_to_claude_stop_reason(None) is None


class TestClaudeRequestToInternal:
    """Tests for converting Claude Messages API requests to TextGenerationTaskParams."""

    async def test_basic_request_conversion(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(role="user", content="Hello"),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.model == "claude-3-opus"
        assert params.max_output_tokens == 100
        assert isinstance(params.input, list)
        assert len(params.input) == 1
        assert params.input[0].role == "user"
        assert params.input[0].content == "Hello"
        assert params.instructions is None

    async def test_request_with_system_string(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            system="You are a helpful assistant.",
            messages=[
                ClaudeMessage(role="user", content="Hello"),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.instructions == "You are a helpful assistant."
        assert isinstance(params.input, list)
        assert len(params.input) == 1
        assert params.input[0].role == "user"
        assert params.input[0].content == "Hello"

    async def test_request_with_system_text_blocks(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            system=[
                ClaudeTextBlock(text="You are helpful. "),
                ClaudeTextBlock(text="Be concise."),
            ],
            messages=[
                ClaudeMessage(role="user", content="Hello"),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.instructions == "You are helpful. Be concise."
        assert isinstance(params.input, list)
        assert len(params.input) == 1

    async def test_request_with_content_blocks(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeTextBlock(text="First part. "),
                        ClaudeTextBlock(text="Second part."),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert isinstance(params.input, list)
        assert len(params.input) == 1
        assert params.input[0].content == "First part. Second part."

    async def test_request_with_multi_turn_conversation(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(role="user", content="Hello"),
                ClaudeMessage(role="assistant", content="Hi there!"),
                ClaudeMessage(role="user", content="How are you?"),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert isinstance(params.input, list)
        assert len(params.input) == 3
        assert params.input[0].role == "user"
        assert params.input[1].role == "assistant"
        assert params.input[2].role == "user"

    async def test_request_with_optional_parameters(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[ClaudeMessage(role="user", content="Hello")],
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            stop_sequences=["STOP", "END"],
            stream=True,
        )
        params = await claude_request_to_text_generation(request)

        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 40
        assert params.stop == ["STOP", "END"]
        assert params.stream is True


class TestClaudeToolResultConversion:
    """Tests for tool_result blocks mixed with other content."""

    async def test_text_after_a_tool_result_reaches_the_template(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeToolResultBlock(
                            tool_use_id="toolu_1", content="sunny, 72F"
                        ),
                        ClaudeTextBlock(text="what should I wear?"),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.chat_template_messages == [
            {"role": "tool", "tool_call_id": "toolu_1", "content": "sunny, 72F"},
            {"role": "user", "content": "what should I wear?"},
        ]
        assert len(params.input) == 1
        assert params.input[0].content == "what should I wear?"

    async def test_text_before_a_tool_result_emits_the_result_first(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeTextBlock(text="check this:"),
                        ClaudeToolResultBlock(tool_use_id="toolu_1", content="42"),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.chat_template_messages == [
            {"role": "tool", "tool_call_id": "toolu_1", "content": "42"},
            {"role": "user", "content": "check this:"},
        ]

    async def test_lone_tool_result_emits_only_the_tool_message(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeToolResultBlock(tool_use_id="toolu_1", content="done"),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.chat_template_messages == [
            {"role": "tool", "tool_call_id": "toolu_1", "content": "done"},
        ]

    async def test_image_inside_a_tool_result_gets_a_placeholder(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeToolResultBlock(
                            tool_use_id="toolu_1",
                            content=[
                                ClaudeTextBlock(text="screenshot"),
                                ClaudeImageBlock(
                                    source=ClaudeImageSource(
                                        type="base64",
                                        data="aGVsbG8=",
                                    )
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert len(params.images) == 1
        assert params.chat_template_messages == [
            {"role": "tool", "tool_call_id": "toolu_1", "content": "screenshot"},
            {"role": "user", "content": [{"type": "image"}]},
        ]

    async def test_tool_result_with_no_content_produces_empty_text(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[ClaudeToolResultBlock(tool_use_id="toolu_1")],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.chat_template_messages == [
            {"role": "tool", "tool_call_id": "toolu_1", "content": ""},
        ]

    async def test_interleaved_text_and_results_emit_results_first(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeToolResultBlock(
                            tool_use_id="toolu_1", content="first result"
                        ),
                        ClaudeTextBlock(text="first note"),
                        ClaudeToolResultBlock(
                            tool_use_id="toolu_2", content="second result"
                        ),
                        ClaudeTextBlock(text="second note"),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.chat_template_messages == [
            {
                "role": "tool",
                "tool_call_id": "toolu_1",
                "content": "first result",
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_2",
                "content": "second result",
            },
            {"role": "user", "content": "first note\nsecond note"},
        ]

    async def test_list_text_content_in_a_tool_result_is_concatenated(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeToolResultBlock(
                            tool_use_id="toolu_1",
                            content=[
                                ClaudeTextBlock(text="part a"),
                                ClaudeTextBlock(text="part b"),
                            ],
                        ),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.chat_template_messages == [
            {
                "role": "tool",
                "tool_call_id": "toolu_1",
                "content": "part apart b",
            },
        ]

    async def test_image_in_a_tool_result_rides_the_text_message(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeTextBlock(text="read this:"),
                        ClaudeToolResultBlock(
                            tool_use_id="toolu_1",
                            content=[
                                ClaudeImageBlock(
                                    source=ClaudeImageSource(
                                        type="base64",
                                        data="aGVsbG8=",
                                    )
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert len(params.images) == 1
        assert params.chat_template_messages == [
            {"role": "tool", "tool_call_id": "toolu_1", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "read this:"},
                    {"type": "image"},
                ],
            },
        ]

    async def test_image_in_a_tool_result_on_an_assistant_turn(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeTextBlock(text="here is the chart"),
                        ClaudeToolResultBlock(
                            tool_use_id="toolu_1",
                            content=[
                                ClaudeImageBlock(
                                    source=ClaudeImageSource(
                                        type="base64",
                                        data="aGVsbG8=",
                                    )
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert len(params.images) == 1
        assert params.chat_template_messages == [
            {"role": "tool", "tool_call_id": "toolu_1", "content": ""},
            {"role": "user", "content": [{"type": "image"}]},
            {"role": "assistant", "content": "here is the chart"},
        ]

    async def test_a_failed_image_fetch_emits_no_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        async def fail_fetch(_url: str):
            raise OSError("unreachable")

        monkeypatch.setattr("exo.api.adapters.claude.fetch_image_url", fail_fetch)
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeImageBlock(
                            source=ClaudeImageSource(
                                type="url", url="https://example.com/a.png"
                            )
                        ),
                        ClaudeTextBlock(text="describe it"),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.images == []
        assert params.chat_template_messages == [
            {"role": "user", "content": "describe it"},
        ]

    async def test_tool_result_is_error_is_preserved(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeToolResultBlock(
                            tool_use_id="toolu_1",
                            content="command not found",
                            is_error=True,
                        ),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.chat_template_messages == [
            {
                "role": "tool",
                "tool_call_id": "toolu_1",
                "content": "command not found",
                "is_error": True,
            },
        ]

    async def test_empty_text_beside_a_tool_result_emits_no_user_message(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeToolResultBlock(tool_use_id="toolu_1", content="done"),
                        ClaudeTextBlock(text=""),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.chat_template_messages == [
            {"role": "tool", "tool_call_id": "toolu_1", "content": "done"},
        ]

    async def test_empty_text_blocks_do_not_pad_the_joined_text(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeToolResultBlock(tool_use_id="toolu_1", content="done"),
                        ClaudeTextBlock(text=""),
                        ClaudeTextBlock(text="hi"),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.chat_template_messages == [
            {"role": "tool", "tool_call_id": "toolu_1", "content": "done"},
            {"role": "user", "content": "hi"},
        ]

    async def test_thinking_rides_the_placeholder_for_a_non_user_turn(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeThinkingBlock(thinking="hmm"),
                        ClaudeToolResultBlock(
                            tool_use_id="toolu_1",
                            content=[
                                ClaudeImageBlock(
                                    source=ClaudeImageSource(
                                        type="base64", data="aGVsbG8="
                                    )
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.chat_template_messages == [
            {"role": "tool", "tool_call_id": "toolu_1", "content": ""},
            {
                "role": "user",
                "content": [{"type": "image"}],
                "reasoning_content": "hmm",
            },
        ]

    async def test_thinking_beside_a_tool_result_is_kept(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeThinkingBlock(thinking="hmm"),
                        ClaudeToolResultBlock(tool_use_id="toolu_1", content="done"),
                        ClaudeTextBlock(text="thanks"),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.chat_template_messages == [
            {"role": "tool", "tool_call_id": "toolu_1", "content": "done"},
            {
                "role": "user",
                "content": "thanks",
                "reasoning_content": "hmm",
            },
        ]

    async def test_plain_multimodal_message_keeps_block_order(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeImageBlock(
                            source=ClaudeImageSource(type="base64", data="aGVsbG8=")
                        ),
                        ClaudeTextBlock(text="what is this"),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.chat_template_messages == [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "what is this"},
                ],
            },
        ]

    async def test_url_image_inside_a_tool_result_uses_a_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        async def fake_fetch(_url: str):
            return Base64Image("aGVsbG8=")

        monkeypatch.setattr("exo.api.adapters.claude.fetch_image_url", fake_fetch)
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeToolResultBlock(
                            tool_use_id="toolu_1",
                            content=[
                                ClaudeImageBlock(
                                    source=ClaudeImageSource(
                                        type="url",
                                        url="https://example.com/a.png",
                                    )
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.images == [Base64Image("aGVsbG8=")]
        assert params.chat_template_messages == [
            {"role": "tool", "tool_call_id": "toolu_1", "content": ""},
            {"role": "user", "content": [{"type": "image"}]},
        ]

    async def test_failed_url_image_inside_a_tool_result_leaves_no_marker(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        async def fail_fetch(_url: str):
            raise OSError("unreachable")

        monkeypatch.setattr("exo.api.adapters.claude.fetch_image_url", fail_fetch)
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeToolResultBlock(
                            tool_use_id="toolu_1",
                            content=[
                                ClaudeImageBlock(
                                    source=ClaudeImageSource(
                                        type="url",
                                        url="https://example.com/a.png",
                                    )
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.images == []
        assert params.chat_template_messages == [
            {"role": "tool", "tool_call_id": "toolu_1", "content": ""},
        ]

    async def test_multiple_images_in_one_tool_result(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeToolResultBlock(
                            tool_use_id="toolu_1",
                            content=[
                                ClaudeImageBlock(
                                    source=ClaudeImageSource(
                                        type="base64", data="aGVsbG8="
                                    )
                                ),
                                ClaudeImageBlock(
                                    source=ClaudeImageSource(
                                        type="base64", data="d29ybGQ="
                                    )
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.images == [Base64Image("aGVsbG8="), Base64Image("d29ybGQ=")]
        assert params.chat_template_messages == [
            {"role": "tool", "tool_call_id": "toolu_1", "content": ""},
            {"role": "user", "content": [{"type": "image"}, {"type": "image"}]},
        ]

    async def test_thinking_rides_a_placeholder_only_message(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeThinkingBlock(thinking="hmm"),
                        ClaudeToolResultBlock(
                            tool_use_id="toolu_1",
                            content=[
                                ClaudeImageBlock(
                                    source=ClaudeImageSource(
                                        type="base64", data="aGVsbG8="
                                    )
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.chat_template_messages == [
            {"role": "tool", "tool_call_id": "toolu_1", "content": ""},
            {
                "role": "user",
                "content": [{"type": "image"}],
                "reasoning_content": "hmm",
            },
        ]

    async def test_identical_image_blocks_only_mark_successful_fetches(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        calls = 0

        async def flaky_fetch(_url: str):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("unreachable")
            return Base64Image("aGVsbG8=")

        monkeypatch.setattr("exo.api.adapters.claude.fetch_image_url", flaky_fetch)
        source = ClaudeImageSource(type="url", url="https://example.com/a.png")
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeImageBlock(source=source),
                        ClaudeImageBlock(source=source),
                        ClaudeTextBlock(text="two shots"),
                    ],
                ),
            ],
        )
        params = await claude_request_to_text_generation(request)

        assert params.images == [Base64Image("aGVsbG8=")]
        assert params.chat_template_messages == [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "two shots"},
                ],
            },
        ]


class TestClaudeMessagesRequestValidation:
    """Tests for Claude Messages API request validation."""

    def test_request_requires_model(self):
        with pytest.raises(pydantic.ValidationError):
            ClaudeMessagesRequest.model_validate(
                {
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            )

    def test_request_requires_max_tokens(self):
        with pytest.raises(pydantic.ValidationError):
            ClaudeMessagesRequest.model_validate(
                {
                    "model": "claude-3-opus",
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            )

    def test_request_requires_messages(self):
        with pytest.raises(pydantic.ValidationError):
            ClaudeMessagesRequest.model_validate(
                {
                    "model": "claude-3-opus",
                    "max_tokens": 100,
                }
            )
