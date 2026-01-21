from unittest.mock import Mock

from exo.shared.types.api import ChatCompletionMessage
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.worker.engines.mlx.utils_mlx import (
    normalize_tool_call_arguments,
    apply_chat_template,
)


class TestToolCallNormalization:
    """Test tool call argument normalization from JSON strings to dictionaries."""

    def test_normalize_json_string_arguments(self):
        """Test that JSON string arguments are converted to dictionaries."""
        # Create a message with tool calls in OpenAI format
        message = ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris", "unit": "celsius"}',  # JSON string
                    },
                }
            ],
        )

        # Apply the normalization
        normalized = normalize_tool_call_arguments(message.model_dump())

        # Verify the arguments are now a dict
        assert normalized["tool_calls"][0]["function"]["arguments"] == {
            "location": "Paris",
            "unit": "celsius",
        }

    def test_normalize_already_dict_arguments(self):
        """Test that dictionary arguments remain unchanged (idempotent)."""
        message = ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {
                            "location": "Paris",
                            "unit": "celsius",
                        },  # Already dict
                    },
                }
            ],
        )

        # Apply the normalization
        normalized = normalize_tool_call_arguments(message.model_dump())

        # Verify the arguments remain a dict
        assert normalized["tool_calls"][0]["function"]["arguments"] == {
            "location": "Paris",
            "unit": "celsius",
        }

    def test_normalize_invalid_json_string(self):
        """Test that invalid JSON strings are handled gracefully."""
        message = ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris", "unit": ',  # Invalid JSON
                    },
                }
            ],
        )

        # Apply the normalization - should not crash
        normalized = normalize_tool_call_arguments(message.model_dump())

        # Should fallback to empty dict
        assert normalized["tool_calls"][0]["function"]["arguments"] == {}

    def test_normalize_legacy_function_call_format(self):
        """Test normalization of legacy function_call format."""
        message = ChatCompletionMessage(
            role="assistant",
            function_call={
                "name": "get_weather",
                "arguments": '{"location": "Paris", "unit": "celsius"}',  # JSON string
            },
        )

        # Apply the normalization
        normalized = normalize_tool_call_arguments(message.model_dump())

        # Verify the arguments are now a dict
        assert normalized["function_call"]["arguments"] == {
            "location": "Paris",
            "unit": "celsius",
        }

    def test_normalize_non_dict_non_string_arguments(self):
        """Test handling of non-dict, non-string arguments."""
        message = ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": 42,  # Integer, not string or dict
                    },
                }
            ],
        )

        # Apply the normalization
        normalized = normalize_tool_call_arguments(message.model_dump())

        # Should convert to dict with value key
        assert normalized["tool_calls"][0]["function"]["arguments"] == {"value": 42}

    def test_normalize_no_tool_calls(self):
        """Test that messages without tool_calls are left unchanged."""
        message = ChatCompletionMessage(role="user", content="Hello, world!")

        # Apply the normalization
        normalized = normalize_tool_call_arguments(message.model_dump())

        # Should be identical
        assert normalized == message.model_dump()

    def test_apply_chat_template_with_normalized_tool_calls(self):
        """Test that apply_chat_template works with normalized tool calls."""
        # Create a message with tool calls in OpenAI format
        message = ChatCompletionMessage(
            role="assistant",
            content="I need to check the weather",  # Add content so message isn't filtered out
            tool_calls=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris", "unit": "celsius"}',
                    },
                }
            ],
        )

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "Test prompt"

        # Create task params
        task_params = ChatCompletionTaskParams(
            model="test-model", messages=[message], tools=[]
        )

        # Apply chat template - should not raise Jinja2 error
        result = apply_chat_template(mock_tokenizer, task_params)

        # Verify the tokenizer was called with normalized messages
        assert result == "Test prompt"
        mock_tokenizer.apply_chat_template.assert_called_once()

        # Check that the arguments in the call are normalized
        call_args = mock_tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]  # First positional argument

        assert messages[0]["tool_calls"][0]["function"]["arguments"] == {
            "location": "Paris",
            "unit": "celsius",
        }
