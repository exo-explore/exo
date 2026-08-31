"""Tests for the Kimi K2 textual tool-call parser."""

import pytest

from exo.worker.engines.mlx.utils_mlx import _parse_kimi_tool_calls  # pyright: ignore[reportPrivateUsage]

_ARGS = '{"city": "Melbourne, Australia"}'


def _wrap(tool_id: str) -> str:
    return (
        "<|tool_calls_section_begin|>"
        f"<|tool_call_begin|>{tool_id}<|tool_call_argument_begin|>{_ARGS}<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )


class TestParseKimiToolCalls:
    def test_canonical_id(self):
        calls = _parse_kimi_tool_calls(_wrap("functions.get_weather:1"))
        assert calls == [
            {
                "id": "functions.get_weather:1",
                "name": "get_weather",
                "arguments": {"city": "Melbourne, Australia"},
            }
        ]

    def test_sanitised_id_is_accepted_and_canonicalised(self):
        # Clients that restrict ids to [A-Za-z0-9_-] echo this form back in the
        # history and the model then emits it; previously a hard parse failure.
        calls = _parse_kimi_tool_calls(_wrap("functions_get_weather_1"))
        assert calls[0]["id"] == "functions.get_weather:1"
        assert calls[0]["name"] == "get_weather"

    def test_sanitised_id_with_underscores_in_name(self):
        calls = _parse_kimi_tool_calls(_wrap("functions_read_file_3"))
        assert calls[0]["id"] == "functions.read_file:3"
        assert calls[0]["name"] == "read_file"

    def test_missing_prefix_is_canonicalised(self):
        calls = _parse_kimi_tool_calls(_wrap("get_weather:0"))
        assert calls[0]["id"] == "functions.get_weather:0"

    def test_multiple_calls(self):
        text = (
            "<|tool_calls_section_begin|>"
            f"<|tool_call_begin|>functions.a:0<|tool_call_argument_begin|>{{}}<|tool_call_end|>"
            f"<|tool_call_begin|>functions_b_1<|tool_call_argument_begin|>{{}}<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        calls = _parse_kimi_tool_calls(text)
        assert [c["id"] for c in calls] == ["functions.a:0", "functions.b:1"]

    def test_no_index_still_fails(self):
        with pytest.raises(ValueError):
            _parse_kimi_tool_calls(_wrap("functions.get_weather"))
