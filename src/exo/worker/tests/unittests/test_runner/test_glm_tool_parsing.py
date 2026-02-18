"""Tests for GLM tool call argument parsing regex."""

import regex as re

# Replicate the regex patterns from runner.py to test them in isolation
_func_name_regex = re.compile(r"^(.*?)<arg_key>", re.DOTALL)
_func_arg_regex = re.compile(
    r"<arg_key>(.*?)</arg_key>(?:\n|\s)*<arg_value>(.*?)(?:</arg_value>|(?=<arg_key>)|$)",
    re.DOTALL,
)


def _parse_args(text: str) -> list[tuple[str, str]]:
    """Extract (key, value) pairs from GLM tool call text."""
    pairs = _func_arg_regex.findall(text)
    return [(k.strip(), v.strip()) for k, v in pairs]  # pyright: ignore[reportAny]


def _parse_func_name(text: str) -> str:
    """Extract function name from GLM tool call text."""
    match = _func_name_regex.search(text)
    if match is None:
        raise ValueError(f"Could not parse function name: {text!r}")
    return match.group(1).strip()


class TestGlmToolParsingWithClosingTags:
    """Tests for normal format with closing tags present."""

    def test_single_argument(self):
        text = (
            "get_weather<arg_key>location</arg_key><arg_value>San Francisco</arg_value>"
        )
        assert _parse_func_name(text) == "get_weather"
        pairs = _parse_args(text)
        assert pairs == [("location", "San Francisco")]

    def test_multiple_arguments(self):
        text = (
            "search<arg_key>query</arg_key><arg_value>python</arg_value>"
            "<arg_key>limit</arg_key><arg_value>10</arg_value>"
        )
        assert _parse_func_name(text) == "search"
        pairs = _parse_args(text)
        assert pairs == [("query", "python"), ("limit", "10")]

    def test_arguments_with_whitespace_between(self):
        text = (
            "fn<arg_key>a</arg_key>\n<arg_value>1</arg_value>\n"
            "<arg_key>b</arg_key> <arg_value>2</arg_value>"
        )
        pairs = _parse_args(text)
        assert pairs == [("a", "1"), ("b", "2")]


class TestGlmToolParsingMissingClosingTags:
    """Tests for format where </arg_value> closing tags are missing."""

    def test_single_argument_no_closing(self):
        text = "get_weather<arg_key>location</arg_key><arg_value>San Francisco"
        assert _parse_func_name(text) == "get_weather"
        pairs = _parse_args(text)
        assert pairs == [("location", "San Francisco")]

    def test_multiple_arguments_no_closing(self):
        text = (
            "search<arg_key>query</arg_key><arg_value>python"
            "<arg_key>limit</arg_key><arg_value>10"
        )
        assert _parse_func_name(text) == "search"
        pairs = _parse_args(text)
        assert pairs == [("query", "python"), ("limit", "10")]

    def test_mixed_closing_tags(self):
        """First arg has closing tag, second does not."""
        text = (
            "fn<arg_key>a</arg_key><arg_value>1</arg_value>"
            "<arg_key>b</arg_key><arg_value>2"
        )
        pairs = _parse_args(text)
        assert pairs == [("a", "1"), ("b", "2")]

    def test_value_with_trailing_whitespace(self):
        text = "fn<arg_key>x</arg_key><arg_value>hello world  \n"
        pairs = _parse_args(text)
        assert pairs == [("x", "hello world")]

    def test_value_with_newlines_no_closing(self):
        text = "fn<arg_key>data</arg_key><arg_value>line1\nline2"
        pairs = _parse_args(text)
        assert pairs == [("data", "line1\nline2")]


class TestGlmToolParsingEdgeCases:
    """Edge case tests for GLM tool call parsing."""

    def test_empty_value_with_closing(self):
        text = "fn<arg_key>empty</arg_key><arg_value></arg_value>"
        pairs = _parse_args(text)
        assert pairs == [("empty", "")]

    def test_value_with_json_content(self):
        text = 'fn<arg_key>data</arg_key><arg_value>{"key": "value"}</arg_value>'
        pairs = _parse_args(text)
        assert pairs == [("data", '{"key": "value"}')]

    def test_value_with_json_no_closing(self):
        text = 'fn<arg_key>data</arg_key><arg_value>{"key": "value"}'
        pairs = _parse_args(text)
        assert pairs == [("data", '{"key": "value"}')]
