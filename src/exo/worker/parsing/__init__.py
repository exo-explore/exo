"""Tool-call + reasoning parsing for streaming LLM output.

This package is designed to be *dropped into* a project that streams raw text
deltas from an LLM and needs to convert model-specific markup into:

- assistant content
- reasoning_content (thinking)
- tool calls (name + JSON arguments)

The output shapes intentionally match what an OpenAI-compatible chat.completions
streaming wrapper expects (tool_calls deltas, finish_reason=tool_calls, etc.).

Primary entrypoints:
- toolcall_parser.stream.ChunkParser
- toolcall_parser.stream.parse_stream
- toolcall_parser.parsers.ParserManager (registry/factory)
"""

from .stream import ChunkParser, ChunkParserConfig, parse_stream

__all__ = ["ChunkParser", "ChunkParserConfig", "parse_stream"]
