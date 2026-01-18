from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

_HARMONY_IMPORT_ERROR: Optional[Exception] = None
try:
    from openai_harmony import (
        load_harmony_encoding,
        HarmonyEncodingName,
        StreamableParser,
        Role,
    )
except Exception as e:  # pragma: no cover
    _HARMONY_IMPORT_ERROR = e
    load_harmony_encoding = None  # type: ignore[assignment]
    HarmonyEncodingName = None  # type: ignore[assignment]
    StreamableParser = None  # type: ignore[assignment]
    Role = None  # type: ignore[assignment]


class ChannelType(Enum):
    ANALYSIS = "analysis"
    COMMENTARY = "commentary"
    FINAL = "final"


class ToolParserState(Enum):
    NORMAL = "normal"
    FOUND_ARGUMENTS = "found_arguments"
    END_STREAM = "end_stream"


class HarmonyParser:
    """Parser for OpenAI Harmony-style encodings.

    This is a *unified* parser: it can emit reasoning_content, tool_calls, and content.

    Notes
    -----
    * Requires the optional dependency `openai-harmony`.
    * Mirrors the behavior used by mlx-openai-server's Harmony parser.
    """

    def __init__(self):
        if _HARMONY_IMPORT_ERROR is not None:
            raise ImportError(
                "HarmonyParser requires the optional dependency 'openai-harmony'. "
                "Install it (pip install openai-harmony) or avoid selecting the 'harmony' parser."
            ) from _HARMONY_IMPORT_ERROR

        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.parser = StreamableParser(self.encoding, role=Role.ASSISTANT)

        self.end_tool_chunk = "<|call|>"
        self.state = ToolParserState.NORMAL
        self.arguments_buffer: List[str] = []
        self.function_name_buffer = ""

    def parse(self, text: str) -> Dict[str, Any] | None:
        if self.end_tool_chunk in text:
            text = text.split(self.end_tool_chunk)[0]

        result: Dict[str, Any] = {
            "content": None,
            "tool_calls": [],
            "reasoning_content": None,
        }

        tokens = self.encoding.encode(text, allowed_special="all")
        parsed_messages = self.encoding.parse_messages_from_completion_tokens(tokens, role=Role.ASSISTANT)
        for message in parsed_messages:
            if message.channel == ChannelType.ANALYSIS.value:
                result["reasoning_content"] = message.content[0].text
            elif message.channel == ChannelType.COMMENTARY.value:
                result["tool_calls"].append({
                    "name":
                    message.recipient.replace("functions.", ""),
                    "arguments":
                    message.content[0].text,
                })
            elif message.channel == ChannelType.FINAL.value:
                result["content"] = message.content[0].text
        return result

    def _build_result(
        self,
        reasoning_contents: List[str],
        tool_calls: Optional[List[Dict[str, str]]],
        contents: List[str],
    ) -> Dict[str, Any]:
        return {
            "reasoning_content": "".join(reasoning_contents) or None,
            "tool_calls": tool_calls,
            "content": "".join(contents) or None,
        }

    def parse_streaming(self, chunk: str) -> Tuple[Optional[Dict[str, Any]], bool]:
        if self.state == ToolParserState.END_STREAM:
            return None, True

        reasoning_contents: List[str] = []
        contents: List[str] = []
        end_stream_state = False

        if self.end_tool_chunk in chunk:
            chunk = chunk[:chunk.find(self.end_tool_chunk)]
            end_stream_state = True

        chunk_tokens = self.encoding.encode(chunk, allowed_special="all")
        for chunk_token in chunk_tokens:
            stream_text = self.parser.process(chunk_token)
            content = stream_text.last_content_delta

            if not content:
                continue

            if self.state == ToolParserState.FOUND_ARGUMENTS:
                self.arguments_buffer.append(content)
                continue

            current_channel = stream_text.current_channel
            if current_channel == ChannelType.ANALYSIS.value:
                reasoning_contents.append(content)
            elif current_channel == ChannelType.COMMENTARY.value:
                self.state = ToolParserState.FOUND_ARGUMENTS
                self.arguments_buffer.append(content)
                self.function_name_buffer = stream_text.current_recipient.replace("functions.", "")
            elif current_channel == ChannelType.FINAL.value:
                contents.append(content)

        if end_stream_state:
            tool_calls = [{
                "name": self.function_name_buffer,
                "arguments": "".join(self.arguments_buffer),
            }]
            self.arguments_buffer = []
            self.function_name_buffer = ""
            self.state = ToolParserState.END_STREAM
            return self._build_result(reasoning_contents, tool_calls, contents), True

        return self._build_result(reasoning_contents, None, contents), False
