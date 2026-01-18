from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from .parsers import ParserManager, ParsersResult, ReasoningParserState

ParsedChunk = Union[str, Dict[str, Any]]


@dataclass
class ChunkParserConfig:
    """Configuration for ChunkParser.

    Parameters
    ----------
    reasoning_parser_name:
        Name of a reasoning parser (e.g. 'hermes', 'glm4_moe', 'solar_open').
    tool_parser_name:
        Name of a tool-call parser (e.g. 'hermes', 'functiongemma', 'solar_open').
    enable_thinking:
        If False, and the selected reasoning parser respects enable_thinking,
        reasoning parsing is disabled.
    disable_all_parsing:
        If True, parser becomes a no-op pass-through (useful when enforcing JSON schema elsewhere).
    """

    reasoning_parser_name: Optional[str] = None
    tool_parser_name: Optional[str] = None
    enable_thinking: bool = True
    disable_all_parsing: bool = False


class ChunkParser:
    """Stateful streaming chunk parser.

    Feed it raw model text deltas (token strings). It emits a sequence of "parsed chunks":

    * plain strings => assistant content deltas
    * dicts with {"reasoning_content": ...} => reasoning deltas
    * dicts with {"name": ..., "arguments": ...} => tool call deltas

    This mirrors the shape expected by mlx-openai-server's stream wrapper, but the
    class is framework-agnostic.
    """

    def __init__(self, cfg: ChunkParserConfig):
        self.cfg = cfg
        self.parsers: ParsersResult = ParserManager.create_parsers(
            reasoning_parser_name=cfg.reasoning_parser_name,
            tool_parser_name=cfg.tool_parser_name,
        )

        if cfg.disable_all_parsing:
            self.parsers.reasoning_parser = None
            self.parsers.tool_parser = None
            self.parsers.unified_parser = None

        # Respect enable_thinking when the reasoning parser says it should.
        if (not cfg.enable_thinking) and self.parsers.reasoning_parser:
            if self.parsers.reasoning_parser.respects_enable_thinking():
                self.parsers.reasoning_parser = None

        self._is_first_chunk = True
        self._tool_call_index = 0

    def feed(self, text: str) -> List[ParsedChunk]:
        """Process one streamed text delta and return zero or more parsed chunks."""
        if not text:
            return []

        out: List[ParsedChunk] = []

        # Unified parser (Harmony) handles reasoning+tools+content together.
        if self.parsers.is_unified:
            parsed, _done = self.parsers.unified_parser.parse_streaming(text)
            if not parsed:
                return []

            if parsed.get("reasoning_content"):
                out.append({"reasoning_content": parsed["reasoning_content"]})

            tool_calls = parsed.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    out.append({
                        "name": tc.get("name"),
                        "arguments": tc.get("arguments")
                    })

            if parsed.get("content"):
                out.append(parsed["content"])
            return out

        reasoning_parser = self.parsers.reasoning_parser
        tool_parser = self.parsers.tool_parser

        # Some reasoning parsers need the opening token prefixed for the first chunk.
        if self._is_first_chunk and reasoning_parser:
            if hasattr(reasoning_parser, "needs_redacted_reasoning_prefix") and reasoning_parser.needs_redacted_reasoning_prefix():
                text = reasoning_parser.get_reasoning_open() + text
            self._is_first_chunk = False

        # 1) Reasoning parsing happens before tool parsing.
        after_reasoning_close: Optional[str] = None
        if reasoning_parser:
            parsed_content, is_complete = reasoning_parser.extract_reasoning_streaming(text)
            if isinstance(parsed_content, dict):
                if parsed_content.get("reasoning_content"):
                    out.append({
                        "reasoning_content":
                        parsed_content["reasoning_content"]
                    })
                
                if parsed_content.get("content"):
                    # If the reasoning parser returns content, we should use it as the text for the next stage
                    # but we generally do not want to double emit it if we are chaining parsers.
                    # However, ChunkParser logic seems to assume reasoning parser consumes input if it returns reasoning.
                    # If it returns content, it's passthrough.
                    text = parsed_content["content"]

                after_reasoning_close = parsed_content.get("after_reasoning_close_content")

                if is_complete:
                    self.parsers.reasoning_parser = None
                    reasoning_parser = None

            elif isinstance(parsed_content, str):
                # If it returned a string, treat it as content to be processed by tool parser
                text = parsed_content
                # out.append(parsed_content) # Removed direct append to allow tool parsing

            # If still inside reasoning and nothing after close, stop here.
            # We check if we are actually parsing reasoning (state != NORMAL).
            if reasoning_parser and reasoning_parser.state != ReasoningParserState.NORMAL and not after_reasoning_close:
                return out

            # Continue parsing the remainder (content after </think>, etc.)
            if after_reasoning_close:
                text = after_reasoning_close

        # 2) Tool-call parsing (may buffer until close token).
        if tool_parser:
            parsed_content, should_emit = tool_parser.extract_tool_calls_streaming(text)
            if not should_emit:
                return out

            if isinstance(parsed_content, dict):
                content = parsed_content.get("content")
                if isinstance(content, str) and content:
                    out.append(content)

                tool_calls = parsed_content.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        out.append({
                            "index": self._tool_call_index,
                            "id": f"call_{uuid.uuid4()}",
                            "type": "function",
                            "function": {
                                "name": tc.get("name"),
                                "arguments": tc.get("arguments")
                            }
                        })
                        self._tool_call_index += 1

                return out

            if isinstance(parsed_content, str):
                out.append(parsed_content)
                return out

            return out

        # 3) Plain content
        out.append(text)
        return out

    def flush(self) -> List[ParsedChunk]:
        """Flush any buffered state at end-of-stream.

        If a tool or reasoning parser was buffering an incomplete block, we return it
        as plain content so callers don't lose output.
        """
        out: List[ParsedChunk] = []

        if self.parsers.is_unified:
            up = self.parsers.unified_parser
            if getattr(up, "arguments_buffer", None) and getattr(up, "function_name_buffer", ""):
                out.append({
                    "name": up.function_name_buffer,
                    "arguments": "".join(up.arguments_buffer),
                })
                up.arguments_buffer = []
                up.function_name_buffer = ""
            return out

        rp = self.parsers.reasoning_parser
        if rp and getattr(rp, "buffer", ""):
            out.append(rp.buffer)
            rp.buffer = ""

        tp = self.parsers.tool_parser
        if tp and getattr(tp, "buffer", ""):
            out.append(tp.buffer)
            tp.buffer = ""

        return out


def parse_stream(
    text_deltas: Iterable[str],
    parser: ChunkParser,
) -> Iterator[ParsedChunk]:
    """Convenience wrapper: parse an iterable of raw text deltas."""
    for delta in text_deltas:
        for item in parser.feed(delta):
            yield item

    for item in parser.flush():
        yield item
