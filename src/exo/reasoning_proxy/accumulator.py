"""Accumulators for capturing the emitted assistant shape from a streaming response.

Both accumulators are fed raw SSE chunks (bytes) as they pass through. At stream
end, they expose a canonical assistant-message shape suitable for hashing, plus
the reasoning text that should be cached against that hash.
"""

import json
import logging
from typing import cast

from exo.reasoning_proxy._helpers import (
    as_dict,
    as_str,
    dict_get_dict,
    dict_get_list,
    dict_get_str,
)

logger = logging.getLogger(__name__)


class OpenAIAccumulator:
    """Captures content, tool_calls, and reasoning_content from OpenAI SSE chunks.

    OpenAI can emit multiple choices per chunk; we only track choice index 0
    (the common case for chat completions; n>1 is uncommon and re-hash misses
    there degrade gracefully to no-op cache insert).
    """

    def __init__(self) -> None:
        self._content_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._tool_calls_by_index: dict[int, dict[str, object]] = {}
        self._buffer = ""

    def feed_bytes(self, chunk: bytes) -> None:
        self._buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]" or not payload:
                continue
            try:
                parsed = cast(object, json.loads(payload))
            except json.JSONDecodeError:
                continue
            event = as_dict(parsed)
            if event is None:
                continue
            self._consume_event(event)

    def _consume_event(self, event: dict[str, object]) -> None:
        choices = dict_get_list(event, "choices")
        if not choices:
            return
        for choice_raw in choices:
            choice = as_dict(choice_raw)
            if choice is None:
                continue
            index_val = choice.get("index", 0)
            if (
                not (isinstance(index_val, int) and not isinstance(index_val, bool))
                or index_val != 0
            ):
                continue
            delta = dict_get_dict(choice, "delta")
            if delta is None:
                continue
            content = dict_get_str(delta, "content")
            if content is not None:
                self._content_parts.append(content)
            reasoning = dict_get_str(delta, "reasoning_content")
            if reasoning is not None:
                self._reasoning_parts.append(reasoning)
            tool_calls = dict_get_list(delta, "tool_calls")
            if tool_calls is not None:
                self._merge_tool_calls(tool_calls)

    def _merge_tool_calls(self, deltas: list[object]) -> None:
        for raw in deltas:
            d = as_dict(raw)
            if d is None:
                continue
            index_val = d.get("index", 0)
            if not (isinstance(index_val, int) and not isinstance(index_val, bool)):
                continue
            entry = self._tool_calls_by_index.setdefault(
                index_val,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            tc_id = dict_get_str(d, "id")
            if tc_id is not None:
                entry["id"] = tc_id
            tc_type = dict_get_str(d, "type")
            if tc_type is not None:
                entry["type"] = tc_type
            fn = dict_get_dict(d, "function")
            if fn is not None:
                entry_fn = entry.get("function")
                if not isinstance(entry_fn, dict):
                    entry_fn = {"name": "", "arguments": ""}
                    entry["function"] = entry_fn
                entry_fn_typed = cast(dict[str, object], entry_fn)
                name = dict_get_str(fn, "name")
                if name is not None:
                    prev_name = as_str(entry_fn_typed.get("name")) or ""
                    entry_fn_typed["name"] = prev_name + name
                args = dict_get_str(fn, "arguments")
                if args is not None:
                    prev_args = as_str(entry_fn_typed.get("arguments")) or ""
                    entry_fn_typed["arguments"] = prev_args + args

    @property
    def content(self) -> str | None:
        joined = "".join(self._content_parts)
        return joined if joined else None

    @property
    def tool_calls(self) -> list[dict[str, object]] | None:
        if not self._tool_calls_by_index:
            return None
        ordered = [
            self._tool_calls_by_index[i] for i in sorted(self._tool_calls_by_index)
        ]
        return ordered

    @property
    def reasoning(self) -> str:
        return "".join(self._reasoning_parts)


class ClaudeAccumulator:
    """Captures Claude streaming content blocks.

    Tracks per-index content blocks. At end, exposes the final `content_blocks`
    list (excluding thinking blocks — those go into `reasoning` as joined text)
    in a shape suitable for hashing.
    """

    def __init__(self) -> None:
        self._blocks_by_index: dict[int, dict[str, object]] = {}
        self._buffer = ""
        self._current_event: str | None = None

    def feed_bytes(self, chunk: bytes) -> None:
        self._buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                self._current_event = None
                continue
            if line.startswith("event:"):
                self._current_event = line[len("event:") :].strip()
                continue
            if line.startswith("data:"):
                payload = line[len("data:") :].strip()
                try:
                    parsed = cast(object, json.loads(payload))
                except json.JSONDecodeError:
                    continue
                event = as_dict(parsed)
                if event is None:
                    continue
                self._consume_event(event)

    def _consume_event(self, event: dict[str, object]) -> None:
        event_type = dict_get_str(event, "type") or self._current_event
        if event_type == "content_block_start":
            index_val = event.get("index", 0)
            if not (isinstance(index_val, int) and not isinstance(index_val, bool)):
                return
            block = dict_get_dict(event, "content_block")
            if block is None:
                return
            btype = dict_get_str(block, "type")
            if btype == "text":
                self._blocks_by_index[index_val] = {"type": "text", "text": ""}
            elif btype == "thinking":
                self._blocks_by_index[index_val] = {
                    "type": "thinking",
                    "thinking": "",
                }
            elif btype == "tool_use":
                self._blocks_by_index[index_val] = {
                    "type": "tool_use",
                    "id": dict_get_str(block, "id") or "",
                    "name": dict_get_str(block, "name") or "",
                    "input_json": "",
                }
        elif event_type == "content_block_delta":
            index_val = event.get("index", 0)
            if not (isinstance(index_val, int) and not isinstance(index_val, bool)):
                return
            delta = dict_get_dict(event, "delta")
            if delta is None:
                return
            block = self._blocks_by_index.get(index_val)
            if block is None:
                return
            dtype = dict_get_str(delta, "type")
            if dtype == "text_delta":
                text = dict_get_str(delta, "text")
                if text is not None:
                    prev = as_str(block.get("text")) or ""
                    block["text"] = prev + text
            elif dtype == "thinking_delta":
                thinking = dict_get_str(delta, "thinking")
                if thinking is not None:
                    prev = as_str(block.get("thinking")) or ""
                    block["thinking"] = prev + thinking
            elif dtype == "input_json_delta":
                partial = dict_get_str(delta, "partial_json")
                if partial is not None:
                    prev = as_str(block.get("input_json")) or ""
                    block["input_json"] = prev + partial

    @property
    def content_blocks(self) -> list[dict[str, object]]:
        """Public blocks (excludes thinking), with tool_use input parsed from JSON."""
        public: list[dict[str, object]] = []
        for index in sorted(self._blocks_by_index):
            block = self._blocks_by_index[index]
            if block.get("type") == "thinking":
                continue
            if block.get("type") == "tool_use":
                input_json = as_str(block.get("input_json")) or "{}"
                parsed_input_raw: object
                try:
                    parsed_input_raw = cast(object, json.loads(input_json))
                except json.JSONDecodeError:
                    parsed_input_raw = {}
                parsed_input = as_dict(parsed_input_raw) or {}
                public.append(
                    {
                        "type": "tool_use",
                        "id": as_str(block.get("id")) or "",
                        "name": as_str(block.get("name")) or "",
                        "input": parsed_input,
                    }
                )
            else:
                public.append({k: v for k, v in block.items() if k != "input_json"})
        return public

    @property
    def reasoning(self) -> str:
        parts: list[str] = []
        for index in sorted(self._blocks_by_index):
            block = self._blocks_by_index[index]
            if block.get("type") == "thinking":
                parts.append(as_str(block.get("thinking")) or "")
        return "".join(parts)
