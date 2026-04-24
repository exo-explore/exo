"""Dialect strategies for deciding which assistant history indices should receive
cached reasoning on inbound requests.

Each dialect inspects the message list and returns the set of indices where
reasoning_content (OpenAI) or a thinking block (Claude) should be reattached if
the cache has it. The dialect does not mutate messages — the caller does.

Dialect selection is driven by the `reasoning_dialect` field on each model card,
surfaced through /v1/models.
"""

from typing import Protocol

from exo.reasoning_proxy._helpers import as_dict, as_list, dict_get_list, dict_get_str
from exo.shared.types.text_generation import ReasoningDialect


class Dialect(Protocol):
    def select_attach_indices(
        self, messages: list[dict[str, object]], has_tools: bool
    ) -> set[int]: ...


def _is_assistant(msg: dict[str, object]) -> bool:
    return msg.get("role") == "assistant"


def _is_user(msg: dict[str, object]) -> bool:
    return msg.get("role") == "user"


class NoneDialect:
    def select_attach_indices(
        self, messages: list[dict[str, object]], has_tools: bool
    ) -> set[int]:
        return set()


class PostLastUserDialect:
    """MiniMax / GLM / Qwen-thinking / V4-with-tools.

    Preserve reasoning on every assistant message appearing after the last
    non-tool-response user message. Tool-response user messages (role=tool, or
    role=user with tool_call_id set, or Claude's tool_result block) don't count
    as "real" user turns — they're part of the assistant's tool-calling chain.
    """

    def select_attach_indices(
        self, messages: list[dict[str, object]], has_tools: bool
    ) -> set[int]:
        last_user_index = -1
        for i, msg in enumerate(messages):
            if _is_user(msg) and not _is_tool_response(msg):
                last_user_index = i
        return {
            i
            for i, msg in enumerate(messages)
            if i > last_user_index and _is_assistant(msg)
        }


class SuffixDialect:
    """Kimi K2 Thinking / K2.6.

    Preserve reasoning only on the tail run of tool-call-carrying assistant
    messages (the current, unresolved tool-call chain). Walk backward: include
    every assistant with tool_calls until we hit an assistant without tool_calls
    or a non-assistant message.
    """

    def select_attach_indices(
        self, messages: list[dict[str, object]], has_tools: bool
    ) -> set[int]:
        indices: set[int] = set()
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if not _is_assistant(msg):
                if _is_tool_response(msg):
                    continue
                break
            if not _has_tool_calls(msg):
                break
            indices.add(i)
        return indices


class ChannelDialect:
    """GPT-OSS Harmony format.

    Preserve analysis-channel content on assistant turns that follow the most
    recent assistant message tagged with a "final" channel marker. If no prior
    final exists, the whole conversation is one unresolved chain.
    """

    def select_attach_indices(
        self, messages: list[dict[str, object]], has_tools: bool
    ) -> set[int]:
        last_final_index = -1
        for i, msg in enumerate(messages):
            if _is_assistant(msg) and _has_final_channel(msg):
                last_final_index = i
        return {
            i
            for i, msg in enumerate(messages)
            if i > last_final_index and _is_assistant(msg)
        }


class ToolConditionalDialect:
    """DeepSeek V4 Flash.

    If the request has tools, behave as PostLastUserDialect; otherwise passthrough.
    """

    def __init__(self) -> None:
        self._inner = PostLastUserDialect()

    def select_attach_indices(
        self, messages: list[dict[str, object]], has_tools: bool
    ) -> set[int]:
        if not has_tools:
            return set()
        return self._inner.select_attach_indices(messages, has_tools)


def _is_tool_response(msg: dict[str, object]) -> bool:
    if msg.get("role") == "tool":
        return True
    if msg.get("role") == "user" and msg.get("tool_call_id"):
        return True
    content = as_list(msg.get("content"))
    if content is not None:
        for raw in content:
            block = as_dict(raw)
            if block is not None and block.get("type") == "tool_result":
                return True
    return False


def _has_tool_calls(msg: dict[str, object]) -> bool:
    tc = dict_get_list(msg, "tool_calls")
    if tc:
        return True
    content = as_list(msg.get("content"))
    if content is not None:
        for raw in content:
            block = as_dict(raw)
            if block is not None and block.get("type") == "tool_use":
                return True
    return False


def _has_final_channel(msg: dict[str, object]) -> bool:
    if msg.get("channel") == "final":
        return True
    content = dict_get_str(msg, "content")
    return bool(content and content.strip())


_DIALECTS: dict[ReasoningDialect, Dialect] = {
    "none": NoneDialect(),
    "post_last_user": PostLastUserDialect(),
    "suffix": SuffixDialect(),
    "channel": ChannelDialect(),
    "tool_conditional": ToolConditionalDialect(),
}


def get_dialect(name: ReasoningDialect) -> Dialect:
    return _DIALECTS[name]
