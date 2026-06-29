"""tool_choice forces a tool call via an assistant prefill marker."""

import pytest

pytest.importorskip("mlx")

from types import SimpleNamespace

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.worker.engines.mlx.utils_mlx import _forced_tool_call_prefill

# Minimal fragment that infer_tool_parser recognizes as the Hermes/Qwen format.
_TEMPLATE = "{{ '<tool_call>' }}{{ tool_call.name }}{{ '</tool_call>' }}"
_TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]


def _params(*, tools=None, tool_choice=None):
    return TextGenerationTaskParams(
        model=ModelId("test/model"),
        input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        tools=tools,
        tool_choice=tool_choice,
    )


def _tokenizer(template):
    return SimpleNamespace(chat_template=template)


def test_named_function_seeds_the_call():
    prefill = _forced_tool_call_prefill(
        _tokenizer(_TEMPLATE),
        _params(
            tools=_TOOLS,
            tool_choice={"type": "function", "function": {"name": "bash"}},
        ),
    )
    assert prefill == '<tool_call>\n{"name": "bash", "arguments": '


def test_required_forces_some_call():
    prefill = _forced_tool_call_prefill(
        _tokenizer(_TEMPLATE), _params(tools=_TOOLS, tool_choice="required")
    )
    assert prefill == "<tool_call>"


def test_auto_leaves_the_model_free():
    assert (
        _forced_tool_call_prefill(
            _tokenizer(_TEMPLATE), _params(tools=_TOOLS, tool_choice="auto")
        )
        is None
    )


def test_none_choice_is_noop():
    assert (
        _forced_tool_call_prefill(
            _tokenizer(_TEMPLATE), _params(tools=_TOOLS, tool_choice=None)
        )
        is None
    )


def test_no_tools_is_noop():
    assert (
        _forced_tool_call_prefill(
            _tokenizer(_TEMPLATE), _params(tools=None, tool_choice="required")
        )
        is None
    )


def test_uninferable_template_is_noop():
    assert (
        _forced_tool_call_prefill(
            _tokenizer("no tool markers here"),
            _params(tools=_TOOLS, tool_choice="required"),
        )
        is None
    )
