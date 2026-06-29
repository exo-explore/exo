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
from exo.worker.engines.mlx.utils_mlx import (
    _forced_tool_call_prefill,
    detect_tool_call_prompt_suffix,
)

_START = "<tool_call>"
_TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]


def _params(*, tools=None, tool_choice=None):
    return TextGenerationTaskParams(
        model=ModelId("test/model"),
        input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        tools=tools,
        tool_choice=tool_choice,
    )


def _tokenizer(tool_call_start):
    return SimpleNamespace(tool_call_start=tool_call_start)


def test_named_function_forces_the_start_marker():
    prefill = _forced_tool_call_prefill(
        _tokenizer(_START),
        _params(
            tools=_TOOLS,
            tool_choice={"type": "function", "function": {"name": "bash"}},
        ),
    )
    assert prefill == _START


def test_required_forces_the_start_marker():
    prefill = _forced_tool_call_prefill(
        _tokenizer(_START), _params(tools=_TOOLS, tool_choice="required")
    )
    assert prefill == _START


def test_auto_leaves_the_model_free():
    assert (
        _forced_tool_call_prefill(
            _tokenizer(_START), _params(tools=_TOOLS, tool_choice="auto")
        )
        is None
    )


def test_none_choice_is_noop():
    assert (
        _forced_tool_call_prefill(
            _tokenizer(_START), _params(tools=_TOOLS, tool_choice=None)
        )
        is None
    )


def test_no_tools_is_noop():
    assert (
        _forced_tool_call_prefill(
            _tokenizer(_START), _params(tools=None, tool_choice="required")
        )
        is None
    )


def test_model_without_tool_marker_is_noop():
    assert (
        _forced_tool_call_prefill(
            _tokenizer(None), _params(tools=_TOOLS, tool_choice="required")
        )
        is None
    )


def test_prompt_suffix_detected_when_prefill_present():
    assert detect_tool_call_prompt_suffix(
        "...<|im_start|>assistant\n<tool_call>", _tokenizer(_START)
    )
    assert not detect_tool_call_prompt_suffix(
        "...<|im_start|>assistant\n", _tokenizer(_START)
    )
