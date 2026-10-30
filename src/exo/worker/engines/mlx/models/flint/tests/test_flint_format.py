"""Flint's stop tokens and tool-call format.

The tool-call strings are what Flint's official chat template renders for an assistant
turn: the function name straight after <tool_call>, no newlines, string values as
written and other values as JSON.
"""

import json

import pytest
from mlx_lm.tool_parsers import glm47

from exo.worker.engines.mlx.models.spec_registry import eos_token_ids_for_model_id
from exo.worker.runner.llm_inference.tool_parsers import ToolParser, make_mlx_parser

# <|end_of_text|>, <|user|>, <|observation|>, <|developer|>
_FLINT_EOS = [150001, 150003, 150005, 150018]

_RUN_SHELL = {
    "type": "function",
    "function": {
        "name": "run_shell",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_s": {"type": "integer"},
                "env": {"type": "object"},
            },
            "required": ["command"],
        },
    },
}


@pytest.mark.parametrize(
    "model_id",
    [
        "RecursiveSI/flint",
        "RecursiveSI/flint-int4",
        "someone/flint-8bit",
        "mlx-community/flint-mixed-3-4bit",
        "someone/Flint-40B-A2.5B-MLX-4bit",
    ],
)
def test_flint_stops_where_the_next_turn_would_start(model_id: str) -> None:
    assert eos_token_ids_for_model_id(model_id) == _FLINT_EOS


@pytest.mark.parametrize(
    "model_id",
    [
        "someone/granite-4.1-8b-FlintStones-V1",
        "someone/gemma-4-E2B-it-flint-GGUF",
        "someone/Unrelated-Model",
    ],
)
def test_other_ids_mentioning_flint_keep_their_own_stop_tokens(model_id: str) -> None:
    assert eos_token_ids_for_model_id(model_id) != _FLINT_EOS


def _parser() -> ToolParser:
    return make_mlx_parser(
        glm47.tool_call_start, glm47.tool_call_end, glm47.parse_tool_call
    )


def test_a_rendered_tool_call_parses_back_to_its_arguments() -> None:
    text = (
        "<tool_call>run_shell<arg_key>command</arg_key>"
        "<arg_value>pytest -q tests/test_queue.py</arg_value>"
        "<arg_key>timeout_s</arg_key><arg_value>30</arg_value>"
        '<arg_key>env</arg_key><arg_value>{"CI": "1", "N": 2}</arg_value></tool_call>'
    )
    calls = _parser().parse(text, [_RUN_SHELL])
    assert calls is not None and len(calls) == 1
    assert calls[0].name == "run_shell"
    assert json.loads(calls[0].arguments) == {
        "command": "pytest -q tests/test_queue.py",
        "timeout_s": 30,
        "env": {"CI": "1", "N": 2},
    }


def test_a_tool_call_without_arguments_parses() -> None:
    calls = _parser().parse("<tool_call>list_open_files</tool_call>", None)
    assert calls is not None and len(calls) == 1
    assert calls[0].name == "list_open_files"
    assert json.loads(calls[0].arguments) == {}
