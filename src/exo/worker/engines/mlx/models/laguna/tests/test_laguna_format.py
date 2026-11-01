"""Laguna's stop tokens and tool-call format.

The tool-call strings are what Laguna's official chat template renders for an assistant
turn: the function name straight after <tool_call>, no newlines, string values as
written and other values as JSON.
"""

import json

import pytest
from mlx_lm.tool_parsers import glm47

from exo.worker.engines.mlx.models.spec_registry import eos_token_ids_for_model_id
from exo.worker.runner.llm_inference.tool_parsers import ToolParser, make_mlx_parser

# 〈|EOS|〉, </assistant>
_LAGUNA_EOS = [2, 24]

_EDIT_FILE = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "line": {"type": "integer"},
                "options": {"type": "object"},
            },
            "required": ["path", "content"],
        },
    },
}


@pytest.mark.parametrize(
    "model_id",
    [
        "poolside/Laguna-S-2.1",
        "poolside/Laguna-XS-2.1",
        "poolside/Laguna-XS-2.1-NVFP4-mlx",
        "mlx-community/Laguna-S-2.1-oQ4e",
        "poolside/Laguna-XS.2",
        "someone/Laguna-M.1-8bit",
    ],
)
def test_laguna_stops_at_the_end_of_its_turn(model_id: str) -> None:
    assert eos_token_ids_for_model_id(model_id) == _LAGUNA_EOS


@pytest.mark.parametrize(
    "model_id",
    [
        "Laguna198/OpenThinker3-7B",
        "someone/qwen3-8b-laguna",
        "someone/Unrelated-Model",
    ],
)
def test_other_ids_mentioning_laguna_keep_their_own_stop_tokens(model_id: str) -> None:
    assert eos_token_ids_for_model_id(model_id) != _LAGUNA_EOS


def _parser() -> ToolParser:
    return make_mlx_parser(
        glm47.tool_call_start, glm47.tool_call_end, glm47.parse_tool_call
    )


def test_a_rendered_tool_call_parses_back_to_its_arguments() -> None:
    text = (
        "<tool_call>edit_file<arg_key>path</arg_key><arg_value>src/app.py</arg_value>"
        "<arg_key>content</arg_key><arg_value>[1, 2]</arg_value>"
        "<arg_key>line</arg_key><arg_value>12</arg_value>"
        '<arg_key>options</arg_key><arg_value>{"dry_run": true}</arg_value></tool_call>'
    )
    calls = _parser().parse(text, [_EDIT_FILE])
    assert calls is not None and len(calls) == 1
    assert calls[0].name == "edit_file"
    # content is declared a string, so it stays as written even though it is valid JSON.
    assert json.loads(calls[0].arguments) == {
        "path": "src/app.py",
        "content": "[1, 2]",
        "line": 12,
        "options": {"dry_run": True},
    }


def test_a_tool_call_without_arguments_parses() -> None:
    calls = _parser().parse("<tool_call>list_files</tool_call>", None)
    assert calls is not None and len(calls) == 1
    assert calls[0].name == "list_files"
    assert json.loads(calls[0].arguments) == {}


def test_laguna_prompts_keep_a_bare_think_end() -> None:
    """With thinking off the template renders `<assistant></think>`; exo must not put a
    <think> before that </think> for Laguna, and still does for other models."""
    import mlx.core as mx
    from mlx_lm.models.laguna import Model, ModelArgs

    from exo.shared.types.common import ModelId
    from exo.shared.types.text_generation import (
        InputMessage,
        InputMessageContent,
        TextGenerationTaskParams,
    )
    from exo.worker.engines.mlx.models.registry import keeps_think_tags, model_type_of

    laguna = Model(
        ModelArgs(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            vocab_size=64,
        )
    )
    mx.eval(laguna.parameters())
    task = TextGenerationTaskParams(
        model=ModelId("poolside/Laguna-XS-2.1"),
        input=[InputMessage(role="user", content=InputMessageContent("hi"))],
    )
    assert model_type_of(laguna) == "laguna"
    assert keeps_think_tags(task, model_type_of(laguna))
    assert not keeps_think_tags(task, "llama")
