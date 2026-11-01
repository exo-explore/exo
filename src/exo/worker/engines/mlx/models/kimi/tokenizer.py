"""Kimi's tokenizer: a custom TikToken tokenizer shipped as remote code, which
transformers 5.x can't load through AutoTokenizer, and its tool-call format."""

import json
import sys
from pathlib import Path
from typing import Any

from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.common import ModelId


def applies_to(model_id: ModelId) -> bool:
    return "kimi-k2" in model_id.lower()


def load_tokenizer(
    model_path: Path, eos_token_ids: list[int] | None
) -> TokenizerWrapper:
    import importlib.util
    import types

    sys.path.insert(0, str(model_path))

    # Load tool_declaration_ts first (tokenization_kimi imports it with relative import)
    tool_decl_path = model_path / "tool_declaration_ts.py"
    if tool_decl_path.exists():
        spec = importlib.util.spec_from_file_location(
            "tool_declaration_ts", tool_decl_path
        )
        if spec and spec.loader:
            tool_decl_module = importlib.util.module_from_spec(spec)
            sys.modules["tool_declaration_ts"] = tool_decl_module
            spec.loader.exec_module(tool_decl_module)

    # Load tokenization_kimi with patched source (convert relative to absolute import)
    tok_path = model_path / "tokenization_kimi.py"
    source = tok_path.read_text()
    source = source.replace("from .tool_declaration_ts", "from tool_declaration_ts")
    spec = importlib.util.spec_from_file_location("tokenization_kimi", tok_path)
    if spec:
        tok_module = types.ModuleType("tokenization_kimi")
        tok_module.__file__ = str(tok_path)
        sys.modules["tokenization_kimi"] = tok_module
        exec(compile(source, tok_path, "exec"), tok_module.__dict__)  # noqa: S102
        TikTokenTokenizer = tok_module.TikTokenTokenizer  # type: ignore[attr-defined]  # noqa: N806
    else:
        from tokenization_kimi import TikTokenTokenizer  # type: ignore[import-not-found]  # noqa: I001

    hf_tokenizer: Any = TikTokenTokenizer.from_pretrained(model_path)  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]

    # Patch encode to use internal tiktoken model directly
    # transformers 5.x has a bug in the encode->pad path for slow tokenizers
    def _patched_encode(text: str, **_kwargs: object) -> list[int]:
        # Pass allowed_special="all" to handle special tokens like <|im_user|>
        return list(hf_tokenizer.model.encode(text, allowed_special="all"))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

    hf_tokenizer.encode = _patched_encode
    return TokenizerWrapper(
        hf_tokenizer,
        eos_token_ids=eos_token_ids,
        tool_call_start="<|tool_calls_section_begin|>",
        tool_call_end="<|tool_calls_section_end|>",
        tool_parser=_parse_kimi_tool_calls,
    )


def _parse_kimi_tool_calls(text: str):
    import regex as re

    # kimi has a fixed function naming scheme, with a json formatted arg
    #   functions.multiply:0<|tool_call_argument_begin|>{"a": 2, "b": 3}
    _func_name_regex = re.compile(
        r"^\s*((?:functions\.)?(.+?):\d+)\s*<\|tool_call_argument_begin\|>", re.DOTALL
    )
    _func_arg_regex = re.compile(r"<\|tool_call_argument_begin\|>\s*(.*)\s*", re.DOTALL)
    _tool_call_split_regex = re.compile(
        r"<\|tool_call_begin\|>(.*?)<\|tool_call_end\|>", re.DOTALL
    )

    def _parse_single_tool(text: str) -> dict[str, Any]:
        func_name_match = _func_name_regex.search(text)
        if func_name_match is None:
            raise ValueError("No tool call found.")
        tool_call_id = func_name_match.group(1)  # e.g. "functions.get_weather:0"
        func_name = func_name_match.group(2)  # e.g. "get_weather"

        func_args_match = _func_arg_regex.search(text)
        if func_args_match is None:
            raise ValueError("No tool call arguments found.")
        func_args = func_args_match.group(1)
        arg_dct = json.loads(func_args)  # pyright: ignore[reportAny]

        return dict(id=tool_call_id, name=func_name, arguments=arg_dct)  # pyright: ignore[reportAny]

    tool_matches = _tool_call_split_regex.findall(text)
    if tool_matches:
        return [_parse_single_tool(match) for match in tool_matches]  # pyright: ignore[reportAny]
    else:
        return [_parse_single_tool(text)]
