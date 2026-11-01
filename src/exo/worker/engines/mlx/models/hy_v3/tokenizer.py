"""Hy3's tokenizer: mlx-lm's wrapper, told Hy3's reasoning tags and tool-call format.

The Hy3 release renames the preview's chat tokens with a ``:opensource`` suffix at the
same ids: ``<think:opensource>`` / ``</think:opensource>`` and
``<tool_calls:opensource>`` ... ``</tool_calls:opensource>``. mlx-lm knows neither, so it
found no reasoning tags (reasoning came back as content) and no tool parser (tool calls
came back as text). The MLX conversions' tokenizer_config.json names
``tool_parser_type: hy_v3_opensource``, a parser only the unmerged upstream port had, and
mlx-lm's tokenizer loader fails to import it. This loader doesn't use mlx-lm's.

A tool call, as the official chat template renders it::

    <tool_calls:opensource>
    <tool_call:opensource>get_weather<tool_sep:opensource>
    <arg_key:opensource>city</arg_key:opensource>
    <arg_value:opensource>Paris</arg_value:opensource>
    </tool_call:opensource>
    </tool_calls:opensource>

String arguments are written as they are, every other value as JSON
(``tojson(ensure_ascii=False)``), so an argument is decoded as JSON unless the tool's
schema declares it a string.
"""

import json
import re
from pathlib import Path
from typing import Any, cast

from mlx_lm.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from exo.shared.types.common import ModelId
from exo.worker.engines.mlx.models.hy_v3.model_ids import is_hy3_model_id

_SUFFIX = ":opensource"
# Optional tag suffix, so Hy3-preview's unsuffixed tags parse too.
_TAG_SUFFIX = r"(?::[\w-]+)?"
_TOOL_CALL = re.compile(
    rf"<tool_call{_TAG_SUFFIX}>(.*?)<tool_sep{_TAG_SUFFIX}>(.*?)</tool_call{_TAG_SUFFIX}>",
    re.DOTALL,
)
_ARGUMENT = re.compile(
    rf"<arg_key{_TAG_SUFFIX}>(.*?)</arg_key{_TAG_SUFFIX}>\s*"
    rf"<arg_value{_TAG_SUFFIX}>(.*?)</arg_value{_TAG_SUFFIX}>",
    re.DOTALL,
)


def applies_to(model_id: ModelId) -> bool:
    return is_hy3_model_id(model_id)


def _string_arguments(
    tool_name: str, tools: list[dict[str, Any]] | None
) -> set[str] | None:
    """The argument names the tool's schema declares as strings, or None when the
    request doesn't describe the tool."""
    for tool in tools or []:
        function = cast(dict[str, Any], tool.get("function") or {})
        if function.get("name") != tool_name:
            continue
        parameters = cast(dict[str, object], function.get("parameters") or {})
        properties = cast(dict[str, object], parameters.get("properties") or {})
        strings: set[str] = set()
        for name, schema in properties.items():
            if (
                isinstance(schema, dict)
                and cast(dict[str, object], schema).get("type") == "string"
            ):
                strings.add(name)
        return strings
    return None


def _decode_value(raw: str) -> object:
    try:
        return cast(object, json.loads(raw))
    except json.JSONDecodeError:
        return raw


def parse_tool_calls(
    text: str, tools: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Parse the text between ``<tool_calls...>`` and ``</tool_calls...>``."""
    calls: list[dict[str, Any]] = []
    for name, body in cast(list[tuple[str, str]], _TOOL_CALL.findall(text)):
        tool_name = name.strip()
        strings = _string_arguments(tool_name, tools)
        arguments: dict[str, object] = {}
        for key, raw in cast(list[tuple[str, str]], _ARGUMENT.findall(body)):
            key = key.strip()
            arguments[key] = (
                raw if strings is not None and key in strings else _decode_value(raw)
            )
        calls.append({"name": tool_name, "arguments": arguments})
    if not calls:
        raise ValueError(f"no Hy3 tool call in {text!r}")
    return calls


class HyV3TokenizerWrapper(TokenizerWrapper):
    """mlx-lm's TokenizerWrapper with Hy3's reasoning tags and tool-call parser."""

    def __init__(
        self, tokenizer: PreTrainedTokenizerFast, eos_token_ids: list[int] | None
    ) -> None:
        vocab = tokenizer.get_vocab()
        suffix = _SUFFIX if f"<tool_calls{_SUFFIX}>" in vocab else ""
        super().__init__(
            tokenizer,
            # tokenizer.json's decoder is ByteLevel, for which mlx-lm picks this one.
            BPEStreamingDetokenizer,
            eos_token_ids=eos_token_ids,
            tool_parser=parse_tool_calls,
            tool_call_start=f"<tool_calls{suffix}>",
            tool_call_end=f"</tool_calls{suffix}>",
        )
        think_start, think_end = f"<think{suffix}>", f"</think{suffix}>"
        if think_start in vocab and think_end in vocab:
            self._think_start: str | None = think_start
            self._think_end: str | None = think_end
            self._think_start_tokens: tuple[int, ...] | None = (vocab[think_start],)
            self._think_end_tokens: tuple[int, ...] | None = (vocab[think_end],)


def load_tokenizer(
    model_path: Path, eos_token_ids: list[int] | None
) -> TokenizerWrapper:
    tokenizer = cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained(model_path),  # pyright: ignore[reportUnknownMemberType]
    )
    return HyV3TokenizerWrapper(tokenizer, eos_token_ids)
