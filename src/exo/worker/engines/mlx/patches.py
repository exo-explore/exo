"""
MLX-specific model patches for special models (Kimi, GLM, GPT-OSS, thinking models).
These patches are applied in MlxEngine.generate() to handle model-specific quirks.
"""

import ast
import json
from collections.abc import Generator
from typing import Any, Callable

import regex as re
from mlx_lm.tokenizer_utils import TokenizerWrapper
from openai_harmony import Role, StreamableParser  # pyright: ignore[reportMissingTypeStubs]
from pydantic import ValidationError

from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallItem,
    ToolCallResponse,
)


def filter_kimi_tokens(
    responses: Generator[GenerationResponse | ToolCallResponse, None, None],
) -> Generator[GenerationResponse, None, None]:
    """Filter out Kimi-specific tool call section markers."""
    for resp in responses:
        assert isinstance(resp, GenerationResponse)
        if (
            resp.text == "<|tool_calls_section_begin|>"
            or resp.text == "<|tool_calls_section_end|>"
        ):
            continue
        yield resp


def parse_gpt_oss(
    responses: Generator[GenerationResponse | ToolCallResponse, None, None],
    encoding: Any,
) -> Generator[GenerationResponse | ToolCallResponse, None, None]:
    """Parse GPT-OSS model outputs to match standard format."""
    stream = StreamableParser(encoding, role=Role.ASSISTANT)
    thinking = False
    current_tool_name: str | None = None
    tool_arg_parts: list[str] = []

    for response in responses:
        assert isinstance(response, GenerationResponse)
        stream.process(response.token)

        delta = stream.last_content_delta
        ch = stream.current_channel
        recipient = stream.current_recipient

        if recipient != current_tool_name:
            if current_tool_name is not None:
                prefix = "functions."
                if current_tool_name.startswith(prefix):
                    current_tool_name = current_tool_name[len(prefix) :]
                yield ToolCallResponse(
                    tool_calls=[
                        ToolCallItem(
                            name=current_tool_name,
                            arguments="".join(tool_arg_parts).strip(),
                        )
                    ],
                    usage=response.usage,
                )
                tool_arg_parts = []
            current_tool_name = recipient

        # If inside a tool call, accumulate arguments
        if current_tool_name is not None:
            if delta:
                tool_arg_parts.append(delta)
            continue

        if ch == "analysis" and not thinking:
            thinking = True
            yield response.model_copy(update={"text": "<think>"})

        if ch != "analysis" and thinking:
            thinking = False
            yield response.model_copy(update={"text": "</think>"})

        if delta:
            yield response.model_copy(update={"text": delta})

        if response.finish_reason is not None:
            if thinking:
                yield response.model_copy(update={"text": "</think>"})
            yield response


def parse_thinking_models(
    responses: Generator[GenerationResponse | ToolCallResponse, None, None],
    tokenizer: TokenizerWrapper,
) -> Generator[GenerationResponse | ToolCallResponse, None, None]:
    """
    For models that inject thinking tags in the prompt (like GLM-4.7),
    prepend the thinking tag to the output stream so the frontend
    can properly parse thinking content.
    """
    first = True
    for response in responses:
        if isinstance(response, ToolCallResponse):
            yield response
            continue
        if first:
            first = False
            yield response.model_copy(
                update={
                    "text": tokenizer.think_start,
                    "token": tokenizer.think_start_id,  # type: ignore
                }
            )
        yield response


def parse_tool_calls(
    responses: Generator[GenerationResponse | ToolCallResponse, None, None],
    tool_call_start: str,
    tool_call_end: str,
    tool_parser: Callable[[str], dict[str, Any] | list[dict[str, Any]]],
    logger: Any,
) -> Generator[GenerationResponse | ToolCallResponse, None, None]:
    """Parse tool calls from the generation stream."""
    in_tool_call = False
    tool_call_text_parts: list[str] = []
    for response in responses:
        assert isinstance(response, GenerationResponse)
        # assumption: the tool call start is one token
        if response.text == tool_call_start:
            in_tool_call = True
            continue
        # assumption: the tool call end is one token
        if in_tool_call and response.text == tool_call_end:
            try:
                # tool_parser returns an arbitrarily nested python dictionary
                # we actually don't want the python dictionary, we just want to
                # parse the top level { function: ..., arguments: ... } structure
                # as we're just gonna hand it back to the api anyway
                parsed = tool_parser("".join(tool_call_text_parts).strip())
                logger.info(f"parsed {tool_call_text_parts=} into {parsed=}")
                if isinstance(parsed, list):
                    tools = [_validate_single_tool(tool) for tool in parsed]
                else:
                    tools = [_validate_single_tool(parsed)]
                yield ToolCallResponse(tool_calls=tools, usage=response.usage)

            except (
                json.JSONDecodeError,
                ValidationError,
                ValueError,
                AttributeError,
            ) as e:
                # ValueError: our parsers raise this for malformed tool calls
                # AttributeError: upstream parsers (e.g. glm47) may raise this when regex doesn't match
                logger.opt(exception=e).warning("tool call parsing failed")
                # assumption: talking about tool calls, not making a tool call
                response.text = (
                    tool_call_start + "".join(tool_call_text_parts) + tool_call_end
                )
                yield response

            in_tool_call = False
            tool_call_text_parts = []
            continue

        if in_tool_call:
            tool_call_text_parts.append(response.text)
            continue
        # fallthrough
        yield response


def patch_kimi_tokenizer(tokenizer: TokenizerWrapper) -> None:
    """
    Version of to-be-upstreamed kimi-k2 tool parser.
    """
    # kimi has a fixed function naming scheme, with a json formatted arg
    #   functions.multiply:0 <|tool_call_argument_begin|> {"a": 2, "b": 3}
    _func_name_regex = re.compile(
        r"^\s*(.+):\d+\s*<\|tool_call_argument_begin\|>", re.DOTALL
    )
    _func_arg_regex = re.compile(r"<\|tool_call_argument_begin\|>\s*(.*)\s*", re.DOTALL)

    # kimi has a tool_calls_section - we're leaving this up to the caller to handle
    tool_call_start = "<|tool_call_begin|>"
    tool_call_end = "<|tool_call_end|>"

    def _deserialize(value: str) -> Any:  # pyright: ignore[reportAny]
        try:
            return json.loads(value)  # pyright: ignore[reportAny]
        except Exception:
            pass

        try:
            return ast.literal_eval(value)  # pyright: ignore[reportAny]
        except Exception:
            pass
        return value

    def parse_tool_call(text: str, tools: Any | None = None):
        func_name_match = _func_name_regex.search(text)
        if func_name_match is None:
            raise ValueError(f"Could not parse function name from tool call: {text!r}")
        func_name = func_name_match.group(1)
        # strip off the `functions.` prefix, if it exists.
        func_name = func_name[func_name.find(".") + 1 :]

        func_args_match = _func_arg_regex.search(text)
        if func_args_match is None:
            raise ValueError(f"Could not parse function args from tool call: {text!r}")
        func_args = func_args_match.group(1)
        # the args should be valid json - no need to check against our tools to deserialize
        arg_dct = _deserialize(func_args)  # pyright: ignore[reportAny]

        return dict(name=func_name, arguments=arg_dct)  # pyright: ignore[reportAny]

    tokenizer._tool_call_start = tool_call_start
    tokenizer._tool_call_end = tool_call_end
    tokenizer._tool_parser = parse_tool_call


def patch_glm_tokenizer(tokenizer: TokenizerWrapper) -> None:
    """
    Fixed version of mlx_lm's glm47 tool parser that handles regex match failures.
    """
    _func_name_regex = re.compile(r"^(.*?)<arg_key>", re.DOTALL)
    _func_arg_regex = re.compile(
        r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>",
        re.DOTALL,
    )

    tool_call_start = "<tool_call>"
    tool_call_end = "</tool_call>"

    def _is_string_type(
        tool_name: str,
        arg_name: str,
        tools: list[Any] | None,
    ) -> bool:
        if tools is None:
            return False
        for tool in tools:  # pyright: ignore[reportAny]
            func = tool["function"]  # pyright: ignore[reportAny]
            if func["name"] == tool_name:
                params = func["parameters"]  # pyright: ignore[reportAny]
                if params is None:
                    return False
                props = params.get("properties", {})  # pyright: ignore[reportAny]
                arg_props = props.get(arg_name, {})  # pyright: ignore[reportAny]
                arg_type = arg_props.get("type", None)  # pyright: ignore[reportAny]
                return arg_type == "string"  # pyright: ignore[reportAny]
        return False

    def _deserialize(value: str) -> Any:  # pyright: ignore[reportAny]
        try:
            return json.loads(value)  # pyright: ignore[reportAny]
        except Exception:
            pass
        try:
            return ast.literal_eval(value)  # pyright: ignore[reportAny]
        except Exception:
            pass
        return value

    def parse_tool_call(text: str, tools: list[Any] | None = None):
        func_name_match = _func_name_regex.search(text)
        if func_name_match is None:
            raise ValueError(f"Could not parse function name from tool call: {text!r}")
        func_name = func_name_match.group(1)

        pairs = _func_arg_regex.findall(text)
        arg_dct: dict[str, Any] = {}
        for key, value in pairs:  # pyright: ignore[reportAny]
            arg_key = key.strip()  # pyright: ignore[reportAny]
            arg_val = value.strip()  # pyright: ignore[reportAny]
            if not _is_string_type(func_name, arg_key, tools):  # pyright: ignore[reportAny]
                arg_val = _deserialize(arg_val)  # pyright: ignore[reportAny]
            arg_dct[arg_key] = arg_val
        return dict(name=func_name, arguments=arg_dct)

    tokenizer._tool_call_start = tool_call_start
    tokenizer._tool_call_end = tool_call_end
    tokenizer._tool_parser = parse_tool_call


def _validate_single_tool(obj: dict[str, Any]) -> ToolCallItem:
    """Validate and convert a tool call dictionary to ToolCallItem."""
    if (
        ((name := obj.get("name")) is not None)
        and ((args := obj.get("arguments")) is not None)
        and isinstance(name, str)
    ):
        return ToolCallItem(name=name, arguments=json.dumps(args))
    else:
        raise ValidationError
