import json
from collections.abc import Generator
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger
from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
    HarmonyEncodingName,
    HarmonyError,  # pyright: ignore[reportUnknownVariableType]
    Role,
    StreamableParser,
    load_harmony_encoding,
)

from exo.shared.types.api import ToolCallItem
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)

if TYPE_CHECKING:
    from mlx_lm.tokenizer_utils import TokenizerWrapper


def parse_thinking_models(
    responses: Generator[GenerationResponse, None, None],
    tokenizer: "TokenizerWrapper",
    starts_in_thinking: bool = True,
) -> Generator[GenerationResponse | ToolCallResponse, None, None]:
    """Route thinking tokens via is_thinking flag.

    Swallows think tag tokens, sets is_thinking on all others.
    Always yields tokens with finish_reason to avoid hanging the chunk stream.
    """
    in_thinking = starts_in_thinking
    for response in responses:
        if isinstance(response, ToolCallResponse):
            yield response
            continue

        is_think_tag = (
            tokenizer.think_end is not None and response.text == tokenizer.think_end
        ) or (
            tokenizer.think_start is not None and response.text == tokenizer.think_start
        )

        if is_think_tag:
            in_thinking = response.text != tokenizer.think_end
            # Never swallow finish_reason — the chunk stream needs it to terminate.
            if response.finish_reason is not None:
                yield response.model_copy(update={"text": "", "is_thinking": False})
            continue
        yield response.model_copy(update={"is_thinking": in_thinking})


@dataclass
class ToolParser:
    start_parsing: str
    end_parsing: str
    parse_tool_calls: Callable[[str], list[ToolCallItem] | None]


def make_mlx_parser(
    tool_call_start: str,
    tool_call_end: str,
    tool_parser: Callable[[str], dict[str, Any] | list[dict[str, Any]]],
) -> ToolParser:
    def parse_tool_calls(text: str) -> list[ToolCallItem] | None:
        try:
            text = text.removeprefix(tool_call_start)
            text = text.removesuffix(tool_call_end)
            parsed = tool_parser(text)
            if isinstance(parsed, list):
                return [ToolCallItem.model_validate(_flatten(p)) for p in parsed]
            else:
                return [ToolCallItem.model_validate(_flatten(parsed))]

        except Exception:
            return None

    return ToolParser(
        start_parsing=tool_call_start,
        end_parsing=tool_call_end,
        parse_tool_calls=parse_tool_calls,
    )


# TODO / example code:
def _parse_json_calls(text: str) -> list[ToolCallItem] | None:
    try:
        text = text.removeprefix("<tool_call>")
        text = text.removesuffix("</tool_call>")
        top_level = {
            k: json.dumps(v) if isinstance(v, (dict, list)) else v
            for k, v in json.loads(text).items()  # pyright: ignore[reportAny]
        }
        return [ToolCallItem.model_validate(top_level)]
    except Exception:
        return None


def _flatten(p: dict[str, Any]) -> dict[str, str]:
    return {
        k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)  # pyright: ignore[reportAny]
        for k, v in p.items()  # pyright: ignore[reportAny]
    }


json_tool_parser = ToolParser(
    start_parsing="<tool_call>",
    end_parsing="</tool_call>",
    parse_tool_calls=_parse_json_calls,
)


def infer_tool_parser(chat_template: str) -> ToolParser | None:
    """Attempt to auto-infer a tool parser from the chat template."""
    if "<tool_call>" in chat_template and "tool_call.name" in chat_template:
        return json_tool_parser
    return None


@cache
def _get_gpt_oss_encoding():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoding


def parse_gpt_oss(
    responses: Generator[GenerationResponse, None, None],
) -> Generator[GenerationResponse | ToolCallResponse, None, None]:
    """Parse GPT-OSS model outputs using Harmony encoding."""
    encoding = _get_gpt_oss_encoding()
    stream = StreamableParser(encoding, role=Role.ASSISTANT)
    thinking = False
    current_tool_name: str | None = None
    tool_arg_parts: list[str] = []

    for response in responses:
        assert isinstance(response, GenerationResponse)
        try:
            stream.process(response.token)
        except HarmonyError:
            logger.error("Encountered critical Harmony Error, returning early")
            return

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

        if current_tool_name is not None:
            if delta:
                tool_arg_parts.append(delta)
            continue

        if ch == "analysis" and not thinking:
            thinking = True

        if ch != "analysis" and thinking:
            thinking = False

        if delta:
            yield response.model_copy(update={"text": delta, "is_thinking": thinking})

        if response.finish_reason is not None:
            yield response


def parse_deepseek_v32(
    responses: Generator[GenerationResponse, None, None],
) -> Generator[GenerationResponse | ToolCallResponse, None, None]:
    """Parse DeepSeek V3.2 DSML tool calls from the generation stream.

    Uses accumulated-text matching (not per-token marker checks) because
    DSML markers like <｜DSML｜function_calls> may span multiple tokens.
    Also handles <think>...</think> blocks for thinking mode.
    """
    from exo.worker.engines.mlx.dsml_encoding import (
        THINKING_END,
        THINKING_START,
        TOOL_CALLS_END,
        TOOL_CALLS_START,
        parse_dsml_output,
    )

    accumulated = ""
    in_tool_call = False
    thinking = False
    # Tokens buffered while we detect the start of a DSML block
    pending_buffer: list[GenerationResponse] = []
    # Text accumulated during a tool call block
    tool_call_text = ""

    for response in responses:
        assert isinstance(response, GenerationResponse)

        # ── Handle thinking tags ──
        if not thinking and THINKING_START in response.text:
            thinking = True
            # Yield any text before the <think> tag
            before = response.text[: response.text.index(THINKING_START)]
            if before:
                yield response.model_copy(update={"text": before})
            continue

        if thinking and THINKING_END in response.text:
            thinking = False
            # Yield any text after the </think> tag
            after = response.text[
                response.text.index(THINKING_END) + len(THINKING_END) :
            ]
            if after:
                yield response.model_copy(update={"text": after, "is_thinking": False})
            continue

        if thinking:
            yield response.model_copy(update={"is_thinking": True})
            continue

        # ── Handle tool call accumulation ──
        if in_tool_call:
            tool_call_text += response.text
            if TOOL_CALLS_END in tool_call_text:
                # Parse the accumulated DSML block
                parsed = parse_dsml_output(tool_call_text)
                if parsed is not None:
                    logger.info(f"parsed DSML tool calls: {parsed}")
                    yield ToolCallResponse(
                        tool_calls=parsed,
                        usage=response.usage,
                        stats=response.stats,
                    )
                else:
                    logger.warning(
                        f"DSML tool call parsing failed for: {tool_call_text}"
                    )
                    yield response.model_copy(update={"text": tool_call_text})
                in_tool_call = False
                tool_call_text = ""
                continue

            # EOS reached before end marker — yield buffered text as-is
            if response.finish_reason is not None:
                logger.info("DSML tool call parsing interrupted by EOS")
                yield response.model_copy(update={"text": tool_call_text})
                in_tool_call = False
                tool_call_text = ""
            continue

        # ── Detect start of tool call block ──
        accumulated += response.text

        if TOOL_CALLS_START in accumulated:
            # The start marker might be split across pending_buffer + current token
            start_idx = accumulated.index(TOOL_CALLS_START)
            # Yield any pending tokens that are purely before the marker
            pre_text = accumulated[:start_idx]
            if pre_text:
                # Flush pending buffer tokens that contributed text before the marker
                for buf_resp in pending_buffer:
                    if pre_text:
                        chunk = buf_resp.text
                        if len(chunk) <= len(pre_text):
                            yield buf_resp
                            pre_text = pre_text[len(chunk) :]
                        else:
                            yield buf_resp.model_copy(update={"text": pre_text})
                            pre_text = ""
            pending_buffer = []
            tool_call_text = accumulated[start_idx:]
            accumulated = ""

            # Check if the end marker is already present (entire tool call in one token)
            if TOOL_CALLS_END in tool_call_text:
                parsed = parse_dsml_output(tool_call_text)
                if parsed is not None:
                    logger.info(f"parsed DSML tool calls: {parsed}")
                    yield ToolCallResponse(
                        tool_calls=parsed,
                        usage=response.usage,
                        stats=response.stats,
                    )
                else:
                    logger.warning(
                        f"DSML tool call parsing failed for: {tool_call_text}"
                    )
                    yield response.model_copy(update={"text": tool_call_text})
                tool_call_text = ""
            else:
                in_tool_call = True
            continue

        # Check if accumulated text might be the start of a DSML marker
        # Buffer tokens if we see a partial match at the end
        if _could_be_dsml_prefix(accumulated):
            pending_buffer.append(response)
            continue

        # No partial match — flush all pending tokens and the current one
        for buf_resp in pending_buffer:
            yield buf_resp
        pending_buffer = []
        accumulated = ""
        yield response

    # Flush any remaining pending buffer at generator end
    for buf_resp in pending_buffer:
        yield buf_resp


def _could_be_dsml_prefix(text: str) -> bool:
    """Check if the end of text could be the start of a DSML function_calls marker.

    We look for suffixes of text that are prefixes of the TOOL_CALLS_START pattern.
    This allows us to buffer tokens until we can determine if a tool call is starting.
    """
    from exo.worker.engines.mlx.dsml_encoding import TOOL_CALLS_START

    # Only check the last portion of text that could overlap with the marker
    max_check = len(TOOL_CALLS_START)
    tail = text[-max_check:] if len(text) > max_check else text

    # Check if any suffix of tail is a prefix of TOOL_CALLS_START
    for i in range(len(tail)):
        suffix = tail[i:]
        if TOOL_CALLS_START.startswith(suffix):
            return True
    return False
