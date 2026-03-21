from collections.abc import Generator
from functools import cache
from typing import Any

from exo_core.types.common import ModelId
from exo_core.types.runner_response import GenerationResponse, ToolCallResponse
from mlx_lm.tokenizer_utils import TokenizerWrapper
from openai_harmony import (
    HarmonyEncodingName,
    HarmonyError,
    Role,
    StreamableParser,
    load_harmony_encoding,
)

from exo.api.types import ToolCallItem
from exo.worker.engines.mlx.utils_mlx import (
    detect_thinking_prompt_suffix,
)
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.tool_parsers import ToolParser


@cache
def get_gpt_oss_encoding():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoding


def apply_all_parsers(
    receiver: Generator[GenerationResponse | None],
    prompt: str,
    tool_parser: ToolParser | None,
    tokenizer: TokenizerWrapper,
    model_id: ModelId,
    tools: list[dict[str, Any]] | None,
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    gen = receiver

    if tokenizer.has_thinking:
        gen = parse_thinking_models(
            gen,
            tokenizer.think_start,
            tokenizer.think_end,
            starts_in_thinking=detect_thinking_prompt_suffix(prompt, tokenizer),
        )

    lower = model_id.normalize().lower()
    if "gpt-oss" in lower or "gpt_oss" in lower:
        gen = parse_gpt_oss(gen)
    elif "deepseek" in lower:
        gen = parse_deepseek_v32(gen)
    elif tool_parser:
        gen = parse_tool_calls(gen, tool_parser, tools)

    return gen


_GPT_OSS_CHANNEL_TOKEN = 200005
_GPT_OSS_MESSAGE_TOKEN = 200008


def parse_gpt_oss(
    responses: Generator[GenerationResponse | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    encoding = get_gpt_oss_encoding()
    stream = StreamableParser(encoding, role=Role.ASSISTANT)
    thinking = False
    current_tool_name: str | None = None
    tool_arg_parts: list[str] = []

    for response in responses:
        if response is None:
            yield None
            continue

        token_id = response.token

        try:
            stream.process(token_id)
        except HarmonyError as e:
            logger.error(
                f"HarmonyError on token_id={response.token} text={response.text!r}: {e}"
            )
            return

        delta = stream.last_content_delta
        ch = stream.current_channel
        recipient = stream.current_recipient

        effective_recipient = (
            recipient
            if (recipient is not None and recipient.startswith("functions."))
            else None
        )
        if effective_recipient != current_tool_name:
            if current_tool_name is not None:
                tool_name = current_tool_name.removeprefix("functions.")
                logger.info(f"parse_gpt_oss yielding tool call: name={tool_name!r}")
                yield ToolCallResponse(
                    tool_calls=[
                        ToolCallItem(
                            name=tool_name,
                            arguments="".join(tool_arg_parts).strip(),
                        )
                    ],
                    usage=response.usage,
                )
                tool_arg_parts = []
            current_tool_name = effective_recipient

        if current_tool_name is not None:
            if delta:
                tool_arg_parts.append(delta)
            if response.finish_reason is not None:
                yield response.model_copy(update={"text": "".join(tool_arg_parts)})
                tool_arg_parts = []
            continue

        is_suppressed = ch == "analysis" or (
            recipient is not None and recipient.startswith("!")
        )

        if is_suppressed and not thinking:
            thinking = True

        if not is_suppressed and thinking:
            thinking = False

        if delta:
            yield response.model_copy(update={"text": delta, "is_thinking": thinking})

        if response.finish_reason is not None:
            yield response.model_copy(update={"text": ""})


def parse_deepseek_v32(
    responses: Generator[GenerationResponse | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
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
        if response is None:
            yield None
            continue

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


def parse_thinking_models(
    responses: Generator[GenerationResponse | None],
    think_start: str | None,
    think_end: str | None,
    starts_in_thinking: bool = True,
) -> Generator[GenerationResponse | None]:
    """Route thinking tokens via is_thinking flag.

    Swallows think tag tokens, sets is_thinking on all others.
    Always yields tokens with finish_reason to avoid hanging the chunk stream.
    """
    is_thinking = starts_in_thinking
    for response in responses:
        if response is None:
            yield None
            continue
        if response.finish_reason is not None:
            yield response.model_copy(update={"is_thinking": False})
            continue

        if response.text == think_start:
            is_thinking = True
            continue
        if response.text == think_end:
            is_thinking = False
            continue

        yield response.model_copy(update={"is_thinking": is_thinking})


def parse_tool_calls(
    responses: Generator[GenerationResponse | None],
    tool_parser: ToolParser,
    tools: list[dict[str, Any]] | None,
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    in_tool_call = False
    tool_call_text_parts: list[str] = []
    for response in responses:
        if response is None:
            yield None
            continue

        if not in_tool_call and response.text.startswith(tool_parser.start_parsing):
            in_tool_call = True

        if not in_tool_call:
            yield response
            continue

        tool_call_text_parts.append(response.text)
        if response.text.endswith(tool_parser.end_parsing):
            # parse the actual tool calls from the tool call text
            combined = "".join(tool_call_text_parts)
            parsed = tool_parser.parse(combined.strip(), tools=tools)
            logger.info(f"parsed {tool_call_text_parts=} into {parsed=}")
            in_tool_call = False
            tool_call_text_parts = []

            if parsed is None:
                logger.warning(f"tool call parsing failed for text {combined}")
                yield response.model_copy(update={"text": combined})
                continue

            yield ToolCallResponse(
                tool_calls=parsed, usage=response.usage, stats=response.stats
            )
            continue

        if response.finish_reason is not None:
            logger.info(
                "tool call parsing interrupted, yield partial tool call as text"
            )
            response = response.model_copy(
                update={
                    "text": "".join(tool_call_text_parts),
                    "token": 0,
                }
            )
            yield response
