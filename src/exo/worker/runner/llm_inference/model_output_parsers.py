from collections.abc import Generator
from functools import cache

from mlx_lm.tokenizer_utils import TokenizerWrapper
from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
    HarmonyEncodingName,
    HarmonyError,  # pyright: ignore[reportUnknownVariableType]
    Role,
    StreamableParser,
    load_harmony_encoding,
)

from exo.shared.types.api import ToolCallItem
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.tool_parsers import ToolParser


@cache
def get_gpt_oss_encoding():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoding


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
        try:
            stream.process(response.token)
        except HarmonyError:
            logger.error("Encountered critical Harmony Error, returning early")
            return

        delta = stream.last_content_delta
        ch = stream.current_channel
        recipient = stream.current_recipient

        # Debug: log every token with state
        logger.debug(
            f"parse_gpt_oss token={response.token} text={response.text!r} "
            f"recipient={recipient!r} ch={ch!r} delta={delta!r} "
            f"state={stream.state} current_tool={current_tool_name!r}"
        )

        if recipient != current_tool_name:
            if current_tool_name is not None:
                prefix = "functions."
                if current_tool_name.startswith(prefix):
                    current_tool_name = current_tool_name[len(prefix) :]
                logger.info(
                    f"parse_gpt_oss yielding tool call: name={current_tool_name!r}"
                )
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

        if ch != "analysis" and thinking:
            thinking = False

        if delta:
            yield response.model_copy(update={"text": delta, "is_thinking": thinking})

        if response.finish_reason is not None:
            yield response


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
    tokenizer: TokenizerWrapper,
    starts_in_thinking: bool = True,
) -> Generator[GenerationResponse | None]:
    """Route thinking tokens via is_thinking flag.

    Swallows think tag tokens, sets is_thinking on all others.
    Always yields tokens with finish_reason to avoid hanging the chunk stream.
    """
    in_thinking = starts_in_thinking
    for response in responses:
        if response is None:
            yield None
            continue
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


def parse_tool_calls(
    responses: Generator[GenerationResponse | None], tool_parser: ToolParser
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    in_tool_call = False
    tool_call_text_parts: list[str] = []
    for response in responses:
        if response is None:
            yield None
            continue
        if not in_tool_call and response.text.startswith(tool_parser.start_parsing):
            in_tool_call = True

        if in_tool_call:
            tool_call_text_parts.append(response.text)
            if response.text.endswith(tool_parser.end_parsing):
                # parse the actual tool calls from the tool call text
                parsed = tool_parser.parse_tool_calls(
                    "".join(tool_call_text_parts).strip()
                )
                logger.info(f"parsed {tool_call_text_parts=} into {parsed=}")
                if parsed is not None:
                    yield ToolCallResponse(
                        tool_calls=parsed, usage=response.usage, stats=response.stats
                    )
                else:
                    logger.warning(
                        f"tool call parsing failed for text {''.join(tool_call_text_parts)}"
                    )
                    response.text = "".join(tool_call_text_parts)
                    yield response

                in_tool_call = False
                tool_call_text_parts = []
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

        else:
            # fallthrough
            yield response
