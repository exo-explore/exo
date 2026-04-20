from collections.abc import Generator, Iterator
from functools import cache
from typing import Any

from mlx_lm.models.deepseek_v32 import Model as DeepseekV32Model
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.tokenizer_utils import TokenizerWrapper
from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
    HarmonyEncodingName,
    HarmonyError,  # pyright: ignore[reportUnknownVariableType]
    Role,
    StreamableParser,
    load_harmony_encoding,
)

from exo.api.types import ToolCallItem
from exo.shared.types.chunks import (
    ErrorChunk,
    GenerationChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import ModelId
from exo.shared.types.mlx import Model
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.engines.mlx.utils_mlx import (
    detect_thinking_prompt_suffix,
)
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.tool_parsers import ToolParser


@cache
def get_gpt_oss_encoding():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoding


def count_reasoning_tokens(
    responses: Generator[GenerationResponse | ToolCallResponse | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    """Count tokens with is_thinking=True and patch the total into Usage on the final response."""
    reasoning_tokens = 0
    for response in responses:
        if response is None:
            yield None
            continue
        if isinstance(response, GenerationResponse) and response.is_thinking:
            reasoning_tokens += 1
        if response.usage is not None and reasoning_tokens > 0:
            response = response.model_copy(
                update={
                    "usage": response.usage.model_copy(
                        update={
                            "completion_tokens_details": response.usage.completion_tokens_details.model_copy(
                                update={"reasoning_tokens": reasoning_tokens}
                            )
                        }
                    )
                }
            )
        yield response


def apply_all_parsers(
    receiver: Generator[GenerationResponse | None],
    prompt: str,
    tool_parser: ToolParser | None,
    tokenizer: TokenizerWrapper,
    model_type: type[Model],
    model_id: ModelId,
    tools: list[dict[str, Any]] | None,
) -> Iterator[GenerationChunk | None]:
    generator = receiver

    if issubclass(model_type, GptOssModel):
        generator = parse_gpt_oss(generator)
    elif (
        issubclass(model_type, DeepseekV32Model)
        and "deepseek" in model_id.normalize().lower()
    ):
        generator = parse_deepseek_v32(generator)
    else:
        if tokenizer.has_thinking:
            generator = parse_thinking_models(
                generator,
                tokenizer.think_start,
                tokenizer.think_end,
                starts_in_thinking=detect_thinking_prompt_suffix(prompt, tokenizer),
            )

        if tool_parser:
            generator = parse_tool_calls(generator, tool_parser, tools)

    generator = count_reasoning_tokens(generator)

    return map(lambda r: map_responses_to_chunks(r, model_id), generator)


def map_responses_to_chunks(
    response: GenerationResponse | ToolCallResponse | None, model_id: ModelId
) -> GenerationChunk | None:
    match response:
        case None:
            return None
        case GenerationResponse():
            if response.finish_reason == "error":
                return ErrorChunk(
                    error_message=response.text,
                    model=model_id,
                )
            else:
                finish_reason = response.finish_reason
                assert finish_reason not in (
                    "error",
                    "tool_calls",
                    "function_call",
                )
                return TokenChunk(
                    model=model_id,
                    text=response.text,
                    token_id=response.token,
                    usage=response.usage,
                    finish_reason=finish_reason,
                    stats=response.stats,
                    logprob=response.logprob,
                    top_logprobs=response.top_logprobs,
                    is_thinking=response.is_thinking,
                )
        case ToolCallResponse():
            return ToolCallChunk(
                tool_calls=response.tool_calls,
                model=model_id,
                usage=response.usage,
                stats=response.stats,
            )


def parse_gpt_oss(
    responses: Generator[GenerationResponse | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    encoding = get_gpt_oss_encoding()
    stream = StreamableParser(encoding, role=Role.ASSISTANT)
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
            if response.finish_reason is not None:
                yield response.model_copy(update={"text": "".join(tool_arg_parts)})
                tool_arg_parts = []
            continue

        if delta:
            yield response.model_copy(
                update={"text": delta, "is_thinking": ch == "analysis"}
            )

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

    def _try_parse_tool_call(
        text: str, response: GenerationResponse
    ) -> ToolCallResponse | GenerationResponse:
        parsed = parse_dsml_output(text)
        if parsed is not None:
            return ToolCallResponse(
                tool_calls=parsed, usage=response.usage, stats=response.stats
            )
        logger.warning(f"DSML tool call parsing failed for: {text}")
        return response.model_copy(update={"text": text})

    for response in responses:
        if response is None:
            yield None
            continue

        if response.finish_reason is not None:
            yield from pending_buffer
            pending_buffer.clear()
            if in_tool_call:
                tool_call_text += response.text
                yield (
                    _try_parse_tool_call(tool_call_text, response)
                    if TOOL_CALLS_END in tool_call_text
                    else response.model_copy(update={"text": tool_call_text})
                )
            elif TOOL_CALLS_START in response.text and TOOL_CALLS_END in response.text:
                dsml_start = response.text.index(TOOL_CALLS_START)
                before = response.text[:dsml_start]
                if before:
                    yield response.model_copy(update={"text": before})
                yield _try_parse_tool_call(response.text[dsml_start:], response)
            else:
                yield response
            break

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
                yield _try_parse_tool_call(tool_call_text, response)
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
                    if not pre_text:
                        break
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
                yield _try_parse_tool_call(tool_call_text, response)
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
        yield from pending_buffer
        pending_buffer.clear()
        accumulated = ""
        yield response

    # Flush any remaining pending buffer at generator end
    yield from pending_buffer


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
    accumulated = ""
    pending_buffer: list[GenerationResponse] = []

    def drain_pending(_is_thinking: bool):
        for buffered in pending_buffer:
            yield buffered.model_copy(update={"is_thinking": _is_thinking})
        pending_buffer.clear()

    for response in responses:
        if response is None:
            yield None
            continue

        accumulated += response.text

        if response.finish_reason is not None:
            yield from drain_pending(is_thinking)
            yield response.model_copy(update={"is_thinking": False})
            continue

        if accumulated == think_start and not is_thinking:
            is_thinking = True
            accumulated = ""
            pending_buffer.clear()
            continue
        if accumulated == think_end and is_thinking:
            is_thinking = False
            accumulated = ""
            pending_buffer.clear()
            continue

        if (think_start and accumulated == think_start[: len(accumulated)]) or (
            think_end and accumulated == think_end[: len(accumulated)]
        ):
            pending_buffer.append(response)
            continue

        accumulated = ""

        yield from drain_pending(is_thinking)
        yield response.model_copy(update={"is_thinking": is_thinking})


def parse_tool_calls(
    responses: Generator[GenerationResponse | None],
    tool_parser: ToolParser,
    tools: list[dict[str, Any]] | None,
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    in_tool_call = False
    tool_call_text_parts: list[str] = []
    accumulated_tool_calls: list[ToolCallItem] = []

    for response in responses:
        if response is None:
            yield None
            continue

        if not in_tool_call and response.text.startswith(tool_parser.start_parsing):
            in_tool_call = True

        if (
            not in_tool_call
            and accumulated_tool_calls
            and (response.stats is not None or response.finish_reason is not None)
        ):
            yield ToolCallResponse(
                tool_calls=accumulated_tool_calls,
                usage=response.usage,
                stats=response.stats,
            )
            accumulated_tool_calls.clear()
            continue

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
                yield response.model_copy(
                    update={"text": combined, "token": 0, "finish_reason": "error"}
                )
                break

            accumulated_tool_calls.extend(parsed)
            if accumulated_tool_calls and (
                response.finish_reason is not None or response.stats is not None
            ):
                yield ToolCallResponse(
                    tool_calls=accumulated_tool_calls,
                    usage=response.usage,
                    stats=response.stats,
                )
                accumulated_tool_calls.clear()
            continue

        if response.finish_reason is not None:
            logger.info(
                "tool call parsing interrupted, yield partial tool call as text"
            )
            response = response.model_copy(
                update={
                    "text": "".join(tool_call_text_parts),
                    "token": 0,
                    "finish_reason": "error",
                }
            )
            yield response

    if not accumulated_tool_calls:
        logger.warning("Tool calls should have all been emitted but were not")
