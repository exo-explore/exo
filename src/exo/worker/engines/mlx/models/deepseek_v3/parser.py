from collections.abc import Callable, Generator

from exo.api.types import ToolCallItem
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.runner.bootstrap import logger


def parse_deepseek_v32(
    responses: Generator[GenerationResponse | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    """Parse DeepSeek V3.2 DSML tool calls from the generation stream.

    Uses accumulated-text matching (not per-token marker checks) because
    DSML markers like <｜DSML｜function_calls> may span multiple tokens.
    Thinking tag handling is delegated to parse_thinking_models, which
    wraps this parser in apply_all_parsers.
    """
    from exo.worker.engines.mlx.models.deepseek_v3.dsml_encoding import (
        TOOL_CALLS_END,
        TOOL_CALLS_START,
        parse_dsml_output,
    )

    return parse_dsml_stream(
        responses, TOOL_CALLS_START, TOOL_CALLS_END, parse_dsml_output
    )


def parse_dsml_stream(
    responses: Generator[GenerationResponse | None],
    tool_calls_start: str,
    tool_calls_end: str,
    parse_body: Callable[[str], list[ToolCallItem] | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    accumulated = ""
    in_tool_call = False
    # Tokens buffered while we detect the start of a DSML block
    pending_buffer: list[GenerationResponse] = []
    # Text accumulated during a tool call block
    tool_call_text = ""

    def _try_parse_tool_call(
        text: str, response: GenerationResponse
    ) -> ToolCallResponse | GenerationResponse:
        parsed = parse_body(text)
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
                    if tool_calls_end in tool_call_text
                    else response.model_copy(update={"text": tool_call_text})
                )
            elif tool_calls_start in response.text and tool_calls_end in response.text:
                dsml_start = response.text.index(tool_calls_start)
                before = response.text[:dsml_start]
                if before:
                    yield response.model_copy(update={"text": before})
                yield _try_parse_tool_call(response.text[dsml_start:], response)
            else:
                yield response
            break

        if in_tool_call:
            tool_call_text += response.text
            if tool_calls_end in tool_call_text:
                yield _try_parse_tool_call(tool_call_text, response)
                in_tool_call = False
                tool_call_text = ""
            continue

        accumulated += response.text

        if tool_calls_start in accumulated:
            start_idx = accumulated.index(tool_calls_start)
            pre_text = accumulated[:start_idx]
            # Flush pending buffer tokens that contributed text before the marker
            if pre_text:
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

            if tool_calls_end in tool_call_text:
                yield _try_parse_tool_call(tool_call_text, response)
                tool_call_text = ""
            else:
                in_tool_call = True
            continue

        if _could_be_marker_prefix(accumulated, tool_calls_start):
            pending_buffer.append(response)
            continue

        # No partial match — flush all pending tokens and the current one
        yield from pending_buffer
        pending_buffer.clear()
        accumulated = ""
        yield response

    # Flush any remaining pending buffer at generator end
    yield from pending_buffer


def _could_be_marker_prefix(text: str, marker: str) -> bool:
    max_check = len(marker)
    tail = text[-max_check:] if len(text) > max_check else text
    for i in range(len(tail)):
        suffix = tail[i:]
        if marker.startswith(suffix):
            return True
    return False
