from collections.abc import Generator
from functools import cache

from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
    HarmonyEncodingName,
    HarmonyError,  # pyright: ignore[reportUnknownVariableType]
    Role,
    StreamableParser,
    load_harmony_encoding,
)

from exo.api.types import ToolCallItem
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.runner.bootstrap import logger


@cache
def get_gpt_oss_encoding():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoding


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
