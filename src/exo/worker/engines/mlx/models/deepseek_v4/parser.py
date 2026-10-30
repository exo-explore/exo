from collections.abc import Generator

from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.engines.mlx.models.deepseek_v3.dsml_encoding import parse_dsml_output
from exo.worker.engines.mlx.models.deepseek_v3.parser import parse_dsml_stream


def parse_deepseek_v4(
    responses: Generator[GenerationResponse | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    dsml_token = "｜DSML｜"
    start = f"<{dsml_token}tool_calls>"
    end = f"</{dsml_token}tool_calls>"
    return parse_dsml_stream(responses, start, end, parse_dsml_output)
