from mlx_lm.models.minimax import Model as MiniMaxModel
from mlx_lm.models.minimax_m3_vl import Model as MiniMaxM3Model

from exo.worker.engines.mlx.models.family import (
    Architecture,
    ModelFamily,
    OutputParser,
)
from exo.worker.engines.mlx.models.minimax.parser import parse_minimax_m3
from exo.worker.engines.mlx.models.minimax.reasoning import (
    HOOK as REASONING_EFFORT_HOOK,
)
from exo.worker.engines.mlx.models.minimax.reasoning import (
    KWARGS_HOOK as TEMPLATE_KWARGS_HOOK,
)
from exo.worker.engines.mlx.models.minimax.sharding import (
    MiniMaxM3ShardingStrategy,
    MiniMaxShardingStrategy,
)
from exo.worker.engines.mlx.models.minimax.spec import SPEC

FAMILY = ModelFamily(
    spec=SPEC,
    architectures=(
        # Also minimax_m2, which mlx-lm loads as minimax.
        Architecture(
            model_type="minimax",
            model_class=MiniMaxModel,
            tensor_parallel=MiniMaxShardingStrategy,
        ),
        # mlx-lm's minimax_m3_vl is the text model; the checkpoints' vision tower isn't
        # loaded. Every layer keeps its own KV cache, so a pipeline split needs no fixup.
        Architecture(
            model_type="minimax_m3_vl",
            model_class=MiniMaxM3Model,
            tensor_parallel=MiniMaxM3ShardingStrategy,
            # <mm:think> tags and namespaced XML tool calls. mlx-lm would detect <think>
            # (in the vocabulary, unused) and a JSON tool-call parser from the template.
            output_parser=OutputParser(
                parse=parse_minimax_m3,
                after_thinking_parser=False,
                coerces_to_tool_schemas=True,
            ),
        ),
    ),
    reasoning_effort_hook=REASONING_EFFORT_HOOK,
    template_kwargs_hook=TEMPLATE_KWARGS_HOOK,
)
