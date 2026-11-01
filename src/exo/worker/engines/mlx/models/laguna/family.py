from mlx_lm.models.laguna import Model as LagunaModel

from exo.worker.engines.mlx.models.family import Architecture, ModelFamily
from exo.worker.engines.mlx.models.laguna.sharding import (
    LagunaShardingStrategy,
    fix_pipeline_shard,
)
from exo.worker.engines.mlx.models.laguna.spec import SPEC

# Tool calls (<tool_call>name<arg_key>..</arg_key><arg_value>..</arg_value></tool_call>) go
# through mlx-lm's glm47 parser, which mlx-lm picks from the chat template. Reasoning is
# <think>…</think>, which the generic thinking parser handles.
#
# With thinking off, the template closes each assistant turn's (empty) reasoning with a
# bare </think> that is never opened: `<assistant></think>answer`. That is the format the
# model was trained on, so exo must not put a <think> before it.
FAMILY = ModelFamily(
    spec=SPEC,
    architectures=(
        Architecture(
            model_type="laguna",
            model_class=LagunaModel,
            tensor_parallel=LagunaShardingStrategy,
            fix_pipeline_shard=fix_pipeline_shard,
            exact_think_tags=True,
        ),
    ),
)
