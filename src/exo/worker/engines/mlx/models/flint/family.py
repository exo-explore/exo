from mlx_lm.models.flint import Model as FlintModel

from exo.worker.engines.mlx.models.family import Architecture, ModelFamily
from exo.worker.engines.mlx.models.flint.sharding import (
    FlintShardingStrategy,
    fix_pipeline_shard,
)
from exo.worker.engines.mlx.models.flint.spec import SPEC

# Tool calls (<tool_call>name<arg_key>..</arg_key><arg_value>..</arg_value></tool_call>) go
# through mlx-lm's glm47 parser, which mlx-lm picks from the chat template.
FAMILY = ModelFamily(
    spec=SPEC,
    architectures=(
        Architecture(
            model_type="flint",
            model_class=FlintModel,
            tensor_parallel=FlintShardingStrategy,
            fix_pipeline_shard=fix_pipeline_shard,
        ),
    ),
)
