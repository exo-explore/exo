from exo.worker.engines.mlx.models.family import (
    Architecture,
    ModelClasses,
    ModelFamily,
)
from exo.worker.engines.mlx.models.flint.model import Model as FlintModel
from exo.worker.engines.mlx.models.flint.model import ModelArgs as FlintModelArgs
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
    # mlx-lm's Flint with its RMSNorms computed as the reference does (model.py).
    model_classes=(
        ModelClasses(
            model_type="flint",
            model_class=FlintModel,
            model_args_class=FlintModelArgs,
        ),
    ),
)
