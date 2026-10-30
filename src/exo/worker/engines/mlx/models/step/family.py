from mlx_lm.models.step3p5 import Model as Step35Model

from exo.worker.engines.mlx.models.family import Architecture, ModelFamily
from exo.worker.engines.mlx.models.step.sharding import (
    Step35ShardingStrategy,
    fix_pipeline_shard,
)
from exo.worker.engines.mlx.models.step.spec import SPEC

FAMILY = ModelFamily(
    spec=SPEC,
    architectures=(
        Architecture(
            model_type="step3p5",
            model_class=Step35Model,
            tensor_parallel=Step35ShardingStrategy,
            fix_pipeline_shard=fix_pipeline_shard,
        ),
    ),
)
