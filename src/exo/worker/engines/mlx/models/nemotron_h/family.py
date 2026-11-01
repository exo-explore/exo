from mlx_lm.models.nemotron_h import Model as NemotronHModel

from exo.worker.engines.mlx.models.family import Architecture, ModelFamily
from exo.worker.engines.mlx.models.nemotron_h.sharding import (
    NemotronHShardingStrategy,
    fix_pipeline_shard,
)
from exo.worker.engines.mlx.models.nemotron_h.spec import SPEC

FAMILY = ModelFamily(
    spec=SPEC,
    architectures=(
        Architecture(
            model_type="nemotron_h",
            model_class=NemotronHModel,
            tensor_parallel=NemotronHShardingStrategy,
            fix_pipeline_shard=fix_pipeline_shard,
        ),
    ),
)
