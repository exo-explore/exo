from mlx_lm.models.gemma4 import Model as Gemma4Model

from exo.worker.engines.mlx.models.family import Architecture, ModelFamily
from exo.worker.engines.mlx.models.gemma.sharding import Gemma4ShardingStrategy
from exo.worker.engines.mlx.models.gemma.spec import SPEC

FAMILY = ModelFamily(
    spec=SPEC,
    architectures=(
        Architecture(
            model_type="gemma4",
            model_class=Gemma4Model,
            tensor_parallel=Gemma4ShardingStrategy,
        ),
    ),
)
