from mlx_lm.models.minimax import Model as MiniMaxModel

from exo.worker.engines.mlx.models.family import Architecture, ModelFamily
from exo.worker.engines.mlx.models.minimax.sharding import MiniMaxShardingStrategy
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
    ),
)
