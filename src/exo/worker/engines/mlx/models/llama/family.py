from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.ministral3 import Model as Ministral3Model

from exo.worker.engines.mlx.models.family import Architecture, ModelFamily
from exo.worker.engines.mlx.models.llama.sharding import LlamaShardingStrategy
from exo.worker.engines.mlx.models.llama.spec import SPEC

FAMILY = ModelFamily(
    spec=SPEC,
    architectures=(
        # Also mistral and iquestcoder, which mlx-lm loads as llama.
        Architecture(
            model_type="llama",
            model_class=LlamaModel,
            tensor_parallel=LlamaShardingStrategy,
        ),
        Architecture(
            model_type="ministral3",
            model_class=Ministral3Model,
            tensor_parallel=LlamaShardingStrategy,
        ),
    ),
)
