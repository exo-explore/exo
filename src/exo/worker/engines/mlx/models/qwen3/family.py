from mlx_lm.models.qwen3 import Model as Qwen3Model
from mlx_lm.models.qwen3_5 import Model as Qwen3_5TextModel
from mlx_lm.models.qwen3_5_moe import Model as Qwen3_5MoeModel
from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel
from mlx_lm.models.qwen3_next import Model as Qwen3NextModel
from mlx_lm.models.qwen3_vl import Model as Qwen3VLModel

from exo.worker.engines.mlx.models.family import Architecture, ModelFamily
from exo.worker.engines.mlx.models.qwen3.sharding import (
    QwenShardingStrategy,
    fix_pipeline_shard,
)
from exo.worker.engines.mlx.models.qwen3.spec import SPEC

FAMILY = ModelFamily(
    spec=SPEC,
    architectures=(
        Architecture(
            model_type="qwen3",
            model_class=Qwen3Model,
            tensor_parallel=QwenShardingStrategy,
        ),
        Architecture(
            model_type="qwen3_moe",
            model_class=Qwen3MoeModel,
            tensor_parallel=QwenShardingStrategy,
            pipeline_layer_count="num_hidden_layers",
        ),
        # The hybrid (Gated DeltaNet + full attention) models.
        Architecture(
            model_type="qwen3_next",
            model_class=Qwen3NextModel,
            tensor_parallel=QwenShardingStrategy,
            fix_pipeline_shard=fix_pipeline_shard,
        ),
        Architecture(
            model_type="qwen3_5",
            model_class=Qwen3_5TextModel,
            tensor_parallel=QwenShardingStrategy,
            fix_pipeline_shard=fix_pipeline_shard,
        ),
        Architecture(
            model_type="qwen3_5_moe",
            model_class=Qwen3_5MoeModel,
            tensor_parallel=QwenShardingStrategy,
            fix_pipeline_shard=fix_pipeline_shard,
        ),
        Architecture(
            model_type="qwen3_vl",
            model_class=Qwen3VLModel,
            tensor_parallel=QwenShardingStrategy,
        ),
    ),
)
