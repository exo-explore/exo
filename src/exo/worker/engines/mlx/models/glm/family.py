from mlx_lm.models.glm4_moe import Model as Glm4MoeModel
from mlx_lm.models.glm4_moe_lite import Model as GLM4MoeLiteModel

from exo.worker.engines.mlx.models.deepseek_v3.family import (
    DEEPSEEK_V32_OUTPUT_PARSER,
)
from exo.worker.engines.mlx.models.deepseek_v3.sharding import DeepSeekShardingStrategy
from exo.worker.engines.mlx.models.family import (
    Architecture,
    ModelClasses,
    ModelFamily,
)
from exo.worker.engines.mlx.models.glm.glm5_next import Model as Glm5NextModel
from exo.worker.engines.mlx.models.glm.glm5_next import (
    ModelArgs as Glm5NextModelArgs,
)
from exo.worker.engines.mlx.models.glm.glm_moe_dsa import Model as GlmMoeDsaModel
from exo.worker.engines.mlx.models.glm.glm_moe_dsa import (
    ModelArgs as GlmMoeDsaModelArgs,
)
from exo.worker.engines.mlx.models.glm.reasoning import HOOK as REASONING_EFFORT_HOOK
from exo.worker.engines.mlx.models.glm.sharding import (
    GLM4MoeLiteShardingStrategy,
    Glm4MoeShardingStrategy,
    Glm5NextShardingStrategy,
    forward_shared_topk,
)
from exo.worker.engines.mlx.models.glm.spec import SPEC

FAMILY = ModelFamily(
    spec=SPEC,
    architectures=(
        # Also solar_open, whose mlx-lm module reuses this class.
        Architecture(
            model_type="glm4_moe",
            model_class=Glm4MoeModel,
            tensor_parallel=Glm4MoeShardingStrategy,
            pipeline_layer_count="num_layers",
        ),
        Architecture(
            model_type="glm4_moe_lite",
            model_class=GLM4MoeLiteModel,
            tensor_parallel=GLM4MoeLiteShardingStrategy,
        ),
        # GLM-5's mlx-lm class subclasses DeepSeek V3.2's, so it is sharded and laid out
        # like it, and keeps V3.2's output parser (which only applies to DeepSeek ids).
        Architecture(
            model_type="glm_moe_dsa",
            model_class=GlmMoeDsaModel,
            tensor_parallel=DeepSeekShardingStrategy,
            fix_pipeline_shard=forward_shared_topk,
            pipeline_layer_count="num_layers",
            output_parser=DEEPSEEK_V32_OUTPUT_PARSER,
        ),
        # GLM-5.3-Flash: KDA linear attention, NoPE MLA with a k-pool DSA indexer and
        # hyper-connections (glm5_next.py in this folder). Each layer builds its attention
        # mask from its own cache, so pipeline shards need no cache-index fixup.
        Architecture(
            model_type="glm5_next",
            model_class=Glm5NextModel,
            tensor_parallel=Glm5NextShardingStrategy,
        ),
    ),
    # mlx-lm's GLM-5.x model with the reference DSA indexer key-norm epsilon
    # (glm_moe_dsa.py in this folder), and GLM-5.3-Flash, which mlx-lm doesn't have
    # (glm5_next.py).
    model_classes=(
        ModelClasses(
            model_type="glm_moe_dsa",
            model_class=GlmMoeDsaModel,
            model_args_class=GlmMoeDsaModelArgs,
        ),
        ModelClasses(
            model_type="glm5_next",
            model_class=Glm5NextModel,
            model_args_class=Glm5NextModelArgs,
        ),
    ),
    reasoning_effort_hook=REASONING_EFFORT_HOOK,
)
