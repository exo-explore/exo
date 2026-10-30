from exo.worker.engines.mlx.models.family import (
    Architecture,
    ModelClasses,
    ModelFamily,
)
from exo.worker.engines.mlx.models.nemotron_h.model import Model as NemotronHModel
from exo.worker.engines.mlx.models.nemotron_h.model import (
    ModelArgs as NemotronHModelArgs,
)
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
    model_classes=(
        # mlx-lm's model with the router logits in float32 (model.py in this folder)
        ModelClasses(
            model_type="nemotron_h",
            model_class=NemotronHModel,
            model_args_class=NemotronHModelArgs,
        ),
    ),
)
