from mlx_lm.models.muse_spark import Model as MuseSparkModel

from exo.worker.engines.mlx.models.family import (
    Architecture,
    ModelFamily,
    OutputParser,
)
from exo.worker.engines.mlx.models.muse_spark.parser import parse_muse_spark
from exo.worker.engines.mlx.models.muse_spark.reasoning import (
    HOOK as REASONING_EFFORT_HOOK,
)
from exo.worker.engines.mlx.models.muse_spark.sharding import (
    MuseSparkShardingStrategy,
    pipeline_parallel,
)
from exo.worker.engines.mlx.models.muse_spark.spec import SPEC

FAMILY = ModelFamily(
    spec=SPEC,
    architectures=(
        # mlx-lm's muse_spark is the text model, with the checkpoint's vision tower (exo
        # serves the model as text only); the audio tower isn't loaded.
        Architecture(
            model_type="muse_spark",
            model_class=MuseSparkModel,
            tensor_parallel=MuseSparkShardingStrategy,
            pipeline_parallel=pipeline_parallel,
            # Messages with recipients (reasoning to=self, answers to=user) and ATEM XML
            # tool calls, whose values are text: typed against the request's schemas.
            output_parser=OutputParser(
                parse=parse_muse_spark,
                after_thinking_parser=False,
                coerces_to_tool_schemas=True,
            ),
        ),
    ),
    reasoning_effort_hook=REASONING_EFFORT_HOOK,
)
