from mlx_lm.models.gpt_oss import Model as GptOssModel

from exo.worker.engines.mlx.models.family import (
    Architecture,
    ModelFamily,
    OutputParser,
    TemplateMessagesHook,
)
from exo.worker.engines.mlx.models.gpt_oss import prompt
from exo.worker.engines.mlx.models.gpt_oss.parser import parse_gpt_oss
from exo.worker.engines.mlx.models.gpt_oss.sharding import (
    GptOssShardingStrategy,
    fix_pipeline_shard,
)
from exo.worker.engines.mlx.models.gpt_oss.spec import SPEC

FAMILY = ModelFamily(
    spec=SPEC,
    architectures=(
        Architecture(
            model_type="gpt_oss",
            model_class=GptOssModel,
            tensor_parallel=GptOssShardingStrategy,
            fix_pipeline_shard=fix_pipeline_shard,
            # Harmony carries reasoning and tool calls in its own channels.
            output_parser=OutputParser(
                parse=parse_gpt_oss, after_thinking_parser=False
            ),
        ),
    ),
    template_messages_hook=TemplateMessagesHook(
        applies_to=prompt.applies_to, prepare=prompt.prepare_template_messages
    ),
)
