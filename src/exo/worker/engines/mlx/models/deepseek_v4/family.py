from mlx_lm.models.deepseek_v4 import Model as DeepseekV4Model

from exo.worker.engines.mlx.models.deepseek_v4 import prompt
from exo.worker.engines.mlx.models.deepseek_v4.parser import parse_deepseek_v4
from exo.worker.engines.mlx.models.deepseek_v4.sharding import (
    DeepseekV4ShardingStrategy,
)
from exo.worker.engines.mlx.models.deepseek_v4.spec import SPEC
from exo.worker.engines.mlx.models.family import (
    Architecture,
    ModelFamily,
    OutputParser,
    PromptEncoder,
)


def _is_deepseek_v4(normalized_model_id: str) -> bool:
    return "deepseek-v4" in normalized_model_id


FAMILY = ModelFamily(
    spec=SPEC,
    architectures=(
        Architecture(
            model_type="deepseek_v4",
            model_class=DeepseekV4Model,
            tensor_parallel=DeepseekV4ShardingStrategy,
            pipeline_layer_count="num_layers",
            output_parser=OutputParser(
                parse=parse_deepseek_v4,
                after_thinking_parser=True,
                applies_to_model_id=_is_deepseek_v4,
            ),
        ),
    ),
    prompt_encoder=PromptEncoder(
        applies_to=prompt.applies_to, encode=prompt.encode_prompt
    ),
)
