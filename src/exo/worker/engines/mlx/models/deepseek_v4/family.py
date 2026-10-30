from exo.worker.engines.mlx.models.deepseek_v4 import prompt
from exo.worker.engines.mlx.models.deepseek_v4.model import Model as DeepseekV4Model
from exo.worker.engines.mlx.models.deepseek_v4.model import (
    ModelArgs as DeepseekV4ModelArgs,
)
from exo.worker.engines.mlx.models.deepseek_v4.parser import parse_deepseek_v4
from exo.worker.engines.mlx.models.deepseek_v4.sharding import (
    DeepseekV4ShardingStrategy,
)
from exo.worker.engines.mlx.models.deepseek_v4.spec import SPEC
from exo.worker.engines.mlx.models.family import (
    Architecture,
    ModelClasses,
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
    # mlx-lm has no deepseek_v4: the model is in this folder's model.py
    model_classes=(
        ModelClasses(
            model_type="deepseek_v4",
            model_class=DeepseekV4Model,
            model_args_class=DeepseekV4ModelArgs,
        ),
    ),
    prompt_encoder=PromptEncoder(
        applies_to=prompt.applies_to, encode=prompt.encode_prompt
    ),
)
