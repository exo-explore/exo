from mlx_lm.models.deepseek_v3 import Model as DeepseekV3Model
from mlx_lm.models.deepseek_v32 import Model as DeepseekV32Model

from exo.worker.engines.mlx.models.deepseek_v3 import prompt
from exo.worker.engines.mlx.models.deepseek_v3.parser import parse_deepseek_v32
from exo.worker.engines.mlx.models.deepseek_v3.sharding import DeepSeekShardingStrategy
from exo.worker.engines.mlx.models.deepseek_v3.spec import SPEC
from exo.worker.engines.mlx.models.family import (
    Architecture,
    ModelFamily,
    OutputParser,
    PromptEncoder,
)


def _is_deepseek(normalized_model_id: str) -> bool:
    return "deepseek" in normalized_model_id


# DSML tool calls after the thinking block. Only for DeepSeek ids: GLM-5's mlx-lm class
# subclasses V3.2's, but GLM-5 writes its tool calls in its own format.
DEEPSEEK_V32_OUTPUT_PARSER = OutputParser(
    parse=parse_deepseek_v32,
    after_thinking_parser=True,
    applies_to_model_id=_is_deepseek,
)

FAMILY = ModelFamily(
    spec=SPEC,
    architectures=(
        # Also Kimi K2, whose model_type kimi_k2 mlx-lm loads as deepseek_v3.
        Architecture(
            model_type="deepseek_v3",
            model_class=DeepseekV3Model,
            tensor_parallel=DeepSeekShardingStrategy,
            pipeline_layer_count="num_layers",
        ),
        Architecture(
            model_type="deepseek_v32",
            model_class=DeepseekV32Model,
            tensor_parallel=DeepSeekShardingStrategy,
            pipeline_layer_count="num_layers",
            output_parser=DEEPSEEK_V32_OUTPUT_PARSER,
        ),
    ),
    prompt_encoder=PromptEncoder(
        applies_to=prompt.applies_to, encode=prompt.encode_prompt
    ),
)
