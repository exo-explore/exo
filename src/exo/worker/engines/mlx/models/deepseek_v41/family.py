from pathlib import Path

import mlx.nn as nn

from exo.worker.engines.mlx.models.deepseek_v41 import prompt
from exo.worker.engines.mlx.models.deepseek_v41.model import Model, ModelArgs
from exo.worker.engines.mlx.models.deepseek_v41.parser import parse_deepseek_v41
from exo.worker.engines.mlx.models.deepseek_v41.sharding import (
    DeepseekV41ShardingStrategy,
    pipeline_parallel,
)
from exo.worker.engines.mlx.models.deepseek_v41.spec import SPEC
from exo.worker.engines.mlx.models.family import (
    Architecture,
    ModelClasses,
    ModelFamily,
    OutputParser,
    PromptEncoder,
)


def _load_engram(model: nn.Module, model_path: Path) -> None:
    assert isinstance(model, Model)
    model.load_engram(model_path)


FAMILY = ModelFamily(
    spec=SPEC,
    architectures=(
        Architecture(
            model_type="deepseek_v41",
            model_class=Model,
            tensor_parallel=DeepseekV41ShardingStrategy,
            pipeline_parallel=pipeline_parallel,
            output_parser=OutputParser(
                parse=parse_deepseek_v41, after_thinking_parser=True
            ),
            post_load=_load_engram,
            # Its per-layer cache can't be merged into or split from a batch yet.
            batched_generation=False,
        ),
    ),
    model_classes=(
        ModelClasses(
            model_type="deepseek_v41", model_class=Model, model_args_class=ModelArgs
        ),
    ),
    prompt_encoder=PromptEncoder(
        applies_to=prompt.applies_to,
        encode=prompt.encode_prompt,
        model_types=frozenset({"deepseek_v41"}),
        # chat mode is `<｜Assistant｜></think>`: thinking closed without an opener
        exact_think_tags=True,
    ),
)
