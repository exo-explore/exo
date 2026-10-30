from pathlib import Path
from typing import cast

import mlx.nn as nn
from mlx_lm.models.laguna import Model as LagunaModel

from exo.worker.engines.mlx.models.family import Architecture, ModelFamily
from exo.worker.engines.mlx.models.laguna.fast import fuse_for_inference
from exo.worker.engines.mlx.models.laguna.sharding import (
    LagunaShardingStrategy,
    fix_pipeline_shard,
)
from exo.worker.engines.mlx.models.laguna.spec import SPEC
from exo.worker.engines.mlx.sharding.pipeline import get_inner_model, get_layers


def _post_load(model: nn.Module, _model_path: Path) -> None:
    """Switch the loaded (and sharded) blocks to the fused forward pass: same weights and
    results, fewer and larger kernels (see fast.py)."""
    layers = cast(list[nn.Module], get_layers(get_inner_model(model)))
    fuse_for_inference(model, layers)


# Tool calls (<tool_call>name<arg_key>..</arg_key><arg_value>..</arg_value></tool_call>) go
# through mlx-lm's glm47 parser, which mlx-lm picks from the chat template. Reasoning is
# <think>…</think>, which the generic thinking parser handles.
#
# With thinking off, the template closes each assistant turn's (empty) reasoning with a
# bare </think> that is never opened: `<assistant></think>answer`. That is the format the
# model was trained on, so exo must not put a <think> before it.
FAMILY = ModelFamily(
    spec=SPEC,
    architectures=(
        Architecture(
            model_type="laguna",
            model_class=LagunaModel,
            tensor_parallel=LagunaShardingStrategy,
            fix_pipeline_shard=fix_pipeline_shard,
            post_load=_post_load,
            exact_think_tags=True,
        ),
    ),
)
