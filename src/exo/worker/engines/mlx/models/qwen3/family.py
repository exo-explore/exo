from pathlib import Path

import mlx.nn as nn
from mlx_lm.models.qwen3 import Model as Qwen3Model
from mlx_lm.models.qwen3_5 import Model as Qwen3_5TextModel
from mlx_lm.models.qwen3_5_moe import Model as Qwen3_5MoeModel
from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel
from mlx_lm.models.qwen3_next import Model as Qwen3NextModel
from mlx_lm.models.qwen3_vl import Model as Qwen3VLModel

from exo.worker.engines.mlx.models.family import (
    Architecture,
    ModelClasses,
    ModelFamily,
)
from exo.worker.engines.mlx.models.qwen3.qwen4_exp import Model as Qwen4ExpModel
from exo.worker.engines.mlx.models.qwen3.qwen4_exp import (
    ModelArgs as Qwen4ExpModelArgs,
)
from exo.worker.engines.mlx.models.qwen3.sharding import (
    Qwen4ExpShardingStrategy,
    QwenShardingStrategy,
    fix_pipeline_shard,
    fix_qwen4_exp_pipeline_shard,
)
from exo.worker.engines.mlx.models.qwen3.spec import SPEC


def _attach_ngram_table(model: nn.Module, model_path: Path) -> None:
    assert isinstance(model, Qwen4ExpModel)
    model.attach_ngram_table(model_path)


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
        # Qwen3.8-Flash-Next
        Architecture(
            model_type="qwen4_exp",
            model_class=Qwen4ExpModel,
            tensor_parallel=Qwen4ExpShardingStrategy,
            fix_pipeline_shard=fix_qwen4_exp_pipeline_shard,
            # The n-gram table is read from the checkpoint files while running
            post_load=_attach_ngram_table,
            # Its indexer keys and n-gram context aren't merged into or split from a
            # batch: requests run one at a time.
            batched_generation=False,
        ),
    ),
    # mlx-lm has no qwen4_exp: the model is in this folder's qwen4_exp.py
    model_classes=(
        ModelClasses(
            model_type="qwen4_exp",
            model_class=Qwen4ExpModel,
            model_args_class=Qwen4ExpModelArgs,
        ),
    ),
)
