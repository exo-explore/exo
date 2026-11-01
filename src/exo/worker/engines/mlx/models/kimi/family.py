from mlx_lm.models.kimi_k25 import Model as KimiK25Model

from exo.worker.engines.mlx.models.deepseek_v3.sharding import DeepSeekShardingStrategy
from exo.worker.engines.mlx.models.family import (
    Architecture,
    ModelFamily,
    TokenizerLoader,
)
from exo.worker.engines.mlx.models.kimi import tokenizer
from exo.worker.engines.mlx.models.kimi.spec import SPEC

FAMILY = ModelFamily(
    spec=SPEC,
    # Kimi K2 (model_type kimi_k2) loads as mlx-lm's deepseek_v3: see the deepseek_v3 family.
    architectures=(
        Architecture(
            model_type="kimi_k25",
            model_class=KimiK25Model,
            tensor_parallel=DeepSeekShardingStrategy,
            pipeline_layer_count="num_layers",
        ),
    ),
    tokenizer_loader=TokenizerLoader(
        applies_to=tokenizer.applies_to, load=tokenizer.load_tokenizer
    ),
)
