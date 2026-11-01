from exo.worker.engines.mlx.models.family import (
    Architecture,
    ModelClasses,
    ModelFamily,
    TokenizerLoader,
)
from exo.worker.engines.mlx.models.hy_v3 import tokenizer
from exo.worker.engines.mlx.models.hy_v3.model import Model as HyV3Model
from exo.worker.engines.mlx.models.hy_v3.model import ModelArgs as HyV3ModelArgs
from exo.worker.engines.mlx.models.hy_v3.reasoning import HOOK as REASONING_EFFORT_HOOK
from exo.worker.engines.mlx.models.hy_v3.sharding import HyV3ShardingStrategy
from exo.worker.engines.mlx.models.hy_v3.spec import SPEC

# exo's mlx-lm has no hy_v3 model: model.py is a port of Hugging Face transformers'.
# Pipeline parallelism needs no fixup: every layer is full attention with its own KV cache.
FAMILY = ModelFamily(
    spec=SPEC,
    architectures=(
        Architecture(
            model_type="hy_v3",
            model_class=HyV3Model,
            tensor_parallel=HyV3ShardingStrategy,
        ),
    ),
    model_classes=(
        ModelClasses(
            model_type="hy_v3",
            model_class=HyV3Model,
            model_args_class=HyV3ModelArgs,
        ),
    ),
    reasoning_effort_hook=REASONING_EFFORT_HOOK,
    tokenizer_loader=TokenizerLoader(
        applies_to=tokenizer.applies_to, load=tokenizer.load_tokenizer
    ),
)
