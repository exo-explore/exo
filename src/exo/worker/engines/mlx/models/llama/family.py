from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.ministral3 import Model as Ministral3Model

from exo.worker.engines.mlx.models.family import (
    Architecture,
    ModelFamily,
    OutputParser,
    TemplateMessagesHook,
)
from exo.worker.engines.mlx.models.llama import prompt
from exo.worker.engines.mlx.models.llama.parser import (
    in_ipython_environment,
    parse_llama3,
)
from exo.worker.engines.mlx.models.llama.sharding import LlamaShardingStrategy
from exo.worker.engines.mlx.models.llama.spec import SPEC

FAMILY = ModelFamily(
    spec=SPEC,
    architectures=(
        # Also mistral and iquestcoder, which mlx-lm loads as llama.
        Architecture(
            model_type="llama",
            model_class=LlamaModel,
            tensor_parallel=LlamaShardingStrategy,
            # Llama 3's tool calls: bare JSON or after <|python_tag|>, with no markers
            # mlx-lm could find a tool parser by. Only for prompts with Llama 3's ipython
            # environment (a request with tools), so other checkpoints of the
            # architecture, and requests without tools, keep the generic parsers.
            output_parser=OutputParser(
                parse=parse_llama3,
                after_thinking_parser=True,
                applies_to_prompt=in_ipython_environment,
                # Small models write numbers as strings now and then ("a": "17").
                coerces_to_tool_schemas=True,
                # A JSON object that names none of the tools is no call (e.g. one shown
                # in an answer).
                parse_with_tool_names=parse_llama3,
            ),
        ),
        Architecture(
            model_type="ministral3",
            model_class=Ministral3Model,
            tensor_parallel=LlamaShardingStrategy,
        ),
    ),
    template_messages_hook=TemplateMessagesHook(
        applies_to=prompt.applies_to, prepare=prompt.split_parallel_tool_calls
    ),
)
