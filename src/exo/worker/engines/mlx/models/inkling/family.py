from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.models.family import (
    Architecture,
    ModelClasses,
    ModelFamily,
    OutputParser,
    TemplateMessagesHook,
)
from exo.worker.engines.mlx.models.inkling import reasoning
from exo.worker.engines.mlx.models.inkling.inkling2 import Model as Inkling2Model
from exo.worker.engines.mlx.models.inkling.inkling2 import (
    ModelArgs as Inkling2ModelArgs,
)
from exo.worker.engines.mlx.models.inkling.model import Model as InklingModel
from exo.worker.engines.mlx.models.inkling.model import ModelArgs as InklingModelArgs
from exo.worker.engines.mlx.models.inkling.parser import parse_inkling
from exo.worker.engines.mlx.models.inkling.sharding import InklingShardingStrategy
from exo.worker.engines.mlx.models.inkling.spec import SPEC


def _is_inkling(normalized_model_id: str) -> bool:
    return "inkling" in normalized_model_id


def _is_inkling_task(task_params: TextGenerationTaskParams) -> bool:
    return "inkling" in task_params.model.lower()


def _keep_messages(_messages: list[dict[str, object]]) -> None:
    return None


_INKLING_OUTPUT = OutputParser(
    parse=parse_inkling,
    after_thinking_parser=False,
    applies_to_model_id=_is_inkling,
)

FAMILY = ModelFamily(
    spec=SPEC,
    architectures=(
        Architecture(
            model_type="inkling_mm_model",
            model_class=InklingModel,
            tensor_parallel=InklingShardingStrategy,
            # Its reasoning, answers and tool calls are messages marked by special tokens
            # (parser.py); the generic thinking-tag and tool-call parsers don't apply.
            output_parser=_INKLING_OUTPUT,
        ),
        # Inkling 2 Mini (config model_type inkling2_loop): mlx-lm's Inkling 2 model
        # (inkling2/, vendored from ml-explore/mlx-lm#2086). Same message tokens and ids as
        # Inkling-Small, so the same parser. Single node: no tensor-parallel strategy yet.
        Architecture(
            model_type="inkling2_loop",
            model_class=Inkling2Model,
            output_parser=_INKLING_OUTPUT,
        ),
    ),
    # mlx-lm has no Inkling: the text model is in this folder's model.py. A pipeline
    # shard builds its attention masks from its own layers, so it needs no fixup.
    model_classes=(
        ModelClasses(
            model_type="inkling_mm_model",
            model_class=InklingModel,
            model_args_class=InklingModelArgs,
        ),
        ModelClasses(
            model_type="inkling2_loop",
            model_class=Inkling2Model,
            model_args_class=Inkling2ModelArgs,
        ),
    ),
    # An assistant prefill continues a text message: the template opens one with
    # <|message_model|><|content_text|>, and exo's generation prompt ends after the first.
    template_messages_hook=TemplateMessagesHook(
        applies_to=_is_inkling_task,
        prepare=_keep_messages,
        prefill_prefix="<|content_text|>",
    ),
    reasoning_effort_hook=reasoning.HOOK,
)
