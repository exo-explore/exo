"""The model families on the worker: sharding, prompt encoders, output parsers, tokenizer
loaders and family-provided model classes, keyed by model_type.

The master-side part of each family (placement rules, EOS ids) is in ``spec_registry``,
which this registry follows family for family.
"""

from typing import Any, cast

import mlx.nn as nn
from mlx_lm.utils import (
    _get_classes as mlx_lm_get_classes,  # pyright: ignore[reportPrivateUsage]
)

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.models.deepseek_v3.family import FAMILY as DEEPSEEK_V3
from exo.worker.engines.mlx.models.deepseek_v4.family import FAMILY as DEEPSEEK_V4
from exo.worker.engines.mlx.models.family import (
    Architecture,
    ModelClasses,
    ModelFamily,
    PromptEncoder,
    ReasoningEffortHook,
    TemplateMessagesHook,
    TokenizerLoader,
)
from exo.worker.engines.mlx.models.gemma.family import FAMILY as GEMMA
from exo.worker.engines.mlx.models.glm.family import FAMILY as GLM
from exo.worker.engines.mlx.models.gpt_oss.family import FAMILY as GPT_OSS
from exo.worker.engines.mlx.models.inkling.family import FAMILY as INKLING
from exo.worker.engines.mlx.models.kimi.family import FAMILY as KIMI
from exo.worker.engines.mlx.models.llama.family import FAMILY as LLAMA
from exo.worker.engines.mlx.models.minimax.family import FAMILY as MINIMAX
from exo.worker.engines.mlx.models.nemotron_h.family import FAMILY as NEMOTRON_H
from exo.worker.engines.mlx.models.qwen3.family import FAMILY as QWEN3
from exo.worker.engines.mlx.models.spec_registry import FAMILY_SPECS
from exo.worker.engines.mlx.models.step.family import FAMILY as STEP

# Same order as spec_registry.FAMILY_SPECS: model-id rules (prompt encoders, tokenizer
# loaders) are tried in this order, first match wins.
FAMILIES: tuple[ModelFamily, ...] = (
    KIMI,
    GLM,
    GPT_OSS,
    QWEN3,
    GEMMA,
    DEEPSEEK_V3,
    DEEPSEEK_V4,
    LLAMA,
    MINIMAX,
    NEMOTRON_H,
    STEP,
    INKLING,
)
assert tuple(family.spec for family in FAMILIES) == FAMILY_SPECS


def _architectures_by_class() -> dict[type[nn.Module], Architecture]:
    by_class: dict[type[nn.Module], Architecture] = {}
    for family in FAMILIES:
        for architecture in family.architectures:
            assert architecture.model_type in family.spec.model_types, (
                f"{architecture.model_type} is not listed in {family.spec.name}'s spec"
            )
            assert architecture.model_class not in by_class
            by_class[architecture.model_class] = architecture
    return by_class


def _model_classes_by_model_type() -> dict[str, ModelClasses]:
    by_model_type: dict[str, ModelClasses] = {}
    for family in FAMILIES:
        for classes in family.model_classes:
            assert classes.model_type not in by_model_type
            by_model_type[classes.model_type] = classes
    return by_model_type


_ARCHITECTURES_BY_CLASS = _architectures_by_class()
_MODEL_CLASSES_BY_MODEL_TYPE = _model_classes_by_model_type()


def architecture_for_model_class(model_class: type) -> Architecture | None:
    """The architecture registered for the class or, failing that, its nearest registered
    base class."""
    for cls in model_class.__mro__:
        architecture = _ARCHITECTURES_BY_CLASS.get(cast(type[nn.Module], cls))
        if architecture is not None:
            return architecture
    return None


def architecture_for_model(model: nn.Module) -> Architecture | None:
    return architecture_for_model_class(type(model))


def get_model_classes(config: dict[str, Any]) -> tuple[type[nn.Module], type]:
    """mlx-lm's load_model hook: the family's own model classes for checkpoints whose
    model_type a family provides them for, otherwise mlx-lm's."""
    model_type = cast(str, config["model_type"])
    classes = _MODEL_CLASSES_BY_MODEL_TYPE.get(model_type)
    if classes is not None:
        return classes.model_class, classes.model_args_class
    return mlx_lm_get_classes(config)


def prompt_encoder_for(task_params: TextGenerationTaskParams) -> PromptEncoder | None:
    for family in FAMILIES:
        encoder = family.prompt_encoder
        if encoder is not None and encoder.applies_to(task_params):
            return encoder
    return None


def template_messages_hooks_for(
    task_params: TextGenerationTaskParams,
) -> list[TemplateMessagesHook]:
    return [
        hook
        for family in FAMILIES
        if (hook := family.template_messages_hook) is not None
        and hook.applies_to(task_params)
    ]


def reasoning_effort_hook_for(
    task_params: TextGenerationTaskParams,
) -> ReasoningEffortHook | None:
    for family in FAMILIES:
        hook = family.reasoning_effort_hook
        if hook is not None and hook.applies_to(task_params):
            return hook
    return None


def tokenizer_loader_for(model_id: ModelId) -> TokenizerLoader | None:
    for family in FAMILIES:
        loader = family.tokenizer_loader
        if loader is not None and loader.applies_to(model_id):
            return loader
    return None
