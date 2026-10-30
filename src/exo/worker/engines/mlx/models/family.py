"""What the worker needs to know about a model family: how to shard it, how to encode its
prompts, how to parse its output, how to load its tokenizer, and optionally its own model
classes. Each family folder's ``family.py`` defines one ``ModelFamily``; ``registry.py``
collects them."""

from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, final

import mlx.nn as nn
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.engines.mlx.models.spec import FamilySpec
from exo.worker.engines.mlx.sharding.pipeline import PipelineShardFixup
from exo.worker.engines.mlx.sharding.tensor import TensorParallelShardingStrategy


def _any_model_id(_normalized_model_id: str) -> bool:
    return True


@final
@dataclass(frozen=True)
class OutputParser:
    """Parses the model's output stream (reasoning, tool calls) in place of the generic
    tool-call parser."""

    parse: Callable[
        [Generator[GenerationResponse | None]],
        Generator[GenerationResponse | ToolCallResponse | None],
    ]
    # Whether the generic thinking-tag parser runs on the stream first (when the tokenizer
    # has thinking tags).
    after_thinking_parser: bool
    # The (normalized, lowercased) model ids it applies to; others get the generic parsers.
    applies_to_model_id: Callable[[str], bool] = _any_model_id


@final
@dataclass(frozen=True)
class Architecture:
    """One mlx-lm model class, i.e. one config.json model_type after mlx-lm's remapping.

    A loaded model gets the architecture of the first class in its MRO that one is
    registered for, so an mlx-lm class that subclasses a registered one (as GLM-5's does
    DeepSeek V3.2's) is handled like it unless it is registered itself.
    """

    model_type: str
    model_class: type[nn.Module]
    tensor_parallel: type[TensorParallelShardingStrategy] | None = None
    fix_pipeline_shard: PipelineShardFixup | None = None
    # The inner model's layer counter to set to the shard's layer count in pipeline
    # parallelism, for models that loop over it rather than over their layers.
    pipeline_layer_count: Literal["num_layers", "num_hidden_layers"] | None = None
    output_parser: OutputParser | None = None
    # PromptEncoder.exact_think_tags for a model rendered through its Jinja chat template:
    # its template's think tags are exactly the model's format (e.g. a bare `</think>`
    # that closes thinking when it is off), so exo leaves an unmatched `</think>` as it is.
    exact_think_tags: bool = False


@final
@dataclass(frozen=True)
class ModelClasses:
    """A model definition the family ships itself, in its folder's model.py, for a model
    that exo's mlx-lm doesn't have. Checkpoints whose config.json model_type matches load
    with these classes (through mlx-lm's load_model(get_model_classes=...))."""

    model_type: str
    model_class: type[nn.Module]
    model_args_class: type


@final
@dataclass(frozen=True)
class PromptEncoder:
    """Renders the prompt for models that ship no Jinja chat template."""

    applies_to: Callable[[TextGenerationTaskParams], bool]
    encode: Callable[[list[dict[str, Any]], TextGenerationTaskParams], str]
    # The config.json model_types it renders for. When the loaded model's type is known,
    # an encoder that lists its model_types is picked by them alone, not by applies_to.
    model_types: frozenset[str] = frozenset()
    # True when its think tags are exactly the model's format (e.g. a bare `</think>` that
    # closes thinking in chat mode): exo then leaves an unmatched `</think>` as it is
    # instead of putting a `<think>` before it.
    exact_think_tags: bool = False


@final
@dataclass(frozen=True)
class TemplateMessagesHook:
    """Adjusts the messages before they go through the model's Jinja chat template."""

    applies_to: Callable[[TextGenerationTaskParams], bool]
    prepare: Callable[[list[dict[str, Any]]], None]


@final
@dataclass(frozen=True)
class TokenizerLoader:
    """Loads tokenizers that mlx-lm's loader can't."""

    applies_to: Callable[[ModelId], bool]
    load: Callable[[Path, list[int] | None], TokenizerWrapper]


@final
@dataclass(frozen=True)
class ModelFamily:
    spec: FamilySpec
    architectures: tuple[Architecture, ...] = ()
    model_classes: tuple[ModelClasses, ...] = ()
    prompt_encoder: PromptEncoder | None = None
    template_messages_hook: TemplateMessagesHook | None = None
    tokenizer_loader: TokenizerLoader | None = None
