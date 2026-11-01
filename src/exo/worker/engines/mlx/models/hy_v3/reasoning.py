"""reasoning_effort for Hy3's chat template.

The template understands exactly three values and raises on any other:

- ``no_think`` (its default): the generation prompt closes the reasoning block
  (``<think:opensource></think:opensource>``);
- ``low`` and ``high``: the generation prompt opens it, and the system prompt (or, with
  tools, the tool instructions) gets ``<｜reasoning_mode:opensource｜>reasoning_effort:<value>``.

exo's own levels are none / minimal / low / medium / high / xhigh, and it resolves
``enable_thinking=True`` to "medium" and ``enable_thinking=False`` to "none". Without a
mapping, every request that set either failed to render.
"""

from exo.shared.types.text_generation import (
    TextGenerationTaskParams,
    resolve_reasoning_params,
)
from exo.worker.engines.mlx.models.family import ReasoningEffortHook
from exo.worker.engines.mlx.models.hy_v3.model_ids import is_hy3_model_id


def applies_to(task_params: TextGenerationTaskParams) -> bool:
    return is_hy3_model_id(task_params.model)


def template_effort(task_params: TextGenerationTaskParams) -> str | None:
    """Unset -> the template's default (no_think); none -> no_think; minimal, low -> low;
    medium, high, xhigh -> high. A request that sets only enable_thinking counts as
    medium (on) or none (off), as exo's API adapters resolve it."""
    effort, _ = resolve_reasoning_params(
        task_params.reasoning_effort, task_params.enable_thinking
    )
    match effort:
        case None:
            return None
        case "none":
            return "no_think"
        case "minimal" | "low":
            return "low"
        case "medium" | "high" | "xhigh":
            return "high"


def thinking_off_suffix(_task_params: TextGenerationTaskParams) -> str | None:
    # no_think already closes the reasoning block in the generation prompt.
    return None


HOOK = ReasoningEffortHook(
    applies_to=applies_to,
    template_effort=template_effort,
    thinking_off_suffix=thinking_off_suffix,
)
