"""reasoning_effort for the GLM-5.2 and GLM-5.3 chat templates.

Both templates understand only part of exo's effort vocabulary and render every other
value as "Reasoning Effort: Max":

- GLM-5.2: "high" -> High; anything else -> Max. ``enable_thinking=false`` renders
  ``<think></think>`` and no effort line.
- GLM-5.3 and GLM-5.3-Flash: "low" / "high" -> Low / High; anything else -> Max. There is
  no ``enable_thinking`` switch: the generation prompt always opens ``<think>``.

exo resolves ``enable_thinking=True`` to "medium", so without a mapping a plain thinking
request ran at Max.
"""

from typing import Literal

from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.models.family import ReasoningEffortHook

GlmReasoningTemplate = Literal["glm-5.2", "glm-5.3"]


def _template(task_params: TextGenerationTaskParams) -> GlmReasoningTemplate | None:
    model = task_params.model.lower()
    if "glm-5.3" in model:
        return "glm-5.3"
    if "glm-5.2" in model:
        return "glm-5.2"
    return None


def applies_to(task_params: TextGenerationTaskParams) -> bool:
    return _template(task_params) is not None


def template_effort(task_params: TextGenerationTaskParams) -> str | None:
    """None or xhigh -> the template's default (Max); medium/high -> High; none,
    minimal, low -> Low on GLM-5.3 (GLM-5.2 has no Low, so High)."""
    match task_params.reasoning_effort:
        case None | "xhigh":
            return None
        case "medium" | "high":
            return "high"
        case "none" | "minimal" | "low":
            return "low" if _template(task_params) == "glm-5.3" else "high"


def thinking_off_suffix(task_params: TextGenerationTaskParams) -> str | None:
    """GLM-5.3 has no non-thinking mode in its template: close the reasoning block the
    way GLM-5.2's template does for enable_thinking=false (``<think></think>``)."""
    return "</think>" if _template(task_params) == "glm-5.3" else None


HOOK = ReasoningEffortHook(
    applies_to=applies_to,
    template_effort=template_effort,
    thinking_off_suffix=thinking_off_suffix,
)
